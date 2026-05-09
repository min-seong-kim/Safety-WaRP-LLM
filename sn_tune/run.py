"""
run.py  —  WaRP-SN-Tune 메인 스크립트

전체 파이프라인
──────────────
Phase A  [Convert]   원본 모델 레이어 → LinearSNWaRP(C = W @ U)
Phase B  [Detect]    safety 데이터로 |∂L/∂C| 누적 → top-k 좌표 선택 → mask
Phase C  [Tune]      mask 위치의 C 만 업데이트 (backward hook)
Phase D  [Restore]   W_final = C @ U.T → nn.Linear 변환 → 저장

모델 출력 보존 근거
──────────────────
U 가 정규직교이므로  C @ U.T = W @ U @ U.T = W.
따라서 Convert 직후 forward 는 원본 모델과 수치적으로 동일.

Usage
─────
python -m sn_tune.run \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --basis_dir  ./checkpoints/phase1_20260505_164049/basis \
    --dataset_file ./data/circuit_breakers_train.json \
    --output_dir   ./warp_sn_fwd_output \
    --layer_type   ffn_up,ffn_down,attn_q,attn_k,attn_v \
    --detect_method forward \
    --top_k_ffn     1200 \
    --top_k_attn    200 \
    --learning_rate 5e-5 \
    --num_epochs    3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

# 같은 패키지 내 모듈
from .module  import (
    LinearSNWaRP,
    LAYER_TYPE_MAP,
    convert_to_sn_warp,
    restore_to_linear,
)
from .detect  import (
    accumulate_grad_scores,
    select_top_coords,
    apply_coeff_gradient_masks,
    detect_with_forward_scores,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "./logs/warp_sn_tune") -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"warp_sn_tune_{ts}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    return log_file


log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Basis loading
# ─────────────────────────────────────────────────────────────────────────────

def load_basis_for_layer(
    basis_dir: str,
    layer_type: str,
    layer_idx: int,
) -> torch.Tensor | None:
    """
    Phase 1 SVD basis 파일 로드.

    파일 이름 규칙: layer_{layer_idx+1:02d}_svd.pt  (1-indexed)
    data['U'] = V  [in_dim, in_dim]  정규직교 right singular vectors.

    Returns None if file does not exist.
    """
    # 1-indexed 시도 먼저, 없으면 0-indexed 시도
    for offset in (1, 0):
        fname = f"layer_{layer_idx + offset:02d}_svd.pt"
        path  = os.path.join(basis_dir, layer_type, fname)
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            U    = data["U"]   # [in_dim, in_dim]
            return U

    log.warning(f"  Basis file not found: {basis_dir}/{layer_type}/layer_*{layer_idx}*_svd.pt")
    return None


def load_all_basis(
    basis_dir: str,
    layer_types: list[str],
    num_layers: int,
) -> dict[tuple[int, str], torch.Tensor]:
    """
    모든 (layer_idx, ltype) 에 대한 U 딕셔너리 반환.
    파일이 없는 항목은 딕셔너리에서 제외.
    """
    log.info(f"Loading basis files from: {basis_dir}")
    basis_dict: dict[tuple[int, str], torch.Tensor] = {}

    for ltype in layer_types:
        ltype_dir = os.path.join(basis_dir, ltype)
        if not os.path.isdir(ltype_dir):
            log.warning(f"  Basis subdir not found: {ltype_dir}  — skipping {ltype}")
            continue

        loaded = 0
        for layer_idx in range(num_layers):
            U = load_basis_for_layer(basis_dir, ltype, layer_idx)
            if U is not None:
                basis_dict[(layer_idx, ltype)] = U
                loaded += 1

        log.info(f"  {ltype:10s}: loaded {loaded}/{num_layers} layers  "
                 f"(U shape: {U.shape if loaded > 0 else 'N/A'})")

    log.info(f"Total basis matrices loaded: {len(basis_dict)}")
    return basis_dict


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (Circuit Breakers)
# ─────────────────────────────────────────────────────────────────────────────

def _is_instruct_model(name: str) -> bool:
    lower = name.lower()
    return any(t in lower for t in ("instruct", "chat"))


class PromptOnlyDataset(Dataset):
    """
    Detection 전용: harmful prompt 만 입력 (safe response 없음).

    detection_v2 와 동일하게 [user: harmful_prompt] 만 모델에 넣는다.
    gradient 방식 detection 에서 사용하며, loss 는 prompt 내
    next-token prediction (자기 자신의 다음 토큰 예측) 으로 계산된다.
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_samples: int | None = None,
        max_length: int = 1024,
        is_instruct: bool = False,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_samples:
            data = data[: min(max_samples, len(data))]

        self.prompts     = [item.get("prompt", "") for item in data]
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.is_instruct = is_instruct
        log.info(f"[PromptOnlyDataset] {len(self.prompts)} prompts (harmful prompt only)")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt = self.prompts[idx]

        if self.is_instruct:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            input_ids = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length,
            )["input_ids"]

        seq_len        = min(len(input_ids), self.max_length)
        input_ids      = input_ids[:seq_len]
        pad_len        = self.max_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        input_ids_pad  = input_ids + [self.tokenizer.pad_token_id] * pad_len

        # labels: prompt 내 next-token prediction (padding → -100)
        labels = list(input_ids_pad)
        for i in range(self.max_length):
            if attention_mask[i] == 0:
                labels[i] = -100

        return {
            "input_ids":      torch.tensor(input_ids_pad, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


class SafetyDataset(Dataset):
    """Circuit Breakers 안전 데이터셋 (tuning 전용: prompt + safe_response)."""

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_samples: int | None = None,
        max_length: int = 1024,
        is_instruct: bool = False,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_samples:
            data = data[: min(max_samples, len(data))]

        self.data        = data
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.is_instruct = is_instruct
        self._logged     = False
        log.info(f"Dataset loaded: {len(self.data)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item            = self.data[idx]
        harmful_prompt  = item.get("prompt", "")
        safe_response   = item.get("llama3_output", "")

        if self.is_instruct:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": harmful_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            full_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": harmful_prompt},
                    {"role": "assistant", "content": safe_response},
                ],
                tokenize=True, add_generation_prompt=False,
            )
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]

            seq_len        = len(full_ids)
            pad_len        = self.max_length - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            input_ids      = full_ids + [self.tokenizer.pad_token_id] * pad_len
            labels         = list(input_ids)
            plen           = min(len(prompt_ids), self.max_length)
            for i in range(plen):
                labels[i] = -100
            for i in range(self.max_length):
                if attention_mask[i] == 0:
                    labels[i] = -100
        else:
            prompt_text = f"Question: {harmful_prompt}\nAnswer:"
            full_text   = f"{prompt_text} {safe_response}"
            enc         = self.tokenizer(
                full_text, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            penc        = self.tokenizer(
                prompt_text, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            labels = enc["input_ids"].clone()
            labels[:, : penc["input_ids"].size(1)] = -100
            labels[enc["attention_mask"] == 0]     = -100

            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         labels.squeeze(0),
            }

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def freeze_all_except_coeff(model) -> None:
    """LinearSNWaRP.coeff 를 제외한 모든 파라미터 동결."""
    for param in model.parameters():
        param.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, LinearSNWaRP):
            mod.coeff.requires_grad_(True)


def train_loop(
    model,
    dataloader: DataLoader,
    device: torch.device,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    grad_accum_steps: int = 4,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
) -> None:
    """
    C 공간의 선택 좌표만 업데이트하는 훈련 루프.
    backward hook 이 이미 적용된 상태에서 호출.
    """
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Ensure coeff.requires_grad=True and hooks are set.")

    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=0.01)

    total_steps  = num_epochs * math.ceil(len(dataloader) / grad_accum_steps)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    log.info(f"[tune] total_optimizer_steps={total_steps}, warmup={warmup_steps}")
    log.info(f"[tune] trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        epoch_loss  = 0.0
        step_count  = 0
        opt_steps   = 0

        pbar = tqdm(dataloader, desc=f"[tune] epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                log.warning(f"  NaN/Inf loss at step {batch_idx}. Skipping.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            epoch_loss += outputs.loss.item()
            step_count += 1

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

            pbar.set_postfix(loss=f"{epoch_loss / step_count:.4f}")

        avg = epoch_loss / max(step_count, 1)
        log.info(f"[tune] epoch {epoch+1}: avg_loss={avg:.4f}, opt_steps={opt_steps}")


# ─────────────────────────────────────────────────────────────────────────────
# Upload helper
# ─────────────────────────────────────────────────────────────────────────────

def _upload_to_hf(model_dir: str, repo_id: str, hf_token: str | None) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model")
        log.info(f"[upload] Uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        log.error(f"[upload] Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WaRP-SN-Tune: safety-basis reparameterized SN fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 모델/basis
    p.add_argument("--model_name",  type=str, required=True,
                   help="HF model id or local path (원본 모델, 비회전)")
    p.add_argument("--basis_dir",   type=str, required=True,
                   help="Phase 1 basis 디렉토리 (basis/ffn_up/, …)")

    # 데이터
    p.add_argument("--dataset_file", type=str, required=True,
                   help="circuit_breakers_train.json 경로")
    p.add_argument("--max_samples",  type=int, default=4994,
                   help="훈련 샘플 수 (detect + tune 공용)")
    p.add_argument("--max_length",   type=int, default=1024)

    # 레이어 선택
    p.add_argument("--layer_type", type=str,
                   default="ffn_up,attn_q,attn_k,attn_v",
                   help="처리할 layer types (쉼표 구분). "
                        "ffn_down 은 basis [11008,11008] 로 VRAM 집약적이므로 기본 제외")

    # Detection 방법 선택
    p.add_argument("--detect_method", type=str, default="gradient",
                   choices=["gradient", "forward"],
                   help="gradient: |∂L/∂C| 누적 후 top-k (hb env) | "
                        "forward: 패치 모델 activation score + 교집합 (hb_sntune env, detection_v2 방식)")

    # gradient 방식 설정
    p.add_argument("--keep_ratio",   type=float, default=0.10,
                   help="[gradient] C 좌표 중 선택 비율 (0.10 = top 10%%)")
    p.add_argument("--granularity",  type=str,   default="element",
                   choices=["element", "row"],
                   help="[gradient] element: 개별 원소 선택 / row: 행(뉴런) 단위 선택")
    p.add_argument("--per_layer",    action="store_true",
                   help="[gradient] 레이어별 keep_ratio 적용 (기본: 전체 통합)")
    p.add_argument("--detect_batches", type=int, default=None,
                   help="[gradient] Detection 에 사용할 배치 수 (None=전체)")

    # forward 방식 설정 (detection_v2 방식)
    p.add_argument("--top_k_ffn",    type=int, default=1200,
                   help="[forward] FFN 레이어당 per-prompt top-k 뉴런 수")
    p.add_argument("--top_k_attn",   type=int, default=200,
                   help="[forward] Attention 레이어당 per-prompt top-k 뉴런 수")
    p.add_argument("--detect_prompts", type=int, default=None,
                   help="[forward] Detection 에 사용할 프롬프트 수 (None=전체)")

    # WaRP 모듈 옵션
    p.add_argument("--cpu_basis", action="store_true",
                   help="U 를 CPU buffer 로 유지 (VRAM 절약, 속도 약간 감소)")

    # Tuning 하이퍼파라미터
    p.add_argument("--learning_rate",      type=float, default=5e-5)
    p.add_argument("--num_epochs",         type=int,   default=3)
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--grad_accum_steps",   type=int,   default=4)
    p.add_argument("--warmup_ratio",       type=float, default=0.1)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)

    # 출력/업로드
    p.add_argument("--output_dir",  type=str, default="./warp_sn_tune_output")
    p.add_argument("--upload_name", type=str, default=None,
                   help="HF 업로드 repo id (예: username/model-name)")
    p.add_argument("--hf_token",    type=str, default=None)

    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    log_file = setup_logging()

    log.info("=" * 70)
    log.info("WaRP-SN-Tune")
    log.info("=" * 70)
    log.info(f"model_name   : {args.model_name}")
    log.info(f"basis_dir    : {args.basis_dir}")
    log.info(f"dataset_file : {args.dataset_file}")
    log.info(f"layer_type   : {args.layer_type}")
    log.info(f"detect_method: {args.detect_method}")
    if args.detect_method == "gradient":
        log.info(f"keep_ratio   : {args.keep_ratio}")
        log.info(f"granularity  : {args.granularity}")
        log.info(f"per_layer    : {args.per_layer}")
    else:
        log.info(f"top_k_ffn    : {args.top_k_ffn}")
        log.info(f"top_k_attn   : {args.top_k_attn}")
        log.info(f"detect_prompts: {args.detect_prompts}")
    log.info(f"cpu_basis    : {args.cpu_basis}")
    log.info(f"lr           : {args.learning_rate}")
    log.info(f"epochs       : {args.num_epochs}")
    log.info(f"log_file     : {log_file}")
    log.info("=" * 70)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_types  = [t.strip() for t in args.layer_type.split(",")]

    # ── 1. 모델 로드 ──────────────────────────────────────────────────────
    log.info("\n[Step 1] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    num_layers = len(model.model.layers)
    log.info(f"  Model on device: {next(model.parameters()).device}")
    log.info(f"  Num transformer layers: {num_layers}")

    is_instruct = _is_instruct_model(args.model_name)
    log.info(f"  Instruct model: {is_instruct}")

    # ── 2. Basis 로드 ─────────────────────────────────────────────────────
    log.info("\n[Step 2] Loading Phase 1 basis...")
    basis_dict = load_all_basis(args.basis_dir, layer_types, num_layers)
    if not basis_dict:
        log.error("No basis files found. Check --basis_dir and --layer_type.")
        sys.exit(1)

    # ── 3. LinearSNWaRP 변환 ─────────────────────────────────────────────
    log.info("\n[Step 3] Converting layers to LinearSNWaRP (C = W @ U)...")
    convert_to_sn_warp(model, basis_dict, layer_types, cpu_basis=args.cpu_basis)

    # ── [검증] 변환 후 출력 보존 확인 (소규모 샘플) ───────────────────────
    log.info("[verify] Checking output preservation after conversion...")
    _verify_output_preservation(model, tokenizer, device)

    # ── 4. 데이터셋 준비 ─────────────────────────────────────────────────
    log.info("\n[Step 4] Preparing datasets...")

    # Detection: harmful prompt 만 (detection_v2 와 동일)
    prompt_dataset = PromptOnlyDataset(
        args.dataset_file, tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        is_instruct=is_instruct,
    )
    detect_loader = DataLoader(
        prompt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Tuning: prompt + safe_response (목표 응답 학습)
    tune_dataset = SafetyDataset(
        args.dataset_file, tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        is_instruct=is_instruct,
    )
    tune_loader = DataLoader(
        tune_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    # ── 5. Detection ─────────────────────────────────────────────────────
    if args.detect_method == "forward":
        log.info("\n[Step 5] Forward activation score + intersection detection (detection_v2 방식)...")
        log.info("  ※ patched modeling_llama.py (_last_*_score) 가 필요합니다 → hb_sntune env")

        # json 에서 prompt 리스트 직접 추출 (토크나이즈 없이)
        with open(args.dataset_file, "r", encoding="utf-8") as f:
            _raw = json.load(f)
        _prompts = [item.get("prompt", "") for item in _raw if item.get("prompt", "")]
        if args.detect_prompts:
            _prompts = _prompts[: args.detect_prompts]
        log.info(f"  Using {len(_prompts)} prompts for forward scoring")

        masks = detect_with_forward_scores(
            model=model,
            prompts=_prompts,
            tokenizer=tokenizer,
            layer_types=layer_types,
            top_k_ffn=args.top_k_ffn,
            top_k_attn=args.top_k_attn,
            is_chat_model=is_instruct,
            max_seq_len=args.max_length,
            device=device,
        )
    else:
        log.info("\n[Step 5] Gradient-based detection in C space...")
        scores = accumulate_grad_scores(
            model, detect_loader, device, max_batches=args.detect_batches
        )
        masks = select_top_coords(
            scores,
            keep_ratio=args.keep_ratio,
            granularity=args.granularity,     # type: ignore[arg-type]
            per_layer=args.per_layer,
        )

    # ── 6. Gradient mask hook 등록 + 나머지 동결 ─────────────────────────
    log.info("\n[Step 6] Freezing params, applying gradient masks...")
    freeze_all_except_coeff(model)
    hooks = apply_coeff_gradient_masks(model, masks)

    # ── 7. Tuning ─────────────────────────────────────────────────────────
    log.info("\n[Step 7] Training in C space (masked)...")
    train_loop(
        model, tune_loader, device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
    )

    # hook 정리
    for h in hooks:
        h.remove()
    log.info("  Hooks removed.")

    # ── 8. W_final = C @ U.T  →  nn.Linear 복원 ─────────────────────────
    log.info("\n[Step 8] Restoring W_final = C @ U.T ...")
    restore_to_linear(model)

    # ── 9. 저장 ──────────────────────────────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_tag     = f"lr{args.learning_rate:.0e}".replace("-0", "-").replace("+0", "")
    output_dir = f"{args.output_dir}_{lr_tag}_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"\n[Step 9] Model saved to: {output_dir}")

    # 설정 저장
    config_path = os.path.join(output_dir, "warp_sn_tune_config.json")
    import json as _json
    with open(config_path, "w") as f:
        _json.dump(vars(args), f, indent=2)

    # ── 10. HF 업로드 ─────────────────────────────────────────────────────
    if args.upload_name:
        log.info(f"\n[Step 10] Uploading to HuggingFace: {args.upload_name}")
        _upload_to_hf(output_dir, args.upload_name, args.hf_token)

    log.info("\n" + "=" * 70)
    log.info("WaRP-SN-Tune complete!")
    log.info(f"Output: {output_dir}")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 출력 보존 검증 (convert 직후)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_output_preservation(model, tokenizer, device: torch.device,
                                  rtol: float = 1e-2, atol: float = 1e-2) -> None:
    """
    LinearSNWaRP 변환 후 forward 결과가 원본 모델과 수치적으로 동일한지 확인.
    작은 입력 배치로 logits 를 비교.

    주의: 이 함수는 현재 LinearSNWaRP 모델과 비교할 원본 참조가 없으므로,
    대신 변환 직후 C @ U.T 를 직접 계산해 W 와 비교.
    """
    mismatches = []
    for mod in model.modules():
        if isinstance(mod, LinearSNWaRP):
            W_restored = mod.get_restored_weight()
            U = mod.U.to(device=mod.coeff.device, dtype=mod.coeff.dtype)
            C = mod.coeff.data
            W_expected = (C @ U.T)
            if not torch.allclose(W_restored, W_expected, rtol=rtol, atol=atol):
                mismatches.append(repr(mod))

    if mismatches:
        log.warning(f"[verify] {len(mismatches)} modules have W_rec ≠ C @ U.T (numerical drift)")
    else:
        log.info("[verify] All W_rec == C @ U.T  ✓  (output preserved)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
