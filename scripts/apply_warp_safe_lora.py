#!/usr/bin/env python
"""
apply_warp_safe_lora.py

LoRA 어댑터 + WaRP basis → WaRP-Rotated Safe LoRA projection 적용 후 모델 저장.

사용법:
    python scripts/apply_warp_safe_lora.py \
        --base-model     meta-llama/Llama-2-7b-hf \
        --aligned-model  kmseong/llama2_7b-Safety-FT-lr3e-5 \
        --lora-adapter-path ./checkpoints/lora_gsm8k_<ts> \
        --basis-dir      ./checkpoints/phase1_<ts>/basis \
        --output-path    ./checkpoints/warp_safelora_<ts> \
        --top-k-ratio    0.5 \
        --select-type    threshold \
        --threshold      0.5 \
        --use-approx
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Safety-WaRP-LLM 루트를 path에 추가 (models/ 패키지 접근)
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

from models.safe_lora_basis_rotation import WaRPSafeLoRA, WaRPSafeLoRAConfig


# ─────────────────────────────────────────────────────────────────────────────
# Logger 설정
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("warp_safelora")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Apply WaRP-Rotated Safe LoRA projection to a trained LoRA adapter"
    )
    p.add_argument("--base-model",          type=str, required=True)
    p.add_argument("--aligned-model",       type=str, required=True)
    p.add_argument("--lora-adapter-path",   type=str, required=True)
    p.add_argument("--basis-dir",           type=str, required=True)
    p.add_argument("--output-path",         type=str, required=True)
    p.add_argument("--top-k",               type=int,   default=None,
                   help="절대 k값 (미지정 시 top-k-ratio 사용)")
    p.add_argument("--top-k-ratio",         type=float, default=0.5,
                   help="top-k = in_dim * top_k_ratio (기본 0.5)")
    p.add_argument("--select-type",         type=str,   default="threshold",
                   choices=["threshold", "number"])
    p.add_argument("--threshold",           type=float, default=0.5)
    p.add_argument("--num-proj-layers",     type=int,   default=30)
    p.add_argument("--use-approx",          action="store_true",
                   help="approximate projector 사용 (기본 False → exact)")
    p.add_argument("--device",              type=str,   default="cuda")
    p.add_argument("--hf-token",            type=str,   default=None)
    p.add_argument("--log-path",            type=str,   default="")
    p.add_argument("--no-merge",            action="store_true",
                   help="merged model 저장 생략 (어댑터만 저장)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.log_path:
        os.makedirs(os.path.join(str(REPO_DIR), "logs"), exist_ok=True)
        args.log_path = str(REPO_DIR / "logs" / f"warp_safelora_projection_{ts}.log")

    logger = setup_logger(args.log_path)

    logger.info("=" * 70)
    logger.info("WaRP-Rotated Safe LoRA — apply_warp_safe_lora.py")
    logger.info("=" * 70)
    logger.info("  base_model        : %s", args.base_model)
    logger.info("  aligned_model     : %s", args.aligned_model)
    logger.info("  lora_adapter_path : %s", args.lora_adapter_path)
    logger.info("  basis_dir         : %s", args.basis_dir)
    logger.info("  output_path       : %s", args.output_path)
    logger.info("  top_k             : %s  (ratio=%.2f)", args.top_k, args.top_k_ratio)
    logger.info("  select_type       : %s", args.select_type)
    logger.info("  threshold         : %.3f", args.threshold)
    logger.info("  num_proj_layers   : %d",  args.num_proj_layers)
    logger.info("  use_approx        : %s",  args.use_approx)
    logger.info("")

    # ── 1. aligned 모델 + LoRA 어댑터 로드 ──────────────────────────────────
    load_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    hf_kwargs = {"token": args.hf_token} if args.hf_token else {}

    logger.info("[Step 1] Loading aligned model: %s ...", args.aligned_model)
    base_for_peft = AutoModelForCausalLM.from_pretrained(
        args.aligned_model,
        torch_dtype=load_dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
        **hf_kwargs,
    )
    logger.info("[Step 1] Loading LoRA adapter: %s ...", args.lora_adapter_path)
    peft_model = PeftModel.from_pretrained(
        base_for_peft,
        args.lora_adapter_path,
        torch_dtype=load_dtype,
    )
    logger.info("[Step 1] LoRA adapter loaded.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.aligned_model, **hf_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. WaRP-Rotated Safe LoRA config ────────────────────────────────────
    config = WaRPSafeLoRAConfig(
        base_model_path    = args.base_model,
        aligned_model_path = args.aligned_model,
        basis_dir          = args.basis_dir,
        top_k              = args.top_k,
        top_k_ratio        = args.top_k_ratio,
        select_layers_type = args.select_type,
        threshold          = args.threshold,
        num_proj_layers    = args.num_proj_layers,
        use_approximation  = args.use_approx,
        devices            = args.device,
        hf_token           = args.hf_token,
    )

    # ── 3. WaRP-Rotated Safe LoRA 적용 ──────────────────────────────────────
    logger.info("[Step 2] Applying WaRP-Rotated Safe LoRA projection...")
    warp_safelora = WaRPSafeLoRA(peft_model, config, logger=logger)
    projected_model = warp_safelora.model

    # ── 4. Stats 저장 ────────────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)
    stats_path = os.path.join(args.output_path, "warp_safelora_stats.json")
    stats = warp_safelora.stats
    # tensor → python 변환
    def _convert(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_convert(stats), f, indent=2, ensure_ascii=False)
    logger.info("[Step 3] Stats saved to: %s", stats_path)

    # ── 5. 어댑터 저장 (projected) ──────────────────────────────────────────
    adapter_save_path = os.path.join(args.output_path, "lora_adapter")
    projected_model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    logger.info("[Step 3] Projected LoRA adapter saved: %s", adapter_save_path)

    # ── 6. Merged full model 저장 ────────────────────────────────────────────
    if not args.no_merge:
        logger.info("[Step 4] Merging LoRA into base weights...")
        merged = projected_model.merge_and_unload()
        merged_path = os.path.join(args.output_path, "merged_model")
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        logger.info("[Step 4] Merged model saved: %s", merged_path)
    else:
        logger.info("[Step 4] Skipping merge (--no-merge set).")

    # ── 7. 메타데이터 저장 ────────────────────────────────────────────────────
    metadata = {
        "timestamp":         ts,
        "base_model":        args.base_model,
        "aligned_model":     args.aligned_model,
        "lora_adapter_path": args.lora_adapter_path,
        "basis_dir":         args.basis_dir,
        "top_k":             args.top_k,
        "top_k_ratio":       args.top_k_ratio,
        "select_type":       args.select_type,
        "threshold":         args.threshold,
        "num_proj_layers":   args.num_proj_layers,
        "use_approx":        args.use_approx,
        "num_projected":     stats["num_projected_layers"],
        "num_candidates":    stats["num_candidate_layers"],
        "log_path":          args.log_path,
    }
    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Done!  Output: %s", args.output_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
