"""
detect_warp_rotated.py — safety basis 로 회전된 입력으로 SN 뉴런 검출 (WSR-(R)SN-Tune 용).

원공간 검출(`detect_original.py`)과 **점수 공식이 같고 입력만 회전**한다:

    원공간   score[i] = Σ_{b,t} | (x  @ Wᵀ)_{b,t,i} |
    회전     score[i] = Σ_{b,t} | (x U @ Wᵀ)_{b,t,i} |      (M = U @ Wᵀ 로 1회 matmul)

따라서 선택되는 인덱스는 **행(출력 뉴런)** 이고, 원공간 SN 과 같은 대상이다 —
SN-Tune 과 1:1 로 비교된다. 모델의 forward/출력은 바뀌지 않고, 패치가 심어둔
`_last_*_score` 만 회전 버전으로 덮어쓴다.

⚠️ **패치된 transformers 가 필요하다.** 훅이 덮어쓰는 대상이 패치의 스태시이고,
basis 가 없는 layer_type 은 패치 점수를 그대로 쓰기 때문이다.

    bash sn_tune/setup_patched_transformers.sh            # 99MB 오버레이 (권장)
    PYTHONPATH=.transformers_patched python -m sn_tune.detect_warp_rotated ...
    # 또는 conda 환경째로: bash sn_tune/setup_hb_sn.sh (채널 ToS 동의 필요)

**행 마스크는 원공간과 등가다.** W̃ = W U 가 행을 보존하므로, W̃ 의 행 j 를 얼리고
학습한 뒤 W = W̃ Uᵀ 로 복원하면 행 j 는 정확히 보존된다. 그래서 이 검출 결과는
원공간 학습 도구(`sn_tune_original.py` / `finetune_freeze_sn.py`)로 그대로 쓴다 —
2026-05-05 `basis_rotation` 세대가 그렇게 했다.

사용
────
1) safety 뉴런
   PYTHONPATH=.transformers_patched python -m sn_tune.detect_warp_rotated 4994 \
       --model_name meta-llama/Llama-2-7b-chat-hf \
       --top_number_ffn 1200 --top_number_attn 200 \
       --use_basis_rotation_score --basis_dir <phase1>/basis \
       --safety_neuron --output_file <out>.txt

2) utility 뉴런 — `--utility_neuron` 으로 바꾸고 top-k 를 300/50 으로.
"""

import os
import argparse
from typing import Dict, Set, List, Tuple, Optional
import sys
import json
from tqdm import tqdm
import logging
import random
import math
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from sn_tune.neuron_file import (
    assert_nonempty,
    detected_parameter_count,
    save_neuron_file,
    total_model_neurons_from_config as calculate_total_model_neurons_from_config,
    total_model_parameters_from_config,
)
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from datasets import load_dataset

# 원본은 여기서 CUDA_VISIBLE_DEVICES 를 "7" 로 **하드 지정**해 셸 설정을 덮어썼다.
# 저장소 규칙대로 제거한다 — GPU 는 호출자가 정한다.

# 로거 초기 설정 (나중에 파일 핸들러 추가됨)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # = <repo>/sn_tune
GPU_INDEX = 0                                                    # --gpu 로 덮어쓴다
DEFAULT_SAFETY_CORPUS = os.path.join(_REPO_ROOT, "data", "circuit_breakers_train.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_neurons_warp")

random.seed(112)
torch.manual_seed(112)

# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------
def is_instruct_model(name: str) -> bool:
    name = name.lower()
    return ("instruct" in name) or ("chat" in name)

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
model_name = DEFAULT_MODEL_NAME
tokenizer = None
model = None
NUM_LAYERS = 0
ATTN_IMPLEMENTATION = "sdpa"

# ------------------------------------------------------------------
# Accelerated detection hyperparameters
# ------------------------------------------------------------------
DETAIL_LOG_PROMPT_LIMIT = 3
TOP_NUMBER_ATTN = 2000
TOP_NUMBER_FFN = 12000

# ------------------------------------------------------------------
# WSR-style basis-rotated scoring options
# ------------------------------------------------------------------
# The real model forward is NOT changed.  These hooks only replace the
# per-module score tensors used by the SN-Tune top-k/intersection detector.
# 원본 기본값은 예전 박스의 절대경로였다. --basis_dir 로 반드시 지정하라.
DEFAULT_BASIS_DIR = ""
USE_BASIS_ROTATION_SCORE = False
BASIS_DIR = DEFAULT_BASIS_DIR
BASIS_LAYER_TYPES = {"ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v"}
BASIS_HOOKS = []
BASIS_SCORE_MODULES = {}

LAYER_TYPE_TO_MODULE = {
    "ffn_up": ("mlp", "up_proj", "_last_ffn_up_score"),
    "ffn_down": ("mlp", "down_proj", "_last_ffn_down_score"),
    "attn_q": ("self_attn", "q_proj", "_last_q_score"),
    "attn_k": ("self_attn", "k_proj", "_last_k_score"),
    "attn_v": ("self_attn", "v_proj", "_last_v_score"),
}


def initialize_model_and_tokenizer(selected_model_name: str):
    """Initialize global model/tokenizer after CLI args are parsed."""
    global model_name, model, tokenizer, NUM_LAYERS

    model_name = selected_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "device_map": {"": GPU_INDEX},
        "torch_dtype": torch.bfloat16,
    }
    if ATTN_IMPLEMENTATION and ATTN_IMPLEMENTATION.lower() != "auto":
        load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        if "attn_implementation" in load_kwargs:
            logger.warning(
                f"Failed to load with attn_implementation={ATTN_IMPLEMENTATION} ({e}); retrying with library default."
            )
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            raise
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Safety neuron detection with configurable model and per-layer top-k"
    )
    parser.add_argument(
        "num_prompts",
        type=int,
        help="Number of samples to process (prompts for --safety_neuron, documents for --utility_neuron)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--top_number_ffn",
        type=int,
        default=TOP_NUMBER_FFN,
        help="Per-layer top-k for FFN neuron selection",
    )
    parser.add_argument(
        "--top_number_attn",
        type=int,
        default=TOP_NUMBER_ATTN,
        help="Per-layer top-k for attention neuron selection",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        help=(
            "Attention kernel selection for model forward. "
            "Use sdpa or flash_attention_2 for speed; avoid eager for large runs."
        ),
    )
    parser.add_argument(
        "--use_basis_rotation_score",
        action="store_true",
        help=(
            "Use WSR-style safety-basis-rotated inputs only for neuron scoring. "
            "The model forward/output is not modified."
        ),
    )
    parser.add_argument(
        "--basis_dir",
        type=str,
        default=DEFAULT_BASIS_DIR,
        help="Phase 1 safety basis directory containing ffn_up/, attn_q/, ... subfolders.",
    )
    parser.add_argument(
        "--basis_layer_types",
        type=str,
        default="ffn_up,ffn_down,attn_q,attn_k,attn_v",
        help="Comma-separated layer types where basis-rotated scoring is applied.",
    )

    parser.add_argument(
        "--dataset_file", type=str, default=None,
        help=f"safety 코퍼스 JSON (--safety_neuron 용). 기본 {DEFAULT_SAFETY_CORPUS}",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="검출 결과 디렉토리 (기본 sn_tune/output_neurons_warp)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="정확한 출력 경로. --output_dir 와 자동 파일명을 덮어쓴다.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA_VISIBLE_DEVICES 안에서의 인덱스 (기본 0)",
    )
    parser.add_argument(
        "--skip_patch_check", action="store_true",
        help="패치 검사 생략 (권장하지 않음 — 빈 뉴런 파일이 나온다)",
    )

    parser.add_argument(
        "--utility_json", type=str, default=None,
        help=("utility 코퍼스를 Hub 에서 받는 대신 이 JSON 에서 읽는다 "
              '([{"prompt": ...}] 형식, sn_tune/build_utility_corpus.py 산출물). '
              "오프라인 환경에서 필요하고, 행/열 경로가 같은 문서를 보게 하는 효과도 있다."),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--safety_neuron",
        action="store_true",
        help="Detect safety neurons from circuit_breakers_train.json",
    )
    mode_group.add_argument(
        "--utility_neuron",
        action="store_true",
        help="Detect utility neurons from Wikipedia dataset",
    )

    args = parser.parse_args(argv)

    if args.top_number_ffn <= 0:
        parser.error("--top_number_ffn must be a positive integer.")
    if args.top_number_attn <= 0:
        parser.error("--top_number_attn must be a positive integer.")

    return args



def _basis_file_candidates(basis_dir: str, layer_type: str, layer_idx: int) -> List[str]:
    """Return likely Phase-1 basis file paths for both 0- and 1-indexed naming."""
    return [
        os.path.join(basis_dir, layer_type, f"layer_{layer_idx:02d}_svd.pt"),
        os.path.join(basis_dir, layer_type, f"layer_{layer_idx + 1:02d}_svd.pt"),
    ]


def _load_safety_basis(basis_dir: str, layer_type: str, layer_idx: int) -> Optional[torch.Tensor]:
    """Load Phase-1 safety basis U for one layer/module.

    The saved file is expected to contain data['U'] with shape [in_dim, in_dim].
    This function keeps U on CPU in float32; the hook moves it to the module device.
    """
    for path in _basis_file_candidates(basis_dir, layer_type, layer_idx):
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            U = data.get("U")
            if U is None:
                raise ValueError(f"Missing key 'U' in basis file: {path}")
            return U.float().contiguous()
    return None


def _score_from_rotated_input_precomputed(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Compute SN-style activation score using precomputed M = U @ W.T.

    score[i] = sum_{b,t} |( x @ M )_{b,t,i}|
    where M = U @ W.T  [in_features, out_features] is precomputed once.

    Reduces 2 matmuls (x@U, then @W.T) to 1 matmul (x @ M).
    x 는 모델의 bfloat16 그대로 사용 — float32 형변환 없음.
    """
    # x: [B, T, in_features], M: [in_features, out_features] (same device, bfloat16)
    out_rot = x.detach() @ M          # [B, T, out_features]
    return out_rot.abs().sum(dim=(0, 1)).float()  # [out_features] on GPU


def _register_basis_rotation_score_hooks() -> None:
    """Register forward hooks that overwrite patched modeling_llama score tensors.

    The hooks do not alter module outputs.  They only replace _last_*_score
    after each projection module has run, allowing the rest of the SN-Tune
    detector, including per-prompt top-k and exact intersection, to remain
    unchanged.

    속도 최적화:
    1. U 를 hook 등록 시 GPU 로 미리 이동 (매 호출마다 host→device 전송 제거)
    2. M = U @ W.T 를 사전 계산 (hook 당 matmul 2개 → 1개)
    3. bfloat16 그대로 사용 (float32 형변환 제거)
    """
    global BASIS_HOOKS, BASIS_SCORE_MODULES

    # Remove stale hooks if this function is called more than once.
    for h in BASIS_HOOKS:
        try:
            h.remove()
        except Exception:
            pass
    BASIS_HOOKS = []
    BASIS_SCORE_MODULES = {}

    loaded = 0
    skipped = 0
    mismatched = 0

    device = next(model.parameters()).device

    for layer_idx in range(NUM_LAYERS):
        layer = model.model.layers[layer_idx]
        for layer_type in sorted(BASIS_LAYER_TYPES):
            if layer_type not in LAYER_TYPE_TO_MODULE:
                continue

            sub_name, proj_name, score_attr = LAYER_TYPE_TO_MODULE[layer_type]
            sub = getattr(layer, sub_name, None)
            proj = getattr(sub, proj_name, None) if sub is not None else None
            if proj is None or not hasattr(proj, "weight"):
                skipped += 1
                continue

            U_cpu = _load_safety_basis(BASIS_DIR, layer_type, layer_idx)
            if U_cpu is None:
                skipped += 1
                continue
            if proj.weight.shape[1] != U_cpu.shape[0]:
                logger.warning(
                    f"[BasisScore] Skip L{layer_idx} {layer_type}: "
                    f"weight={tuple(proj.weight.shape)}, U={tuple(U_cpu.shape)}"
                )
                mismatched += 1
                continue

            # ── 최적화 1: GPU 로 미리 이동 + bfloat16 으로 변환 ──────────
            # ── 최적화 2: M = U @ W.T 사전 계산 (hook 당 matmul 1회로 감소) ─
            with torch.no_grad():
                U_gpu  = U_cpu.to(device=device, dtype=proj.weight.dtype)    # [in, in]
                W      = proj.weight.detach()                                 # [out, in]
                M      = U_gpu @ W.T                                          # [in, out]

            def _make_hook(parent_module, attr_name, M_saved, lidx, ltype):
                def hook(linear_module, inputs, output):
                    try:
                        score = _score_from_rotated_input_precomputed(inputs[0], M_saved)
                        setattr(parent_module, attr_name, score)
                    except Exception as e:
                        logger.error(f"[BasisScore] Failed at layer={lidx}, type={ltype}: {e}")
                        setattr(parent_module, attr_name, None)
                return hook

            BASIS_HOOKS.append(proj.register_forward_hook(_make_hook(sub, score_attr, M, layer_idx, layer_type)))
            BASIS_SCORE_MODULES[(layer_idx, layer_type)] = True
            loaded += 1

    logger.info("=" * 70)
    logger.info("WSR-style basis-rotated scoring hooks registered")
    logger.info(f"  - basis_dir: {BASIS_DIR}")
    logger.info(f"  - layer_types: {sorted(BASIS_LAYER_TYPES)}")
    logger.info(f"  - hooks loaded: {loaded}")
    logger.info(f"  - skipped: {skipped}")
    logger.info(f"  - mismatched: {mismatched}")
    logger.info("  - model forward/output is unchanged; only _last_*_score is overwritten")
    logger.info("  - [opt] M=U@W.T precomputed; U cached on GPU (no per-prompt transfers)")
    logger.info("=" * 70)


def calculate_model_total_neurons() -> int:
    """
    Same denominator as calculate_safety_neuron_percentage.py:
    q/k/v/o + gate/up/down output channels across all layers.
    """
    return calculate_total_model_neurons_from_config(model.config)

def should_log_detail(prompt_idx: int) -> bool:
    return prompt_idx < DETAIL_LOG_PROMPT_LIMIT


def log_tensor_stats(name: str, tensor: Optional[torch.Tensor], prompt_idx: int, layer_idx: int):
    if tensor is None:
        logger.debug(f"[Prompt {prompt_idx}][Layer {layer_idx}] {name}: None")
        return

    try:
        t = tensor.detach().float()
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        logger.debug(
            f"[Prompt {prompt_idx}][Layer {layer_idx}] {name}: "
            f"shape={tuple(t.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
            f"min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}, "
            f"nan={nan_count}, inf={inf_count}"
        )
    except Exception as e:
        logger.debug(f"[Prompt {prompt_idx}][Layer {layer_idx}] {name}: stats failed: {e}")

def get_attention_metadata(attn_module):
    """
    Robustly extract attention metadata across different HF LlamaAttention implementations.
    """
    cfg = getattr(attn_module, "config", None)
    if cfg is None:
        cfg = model.config

    # num_heads
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        raise RuntimeError("Cannot determine num_heads from attention module or config.")

    # num_kv_heads
    num_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if num_kv_heads is None:
        # fallback: infer from k_proj output shape
        k_out = attn_module.k_proj.weight.shape[0]
        q_out = attn_module.q_proj.weight.shape[0]
        inferred_head_dim = q_out // num_heads
        num_kv_heads = k_out // inferred_head_dim

    # head_dim
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        q_out = attn_module.q_proj.weight.shape[0]
        if q_out % num_heads != 0:
            raise RuntimeError(
                f"q_proj out_features ({q_out}) is not divisible by num_heads ({num_heads})."
            )
        head_dim = q_out // num_heads

    if num_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})."
        )

    num_kv_groups = num_heads // num_kv_heads

    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_kv_groups": num_kv_groups,
    }

def repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [B, T, H_kv, D]
    return: [B, T, H_q, D]
    """
    if n_rep == 1:
        return x
    bsz, seqlen, num_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(bsz, seqlen, num_kv_heads, n_rep, head_dim)
    return x.reshape(bsz, seqlen, num_kv_heads * n_rep, head_dim)


def build_causal_mask_for_query_subset(
    seq_len: int,
    query_start: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns bool mask of shape [Q, T]
    where Q = seq_len - query_start
    """
    q_positions = torch.arange(query_start, seq_len, device=device)  # [Q]
    k_positions = torch.arange(seq_len, device=device)               # [T]
    return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)      # [Q, T]


def select_topk_indices(score: Optional[torch.Tensor], top_k: int) -> Set[int]:
    """Select top-k indices from a 1D score vector."""
    if score is None:
        return set()

    if not isinstance(score, torch.Tensor):
        score = torch.tensor(score, dtype=torch.float32, device=next(model.parameters()).device)

    vec = score.detach().view(-1)
    if vec.numel() == 0:
        return set()

    if vec.device.type != "cuda":
        vec = vec.to(next(model.parameters()).device, non_blocking=True)

    vec = vec.float()
    k = min(top_k, int(vec.numel()))
    if k <= 0:
        return set()

    _, indices = torch.topk(vec, k=k, largest=True, sorted=False)
    return {int(idx) for idx in indices.detach().cpu().tolist()}


def detect_safety_neurons_threshold(
    prompt: str,
    prompt_idx: int = 0,
) -> Optional[
    Tuple[
        Dict[int, Set[int]],
        Dict[int, Set[int]],
        Dict[int, Set[int]],
        Dict[int, Set[int]],
        Dict[int, Set[int]],
    ]
]:
    """Original GitHub style detection: use forward-stored scores + per-layer fixed top-k."""
    ffn_up_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    ffn_down_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    q_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    k_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    v_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    try:
        # ------------------------------------------------------------
        # 1) Tokenize
        # ------------------------------------------------------------
        if is_instruct_model(model_name):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {"input_ids": input_ids}
        else:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        seq_len = inputs["input_ids"].shape[1]
        logger.debug(
            f"[Prompt {prompt_idx}] tokenized: seq_len={seq_len}, "
            f"has_attention_mask={'attention_mask' in inputs}, device={device}"
        )

        # --------------------------------------------------------
        # 2) Run real model forward once to populate module-internal scores
        # --------------------------------------------------------
        with torch.no_grad():
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=False,
                return_dict=True,
            )

        # --------------------------------------------------------
        # 3) Per-layer fixed top-k selection from forward-stored scores
        # --------------------------------------------------------
        for layer_idx in range(NUM_LAYERS):
            layer = model.model.layers[layer_idx]

            ffn_up_score = getattr(layer.mlp, "_last_ffn_up_score", None)
            ffn_down_score = getattr(layer.mlp, "_last_ffn_down_score", None)
            q_score = getattr(layer.self_attn, "_last_q_score", None)
            k_score = getattr(layer.self_attn, "_last_k_score", None)
            v_score = getattr(layer.self_attn, "_last_v_score", None)

            if any(score is None for score in [ffn_up_score, ffn_down_score, q_score, k_score, v_score]):
                raise RuntimeError(
                    f"Missing forward-captured score tensor(s) at layer {layer_idx}. "
                    "Ensure patched modeling_llama.py is loaded."
                )

            ffn_up_dict[layer_idx] = select_topk_indices(ffn_up_score, TOP_NUMBER_FFN)
            ffn_down_dict[layer_idx] = select_topk_indices(ffn_down_score, TOP_NUMBER_FFN)
            q_dict[layer_idx] = select_topk_indices(q_score, TOP_NUMBER_ATTN)
            k_dict[layer_idx] = select_topk_indices(k_score, TOP_NUMBER_ATTN)
            v_dict[layer_idx] = select_topk_indices(v_score, TOP_NUMBER_ATTN)

        if should_log_detail(prompt_idx):
            ffn_up_total = sum(len(v) for v in ffn_up_dict.values())
            ffn_down_total = sum(len(v) for v in ffn_down_dict.values())
            q_total = sum(len(v) for v in q_dict.values())
            k_total = sum(len(v) for v in k_dict.values())
            v_total = sum(len(v) for v in v_dict.values())

            logger.debug(
                f"[Prompt {prompt_idx}] selected neurons by top-k: "
                f"ffn_up={ffn_up_total}, ffn_down={ffn_down_total}, "
                f"q={q_total}, k={k_total}, v={v_total}"
            )

    except Exception as e:
        logger.exception(f"Error in neuron detection (Prompt {prompt_idx}): {e}")
        return None

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


def compute_intersection(
    neuron_sets_list: List[Dict[int, Set[int]]],
    module_name: str = "module"
) -> Dict[int, Set[int]]:
    """
    Compute exact intersection across all prompts (Eq. 3).

    Eq. (3): N_safe = ⋂_{x in X} Nx
    - A neuron must appear in EVERY prompt-specific set Nx.
    - If any prompt has an empty set at a layer, intersection becomes empty.
    """
    if not neuron_sets_list:
        logger.info(f"[compute_intersection][{module_name}] no neuron sets; reduced=0")
        return {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    intersection_dict: Dict[int, Set[int]] = {}
    before_union_total = 0
    after_intersection_total = 0

    for layer_idx in range(NUM_LAYERS):
        layer_sets = [
            neuron_dict.get(layer_idx, set())
            for neuron_dict in neuron_sets_list
        ]

        # Union is just for logging/diagnostics
        union_set = set().union(*layer_sets) if layer_sets else set()

        # Exact intersection across ALL prompts
        if not layer_sets:
            common = set()
        else:
            common = set(layer_sets[0])
            for s in layer_sets[1:]:
                common &= s

        before_union_total += len(union_set)
        after_intersection_total += len(common)
        intersection_dict[layer_idx] = common

    reduced = before_union_total - after_intersection_total
    logger.info(
        f"[compute_intersection][{module_name}] prompts={len(neuron_sets_list)}, "
        f"before(union)={before_union_total}, after(intersection)={after_intersection_total}, reduced={reduced}"
    )

    return intersection_dict


def load_utility_texts_from_json(path: str, num_samples: int) -> List[str]:
    """미리 덤프해 둔 utility 코퍼스 JSON 을 읽는다.

    `build_utility_corpus.py` 가 `detect_*.load_wikipedia_data` 와 같은 seed(112)·
    같은 자르기 길이(2000자)로 만들어 두므로, Hub 에서 직접 받는 것과 같은 문서다.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r.get("prompt", "") for r in records if r.get("prompt")]
    if len(texts) > num_samples:
        texts = texts[:num_samples]
    logger.info(f"Loaded {len(texts)} utility documents from {path}")
    return texts


def load_wikipedia_data(num_samples: int = 1000) -> List[str]:
    """Load Wikipedia data from Hugging Face."""
    logger.info("Loading Wikipedia dataset (subset: 20231101.en)...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=False,
            cache_dir=os.path.join(SCRIPT_DIR, "wikipedia_cache"),
        )

        total_size = len(dataset)
        random.seed(112)
        random_indices = random.sample(range(total_size), min(num_samples, total_size))

        texts = []
        for idx in tqdm(random_indices, desc="Loading Wikipedia docs"):
            try:
                item = dataset[idx]
                text = item.get("text", "").strip()
                if text:
                    texts.append(text[:2000])
            except Exception as e:
                logger.warning(f"Failed to load Wikipedia doc {idx}: {e}")

        logger.info(f"Successfully loaded {len(texts)} Wikipedia samples")
        return texts

    except Exception as e:
        logger.error(f"Error loading Wikipedia dataset: {e}")
        raise


def main(argv):
    global TOP_NUMBER_FFN, TOP_NUMBER_ATTN
    global USE_BASIS_ROTATION_SCORE, BASIS_DIR, BASIS_LAYER_TYPES, ATTN_IMPLEMENTATION

    args = parse_args(argv)

    TOP_NUMBER_FFN = args.top_number_ffn
    TOP_NUMBER_ATTN = args.top_number_attn
    ATTN_IMPLEMENTATION = args.attn_implementation
    USE_BASIS_ROTATION_SCORE = args.use_basis_rotation_score
    BASIS_DIR = args.basis_dir
    globals()["GPU_INDEX"] = args.gpu

    if args.use_basis_rotation_score and not BASIS_DIR:
        logger.error("--use_basis_rotation_score 를 쓰려면 --basis_dir 이 필요하다.")
        sys.exit(1)

    # 패치 검사를 모델 로딩보다 **먼저**. 정품으로 돌리면 몇 시간 뒤 빈 파일만 남는다.
    if not args.skip_patch_check:
        from sn_tune.verify_patch import check, family_from_model_name
        if check(family_from_model_name(args.model_name), "present") != 0:
            logger.error(
                "패치된 transformers 가 아니다. 이 검출기는 패치의 _last_*_score 를 덮어쓰는 "
                "구조라 정품에서는 동작하지 않는다.\n"
                "  bash sn_tune/setup_patched_transformers.sh\n"
                "  PYTHONPATH=.transformers_patched python -m sn_tune.detect_warp_rotated ...\n"
                "정말 이대로 진행하려면 --skip_patch_check 를 주어라."
            )
            sys.exit(1)
    BASIS_LAYER_TYPES = {x.strip() for x in args.basis_layer_types.split(",") if x.strip()}

    initialize_model_and_tokenizer(args.model_name)

    # =====================================================================
    # 로깅 설정: 파일 핸들러 추가
    # =====================================================================
    log_dir = os.path.join(SCRIPT_DIR, "logs", "neuron_detection")
    os.makedirs(log_dir, exist_ok=True)

    # 파일 이름: 현재 날짜 및 시간
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = "safety_neuron" if args.safety_neuron else "utility_neuron"
    log_file = os.path.join(log_dir, f"{log_prefix}_{log_timestamp}.log")
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러도 추가 (기존 출력 유지)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"ATTN_IMPLEMENTATION: {ATTN_IMPLEMENTATION}")
    logger.info(f"TOP_NUMBER_FFN: {TOP_NUMBER_FFN}, TOP_NUMBER_ATTN: {TOP_NUMBER_ATTN}")
    logger.info(f"USE_BASIS_ROTATION_SCORE: {USE_BASIS_ROTATION_SCORE}")
    if USE_BASIS_ROTATION_SCORE:
        logger.info(f"BASIS_DIR: {BASIS_DIR}")
        logger.info(f"BASIS_LAYER_TYPES: {sorted(BASIS_LAYER_TYPES)}")
        _register_basis_rotation_score_hooks()

    num_samples = args.num_prompts

    if args.safety_neuron:
        logger.info("[Mode] Safety Neuron Detection")
        logger.info(f"Number of prompts to process: {num_samples}")
        file_path = args.dataset_file or DEFAULT_SAFETY_CORPUS
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            logger.error(f"No valid 'prompt' entries found in: {file_path}")
            sys.exit(1)

        if len(records) > num_samples:
            records = records[:num_samples]

        lines = [item.get("prompt", "") for item in records]
        logger.info(f"Processing {len(lines)} prompts from {file_path}")

    else:  # args.utility_neuron
        logger.info("[Mode] Utility Neuron Detection (Wikipedia)")
        logger.info(f"Number of Wikipedia documents to process: {num_samples}")
        if args.utility_json:
            lines = load_utility_texts_from_json(args.utility_json, num_samples)
        else:
            lines = load_wikipedia_data(num_samples=num_samples)
        if not lines:
            logger.error("Failed to load Wikipedia data")
            sys.exit(1)
        logger.info(f"Processing {len(lines)} Wikipedia documents")

    # 각 prompt x에 대해 Nx를 수집
    ffn_up_sets: List[Dict[int, Set[int]]] = []
    ffn_down_sets: List[Dict[int, Set[int]]] = []
    q_sets: List[Dict[int, Set[int]]] = []
    k_sets: List[Dict[int, Set[int]]] = []
    v_sets: List[Dict[int, Set[int]]] = []

    failed_count = 0
    successful_count = 0

    for idx, prompt in enumerate(tqdm(lines, desc="Detecting neurons")):
        result = detect_safety_neurons_threshold(prompt, prompt_idx=idx)

        if result is None:
            failed_count += 1
            logger.warning(f"Failed prompt idx={idx}")
            continue

        ffn_up, ffn_down, q, k, v = result
        ffn_up_sets.append(ffn_up)
        ffn_down_sets.append(ffn_down)
        q_sets.append(q)
        k_sets.append(k)
        v_sets.append(v)
        successful_count += 1
    logger.info(f"Detection complete: success={successful_count}, failed={failed_count}")

    # Eq. (3): N_safe = ⋂_x N_x
    ffn_up_common = compute_intersection(ffn_up_sets, module_name="ffn_up")
    ffn_down_common = compute_intersection(ffn_down_sets, module_name="ffn_down")
    q_common = compute_intersection(q_sets, module_name="q")
    k_common = compute_intersection(k_sets, module_name="k")
    v_common = compute_intersection(v_sets, module_name="v")

    # 결과 저장 — 5줄 포맷. 줄 순서가 곧 계약이다 (sn_tune/neuron_file.py).
    if args.output_file:
        output_file = args.output_file
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if args.safety_neuron:
            output_file = os.path.join(output_dir, f"safety_neuron_accelerated_{log_timestamp}.txt")
        else:
            output_file = os.path.join(output_dir, f"utility_neurons_{len(ffn_up_sets)}_{log_timestamp}.txt")

    detected = {
        "ffn_up": ffn_up_common, "ffn_down": ffn_down_common,
        "q": q_common, "k": k_common, "v": v_common,
    }
    save_neuron_file(output_file, detected)

    # 최종 결과 계산
    total_safety_neurons = 0
    for layer_idx in range(NUM_LAYERS):
        ffn_up_count = len(ffn_up_common.get(layer_idx, set()))
        ffn_down_count = len(ffn_down_common.get(layer_idx, set()))
        q_count = len(q_common.get(layer_idx, set()))
        k_count = len(k_common.get(layer_idx, set()))
        v_count = len(v_common.get(layer_idx, set()))
        total_safety_neurons += ffn_up_count + ffn_down_count + q_count + k_count + v_count

    total_model_neurons = calculate_model_total_neurons()
    actual_sparsity = total_safety_neurons / total_model_neurons if total_model_neurons > 0 else 0

    # 논문의 "≤1%" 는 **파라미터** 기준이다. 뉴런 비율과 다르니 둘 다 찍는다.
    param_pct = None
    try:
        param_pct = 100.0 * detected_parameter_count(detected, model.config) \
            / total_model_parameters_from_config(model.config)
    except Exception as exc:
        logger.warning(f"파라미터 비율 계산 실패: {type(exc).__name__}: {exc}")
    
    mode_label = "Safety" if args.safety_neuron else "Utility"
    logger.info(f"\n{'='*70}")
    logger.info(f"{mode_label} Neuron Detection Results")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total {mode_label.lower()} neurons: {total_safety_neurons:,}")
    logger.info(f"Total model neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    logger.info(f"Detected {mode_label.lower()} neuron percentage: {actual_sparsity*100:.4f}%  (뉴런 기준)")
    if param_pct is not None:
        logger.info(f"Detected {mode_label.lower()} PARAMETER percentage: {param_pct:.4f}%  "
                    f"← 논문 기준(≤1%)은 이 값이다.")
    logger.info(f"Output: {output_file}")
    logger.info(f"Log: {log_file}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
