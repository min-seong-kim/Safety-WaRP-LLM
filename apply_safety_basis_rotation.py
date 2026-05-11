"""
apply_safety_basis_rotation.py

WaRP Phase 1 safety basis를 이용해 모델 가중치를 회전 변환한 후 저장합니다.
저장된 모델은 Safety-Neuron의 safety_neuron_detection_v2.py 에 그대로 입력할 수 있습니다.

────────────────────────────────────────────────────────────────────────
회전 변환 원리
────────────────────────────────────────────────────────────────────────
Phase 1 에서 safety 데이터로 수집한 Gram 행렬의 SVD:
    Gram = U_gram @ diag(S) @ U_gram^T
    V (= 'U' 필드) = right singular vectors [in_dim, in_dim]
    V[:,0] = safety 입력의 분산이 가장 큰 방향 (제 1 safety principal direction)

회전:
    W_new = W @ V^T            (W: [out, in],  V: [in, in])
    activation_new = W_new @ x = W @ V^T @ x

V^T @ x 를 계산하면:
    (V^T @ x)[0] = V[:,0] · x  ← safety direction 0 성분 (safety 입력에서 일관되게 큼)
    (V^T @ x)[1] = V[:,1] · x  ← safety direction 1 성분

결과:
  - Safety 입력이 들어올 때 특정 뉴런들이 일관되게 높은 activation을 가짐
  - SN detection (교집합 기반) 에서 더 신뢰할 수 있는 safety neuron 후보를 찾을 수 있음
  - 비회전 모델에서 safety direction 성분이 여러 입력 차원에 흩어져 있던 것이
    회전 후 앞쪽 차원에 집중되어 교집합 조건이 더 잘 충족됨

────────────────────────────────────────────────────────────────────────
주의
────────────────────────────────────────────────────────────────────────
- 회전된 모델은 원본과 출력이 다릅니다 (FFN SiLU 비선형성으로 인해 정확히 보존되지 않음).
- 탐지 목적으로만 사용하고, SN-Tune fine-tuning도 회전된 모델 기준으로 수행해야 합니다.
- attn (q/k/v) 는 선형 변환이므로 출력 보존 조건을 추가 변환으로 맞출 수 있지만,
  본 스크립트는 detection 전처리 목적이므로 그 처리를 생략합니다.

────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────
python apply_safety_basis_rotation.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --basis_dir ./checkpoints/phase1_20260505_164049/basis \
    --output_path ./Llama-2-7b-chat-hf-rotated_model_for_sn_detection \
    --layer_type ffn_up,ffn_down,attn_q,attn_k,attn_v \
    --device cuda

이후 SN detection:
cd /home/yonsei_jong/Safety-Neuron/neuron_detection
python safety_neuron_detection_v2.py 4994 \\
    --model_name /path/to/rotated_model_for_sn_detection \\
    --top_number_ffn 1200 --top_number_attn 200 \\
    --safety_neuron
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# layer_type → (sub-module 이름, weight 속성)
LAYER_TYPE_TO_MODULE = {
    "ffn_up":   ("mlp", "up_proj"),
    "ffn_gate": ("mlp", "gate_proj"),
    "ffn_down": ("mlp", "down_proj"),
    "attn_q":   ("self_attn", "q_proj"),
    "attn_k":   ("self_attn", "k_proj"),
    "attn_v":   ("self_attn", "v_proj"),
    "attn_o":   ("self_attn", "o_proj"),
}


def _get_linear(layer, layer_type: str) -> torch.nn.Linear:
    sub, name = LAYER_TYPE_TO_MODULE[layer_type]
    return getattr(getattr(layer, sub), name)


def _load_V(basis_dir: str, layer_type: str, layer_idx: int) -> torch.Tensor:
    """
    Phase 1 SVD 파일에서 V (right singular vectors) 로드.

    저장 규칙 (phase1_basis.py):
        U, S, Vh = torch.linalg.svd(gram, full_matrices=False)
        V = Vh.t()
        torch.save({'U': V, 'S': S, 'UT': Vh}, path)

    따라서 data['U'] = V [in_dim, in_dim],
             data['UT'] = Vh [in_dim, in_dim] = V^T
    Gram 이 정방행렬이므로 full_matrices=False 라도 V는 [h, h] full-rank.
    """
    path = os.path.join(basis_dir, layer_type, f"layer_{layer_idx:02d}_svd.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    V = data.get("U")
    if V is None:
        raise ValueError(f"'U' key missing in {path}")
    return V.float()  # [in_dim, in_dim]


def _apply_rotation(model, layer_idx: int, layer_type: str,
                    basis_dir: str, target_dtype: torch.dtype) -> bool:
    """
    한 (레이어, 타입) 쌍에 safety basis rotation 적용.

    W_new = W @ V^T
      - V[:,0] = 1st safety principal direction
      - V^T @ x: x 를 safety basis 로 투영 (safety 입력이면 첫 성분이 일관되게 큼)
      - 회전 후 safety 방향에 민감한 뉴런들이 safety prompt에서 더 일관된 activation을 가짐
    """
    try:
        V = _load_V(basis_dir, layer_type, layer_idx)  # [in_dim, in_dim]
    except FileNotFoundError:
        logger.debug(f"  [L{layer_idx}][{layer_type}] basis 파일 없음, 건너뜀")
        return False

    try:
        linear = _get_linear(model.model.layers[layer_idx], layer_type)
    except AttributeError as e:
        logger.warning(f"  [L{layer_idx}][{layer_type}] 모듈 접근 실패: {e}")
        return False

    W = linear.weight.data  # [out_dim, in_dim]
    V = V.to(W.device)
    if W.shape[1] != V.shape[0]:
        logger.warning(
            f"  [L{layer_idx}][{layer_type}] 차원 불일치: "
            f"W.shape={tuple(W.shape)}, V.shape={tuple(V.shape)}. 건너뜀."
        )
        return False

    # W_new = W @ V^T  →  activation_new = W @ V^T @ x
    # V^T @ x 의 첫 성분 = safety direction 0 성분 (safety 입력에서 일관되게 큼)
    W_new = (W.float() @ V).to(target_dtype)
    linear.weight.data = W_new
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply WaRP safety basis rotation for SN detection."
    )
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model ID 또는 로컬 경로")
    p.add_argument("--basis_dir", type=str, required=True,
                   help="Phase 1 basis 디렉토리 (하위에 ffn_up/, ffn_down/, ... 폴더 포함)")
    p.add_argument("--output_path", type=str, required=True,
                   help="회전된 모델 저장 경로")
    p.add_argument("--layer_type", type=str,
                   default="ffn_up,ffn_down,attn_q,attn_k,attn_v",
                   help="회전을 적용할 layer type (쉼표 구분). "
                        "ffn_gate 포함 시 FFN 전체 회전")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cpu",
                   help="모델 로드 device (cpu: GPU 메모리 절약, cuda: 빠름)")
    return p.parse_args()


def main():
    args = parse_args()

    layer_types = [lt.strip() for lt in args.layer_type.split(",")]
    for lt in layer_types:
        if lt not in LAYER_TYPE_TO_MODULE:
            raise ValueError(f"Unknown layer_type: '{lt}'. "
                             f"Choose from {list(LAYER_TYPE_TO_MODULE.keys())}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    target_dtype = dtype_map[args.dtype]

    logger.info("=" * 70)
    logger.info("Safety Basis Rotation for SN Detection")
    logger.info("=" * 70)
    logger.info(f"Model path  : {args.model_path}")
    logger.info(f"Basis dir   : {args.basis_dir}")
    logger.info(f"Output path : {args.output_path}")
    logger.info(f"Layer types : {layer_types}")
    logger.info(f"dtype       : {args.dtype}")
    logger.info(f"device      : {args.device}")
    logger.info("")
    logger.info("Rotation: W_new = W @ V^T")
    logger.info("  V[:,0] = safety 1st principal direction")
    logger.info("  (V^T @ x)[0] : safety 입력에서 일관되게 큼")
    logger.info("  → safety 뉴런이 safety prompt에 더 일관된 activation")
    logger.info("=" * 70)

    # ── 1. 모델 로드 ──────────────────────────────────────────────────
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=target_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    num_layers = len(model.model.layers)
    logger.info(f"✓ Model loaded: {num_layers} layers, dtype={model.dtype}")

    # ── 2. Rotation 적용 ──────────────────────────────────────────────
    logger.info("")
    logger.info("Applying safety basis rotation...")

    applied = 0
    skipped = 0

    for layer_idx in range(num_layers):
        for layer_type in layer_types:
            ok = _apply_rotation(model, layer_idx, layer_type,
                                 args.basis_dir, target_dtype)
            if ok:
                applied += 1
            else:
                skipped += 1

        if (layer_idx + 1) % 8 == 0 or layer_idx == num_layers - 1:
            logger.info(f"  {layer_idx + 1}/{num_layers} layers processed  "
                        f"(applied={applied}, skipped={skipped})")

    logger.info(f"✓ Rotation 완료: applied={applied}, skipped={skipped}")

    # ── 3. 저장 ──────────────────────────────────────────────────────
    logger.info("")
    os.makedirs(args.output_path, exist_ok=True)
    logger.info(f"Saving rotated model → {args.output_path}")
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)
    logger.info("✓ Model + tokenizer saved")

    # ── 4. 메타데이터 저장 ────────────────────────────────────────────
    metadata = {
        "rotation_formula": "W_new = W @ V^T",
        "V_description": (
            "V = right singular vectors of Gram(safety_activations), "
            "V[:,0] = most important safety principal direction"
        ),
        "effect": (
            "Safety inputs produce consistently high activations in safety-aligned neurons. "
            "SN detection (intersection across prompts) finds more reliable safety neurons."
        ),
        "model_path":          args.model_path,
        "basis_dir":           args.basis_dir,
        "layer_types_rotated": layer_types,
        "applied":             applied,
        "skipped":             skipped,
        "num_layers":          num_layers,
        "dtype":               args.dtype,
        "timestamp":           datetime.now().strftime("%Y%m%d_%H%M%S"),
        "next_step": (
            "cd Safety-Neuron/neuron_detection && "
            "python safety_neuron_detection_v2.py 4994 "
            f"--model_name {os.path.abspath(args.output_path)} "
            "--top_number_ffn 1200 --top_number_attn 200 --safety_neuron"
        ),
    }
    meta_path = os.path.join(args.output_path, "rotation_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("")
    logger.info("Next step — SN detection on rotated model:")
    logger.info(f"  cd /home/yonsei_jong/Safety-Neuron/neuron_detection")
    logger.info(f"  python safety_neuron_detection_v2.py 4994 \\")
    logger.info(f"      --model_name {os.path.abspath(args.output_path)} \\")
    logger.info(f"      --top_number_ffn 1200 --top_number_attn 200 \\")
    logger.info(f"      --safety_neuron")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
