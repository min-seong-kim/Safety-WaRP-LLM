"""
WaRP 열(column) 뉴런 파일 → Phase 3 마스크 디렉토리 변환기.

WSR-(R)SN-Tune 열 버전의 핵심 접착제다. `warp_sn_detection.py` 가 만든 5줄 뉴런 파일
(인덱스 = `basis_coeff = W @ U` 의 **열** = 입력 basis 방향)을, 저장소 본체의
`train.py --phase 3` 가 그대로 읽는 마스크 포맷으로 바꾼다:

    <masks_dir>/<layer_type>/layer_XX_mask.pt   = {'mask': torch.bool [out, in]}
    <masks_dir>/metadata.json

Phase 3 의 규약은 **mask=1 이 동결(stopgrad), mask=0 이 학습** 이다. 두 모드가 있다.

    --mode freeze : 선택된 열을 동결, 나머지 학습          → downstream(GSM8K) 단계
    --mode tune   : 선택된 열만 학습, 나머지 전부 동결      → (R)SN-Tune 단계

즉 같은 뉴런 파일에서 서로 보수(complement)인 두 마스크가 나온다. 이렇게 하면
RSN-Tune 단계와 downstream 단계 모두 기존 Phase 3 트레이너(검증된 LinearWaRP forward,
detach 마스크, restore 콜백)로 돌릴 수 있어, 이전 세대의 전용 트레이너
(`finetune_downstream_freeze_warp_sn.py`, 10.8 s/it · 178 GB OOM)가 필요 없다.

행렬 모양은 모델 config 에서 계산한다 (Llama 계열, GQA 포함):
    ffn_up   [intermediate, hidden]     ffn_down [hidden, intermediate]
    attn_q   [hidden, hidden]           attn_k/v [n_kv * head_dim, hidden]
열 인덱스는 항상 두 번째 축(in_features)이다. 범위를 벗어난 인덱스는 에러다 —
행 인덱스 파일(원공간 (R)SN)을 실수로 넣으면 ffn_up 에서 즉시 걸린다
(행은 intermediate 크기라 hidden 을 넘는 인덱스가 반드시 나온다).

사용:
    python -m sn_tune.warp_col_masks --neuron_file <critical.txt> \
        --model_name meta-llama/Llama-2-7b-chat-hf --mode freeze --output_dir <cell>/masks_freeze
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Set, Tuple

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sn_tune.neuron_file import (  # noqa: E402
    MODULE_ORDER,
    count_neurons,
    load_neuron_file,
    total_model_parameters_from_config,
)

# 뉴런 파일의 모듈 키 → Phase 1/2/3 의 layer_type 철자
MODULE_TO_LAYER_TYPE: Dict[str, str] = {
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
    "q": "attn_q",
    "k": "attn_k",
    "v": "attn_v",
}
LAYER_TYPE_TO_MODULE = {v: k for k, v in MODULE_TO_LAYER_TYPE.items()}


def module_shapes_from_config(cfg) -> Dict[str, Tuple[int, int]]:
    """layer_type → (out_features, in_features)."""
    hidden = int(cfg.hidden_size)
    inter = int(cfg.intermediate_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden // n_heads)
    return {
        "ffn_up": (inter, hidden),
        "ffn_down": (hidden, inter),
        "attn_q": (n_heads * head_dim, hidden),
        "attn_k": (n_kv * head_dim, hidden),
        "attn_v": (n_kv * head_dim, hidden),
    }


def build_masks(
    neurons: Dict[str, Dict[int, Set[int]]],
    shapes: Dict[str, Tuple[int, int]],
    num_layers: int,
    mode: str,
    layer_types: List[str],
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[str, Dict[str, int]]]:
    if mode not in ("freeze", "tune"):
        raise ValueError(f"mode 는 freeze|tune 이어야 한다: {mode!r}")

    masks: Dict[Tuple[int, str], torch.Tensor] = {}
    stats: Dict[str, Dict[str, int]] = {}

    for layer_type in layer_types:
        module = LAYER_TYPE_TO_MODULE[layer_type]
        out_f, in_f = shapes[layer_type]
        per_layer = neurons.get(module, {})
        n_cols = 0
        n_frozen = 0
        for layer_idx in range(num_layers):
            cols = sorted(int(c) for c in per_layer.get(layer_idx, set()))
            if cols and (cols[0] < 0 or cols[-1] >= in_f):
                raise ValueError(
                    f"{layer_type} layer {layer_idx}: 열 인덱스 범위 초과 "
                    f"(max={cols[-1]}, in_features={in_f}). 행 인덱스 파일(원공간)을 넣지 않았는지 확인."
                )
            if mode == "freeze":
                mask = torch.zeros(out_f, in_f, dtype=torch.bool)
                if cols:
                    mask[:, cols] = True
            else:  # tune: 선택 열만 학습
                mask = torch.ones(out_f, in_f, dtype=torch.bool)
                if cols:
                    mask[:, cols] = False
            masks[(layer_idx, layer_type)] = mask
            n_cols += len(cols)
            n_frozen += int(mask.sum().item())
        stats[layer_type] = {
            "out_features": out_f,
            "in_features": in_f,
            "layers": num_layers,
            "selected_columns": n_cols,
            "frozen_elems": n_frozen,
            "total_elems": out_f * in_f * num_layers,
        }
    return masks, stats


def save_masks(
    masks: Dict[Tuple[int, str], torch.Tensor],
    output_dir: str,
    metadata: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for (layer_idx, layer_type), mask in masks.items():
        d = os.path.join(output_dir, layer_type)
        os.makedirs(d, exist_ok=True)
        # Phase 2 와 동일한 포맷: {'mask': tensor}. Phase 3 는 bool 이 아니면 >0.5 로 이진화하므로
        # bool 로 저장해 두면 그대로 쓴다.
        torch.save({"mask": mask}, os.path.join(d, f"layer_{layer_idx:02d}_mask.pt"))
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neuron_file", required=True, help="WaRP 열 뉴런 파일 (5줄, 열 인덱스)")
    ap.add_argument("--model_name", required=True, help="HF id 또는 로컬 경로 (config 로 모양 계산)")
    ap.add_argument("--mode", choices=["freeze", "tune"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--layer_types", default="ffn_up,ffn_down,attn_q,attn_k,attn_v",
                    help="Phase 1/3 와 **동일**해야 한다")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    from transformers import AutoConfig

    if os.path.isdir(args.output_dir) and os.path.exists(os.path.join(args.output_dir, "metadata.json")):
        if not args.overwrite:
            print(f"[warp_col_masks] 이미 있다 (--overwrite 로 덮어쓰기): {args.output_dir}")
            return 0

    layer_types = [x.strip() for x in args.layer_types.split(",") if x.strip()]
    unknown = [lt for lt in layer_types if lt not in LAYER_TYPE_TO_MODULE]
    if unknown:
        raise ValueError(f"지원하지 않는 layer_type: {unknown}")

    cfg = AutoConfig.from_pretrained(args.model_name)
    shapes = module_shapes_from_config(cfg)
    num_layers = int(cfg.num_hidden_layers)

    neurons = load_neuron_file(args.neuron_file)
    counts = count_neurons(neurons)
    if counts["total"] == 0:
        raise ValueError(f"뉴런 파일이 비어 있다: {args.neuron_file}")

    masks, stats = build_masks(neurons, shapes, num_layers, args.mode, layer_types)

    frozen = sum(s["frozen_elems"] for s in stats.values())
    total = sum(s["total_elems"] for s in stats.values())
    selected_elems = sum(
        s["selected_columns"] * s["out_features"] for s in stats.values()
    )
    total_model_params = int(total_model_parameters_from_config(cfg))

    metadata = {
        "phase": 2,                      # Phase 3 load_masks 가 기대하는 필드들
        "keep_ratio": frozen / max(total, 1),
        "masking_strategy": f"warp_sn_column_{args.mode}",
        "layer_types": layer_types,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "two_mask": False,
        # 이 파일만의 정보
        "mask_unit": "column",
        "mode": args.mode,
        "mode_semantics": (
            "freeze: 선택 열 동결(mask=1), 나머지 학습" if args.mode == "freeze"
            else "tune: 선택 열만 학습(mask=0), 나머지 동결(mask=1)"
        ),
        "neuron_file": os.path.abspath(args.neuron_file),
        "model_name": args.model_name,
        "num_layers": num_layers,
        "neuron_counts": counts,
        "per_layer_type": stats,
        "frozen_elems": frozen,
        "warp_module_elems": total,
        "selected_column_elems": selected_elems,
        "selected_column_param_fraction_of_model": selected_elems / max(total_model_params, 1),
        "total_model_params": total_model_params,
    }
    save_masks(masks, args.output_dir, metadata)

    print(f"[warp_col_masks] mode={args.mode}  neuron_file={args.neuron_file}")
    print(f"  layer_types={layer_types}  layers={num_layers}  masks={len(masks)}")
    for lt, s in stats.items():
        print(f"  {lt:9s} cols={s['selected_columns']:6d}  frozen={s['frozen_elems']:>13,d}/{s['total_elems']:>13,d}"
              f"  ({100.0 * s['frozen_elems'] / s['total_elems']:.2f}%)")
    print(f"  선택 열 파라미터 = {selected_elems:,} = 모델 전체의 {100.0 * selected_elems / total_model_params:.4f}%")
    print(f"  동결 비율(WaRP 모듈 기준) = {100.0 * frozen / total:.2f}%  → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
