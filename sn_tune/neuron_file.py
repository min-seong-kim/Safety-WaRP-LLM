"""
neuron_file.py — safety neuron 파일(5줄 포맷)의 읽기/쓰기와 비율 계산.

원공간 검출기와 WaRP 공간 검출기가 **같은 포맷**을 쓰므로 이 모듈은 양쪽 공용이다.
(WaRP 쪽 인덱스는 basis_coeff 의 '열' 을 뜻한다는 점만 다르다 — 의미는 다르고 포맷은 같다.)

포맷 — 순서가 곧 계약이다
─────────────────────────
5줄, 각 줄이 `{layer_idx: [neuron_indices]}` 형태의 JSON/파이썬-repr dict.

    line 0: ffn_up     line 1: ffn_down     line 2: q     line 3: k     line 4: v

**파일에서 키를 읽지 않는다. 줄 위치로만 배정한다.** 코드상의 dict 키도
`ffn_up, ffn_down, q, k, v` 이지 `attn_q/attn_k/attn_v` 가 아니다
(WaRP 쪽 `--layer_type` 의 `attn_*` 철자와 혼동하지 말 것).

주의: 이 포맷을 읽는 쪽이 모르는 키를 조용히 건너뛰므로, 키를 잘못 쓰면
에러가 아니라 **0개로 집계**된다. 5줄 구조를 절대 깨지 마라.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set

# 줄 순서 = 계약. 절대 재정렬하지 말 것.
MODULE_ORDER: tuple = ("ffn_up", "ffn_down", "q", "k", "v")

# 5줄 포맷의 키 ↔ WaRP 쪽 layer_type 철자 대응
WARP_LAYER_TYPE = {
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
    "q": "attn_q",
    "k": "attn_k",
    "v": "attn_v",
}

LLAMA31_8B_FALLBACK_DIMS = {
    "num_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}


# ---------------------------------------------------------------------------
# 읽기 / 쓰기
# ---------------------------------------------------------------------------

def _parse_dict_line(raw: str) -> Dict:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def load_neuron_file(path: str) -> Dict[str, Dict[int, Set[int]]]:
    """5줄 뉴런 파일 → {module: {layer_idx: set(indices)}}.

    줄이 5개보다 적으면 없는 모듈은 빈 dict 가 된다 (에러가 아니다 — 원본 동작과 동일).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"뉴런 파일이 없다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out: Dict[str, Dict[int, Set[int]]] = {}
    for i, module in enumerate(MODULE_ORDER):
        raw = lines[i] if i < len(lines) else ""
        parsed = _parse_dict_line(raw)
        out[module] = {int(k): set(v) for k, v in parsed.items()}
    return out


def save_neuron_file(path: str, neurons: Dict[str, Dict[int, Iterable[int]]]) -> None:
    """{module: {layer_idx: indices}} → 5줄 뉴런 파일. 인덱스는 정렬해서 쓴다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for module in MODULE_ORDER:
            layer_map = neurons.get(module, {})
            serialized = {str(int(k)): sorted(int(i) for i in v)
                          for k, v in sorted(layer_map.items(), key=lambda kv: int(kv[0]))}
            f.write(json.dumps(serialized) + "\n")


def count_neurons(neurons: Dict[str, Dict[int, Set[int]]]) -> Dict[str, int]:
    """모듈별/전체 뉴런 개수. 검출이 빈 파일을 뱉었는지 확인하는 용도."""
    counts = {m: sum(len(v) for v in neurons.get(m, {}).values()) for m in MODULE_ORDER}
    counts["ffn"] = counts["ffn_up"] + counts["ffn_down"]
    counts["attn"] = counts["q"] + counts["k"] + counts["v"]
    counts["total"] = counts["ffn"] + counts["attn"]
    return counts


def assert_nonempty(path: str, neurons: Optional[Dict] = None) -> Dict[str, int]:
    """검출 결과가 비어 있으면 즉시 실패시킨다.

    원본 검출기는 프롬프트마다 try/except 로 감싸여 있어 **100% 실패해도 exit 0** 이고
    `{"0": [], "1": [], ...}` 를 써버린다. 그 빈 파일은 몇 시간 뒤 학습 단계에서
    `ValueError: optimizer got an empty parameter list` 로야 드러난다. 여기서 끊는다.
    """
    neurons = neurons if neurons is not None else load_neuron_file(path)
    counts = count_neurons(neurons)
    if counts["total"] == 0:
        raise ValueError(
            f"뉴런 파일이 비었다: {path}\n"
            "검출이 전부 실패했다는 뜻이다. 거의 항상 원인은 패치되지 않은 transformers 다.\n"
            "  python -m sn_tune.verify_patch --model_family <llama|gemma2|qwen2>\n"
            "검출 로그의 'Detection complete: success=N, failed=M' 줄도 확인하라."
        )
    return counts


# ---------------------------------------------------------------------------
# 비율 계산 (모델 전체 대비 몇 % 인가 — 논문 기준 ≤1%)
# ---------------------------------------------------------------------------

def total_model_neurons_from_dims(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int] = None,
) -> int:
    """모델 전체 뉴런 분모: 레이어마다 q/k/v/o + gate/up/down 의 출력 채널 수."""
    num_kv_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_kv_heads * head_dim
    return num_layers * (
        hidden_size        # q
        + kv_dim           # k
        + kv_dim           # v
        + hidden_size      # o
        + intermediate_size  # gate
        + intermediate_size  # up
        + hidden_size      # down
    )


def total_model_neurons_from_config(cfg: Any) -> int:
    return total_model_neurons_from_dims(
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    )


def detected_parameter_count(neurons: Dict[str, Dict[int, Set[int]]], cfg: Any) -> int:
    """선택된 뉴런이 차지하는 파라미터 수. 인덱스 하나 = 가중치 한 줄(전체 열/행)."""
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    per_neuron = {
        "ffn_up": intermediate_size,
        "ffn_down": hidden_size,
        "q": hidden_size,
        "k": kv_dim,
        "v": kv_dim,
    }
    total = 0
    for module, layer_map in neurons.items():
        unit = per_neuron.get(module)
        if unit is None:      # 모르는 키는 0 으로 집계된다 — 모듈 docstring 참고
            continue
        total += sum(len(idx) * unit for idx in layer_map.values())
    return total


def total_model_parameters_from_config(cfg: Any) -> int:
    """config 로부터 전체 파라미터 수 추정 (LLaMA 계열 구조 가정)."""
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    per_layer = (
        hidden_size * hidden_size          # q_proj
        + kv_dim * hidden_size             # k_proj
        + kv_dim * hidden_size             # v_proj
        + hidden_size * hidden_size        # o_proj
        + intermediate_size * hidden_size  # gate_proj
        + intermediate_size * hidden_size  # up_proj
        + hidden_size * intermediate_size  # down_proj
        + 2 * hidden_size                  # layernorm ×2
    )
    embedding = vocab_size * hidden_size
    lm_head = 0 if getattr(cfg, "tie_word_embeddings", True) else hidden_size * vocab_size
    return embedding + num_layers * per_layer + hidden_size + lm_head


def summarize(path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """뉴런 파일 요약: 개수 + (모델을 알면) 뉴런/파라미터 비율."""
    neurons = load_neuron_file(path)
    out: Dict[str, Any] = {"path": path, "counts": count_neurons(neurons)}

    if model_name:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        total_neurons = total_model_neurons_from_config(cfg)
        total_params = total_model_parameters_from_config(cfg)
        det_params = detected_parameter_count(neurons, cfg)
        out["model_name"] = model_name
        out["neuron_pct"] = 100.0 * out["counts"]["total"] / total_neurons
        out["param_pct"] = 100.0 * det_params / total_params
        out["detected_params"] = det_params
        out["total_params"] = total_params
    return out


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="뉴런 파일 요약/검증")
    ap.add_argument("neuron_file")
    ap.add_argument("--model_name", default=None,
                    help="주면 모델 대비 뉴런/파라미터 비율까지 계산한다 (논문 기준 ≤1%%)")
    ap.add_argument("--require_nonempty", action="store_true",
                    help="비어 있으면 종료 코드 1")
    args = ap.parse_args(argv)

    info = summarize(args.neuron_file, args.model_name)
    c = info["counts"]
    print(f"파일: {info['path']}")
    for m in MODULE_ORDER:
        print(f"  {m:<9} {c[m]:>8,}")
    print(f"  {'ffn':<9} {c['ffn']:>8,}\n  {'attn':<9} {c['attn']:>8,}\n  {'total':<9} {c['total']:>8,}")
    if "neuron_pct" in info:
        print(f"\n모델: {info['model_name']}")
        print(f"  뉴런 비율     {info['neuron_pct']:.4f}%")
        print(f"  파라미터 비율 {info['param_pct']:.4f}%  "
              f"({info['detected_params']:,} / {info['total_params']:,})")
    if args.require_nonempty and c["total"] == 0:
        print("\n[FAIL] 뉴런이 0개다 — 검출이 전부 실패했다.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
