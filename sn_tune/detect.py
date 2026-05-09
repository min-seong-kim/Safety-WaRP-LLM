"""
detect.py  —  C 공간에서 gradient 기반 safety coordinate 검출

흐름
────
1. Safety 데이터셋으로 forward+backward 수행
2. C.grad (= ∂L_safety/∂C) 를 배치마다 abs 누적
3. keep_ratio 에 따라 top-k 좌표 선택 → bool mask dict 반환

gradient 의미
─────────────
forward:  W_rec = C @ U.T  →  output = F.linear(x, W_rec)
∂L/∂C  = ∂L/∂W_rec @ U       (autograd 자동 계산)
         shape: [out_features, in_features]  (C 와 동일)

row i 의 기여도  = ||∂L/∂C[i, :]||   → out-neuron i 의 safety 민감도
element [i,j]   = |∂L/∂C[i,j]|      → safety basis 방향 j 에 대한 neuron i 의 민감도
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .module import LinearSNWaRP, LAYER_TYPE_MAP

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gradient 누적
# ─────────────────────────────────────────────────────────────────────────────

def accumulate_grad_scores(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[tuple[int, str], torch.Tensor]:
    """
    Safety 데이터에서 |∂L/∂C| 를 누적한 score tensor 반환.

    Returns
    -------
    scores : {(layer_idx, ltype): FloatTensor [out, in]}
        누적 절댓값 gradient.  각 원소가 클수록 safety 민감도 높음.
    """
    # ── LinearSNWaRP 모듈 목록 수집 ──────────────────────────────────────
    keys: list[tuple[int, str]] = []
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        for ltype in LAYER_TYPE_MAP:
            sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
            sub  = getattr(model.model.layers[layer_idx], sub_attr, None)
            if sub is None:
                continue
            proj = getattr(sub, proj_attr, None)
            if isinstance(proj, LinearSNWaRP):
                keys.append((layer_idx, ltype))

    if not keys:
        raise RuntimeError("No LinearSNWaRP modules found. Run convert_to_sn_warp() first.")

    log.info(f"[detect] Found {len(keys)} LinearSNWaRP modules")

    # ── score 버퍼 초기화 (float32, CPU) ──────────────────────────────────
    scores: dict[tuple[int, str], torch.Tensor] = {}
    for (layer_idx, ltype) in keys:
        sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
        sub  = getattr(model.model.layers[layer_idx], sub_attr)
        proj = getattr(sub, proj_attr)
        scores[(layer_idx, ltype)] = torch.zeros_like(proj.coeff.data, dtype=torch.float32, device="cpu")

    # ── coeff.requires_grad = True 확인 ────────────────────────────────────
    model.eval()
    for (layer_idx, ltype) in keys:
        sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
        sub  = getattr(model.model.layers[layer_idx], sub_attr)
        proj = getattr(sub, proj_attr)
        proj.coeff.requires_grad_(True)

    # ── 배치 반복 ──────────────────────────────────────────────────────────
    count = 0
    for batch in tqdm(dataloader, desc="[detect] accumulating |∂L/∂C|"):
        if max_batches is not None and count >= max_batches:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # gradient 초기화
        for (layer_idx, ltype) in keys:
            sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
            sub  = getattr(model.model.layers[layer_idx], sub_attr)
            proj = getattr(sub, proj_attr)
            if proj.coeff.grad is not None:
                proj.coeff.grad.zero_()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        outputs.loss.backward()

        # 누적
        with torch.no_grad():
            for (layer_idx, ltype) in keys:
                sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
                sub  = getattr(model.model.layers[layer_idx], sub_attr)
                proj = getattr(sub, proj_attr)
                if proj.coeff.grad is not None:
                    scores[(layer_idx, ltype)] += proj.coeff.grad.abs().float().cpu()

        count += 1

    log.info(f"[detect] Accumulated gradients over {count} batches")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top-k 선택 → bool mask
# ─────────────────────────────────────────────────────────────────────────────

def select_top_coords(
    scores: dict[tuple[int, str], torch.Tensor],
    keep_ratio: float = 0.10,
    granularity: Literal["element", "row"] = "element",
    per_layer: bool = False,
) -> dict[tuple[int, str], torch.Tensor]:
    """
    score 에서 top-k 좌표를 선택해 bool mask 반환.

    Parameters
    ----------
    scores       : {(layer_idx, ltype): FloatTensor [out, in]}
    keep_ratio   : 선택할 비율 (예: 0.10 → 상위 10%)
    granularity  : "element"  → 개별 원소 선택
                   "row"      → 행(out-neuron) 단위 선택 (행 전체 포함/제외)
    per_layer    : True 이면 레이어별로 keep_ratio 적용
                   False 이면 전체 통합 threshold

    Returns
    -------
    masks : {(layer_idx, ltype): BoolTensor [out, in]}
        True 인 위치가 업데이트 대상.
    """
    masks: dict[tuple[int, str], torch.Tensor] = {}

    if granularity == "row":
        # 행별 score = row L1 norm
        row_scores: dict[tuple[int, str], torch.Tensor] = {
            k: v.sum(dim=1) for k, v in scores.items()   # [out]
        }

        if per_layer:
            for key, rscore in row_scores.items():
                n_total = rscore.numel()
                n_keep  = max(1, int(n_total * keep_ratio))
                thresh  = rscore.kthvalue(n_total - n_keep + 1).values.item()
                row_mask = rscore >= thresh                  # [out]
                masks[key] = row_mask.unsqueeze(1).expand_as(scores[key]).clone()
        else:
            all_row = torch.cat([v for v in row_scores.values()])
            n_total = all_row.numel()
            n_keep  = max(1, int(n_total * keep_ratio))
            thresh  = all_row.kthvalue(n_total - n_keep + 1).values.item()
            for key, rscore in row_scores.items():
                row_mask = rscore >= thresh
                masks[key] = row_mask.unsqueeze(1).expand_as(scores[key]).clone()

    else:  # granularity == "element"
        if per_layer:
            for key, score in scores.items():
                flat    = score.flatten()
                n_total = flat.numel()
                n_keep  = max(1, int(n_total * keep_ratio))
                thresh  = flat.kthvalue(n_total - n_keep + 1).values.item()
                masks[key] = score >= thresh
        else:
            all_vals = torch.cat([s.flatten() for s in scores.values()])
            n_total  = all_vals.numel()
            n_keep   = max(1, int(n_total * keep_ratio))
            thresh   = all_vals.kthvalue(n_total - n_keep + 1).values.item()
            for key, score in scores.items():
                masks[key] = score >= thresh

    # 통계 출력
    total_params  = sum(m.numel() for m in masks.values())
    selected      = sum(m.sum().item() for m in masks.values())
    log.info(
        f"[detect] Selected {selected:,} / {total_params:,} coords "
        f"({selected / total_params * 100:.2f}%) "
        f"[granularity={granularity}, per_layer={per_layer}]"
    )
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backward hook 등록 (tuning 단계에서 사용)
# ─────────────────────────────────────────────────────────────────────────────

def apply_coeff_gradient_masks(
    model,
    masks: dict[tuple[int, str], torch.Tensor],
) -> list:
    """
    masks 에 따라 C.grad 를 zero-masking 하는 backward hook 등록.
    선택된 위치(True)만 gradient 통과 → 나머지 0.

    Returns
    -------
    hooks : 등록된 RemovableHook 리스트  (학습 후 remove 할 것)
    """
    hooks = []

    for (layer_idx, ltype), mask in masks.items():
        sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
        sub  = getattr(model.model.layers[layer_idx], sub_attr, None)
        if sub is None:
            continue
        proj = getattr(sub, proj_attr, None)
        if not isinstance(proj, LinearSNWaRP):
            continue

        # mask 를 coeff 와 같은 device/dtype 으로 미리 이동
        mask_device = proj.coeff.device
        mask_gpu    = mask.to(device=mask_device, dtype=proj.coeff.dtype)

        def _make_hook(m: torch.Tensor):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                return grad * m
            return hook

        h = proj.coeff.register_hook(_make_hook(mask_gpu))
        hooks.append(h)

    log.info(f"[detect] Registered {len(hooks)} coeff gradient hooks")
    return hooks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forward-score 기반 교집합 detection  (detection_v2 방식)
# ─────────────────────────────────────────────────────────────────────────────

# patched modeling_llama.py 에서 저장하는 score attribute 이름 매핑
_SCORE_ATTR: dict[str, tuple[str, str]] = {
    "ffn_up":   ("mlp",       "_last_ffn_up_score"),
    "ffn_down": ("mlp",       "_last_ffn_down_score"),
    "attn_q":   ("self_attn", "_last_q_score"),
    "attn_k":   ("self_attn", "_last_k_score"),
    "attn_v":   ("self_attn", "_last_v_score"),
}


def detect_with_forward_scores(
    model,
    prompts: list[str],
    tokenizer,
    layer_types: list[str],
    top_k_ffn:  int  = 1200,
    top_k_attn: int  = 200,
    is_chat_model: bool = True,
    max_seq_len:   int  = 1024,
    device: torch.device | None = None,
) -> dict[tuple[int, str], torch.Tensor]:
    """
    patched modeling_llama.py 의 forward activation score 를 이용해
    safety neuron 을 교집합 방식으로 탐지한다 (detection_v2.py 와 동일한 Eq.3).

    WaRP 모듈(LinearSNWaRP)이 이미 적용된 모델 위에서 동작한다.
    LinearSNWaRP.forward() 가 원본 출력을 보존하므로 score 는 완전히 유효.

    Parameters
    ----------
    model        : WaRP 변환된 LlamaForCausalLM (patched transformers 필요)
    prompts      : harmful prompt 문자열 리스트
    tokenizer    : 모델 토크나이저
    layer_types  : 탐지할 ltype 리스트  (예: ["ffn_up", "attn_q", "attn_v"])
    top_k_ffn    : FFN 레이어당 per-prompt top-k
    top_k_attn   : attention 레이어당 per-prompt top-k
    is_chat_model: True 이면 apply_chat_template 사용
    max_seq_len  : 최대 토큰 길이
    device       : 기본값 = model 의 첫 번째 파라미터 device

    Returns
    -------
    masks : {(layer_idx, ltype): BoolTensor[out_features, in_features]}
        교집합으로 선택된 row 에 해당하는 모든 C 좌표가 True.
        선택된 row i → masks[key][i, :] = True (행 전체 tune).

    Notes
    -----
    - patched transformers 가 필요하므로 hb_sntune 환경에서 실행할 것.
    - 점수 attribute 가 없으면 RuntimeError 발생 → patched 모델 확인.
    - q/k score 는 patched 코드에서 동일한 qk_score 를 공유한다.
      GQA 모델(k/v head 수가 q head 수와 다른 경우)에서는 k_proj output dim 과
      score size 가 불일치할 수 있음 — Llama-2 (MHA) 에서는 문제없음.
    """
    if device is None:
        device = next(model.parameters()).device

    num_layers = len(model.model.layers)

    # ── 대상 ltype 만 추려 top_k 결정 ──────────────────────────────────────
    ffn_ltypes  = {"ffn_up", "ffn_down"}
    attn_ltypes = {"attn_q", "attn_k", "attn_v"}
    target_ltypes = [lt for lt in layer_types if lt in _SCORE_ATTR]

    def _top_k(ltype: str) -> int:
        return top_k_ffn if ltype in ffn_ltypes else top_k_attn

    # ── (layer_idx, ltype) 마다 per-prompt Set[int] 을 수집 ───────────────
    # prompt_sets[(layer_idx, ltype)] = List[Set[int]]
    prompt_sets: dict[tuple[int, str], list[set[int]]] = {
        (li, lt): []
        for li in range(num_layers)
        for lt in target_ltypes
    }

    model.eval()
    failed = 0

    for p_idx, prompt in enumerate(tqdm(prompts, desc="[detect_fwd] forward scoring")):
        # ── 토크나이즈 ────────────────────────────────────────────────────
        try:
            if is_chat_model:
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                inputs = {"input_ids": input_ids}
            else:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as e:
            log.warning(f"[detect_fwd] tokenize failed (prompt {p_idx}): {e}")
            failed += 1
            continue

        # ── forward (no grad) ─────────────────────────────────────────────
        try:
            with torch.no_grad():
                model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=False,
                    return_dict=True,
                )
        except Exception as e:
            log.warning(f"[detect_fwd] forward failed (prompt {p_idx}): {e}")
            failed += 1
            continue

        # ── 레이어별 score 읽기 → per-prompt top-k indices ───────────────
        ok = True
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            for ltype in target_ltypes:
                sub_attr, score_attr = _SCORE_ATTR[ltype]
                sub = getattr(layer, sub_attr, None)
                if sub is None:
                    continue
                score = getattr(sub, score_attr, None)
                if score is None:
                    log.error(
                        f"[detect_fwd] Missing '{score_attr}' at layer {layer_idx}. "
                        "Patched modeling_llama.py 가 로드되지 않았을 수 있음."
                    )
                    ok = False
                    break

                # score: [out_features] (배치 크기 1 기준 squeeze 됨)
                score_1d = score.detach().float().view(-1).cpu()
                k = min(_top_k(ltype), score_1d.numel())
                if k > 0:
                    top_indices = score_1d.topk(k).indices.tolist()
                    prompt_sets[(layer_idx, ltype)].append(set(top_indices))
                else:
                    prompt_sets[(layer_idx, ltype)].append(set())
            if not ok:
                failed += 1
                break

    log.info(
        f"[detect_fwd] Processed {len(prompts) - failed}/{len(prompts)} prompts "
        f"(failed={failed})"
    )

    # ── 교집합 계산 (Eq. 3:  N_safe = ⋂_x N_x) ──────────────────────────
    masks: dict[tuple[int, str], torch.Tensor] = {}

    for (layer_idx, ltype), sets_list in prompt_sets.items():
        # LinearSNWaRP 존재 여부 확인 → out/in shape 얻기
        sub_attr_w, proj_attr_w = LAYER_TYPE_MAP[ltype]
        sub_w  = getattr(model.model.layers[layer_idx], sub_attr_w, None)
        if sub_w is None:
            continue
        proj_w = getattr(sub_w, proj_attr_w, None)
        if proj_w is None:
            continue
        out_features, in_features = proj_w.coeff.shape   # [out, in]

        # 교집합
        if not sets_list:
            common: set[int] = set()
        else:
            common = set(sets_list[0])
            for s in sets_list[1:]:
                common &= s

        # row mask 생성: 선택된 뉴런의 행 전체를 True
        row_mask = torch.zeros(out_features, dtype=torch.bool)
        for idx in common:
            if 0 <= idx < out_features:
                row_mask[idx] = True

        # [out, in] 으로 확장
        masks[(layer_idx, ltype)] = row_mask.unsqueeze(1).expand(out_features, in_features).clone()

    # ── 통계 로그 ─────────────────────────────────────────────────────────
    total_params = sum(m.numel() for m in masks.values())
    selected     = sum(m.sum().item() for m in masks.values())
    pct = selected / total_params * 100 if total_params > 0 else 0.0

    # 레이어별 선택 뉴런 수 집계
    per_ltype: dict[str, int] = {}
    for (li, lt), m in masks.items():
        n = int(m[:, 0].sum().item())   # 행 단위 → 첫 열 합산
        per_ltype[lt] = per_ltype.get(lt, 0) + n

    log.info(
        f"[detect_fwd] Intersection result: "
        f"{selected:,}/{total_params:,} C-coords ({pct:.2f}%) selected"
    )
    log.info(f"[detect_fwd] Per-ltype neuron counts: {per_ltype}")

    return masks
