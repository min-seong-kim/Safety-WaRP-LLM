"""
WSR-Tune vs ActSVD mask-structure ablation: arm 정의 · 마스크 생성 · 예산(budget) 회계.

`wsr_actsvd_ablation_spec.md` §1–§2 구현. 순수 함수만 담고 있어 GPU/모델 없이 테스트된다.

────────────────────────────────────────────────────────────────────────────
Arm 정의 (spec §1 "The crucial mapping")

  W ∈ R^{m×n}  (m=out_features, n=in_features),  W̃ = V^T W U

  arm  basis                              mask unit   대응
  ───  ─────────────────────────────────  ─────────   ─────────────────────────
  A    U=I, V=I (원본 공간)                entry       SN-Tune / 논문 Table 5
  B    V=U_out (ActSVD 출력측), U=I         row         ActSVD rank freezing ★
  C    U=U_in (safety 입력측), V=I          column      입력측 subspace 제약
  D    U=U_in, V=I                          entry       WSR-Tune (ours)
  D_perm  U=U_in, V=signed permutation      entry       sanity: D와 정확히 같아야 함

  ⚠️ ActSVD의 U는 W X_in 의 **left** singular vector(출력 공간, m×m)이고
     WSR-Tune의 U는 H H^T 의 eigenbasis(입력 공간, n×n)다. 절대 섞지 말 것.
     행(row) 동결 = 출력 방향 u_i 고정 = Wei et al. footnote 9이 "어렵다"고 한 rank-level freezing.

────────────────────────────────────────────────────────────────────────────
예산 매칭 (spec §2 CRITICAL)

  동결되는 **스칼라 파라미터 수**를 arm 간 동일하게 맞춘다.
    entry  : round(ρ·m·n)
    row    : k_row = round(ρ·m)  →  k_row · n
    column : k_col = round(ρ·n)  →  k_col · m
  ±1% 이내여야 하며 `check_budget_match`가 이를 검증한다.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Arm 사양
# ────────────────────────────────────────────────────────────────────────────

ARM_SPECS: Dict[str, Dict] = {
    "A": {
        "basis_side": None,          # 재파라미터화 없음 (U=V=I)
        "mask_unit": "entry",
        "v_mode": "identity",
        "label": "original space + entry mask",
        "note": "논문 Table 5 'Original space + important mask FT' 재현 대상 (JB≈9.17 / GSM8K≈40.41)",
    },
    "B": {
        "basis_side": "output",      # V = ActSVD left singular vectors
        "mask_unit": "row",
        "v_mode": "basis",
        "label": "ActSVD output basis + row (rank) freezing",
        "note": "핵심 비교 arm. Wei et al. footnote 9의 rank-level freezing을 실제로 구현한 것.",
    },
    "C": {
        "basis_side": "input",       # U = safety activation covariance eigenbasis
        "mask_unit": "column",
        "v_mode": "identity",
        "label": "safety input basis + column freezing",
        "note": "입력측 subspace 제약 (AlphaEdit / Safe-LoRA 계열이 사는 위치)",
    },
    "D": {
        "basis_side": "input",
        "mask_unit": "entry",
        "v_mode": "identity",
        "label": "WSR-Tune (ours)",
        "note": "논문 Table 2 재현 대상 (JB≈6.90 / GSM8K≈38.99)",
    },
    "D_perm": {
        "basis_side": "input",
        "mask_unit": "entry",
        "v_mode": "signed_permutation",
        "label": "sanity arm: signed-permutation V",
        "note": "행 relabeling일 뿐이므로 결과가 D와 정확히 같아야 한다. 다르면 구현 버그.",
    },
}

# 참고: Wei et al. §2.2의 utility disentanglement `(I−Π^u)Π^s` (spec §4 optional arm)는
# 구현하지 않는다. 본 실험의 셋업은 WSR-Tune 본문과 동일하게 **safety 데이터만**
# (circuit_breakers) 사용해 safety-tuned 모델의 안전성 보존을 측정하는 것이며,
# utility corpus를 끌어들이면 arm 간 유일한 변수가 basis/mask 구조라는 전제가 깨진다.

MASK_UNITS = ("entry", "row", "column")


def arm_spec(arm: str) -> Dict:
    """arm 이름 → 사양 dict. 알 수 없는 arm이면 명확한 에러."""
    if arm not in ARM_SPECS:
        raise ValueError(
            f"Unknown ablation arm: {arm!r}. Choose from {sorted(ARM_SPECS)}"
        )
    return dict(ARM_SPECS[arm])


# ────────────────────────────────────────────────────────────────────────────
# 예산 계산
# ────────────────────────────────────────────────────────────────────────────

def budget_entries(rho: float, m: int, n: int) -> int:
    """entry 마스크가 동결하는 스칼라 수 = round(ρ·m·n). 모든 arm의 기준 예산."""
    _check_rho(rho)
    return int(round(rho * m * n))


def structured_k(rho: float, dim: int) -> int:
    """row/column 마스크가 동결할 방향 수 k = round(ρ·dim) (최소 1, 최대 dim)."""
    _check_rho(rho)
    return int(min(dim, max(1, round(rho * dim))))


def _check_rho(rho: float) -> None:
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"keep_ratio(ρ)는 (0, 1] 범위여야 합니다: {rho}")


def planned_frozen(rho: float, shape: Tuple[int, int], mask_unit: str) -> int:
    """mask_unit이 동결할 예정인 스칼라 파라미터 수."""
    m, n = shape
    if mask_unit == "entry":
        return budget_entries(rho, m, n)
    if mask_unit == "row":
        return structured_k(rho, m) * n
    if mask_unit == "column":
        return structured_k(rho, n) * m
    raise ValueError(f"Unknown mask_unit: {mask_unit!r}. Choose from {MASK_UNITS}")


# ────────────────────────────────────────────────────────────────────────────
# 마스크 생성 (True = 동결)
# ────────────────────────────────────────────────────────────────────────────

def top_k_entry_mask(scores: np.ndarray, rho: float) -> np.ndarray:
    """
    상위 ρ 비율의 **원소**를 동결하는 마스크.

    기존 Phase 2는 `np.quantile` 임계값을 쓰지만, 여기서는 argpartition으로
    정확히 k = round(ρ·size)개를 고른다. 예산 매칭 ±1% 검증을 하려면 동점(tie)
    때문에 개수가 흔들리면 안 되기 때문이다. gradient 크기는 연속값이라
    실전에서 두 방식은 사실상 동일하다.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores는 2차원이어야 합니다: {scores.shape}")
    k = budget_entries(rho, *scores.shape)
    mask = np.zeros(scores.shape, dtype=bool)
    if k <= 0:
        return mask
    if k >= scores.size:
        mask[:] = True
        return mask
    flat = scores.reshape(-1)
    idx = np.argpartition(flat, flat.size - k)[flat.size - k:]
    mask.reshape(-1)[idx] = True
    return mask


def aggregate_scores(scores: np.ndarray, mask_unit: str, agg: str = "l2") -> np.ndarray:
    """
    row/column 단위 집계 점수.

    mask_unit='row'    → axis=1로 집계 → shape (m,)
    mask_unit='column' → axis=0로 집계 → shape (n,)
    agg ∈ {l2, sum, mean}  (spec §3 arm B step 4: "row L2 of G̃, or Σ over row")
    """
    scores = np.asarray(scores, dtype=np.float64)
    if mask_unit == "row":
        axis = 1
    elif mask_unit == "column":
        axis = 0
    else:
        raise ValueError(f"aggregate_scores는 row/column만 지원합니다: {mask_unit!r}")

    if agg == "l2":
        return np.sqrt((scores ** 2).sum(axis=axis))
    if agg == "sum":
        return scores.sum(axis=axis)
    if agg == "mean":
        return scores.mean(axis=axis)
    raise ValueError(f"Unknown agg: {agg!r}. Choose from ('l2', 'sum', 'mean')")


def structured_mask(
    scores: np.ndarray,
    rho: float,
    mask_unit: str,
    agg: str = "l2",
) -> np.ndarray:
    """
    상위 k개 **행 전체** 또는 **열 전체**를 동결하는 마스크 (gradient 기반 랭킹).

    row 동결 = 출력 방향(rank) 고정, column 동결 = 입력 방향 고정.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores는 2차원이어야 합니다: {scores.shape}")
    m, n = scores.shape
    agg_scores = aggregate_scores(scores, mask_unit, agg=agg)
    dim = m if mask_unit == "row" else n
    k = structured_k(rho, dim)

    order = np.argsort(-agg_scores, kind="stable")
    selected = order[:k]

    mask = np.zeros((m, n), dtype=bool)
    if mask_unit == "row":
        mask[selected, :] = True
    else:
        mask[:, selected] = True
    return mask


def spectral_structured_mask(
    shape: Tuple[int, int],
    rho: float,
    mask_unit: str,
) -> np.ndarray:
    """
    ActSVD **자체 기준**(특이값 순서)으로 상위 k개 방향을 동결하는 마스크.

    Phase 1/ActSVD basis의 열은 이미 특이값 내림차순으로 정렬되어 있으므로,
    "가장 중요한 r개 rank"는 곧 좌표계의 앞쪽 k개 행(=출력 방향)이다.
    gradient 랭킹(`structured_mask`)과 비교해 ActSVD 원논문 기준을 그대로 옮긴 변형.
    """
    m, n = shape
    dim = m if mask_unit == "row" else n
    k = structured_k(rho, dim)
    mask = np.zeros((m, n), dtype=bool)
    if mask_unit == "row":
        mask[:k, :] = True
    elif mask_unit == "column":
        mask[:, :k] = True
    else:
        raise ValueError(f"spectral 랭킹은 row/column만 지원합니다: {mask_unit!r}")
    return mask


def build_mask(
    scores: Optional[np.ndarray],
    rho: float,
    mask_unit: str,
    agg: str = "l2",
    rank_by: str = "grad",
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    arm 사양에 맞는 마스크 하나를 만든다.

    rank_by='grad'     : safety gradient 크기로 랭킹 (모든 arm 기본값)
    rank_by='spectral' : 기저의 특이값 순서로 랭킹 (row/column 전용, ActSVD 원기준)
    """
    if rank_by == "spectral":
        if shape is None:
            if scores is None:
                raise ValueError("spectral 랭킹에는 shape 또는 scores가 필요합니다")
            shape = np.asarray(scores).shape
        return spectral_structured_mask(shape, rho, mask_unit)

    if rank_by != "grad":
        raise ValueError(f"Unknown rank_by: {rank_by!r}. Choose from ('grad', 'spectral')")
    if scores is None:
        raise ValueError("gradient 랭킹에는 importance scores가 필요합니다")

    if mask_unit == "entry":
        return top_k_entry_mask(scores, rho)
    return structured_mask(scores, rho, mask_unit, agg=agg)


# ────────────────────────────────────────────────────────────────────────────
# 예산 회계 / 검증
# ────────────────────────────────────────────────────────────────────────────

def mask_report(
    masks: Dict[Tuple[int, str], np.ndarray],
    rho: float,
    mask_unit: str,
) -> Dict:
    """
    per-layer 및 전체 동결 스칼라 수 리포트 (spec §2: "Log the actual frozen-parameter
    count per layer for every arm").
    """
    layers = []
    total_frozen = 0
    total_elems = 0
    total_reference = 0

    for key in sorted(masks.keys()):
        layer_idx, layer_type = key
        mask = np.asarray(masks[key])
        m, n = mask.shape
        frozen = int(mask.sum())
        reference = budget_entries(rho, m, n)  # entry-mask 기준 예산
        # 구조적 마스크는 동결 방향 수 k가 정수라, 예산 오차에 원리적 하한이 있다:
        # k = round(ρ·dim) 로 고정된 이상 |k·other − round(ρ·m·n)| 만큼은 반드시 어긋난다.
        # LLaMA 차원(4096/11008)에서는 ≤0.1%지만 작은 모델에서는 1%를 넘을 수 있으므로,
        # "구현 버그로 인한 불일치"와 "차원 때문에 불가피한 불일치"를 구별하기 위해 기록한다.
        floor = abs(planned_frozen(rho, (m, n), mask_unit) - reference) / max(reference, 1)

        layers.append({
            "layer_idx": int(layer_idx),
            "layer_type": layer_type,
            "shape": [int(m), int(n)],
            "mask_unit": mask_unit,
            "rounding_floor_rel_err": floor,
            "frozen": frozen,
            "numel": int(mask.size),
            "frozen_ratio": frozen / max(mask.size, 1),
            "reference_entry_budget": reference,
            "budget_rel_err": (frozen - reference) / max(reference, 1),
            "k_directions": (
                int(mask.any(axis=1).sum()) if mask_unit == "row"
                else int(mask.any(axis=0).sum()) if mask_unit == "column"
                else None
            ),
        })
        total_frozen += frozen
        total_elems += int(mask.size)
        total_reference += reference

    return {
        "keep_ratio": rho,
        "mask_unit": mask_unit,
        "num_modules": len(layers),
        "total_frozen": total_frozen,
        "total_numel": total_elems,
        "total_frozen_ratio": total_frozen / max(total_elems, 1),
        "total_reference_entry_budget": total_reference,
        "total_budget_rel_err": (total_frozen - total_reference) / max(total_reference, 1),
        "max_layer_budget_rel_err": max((abs(l["budget_rel_err"]) for l in layers), default=0.0),
        "rounding_floor_rel_err": max((l["rounding_floor_rel_err"] for l in layers), default=0.0),
        "layers": layers,
    }


def check_budget_match(report: Dict, tol: float = 0.01) -> Tuple[bool, str]:
    """
    한 arm의 리포트가 entry-mask 기준 예산과 ±tol 이내인지 검증 (spec §8 항목 4).

    row/column 마스크는 방향 수가 정수라 오차의 원리적 하한(`rounding_floor_rel_err`)이
    존재한다. 하한 자체가 tol을 넘는 설정(작은 모델/작은 ρ)에서는 하한까지 달성했으면
    "가능한 최선"으로 보고 통과시키되 메시지에 명시한다 — 구현 버그와 구별하기 위함이다.

    Returns: (ok, message)
    """
    worst = report.get("max_layer_budget_rel_err", 0.0)
    total = abs(report.get("total_budget_rel_err", 0.0))
    floor = report.get("rounding_floor_rel_err", 0.0)
    limit = max(tol, floor * 1.01)

    ok = worst <= limit and total <= limit
    verdict = "OK" if ok else "FAIL"
    if ok and floor > tol:
        verdict = f"OK (반올림 하한 {floor:.4%} 에 도달, tol보다 큼 — 차원이 작아 불가피)"
    msg = (
        f"budget check: total_rel_err={total:+.4%}, worst_layer_rel_err={worst:.4%}, "
        f"tol=±{tol:.1%}, rounding_floor={floor:.4%} → {verdict}"
    )
    return ok, msg


def compare_arm_budgets(reports: Dict[str, Dict], tol: float = 0.01) -> Tuple[bool, str]:
    """
    여러 arm의 동결 파라미터 총량이 서로 ±tol 이내인지 검증.
    arm 간 비교가 공정한지 판정하는 최종 게이트.
    """
    if not reports:
        return False, "no reports"
    totals = {arm: rep["total_frozen"] for arm, rep in reports.items()}
    ref = float(np.mean(list(totals.values())))
    lines = [f"  {arm:8s} frozen={cnt:,} ({(cnt - ref) / max(ref, 1):+.4%} vs mean)"
             for arm, cnt in sorted(totals.items())]
    worst = max(abs(cnt - ref) / max(ref, 1) for cnt in totals.values())
    ok = worst <= tol
    msg = "\n".join([f"cross-arm budget match (mean={ref:,.0f}, tol=±{tol:.1%}) → "
                     f"{'OK' if ok else 'FAIL'}"] + lines)
    return ok, msg
