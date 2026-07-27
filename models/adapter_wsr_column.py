"""Adapter-aware, column-structured WSR projection for LoRA.

명세: `adapter_aware_column_wsr_lora_implementation.md`

핵심 아이디어
-------------
safety LoRA update 를 **safety-conditioned WSR activation basis** `U_l` 로 회전하고,
회전된 adapter coefficient 의 safety importance 를 측정한 뒤, 중요한 **column direction**
을 downstream LoRA 가 쓰지 못하게 막는다.

    ΔW̃_s,l = s_s · B_s,l A_s,l U_l                  (정확한 좌표변환, 근사 아님)
    G_l     = Σ_x |∂L_safe(x) / ∂(W_safe,l U_l)|    (배치별 abs 후 누적)
    T_l     = |ΔW̃_s,l| ⊙ G_l
    c_l,j   = ‖T_l[:,j]‖₂                            (또는 L1)
    U_S,l   = U_l[:, TopK(c_l)]
    ΔW_d^⊥  = s_d B_d A_d (I − U_S U_Sᵀ)

원소별 WSR mask 는 LoRA 의 rank 를 깨뜨리므로(Hadamard 곱은 rank 비보존),
importance 를 **column 단위로 집계**하고 **오른쪽 projection** 으로 적용한다.
오른쪽 곱셈은 rank(XP) ≤ rank(X) 이므로 rank-r 이 보존되고 진짜 PEFT adapter 로 merge 된다.

`adapter_subspace_lora` 와의 차이
--------------------------------
`adapter_subspace_lora` 는 safety adapter **자체의 compact SVD** right singular subspace
`Q_S` 를 쓴다. 이 모듈은 **activation covariance basis** `U_l` 위에서 gradient importance 를
재어 방향을 고른다. 수식의 형태(오른쪽 projection)만 같고 선택 기준이 완전히 다르다.

⚠️ `U_l` 은 반드시 safety adapter 가 반영된 `W_safe = W_base + ΔW_s` 에서 수집한
activation 으로 만들어야 한다. base model 만으로 만든 basis 는 이 방법의 전제를 깬다.
"""

import json
import os

import torch

from models.adapter_subspace import PROJ_TO_LT, module_name_to_key  # noqa: F401  (재사용)

EPS = 1e-12

VALID_IMPORTANCE_MODES = ("adapter_taylor", "gradient_only", "adapter_magnitude_only")
VALID_AGGREGATIONS = ("l2", "l1")


# ═══════════════════════════════════════════════════════════════════════
# 1. 회전 + chunked column score
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rotated_adapter_chunk(B_s, A_s, scaling, U_chunk):
    """ΔW̃_s[:, J] = s_s · B_s (A_s U[:, J]) 를 계산한다.

    괄호 순서가 중요하다. `(B_s A_s) U_J` 로 하면 dense (m×n) 을 만들게 되므로
    반드시 `A_s U_J` (r×cj) 를 먼저 곱한다 — 비용 O(r·n·cj), 메모리 O(m·cj).

    Args:
        B_s: (m, r), A_s: (r, n), U_chunk: (n, cj)
    Returns:
        (m, cj)
    """
    return scaling * (B_s @ (A_s @ U_chunk))


def _aggregate_columns(mat, aggregation):
    """(m, cj) → (cj,). l2 = 열별 유클리드 노름, l1 = 열별 절대값 합."""
    if aggregation == "l2":
        return torch.linalg.vector_norm(mat, ord=2, dim=0)
    if aggregation == "l1":
        return mat.abs().sum(dim=0)
    raise ValueError(f"unknown aggregation: {aggregation} (valid: {VALID_AGGREGATIONS})")


@torch.no_grad()
def compute_column_scores(B_s, A_s, scaling, U, G, modes=VALID_IMPORTANCE_MODES,
                          aggregation="l2", chunk_size=128, compute_dtype=torch.float32):
    """세 가지 importance mode 의 column score 를 **한 번의 순회**로 계산한다.

    dense ΔW̃_s (down_proj 면 4096×11008) 를 전체 materialize 하지 않는다.
    basis column chunk 단위로 만들고 즉시 열 축약한 뒤 버린다.

    Args:
        B_s: (m, r) safety lora_B
        A_s: (r, n) safety lora_A
        scaling: s_s = α_s / r_s
        U: (n, n) WSR activation basis (orthonormal)
        G: (m, n) 누적 gradient importance = Σ_b |G_W^(b) U|.  None 이면
           gradient 를 쓰는 mode 는 계산하지 않는다.
        modes: 계산할 mode 들
        aggregation: "l2" | "l1"
        chunk_size: basis column chunk 크기
        compute_dtype: 누적 dtype (fp32 권장)

    Returns:
        {mode: (n,) float32 CPU tensor}
    """
    for m_ in modes:
        if m_ not in VALID_IMPORTANCE_MODES:
            raise ValueError(f"unknown importance mode: {m_} (valid: {VALID_IMPORTANCE_MODES})")
    needs_grad = any(m_ in ("gradient_only", "adapter_taylor") for m_ in modes)
    needs_adapter = any(m_ in ("adapter_magnitude_only", "adapter_taylor") for m_ in modes)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"U shape must be square (n,n), got {tuple(U.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if needs_adapter and (B_s.ndim != 2 or A_s.ndim != 2 or
                          B_s.shape[1] != A_s.shape[0] or A_s.shape[1] != U.shape[0]):
        raise ValueError(f"adapter shape mismatch: B={tuple(B_s.shape)} A={tuple(A_s.shape)} U={tuple(U.shape)}")
    expected_m = B_s.shape[0] if needs_adapter else (G.shape[0] if G is not None else None)
    if G is not None and (G.ndim != 2 or G.shape != (expected_m, U.shape[1])):
        raise ValueError(f"gradient shape mismatch: G={tuple(G.shape)} expected={(expected_m, U.shape[1])}")
    if needs_grad and G is None:
        raise ValueError(f"modes={modes} 는 gradient importance G 가 필요합니다 (G=None).")

    device = U.device
    n = U.shape[1]
    B_s = B_s.to(device=device, dtype=compute_dtype) if needs_adapter else None
    A_s = A_s.to(device=device, dtype=compute_dtype) if needs_adapter else None
    U = U.to(compute_dtype)

    out = {m_: torch.zeros(n, dtype=torch.float32) for m_ in modes}

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        U_j = U[:, start:stop]                                   # (n, cj)

        dW_j = None
        if needs_adapter:
            dW_j = rotated_adapter_chunk(B_s, A_s, scaling, U_j)  # (m, cj)

        G_j = None
        if needs_grad:
            # G 는 CPU 에 있을 수 있다(모듈 하나가 fp32 로 최대 180MB).
            # chunk 만 device 로 올려 peak 메모리를 억제한다.
            G_j = G[:, start:stop].to(device=device, dtype=compute_dtype)

        for m_ in modes:
            if m_ == "gradient_only":
                col = _aggregate_columns(G_j, aggregation)
            elif m_ == "adapter_magnitude_only":
                col = _aggregate_columns(dW_j, aggregation)
            else:  # adapter_taylor
                col = _aggregate_columns(dW_j.abs() * G_j, aggregation)
            out[m_][start:stop] = col.float().cpu()

        del U_j, dW_j, G_j

    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. Direction 선택 규칙 (명세 §8)
# ═══════════════════════════════════════════════════════════════════════

def select_directions(scores, top_k=None, keep_ratio=None, score_energy=None):
    """column score 에서 보호할 direction index 를 고른다.

    우선순위 (명세 §8 말미): top_k → keep_ratio → score_energy.
    셋 다 None 이면 오류 (명시적 default 를 두지 않는다 — 조용한 기본값이 실험을 오염시킨다).

    Returns: (indices LongTensor(k,), k, selection_mode, selection_value)
    """
    n = int(scores.numel())
    given = [x is not None for x in (top_k, keep_ratio, score_energy)]
    if sum(given) == 0:
        raise ValueError("direction 선택 규칙이 없습니다. "
                         "--direction_top_k / --direction_keep_ratio / --direction_score_energy "
                         "중 하나를 지정하세요.")

    order = torch.argsort(scores, descending=True)

    if top_k is not None:
        k = max(0, min(int(top_k), n))
        return order[:k].clone(), k, "top_k", float(top_k)

    if keep_ratio is not None:
        import math
        if not 0.0 < float(keep_ratio) <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        k = int(math.ceil(float(keep_ratio) * n))
        k = max(0, min(k, n))
        return order[:k].clone(), k, "keep_ratio", float(keep_ratio)

    if not 0.0 < float(score_energy) <= 1.0:
        raise ValueError(f"score_energy must be in (0, 1], got {score_energy}")
    s = scores[order].to(torch.float64) ** 2
    total = float(s.sum())
    if total <= 0:
        return order[:0].clone(), 0, "score_energy", float(score_energy)
    cum = torch.cumsum(s, dim=0) / total
    k = int(torch.searchsorted(cum, torch.tensor(float(score_energy), dtype=cum.dtype)).item()) + 1
    k = max(1, min(k, n))
    return order[:k].clone(), k, "score_energy", float(score_energy)


def select_random_directions(n, k, generator):
    """§17.6 random-column control. 같은 U 에서 동일한 k 개를 무작위로 고른다.

    개선이 safety-aware 선택 덕인지 단순 capacity 감소 덕인지 분리하기 위한 통제군.
    """
    return torch.randperm(n, generator=generator)[:k].clone()


# ═══════════════════════════════════════════════════════════════════════
# 3. 진단 지표 (명세 §15.6)
# ═══════════════════════════════════════════════════════════════════════

def spearman(a, b):
    """Spearman rank correlation. tie 는 평균 순위로 처리한다."""
    def _rank(x):
        x = x.to(torch.float64)
        order = torch.argsort(x)
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(x.numel(), dtype=torch.float64)
        # tie 평균 처리
        vals, inv, counts = torch.unique(x, return_inverse=True, return_counts=True)
        if int(counts.max()) > 1:
            sums = torch.zeros(vals.numel(), dtype=torch.float64).scatter_add_(0, inv, ranks)
            ranks = (sums / counts)[inv]
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(torch.linalg.vector_norm(ra) * torch.linalg.vector_norm(rb))
    if denom < EPS:
        return 0.0
    return float((ra @ rb) / denom)


def score_summary(scores, selected_idx):
    """score 분포 진단 (명세 §15.6)."""
    n = int(scores.numel())
    k = int(selected_idx.numel())
    total = float(scores.sum())
    sel = scores[selected_idx]
    sel_sum = float(sel.sum())
    mask = torch.ones(n, dtype=torch.bool)
    mask[selected_idx] = False
    unsel = scores[mask]
    return {
        "n": n, "k": k,
        "score_min": float(scores.min()), "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "topk_score_mass": (sel_sum / total) if total > EPS else 0.0,
        "selected_mean": float(sel.mean()) if k > 0 else 0.0,
        "unselected_mean": float(unsel.mean()) if unsel.numel() > 0 else 0.0,
        "selected_over_unselected": (float(sel.mean()) / float(unsel.mean()))
        if (k > 0 and unsel.numel() > 0 and float(unsel.mean()) > EPS) else None,
    }


@torch.no_grad()
def subspace_orthogonality_error(U_S):
    """‖U_Sᵀ U_S − I‖_F (fp64). U_S 는 orthonormal basis 의 부분집합이므로 ~1e-6 이하여야 한다."""
    if U_S.numel() == 0:
        return 0.0
    X = U_S.to(torch.float64)
    G = X.transpose(0, 1) @ X
    return float(torch.linalg.matrix_norm(G - torch.eye(G.shape[0], dtype=G.dtype, device=G.device)))


# ═══════════════════════════════════════════════════════════════════════
# 4. Artifact I/O
# ═══════════════════════════════════════════════════════════════════════

def subspace_path(root, layer_type, layer_idx):
    return os.path.join(root, layer_type, f"layer_{layer_idx:02d}_subspace.pt")


def save_subspace(root, key, payload):
    layer_idx, layer_type = key
    os.makedirs(os.path.join(root, layer_type), exist_ok=True)
    torch.save(payload, subspace_path(root, layer_type, layer_idx))


def load_subspaces(root, layer_types=None):
    """{(layer_idx, layer_type): payload}. payload['U_S'] 가 보호 subspace."""
    out = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"adapter-wsr subspace dir not found: {root}")
    types = layer_types or [d for d in sorted(os.listdir(root))
                            if os.path.isdir(os.path.join(root, d))]
    for lt in types:
        d = os.path.join(root, lt)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith("layer_") and fname.endswith("_subspace.pt"):
                li = int(fname.split("_")[1])
                out[(li, lt)] = torch.load(os.path.join(d, fname), map_location="cpu",
                                           weights_only=False)
    return out


def scores_path(root, layer_type, layer_idx):
    return os.path.join(root, layer_type, f"layer_{layer_idx:02d}_scores.pt")


def save_scores(root, key, payload):
    layer_idx, layer_type = key
    os.makedirs(os.path.join(root, layer_type), exist_ok=True)
    torch.save(payload, scores_path(root, layer_type, layer_idx))


def load_scores(root, layer_types=None):
    """{(layer_idx, layer_type): {mode: (n,) tensor, ...}}"""
    out = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"column score dir not found: {root}")
    types = layer_types or [d for d in sorted(os.listdir(root))
                            if os.path.isdir(os.path.join(root, d))]
    for lt in types:
        d = os.path.join(root, lt)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith("layer_") and fname.endswith("_scores.pt"):
                li = int(fname.split("_")[1])
                out[(li, lt)] = torch.load(os.path.join(d, fname), map_location="cpu",
                                           weights_only=False)
    return out


def read_report(root):
    p = os.path.join(root, "report.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)
