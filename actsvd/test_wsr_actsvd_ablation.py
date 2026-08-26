"""
WSR-Tune vs ActSVD mask-structure ablation의 수학·구현 회귀 테스트 (CPU 전용).

`wsr_actsvd_ablation_spec.md` §8 correctness checklist 중 모델 학습 없이 검증 가능한 항목:
  3. signed-permutation sanity arm == arm D
  4. arm 간 동결 파라미터 수가 ±1% 이내
  5. U_out = left singular of W X_in (출력, m×m) / U_in = eigenbasis of X X^T (입력, n×n)
  + 재파라미터화 왕복, row 동결이 실제로 출력 방향을 보존하는지, mask granularity 동작

실행: pytest -q actsvd/test_wsr_actsvd_ablation.py
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from actsvd import wsr_ablation_masks as M
from models.warp_modules import LinearWaRP, make_signed_permutation, restore_weight
from actsvd.wsr_ablation_reparam import apply_arm_reparameterization

torch.manual_seed(0)
np.random.seed(0)

LLAMA_SHAPES = [
    ("attn_q", 4096, 4096),
    ("attn_k", 4096, 4096),
    ("attn_v", 4096, 4096),
    ("ffn_up", 11008, 4096),
    ("ffn_down", 4096, 11008),
]


# ────────────────────────────────────────────────────────────────────────────
# 예산 매칭 (spec §2 CRITICAL, §8 항목 4)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,m,n", LLAMA_SHAPES)
@pytest.mark.parametrize("rho", [0.01, 0.05, 0.10, 0.20])
def test_budget_matched_across_mask_units(name, m, n, rho):
    """entry / row / column 이 동결하는 스칼라 수가 ±1% 이내여야 한다."""
    entry = M.planned_frozen(rho, (m, n), "entry")
    row = M.planned_frozen(rho, (m, n), "row")
    col = M.planned_frozen(rho, (m, n), "column")

    for unit, cnt in (("row", row), ("column", col)):
        rel = abs(cnt - entry) / entry
        assert rel <= 0.01, f"{name} {unit}: {cnt} vs entry {entry} ({rel:.3%} > 1%)"


def test_llama_layer_budget_report_and_cross_arm_check():
    """실제 LLaMA-2-7B 형상으로 arm A/B/C/D 리포트를 만들고 교차 검증."""
    rho = 0.10
    rng = np.random.default_rng(0)
    # 형상 비율만 유지한 축소판. ρ·dim 이 충분히 커야 반올림 오차가 1% 안에 들어온다
    # (rel_err ≈ 0.5/(ρ·dim)) — 실제 LLaMA 차원(4096/11008)은 여유롭게 만족한다.
    small = {
        ("attn_q", 1024, 1024), ("ffn_up", 2752, 1024), ("ffn_down", 1024, 2752),
    }
    scores = {(i, name): rng.random((m, n)).astype(np.float32)
              for i, (name, m, n) in enumerate(small)}

    reports = {}
    for arm in ("A", "B", "C", "D"):
        spec = M.arm_spec(arm)
        masks = {k: M.build_mask(v, rho, spec["mask_unit"]) for k, v in scores.items()}
        rep = M.mask_report(masks, rho, spec["mask_unit"])
        ok, msg = M.check_budget_match(rep)
        assert ok, f"arm {arm}: {msg}"
        reports[arm] = rep

    ok, msg = M.compare_arm_budgets(reports)
    assert ok, msg


# ────────────────────────────────────────────────────────────────────────────
# 마스크 구조
# ────────────────────────────────────────────────────────────────────────────

def test_entry_mask_selects_exact_top_k():
    scores = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = M.top_k_entry_mask(scores, 0.10)
    assert mask.sum() == 10
    assert scores[mask].min() == 90.0     # 상위 10개만


def test_row_mask_freezes_whole_rows_by_aggregate():
    scores = np.zeros((4, 6), dtype=np.float32)
    scores[2, :] = 10.0          # 고르게 큰 행 → sum 60, L2 24.5
    scores[0, 0] = 50.0          # 한 원소만 큰 행 → sum 50, L2 50

    mask_sum = M.structured_mask(scores, rho=0.25, mask_unit="row", agg="sum")
    assert mask_sum.sum() == 6                            # 1개 행 × 6열
    assert mask_sum[2].all() and not mask_sum[0].any()    # sum 기준 → 2번 행

    mask_l2 = M.structured_mask(scores, rho=0.25, mask_unit="row", agg="l2")
    assert mask_l2[0].all() and not mask_l2[2].any()      # L2 기준 → 0번 행

    # 행 단위이므로 부분 동결은 존재할 수 없다
    for mask in (mask_sum, mask_l2):
        assert set(mask.sum(axis=1).tolist()) <= {0, 6}


def test_column_mask_freezes_whole_columns():
    scores = np.zeros((6, 4), dtype=np.float32)
    scores[:, 3] = 5.0
    mask = M.structured_mask(scores, rho=0.25, mask_unit="column")
    assert mask.sum() == 6
    assert mask[:, 3].all()
    assert set(mask.sum(axis=0).tolist()) <= {0, 6}


def test_spectral_mask_takes_leading_directions():
    """기저 열은 특이값 내림차순 → ActSVD 기준 top-k는 앞쪽 k개 행."""
    mask = M.spectral_structured_mask((10, 6), rho=0.30, mask_unit="row")
    assert mask[:3].all() and not mask[3:].any()


def test_unknown_arm_and_unit_raise():
    with pytest.raises(ValueError):
        M.arm_spec("Z")
    with pytest.raises(ValueError):
        M.planned_frozen(0.1, (4, 4), "diagonal")


# ────────────────────────────────────────────────────────────────────────────
# ActSVD 수학 (spec §8 항목 5)
# ────────────────────────────────────────────────────────────────────────────

def test_actsvd_left_singular_equals_output_gram_eigenbasis():
    """
    ActSVD:  U S V^T ≈ W X_in,  U ∈ R^{m×r} (출력 공간)
    구현:    Y = W X_in 의 Gram Y Y^T 를 SVD → 같은 U (부호 제외)
    """
    torch.manual_seed(1)
    m, n, N = 12, 9, 200
    W = torch.randn(m, n, dtype=torch.float64)
    X = torch.randn(n, N, dtype=torch.float64)
    Y = W @ X

    U_direct, S_direct, _ = torch.linalg.svd(Y, full_matrices=False)
    gram = Y @ Y.T
    U_gram, S_gram, _ = torch.linalg.svd(gram)

    # 특이값: σ(Y)^2 == λ(Y Y^T)
    r = min(m, n)
    assert torch.allclose(S_direct[:r] ** 2, S_gram[:r], rtol=1e-8, atol=1e-8)

    # 부분공간 일치 (부호 무관): |U_direct^T U_gram| 의 대각이 1
    overlap = (U_direct[:, :r].T @ U_gram[:, :r]).abs()
    assert torch.allclose(overlap.diagonal(), torch.ones(r, dtype=torch.float64), atol=1e-6)


def test_actsvd_projection_is_best_rank_r_approximation():
    """Ŵ = U_r U_r^T W 가 ‖W X − Ŵ X‖_F 를 최소화 (Eckart–Young)."""
    torch.manual_seed(2)
    m, n, N, r = 10, 8, 120, 3
    W = torch.randn(m, n, dtype=torch.float64)
    X = torch.randn(n, N, dtype=torch.float64)
    Y = W @ X

    U, S, _ = torch.linalg.svd(Y, full_matrices=False)
    Ur = U[:, :r]
    W_hat = Ur @ Ur.T @ W

    residual = (Y - W_hat @ X).norm()
    expected = S[r:].pow(2).sum().sqrt()
    assert torch.allclose(residual, expected, rtol=1e-8, atol=1e-8)
    assert torch.linalg.matrix_rank(W_hat) <= r


def test_input_and_output_basis_dimensions_differ():
    """입력측은 n×n, 출력측은 m×m — 절대 바꿔 쓸 수 없음을 형상으로 고정."""
    m, n, N = 7, 4, 50
    W = torch.randn(m, n, dtype=torch.float64)
    X = torch.randn(n, N, dtype=torch.float64)

    U_in, _, _ = torch.linalg.svd(X @ X.T)
    U_out, _, _ = torch.linalg.svd((W @ X) @ (W @ X).T)
    assert U_in.shape == (n, n)
    assert U_out.shape == (m, m)


# ────────────────────────────────────────────────────────────────────────────
# 재파라미터화 왕복 · 마스크 동작
# ────────────────────────────────────────────────────────────────────────────

def _fake_model(m_out, n_in, dtype=torch.float64, num_layers=1):
    """models.model.layers[i].self_attn.q_proj 구조를 흉내내는 최소 모델."""
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LinearWaRP(nn.Linear(n_in, m_out, bias=False, dtype=dtype))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(num_layers)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    return Model()


def _get_q(layer, layer_type):
    assert layer_type == "attn_q"
    return layer.self_attn.q_proj


def _orthonormal(dim, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype))
    return q


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "D_perm"])
def test_reparameterization_roundtrip_preserves_weight(arm):
    """모든 arm에서 W = V @ basis_coeff @ U^T 가 원본 W를 정확히 복원해야 한다."""
    m, n = 6, 5
    model = _fake_model(m, n)
    module = model.model.layers[0].self_attn.q_proj
    W0 = module.weight.data.clone()

    spec = M.arm_spec(arm)
    basis_data = {}
    if spec["basis_side"] == "input":
        basis_data[(0, "attn_q")] = {"U": _orthonormal(n, seed=3)}
    elif spec["basis_side"] == "output":
        basis_data[(0, "attn_q")] = {"U": _orthonormal(m, seed=4)}

    count, diag = apply_arm_reparameterization(
        model, spec=spec, basis_data=basis_data, layer_types=["attn_q"],
        target_layers=[0], get_target_module=_get_q, seed=7)
    assert count == 1
    assert max(diag.values()) < 1e-10, f"arm {arm} 복원 오차 {diag}"

    # forward 가 원본 선형변환과 동일한지
    x = torch.randn(3, n, dtype=torch.float64)
    assert torch.allclose(module(x), x @ W0.T, atol=1e-10)

    # restore_weight 도 같은 규약을 따르는지
    restore_weight(model)
    assert torch.allclose(module.weight.data, W0, atol=1e-10)


def test_output_side_reparam_has_expected_shapes():
    """arm B: UT_backward = U_out^T (m×m), UT_forward 비어 있음."""
    m, n = 6, 4
    model = _fake_model(m, n)
    module = model.model.layers[0].self_attn.q_proj
    U_out = _orthonormal(m, seed=11)

    apply_arm_reparameterization(
        model, spec=M.arm_spec("B"), basis_data={(0, "attn_q"): {"U": U_out}},
        layer_types=["attn_q"], target_layers=[0], get_target_module=_get_q)

    assert module.UT_forward.numel() == 0
    assert module.UT_backward.shape == (m, m)
    assert torch.allclose(module.basis_coeff.data, U_out.T @ module.weight.data, atol=1e-10)


def test_wrong_side_basis_raises():
    """입력측 기저를 출력측 arm에 넣으면 형상 검사에서 막혀야 한다."""
    m, n = 6, 4
    model = _fake_model(m, n)
    with pytest.raises(ValueError, match="차원 불일치"):
        apply_arm_reparameterization(
            model, spec=M.arm_spec("B"),
            basis_data={(0, "attn_q"): {"U": _orthonormal(n, seed=12)}},   # n×n 인데 출력측 요구
            layer_types=["attn_q"], target_layers=[0], get_target_module=_get_q)


def test_row_freezing_preserves_output_direction():
    """
    arm B의 핵심 주장: 행 i를 동결하면 학습 내내 u_i^T ΔW = 0.
    = Wei et al. footnote 9의 rank-level freezing이 좌표변환 후에는 행 마스킹으로 달성된다.
    """
    m, n = 6, 4
    model = _fake_model(m, n)
    module = model.model.layers[0].self_attn.q_proj
    U_out = _orthonormal(m, seed=13)
    W_before = module.weight.data.clone()

    apply_arm_reparameterization(
        model, spec=M.arm_spec("B"), basis_data={(0, "attn_q"): {"U": U_out}},
        layer_types=["attn_q"], target_layers=[0], get_target_module=_get_q)

    frozen_rows = torch.tensor([0, 3])
    mask = torch.zeros(m, n, dtype=torch.bool)
    mask[frozen_rows] = True
    module.coeff_mask.data = mask
    module.mask_mode.fill_(0)

    # gradient가 실제로 차단되는지
    x = torch.randn(8, n, dtype=torch.float64)
    module(x).pow(2).sum().backward()
    grad = module.basis_coeff.grad
    assert torch.allclose(grad[frozen_rows], torch.zeros_like(grad[frozen_rows]))
    assert grad[1].abs().sum() > 0

    # 임의의 optimizer step을 흉내낸 뒤 출력 방향 보존 확인
    with torch.no_grad():
        module.basis_coeff.data += torch.where(mask, torch.zeros_like(grad), torch.randn_like(grad))
    restore_weight(model)
    dW = module.weight.data - W_before
    preserved = U_out[:, frozen_rows].T @ dW      # u_i^T ΔW
    assert preserved.abs().max() < 1e-10


# ────────────────────────────────────────────────────────────────────────────
# signed-permutation sanity arm (spec §3 / §8 항목 3)
# ────────────────────────────────────────────────────────────────────────────

def test_signed_permutation_is_orthogonal():
    P, perm, signs = make_signed_permutation(8, seed=5, dtype=torch.float64)
    assert torch.allclose(P.T @ P, torch.eye(8, dtype=torch.float64), atol=1e-12)
    assert set(perm.tolist()) == set(range(8))
    assert set(signs.tolist()) <= {-1.0, 1.0}
    # P^T W = 행 치환 + 부호 반전
    W = torch.randn(8, 3, dtype=torch.float64)
    assert torch.allclose((P.T @ W).abs(), W[perm].abs(), atol=1e-12)


def test_perm_arm_mask_is_row_permutation_of_arm_D_mask():
    """|∂L/∂W̃| 는 절댓값이므로 D_perm의 importance는 D의 행 치환일 뿐이다."""
    m, n = 8, 6
    rho = 0.25
    G_D = torch.rand(m, n, dtype=torch.float64) + 0.1        # arm D의 importance
    P, perm, signs = make_signed_permutation(m, seed=17, dtype=torch.float64)

    # W̃_perm = P^T W̃_D  →  G̃_perm = |P^T G̃_D| (부호 소거)
    G_perm = (P.T @ (signs.to(torch.float64).unsqueeze(1) * G_D)).abs()

    mask_D = M.top_k_entry_mask(G_D.numpy(), rho)
    mask_perm = M.top_k_entry_mask(G_perm.numpy(), rho)

    # perm[j] = P^T의 j번째 행이 가져오는 원본 행 인덱스
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(m)
    assert np.array_equal(mask_perm, mask_D[perm.numpy()][inv.numpy()][perm.numpy()]) or \
           np.array_equal(mask_perm[inv.numpy()], mask_D), \
           "D_perm 마스크가 D 마스크의 행 치환이 아닙니다"
    assert mask_perm.sum() == mask_D.sum()


def test_perm_arm_training_step_equals_arm_D():
    """
    같은 데이터/손실로 한 스텝 학습했을 때 arm D와 D_perm의 최종 W가 동일해야 한다.
    (spec §3 "Result MUST equal arm D exactly. If it differs → implementation bug.")
    """
    torch.manual_seed(21)
    m, n = 6, 5
    lr = 0.1
    U_in = _orthonormal(n, seed=23)
    x = torch.randn(16, n, dtype=torch.float64)
    target = torch.randn(16, m, dtype=torch.float64)

    base = nn.Linear(n, m, bias=False, dtype=torch.float64)
    W0 = base.weight.data.clone()

    def run(arm):
        model = _fake_model(m, n)
        module = model.model.layers[0].self_attn.q_proj
        module.weight.data = W0.clone()
        apply_arm_reparameterization(
            model, spec=M.arm_spec(arm), basis_data={(0, "attn_q"): {"U": U_in}},
            layer_types=["attn_q"], target_layers=[0], get_target_module=_get_q, seed=99)

        # importance = |grad| 로 entry mask 생성 (양쪽 모두 동일 절차)
        loss = (module(x) - target).pow(2).mean()
        loss.backward()
        G = module.basis_coeff.grad.detach().abs().numpy()
        mask = torch.from_numpy(M.top_k_entry_mask(G, 0.30))
        module.basis_coeff.grad = None

        module.coeff_mask.data = mask
        module.mask_mode.fill_(0)

        # 한 스텝 SGD
        loss = (module(x) - target).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            module.basis_coeff -= lr * module.basis_coeff.grad
        restore_weight(model)
        return module.weight.data.clone(), int(mask.sum())

    W_D, frozen_D = run("D")
    W_perm, frozen_perm = run("D_perm")

    assert frozen_D == frozen_perm
    assert torch.allclose(W_D, W_perm, atol=1e-12), \
        f"D vs D_perm 최대 편차 {(W_D - W_perm).abs().max():.3e}"
    assert not torch.allclose(W_D, W0)     # 실제로 학습이 일어났는지
