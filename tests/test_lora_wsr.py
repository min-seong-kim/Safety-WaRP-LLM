"""핵심 3 test: WSR-LoRA element-wise 수학 검증 (CPU, GPU 불필요).

실행: pytest -q tests/test_lora_wsr.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lora_wsr_elementwise import LinearWSRLoRA  # noqa: E402


torch.manual_seed(0)


def _random_orthonormal(n, dtype=torch.float64):
    a = torch.randn(n, n, dtype=dtype)
    q, _ = torch.linalg.qr(a)
    return q


def _make_module(m=12, n=16, r=4, alpha=8, mask_ratio=0.1, dtype=torch.float64):
    lin = nn.Linear(n, m, bias=False).to(dtype)
    U = _random_orthonormal(n, dtype=dtype)
    # element-wise mask (m×n): 상위 mask_ratio 를 freeze
    imp = torch.rand(m, n, dtype=dtype)
    k = max(1, int(round(mask_ratio * m * n)))
    thresh = torch.quantile(imp.flatten(), 1 - mask_ratio)
    mask = imp >= thresh
    mod = LinearWSRLoRA(lin, r=r, alpha=alpha, dropout=0.0).to(dtype)
    mod.set_basis_and_mask(U, mask)
    return mod, lin, U, mask


def test_initial_output_equivalence():
    """B=0 초기화 → forward 가 시작 모델(W0)과 정확히 동일해야 한다."""
    mod, lin, U, mask = _make_module()
    x = torch.randn(5, lin.in_features, dtype=torch.float64)
    with torch.no_grad():
        out_ref = lin(x)
        out_wsr = mod(x)
    assert torch.allclose(out_wsr, out_ref, atol=1e-10), \
        f"initial output diff = {(out_wsr - out_ref).abs().max().item()}"


def test_wsr_mask_freeze():
    """B,A 에 임의값 주입 후, (W_final - W0) 의 rotated 표현이 mask=1 위치에서 0."""
    mod, lin, U, mask = _make_module()
    with torch.no_grad():
        mod.lora_B.copy_(torch.randn_like(mod.lora_B))
        mod.lora_A.copy_(torch.randn_like(mod.lora_A))
        W0 = lin.weight.data
        W_final = mod.folded_weight()
        delta_W = W_final - W0
        rotated = delta_W @ U  # = (1-M) ∘ (s·BA)
    frozen_vals = rotated[mask]
    assert frozen_vals.abs().max().item() < 1e-10, \
        f"masked(frozen) rotated delta nonzero: max={frozen_vals.abs().max().item()}"
    # sanity: 자유(1-M) 위치는 실제로 업데이트가 있어야 함
    free_vals = rotated[~mask]
    assert free_vals.abs().max().item() > 1e-6, "free positions should carry the update"


def test_folded_forward_matches_module():
    """dense fold 한 nn.Linear 의 forward 가 원래 모듈과 일치 (저장 정확성)."""
    mod, lin, U, mask = _make_module()
    with torch.no_grad():
        mod.lora_B.copy_(torch.randn_like(mod.lora_B))
        mod.lora_A.copy_(torch.randn_like(mod.lora_A))
        x = torch.randn(5, lin.in_features, dtype=torch.float64)
        out_mod = mod(x)
        folded = nn.Linear(mod.in_features, mod.out_features, bias=False).to(torch.float64)
        folded.weight = nn.Parameter(mod.folded_weight())
        out_folded = folded(x)
    assert torch.allclose(out_mod, out_folded, atol=1e-10), \
        f"fold mismatch = {(out_mod - out_folded).abs().max().item()}"


def test_trainable_parameter_equality():
    """세 방법의 trainable param 수 동일: WSR-LoRA 의 학습 파라미터는 표준 LoRA 의 A,B 와 동일 shape."""
    m, n, r = 12, 16, 4
    mod, lin, U, mask = _make_module(m=m, n=n, r=r)
    # WSR-LoRA trainable = lora_A(r×n) + lora_B(m×r)
    wsr_trainable = mod.lora_A.numel() + mod.lora_B.numel()
    # 표준 LoRA(PEFT) 도 동일 target 에 대해 A(r×n)+B(m×r)
    standard_lora = r * n + m * r
    assert wsr_trainable == standard_lora == (r * n + m * r)
    # weight/U/mask 는 buffer(비학습)
    assert not mod.weight.requires_grad
    assert not mod.UT_forward.requires_grad
