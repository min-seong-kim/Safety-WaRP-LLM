"""Adapter-aware column-WSR-LoRA 수학 및 artifact 회귀 테스트 (CPU)."""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter_subspace import AdapterSubspaceProjector
from models.adapter_wsr_column import (
    compute_column_scores,
    load_subspaces,
    rotated_adapter_chunk,
    select_directions,
)


def orthogonal(n, dtype=torch.float64):
    return torch.linalg.qr(torch.randn(n, n, dtype=dtype)).Q


def test_rotation_consistency():
    m, n, r = 7, 9, 3
    B, A, U = torch.randn(m, r), torch.randn(r, n), orthogonal(n).float()
    delta = 2.0 * B @ A
    assert torch.allclose((delta @ U) @ U.T, delta, atol=2e-5)


def test_gradient_transform():
    m, n, b = 5, 7, 4
    U = orthogonal(n)
    C = torch.randn(m, n, dtype=torch.float64, requires_grad=True)
    x = torch.randn(b, n, dtype=torch.float64)
    loss = ((x @ U @ C.T) ** 2).sum()
    grad_c, = torch.autograd.grad(loss, C)
    W = (C.detach() @ U.T).requires_grad_()
    loss_w = ((x @ W.T) ** 2).sum()
    grad_w, = torch.autograd.grad(loss_w, W)
    assert torch.allclose(grad_c, grad_w @ U, atol=1e-10)


@pytest.mark.parametrize("aggregation", ["l1", "l2"])
def test_chunked_scores_equal_dense(aggregation):
    m, n, r = 8, 11, 3
    B, A, U, G = torch.randn(m, r), torch.randn(r, n), orthogonal(n).float(), torch.rand(m, n)
    chunked = compute_column_scores(B, A, 1.7, U, G, aggregation=aggregation, chunk_size=3)
    dense = 1.7 * B @ A @ U
    reduce = (lambda x: x.abs().sum(0)) if aggregation == "l1" else (lambda x: torch.linalg.vector_norm(x, dim=0))
    expected = {
        "gradient_only": reduce(G),
        "adapter_magnitude_only": reduce(dense),
        "adapter_taylor": reduce(dense.abs() * G),
    }
    for mode in expected:
        assert torch.allclose(chunked[mode], expected[mode], atol=2e-5)


def test_projection_rank_mapping_and_column_mask_equivalence():
    m, n, r, k = 8, 10, 3, 4
    U = orthogonal(n)
    idx = torch.tensor([0, 2, 5, 8])
    U_S = U[:, idx]
    B, A, W = torch.randn(m, r, dtype=torch.float64), torch.randn(r, n, dtype=torch.float64), torch.randn(m, n, dtype=torch.float64)
    A_perp = A - (A @ U_S) @ U_S.T
    assert torch.linalg.vector_norm(A_perp @ U_S) < 1e-10
    assert torch.linalg.matrix_rank(B @ A_perp) <= r
    assert torch.allclose((W + B @ A_perp) @ U_S, W @ U_S, atol=1e-10)
    D = torch.ones(n, dtype=torch.float64); D[idx] = 0
    assert torch.allclose((B @ A @ U) * D @ U.T, B @ A_perp, atol=1e-10)


def test_invalid_score_shapes_and_selection_values():
    B, A, U = torch.randn(5, 2), torch.randn(2, 7), torch.eye(7)
    with pytest.raises(ValueError, match="shape"):
        compute_column_scores(B, A, 1.0, U, torch.ones(4, 7))
    with pytest.raises(ValueError):
        select_directions(torch.ones(7), keep_ratio=1.1)


def test_partial_artifact_is_not_complete(tmp_path):
    d = tmp_path / "attn_q"; d.mkdir()
    torch.save({"U_S": torch.eye(4)[:, :1]}, d / "layer_00_subspace.pt")
    assert len(load_subspaces(tmp_path)) == 1
    assert not (tmp_path / "report.json").is_file()


class FakeLoraLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(6, 5, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(6, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 5, bias=False)})
        self.scaling = {"default": 1.0}


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].q_proj = FakeLoraLinear()


def test_projector_uses_u_s_and_rejects_bad_shape():
    model = FakeModel()
    key = (0, "attn_q")
    U_S = orthogonal(6).float()[:, :2]
    projector = AdapterSubspaceProjector(model, {key: {"U_S": U_S}}, subspace_key="U_S")
    projector.project()
    A = model.model.layers[0].q_proj.lora_A["default"].weight
    assert torch.linalg.vector_norm(A @ U_S) < 1e-5
    with pytest.raises(ValueError, match="in_dim"):
        AdapterSubspaceProjector(model, {key: {"U_S": torch.ones(5, 2)}}, subspace_key="U_S")
