"""
module.py  —  LinearSNWaRP

WaRP-style 재매개변수화 모듈로 SN-Tune을 위한 핵심 구성요소.

핵심 원리
─────────
Phase 1 SVD에서 얻은 safety basis U (정규직교, [in_features, in_features]) 를 사용해
weight W 를 coefficient 공간 C 로 재매개변수화:

    C = W @ U                 (초기화)

Forward pass:
    W_rec = C @ U.T           (W 복원)
    output = F.linear(x, W_rec, bias)

U 가 정규직교이므로:
    C @ U.T = W @ U @ U.T = W   → 모델 출력이 그대로 보존됨

학습 중:
    W 가 아닌 C 의 선택된 좌표만 업데이트
    gradient  ∂L/∂C  는 autograd 가 자동으로 계산
    (∂L/∂C = ∂L/∂W_rec @ U,  같은 shape [out, in])

복원:
    W_final = C.data @ U.T   → nn.Linear 로 변환

메모리 주의
───────────
ffn_down 의 U 는 [intermediate, intermediate] = [11008, 11008]
bfloat16 기준 ≈ 243 MB/layer × 32 layers = 7.8 GB.
기본 layer_type 에서는 ffn_up, attn_q/k/v 만 사용하고
ffn_down 은 선택적으로 추가 권장.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSNWaRP(nn.Module):
    """
    Safety-Neuron WaRP Linear 모듈.

    Parameters
    ----------
    linear_layer : nn.Linear
        변환할 원본 레이어.
    U : torch.Tensor, shape [in_features, in_features]
        Safety basis (정규직교).  Phase 1 SVD의 data['U'].
    coeff_dtype : torch.dtype, optional
        C 의 dtype.  None 이면 linear_layer.weight.dtype 사용.
    cpu_basis : bool
        True 이면 U 를 CPU buffer 로 유지 (VRAM 절약, forward 에서 이동).
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        U: torch.Tensor,
        coeff_dtype: torch.dtype = None,
        cpu_basis: bool = False,
    ):
        super().__init__()

        assert isinstance(linear_layer, nn.Linear), "layer must be nn.Linear"

        self.in_features  = linear_layer.in_features
        self.out_features = linear_layer.out_features

        device = linear_layer.weight.device
        w_dtype = linear_layer.weight.dtype
        c_dtype = coeff_dtype if coeff_dtype is not None else w_dtype

        # ── U 를 buffer 로 등록 ─────────────────────────────────────────
        # cpu_basis=True 이면 CPU 에만 올려두고 forward 에서 CUDA 로 이동.
        U_store = U.to(dtype=torch.float32)
        if not cpu_basis:
            U_store = U_store.to(device=device)
        self.register_buffer("U", U_store)           # [in, in]  float32
        self._cpu_basis = cpu_basis

        # ── C = W @ U  초기화 ──────────────────────────────────────────
        with torch.no_grad():
            W = linear_layer.weight.data.to(device=device, dtype=torch.float32)
            C_init = (W @ self.U.to(device)).to(c_dtype)

        self.coeff = nn.Parameter(C_init)            # [out, in]

        # ── bias ────────────────────────────────────────────────────────
        if linear_layer.bias is not None:
            self.register_buffer(
                "bias", linear_layer.bias.data.clone().to(device=device)
            )
        else:
            self.bias = None

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        W_rec = C @ U.T  →  F.linear(x, W_rec, bias)
        dtype: coeff.dtype (bfloat16 등)
        """
        U = self.U
        if self._cpu_basis:
            U = U.to(device=self.coeff.device)

        # U.T 를 coeff 와 같은 dtype 으로 캐스팅 후 matmul
        U_t = U.T.to(self.coeff.dtype)
        W_rec = self.coeff @ U_t            # [out, in]

        return F.linear(x, W_rec, self.bias)

    # ────────────────────────────────────────────────────────────────────
    def get_restored_weight(self) -> torch.Tensor:
        """
        현재 C 로부터 W_final = C @ U.T 를 반환 (no_grad, data 복사).
        nn.Linear 로 복원할 때 사용.
        """
        with torch.no_grad():
            U = self.U
            if self._cpu_basis:
                U = U.to(self.coeff.device)
            U_t = U.T.to(self.coeff.dtype)
            return (self.coeff.data @ U_t).clone()

    def __repr__(self) -> str:
        return (
            f"LinearSNWaRP("
            f"in={self.in_features}, "
            f"out={self.out_features}, "
            f"basis=[{self.U.shape[0]}×{self.U.shape[1]}], "
            f"cpu_basis={self._cpu_basis})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility: 모델 레이어 접근 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

# layer_type → (sub_module_attr, proj_attr)
LAYER_TYPE_MAP: dict[str, tuple[str, str]] = {
    "ffn_up":   ("mlp",       "up_proj"),
    "ffn_down": ("mlp",       "down_proj"),
    "attn_q":   ("self_attn", "q_proj"),
    "attn_k":   ("self_attn", "k_proj"),
    "attn_v":   ("self_attn", "v_proj"),
}


def get_proj(model, layer_idx: int, ltype: str) -> nn.Module:
    """model.model.layers[layer_idx].<sub>.<proj> 반환."""
    sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
    layer  = model.model.layers[layer_idx]
    sub    = getattr(layer, sub_attr)
    return getattr(sub, proj_attr)


def set_proj(model, layer_idx: int, ltype: str, module: nn.Module) -> None:
    """model.model.layers[layer_idx].<sub>.<proj> 교체."""
    sub_attr, proj_attr = LAYER_TYPE_MAP[ltype]
    layer = model.model.layers[layer_idx]
    sub   = getattr(layer, sub_attr)
    setattr(sub, proj_attr, module)


def convert_to_sn_warp(model, basis_dict: dict, layer_types: list[str],
                        cpu_basis: bool = False) -> None:
    """
    지정 레이어를 LinearSNWaRP 로 in-place 변환.

    Parameters
    ----------
    model      : LLaMA 모델 (AutoModelForCausalLM)
    basis_dict : {(layer_idx, ltype): U_tensor}  — load_all_basis() 결과
    layer_types: e.g. ['ffn_up', 'attn_q', 'attn_k', 'attn_v']
    cpu_basis  : U 를 CPU 에만 올릴지 여부
    """
    import logging
    log = logging.getLogger(__name__)

    n_converted = 0
    num_layers  = len(model.model.layers)

    for layer_idx in range(num_layers):
        for ltype in layer_types:
            key = (layer_idx, ltype)
            if key not in basis_dict:
                continue
            U   = basis_dict[key]
            lin = get_proj(model, layer_idx, ltype)

            if not isinstance(lin, nn.Linear):
                log.warning(f"  Skip layer {layer_idx} {ltype}: already converted or not Linear")
                continue

            sn_mod = LinearSNWaRP(lin, U, cpu_basis=cpu_basis)
            set_proj(model, layer_idx, ltype, sn_mod)
            n_converted += 1

    log.info(f"[convert_to_sn_warp] Converted {n_converted} layers → LinearSNWaRP")


def restore_to_linear(model) -> None:
    """
    모든 LinearSNWaRP 를 nn.Linear 로 복원 (in-place).
    W_final = C @ U.T
    """
    import logging
    log = logging.getLogger(__name__)

    replacements = []
    for mod in model.modules():
        for attr, child in mod.named_children():
            if isinstance(child, LinearSNWaRP):
                replacements.append((mod, attr, child))

    for parent, attr, sn_mod in replacements:
        W_final = sn_mod.get_restored_weight()
        new_lin = nn.Linear(
            sn_mod.in_features,
            sn_mod.out_features,
            bias=(sn_mod.bias is not None),
            dtype=W_final.dtype,
            device=W_final.device,
        )
        new_lin.weight = nn.Parameter(W_final)
        if sn_mod.bias is not None:
            new_lin.bias = nn.Parameter(sn_mod.bias.data.clone())
        setattr(parent, attr, new_lin)

    log.info(f"[restore_to_linear] Restored {len(replacements)} LinearSNWaRP → nn.Linear")
