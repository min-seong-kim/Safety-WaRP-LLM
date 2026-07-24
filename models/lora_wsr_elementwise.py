"""
WSR-LoRA (element-wise) module.

논문 WSR-Tune 의 element-wise safety freeze 를 LoRA 파라미터 예산으로 재현한다.
방향(열) 투영이 아니라 reparameterized 공간에서 element 단위 mask 를 사용하므로
논문 eq.(12) 철학과 일치한다.

수식 (target linear layer, frozen safety weight W0 ∈ R^{m×n}, basis U ∈ R^{n×n} orthonormal):
    basis_coeff = W0 @ U                       (frozen buffer, = W̃0)
    delta       = (1 - M) ∘ ( s · (B @ A) )    (s=α/r, B∈R^{m×r}, A∈R^{r×n}, M∈{0,1}^{m×n})
    W̃_eff       = basis_coeff + delta
    W           = W̃_eff @ U^T   =  W0 + [ (1-M) ∘ (s·BA) ] @ U^T

- B 는 0 초기화 → 시작 시 delta=0 → W=W0 (정확히 safety 모델).
- mask=1(safety-critical) element 는 증분이 0 → 안전 방향 보존.
- 저장은 adapter merge 가 아니라 dense fold: restore_wsr_lora_to_linear() 가
  W = W0 + [(1-M)∘(s·BA)] @ U^T 를 계산해 nn.Linear 로 교체한다 (from_pretrained 로 바로 로드).

기존 warp_modules 관례(UT_forward 에 V=U 저장, W@U 로 basis_coeff 초기화)를 그대로 따른다.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# layer_type -> (parent attr path, module attr name)  (LLaMA)
_LAYER_TYPE_TO_ATTR = {
    "ffn_down": ("mlp", "down_proj"),
    "ffn_up": ("mlp", "up_proj"),
    "ffn_gate": ("mlp", "gate_proj"),
    "attn_q": ("self_attn", "q_proj"),
    "attn_k": ("self_attn", "k_proj"),
    "attn_v": ("self_attn", "v_proj"),
    "attn_o": ("self_attn", "o_proj"),
}


class LinearWSRLoRA(nn.Module):
    """basis 공간 element-wise mask 를 적용하는 LoRA linear.

    trainable: lora_A (r×n), lora_B (m×r) 뿐. basis_coeff/UT_forward/coeff_mask 는 frozen buffer.
    """

    def __init__(self, linear_layer: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert isinstance(linear_layer, nn.Linear), "Layer must be nn.Linear"

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        weight = linear_layer.weight.data.clone()
        dtype = weight.dtype
        device = weight.device

        # W0 (frozen). base forward 에 그대로 사용.
        # (증분만 basis 공간에서 masking 후 U^T 로 회전; W0 자체는 basis_coeff@U^T 와 정확히 동일하므로
        #  basis_coeff 를 따로 들 필요 없음 → 메모리 절약)
        self.register_buffer("weight", weight)
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

        # basis / mask (set_basis_and_mask 에서 채움)
        self.register_buffer("UT_forward", torch.empty(0, dtype=dtype, device=device))
        self.register_buffer(
            "coeff_mask", torch.zeros((self.out_features, self.in_features), dtype=torch.bool, device=device)
        )
        self._basis_ready = False
        self._no_rotation = False   # U=None → ΔW=(1-M)∘(s·BA), 원래공간 (rotation ablation)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

        # LoRA factors. PEFT 관례: A ~ kaiming, B = 0 → 초기 ΔW=0.
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @torch.no_grad()
    def set_basis_and_mask(self, U: torch.Tensor, mask: torch.Tensor):
        """U: (n×n) orthonormal (warp_modules 관례상 V=UT.t() 가 저장됨). mask: (m×n) bool, True=freeze.
        U=None 이면 no-rotation ablation: ΔW=(1-M)∘(s·BA) 를 원래공간에서 그대로 사용."""
        W = self.weight
        if U is None:
            self._no_rotation = True
            self.UT_forward = torch.empty(0, dtype=W.dtype, device=W.device)
        else:
            U = U.to(dtype=W.dtype, device=W.device)
            self.UT_forward = U.clone()

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            mask = mask.to(device=W.device)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            assert mask.shape == (self.out_features, self.in_features), (
                f"mask shape {tuple(mask.shape)} != {(self.out_features, self.in_features)}"
            )
            self.coeff_mask = mask
        self._basis_ready = True

    def _delta_weight(self) -> torch.Tensor:
        """ΔW = [ (1-M) ∘ (s·BA) ] @ U^T  (original weight space).
        no_rotation 이면 U 없이 ΔW = (1-M) ∘ (s·BA)."""
        delta_coeff = self.scaling * (self.lora_B @ self.lora_A)  # (m×n), basis 공간
        delta_coeff = torch.where(self.coeff_mask, torch.zeros_like(delta_coeff), delta_coeff)
        if self._no_rotation:
            return delta_coeff
        return delta_coeff @ self.UT_forward.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._basis_ready, "set_basis_and_mask() must be called before forward"
        base = F.linear(x, self.weight, self.bias)
        # LoRA 증분 경로 (dropout 은 adapter 입력에만, PEFT 관례).
        delta_w = self._delta_weight().to(x.dtype)
        adapter = F.linear(self.lora_dropout(x), delta_w)
        return base + adapter

    @torch.no_grad()
    def folded_weight(self) -> torch.Tensor:
        """저장용 dense weight: W0 + ΔW."""
        return (self.weight + self._delta_weight()).to(self.weight.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}"
        )


def _resolve_target_layers(num_layers, target_layers):
    if isinstance(target_layers, str):
        if target_layers == "all":
            return list(range(num_layers))
        if "-" in target_layers:
            s, e = map(int, target_layers.split("-"))
            return list(range(s, e + 1))
        return [int(target_layers)]
    return list(target_layers)


def switch_to_wsr_lora(model, layer_types, target_layers, r, alpha, dropout=0.0):
    """지정 레이어의 nn.Linear 를 LinearWSRLoRA 로 교체. (basis/mask 는 이후 set_basis_and_mask 로 주입)"""
    num_layers = len(model.model.layers)
    indices = _resolve_target_layers(num_layers, target_layers)
    converted = {}
    for layer_idx in indices:
        layer = model.model.layers[layer_idx]
        for lt in layer_types:
            if lt not in _LAYER_TYPE_TO_ATTR:
                raise ValueError(f"Unknown layer type: {lt}")
            parent_name, attr = _LAYER_TYPE_TO_ATTR[lt]
            parent = getattr(layer, parent_name)
            orig = getattr(parent, attr)
            wsr = LinearWSRLoRA(orig, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wsr)
            converted[(layer_idx, lt)] = wsr
    logger.info(f"✓ Converted {len(converted)} modules to LinearWSRLoRA")
    return converted


def mark_only_lora_trainable(model):
    """lora_A/lora_B 만 학습, 나머지 전부 freeze. trainable param 수 반환."""
    trainable = 0
    for name, p in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B") or ".lora_A" in name or ".lora_B" in name:
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
    return trainable


@torch.no_grad()
def restore_wsr_lora_to_linear(model):
    """LinearWSRLoRA → nn.Linear (dense fold). 저장 전 호출."""
    replacements = []
    for module in model.modules():
        for attr, child in module.named_children():
            if isinstance(child, LinearWSRLoRA):
                replacements.append((module, attr, child))
    for parent, attr, wsr in replacements:
        w = wsr.folded_weight()
        new_linear = nn.Linear(
            wsr.in_features, wsr.out_features,
            bias=wsr.bias is not None, dtype=w.dtype, device=w.device,
        )
        new_linear.weight = nn.Parameter(w.clone())
        if wsr.bias is not None:
            new_linear.bias = nn.Parameter(wsr.bias.data.clone())
        setattr(parent, attr, new_linear)
    logger.info(f"✓ Restored {len(replacements)} LinearWSRLoRA → nn.Linear")
    return model
