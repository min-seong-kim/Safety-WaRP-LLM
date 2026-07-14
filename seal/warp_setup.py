"""
WaRP Stage-2 세팅 — models/phase3_extra_learning.py의 setup_warp_modules 로직을 복제.

Phase 1(basis)·Phase 2(mask) 산출물을 로드해 HF 모델의 대상 레이어를 LinearWaRP로 바꾸고
재매개변수화 계수를 초기화한다. 저장 형식/수식은 저장소 원본과 100% 일치:

  basis_dir/{layer_type}/layer_{idx:02d}_svd.pt   → {'U','S','UT'}
  masks_dir/{layer_type}/layer_{idx:02d}_mask.pt  → {'mask'}   (1=freeze/안전)

  basis_coeff = W @ U ,  UT_forward = U ,  UT_backward = ∅(V=I)
  forward:  W = (basis_coeff⊙mask).detach() @ Uᵀ + (basis_coeff⊙(1-mask)) @ Uᵀ
  학습:     basis_coeff 만 requires_grad=True (나머지 전부 동결)
"""

import os
import sys

import numpy as np
import torch

# 저장소 루트의 models/ 패키지 import 보장
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.warp_modules import (  # noqa: E402
    WaRPModule,
    restore_to_linear,
    restore_weight,
    switch_to_warp_module,
)

# layer_type → (부모 속성 경로, LinearWaRP가 놓일 속성명)
_LAYER_TYPE_ATTR = {
    "ffn_down": ("mlp", "down_proj"),
    "ffn_up": ("mlp", "up_proj"),
    "ffn_gate": ("mlp", "gate_proj"),
    "attn_q": ("self_attn", "q_proj"),
    "attn_k": ("self_attn", "k_proj"),
    "attn_v": ("self_attn", "v_proj"),
    "attn_o": ("self_attn", "o_proj"),
}


def parse_layer_types(layer_type_str):
    return [lt.strip() for lt in layer_type_str.split(",") if lt.strip()]


def parse_target_layers(target_layers, num_layers):
    """'all' | 'early|middle|late|last' | 단일 인덱스 | 'a-b' 범위 → 인덱스 리스트."""
    if isinstance(target_layers, (list, tuple)):
        return list(target_layers)
    s = str(target_layers).strip()
    if s == "all":
        return list(range(num_layers))
    if s == "early":
        return list(range(0, num_layers // 3))
    if s == "middle":
        return list(range(num_layers // 3, 2 * num_layers // 3))
    if s == "late":
        return list(range(2 * num_layers // 3, num_layers))
    if s == "last":
        return [num_layers - 1]
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(s)]


def _get_module(layer, layer_type):
    parent_attr, child_attr = _LAYER_TYPE_ATTR[layer_type]
    return getattr(getattr(layer, parent_attr), child_attr)


def load_basis(basis_dir, layer_types):
    """basis_dir → {(layer_idx, layer_type): U tensor}."""
    basis = {}
    for lt in layer_types:
        lt_dir = os.path.join(basis_dir, lt)
        if not os.path.isdir(lt_dir):
            continue
        for fn in sorted(f for f in os.listdir(lt_dir)
                         if f.startswith("layer_") and f.endswith("_svd.pt")):
            idx = int(fn.split("_")[1])
            data = torch.load(os.path.join(lt_dir, fn), map_location="cpu")
            basis[(idx, lt)] = data["U"]
    return basis


def load_masks(basis_masks_dir, layer_types):
    """masks_dir → {(layer_idx, layer_type): mask}."""
    masks = {}
    for lt in layer_types:
        lt_dir = os.path.join(basis_masks_dir, lt)
        if not os.path.isdir(lt_dir):
            continue
        for fn in sorted(f for f in os.listdir(lt_dir)
                         if f.startswith("layer_") and f.endswith("_mask.pt")):
            idx = int(fn.split("_")[1])
            data = torch.load(os.path.join(lt_dir, fn), weights_only=False)
            masks[(idx, lt)] = data["mask"]
    return masks


def apply_warp(model, basis_dir, masks_dir, layer_type_str, target_layers,
               no_masks=False, logger=None):
    """
    대상 레이어를 LinearWaRP로 변환 + basis/mask 세팅. 반환: (변환된 model, 통계 dict).

    Phase 3 setup_warp_modules와 동일한 초기화를 수행한다.
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    layer_types = parse_layer_types(layer_type_str)
    num_layers = len(model.model.layers)
    target_indices = parse_target_layers(target_layers, num_layers)

    # 1) Linear → LinearWaRP 변환
    switch_to_warp_module(model, layer_types, target_indices)

    # 2) basis / mask 로드
    basis = load_basis(basis_dir, layer_types)
    masks = {} if no_masks else load_masks(masks_dir, layer_types)
    log(f"[warp] basis loaded: {len(basis)}  masks loaded: {len(masks)}")

    setup_count, total_frozen, total_elems = 0, 0, 0
    for layer_idx in target_indices:
        layer = model.model.layers[layer_idx]
        for lt in layer_types:
            key = (layer_idx, lt)
            if key not in basis:
                continue
            if key not in masks and not no_masks:
                continue

            module = _get_module(layer, lt)
            if not isinstance(module, WaRPModule):
                continue

            W = module.weight.data.clone()
            U = basis[key].to(dtype=W.dtype, device=W.device)

            # basis_coeff = W @ U  (V = I)
            module.basis_coeff.data = W @ U
            module.UT_forward = U.clone().detach()
            module.UT_backward = torch.empty(0, dtype=W.dtype, device=W.device)

            if key in masks:
                m = masks[key]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m)
                m = m.to(device=W.device)
                if m.dtype != torch.bool:
                    m = m > 0.5
            else:
                m = torch.zeros(module.basis_coeff.shape, dtype=torch.bool, device=W.device)
            module.coeff_mask.data = m

            # mask_mode: 2=전체동결, 0=혼합, 1=전체학습
            if torch.all(m):
                module.mask_mode.fill_(2)
            elif torch.any(m):
                module.mask_mode.fill_(0)
            else:
                module.mask_mode.fill_(1)

            module.flag = True
            module.basis_coeff.requires_grad = True

            setup_count += 1
            total_frozen += int(m.sum().item())
            total_elems += int(m.numel())

    # 3) basis_coeff만 학습 (Phase 3 규약)
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "basis_coeff" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False

    frozen_ratio = (total_frozen / total_elems * 100) if total_elems else 0.0
    stats = {
        "warp_modules": setup_count,
        "frozen_coeff_ratio_pct": frozen_ratio,
        "trainable_params": trainable,
        "total_params": total,
    }
    log(f"[warp] modules={setup_count}  frozen coeff={frozen_ratio:.1f}%  "
        f"trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M")
    return model, stats


def restore_and_delinearize(model, logger=None):
    """학습 후 basis_coeff → weight 복원 + LinearWaRP → nn.Linear 되돌리기 (저장 전 필수)."""
    restore_weight(model)
    restore_to_linear(model)
    if logger:
        logger.info("[warp] restored basis_coeff → weight, LinearWaRP → nn.Linear")
    return model
