"""
WSR-Tune vs ActSVD ablation: arm 사양 → LinearWaRP 재파라미터화 (Phase 2/3 공용).

Phase 2(중요도 측정)와 Phase 3(다운스트림 학습)이 **반드시 동일한 좌표계**를 써야
importance G̃ 와 mask 가 같은 것을 가리킨다. 그래서 세팅 로직을 여기 한 곳에만 둔다.

규약 (models/warp_modules.LinearWaRP 와 일치):

    W  =  V @ basis_coeff @ U^T           (forward에서 매 step 재구성)
    basis_coeff = V^T W U
    UT_forward  = U    (입력측 기저, n×n; 비어 있으면 U=I)
    UT_backward = V^T  (출력측 기저 전치, m×m; 비어 있으면 V=I)

  arm A : U=I,      V=I
  arm B : U=I,      V=U_out   (ActSVD)     → UT_backward = U_out^T
  arm C : U=U_in,   V=I
  arm D : U=U_in,   V=I
  D_perm: U=U_in,   V=P (signed permutation)
"""

import numpy as np
import torch

from .warp_modules import WaRPModule, make_signed_permutation


def permutation_seed(base_seed, layer_idx, layer_type):
    """(layer, type)마다 다르지만 Phase 2/3 간에는 동일한 결정적 시드."""
    return (int(base_seed) * 1_000_003 + int(layer_idx) * 131
            + (hash(layer_type) % 100_000)) % (2 ** 31)


def apply_arm_reparameterization(
    model,
    *,
    spec,
    basis_data,
    layer_types,
    target_layers,
    get_target_module,
    seed=42,
    logger=None,
    log_prefix="ablation",
    basis_dir=None,
):
    """
    arm 사양대로 모든 대상 LinearWaRP를 재파라미터화한다.

    Returns:
        (count, diagnostics) — diagnostics는 모듈별 복원 상대오차
        ‖V·coeff·U^T − W‖_F / ‖W‖_F. bf16 기저에서는 1e-3 수준이 정상이며,
        arm 간에 이 값이 비슷해야 "좌표계만 바꿨다"는 주장이 성립한다.
    """
    side = spec['basis_side']
    v_mode = spec.get('v_mode', 'identity')
    diagnostics = {}
    count = 0

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        for layer_type in layer_types:
            key = (layer_idx, layer_type)
            module = get_target_module(layer, layer_type)
            if not isinstance(module, WaRPModule):
                if logger:
                    logger.warning(f"[{log_prefix}] Layer {layer_idx} {layer_type}: WaRP 모듈이 아님 — 건너뜀")
                continue

            W = module.weight.data
            dtype, device = W.dtype, W.device
            empty = torch.empty(0, dtype=dtype, device=device)

            U = None    # 입력측 (n×n)
            Vt = None   # 출력측 V^T (m×m)

            if side is not None:
                if key not in basis_data:
                    raise ValueError(f"[{log_prefix}] basis에 {key} 가 없습니다 (basis_dir={basis_dir})")
                B = basis_data[key]['U'].to(dtype=dtype, device=device)
                if side == 'input':
                    if B.shape[0] != W.shape[1] or B.shape[1] != W.shape[1]:
                        raise ValueError(
                            f"[{log_prefix}] {key}: 입력측 기저 차원 불일치 "
                            f"basis={tuple(B.shape)} vs W={tuple(W.shape)} (n={W.shape[1]} 필요)")
                    U = B.contiguous()
                elif side == 'output':
                    if B.shape[0] != W.shape[0] or B.shape[1] != W.shape[0]:
                        raise ValueError(
                            f"[{log_prefix}] {key}: 출력측 기저 차원 불일치 "
                            f"basis={tuple(B.shape)} vs W={tuple(W.shape)} (m={W.shape[0]} 필요)")
                    Vt = B.t().contiguous()      # UT_backward = V^T = U_out^T
                else:
                    raise ValueError(f"Unknown basis_side: {side!r}")

            if v_mode == 'signed_permutation':
                P, _perm, _signs = make_signed_permutation(
                    W.shape[0], seed=permutation_seed(seed, layer_idx, layer_type),
                    dtype=dtype, device=device)
                Vt = P.t().contiguous()

            # basis_coeff = V^T W U
            coeff = W
            if Vt is not None:
                coeff = Vt @ coeff
            if U is not None:
                coeff = coeff @ U

            module.basis_coeff.data = coeff.contiguous()
            module.UT_forward = U if U is not None else empty
            module.UT_backward = Vt if Vt is not None else empty
            module.flag = True

            with torch.no_grad():
                recon = module.basis_coeff.data
                if U is not None:
                    recon = recon @ module.UT_forward.t()
                if Vt is not None:
                    recon = module.UT_backward.t() @ recon
                rel = ((recon - W).float().norm()
                       / W.float().norm().clamp_min(1e-12)).item()
            diagnostics[f'{layer_type}/layer_{layer_idx:02d}'] = rel
            count += 1

    if logger and diagnostics:
        values = list(diagnostics.values())
        worst_key = max(diagnostics, key=diagnostics.get)
        logger.info(
            f"[{log_prefix}] reparameterized {count} modules "
            f"(side={side}, V={v_mode}) | reconstruction rel-err "
            f"mean={float(np.mean(values)):.3e} worst={diagnostics[worst_key]:.3e} @ {worst_key}"
        )

    return count, diagnostics
