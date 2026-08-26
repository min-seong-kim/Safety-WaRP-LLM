"""
Phase 3 (variant): WSR-Tune vs ActSVD ablation의 다운스트림 학습.

`Phase3IncrementalLearner`를 그대로 상속하므로 **데이터 로딩 · Trainer 설정 · 옵티마이저 ·
저장 경로가 arm 간 100% 동일**하다 (spec §2 "Fixed setup (identical across all arms)").
바뀌는 것은 `setup_warp_modules` 안의 좌표계(U, V)와 마스크 구조뿐이다.

  arm A : W = basis_coeff                      (원본 공간, entry mask)
  arm B : W = U_out @ basis_coeff              (ActSVD 출력측, row mask) ★
  arm C : W = basis_coeff @ U_in^T             (safety 입력측, column mask)
  arm D : W = basis_coeff @ U_in^T             (WSR-Tune, entry mask)
  D_perm: W = P @ basis_coeff @ U_in^T         (sanity)

동결은 forward의 `torch.where(coeff_mask, basis_coeff.detach(), basis_coeff)`로 이루어지므로,
arm B에서 행 i 전체가 mask=1이면 그 출력 방향의 계수가 학습 중 절대 변하지 않는다:
    u_i^T ΔW = 0  ∀t  →  (W+ΔW) 의 u_i 성분이 safety 모델과 동일하게 유지된다.
이것이 Wei et al. (2024) footnote 9이 "U^u, U^s로는 쉽게 달성할 수 없다"고 한 rank-level
freezing이며, 좌표 변환을 거치면 단순 행 마스킹이 된다.
"""

import json
import os

import numpy as np
import torch

from models.phase3_extra_learning import Phase3IncrementalLearner
from models.warp_modules import WaRPModule
from .wsr_ablation_masks import arm_spec
from .wsr_ablation_reparam import apply_arm_reparameterization


class Phase3AblationLearner(Phase3IncrementalLearner):
    """arm별 좌표계/마스크로 다운스트림 fine-tuning."""

    def __init__(self, args, logger, basis_dir, masks_dir, phase0_model_dir):
        super().__init__(args, logger, basis_dir, masks_dir, phase0_model_dir)

        self.arm = getattr(args, 'ablation_arm', 'D')
        self.spec = arm_spec(self.arm)
        self.reparam_diagnostics = {}

        self.logger.info("=" * 70)
        self.logger.info(f"Phase 3 ablation arm {self.arm}: {self.spec['label']}")
        self.logger.info(f"  - basis side : {self.spec['basis_side']}")
        self.logger.info(f"  - V (output) : {self.spec['v_mode']}")
        self.logger.info(f"  - mask unit  : {self.spec['mask_unit']}")
        self.logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────

    def load_basis(self):
        self.layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]

        if self.spec['basis_side'] is None:
            self.basis_data = {}
            self.logger.info(f"arm {self.arm}: 재파라미터화 없음 (U=V=I) — basis 로드 생략")
            return

        if not self.basis_dir:
            raise ValueError(f"arm {self.arm} 은 --basis_dir 가 필요합니다")

        with open(os.path.join(self.basis_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        declared = metadata.get('basis_side') or 'input'
        if declared != self.spec['basis_side']:
            raise ValueError(
                f"arm {self.arm} 은 basis_side='{self.spec['basis_side']}' 를 요구하지만 "
                f"{self.basis_dir} 는 '{declared}' 입니다 (ActSVD 출력측 ↔ WSR 입력측 혼동)."
            )

        super().load_basis()

    def load_masks(self):
        super().load_masks()

        if getattr(self.args, 'no_masks', False):
            return

        meta_path = os.path.join(self.masks_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                mask_meta = json.load(f)
            mask_arm = mask_meta.get('ablation_arm')
            if mask_arm is not None and mask_arm != self.arm:
                raise ValueError(
                    f"mask는 arm {mask_arm} 으로 만들어졌는데 Phase 3은 arm {self.arm} 으로 "
                    f"실행 중입니다 (masks_dir={self.masks_dir}). "
                    "importance와 마스크가 서로 다른 좌표계를 가리키게 됩니다."
                )
            self.logger.info(
                f"✓ mask metadata: arm={mask_arm} unit={mask_meta.get('mask_unit')} "
                f"rank_by={mask_meta.get('rank_by')} total_frozen={mask_meta.get('total_frozen')}"
            )

    # ──────────────────────────────────────────────────────────────────

    def setup_warp_modules(self):
        self.logger.info("=" * 70)
        self.logger.info(f"Setting up WaRP modules — ablation arm {self.arm}")
        self.logger.info("=" * 70)

        self.warp_monitors = []
        target_indices = self._parse_target_layers(len(self.model.model.layers))
        no_masks = getattr(self.args, 'no_masks', False)

        # 1) 좌표계 설정 (Phase 2와 동일한 규약)
        count, diagnostics = apply_arm_reparameterization(
            self.model,
            spec=self.spec,
            basis_data=self.basis_data,
            layer_types=self.layer_types,
            target_layers=target_indices,
            get_target_module=self._get_target_module,
            seed=getattr(self.args, 'seed', 42),
            logger=self.logger,
            log_prefix=f"phase3/arm{self.arm}",
            basis_dir=self.basis_dir,
        )
        self.reparam_diagnostics = diagnostics

        # 2) 마스크 적용
        total_frozen = 0
        total_elems = 0
        setup_count = 0

        for layer_idx in target_indices:
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                key = (layer_idx, layer_type)
                module = self._get_target_module(layer, layer_type)
                if not isinstance(module, WaRPModule):
                    continue

                coeff_shape = tuple(module.basis_coeff.shape)
                device = module.basis_coeff.device

                if key in self.masks:
                    mask = self.masks[key]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(np.ascontiguousarray(mask))
                    mask = mask.to(device=device)
                    if mask.dtype != torch.bool:
                        mask = mask > 0.5
                    if tuple(mask.shape) != coeff_shape:
                        raise ValueError(
                            f"{key}: mask shape {tuple(mask.shape)} != basis_coeff shape "
                            f"{coeff_shape}. arm {self.arm} 의 좌표계와 다른 마스크입니다."
                        )
                elif no_masks:
                    mask = torch.zeros(coeff_shape, dtype=torch.bool, device=device)
                else:
                    raise ValueError(
                        f"{key} 에 대한 마스크가 없습니다 (masks_dir={self.masks_dir}). "
                        "arm 간 비교에서는 모든 대상 모듈이 동일하게 마스킹되어야 합니다."
                    )

                module.coeff_mask.data = mask
                if hasattr(module, 'mask_mode'):
                    if bool(torch.all(mask)):
                        module.mask_mode.fill_(2)
                    elif bool(torch.any(mask)):
                        module.mask_mode.fill_(0)
                    else:
                        module.mask_mode.fill_(1)

                module.flag = True
                module.basis_coeff.requires_grad = True
                self._register_warp_monitor(module, layer_idx, layer_type, mask)

                frozen = int(mask.sum().item())
                total_frozen += frozen
                total_elems += int(mask.numel())
                setup_count += 1

                extra = ""
                if self.spec['mask_unit'] == 'row':
                    extra = f" | frozen output directions: {int(mask.any(dim=1).sum().item())}/{mask.shape[0]}"
                elif self.spec['mask_unit'] == 'column':
                    extra = f" | frozen input directions: {int(mask.any(dim=0).sum().item())}/{mask.shape[1]}"

                self.logger.info(
                    f"Layer {layer_idx:2d} ({layer_type:9s}) coeff={coeff_shape} "
                    f"frozen={frozen:,} ({frozen / max(mask.numel(), 1) * 100:.2f}%){extra}"
                )

        self.logger.info("=" * 70)
        self.logger.info(f"✓ arm {self.arm}: {setup_count} modules "
                         f"(reparameterized {count})")
        self.logger.info(f"✓ frozen scalars: {total_frozen:,} / {total_elems:,} "
                         f"({total_frozen / max(total_elems, 1) * 100:.3f}%)")
        self.logger.info("  forward: W = V @ where(mask, coeff.detach(), coeff) @ U^T")
        self.logger.info("=" * 70)

        self.ablation_summary = {
            'arm': self.arm,
            'arm_label': self.spec['label'],
            'basis_side': self.spec['basis_side'],
            'v_mode': self.spec['v_mode'],
            'mask_unit': self.spec['mask_unit'],
            'num_modules': setup_count,
            'total_frozen': total_frozen,
            'total_numel': total_elems,
            'reconstruction_rel_err_mean': float(np.mean(list(diagnostics.values()))) if diagnostics else 0.0,
            'reconstruction_rel_err_max': float(max(diagnostics.values())) if diagnostics else 0.0,
        }
        self._log_warp_monitor_overview()

    # ──────────────────────────────────────────────────────────────────

    def train(self):
        """부모 학습을 그대로 쓰고, 결과 디렉토리에 arm 요약을 남긴다."""
        final_model_path = super().train()
        try:
            summary = dict(getattr(self, 'ablation_summary', {}))
            summary.update({
                'basis_dir': self.basis_dir,
                'masks_dir': self.masks_dir,
                'phase0_model': self.phase0_model_dir,
                'final_model': final_model_path,
                'epochs': getattr(self.args, 'epochs', None),
                'learning_rate': getattr(self.args, 'utility_lr', None),
                'batch_size': getattr(self.args, 'batch_size', None),
                'gradient_accumulation_steps': getattr(self.args, 'gradient_accumulation_steps', None),
                'phase3_dataset': getattr(self.args, 'phase3_dataset', None),
                'keep_ratio': getattr(self.args, 'keep_ratio', None),
                'reparam_diagnostics': self.reparam_diagnostics,
            })
            out_dir = os.path.dirname(final_model_path) or final_model_path
            with open(os.path.join(out_dir, 'ablation_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"✓ ablation summary saved to {out_dir}/ablation_summary.json")
        except Exception as e:  # 요약 저장 실패가 학습 결과를 버리게 하지 않는다
            self.logger.warning(f"ablation summary 저장 실패: {e}")
        return final_model_path
