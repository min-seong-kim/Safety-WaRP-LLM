"""
Phase 2 (variant): WSR-Tune vs ActSVD mask-structure ablation의 importance/mask 생성.

`wsr_actsvd_ablation_spec.md` §1–§3 구현. 네 arm이 **완전히 같은 gradient 계산 루프**를
공유하고, 오직 (1) 어떤 좌표계에서 basis_coeff를 잡는지, (2) 마스크 단위가 entry/row/column
중 무엇인지만 달라진다.

  arm A : W̃ = W              , entry  mask   (원본 공간)
  arm B : W̃ = U_out^T W      , row    mask   (ActSVD rank freezing) ★
  arm C : W̃ = W U_in         , column mask
  arm D : W̃ = W U_in         , entry  mask   (WSR-Tune)
  D_perm: W̃ = P^T W U_in     , entry  mask   (sanity: D와 동일해야 함)

importance는 부모 클래스(`Phase2ImportanceScorerPerLayer.compute_importance`)를 그대로 쓴다:
  G̃ = Σ_{x∈D_safe} |∂L(x)/∂W̃|
  - model.eval(), optimizer.step() 없음
  - loss는 response 토큰만 (circuit_breakers 로더가 prompt를 -100으로 마스킹)
따라서 arm 간 차이는 좌표계와 마스크 구조뿐이라는 spec의 요구가 코드 수준에서 보장된다.
"""

import json
import os
from datetime import datetime

import numpy as np
import torch

from models.phase2_importance_per_layer import Phase2ImportanceScorerPerLayer
from models.warp_modules import WaRPModule
from .wsr_ablation_masks import (
    arm_spec,
    build_mask,
    check_budget_match,
    mask_report,
)
from .wsr_ablation_reparam import apply_arm_reparameterization


class Phase2AblationImportanceScorer(Phase2ImportanceScorerPerLayer):
    """arm별 좌표계 + 마스크 구조로 Phase 2를 수행."""

    def __init__(self, args, logger, basis_dir, phase0_model_dir):
        super().__init__(args, logger, basis_dir, phase0_model_dir)

        self.arm = getattr(args, 'ablation_arm', 'D')
        self.spec = arm_spec(self.arm)
        self.mask_unit = getattr(args, 'mask_unit', None) or self.spec['mask_unit']
        self.structured_agg = getattr(args, 'structured_agg', 'l2')
        self.structured_rank = getattr(args, 'structured_rank', 'grad')
        self.budget_report = None
        self.reparam_diagnostics = {}

        if self.mask_unit == 'entry' and self.structured_rank == 'spectral':
            raise ValueError("--structured_rank spectral 은 row/column 마스크에서만 의미가 있습니다")

        self.logger.info("=" * 70)
        self.logger.info(f"Phase 2 ablation arm {self.arm}: {self.spec['label']}")
        self.logger.info(f"  - basis side  : {self.spec['basis_side']}")
        self.logger.info(f"  - V (output)  : {self.spec['v_mode']}")
        self.logger.info(f"  - mask unit   : {self.mask_unit}")
        self.logger.info(f"  - rank by     : {self.structured_rank}"
                         + (f" (agg={self.structured_agg})" if self.mask_unit != 'entry' else ""))
        self.logger.info(f"  - note        : {self.spec['note']}")
        self.logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # basis
    # ──────────────────────────────────────────────────────────────────

    def load_basis(self):
        self.layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]

        if self.spec['basis_side'] is None:
            self.basis_data = {}
            self.logger.info(f"arm {self.arm}: 재파라미터화 없음 (U=V=I) — basis 로드 생략")
            return

        if not self.basis_dir:
            raise ValueError(f"arm {self.arm} 은 --basis_dir 가 필요합니다 "
                             f"(basis_side={self.spec['basis_side']})")

        metadata_path = os.path.join(self.basis_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # ⚠️ spec §8 항목 5: 입력측/출력측 기저를 절대 바꿔 쓰지 않는다.
        declared = metadata.get('basis_side')
        expected = self.spec['basis_side']
        if declared is None:
            self.logger.warning(
                f"basis metadata에 basis_side가 없습니다 (레거시 Phase 1 산출물로 판단). "
                f"arm {self.arm} 은 '{expected}' 기저를 요구하므로 input 으로 간주합니다."
            )
            declared = 'input'
        if declared != expected:
            raise ValueError(
                f"arm {self.arm} 은 basis_side='{expected}' 를 요구하지만 "
                f"{self.basis_dir} 의 기저는 '{declared}' 입니다. "
                "ActSVD(출력측)와 WSR(입력측) 기저를 바꿔 쓰면 실험이 무의미해집니다."
            )

        super().load_basis()
        self.logger.info(f"✓ {declared}-side basis 로드 완료 (token_scope="
                         f"{metadata.get('token_scope', 'all')})")

    # ──────────────────────────────────────────────────────────────────
    # 재파라미터화
    # ──────────────────────────────────────────────────────────────────

    def reparameterize_weights(self):
        target_layers = self._parse_target_layers(len(self.model.model.layers))
        for module in self.model.modules():
            if isinstance(module, WaRPModule):
                module.coeff_mask.data.zero_()
                if hasattr(module, 'mask_mode'):
                    module.mask_mode.fill_(1)   # all-zero mask (Phase 2에서는 동결 없음)

        _count, diagnostics = apply_arm_reparameterization(
            self.model,
            spec=self.spec,
            basis_data=self.basis_data,
            layer_types=self.layer_types,
            target_layers=target_layers,
            get_target_module=self._get_target_module,
            seed=getattr(self.args, 'seed', 42),
            logger=self.logger,
            log_prefix=f"phase2/arm{self.arm}",
            basis_dir=self.basis_dir,
        )
        self.reparam_diagnostics = diagnostics

    # ──────────────────────────────────────────────────────────────────
    # 마스크
    # ──────────────────────────────────────────────────────────────────

    def generate_masks(self, keep_ratio=0.1, two_mask=False, **_kwargs):
        if two_mask:
            raise ValueError("ablation arm 에서는 --two_mask 를 지원하지 않습니다 "
                             "(arm 간 비교의 유일한 변수는 basis/mask 구조여야 합니다)")

        self.logger.info("=" * 70)
        self.logger.info(f"arm {self.arm}: generating {self.mask_unit} masks (ρ={keep_ratio}, "
                         f"rank_by={self.structured_rank})")
        self.logger.info("=" * 70)

        self.masks = {}
        for key in sorted(self.importances.keys()):
            scores = self.importances[key]
            mask = build_mask(
                scores,
                rho=keep_ratio,
                mask_unit=self.mask_unit,
                agg=self.structured_agg,
                rank_by=self.structured_rank,
            )
            self.masks[key] = mask.astype(np.bool_)

            layer_idx, layer_type = key
            frozen = int(mask.sum())
            self.logger.info(
                f"Layer {layer_idx:2d} ({layer_type:9s}) shape={mask.shape} "
                f"frozen={frozen:,} ({frozen / mask.size * 100:.2f}%)"
                + (f" k={int(mask.any(axis=1).sum())} rows" if self.mask_unit == 'row' else "")
                + (f" k={int(mask.any(axis=0).sum())} cols" if self.mask_unit == 'column' else "")
            )

        self.budget_report = mask_report(self.masks, keep_ratio, self.mask_unit)
        self.budget_report.update({
            'arm': self.arm,
            'basis_side': self.spec['basis_side'],
            'v_mode': self.spec['v_mode'],
            'rank_by': self.structured_rank,
            'agg': self.structured_agg if self.mask_unit != 'entry' else None,
        })

        ok, msg = check_budget_match(self.budget_report)
        self.logger.info("=" * 70)
        self.logger.info(f"arm {self.arm} total frozen = "
                         f"{self.budget_report['total_frozen']:,} / "
                         f"{self.budget_report['total_numel']:,} "
                         f"({self.budget_report['total_frozen_ratio'] * 100:.3f}%)")
        self.logger.info(msg)
        self.logger.info("=" * 70)
        if not ok:
            # 실패해도 산출물은 남기되, 리포트에 표시해 비교에서 걸러지게 한다.
            self.logger.error("⚠️ 예산 매칭 실패 — 이 arm 은 다른 arm 과 공정 비교할 수 없습니다.")
        self.budget_report['budget_ok'] = bool(ok)

    def save_masks(self, two_mask=False, **_kwargs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(
            getattr(self.args, 'output_dir', './checkpoints'),
            f'phase2_arm{self.arm}_{timestamp}',
            'checkpoints',
        )
        masks_dir = os.path.join(checkpoint_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)

        for key, mask in self.masks.items():
            layer_idx, layer_type = key
            layer_type_dir = os.path.join(masks_dir, layer_type)
            os.makedirs(layer_type_dir, exist_ok=True)
            torch.save({'mask': mask},
                       os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_mask.pt'))

        metadata = {
            'phase': 2,
            'ablation_arm': self.arm,
            'arm_label': self.spec['label'],
            'basis_side': self.spec['basis_side'],
            'v_mode': self.spec['v_mode'],
            'mask_unit': self.mask_unit,
            'rank_by': self.structured_rank,
            'agg': self.structured_agg,
            'keep_ratio': getattr(self.args, 'keep_ratio', 0.1),
            'masking_strategy': 'per_layer',
            'layer_types': self.layer_types,
            'target_layers': self.args.target_layers,
            'basis_dir': self.basis_dir,
            'phase0_model': self.phase0_model_dir,
            'total_frozen': self.budget_report['total_frozen'] if self.budget_report else None,
            'budget_ok': self.budget_report.get('budget_ok') if self.budget_report else None,
            'importance_samples': self.stats['total_samples'],
            'importance_tokens': self.stats['total_tokens'],
            'timestamp': timestamp,
        }
        with open(os.path.join(masks_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.budget_report is not None:
            with open(os.path.join(masks_dir, 'budget_report.json'), 'w') as f:
                json.dump(self.budget_report, f, indent=2)

        if self.reparam_diagnostics:
            with open(os.path.join(masks_dir, 'reparam_diagnostics.json'), 'w') as f:
                json.dump(self.reparam_diagnostics, f, indent=2)

        self.logger.info(f"✓ arm {self.arm} masks saved to: {masks_dir}")
        return masks_dir
