"""
Phase 2: Importance Scoring (Original Weight Space, Per-layer keep_ratio)

- basis / rotation / basis_coeff를 사용하지 않음
- 원본 모델 weight 공간에서 importance = |dL/dW| 누적
- 각 layer별 keep_ratio로 mask 생성 (mask=1: 중요한 weight, 이후 동결)
"""

import json
import os
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from .phase2_importance_per_layer import Phase2ImportanceScorerPerLayer


class Phase2ImportanceOriginalSpace(Phase2ImportanceScorerPerLayer):
    """Original-space importance scorer (no WaRP, no basis)."""

    def load_basis(self):
        """No-basis 모드: basis 로드를 건너뛴다."""
        layer_types_str = self.args.layer_type
        self.layer_types = [lt.strip() for lt in layer_types_str.split(',')]
        self.basis_data = {}

        self.logger.info("Original-space mode: skipping basis loading")
        self.logger.info(f"✓ Layer types: {self.layer_types}")

    def convert_to_warp_modules(self):
        """No-basis 모드: WaRP 변환을 수행하지 않는다."""
        self.logger.info("Original-space mode: skipping WaRP conversion")

    def reparameterize_weights(self):
        """No-basis 모드: 재매개변수화를 수행하지 않는다."""
        self.logger.info("Original-space mode: skipping weight reparameterization")

    def compute_importance(self):
        """원본 weight 공간에서 per-layer importance 계산."""
        try:
            self.logger.info("=" * 70)
            self.logger.info("Phase 2: Importance Scoring (Original Weight Space)")
            self.logger.info("=" * 70)
            self.logger.info("✅ model.eval() 모드 - 파라미터 업데이트 없음")
            self.logger.info("✅ Importance = |dL/dW| on original weights")
            self.logger.info("=" * 70)

            self.model.eval()

            target_indices = self._parse_target_layers(len(self.model.model.layers))
            target_params = OrderedDict()

            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                for layer_type in self.layer_types:
                    module = self._get_target_module(layer, layer_type)
                    key = (layer_idx, layer_type)
                    target_params[key] = module.weight

            # Target weights만 gradient 추적
            for p in self.model.parameters():
                p.requires_grad = False
            for p in target_params.values():
                p.requires_grad = True

            total_params = sum(p.numel() for p in self.model.parameters())
            tracked_params = sum(p.numel() for p in target_params.values())
            self.logger.info(
                f"✓ Tracking params in original space: {tracked_params:,}/{total_params:,} "
                f"({(tracked_params / max(total_params, 1)) * 100:.2f}%)"
            )

            importances = OrderedDict()
            progress_bar = tqdm(self.dataloader, desc="Computing importance (original space)", total=len(self.dataloader))

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch.get('labels')
                if labels is not None:
                    labels = labels.to(self.model.device)
                else:
                    labels = input_ids.masked_fill(attention_mask == 0, -100)

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss
                valid_tokens = (labels[:, 1:] != -100).sum().item()

                if valid_tokens > 0 and loss is not None and torch.isfinite(loss):
                    loss.backward()

                    for key, param in target_params.items():
                        if param.grad is None:
                            continue
                        grad_abs = param.grad.detach().abs().float()
                        if key not in importances:
                            importances[key] = grad_abs.clone()
                        else:
                            importances[key].add_(grad_abs)

                    self.stats['total_samples'] += int(input_ids.shape[0])
                    self.stats['total_tokens'] += int(valid_tokens)

                self.model.zero_grad(set_to_none=True)

            self.logger.info("=" * 70)
            self.logger.info("✓ Importance computation completed (original space)")
            self.logger.info(f"  - Total samples: {self.stats['total_samples']}")
            self.logger.info(f"  - Total tokens: {self.stats['total_tokens']}")
            self.logger.info("=" * 70)

            self.importances = {
                key: val.detach().cpu().float().numpy() for key, val in importances.items()
            }

        except Exception as e:
            self.logger.error(f"Failed to compute importance (original space): {str(e)}", exc_info=True)
            raise

    def generate_masks(self, keep_ratio=0.1, **kwargs):
        """각 layer별 keep_ratio로 mask 생성 (mask=1: freeze)."""
        try:
            self.logger.info("=" * 70)
            self.logger.info(f"Generating original-space masks per-layer (keep_ratio={keep_ratio})")
            self.logger.info("=" * 70)

            self.masks = {}
            for key, importance in self.importances.items():
                flat = importance.reshape(-1)
                threshold = np.quantile(flat, 1 - keep_ratio)
                mask = (importance >= threshold)
                self.masks[key] = mask.astype(np.bool_)

                frozen_count = int(mask.sum())
                total_count = int(mask.size)
                frozen_ratio = frozen_count / max(total_count, 1) * 100
                layer_idx, layer_type = key
                self.logger.info(
                    f"Layer {layer_idx} ({layer_type}): Frozen {frozen_count}/{total_count} ({frozen_ratio:.2f}%)"
                )

            self.logger.info("=" * 70)
            self.logger.info(f"✓ Masks generated for {len(self.masks)} modules")

        except Exception as e:
            self.logger.error(f"Failed to generate masks (original space): {str(e)}", exc_info=True)
            raise

    def save_masks(self, **kwargs):
        """mask 저장 (Phase3 original-space learner에서 사용)."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                getattr(self.args, 'output_dir', './checkpoints'),
                f'phase2_original_space_{timestamp}',
                'checkpoints',
            )
            masks_dir = os.path.join(checkpoint_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)

            for key, mask in self.masks.items():
                layer_idx, layer_type = key
                layer_type_dir = os.path.join(masks_dir, layer_type)
                os.makedirs(layer_type_dir, exist_ok=True)

                mask_path = os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_mask.pt')
                torch.save({'mask': mask}, mask_path)

            metadata = {
                'phase': 2,
                'space': 'original_weight_space',
                'use_basis': False,
                'masking_strategy': 'per_layer',
                'keep_ratio': getattr(self.args, 'keep_ratio', 0.1),
                'layer_types': self.layer_types,
                'target_layers': self.args.target_layers,
                'timestamp': timestamp,
            }

            metadata_path = os.path.join(masks_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"✓ Original-space masks saved to: {masks_dir}")
            return masks_dir

        except Exception as e:
            self.logger.error(f"Failed to save masks (original space): {str(e)}", exc_info=True)
            raise
