"""
Phase 3: Downstream finetuning with original-space importance mask

- basis / rotation / basis_coeff를 사용하지 않음
- Phase 2에서 만든 original-space mask를 그대로 사용
- mask=1 위치의 gradient를 hook으로 0 처리하여 동결
- 나머지 파라미터는 학습 가능
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from .phase3_extra_learning import Phase3IncrementalLearner


class OriginalSpaceMaskRestoreCallback(TrainerCallback):
    """
    Restores frozen weight elements (mask=1) after every optimizer step.

    gradient hooks zero out the gradient for mask=1 positions, but AdamW
    weight decay (λθ) is applied independently and still modifies those
    elements each step. This callback saves the initial frozen values at
    construction time and writes them back after every optimizer.step(),
    guaranteeing true parameter freeze for mask=1 positions.
    """

    def __init__(self, frozen_weight_specs):
        # frozen_weight_specs: list of (param, bool_mask, frozen_vals)
        self._specs = frozen_weight_specs

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each optimizer step — restore frozen weight elements."""
        for param, mask, frozen_vals in self._specs:
            with torch.no_grad():
                param.data[mask] = frozen_vals
        return control


class Phase3OriginalSpaceMaskedLearner(Phase3IncrementalLearner):
    """Original-space masked finetuning learner (no WaRP)."""

    def __init__(self, args, logger, basis_dir, masks_dir, phase0_model_dir):
        super().__init__(args, logger, basis_dir, masks_dir, phase0_model_dir)
        self.grad_hooks = []
        self.frozen_weight_specs = []  # (param, bool_mask, frozen_vals) for RestoreCallback

    def load_basis(self):
        """No-basis 모드: basis 로드 생략."""
        self.basis_data = {}
        self.layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]
        self.logger.info("Original-space mode: skipping basis loading")

    def load_model(self):
        """원본 모델 로드 (WaRP 변환 없음)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self.logger.info(f"Loading model from {self.phase0_model_dir} (no rotation, no WaRP)...")

            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.phase0_model_dir,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True,
            )

            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.phase0_model_dir,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info("✓ Base model/tokenizer loaded")
            self.logger.info(
                f"  - Input formatting: {'chat template' if self._is_instruct_model() else 'Question/Answer plain text'}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load original-space model: {str(e)}", exc_info=True)
            raise

    def setup_warp_modules(self):
        """
        Original-space mask를 gradient hook으로 적용.

        mask=1 -> grad 0 (freeze)
        mask=0 -> grad 통과
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("Applying original-space freeze masks (no basis, no WaRP)")
            self.logger.info("=" * 70)

            # 모든 파라미터 학습 가능으로 열고, 마스크 훅으로 부분 동결
            total_params = 0
            for p in self.model.parameters():
                p.requires_grad = True
                total_params += p.numel()

            target_indices = self._parse_target_layers(len(self.model.model.layers))
            hook_count = 0
            frozen_weights = 0
            tracked_weights = 0

            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)
                    if key not in self.masks:
                        continue

                    module = self._get_target_module(layer, layer_type)
                    param = module.weight

                    mask = self.masks[key]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    mask = mask.to(device=param.device)
                    if mask.dtype != torch.bool:
                        mask = mask > 0.5

                    if tuple(mask.shape) != tuple(param.shape):
                        raise ValueError(
                            f"Mask shape mismatch at layer {layer_idx} ({layer_type}): "
                            f"mask={tuple(mask.shape)} vs weight={tuple(param.shape)}"
                        )

                    def _make_hook(local_mask):
                        def _hook(grad):
                            if grad is None:
                                return None
                            grad = grad.clone()
                            grad[local_mask] = 0
                            return grad
                        return _hook

                    handle = param.register_hook(_make_hook(mask))
                    self.grad_hooks.append(handle)
                    hook_count += 1

                    # Save frozen values for weight-decay restore callback
                    with torch.no_grad():
                        frozen_vals = param.data[mask].clone()
                    self.frozen_weight_specs.append((param, mask, frozen_vals))

                    fcnt = int(mask.sum().item())
                    tcnt = int(mask.numel())
                    frozen_weights += fcnt
                    tracked_weights += tcnt

                    self.logger.info(
                        f"Layer {layer_idx} ({layer_type}): frozen {fcnt}/{tcnt} ({(fcnt / max(tcnt, 1)) * 100:.2f}%)"
                    )

            self.logger.info("=" * 70)
            self.logger.info(f"✓ Applied hooks: {hook_count}")
            self.logger.info(
                f"✓ Frozen weights in target modules: {frozen_weights:,}/{tracked_weights:,} "
                f"({(frozen_weights / max(tracked_weights, 1)) * 100:.2f}%)"
            )
            self.logger.info(f"✓ Total model params (trainable with partial freeze hooks): {total_params:,}")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"Failed to apply original-space masks: {str(e)}", exc_info=True)
            raise

    def _train_with_trainer_original_space(self):
        from transformers import Trainer, TrainingArguments

        phase3_dataset = getattr(self.args, 'phase3_dataset', 'gsm8k')
        dataset_name_map = {
            'gsm8k': 'GSM8K',
            'metamath': 'MetaMath',
            'math': 'Hendrycks MATH',
            'safety': 'Safety (Circuit Breakers)',
        }
        dataset_name = dataset_name_map.get(phase3_dataset, phase3_dataset)

        self.logger.info("=" * 70)
        self.logger.info(f"Phase 3: Original-space masked finetuning with {dataset_name}")
        self.logger.info("=" * 70)

        epochs = getattr(self.args, 'epochs', 3)
        learning_rate = getattr(self.args, 'utility_lr', 1e-5)
        batch_size = self.args.batch_size
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)
        warmup_ratio = getattr(self.args, 'warmup_ratio', 0.1)
        lr_scheduler_type = getattr(self.args, 'lr_scheduler_type', 'linear')
        max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
        logging_steps = getattr(self.args, 'logging_steps', 10)
        gradient_checkpointing = getattr(self.args, 'gradient_checkpointing', False)
        weight_decay = getattr(self.args, 'base_weight_decay', 0.01)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(
            getattr(self.args, 'output_dir', './checkpoints'),
            f'phase3_original_space_{timestamp}',
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=True if self.args.dtype == 'bfloat16' else False,
            fp16=True if self.args.dtype == 'float16' else False,
            report_to="none",
            remove_unused_columns=False,
            optim="adamw_torch",
            gradient_checkpointing=gradient_checkpointing,
        )

        @dataclass
        class DataCollatorForCausalLMWithPadding:
            tokenizer: object

            def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
                max_len = max(len(f["input_ids"]) for f in features)
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id

                input_ids, attention_mask, labels = [], [], []
                for f in features:
                    l = len(f["input_ids"])
                    pad_len = max_len - l
                    input_ids.append(f["input_ids"] + [pad_id] * pad_len)
                    attention_mask.append(f["attention_mask"] + [0] * pad_len)
                    labels.append(f["labels"] + [-100] * pad_len)

                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }

        data_collator = DataCollatorForCausalLMWithPadding(self.tokenizer)
        self.model.train()

        restore_callback = OriginalSpaceMaskRestoreCallback(self.frozen_weight_specs)
        self.logger.info(
            f"[OriginalSpaceMaskRestoreCallback] registered {len(self.frozen_weight_specs)} "
            f"param tensors for post-step weight-decay restore"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[restore_callback],
        )

        if phase3_dataset == 'safety':
            self.logger.info("Safety dataset: shuffle disabled (sequential order)")

            def _get_train_dataloader_no_shuffle():
                return DataLoader(
                    self.dataset,
                    batch_size=self.args.batch_size,
                    sampler=torch.utils.data.SequentialSampler(self.dataset),
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory,
                    persistent_workers=training_args.dataloader_persistent_workers,
                )

            trainer.get_train_dataloader = _get_train_dataloader_no_shuffle

        trainer.train()

        final_model_path = os.path.join(checkpoint_dir, 'final_model')
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        metadata = {
            'phase': 3,
            'mode': 'original_space_mask',
            'use_basis': False,
            'masks_dir': self.masks_dir,
            'phase0_model': self.phase0_model_dir,
            'phase3_dataset': phase3_dataset,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'lr_scheduler_type': lr_scheduler_type,
            'optimizer': 'adamw_torch',
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'total_samples': len(self.dataset),
            'timestamp': timestamp,
        }

        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info("=" * 70)
        self.logger.info("Phase 3 Completed (Original Space)")
        self.logger.info(f"  - Final model: {final_model_path}")
        self.logger.info(f"  - Metadata: {metadata_path}")
        self.logger.info("=" * 70)

        return final_model_path

    def train(self):
        try:
            return self._train_with_trainer_original_space()
        except Exception as e:
            self.logger.error(f"Original-space training failed: {str(e)}", exc_info=True)
            raise
