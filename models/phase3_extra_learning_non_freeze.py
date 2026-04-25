"""
Phase 3: Incremental Learning (Non-Freeze Variant)

목표:
- WaRP 마스크 기반 중요 파라미터 동결(detach)은 유지
- WaRP 비적용 레이어를 포함한 나머지 파라미터는 학습 가능

핵심 차이:
- 기존 phase3_extra_learning.py: basis_coeff만 학습
- 본 파일: requires_grad를 전체 파라미터에 대해 True로 설정
  (WaRP의 중요 파라미터 동결은 forward의 mask+detach로 계속 보장)
"""

import os
import json
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

from transformers import TrainerCallback

from .warp_modules import WaRPModule, restore_weight, restore_to_linear
from .phase3_extra_learning import Phase3IncrementalLearner


class WaRPMaskRestoreCallback(TrainerCallback):
    """
    Restores frozen basis_coeff elements (mask=1) after every optimizer step.

    WaRP's forward() uses mask+detach() to block gradients for mask=1 elements,
    but AdamW weight decay (λθ) is applied independently of the gradient graph and
    still modifies those elements each step.  This callback saves the initial
    (frozen) values at construction time and writes them back after every
    optimizer.step(), guaranteeing true parameter freeze for mask=1 positions.
    """

    def __init__(self, model):
        self._specs = []  # list of (module, mask, frozen_vals)
        for module in model.modules():
            if isinstance(module, WaRPModule):
                mask = module.coeff_mask
                if mask is not None and mask.any():
                    with torch.no_grad():
                        frozen_vals = module.basis_coeff.data[mask].clone()
                    self._specs.append((module, mask, frozen_vals))

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each optimizer step — restore frozen basis_coeff elements."""
        for module, mask, frozen_vals in self._specs:
            with torch.no_grad():
                module.basis_coeff.data[mask] = frozen_vals
        return control


class Phase3IncrementalLearnerNonFreeze(Phase3IncrementalLearner):
    """
    Phase 3 Non-Freeze 학습기

    - WaRP layer: mask=1은 detach로 동결, mask=0은 학습
    - Non-WaRP layer: 파라미터 업데이트 허용
    """

    def train(self):
        try:
            from transformers import Trainer, TrainingArguments

            phase3_dataset = getattr(self.args, 'phase3_dataset', 'gsm8k')
            dataset_name_map = {
                'gsm8k': 'GSM8K',
                'metamath': 'MetaMath',
                'math': 'Hendrycks MATH',
                'safety': 'Safety (Circuit Breakers)',
            }
            dataset_name = dataset_name_map.get(phase3_dataset, phase3_dataset)

            self.logger.info("="*70)
            self.logger.info(f"Phase 3: Incremental Learning with {dataset_name} (Non-Freeze)")
            self.logger.info("="*70)

            epochs = getattr(self.args, 'epochs', 3)
            learning_rate = getattr(self.args, 'utility_lr', 1e-5)
            configured_weight_decay = getattr(self.args, 'base_weight_decay', 0.01)
            effective_weight_decay = configured_weight_decay
            batch_size = self.args.batch_size
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)

            if configured_weight_decay > 0:
                self.logger.info(
                    f"Non-Freeze 모드: weight_decay={configured_weight_decay} 적용 "
                    f"(WaRPMaskRestoreCallback으로 mask=1 basis_coeff drift 완전 차단)"
                )

            # ✅ 모든 파라미터 학습 허용
            # WaRP의 중요 파라미터 동결은 forward 내부 mask+detach로 적용됨
            trainable_params = 0
            total_params = 0
            basis_coeff_params = 0

            for name, param in self.model.named_parameters():
                total_params += param.numel()
                param.requires_grad = True
                trainable_params += param.numel()
                if 'basis_coeff' in name:
                    basis_coeff_params += param.numel()

            # WaRP mask 기반 "실질 동결" 통계 (requires_grad=False와 별개)
            masked_frozen_coeff_elems = 0
            masked_total_coeff_elems = 0
            for module in self.model.modules():
                if isinstance(module, WaRPModule) and module.coeff_mask is not None and module.coeff_mask.numel() > 0:
                    mask = module.coeff_mask
                    frozen = (mask > 0.5).sum().item()
                    masked_frozen_coeff_elems += int(frozen)
                    masked_total_coeff_elems += mask.numel()

            masked_trainable_coeff_elems = masked_total_coeff_elems - masked_frozen_coeff_elems

            self.logger.info("Parameter freeze status (Non-Freeze mode):")
            self.logger.info(f"  - Total params: {total_params:,}")
            self.logger.info(f"  - Trainable params: {trainable_params:,}")
            self.logger.info(f"  - basis_coeff params: {basis_coeff_params:,}")
            self.logger.info(f"  - Frozen params (requires_grad=False): {total_params - trainable_params:,}")
            self.logger.info(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")
            self.logger.info(f"  - WaRP masked frozen coeff elems: {masked_frozen_coeff_elems:,}/{masked_total_coeff_elems:,}")
            if masked_total_coeff_elems > 0:
                self.logger.info(
                    f"  - WaRP masked trainable coeff elems: {masked_trainable_coeff_elems:,} "
                    f"({masked_trainable_coeff_elems / masked_total_coeff_elems * 100:.2f}%)"
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                getattr(self.args, 'output_dir', '/lustre/gokms0509/Safety-WaRP-LLM/checkpoints'),
                f'phase3_non_freeze_{timestamp}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.logger.info("Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay (requested): {configured_weight_decay}")
            self.logger.info(f"  - Weight decay (effective): {effective_weight_decay}")
            self.logger.info("  - Warmup ratio: 0.1")
            self.logger.info(f"  - Batch size: {batch_size}")
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
            self.logger.info("  - Max gradient norm: 1.0")
            self.logger.info("  - Optimizer: adamw_torch")
            self.logger.info("  - LR scheduler: linear")
            self.logger.info(
                f"  - Input formatting: {'chat template' if self._is_instruct_model() else 'Question/Answer plain text'}"
            )
            self.logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
            self.logger.info("="*70)

            # W&B: 세션 시작 시 config 로깅
            try:
                import wandb as _wb3nf
                _wandb_enabled = _wb3nf.run is not None
            except Exception:
                _wandb_enabled = False

            if _wandb_enabled:
                try:
                    import wandb
                    wandb.log({
                        'phase3/trainable_params': trainable_params,
                        'phase3/total_params': total_params,
                        'phase3/trainable_ratio_pct': trainable_params / max(total_params, 1) * 100,
                        'phase3/masked_frozen_coeff_elems': masked_frozen_coeff_elems,
                        'phase3/masked_total_coeff_elems': masked_total_coeff_elems,
                        'phase3/dataset': phase3_dataset,
                        'phase3/learning_rate': learning_rate,
                        'phase3/epochs': epochs,
                        'phase3/batch_size': batch_size,
                        'phase3/grad_accum_steps': gradient_accumulation_steps,
                        'phase3/mode': 'non_freeze',
                    }, step=0)
                except Exception:
                    pass

            warmup_ratio = getattr(self.args, 'warmup_ratio', 0.1)
            lr_scheduler_type = getattr(self.args, 'lr_scheduler_type', 'cosine')
            max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
            logging_steps = getattr(self.args, 'logging_steps', 10)
            gradient_checkpointing = getattr(self.args, 'gradient_checkpointing', False)

            training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=effective_weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                max_grad_norm=max_grad_norm,
                logging_steps=logging_steps,
                save_strategy="no",
                eval_strategy="no",
                bf16=True if self.args.dtype == 'bfloat16' else False,
                report_to="wandb" if _wandb_enabled else "none",
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

            # 마스크 기반 gradient 동작 사전 점검 (1배치)
            self._run_mask_gradient_sanity_check()

            restore_callback = WaRPMaskRestoreCallback(self.model)
            self.logger.info(
                f"[WaRPMaskRestoreCallback] registered {len(restore_callback._specs)} modules "
                f"for post-step weight-decay restore"
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
                    return torch.utils.data.DataLoader(
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

            self.logger.info("✓ Trainer initialized")
            self.logger.info("Starting training (Non-Freeze mode)...")
            self.logger.info("  WaRP forward still applies masking:")
            self.logger.info("    W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U")
            self.logger.info("    mask=1: gradient 차단 (frozen)")
            self.logger.info("    mask=0: gradient 흐름 (trainable)")

            trainer.train()

            self.logger.info("✓ Training completed")

            # 마스크/학습 반영 여부 사후 점검 (샘플 인덱스 기반)
            self._log_warp_parameter_delta_summary()

            # 가중치 복원: basis_coeff → weight (W = basis_coeff @ U^T)
            restore_weight(self.model)

            # WaRP 모듈 → 표준 nn.Linear 변환 (버퍼/파라미터 제거 → 용량 정상화)
            restore_to_linear(self.model)

            # 최종 모델 저장 (표준 HuggingFace 구조)
            final_model_path = os.path.join(checkpoint_dir, 'final_model')
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)

            self.logger.info("="*70)
            self.logger.info("Phase 3 (Non-Freeze) Completed")
            self.logger.info(f"  - Final model: {final_model_path}")
            self.logger.info("="*70)

            metadata = {
                'phase': 3,
                'mode': 'non_freeze',
                'trainer': 'Trainer',
                'basis_dir': self.basis_dir,
                'masks_dir': self.masks_dir,
                'phase0_model': self.phase0_model_dir,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay_configured': configured_weight_decay,
                'weight_decay_effective': effective_weight_decay,
                'warmup_ratio': 0.1,
                'optimizer': 'adamw_torch',
                'lr_scheduler': 'cosine',
                'max_grad_norm': 1.0,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'total_samples': len(self.dataset),
                'trainable_params': trainable_params,
                'basis_coeff_params': basis_coeff_params,
                'frozen_params_requires_grad_false': total_params - trainable_params,
                'masked_frozen_coeff_elems': masked_frozen_coeff_elems,
                'masked_total_coeff_elems': masked_total_coeff_elems,
                'total_params': total_params,
                'timestamp': timestamp,
            }

            metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"✓ Metadata saved to: {metadata_path}")

            return final_model_path

        except ImportError:
            self.logger.error("TRL library not found! Install with: pip install trl")
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
