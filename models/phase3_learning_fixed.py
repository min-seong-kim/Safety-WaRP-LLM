"""
Phase 3: Incremental Learning with Masked Updates (Fixed - 원본 WaRP 방식)

✅ 원본 FSCIL-WaRP의 incremental learning과 동일한 방식

핵심:
1. Phase 2에서 생성된 마스크를 WaRP 모듈에 설정
2. WaRP 모듈의 forward에서 자동으로 detach() 적용
3. mask=1: 동결 (gradient 차단), mask=0: 학습 가능
4. GSM8K 데이터로 fine-tuning

원본 WaRP의 forward:
```python
weight = UT_backward @ (basis_coeff * mask).clone().detach() + \
         basis_coeff * (1 - mask) @ UT_forward
```

이미 warp_modules.py에 구현되어 있음!
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from .warp_modules import WaRPModule, restore_weight

logger = logging.getLogger(__name__)


class Phase3IncrementalLearner:
    """
    Phase 3: Incremental Learning (원본 WaRP 방식)
    
    목표: 안전 메커니즘 보호하면서 GSM8K로 학습
    
    핵심:
    - WaRP 모듈의 forward에서 자동으로 마스킹 적용
    - mask=1: detach()로 gradient 차단
    - mask=0: 학습 가능
    """
    
    def __init__(self, args, logger, basis_dir, masks_dir, phase0_model_dir):
        self.args = args
        self.logger = logger
        self.basis_dir = basis_dir
        self.masks_dir = masks_dir
        self.phase0_model_dir = phase0_model_dir
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        
        # Basis 및 마스크
        self.basis_data = {}
        self.masks = {}
        self.layer_types = []
        
        # 통계
        self.stats = {
            'best_loss': float('inf'),
            'best_epoch': 0,
        }
    
    def load_basis(self):
        """Phase 1의 basis 로드"""
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")
            
            # 메타데이터
            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                basis_metadata = json.load(f)
            
            # Layer types
            layer_types_str = self.args.layer_type
            layer_types = [lt.strip() for lt in layer_types_str.split(',')]
            self.layer_types = layer_types
            
            # Basis 로드
            total_loaded = 0
            
            for layer_type in layer_types:
                layer_type_dir = os.path.join(self.basis_dir, layer_type)
                if not os.path.exists(layer_type_dir):
                    continue
                
                svd_files = sorted([
                    f for f in os.listdir(layer_type_dir)
                    if f.startswith('layer_') and f.endswith('_svd.pt')
                ])
                
                for svd_file in svd_files:
                    layer_idx = int(svd_file.split('_')[1])
                    svd_path = os.path.join(layer_type_dir, svd_file)
                    
                    svd_data = torch.load(svd_path)
                    key = (layer_idx, layer_type)
                    self.basis_data[key] = {
                        'U': svd_data['U'],
                        'S': svd_data['S'],
                    }
                    total_loaded += 1
            
            self.logger.info(f"✓ Basis loaded: {total_loaded} combinations")
            
        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
            raise
    
    def load_masks(self):
        """Phase 2의 마스크 로드"""
        try:
            self.logger.info(f"Loading masks from {self.masks_dir}...")
            
            # 메타데이터
            metadata_path = os.path.join(self.masks_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                masks_metadata = json.load(f)
            
            self.logger.info(f"✓ Mask metadata loaded:")
            self.logger.info(f"  - Keep ratio: {masks_metadata.get('keep_ratio')}")
            
            # Layer types
            layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]
            
            # 마스크 로드
            total_loaded = 0
            
            for layer_type in layer_types:
                layer_type_dir = os.path.join(self.masks_dir, layer_type)
                if not os.path.exists(layer_type_dir):
                    continue
                
                mask_files = sorted([
                    f for f in os.listdir(layer_type_dir)
                    if f.startswith('layer_') and f.endswith('_mask.pt')
                ])
                
                for mask_file in mask_files:
                    layer_idx = int(mask_file.split('_')[1])
                    mask_path = os.path.join(layer_type_dir, mask_file)
                    
                    # ✅ PyTorch 2.6+ compatibility: weights_only=False for numpy arrays
                    mask_data = torch.load(mask_path, weights_only=False)
                    key = (layer_idx, layer_type)
                    self.masks[key] = mask_data['mask']
                    total_loaded += 1
            
            self.logger.info(f"✓ Masks loaded: {total_loaded} combinations")
            
        except Exception as e:
            self.logger.error(f"Failed to load masks: {str(e)}", exc_info=True)
            raise
    
    def load_model(self):
        """Phase 0 모델 로드 및 WaRP 모듈로 변환"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .warp_modules import switch_to_warp_module
        
        try:
            self.logger.info(f"Loading Phase 0 model from {self.phase0_model_dir}...")
            
            # 데이터 타입
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.phase0_model_dir,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True
            )
            
            # ❌ Gradient checkpointing 비활성화 (WaRP + freeze 문제)
            # WaRP 모듈에서 일부만 학습할 때 gradient checkpointing이 충돌함
            # if hasattr(self.model, 'gradient_checkpointing_enable'):
            #     self.model.gradient_checkpointing_enable()
            #     self.logger.info("✓ Gradient checkpointing enabled")
            
            self.logger.info(f"✓ Model loaded")
            
            # WaRP 모듈로 변환
            self.logger.info("Converting to WaRP modules...")
            self.model = switch_to_warp_module(
                self.model,
                self.layer_types,
                self.args.target_layers
            )
            
            # 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.phase0_model_dir,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"✓ Tokenizer loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def load_utility_data(self):
        """GSM8K 데이터 로드 및 SFT 방식으로 변환 (prompt/labels 분리)"""
        from datasets import load_dataset, Dataset
        
        try:
            self.logger.info("Loading GSM8K dataset...")
            
            # GSM8K 데이터셋 로드
            dataset = load_dataset('openai/gsm8k', 'main', split='train')
            
            # 샘플 수 제한
            max_samples = getattr(self.args, 'gsm8k_samples', 0)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"✓ Limited to {max_samples} samples (out of {len(dataset)} total)")
                else:
                    self.logger.info(f"✓ Using all {len(dataset)} samples (max_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(dataset)} samples (gsm8k_samples=0 or not specified)")
            
            max_length = getattr(self.args, 'max_length', 512)
            # Always disable multiprocessing here to avoid CUDA fork issues.
            num_proc = None

            def build_chat_prompt(question: str) -> str:
                system_msg = (
                    "You are a helpful assistant that solves math problems step by step. "
                    "Always show your reasoning and provide the final numerical answer after ####."
                )
                user_msg = f"Solve this problem step by step:\n\n{question.strip()}"
                return f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"

            def tokenize_sft_example(prompt_text: str, answer_text: str) -> Dict[str, List[int]]:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]

                remain = max(1, max_length - len(prompt_ids))
                answer_ids = self.tokenizer(
                    answer_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=remain,
                )["input_ids"]

                if self.tokenizer.eos_token_id is not None and (
                    len(answer_ids) == 0 or answer_ids[-1] != self.tokenizer.eos_token_id
                ):
                    if len(prompt_ids) + len(answer_ids) < max_length:
                        answer_ids = answer_ids + [self.tokenizer.eos_token_id]

                input_ids = (prompt_ids + answer_ids)[:max_length]
                attention_mask = [1] * len(input_ids)
                labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

            def preprocess(ex):
                prompt = build_chat_prompt(ex["question"])
                answer = ex["answer"]
                return tokenize_sft_example(prompt, answer)

            self.dataset = dataset.map(
                preprocess,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc="Tokenizing train",
            )
            
            self.logger.info(f"✓ Dataset created ({len(self.dataset)} samples)")
            
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K data: {str(e)}", exc_info=True)
            raise
    
    def setup_warp_modules(self):
        """
        WaRP 모듈 설정: basis, mask 적용
        
        ✅ 원본 WaRP:
        - basis_coeff, UT_forward, UT_backward 설정
        - coeff_mask 설정
        - flag = True (WaRP 모드)
        
        Forward는 WaRP 모듈이 자동으로 처리:
        W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Setting up WaRP modules with basis and masks")
            self.logger.info("="*70)
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            setup_count = 0
            
            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)
                    
                    if key not in self.basis_data or key not in self.masks:
                        continue
                    
                    # 타겟 모듈 (WaRP 모듈)
                    target_module = self._get_target_module(layer, layer_type)
                    
                    if not isinstance(target_module, WaRPModule):
                        continue
                    
                    # 원본 가중치
                    W_original = target_module.weight.data.clone()
                    
                    # Basis (U에는 V = UT.t()가 저장되어 있음)
                    # 원본 WaRP: basis_coeff = W @ UT_forward.t() = W @ V
                    U_matrix = self.basis_data[key]['U']  # 실제로는 V (= UT.t())
                    U_matrix = U_matrix.to(dtype=W_original.dtype, device=W_original.device)
                    
                    # basis_coeff 초기화: W @ V (원본 WaRP 방식)
                    basis_coeff_init = W_original @ U_matrix
                    
                    # ✅ WaRP 모듈 설정
                    target_module.basis_coeff.data = basis_coeff_init
                    target_module.UT_forward = U_matrix.clone().detach()
                    target_module.UT_backward = torch.eye(
                        W_original.shape[0],
                        dtype=W_original.dtype,
                        device=W_original.device
                    )
                    
                    # ✅ 마스크 설정
                    mask = self.masks[key]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    mask = mask.to(dtype=W_original.dtype, device=W_original.device)
                    target_module.coeff_mask.data = mask
                    
                    # ✅ WaRP 모드 활성화
                    target_module.flag = True
                    
                    # basis_coeff를 학습 가능하게 설정
                    target_module.basis_coeff.requires_grad = True
                    
                    setup_count += 1
                    
                    frozen_count = mask.sum().item()
                    total_count = mask.numel()
                    frozen_ratio = frozen_count / total_count * 100
                    
                    self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                    self.logger.info(f"  ✓ basis_coeff: {basis_coeff_init.shape}")
                    self.logger.info(f"  ✓ mask: Frozen {frozen_count}/{total_count} ({frozen_ratio:.1f}%)")
            
            self.logger.info("="*70)
            self.logger.info(f"✓ Setup completed: {setup_count} WaRP modules")
            self.logger.info("="*70)
            self.logger.info("Forward에서 자동으로 마스킹 적용:")
            self.logger.info("  W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U")
            self.logger.info("  mask=1: gradient 차단 (동결)")
            self.logger.info("  mask=0: gradient 흐름 (학습 가능)")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Failed to setup WaRP modules: {str(e)}", exc_info=True)
            raise
    
    def train(self):
        """
        GSM8K로 Incremental Learning (Trainer 사용, SFT 방식)
        
        ✅ Phase 0와 동일한 방식:
        - SFTTrainer 사용
        - 안정적인 학습 루프
        - 자동 gradient accumulation
        
        ✅ WaRP 특화:
        - basis_coeff만 학습 가능
        - WaRP 모듈의 forward에서 마스크 적용
        - mask=1 부분은 detach()로 gradient 차단
        """
        try:
            from transformers import Trainer, TrainingArguments
            
            self.logger.info("="*70)
            self.logger.info("Phase 3: Incremental Learning with GSM8K (Trainer/SFT)")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'epochs', 3)
            learning_rate = getattr(self.args, 'utility_lr', 1e-5)
            weight_decay = getattr(self.args, 'base_weight_decay', 0.01)
            batch_size = self.args.batch_size
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)
            
            # ✅ basis_coeff만 학습 가능하게 설정
            trainable_params = 0
            total_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                
                # basis_coeff만 학습 가능
                if 'basis_coeff' in name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            
            self.logger.info(f"Parameter freeze status:")
            self.logger.info(f"  - Total params: {total_params:,}")
            self.logger.info(f"  - Trainable params: {trainable_params:,}")
            self.logger.info(f"  - Frozen params: {total_params - trainable_params:,}")
            self.logger.info(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")
            
            # Checkpoint 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f'phase3_{timestamp}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay: {weight_decay}")
            self.logger.info(f"  - Warmup ratio: 0.1")
            self.logger.info(f"  - Batch size: {batch_size}")
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
            self.logger.info(f"  - Max gradient norm: 1.0")
            self.logger.info(f"  - Optimizer: adamw_bnb_8bit")
            self.logger.info(f"  - LR scheduler: cosine")
            self.logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
            self.logger.info("="*70)
            
            warmup_ratio = getattr(self.args, 'warmup_ratio', 0.1)
            lr_scheduler_type = getattr(self.args, 'lr_scheduler_type', 'cosine')
            max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
            logging_steps = getattr(self.args, 'logging_steps', 10)

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
                optim="adamw_bnb_8bit",
                gradient_checkpointing=False,
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
            
            # ✅ model.train() 모드
            self.model.train()
            
            # Trainer 초기화
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            self.logger.info("✓ Trainer initialized")
            self.logger.info("Starting training...")
            self.logger.info("  WaRP forward will automatically apply masking:")
            self.logger.info("    W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U")
            self.logger.info("    mask=1: gradient 차단 (frozen)")
            self.logger.info("    mask=0: gradient 흐름 (trainable)")
            
            # 훈련 시작
            trainer.train()
            
            self.logger.info("✓ Training completed")
            
            # 가중치 복원 및 저장
            restore_weight(self.model)
            
            # 최종 모델 저장
            final_model_path = os.path.join(checkpoint_dir, 'final_model')
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 3 Completed")
            self.logger.info(f"  - Final model: {final_model_path}")
            self.logger.info("="*70)
            
            # 메타데이터 저장
            metadata = {
                'phase': 3,
                'trainer': 'Trainer',
                'basis_dir': self.basis_dir,
                'masks_dir': self.masks_dir,
                'phase0_model': self.phase0_model_dir,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'warmup_ratio': 0.1,
                'optimizer': 'adamw_bnb_8bit',
                'lr_scheduler': 'cosine',
                'max_grad_norm': 1.0,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'total_samples': len(self.dataset),
                'trainable_params': trainable_params,
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
    
    def _get_target_module(self, layer, layer_type):
        """Layer type에 맞는 모듈 반환"""
        if layer_type == 'ffn_down':
            return layer.mlp.down_proj
        elif layer_type == 'ffn_up':
            return layer.mlp.up_proj
        elif layer_type == 'attn_q':
            return layer.self_attn.q_proj
        elif layer_type == 'attn_k':
            return layer.self_attn.k_proj
        elif layer_type == 'attn_v':
            return layer.self_attn.v_proj
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def _parse_target_layers(self, num_layers):
        """타겟 레이어 파싱"""
        target = self.args.target_layers.strip()
        
        if target == 'all':
            return list(range(num_layers))
        elif '-' in target:
            start, end = map(int, target.split('-'))
            return list(range(start, end + 1))
        else:
            return [int(target)]
