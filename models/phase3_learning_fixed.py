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
            
            # ✅ Gradient checkpointing 활성화 (메모리 절약)
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("✓ Gradient checkpointing enabled")
            
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
        """GSM8K 데이터 로드"""
        from datasets import load_dataset
        
        try:
            self.logger.info("Loading GSM8K dataset...")
            
            # GSM8K 데이터셋 로드
            dataset = load_dataset('openai/gsm8k', 'main', split='train')
            
            # 샘플 수 제한
            # gsm8k_samples=0 → 전체 사용
            # gsm8k_samples>0 → 해당 개수만 사용
            # default=0 → 전체 사용
            max_samples = getattr(self.args, 'gsm8k_samples', 0)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"✓ Limited to {max_samples} samples (out of {len(dataset)} total)")
                else:
                    self.logger.info(f"✓ Using all {len(dataset)} samples (max_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(dataset)} samples (gsm8k_samples=0 or not specified)")
            
            # 데이터셋 클래스
            class GSM8KDataset(torch.utils.data.Dataset):
                def __init__(self, data, tokenizer, max_length=256):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    question = item['question']
                    answer = item['answer']
                    text = f"Question: {question}\nAnswer: {answer}"
                    
                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                    }
            
            dataset_wrapper = GSM8KDataset(dataset, self.tokenizer)
            
            # Collate function
            def collate_fn(batch):
                max_len = max(len(item['input_ids']) for item in batch)
                
                input_ids_list = []
                attention_masks_list = []
                
                for item in batch:
                    input_ids = item['input_ids']
                    attention_mask = item['attention_mask']
                    
                    padding_length = max_len - len(input_ids)
                    if padding_length > 0:
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((padding_length,), self.tokenizer.pad_token_id)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length)
                        ])
                    
                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attention_mask)
                
                return {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }
            
            self.train_loader = DataLoader(
                dataset_wrapper,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            self.logger.info(f"✓ DataLoader created ({len(self.train_loader)} batches)")
            
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
        GSM8K로 Incremental Learning
        
        ✅ 원본 WaRP:
        - model.train() 모드
        - optimizer.step()으로 업데이트
        - 하지만 WaRP 모듈의 forward에서 마스크 적용
        - mask=1 부분은 detach()로 gradient 차단
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Phase 3: Incremental Learning with GSM8K")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'epochs', 20)
            learning_rate = getattr(self.args, 'utility_lr', 1e-5)
            
            # ✅ basis_coeff만 optimizer에 추가
            basis_params = []
            for module in self.model.modules():
                if isinstance(module, WaRPModule):
                    basis_params.append(module.basis_coeff)
            
            # ✅ 8-bit Optimizer (메모리 절약)
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    basis_params,
                    lr=learning_rate
                )
                self.logger.info("✓ Using 8-bit AdamW optimizer")
            except ImportError:
                self.logger.warning("bitsandbytes not available, using standard AdamW")
                optimizer = torch.optim.AdamW(basis_params, lr=learning_rate)
            
            # ✅ Gradient accumulation 설정
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 8)
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Trainable params: {sum(p.numel() for p in basis_params):,}")
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {self.args.batch_size * gradient_accumulation_steps}")
            
            # ✅ model.train() 모드
            self.model.train()
            
            # Checkpoint 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f'phase3_{timestamp}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 훈련 루프
            for epoch in range(epochs):
                epoch_loss = 0.0
                accumulation_counter = 0
                
                progress_bar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    total=len(self.train_loader)
                )
                
                for batch_idx, batch in enumerate(progress_bar):
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    
                    # Forward (WaRP 모듈이 자동으로 마스킹 적용)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    
                    # Loss
                    pred_logits = logits[:, :-1, :].contiguous()
                    target_ids = input_ids[:, 1:].contiguous()
                    attention_mask_shift = attention_mask[:, 1:].contiguous()
                    
                    valid_mask = (attention_mask_shift == 1) & (
                        target_ids != self.tokenizer.pad_token_id
                    )
                    pred_logits_flat = pred_logits[valid_mask]
                    target_ids_flat = target_ids[valid_mask]
                    
                    if len(target_ids_flat) > 0:
                        loss = nn.CrossEntropyLoss()(pred_logits_flat, target_ids_flat)
                        
                        # ✅ Gradient accumulation을 위한 loss scaling
                        loss = loss / gradient_accumulation_steps
                        
                        # Backward
                        loss.backward()
                        
                        accumulation_counter += 1
                        
                        # ✅ Gradient accumulation: N step마다만 optimizer.step()
                        if accumulation_counter % gradient_accumulation_steps == 0:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(basis_params, 1.0)
                            
                            # ✅ Optimizer step (mask=0 부분만 업데이트됨)
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        epoch_loss += loss.item()
                        
                        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 에포크 통계
                avg_loss = epoch_loss / len(self.train_loader)
                
                self.logger.info(f"Epoch {epoch+1}/{epochs}:")
                self.logger.info(f"  - Average loss: {avg_loss:.4f}")
                
                # Best 모델 저장
                if avg_loss < self.stats['best_loss']:
                    self.stats['best_loss'] = avg_loss
                    self.stats['best_epoch'] = epoch + 1
                    
                    # ✅ 가중치 복원 후 저장
                    restore_weight(self.model)
                    
                    best_model_path = os.path.join(checkpoint_dir, 'best_model')
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    
                    self.logger.info(f"  ✓ Best model saved (epoch {epoch+1})")
            
            # 최종 모델 저장
            restore_weight(self.model)
            
            final_model_path = os.path.join(checkpoint_dir, 'final_model')
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 3 Completed")
            self.logger.info(f"  - Best epoch: {self.stats['best_epoch']}")
            self.logger.info(f"  - Best loss: {self.stats['best_loss']:.4f}")
            self.logger.info(f"  - Final model: {final_model_path}")
            self.logger.info("="*70)
            
            return final_model_path
            
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
