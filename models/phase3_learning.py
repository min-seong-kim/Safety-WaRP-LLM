"""
Phase 3: Incremental Learning with Masked Updates
안전 메커니즘을 보호하면서 유틸리티 개선
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import os
import json
from tqdm import tqdm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Phase3IncrementalLearner:
    """
    Phase 3: Incremental Learning with Masked Gradient Updates
    
    절차:
    1. Phase 1 basis + Phase 2 masks 로드
    2. 모델 가중치를 basis 공간으로 재매개변수화
    3. GSM8K 데이터로 미세조정 (마스킹된 gradient)
    4. 마스크된 방향 (안전 중요)은 업데이트 금지
    5. 덜 중요한 방향만 업데이트 가능
    """
    
    def __init__(self, args, logger, basis_dir, masks_dir):
        """
        Args:
            args: 커맨드라인 인자
            logger: 로거 객체
            basis_dir: Phase 1 basis 디렉토리
            masks_dir: Phase 2 masks 디렉토리
        """
        self.args = args
        self.logger = logger
        self.basis_dir = basis_dir
        self.masks_dir = masks_dir
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        
        # Basis 정보
        self.basis_data = {}  # layer_idx -> {'U': U, 'S': S, 'Vh': Vh}
        
        # 마스크
        self.masks = {}  # layer_idx -> binary mask
        
        # 훈련 통계
        self.stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'num_batches': 0,
        }
        
        # Hook 저장용
        self.hook_handles = []
    
    def load_basis(self):
        """Phase 1에서 저장된 basis 로드"""
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")
            
            # 메타데이터 로드
            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                basis_metadata = json.load(f)
            
            self.logger.info(f"✓ Metadata loaded:")
            self.logger.info(f"  - Target layers: {basis_metadata.get('target_layers')}")
            
            # basis 파일 로드
            import glob
            svd_files = sorted(glob.glob(os.path.join(self.basis_dir, 'layer_*_svd.pt')))
            
            for svd_path in svd_files:
                filename = os.path.basename(svd_path)
                layer_idx = int(filename.split('_')[1])
                
                svd_data = torch.load(svd_path, map_location='cpu')
                self.basis_data[layer_idx] = {
                    'U': svd_data['U'].to(self.args.device),
                    'S': svd_data['S'].to(self.args.device),
                    'Vh': svd_data['Vh'].to(self.args.device),
                }
            
            self.logger.info(f"✓ Basis loaded: {len(self.basis_data)} layers")
            self.logger.info(f"  - Layer indices: {sorted(self.basis_data.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
            raise
    
    def load_masks(self):
        """Phase 2에서 저장된 마스크 로드"""
        try:
            self.logger.info(f"Loading masks from {self.masks_dir}...")
            
            # 메타데이터 로드
            metadata_path = os.path.join(self.masks_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                masks_metadata = json.load(f)
            
            self.logger.info(f"✓ Mask metadata loaded:")
            self.logger.info(f"  - Keep ratio: {masks_metadata.get('keep_ratio')}")
            
            # 마스크 파일 로드
            import glob
            mask_files = sorted(glob.glob(os.path.join(self.masks_dir, 'layer_*_mask.pt')))
            
            for mask_path in mask_files:
                filename = os.path.basename(mask_path)
                layer_idx = int(filename.split('_')[1])
                
                mask = torch.load(mask_path, map_location='cpu')
                self.masks[layer_idx] = mask.to(self.args.device)
            
            self.logger.info(f"✓ Masks loaded: {len(self.masks)} layers")
            self.logger.info(f"  - Layer indices: {sorted(self.masks.keys())}")
            
            # 마스크 통계
            for layer_idx in sorted(self.masks.keys()):
                mask = self.masks[layer_idx]
                num_important = (mask == 1).sum().item()
                ratio = num_important / mask.numel()
                self.logger.info(f"  - Layer {layer_idx}: {num_important}/{mask.numel()} important ({ratio*100:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Failed to load masks: {str(e)}", exc_info=True)
            raise
    
    def load_model(self):
        """모델 로드"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            self.logger.info(f"Loading model: {self.args.model_name}")
            
            # 데이터 타입 설정
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True
            )
            
            self.logger.info(f"✓ Model loaded successfully")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"✓ Tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def load_utility_data(self):
        """GSM8K 훈련 데이터 로드"""
        from datasets import load_dataset
        
        try:
            self.logger.info(f"Loading GSM8K (main/train) with max_samples={self.args.utility_samples}...")
            
            dataset = load_dataset('openai/gsm8k', 'main', split='train')
            
            # 샘플 수 제한
            if self.args.utility_samples > 0:
                dataset = dataset.select(range(min(self.args.utility_samples, len(dataset))))
            
            self.logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # 데이터셋 준비 함수
            def format_gsm8k(examples):
                # question -> input, answer는 전체 풀이 과정
                formatted_inputs = []
                formatted_targets = []
                
                for question, answer in zip(examples['question'], examples['answer']):
                    formatted_inputs.append(f"Q: {question}\nA:")
                    formatted_targets.append(answer)
                
                return {
                    'input_text': formatted_inputs,
                    'target_text': formatted_targets,
                }
            
            dataset = dataset.map(format_gsm8k, batched=True, batch_size=100)
            
            # 데이터로더 생성
            class GSM8KDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, tokenizer, max_length=512):
                    self.dataset = dataset
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    sample = self.dataset[idx]
                    input_text = sample['input_text']
                    target_text = sample['target_text']
                    
                    # 결합: "Q: ...\nA: <answer>"
                    combined = f"{input_text}{target_text}"
                    
                    # 토크나이제이션
                    encoding = self.tokenizer(
                        combined,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                    }
            
            gsm8k_dataset = GSM8KDataset(dataset, self.tokenizer, max_length=512)
            
            # Custom collate function for variable length sequences
            def collate_fn_gsm8k(batch):
                """길이가 다른 시퀀스를 padding으로 처리"""
                max_len = max(len(item['input_ids']) for item in batch)
                
                input_ids_list = []
                attention_masks_list = []
                
                for item in batch:
                    input_ids = item['input_ids']
                    attn_mask = item['attention_mask']
                    
                    # Padding
                    pad_len = max_len - len(input_ids)
                    if pad_len > 0:
                        input_ids = torch.nn.functional.pad(
                            input_ids.unsqueeze(0),
                            (0, pad_len),
                            value=self.tokenizer.pad_token_id
                        ).squeeze(0)
                        attn_mask = torch.nn.functional.pad(
                            attn_mask.unsqueeze(0),
                            (0, pad_len),
                            value=0
                        ).squeeze(0)
                    
                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attn_mask)
                
                return {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }
            
            self.train_loader = DataLoader(
                gsm8k_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn_gsm8k
            )
            
            self.logger.info(f"✓ Utility dataloader created:")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Total batches: {len(self.train_loader)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load utility data: {str(e)}", exc_info=True)
            raise
    
    def register_mask_hooks(self):
        """
        Forward와 Backward hook 등록
        - Forward: weight를 basis_coeff @ U.T로 복원
        - Backward: 중요 방향의 gradient를 0으로 설정
        
        Mask shape: (d_out,) e.g., (4096,)
        Weight gradient shape: (d_out, d_in) e.g., (14336, 14336)
        → mask를 (d_out, 1)로 reshape하여 broadcast
        """
        def make_forward_hook(layer_idx, U):
            def forward_hook(module, input, output):
                """
                Forward pass에서 weight를 basis_coeff @ U.T로 복원
                basis_coeff = W_original @ U이므로
                W_reconstructed = basis_coeff @ U.T
                """
                if hasattr(module, 'basis_coeff'):
                    basis_coeff = module.basis_coeff  # (d_out, d_in)
                    # U.T를 이용해 weight 복원
                    weight_restored = basis_coeff @ U.T  # (d_out, d_in)
                    
                    # 복원된 weight로 output 재계산
                    x = input[0]
                    output_restored = torch.nn.functional.linear(x, weight_restored, module.bias)
                    return output_restored
                return output
            return forward_hook
        
        def make_backward_hook(layer_idx, mask):
            def backward_hook(grad):
                """
                grad shape: (d_out, d_in)
                mask shape: (d_out,)
                
                mask를 (d_out, 1)로 reshape하면 (d_out, d_in)으로 broadcast됨
                mask=1인 위치의 gradient를 0으로 설정 (중요 방향 보호)
                """
                # mask를 gradient와 같은 shape로 broadcast
                mask_broadcast = mask.view(-1, 1).to(grad.device).to(grad.dtype)  # (d_out, 1)
                
                # 마스크 적용: mask=1인 곳의 gradient를 0으로
                grad_masked = grad * (1 - mask_broadcast)
                return grad_masked
            return backward_hook
        
        for layer_idx in self.masks:
            layer = self.model.model.layers[layer_idx]
            target_module = layer.mlp.down_proj
            mask = self.masks[layer_idx]
            U = self.basis_data[layer_idx]['U']  # Phase 1에서 로드된 U
            
            # Forward hook 등록: weight 복원 (basis_coeff @ U.T)
            forward_hook = target_module.register_forward_hook(make_forward_hook(layer_idx, U))
            self.hook_handles.append(forward_hook)
            
            # Backward hook 등록: gradient masking
            backward_hook = target_module.weight.register_hook(make_backward_hook(layer_idx, mask))
            self.hook_handles.append(backward_hook)
            
            self.logger.debug(f"Forward and backward hooks registered for layer {layer_idx}")
        
        self.logger.info(f"✓ {len(self.hook_handles)} hooks registered (forward + backward)")
    
    def unregister_hooks(self):
        """Hook 제거"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.logger.info("✓ Hooks unregistered")
    
    def train_epoch(self, epoch: int, optimizer, lr_scheduler=None):
        """
        한 에포크 훈련
        
        Args:
            epoch: 에포크 번호
            optimizer: 옵티마이저
            lr_scheduler: 학습률 스케줄러
        """
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.args.epochs}",
            total=len(self.train_loader)
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 마스킹은 backward hook에서 자동 적용됨
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """
        전체 훈련 루프
        """
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("PHASE 3: INCREMENTAL LEARNING")
            self.logger.info("="*60 + "\n")
            
            # 1. 데이터 및 모델 로드
            self.logger.info("[Step 1] Loading basis and masks...")
            self.load_basis()
            self.load_masks()
            
            self.logger.info("\n[Step 2] Loading model...")
            self.load_model()
            
            self.logger.info("\n[Step 3] Loading utility data...")
            self.load_utility_data()
            
            # 2. 마스킹 hook 등록
            self.logger.info("\n[Step 4] Registering mask hooks...")
            self.register_mask_hooks()
            
            # 3. Optimizer 설정
            self.logger.info("\n[Step 5] Setting up optimizer...")
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
            
            # Learning rate scheduler (간단한 선형 감소)
            total_steps = len(self.train_loader) * self.args.epochs
            
            # Linear scheduler 구현
            def get_linear_lr(step, total_steps, warmup_steps=0.1):
                warmup_steps = int(total_steps * warmup_steps)
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: get_linear_lr(step, total_steps)
            )
            
            self.logger.info(f"✓ Optimizer configured:")
            self.logger.info(f"  - Learning rate: {self.args.learning_rate}")
            self.logger.info(f"  - Weight decay: {self.args.weight_decay}")
            self.logger.info(f"  - Total steps: {total_steps}")
            
            # 4. 훈련
            self.logger.info("\n[Step 6] Starting training...")
            
            best_loss = float('inf')
            
            for epoch in range(self.args.epochs):
                avg_loss = self.train_epoch(epoch, optimizer, lr_scheduler)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch, is_best=False)
            
            # 5. 정리
            self.logger.info("\n[Step 7] Finalizing...")
            self.unregister_hooks()
            
            self.logger.info("\n" + "="*60)
            self.logger.info("Phase 3 Summary:")
            self.logger.info(f"  - Total epochs: {self.args.epochs}")
            self.logger.info(f"  - Best loss: {best_loss:.4f}")
            self.logger.info(f"  - Output directory: {self.args.checkpoint_dir}")
            self.logger.info("="*60 + "\n")
            
            self.logger.info("✓ Phase 3 completed successfully!")
            
        except Exception as e:
            self.logger.error(f"✗ Error in Phase 3: {str(e)}", exc_info=True)
            self.unregister_hooks()
            raise
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        체크포인트 저장
        
        Args:
            epoch: 에포크 번호
            is_best: 최고 성능 모델인지 여부
        """
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.args),
        }
        
        # 일반 체크포인트
        save_path = os.path.join(checkpoint_dir, f'phase3_epoch_{epoch:03d}.pt')
        torch.save(checkpoint, save_path)
        
        # 최고 모델
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'phase3_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.debug(f"Saved best model checkpoint")
