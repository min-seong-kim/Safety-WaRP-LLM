"""
Phase 0: Base Safety Training

원본 FSCIL-WaRP의 base_train()과 동일한 역할
안전 데이터로 모델을 실제로 학습시켜 "안전 지식"을 모델에 각인

목표: 
- 안전 데이터(circuit_breakers)로 모델을 충분히 학습
- 학습된 가중치를 저장하여 이후 Phase에서 보호할 "base class 지식" 확립

절차:
1. 모델 로드
2. 안전 데이터 로드 (circuit_breakers)
3. 200 에포크 학습 (원본 WaRP와 유사)
4. 학습된 모델 저장
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)


class Phase0BaseTrainer:
    """
    Phase 0: Base Safety Training
    
    원본 WaRP의 base_train()과 동일한 개념
    - 안전 데이터로 모델을 충분히 학습
    - optimizer.step()으로 파라미터 실제 업데이트
    - 학습된 가중치가 이후 Phase에서 보호될 "base class 지식"
    """
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = None
        
        # 통계
        self.stats = {
            'total_loss': 0.0,
            'best_loss': float('inf'),
            'best_epoch': 0,
        }
    
    def load_model(self):
        """모델 및 토크나이저 로드"""
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
            
            # ✅ Gradient checkpointing 활성화 (메모리 절약)
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("✓ Gradient checkpointing enabled")
            
            self.logger.info(f"✓ Model loaded successfully")
            
            # 모델 정보
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"  - Total parameters: {total_params:,}")
            self.logger.info(f"  - Model dtype: {self.model.dtype}")
            
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
    
    def load_safety_data(self):
        """
        안전 데이터 로드 (circuit_breakers_train.json)
        
        원본 WaRP의 base class 데이터에 해당
        """
        try:
            circuit_breakers_path = self.args.circuit_breakers_path
            self.logger.info(f"Loading circuit_breakers data from {circuit_breakers_path}...")
            
            with open(circuit_breakers_path, 'r', encoding='utf-8') as f:
                circuit_breakers_data = json.load(f)
            
            # 샘플 수 제한
            if hasattr(self.args, 'circuit_breakers_samples') and self.args.circuit_breakers_samples > 0:
                circuit_breakers_data = circuit_breakers_data[:self.args.circuit_breakers_samples]
            
            self.logger.info(f"✓ Loaded {len(circuit_breakers_data)} circuit_breakers samples")
            
            # 데이터셋 클래스
            class CircuitBreakersDataset(torch.utils.data.Dataset):
                def __init__(self, data, tokenizer, max_length=256):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    
                    # Prompt + Response 결합
                    prompt = item.get('prompt', '')
                    response = item.get('response', '')
                    text = f"{prompt}\n{response}"
                    
                    # Tokenize
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
            
            dataset = CircuitBreakersDataset(
                circuit_breakers_data, 
                self.tokenizer, 
                max_length=256
            )
            
            # Collate function
            def collate_fn(batch):
                max_len = max(len(item['input_ids']) for item in batch)
                
                input_ids_list = []
                attention_masks_list = []
                
                for item in batch:
                    input_ids = item['input_ids']
                    attention_mask = item['attention_mask']
                    
                    # Padding
                    padding_length = max_len - len(input_ids)
                    if padding_length > 0:
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length, dtype=attention_mask.dtype)
                        ])
                    
                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attention_mask)
                
                return {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }
            
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            self.logger.info(f"✓ DataLoader created")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Total batches: {len(self.train_loader)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load safety data: {str(e)}", exc_info=True)
            raise
    
    def train(self):
        """
        Base safety training
        
        원본 FSCIL-WaRP의 base_train()과 동일:
        - model.train() 모드
        - optimizer.step()으로 파라미터 업데이트
        - 여러 에포크 반복
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Phase 0: Base Safety Training")
            self.logger.info("="*70)
            self.logger.info("원본 FSCIL-WaRP의 base_train()과 동일")
            self.logger.info("목표: 안전 데이터로 모델을 충분히 학습하여 'base class 지식' 확립")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'base_epochs', 100)
            learning_rate = getattr(self.args, 'base_lr', 1e-5)
            weight_decay = getattr(self.args, 'base_weight_decay', 0.01)
            
            # ✅ 8-bit Optimizer (메모리 절약)
            # AdamW state를 fp32 대신 8bit로 저장 (~75% 메모리 절약)
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                self.logger.info("✓ Using 8-bit AdamW optimizer")
            except ImportError:
                self.logger.warning("bitsandbytes not available, using standard AdamW")
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            
            # Scheduler (optional)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=epochs
            )
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay: {weight_decay}")
            self.logger.info(f"  - Optimizer: AdamW")
            self.logger.info(f"  - Scheduler: CosineAnnealingLR")
            
            # 모델을 훈련 모드로
            self.model.train()
            
            # 체크포인트 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = os.path.join(
                self.args.output_dir, 
                f'phase0_{timestamp}'
            )
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"  - Checkpoint directory: {self.checkpoint_dir}")
            
            # ✅ Gradient accumulation 설정
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {self.args.batch_size * gradient_accumulation_steps}")
            self.logger.info("="*70)
            
            # 훈련 루프
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_tokens = 0
                accumulation_counter = 0
                
                progress_bar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    total=len(self.train_loader)
                )
                
                for batch_idx, batch in enumerate(progress_bar):
                    # 데이터를 device로 이동
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    
                    # Forward
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    
                    # Teacher forcing loss
                    pred_logits = logits[:, :-1, :].contiguous()
                    target_ids = input_ids[:, 1:].contiguous()
                    attention_mask_shift = attention_mask[:, 1:].contiguous()
                    
                    # 유효한 토큰만
                    valid_mask = (attention_mask_shift == 1) & (
                        target_ids != self.tokenizer.pad_token_id
                    )
                    pred_logits_flat = pred_logits[valid_mask]
                    target_ids_flat = target_ids[valid_mask]
                    
                    if len(target_ids_flat) > 0:
                        # Loss 계산
                        loss = nn.CrossEntropyLoss()(pred_logits_flat, target_ids_flat)
                        
                        # ✅ Gradient accumulation을 위한 loss scaling
                        loss = loss / gradient_accumulation_steps
                        
                        # Backward
                        loss.backward()
                        
                        accumulation_counter += 1
                        
                        # ✅ Gradient accumulation: N step마다만 optimizer.step()
                        if accumulation_counter % gradient_accumulation_steps == 0:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            
                            # Optimizer step (파라미터 업데이트!)
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        # 통계
                        epoch_loss += loss.item()
                        epoch_tokens += len(target_ids_flat)
                        
                        # Progress bar 업데이트
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                        })
                
                # Scheduler step
                scheduler.step()
                
                # 에포크 통계
                avg_loss = epoch_loss / len(self.train_loader)
                
                self.logger.info(f"Epoch {epoch+1}/{epochs} completed:")
                self.logger.info(f"  - Average loss: {avg_loss:.4f}")
                self.logger.info(f"  - Total tokens: {epoch_tokens:,}")
                self.logger.info(f"  - Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                
                # Best 모델 저장
                if avg_loss < self.stats['best_loss']:
                    self.stats['best_loss'] = avg_loss
                    self.stats['best_epoch'] = epoch + 1
                    
                    best_model_path = os.path.join(
                        self.checkpoint_dir, 
                        'best_model'
                    )
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    
                    self.logger.info(f"  ✓ Best model saved (epoch {epoch+1}, loss {avg_loss:.4f})")
                
                # 정기 체크포인트 (매 20 에포크)
                if (epoch + 1) % 20 == 0:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f'checkpoint_epoch_{epoch+1}'
                    )
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    self.logger.info(f"  ✓ Checkpoint saved: epoch {epoch+1}")
            
            # 최종 모델 저장
            final_model_path = os.path.join(
                self.checkpoint_dir,
                'final_model'
            )
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 0: Base Safety Training Completed")
            self.logger.info(f"  - Best epoch: {self.stats['best_epoch']}")
            self.logger.info(f"  - Best loss: {self.stats['best_loss']:.4f}")
            self.logger.info(f"  - Final model saved to: {final_model_path}")
            self.logger.info("="*70)
            
            # 메타데이터 저장
            metadata = {
                'phase': 0,
                'model_name': self.args.model_name,
                'epochs': epochs,
                'best_epoch': self.stats['best_epoch'],
                'best_loss': self.stats['best_loss'],
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'timestamp': timestamp,
            }
            
            metadata_path = os.path.join(self.checkpoint_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✓ Metadata saved to: {metadata_path}")
            
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
