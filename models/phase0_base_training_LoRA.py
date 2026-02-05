"""
Phase 0: Base Safety Training (LoRA version)

LoRA (Low-Rank Adaptation)를 사용한 효율적인 safety fine-tuning
- 전체 모델 대신 low-rank matrices만 학습
- 메모리 효율적이고 더 빠른 학습 가능
- 더 높은 learning rate 사용 가능

LoRA 하이퍼파라미터:
- rank (r): 16 (낮을수록 효율적, 높을수록 표현력 강함)
- alpha: 32 (일반적으로 rank의 2배)
- dropout: 0.05 (overfitting 방지)
- target_modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- learning_rate: 2e-4 (일반 fine-tuning의 1e-5보다 높음)
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


class Phase0LoRATrainer:
    """
    Phase 0: Base Safety Training with LoRA
    
    LoRA 장점:
    - 학습 파라미터 수 대폭 감소 (~0.1-1% of full model)
    - 메모리 효율적 (더 큰 batch size 가능)
    - 빠른 학습 속도
    - 높은 learning rate 사용 가능
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
        """모델 및 토크나이저 로드 + LoRA 적용"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        
        try:
            self.logger.info(f"Loading model: {self.args.model_name}")
            
            # 데이터 타입 설정
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
            
            # 기본 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True
            )
            
            self.logger.info(f"✓ Base model loaded")
            
            # 모델 정보
            total_params = sum(p.numel() for p in base_model.parameters())
            self.logger.info(f"  - Total parameters: {total_params:,}")
            
            # ✅ LoRA 설정
            lora_r = getattr(self.args, 'lora_r', 16)
            lora_alpha = getattr(self.args, 'lora_alpha', 32)
            lora_dropout = getattr(self.args, 'lora_dropout', 0.05)
            
            # LLaMA 3.2 모델의 target modules
            target_modules = [
                "q_proj",    # Attention query
                "k_proj",    # Attention key
                "v_proj",    # Attention value
                "o_proj",    # Attention output
                "gate_proj", # MLP gate
                "up_proj",   # MLP up
                "down_proj", # MLP down
            ]
            
            lora_config = LoraConfig(
                r=lora_r,                          # LoRA rank
                lora_alpha=lora_alpha,             # LoRA alpha (scaling)
                target_modules=target_modules,     # 적용할 모듈
                lora_dropout=lora_dropout,         # Dropout
                bias="none",                       # Bias 학습 안 함
                task_type=TaskType.CAUSAL_LM,      # Causal LM task
            )
            
            # LoRA 모델 생성
            self.model = get_peft_model(base_model, lora_config)
            
            # ✅ 훈련 모드 활성화 (중요!)
            self.model.train()
            
            # ✅ Gradient checkpointing 활성화 (메모리 절약)
            # 주의: gradient_checkpointing_enable()을 train() 이후에 호출
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
                self.logger.info("✓ Input gradients enabled")
            
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("✓ Gradient checkpointing enabled")
            
            # LoRA 파라미터 정보
            self.model.print_trainable_parameters()
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / all_params
            
            self.logger.info(f"✓ LoRA model created")
            self.logger.info(f"  - LoRA rank (r): {lora_r}")
            self.logger.info(f"  - LoRA alpha: {lora_alpha}")
            self.logger.info(f"  - LoRA dropout: {lora_dropout}")
            self.logger.info(f"  - Target modules: {target_modules}")
            self.logger.info(f"  - Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
            self.logger.info(f"  - All params: {all_params:,}")
            
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
                def __init__(self, data, tokenizer, max_length=512):
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
                max_length=512
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
        LoRA를 사용한 safety training
        
        LoRA 특징:
        - 높은 learning rate 사용 (2e-4)
        - 빠른 수렴
        - 메모리 효율적
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Phase 0: Base Safety Training with LoRA")
            self.logger.info("="*70)
            self.logger.info("LoRA: Low-Rank Adaptation for Efficient Fine-tuning")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'base_epochs', 5)
            learning_rate = getattr(self.args, 'lora_lr', 2e-4)  # LoRA는 더 높은 LR
            weight_decay = getattr(self.args, 'lora_weight_decay', 0.01)
            
            # Optimizer (LoRA 파라미터만 학습) - AdamW 8bit
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Scheduler - Cosine (warmup 없음)
            total_steps = len(self.train_loader) * epochs
            
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,  # No warmup
                num_training_steps=total_steps
            )
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate} (LoRA 최적화)")
            self.logger.info(f"  - Weight decay: {weight_decay}")
            self.logger.info(f"  - Optimizer: AdamW 8-bit (LoRA params only)")
            self.logger.info(f"  - Scheduler: Cosine (no warmup)")
            self.logger.info(f"  - Total steps: {total_steps}")
            
            # 모델을 훈련 모드로
            self.model.train()
            
            # 체크포인트 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = os.path.join(
                self.args.output_dir, 
                f'phase0_lora_{timestamp}'
            )
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"  - Checkpoint directory: {self.checkpoint_dir}")
            
            # Gradient accumulation 설정
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {self.args.batch_size * gradient_accumulation_steps}")
            self.logger.info("="*70)
            
            # ✅ 훈련 모드 재확인
            self.model.train()
            
            # 훈련 루프
            global_step = 0
            
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
                        
                        # Gradient accumulation을 위한 loss scaling
                        loss = loss / gradient_accumulation_steps
                        
                        # Backward
                        loss.backward()
                        
                        accumulation_counter += 1
                        
                        # Gradient accumulation: N step마다만 optimizer.step()
                        if accumulation_counter % gradient_accumulation_steps == 0:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(
                                filter(lambda p: p.requires_grad, self.model.parameters()), 
                                1.0
                            )
                            
                            # Optimizer step
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1
                        
                        # 통계
                        epoch_loss += loss.item()
                        epoch_tokens += len(target_ids_flat)
                        
                        # Progress bar 업데이트
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                        })
                
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
                        'best_lora_model'
                    )
                    # LoRA 어댑터만 저장
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    
                    self.logger.info(f"  ✓ Best LoRA model saved (epoch {epoch+1}, loss {avg_loss:.4f})")
            
            # 최종 LoRA 모델 저장
            final_lora_path = os.path.join(
                self.checkpoint_dir,
                'final_lora_model'
            )
            self.model.save_pretrained(final_lora_path)
            self.tokenizer.save_pretrained(final_lora_path)
            
            # ✅ Merged model 저장 (LoRA + base model)
            self.logger.info("Merging LoRA weights with base model...")
            merged_model = self.model.merge_and_unload()
            
            final_merged_path = os.path.join(
                self.checkpoint_dir,
                'final_merged_model'
            )
            merged_model.save_pretrained(final_merged_path)
            self.tokenizer.save_pretrained(final_merged_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 0: LoRA Training Completed")
            self.logger.info(f"  - Best epoch: {self.stats['best_epoch']}")
            self.logger.info(f"  - Best loss: {self.stats['best_loss']:.4f}")
            self.logger.info(f"  - LoRA adapters saved to: {final_lora_path}")
            self.logger.info(f"  - Merged model saved to: {final_merged_path}")
            self.logger.info("="*70)
            self.logger.info("Note: Use 'final_merged_model' for Phase 1-3")
            self.logger.info("="*70)
            
            # 메타데이터 저장
            metadata = {
                'phase': 0,
                'method': 'LoRA',
                'model_name': self.args.model_name,
                'lora_r': getattr(self.args, 'lora_r', 16),
                'lora_alpha': getattr(self.args, 'lora_alpha', 32),
                'lora_dropout': getattr(self.args, 'lora_dropout', 0.05),
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
            
            return final_merged_path
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
