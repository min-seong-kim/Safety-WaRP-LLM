"""
Phase 0: Base Safety Training with TRL SFTTrainer

TRL (Transformer Reinforcement Learning)의 SFTTrainer를 사용한 안전 학습
기존 phase0_base_training.py와 동일한 목표, 더 안정적인 구현

목표: 
- 안전 데이터(circuit_breakers)로 모델을 충분히 학습
- SFTTrainer로 간편하고 안정적인 fine-tuning
- 학습된 가중치를 저장하여 이후 Phase에서 보호할 "base class 지식" 확립

절차:
1. 모델 로드
2. 안전 데이터 로드 (circuit_breakers)
3. SFTTrainer로 학습
4. 학습된 모델 저장
"""

import os
import json
import torch
import logging
from datetime import datetime
from datasets import Dataset

logger = logging.getLogger(__name__)


class Phase0SFTTrainer:
    """
    Phase 0: Base Safety Training with SFTTrainer
    
    TRL의 SFTTrainer 사용:
    - 자동화된 학습 루프
    - 안정적인 gradient accumulation
    - HuggingFace Trainer 기반의 검증된 구현
    """
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        # 모델 및 토크나이저
        self.model = None
        self.tokenizer = None
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = None
        
        # 통계
        self.stats = {
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
        
        SFTTrainer용 Dataset 형식으로 변환
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
            
            # 데이터 토큰화
            # Manual tokenization for compatibility
            tokenized_data = []
            for item in circuit_breakers_data:
                prompt = item.get('prompt', '')
                response = item.get('response', '')
                full_text = f"{prompt}\n{response}"
                
                # Tokenize
                tokenized = self.tokenizer(
                    full_text,
                    max_length=512,
                    truncation=True,
                    padding=False,  # Will pad in collator
                )
                tokenized_data.append(tokenized)
            
            # HuggingFace Dataset으로 변환
            self.dataset = Dataset.from_list(tokenized_data)
            
            self.logger.info(f"✓ Dataset created ({len(self.dataset)} samples)")
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load safety data: {str(e)}", exc_info=True)
            raise
    
    def train(self):
        """
        SFTTrainer로 Base safety training
        
        기존 phase0_base_training.py와 동일한 하이퍼파라미터 사용
        """
        try:
            from trl import SFTTrainer, SFTConfig
            
            self.logger.info("="*70)
            self.logger.info("Phase 0: Base Safety Training with SFTTrainer")
            self.logger.info("="*70)
            self.logger.info("TRL SFTTrainer를 사용한 안전 학습")
            self.logger.info("목표: 안전 데이터로 모델을 충분히 학습하여 'base class 지식' 확립")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'base_epochs', 3)
            learning_rate = getattr(self.args, 'base_lr', 2e-5)
            weight_decay = getattr(self.args, 'base_weight_decay', 0.01)
            batch_size = self.args.batch_size
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)
            
            # 체크포인트 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = os.path.join(
                self.args.output_dir, 
                f'phase0_sft_{timestamp}'
            )
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay: {weight_decay}")
            self.logger.info(f"  - Batch size: {batch_size}")
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
            self.logger.info(f"  - Checkpoint directory: {self.checkpoint_dir}")
            self.logger.info("="*70)
            
            # SFTConfig 설정
            training_args = SFTConfig(
                # 출력 디렉토리
                output_dir=self.checkpoint_dir,
                
                # 학습 하이퍼파라미터 (기존과 동일)
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                
                # Optimizer & Scheduler
                optim="adamw_8bit",  # 8-bit optimizer (기존과 동일)
                lr_scheduler_type="cosine",  # Cosine scheduler (기존과 동일)
                warmup_ratio=0.0,  # No warmup (기존과 동일)
                
                # Gradient 설정
                max_grad_norm=1.0,  # Gradient clipping (기존과 동일)
                gradient_checkpointing=True,  # Memory saving (기존과 동일)
                
                # 데이터 타입
                bf16=True if self.args.dtype == 'bfloat16' else False,
                fp16=True if self.args.dtype == 'float16' else False,
                
                # 로깅 및 저장
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=3,
                
                # Evaluation (optional)
                # evaluation_strategy="no",
                
                # 기타
                remove_unused_columns=False,
                report_to="none",  # Disable wandb/tensorboard
                seed=getattr(self.args, 'seed', 42),
            )
            
            # Data collator for padding
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM
            )
            
            # SFTTrainer 초기화 (최소 파라미터만 사용)
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator,
            )
            
            self.logger.info("✓ SFTTrainer initialized")
            self.logger.info("Starting training...")
            
            # 훈련 시작
            trainer.train()
            
            self.logger.info("✓ Training completed")
            
            # 최종 모델 저장
            final_model_path = os.path.join(
                self.checkpoint_dir,
                'final_model'
            )
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Best 모델 찾기 (가장 낮은 loss의 checkpoint)
            # SFTTrainer는 자동으로 checkpoint를 저장하므로, 마지막 checkpoint를 best로 간주
            best_model_path = os.path.join(
                self.checkpoint_dir,
                'best_model'
            )
            trainer.save_model(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 0: Base Safety Training Completed")
            self.logger.info(f"  - Final model saved to: {final_model_path}")
            self.logger.info(f"  - Best model saved to: {best_model_path}")
            self.logger.info("="*70)
            
            # 메타데이터 저장
            metadata = {
                'phase': 0,
                'trainer': 'SFTTrainer',
                'model_name': self.args.model_name,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'total_samples': len(self.dataset),
                'timestamp': timestamp,
            }
            
            metadata_path = os.path.join(self.checkpoint_dir, 'metadata.json')
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
