"""
Safety-WaRP-LLM: LoRA Training Script

LoRA를 사용한 효율적인 safety fine-tuning

Usage:
    python train_lora.py --phase 0 [options]
"""

import os
import argparse
import logging
from datetime import datetime

from utils import setup_logger, set_seed


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Safety-WaRP-LLM with LoRA'
    )
    
    # Phase 설정
    parser.add_argument('--phase', type=int, default=0,
                        help='실행할 phase (현재는 0만 지원)')
    
    # 모델 설정
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='사용할 LLM 모델')
    
    # 데이터 설정
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기')
    
    # Phase 0 설정
    parser.add_argument('--circuit_breakers_path', type=str, 
                        default='./data/circuit_breakers_train.json',
                        help='안전 데이터 경로 (Phase 0)')
    parser.add_argument('--circuit_breakers_samples', type=int, default=4994,
                        help='안전 데이터 샘플 수 (Phase 0)')
    parser.add_argument('--base_epochs', type=int, default=5,
                        help='기본 훈련 에포크 (Phase 0)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    
    # LoRA 설정
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (낮을수록 효율적, 높을수록 표현력 강함)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (일반적으로 rank의 2배)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (overfitting 방지)')
    parser.add_argument('--lora_lr', type=float, default=2e-4,
                        help='LoRA 학습률 (일반 fine-tuning보다 높음)')
    parser.add_argument('--lora_weight_decay', type=float, default=0.01,
                        help='LoRA weight decay')
    
    # 계산 설정
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용 디바이스')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='모델 정밀도')
    
    # 저장 경로
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='출력 디렉토리')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='로그 디렉토리')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                        help='시드값')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드')
    
    return parser.parse_args()


def run_phase0_lora(args, logger):
    """
    Phase 0: Base Safety Training with LoRA
    """
    logger.info("="*70)
    logger.info("Starting Phase 0: Base Safety Training with LoRA")
    logger.info("="*70)
    
    from models.phase0_base_training_LoRA import Phase0LoRATrainer
    
    trainer = Phase0LoRATrainer(args, logger)
    
    # 모델 로드 (LoRA 적용)
    trainer.load_model()
    
    # 안전 데이터 로드
    trainer.load_safety_data()
    
    # LoRA 훈련
    final_model_path = trainer.train()
    
    logger.info("="*70)
    logger.info(f"Phase 0 (LoRA) Completed!")
    logger.info(f"Merged model saved to: {final_model_path}")
    logger.info("="*70)
    logger.info("Next step:")
    logger.info(f"  python train_fixed.py --phase 1 --phase0_model_dir {final_model_path}")
    logger.info("="*70)


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 로거 설정
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"phase{args.phase}_lora_{timestamp}.log")
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger('safety_warp_lora', log_file=log_file, level=log_level)
    
    logger.info("="*70)
    logger.info("Safety-WaRP-LLM with LoRA")
    logger.info("="*70)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"LoRA dropout: {args.lora_dropout}")
    logger.info(f"Learning rate: {args.lora_lr}")
    logger.info("="*70)
    
    # Phase 0만 지원
    if args.phase == 0:
        run_phase0_lora(args, logger)
    else:
        raise ValueError(f"LoRA version currently supports only Phase 0. Use train_fixed.py for Phase 1-3.")
    
    logger.info("="*70)
    logger.info("Training completed successfully!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
