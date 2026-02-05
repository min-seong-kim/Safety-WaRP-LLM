"""
Safety-WaRP-LLM: 완전히 수정된 버전 (원본 FSCIL-WaRP 방식)

원본 FSCIL-WaRP와 동일한 플로우:
    Phase 0: Base Safety Training (새로 추가!)
    Phase 1: Basis Construction  
    Phase 2: Importance Scoring (eval 모드, optimizer.step 제거)
    Phase 3: Incremental Learning (WaRP 모듈, 마스크 적용)

주요 변경사항:
✅ Phase 0 추가: 안전 데이터로 실제 모델 학습
✅ Phase 1 수정: Φ @ Φ^T 방식으로 SVD
✅ Phase 2 완전 재작성: model.eval() + gradient만 계산
✅ Phase 3 수정: WaRP 모듈로 레이어 교체, 마스크 적용
"""

import os
import argparse
import logging
from datetime import datetime

from utils import setup_logger, set_seed


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Safety-WaRP-LLM (Fixed - 원본 FSCIL-WaRP 방식)'
    )
    
    # Phase 설정
    parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2, 3],
                        help='실행할 phase (0: Base Training, 1: Basis, 2: Importance, 3: Learning)')
    
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
    parser.add_argument('--circuit_breakers_samples', type=int, default=1000,
                        help='안전 데이터 샘플 수 (Phase 0)')
    parser.add_argument('--base_epochs', type=int, default=100,
                        help='기본 훈련 에포크 (Phase 0)')
    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='기본 훈련 학습률 (Phase 0)')
    parser.add_argument('--base_weight_decay', type=float, default=0.01,
                        help='기본 훈련 weight decay (Phase 0)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (메모리 절약)')
    
    # Phase 1 설정
    parser.add_argument('--phase0_model_dir', type=str, default=None,
                        help='Phase 0에서 학습된 모델 경로 (Phase 1, 2, 3에서 사용)')
    parser.add_argument('--safety_dataset', type=str, default='harmful_prompts',
                        choices=['harmful_prompts', 'do-not-answer', 'circuit_breakers'],
                        help='Basis 구성용 안전 데이터셋 (Phase 1)')
    parser.add_argument('--harmful_prompts_path', type=str, 
                        default='./data/harmful_prompts_200.txt',
                        help='Harmful prompts 파일 경로')
    parser.add_argument('--dna_samples', type=int, default=200,
                        help='Do-not-answer 샘플 수')
    parser.add_argument('--circuit_breakers_samples_phase1', type=int, default=200,
                        help='Circuit breakers 샘플 수 (Phase 1 basis 구성용)')
    
    # Phase 2 설정
    parser.add_argument('--basis_dir', type=str, default=None,
                        help='Phase 1의 basis 디렉토리 경로 (Phase 2, 3에서 사용)')
    parser.add_argument('--keep_ratio', type=float, default=0.1,
                        help='유지할 중요 파라미터 비율 (Phase 2)')
    
    # Phase 3 설정
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Phase 2의 masks 디렉토리 경로 (Phase 3에서 사용)')
    parser.add_argument('--gsm8k_samples', type=int, default=1000,
                        help='GSM8K 샘플 수 (Phase 3)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='훈련 에포크 (Phase 3)')
    parser.add_argument('--utility_lr', type=float, default=1e-5,
                        help='Utility 학습률 (Phase 3)')
    
    # 레이어 설정
    parser.add_argument('--target_layers', type=str, default='all',
                        help='타겟 레이어 (all, 0-5, 30-31 등)')
    parser.add_argument('--layer_type', type=str, 
                        default='ffn_down,ffn_up,attn_q,attn_k,attn_v',
                        help='처리할 layer types (쉼표로 구분)')
    
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
    
    # Phase 0 구현 방법 선택
    parser.add_argument('--use_sft', action='store_true',
                        help='Phase 0에서 TRL SFTTrainer 사용 (기본: manual loop)')
    
    return parser.parse_args()


def run_phase0(args, logger):
    """
    Phase 0: Base Safety Training
    
    원본 FSCIL-WaRP의 base_train()과 동일
    
    두 가지 구현 방법:
    1. Manual loop (기본): phase0_base_training.py
    2. TRL SFTTrainer: phase0_base_SFTtraining.py (--use_sft)
    """
    logger.info("="*70)
    logger.info("Starting Phase 0: Base Safety Training")
    if args.use_sft:
        logger.info("Implementation: TRL SFTTrainer")
    else:
        logger.info("Implementation: Manual training loop")
    logger.info("="*70)
    
    if args.use_sft:
        from models.phase0_base_SFTtraining import Phase0SFTTrainer
        trainer = Phase0SFTTrainer(args, logger)
    else:
        from models.phase0_base_training import Phase0BaseTrainer
        trainer = Phase0BaseTrainer(args, logger)
    
    # 모델 로드
    trainer.load_model()
    
    # 안전 데이터 로드
    trainer.load_safety_data()
    
    # 훈련
    final_model_path = trainer.train()
    
    logger.info("="*70)
    logger.info(f"Phase 0 Completed!")
    logger.info(f"Trained model saved to: {final_model_path}")
    logger.info("="*70)
    logger.info(f"Next step: Run Phase 1 with --phase0_model_dir {final_model_path}")
    logger.info("="*70)


def run_phase1(args, logger):
    """
    Phase 1: Basis Construction
    
    ✅ 수정: Φ @ Φ^T 방식으로 SVD
    """
    logger.info("="*70)
    logger.info("Starting Phase 1: Basis Construction")
    logger.info("="*70)
    
    # Phase 0 모델 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 1 requires --phase0_model_dir (trained model from Phase 0)")
        raise ValueError("Missing --phase0_model_dir")
    
    from models.phase1_basis import Phase1BasiBuilder
    
    builder = Phase1BasiBuilder(args, logger)
    
    # Phase 0 모델 로드
    # ⚠️ phase1_basis.py는 아직 phase0_model_dir를 지원하지 않으므로
    # 수동으로 model_name을 phase0_model_dir로 변경
    args.model_name = args.phase0_model_dir
    builder.load_model()
    
    # 안전 데이터 로드 (harmful_prompts 또는 do-not-answer)
    builder.load_safety_data()
    
    # ✅ Phase 1에서는 WaRP module 불필요!
    # 단순히 activation만 수집하면 되므로 원본 모델 그대로 사용
    
    # ✅ Incremental Gram matrix accumulation (hook 등록 + 누적)
    builder.collect_activations_and_accumulate_gram()
    
    # SVD 계산 (✅ 누적된 Gram matrix에서 직접 계산)
    builder.compute_svd()
    
    # Basis 저장
    builder.save_basis()
    
    logger.info("="*70)
    logger.info(f"Phase 1 Completed!")
    logger.info(f"Basis saved to: {builder.checkpoint_dir}")
    logger.info("="*70)
    logger.info(f"Next step: Run Phase 2 with --basis_dir {builder.checkpoint_dir}/basis")
    logger.info("="*70)


def run_phase2(args, logger):
    """
    Phase 2: Importance Scoring
    
    ✅ 완전 재작성: model.eval() + gradient만 계산
    """
    logger.info("="*70)
    logger.info("Starting Phase 2: Importance Scoring (Fixed)")
    logger.info("="*70)
    
    # Phase 0, 1 결과 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 2 requires --phase0_model_dir")
        raise ValueError("Missing --phase0_model_dir")
    
    if args.basis_dir is None:
        logger.error("Phase 2 requires --basis_dir")
        raise ValueError("Missing --basis_dir")
    
    from models.phase2_importance_fixed import Phase2ImportanceScorer
    
    scorer = Phase2ImportanceScorer(args, logger, args.basis_dir, args.phase0_model_dir)
    
    # Basis 로드
    scorer.load_basis()
    
    # Phase 0 모델 로드
    scorer.load_model()
    
    # WaRP 모듈로 변환
    scorer.convert_to_warp_modules()
    
    # 가중치 재매개변수화
    scorer.reparameterize_weights()
    
    # 안전 데이터 로드
    scorer.load_safety_data()
    
    # ✅ Importance 계산 (eval 모드, optimizer.step 없음!)
    scorer.compute_importance()
    
    # 마스크 생성
    scorer.generate_masks(keep_ratio=args.keep_ratio)
    
    # 마스크 저장
    masks_dir = scorer.save_masks()
    
    logger.info("="*70)
    logger.info(f"Phase 2 Completed!")
    logger.info(f"Masks saved to: {masks_dir}")
    logger.info("="*70)
    logger.info(f"Next step: Run Phase 3 with --masks_dir {masks_dir}")
    logger.info("="*70)


def run_phase3(args, logger):
    """
    Phase 3: Incremental Learning
    
    ✅ 수정: WaRP 모듈 사용, 마스크 적용
    """
    logger.info("="*70)
    logger.info("Starting Phase 3: Incremental Learning (Fixed)")
    logger.info("="*70)
    
    # 이전 Phase 결과 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 3 requires --phase0_model_dir")
        raise ValueError("Missing --phase0_model_dir")
    
    if args.basis_dir is None:
        logger.error("Phase 3 requires --basis_dir")
        raise ValueError("Missing --basis_dir")
    
    if args.masks_dir is None:
        logger.error("Phase 3 requires --masks_dir")
        raise ValueError("Missing --masks_dir")
    
    from models.phase3_learning_fixed import Phase3IncrementalLearner
    
    learner = Phase3IncrementalLearner(
        args, logger, args.basis_dir, args.masks_dir, args.phase0_model_dir
    )
    
    # Basis 로드
    learner.load_basis()
    
    # 마스크 로드
    learner.load_masks()
    
    # Phase 0 모델 로드 + WaRP 모듈 변환
    learner.load_model()
    
    # WaRP 모듈 설정 (basis, mask)
    learner.setup_warp_modules()
    
    # GSM8K 데이터 로드
    learner.load_utility_data()
    
    # ✅ 훈련 (WaRP 모듈이 자동으로 마스킹 적용)
    final_model_path = learner.train()
    
    logger.info("="*70)
    logger.info(f"Phase 3 Completed!")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info("="*70)


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 로거 설정
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"phase{args.phase}_{timestamp}.log")
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger('safety_warp', log_file=log_file, level=log_level)
    
    logger.info("="*70)
    logger.info("Safety-WaRP-LLM (Fixed - 원본 FSCIL-WaRP 방식)")
    logger.info("="*70)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layer types: {args.layer_type}")
    logger.info(f"Target layers: {args.target_layers}")
    logger.info("="*70)
    
    # Phase별 실행
    if args.phase == 0:
        run_phase0(args, logger)
    elif args.phase == 1:
        run_phase1(args, logger)
    elif args.phase == 2:
        run_phase2(args, logger)
    elif args.phase == 3:
        run_phase3(args, logger)
    else:
        raise ValueError(f"Invalid phase: {args.phase}")
    
    logger.info("="*70)
    logger.info("All tasks completed successfully!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
