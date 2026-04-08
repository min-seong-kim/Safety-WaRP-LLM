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
        description='Safety-WaRP-LLM'
    )
    
    # Phase 설정
    parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2, 3],
                        help='실행할 phase (0: Base Training, 1: Basis, 2: Importance, 3: Learning)')
    
    # 모델 설정
    parser.add_argument('--model_name', type=str, help='사용할 LLM 모델')
    
    # 데이터 설정
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='토큰 최대 길이 (Phase 2/3 데이터 전처리)')
    
    
    # Phase 1 설정
    parser.add_argument('--phase0_model_dir', type=str, default=None,
                        help='Phase 0에서 학습된 모델 경로 (Phase 1, 2, 3에서 사용)')
    parser.add_argument('--safety_dataset', type=str, default='circuit_breakers',
                        choices=['circuit_breakers', 'wikipedia'],
                        help='Basis 구성용 데이터셋 (Phase 1) - circuit_breakers(Safety), wikipedia(Utility)')
    parser.add_argument('--circuit_breakers_samples_phase1', type=int, default=4994,
                        help='Circuit breakers 샘플 수 (Phase 1 basis 구성용 - Safety Basis)')
    parser.add_argument('--wikipedia_samples_phase1', type=int, default=1000,
                        help='Wikipedia 샘플 수 (Phase 1 basis 구성용 - Utility Basis)')
    
    # Phase 2 설정
    parser.add_argument('--basis_dir', type=str, default=None,
                        help='Phase 1의 basis 디렉토리 경로 (Phase 2, 3에서 사용)')
    parser.add_argument('--dataset_phase2', type=str, default='circuit_breakers',
                        choices=['circuit_breakers', 'wikipedia'],
                        help='Phase 2 importance score 계산용 데이터셋 - circuit_breakers(Safety), wikipedia(Utility)')
    parser.add_argument('--circuit_breakers_path', type=str,
                        default='./data/circuit_breakers_train.json',
                        help='Circuit Breakers JSON 파일 경로 (Phase 2, 3에서 사용)')
    parser.add_argument('--circuit_breakers_samples_phase2', type=int, default=4994,
                        help='Circuit Breakers 샘플 수 (Phase 2 importance scoring용)')
    parser.add_argument('--wikipedia_samples_phase2', type=int, default=4994,
                        help='Wikipedia 샘플 수 (Phase 2 importance scoring용 - Utility)')
    parser.add_argument('--keep_ratio', type=float, default=0.1,
                        help='유지할 중요 파라미터 비율 (Phase 2)')
    parser.add_argument('--perlayer', action='store_true',
                        help='Phase 2에서 layer별 keep_ratio 적용')
    parser.add_argument('--no_rotation', action='store_true',
                        help='Phase 2/3에서 Phase 1 basis 없이 no-rotation(identity basis) 실험 수행')
    parser.add_argument('--original_space_mask', action='store_true',
                        help='Phase 2/3에서 basis/WaRP 없이 original weight space importance mask 사용')
    
    # Phase 3 설정
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Phase 2의 masks 디렉토리 경로 (Phase 3에서 사용)')
    parser.add_argument('--phase3_dataset', type=str, default='gsm8k',
                        choices=['gsm8k', 'safety', 'metamath', 'math'],
                        help='Phase 3 finetuning용 데이터셋 - gsm8k(Utility), safety(안전성 강화), metamath(고급 수학), math(Hendrycks MATH)')
    parser.add_argument('--gsm8k_samples', type=int, default=1000,
                        help='GSM8K 샘플 수 (Phase 3 - GSM8K 선택시만 사용)')
    parser.add_argument('--metamath_samples', type=int, default=0,
                        help='MetaMath 샘플 수 (Phase 3 - MetaMath 선택시만 사용, 0=전체)')
    parser.add_argument('--math_samples', type=int, default=0,
                        help='Hendrycks MATH 샘플 수 (Phase 3 - MATH 선택시만 사용, 0=전체)')
    parser.add_argument('--math_subjects', type=str, default='all',
                        help='Hendrycks MATH 과목 필터 (예: Algebra,Geometry 또는 all)')
    parser.add_argument('--math_levels', type=str, default='all',
                        help='Hendrycks MATH 난이도 필터 (예: 1,2,3 또는 all)')
    parser.add_argument('--math_dataset_source', type=str, default='official',
                        choices=['official', 'flat_competition_math'],
                        help='MATH 데이터 소스 (official=EleutherAI/hendrycks_math, flat=qwedsacf/competition_math)')
    parser.add_argument('--math_official_dataset_path', type=str, default='EleutherAI/hendrycks_math',
                        help='공식 Hendrycks MATH 데이터셋 경로')
    parser.add_argument('--math_flat_dataset_path', type=str, default='qwedsacf/competition_math',
                        help='flat competition_math 데이터셋 경로')
    parser.add_argument('--math_train_on_mixed_formats', action='store_true',
                        help='MATH 타겟을 long/short/minimal 포맷 혼합으로 구성')
    parser.add_argument('--math_use_chat_template', action='store_true',
                        help='MATH 데이터 전처리 시 tokenizer chat template 사용')
    parser.add_argument('--math_system_prompt', type=str,
                        default='You are a careful competition math solver. Solve the problem step by step. On the last line, write exactly one final answer in the form: Final Answer: $<answer>$. Do not use additional dollar signs earlier in the response.',
                        help='MATH chat template 사용 시 시스템 프롬프트')
    parser.add_argument('--circuit_breakers_samples_phase3', type=int, default=4994,
                        help='Circuit Breakers 샘플 수 (Phase 3 - Safety 선택시만 사용)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='훈련 에포크 (Phase 3)')
    parser.add_argument('--utility_lr', type=float, default=1e-5,
                        help='학습률 (Phase 3)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation 스텝 수 (Phase 3)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='LR warmup 비율 (Phase 3)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        help='LR scheduler 타입 (Phase 3)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping max norm (Phase 3)')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Trainer logging 주기 (Phase 3)')
    parser.add_argument('--base_weight_decay', type=float, default=0.01,
                        help='Weight decay (Phase 3)')
    parser.add_argument('--warp_monitor_samples_per_group', type=int, default=4,
                        help='WaRP monitor 샘플 수 (Phase 3 sanity check용)')
    parser.add_argument('--non_freeze', action='store_true',
                        help='Phase 3에서 WaRP 비적용 레이어를 포함해 나머지 파라미터도 학습')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Phase 3에서 gradient checkpointing 사용 (비교 실험 시 freeze/non-freeze 동일하게 설정 권장)')
    
    # 레이어 설정
    parser.add_argument('--target_layers', type=str, default='all',
                        help='타겟 레이어 (all, 0-5, 30-31 등)')
    parser.add_argument('--layer_type', type=str, 
                        default='attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_down,ffn_up',
                        help='처리할 layer types (쉼표로 구분)')
    
    # 계산 설정
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용 디바이스')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='모델 정밀도')
    
    # 저장 경로
    parser.add_argument('--output_dir', type=str, default='/lustre/gokms0509/Safety-WaRP-LLM/checkpoints',
                        help='출력 디렉토리')
    parser.add_argument('--log_dir', type=str, default='/lustre/gokms0509/Safety-WaRP-LLM/logs',
                        help='로그 디렉토리')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                        help='시드값')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드')
    
    
    return parser.parse_args()


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
    
    from models.phase1_basis import Phase1BasisBuilder
    
    builder = Phase1BasisBuilder(args, logger)
    
    # Phase 0 모델 로드
    args.model_name = args.phase0_model_dir
    builder.load_model()
    
    # 안전 데이터 로드 (circuit_breakers 또는 wikipedia)
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

    if (not args.no_rotation) and (not args.original_space_mask) and args.basis_dir is None:
        logger.error("Phase 2 requires --basis_dir")
        raise ValueError("Missing --basis_dir")

    if args.original_space_mask:
        from models.phase2_importance_original_space import Phase2ImportanceOriginalSpace
        scorer = Phase2ImportanceOriginalSpace(args, logger, args.basis_dir, args.phase0_model_dir)
    elif args.no_rotation:
        from models.phase2_importance_per_layer_no_rotation import Phase2ImportanceScorerPerLayerNoRotation
        scorer = Phase2ImportanceScorerPerLayerNoRotation(args, logger, args.basis_dir, args.phase0_model_dir)
    elif args.perlayer:
        from models.phase2_importance_per_layer import Phase2ImportanceScorerPerLayer
        scorer = Phase2ImportanceScorerPerLayer(args, logger, args.basis_dir, args.phase0_model_dir)
    else:
        from models.phase2_importance_whole import Phase2ImportanceScorer
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
    logger.info(f"Mode: {'non_freeze_warp' if args.non_freeze else 'freeze_warp'}")
    logger.info("="*70)
    
    # 이전 Phase 결과 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 3 requires --phase0_model_dir")
        raise ValueError("Missing --phase0_model_dir")
    
    if (not args.no_rotation) and (not args.original_space_mask) and args.basis_dir is None:
        logger.error("Phase 3 requires --basis_dir")
        raise ValueError("Missing --basis_dir")
    
    if args.masks_dir is None:
        logger.error("Phase 3 requires --masks_dir")
        raise ValueError("Missing --masks_dir")
    
    if args.original_space_mask:
        from models.phase3_extra_learning_original_space import Phase3OriginalSpaceMaskedLearner as Phase3Learner
    elif args.no_rotation:
        from models.phase3_extra_learning_no_rotation import Phase3IncrementalLearnerNoRotation as Phase3Learner
    elif args.non_freeze:
        from models.phase3_extra_learning_non_freeze import Phase3IncrementalLearnerNonFreeze as Phase3Learner
    else:
        from models.phase3_extra_learning import Phase3IncrementalLearner as Phase3Learner
    
    learner = Phase3Learner(
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
