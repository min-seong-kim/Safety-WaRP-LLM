"""
Safety-WaRP-LLM: Multi-Layer Support (Phase 1, 2, 3)

Support Layer Types:
    - ffn_down: MLP down projection (default)
    - ffn_up: MLP up projection
    - attn_q: Self-attention Q projection
    - attn_k: Self-attention K projection
    - attn_v: Self-attention V projection

Target Layers options:
    - all: all layers (0-31)
    - early: early layers (0-10)
    - middle: middle layers (11-21)
    - late: late layers (22+)
    - last: last layer only
    - range: e.g., 0-5, 30-31, 31
"""

import os
import argparse
import logging
from datetime import datetime

# 로컬 임포트
from utils import setup_logger, set_seed, ensure_dir, save_config
from models.phase1_basis import Phase1BasiBuilder


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Safety-WaRP-LLM: Phase 1 Basis Construction'
    )
    
    # Phase 설정
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                        help='실행할 phase (1: Basis, 2: Importance, 3: Learning)')
    
    # 모델 설정
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='사용할 LLM 모델 이름')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='저장된 모델 체크포인트 경로')
    
    # 데이터 설정
    parser.add_argument('--safety_samples', type=int, default=100,
                        help='안전 데이터 샘플 수 (do-not-answer)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기')
    
    # 레이어 설정
    parser.add_argument('--target_layers', type=str, default='all',
                        help='타겟 레이어 범위.')
    parser.add_argument('--layer_type', type=str, default='ffn_down',
                        help='''처리할 layer types (쉼표로 구분)''')

    
    # Phase 2 설정
    parser.add_argument('--basis_dir', type=str, default=None,
                        help='Phase 1에서 저장된 basis 디렉토리 경로')
    parser.add_argument('--keep_ratio', type=float, default=0.1,
                        help='유지할 중요 계수의 비율 (Phase 2)')
    
    # Phase 3 설정
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Phase 2에서 저장된 masks 디렉토리 경로')
    parser.add_argument('--utility_samples', type=int, default=1000,
                        help='유틸리티 데이터 샘플 수 (GSM8K train)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='훈련 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='학습률')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # 계산 설정
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용 디바이스 (cuda, cuda:0, cpu 등)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='모델 정밀도')
    
    # 저장 경로
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='결과 저장 디렉토리')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='로그 저장 디렉토리')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='실험 이름 (None이면 자동 생성)')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                        help='시드값 (-1: 난수)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드')
    
    # HuggingFace Hub 설정 (Phase 3)
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Phase 3 완료 후 HuggingFace Hub에 업로드')
    parser.add_argument('--hub_model_id', type=str, default='kmseong/WaRP-Safety-Llama3_8B_Instruct',
                        help='HuggingFace Hub 모델 ID (format: username/model_name)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API 토큰 (None이면 환경변수 HUGGINGFACE_TOKEN 사용)')
    parser.add_argument('--hub_private', action='store_true',
                        help='HuggingFace Hub에 비공개로 업로드')
    
    return parser.parse_args()


def setup_experiment_dirs(args):
    """실험 디렉토리 설정"""
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"phase{args.phase}_{timestamp}"
    
    args.exp_dir = os.path.join(args.output_dir, args.exp_name)
    args.checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
    args.log_file = os.path.join(args.log_dir, f"{args.exp_name}.log")
    
    ensure_dir(args.exp_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.log_dir)
    
    return args


def log_config(logger, args):
    """설정 정보 로깅"""
    logger.info("="*60)
    logger.info("Safety-WaRP-LLM Configuration")
    logger.info("="*60)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Safety Samples: {args.safety_samples}")
    logger.info(f"Target Layers: {args.target_layers}")
    logger.info(f"Layer Type: {args.layer_type}")
    
    # Phase 2 설정
    if args.phase >= 2:
        logger.info(f"Keep Ratio: {args.keep_ratio}")
    
    # Phase 3 설정
    if args.phase >= 3:
        logger.info(f"Utility Samples: {args.utility_samples}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info(f"Weight Decay: {args.weight_decay}")
        
        # HuggingFace Hub 설정
        if args.push_to_hub:
            logger.info(f"Push to Hub: {args.push_to_hub}")
            logger.info(f"Hub Model ID: {args.hub_model_id}")
            logger.info(f"Hub Private: {args.hub_private}")
    
    logger.info(f"Experiment Dir: {args.exp_dir}")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*60)


def main():
    """메인 함수"""
    # 1. 인자 파싱
    args = parse_args()
    
    # 2. 디렉토리 설정
    args = setup_experiment_dirs(args)
    
    # 3. 로거 설정
    logger = setup_logger(
        'Safety-WaRP-LLM',
        log_file=args.log_file,
        level=logging.DEBUG if args.debug else logging.INFO
    )
    
    # 4. 시드 설정
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # 5. 설정 로깅
    log_config(logger, args)
    
    # 6. 설정 저장
    config_path = os.path.join(args.exp_dir, 'config.json')
    config_dict = vars(args)
    save_config(config_dict, config_path)
    logger.info(f"✓ Config saved to {config_path}")
    
    # 7. Phase별 실행
    try:
        if args.phase == 1:
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: BASIS CONSTRUCTION")
            logger.info("="*60)
            run_phase1(args, logger)
            
        elif args.phase == 2:
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: IMPORTANCE SCORING")
            logger.info("="*60)
            run_phase2(args, logger)
            
        elif args.phase == 3:
            logger.info("\n" + "="*60)
            logger.info("PHASE 3: INCREMENTAL LEARNING")
            logger.info("="*60)
            run_phase3(args, logger)
        
        logger.info("="*60)
        logger.info(f"✓ Phase {args.phase} completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"✗ Error in Phase {args.phase}: {str(e)}", exc_info=True)
        raise


def run_phase1(args, logger):
    """
    Phase 1: Basis Construction
    
    절차:
    1. 모델 로드
    2. 안전 데이터 로드
    3. 각 레이어의 FFN down_proj에서 활성화 수집 (forward hook)
    4. 공분산 행렬 계산
    5. SVD 분해
    6. Basis 저장
    """
    logger.info("Starting Phase 1: Basis Construction")
    logger.info("-" * 60)
    
    # Step 1: 모델 로드
    logger.info("[Step 1] Loading model...")
    builder = Phase1BasiBuilder(args, logger)
    builder.load_model()
    logger.info(f"✓ Model loaded: {args.model_name}")
    
    # Step 2: 안전 데이터 로드
    logger.info("[Step 2] Loading safety data (do-not-answer)...")
    builder.load_safety_data()
    logger.info(f"✓ Safety data loaded: batch_size={args.batch_size}")
    
    # Step 3: 활성화 수집 (hook 등록)
    logger.info("[Step 3] Registering forward hooks...")
    builder.register_activation_hooks()
    logger.info(f"✓ Hooks registered for layer types: {args.layer_type}")
    
    # Step 4-5: 활성화 수집 및 공분산 계산
    logger.info("[Step 4-5] Collecting activations and computing covariance...")
    builder.collect_activations()
    logger.info(f"✓ Activations collected from {len(builder.activations)} layers")
    
    # Step 6: SVD 분해
    logger.info("[Step 6] Computing SVD decomposition...")
    builder.compute_svd()
    logger.info(f"✓ SVD computed for all layers")
    
    # Step 7: Basis 저장
    logger.info("[Step 7] Saving basis and metadata...")
    basis_path = builder.save_basis()
    logger.info(f"✓ Basis saved to {basis_path}")
    
    # 최종 리포트
    logger.info("\n" + "-" * 60)
    logger.info("Phase 1 Summary:")
    logger.info(f"  - Total layers processed: {len(builder.activations)}")
    logger.info(f"  - Safety samples processed: {args.safety_samples}")
    logger.info(f"  - Output directory: {args.exp_dir}")


def run_phase2(args, logger):
    """
    Phase 2: Importance Scoring
    
    절차:
    1. Phase 1에서 저장된 basis 로드
    2. 모델 가중치를 basis 공간으로 재매개변수화
    3. 안전 데이터로 모델 실행 (teacher forcing)
    4. 손실 계산 (token-level cross-entropy)
    5. 역전파로 gradient 계산
    6. 계수별 importance 점수 계산 (gradient magnitude)
    7. Quantile 기반 임계값으로 마스크 생성
    8. 마스크 저장
    """
    logger.info("Starting Phase 2: Importance Scoring")
    logger.info("-" * 60)
    
    # 필수 인자 확인
    if args.basis_dir is None:
        logger.error("Phase 2 requires --basis_dir argument")
        logger.error("Usage: python train.py --phase 2 --basis_dir /path/to/basis --layer_type ffn_down")
        raise ValueError("Missing --basis_dir argument")
    
    if not os.path.exists(args.basis_dir):
        logger.error(f"Basis directory not found: {args.basis_dir}")
        raise FileNotFoundError(f"Basis directory not found: {args.basis_dir}")
    
    from models.phase2_importance import Phase2ImportanceScorer
    
    # Phase 2는 여러 layer_type을 동시에 처리 가능
    logger.info(f"Phase 2: Processing layer_type={args.layer_type}")
    
    # Step 1: 모델 로드
    logger.info("[Step 1] Loading model...")
    scorer = Phase2ImportanceScorer(args, logger, args.basis_dir)
    scorer.load_model()
    logger.info(f"✓ Model loaded: {args.model_name}")

    # Step 2: Basis 로드 (여러 layer_type 동시 로드 가능)
    logger.info("[Step 2] Loading basis from Phase 1...")
    scorer.load_basis()
    logger.info(f"✓ Basis loaded: {len(scorer.basis_data)} (layer, type) combinations")

    # Step 3: 안전 데이터 로드
    logger.info("[Step 3] Loading safety data (do-not-answer)...")
    scorer.load_safety_data()
    logger.info(f"✓ Safety data loaded: batch_size={args.batch_size}")

    # Step 4: 가중치 재매개변수화 (모든 layer_type 동시 처리)
    logger.info("[Step 4] Reparameterizing weights to basis space...")
    scorer.reparameterize_weights()
    logger.info(f"✓ Weights reparameterized for all layer types")

    # Step 5: Importance 계산 (모든 layer_type 동시 처리)
    logger.info("[Step 5] Computing importance scores...")
    scorer.compute_importance()
    logger.info(f"✓ Importance scores computed for {len(scorer.importances)} (layer, type) combinations")

    # Step 6: 마스크 생성 (모든 layer_type)
    logger.info("[Step 6] Generating importance masks...")
    scorer.generate_masks(keep_ratio=args.keep_ratio)
    logger.info(f"✓ Masks generated with keep_ratio={args.keep_ratio}")

    # Step 7: 안전하게 fine-tuning된 모델 저장
    logger.info("[Step 7] Saving fine-tuned model...")
    finetuned_model_path = scorer.save_finetuned_model()
    logger.info(f"✓ Fine-tuned model saved to {finetuned_model_path}")

    # Step 8: 학습된 basis coefficients 저장
    logger.info("[Step 8] Saving basis coefficients...")
    coeffs_path = scorer.save_basis_coefficients()
    logger.info(f"✓ Basis coefficients saved to {coeffs_path}")

    # Step 9: 마스크 저장
    logger.info("[Step 9] Saving masks and metadata...")
    masks_path = scorer.save_masks()
    logger.info(f"✓ Masks saved to {masks_path}")
    
    # 최종 리포트
    logger.info("-" * 60)
    logger.info("Phase 2 Summary (Outputs):")
    logger.info(f"  1. Fine-tuned model: {finetuned_model_path}")
    logger.info(f"  2. Basis coefficients: {coeffs_path}")
    logger.info(f"  3. Importance masks: {masks_path}")
    logger.info("Phase 2 Statistics:")
    logger.info(f"  - Total layers processed: {len(scorer.masks)}")
    logger.info(f"  - Safety samples processed: {args.safety_samples}")
    logger.info(f"  - Keep ratio: {args.keep_ratio}")
    logger.info(f"  - Average loss: {scorer.stats['total_loss'] / len(scorer.dataloader):.4f}")
    logger.info(f"  - Output directory: {args.exp_dir}")


def run_phase3(args, logger):
    """
    Phase 3: Incremental Learning with Masked Gradient Updates
    
    Phase 1 basis + Phase 2 masks를 사용하여
    GSM8K 데이터로 미세조정하되, 안전 중요 방향은 보호
    
    ✅ 모든 layer_type을 동시에 처리 (e.g., --layer_type 'ffn_down,ffn_up,attn_q,attn_k,attn_v')
       단일 train loop에서 모든 10개 (layer_idx, layer_type) 조합을 동시 학습
    """
    from models.phase3_learning import Phase3IncrementalLearner
    from utils import upload_model_to_huggingface
    
    try:
        # 필수 인자 확인
        if args.basis_dir is None:
            logger.error("--basis_dir is required for Phase 3")
            raise ValueError("--basis_dir is required for Phase 3")
        
        if args.masks_dir is None:
            logger.error("--masks_dir is required for Phase 3")
            raise ValueError("--masks_dir is required for Phase 3")
        
        # 처리할 layer_types 파싱
        layer_types = [lt.strip() for lt in args.layer_type.split(',')]
        
        logger.info(f"{'='*60}")
        logger.info(f"Phase 3: Processing {len(layer_types)} layer type(s) SIMULTANEOUSLY")
        logger.info(f"{'='*60}")
        logger.info(f"Layer types to process: {layer_types}")
        logger.info(f"Architecture: Single train loop with all {len(layer_types)} types active")
        
        # 모든 layer_type을 한 번에 처리 (순차 루프 제거)
        logger.info(f"[Phase 3] Initializing learner for simultaneous multi-layer training...")
        
        # Phase 3 실행 (모든 layer_type 동시 처리)
        learner = Phase3IncrementalLearner(
            args=args,
            logger=logger,
            basis_dir=args.basis_dir,
            masks_dir=args.masks_dir
        )
        
        learner.train()
        
        logger.info(f"{'='*60}")
        logger.info(f"✓ Phase 3 completed for all {len(layer_types)} layer type(s) simultaneously!")
        logger.info(f"{'='*60}")
        
        # HuggingFace 업로드 (선택사항)
        if args.push_to_hub:
            logger.info("="*60)
            logger.info("Uploading final model to HuggingFace Hub...")
            logger.info("="*60)
            
            # 모델 저장 디렉토리 (transformers 형식)
            model_save_dir = os.path.join(args.exp_dir, 'checkpoints', 'final_model_all_layers')
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 모델 저장 (transformers 형식)
            logger.info(f"Saving final model to {model_save_dir}...")
            learner.model.save_pretrained(model_save_dir)
            learner.tokenizer.save_pretrained(model_save_dir)
            logger.info("✓ Final model saved")
            
            # HuggingFace에 업로드
            upload_success = upload_model_to_huggingface(
                model_path=model_save_dir,
                repo_id=args.hub_model_id,
                hf_token=args.hf_token,
                commit_message=f"WaRP Safety-Aligned Model - Phase 3 Complete (All {len(layer_types)} Layer Types - Simultaneous)",
                private=args.hub_private,
                logger=logger
            )
            
            if upload_success:
                logger.info(f"✓ Model successfully uploaded to https://huggingface.co/{args.hub_model_id}")
            else:
                logger.warning("Failed to upload model to HuggingFace Hub")
        
    except Exception as e:
        logger.error(f"✗ Error in Phase 3: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
