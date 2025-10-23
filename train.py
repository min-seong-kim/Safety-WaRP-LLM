"""
Safety-WaRP-LLM: Phase 1 - Basis Construction
기본 목표: 안전 데이터로부터 FFN down_proj 활성화 기반 SVD 계산

사용 방법:
    python train.py \
        --phase 1 \
        --model_name meta-llama/Llama-3-8B \
        --safety_samples 100 \
        --batch_size 4 \
        --device cuda:0 \
        --seed 42
"""

import os
import sys
import argparse
import json
import torch
import logging
from datetime import datetime
from pathlib import Path

# 로컬 임포트
from utils import setup_logger, set_seed, ensure_dir, save_config, load_config, AverageTracker
from data.data_loader import create_safety_dataloader
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
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3-8B',
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
                        choices=['all', 'early', 'middle', 'late'],
                        help='타겟 레이어 범위 (all: 0-31, early: 0-10, middle: 11-21, late: 22-31)')
    parser.add_argument('--layer_type', type=str, default='ffn_down',
                        choices=['ffn_down', 'ffn_up', 'attn_q', 'attn_k', 'attn_v'],
                        help='타겟 레이어 타입')
    
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
        
        logger.info("\n" + "="*60)
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
    logger.info("\n[Step 1] Loading model...")
    builder = Phase1BasiBuilder(args, logger)
    builder.load_model()
    logger.info(f"✓ Model loaded: {args.model_name}")
    
    # Step 2: 안전 데이터 로드
    logger.info("\n[Step 2] Loading safety data (do-not-answer)...")
    builder.load_safety_data()
    logger.info(f"✓ Safety data loaded: batch_size={args.batch_size}")
    
    # Step 3: 활성화 수집 (hook 등록)
    logger.info("\n[Step 3] Registering forward hooks...")
    builder.register_activation_hooks()
    logger.info(f"✓ Hooks registered for layer type: {args.layer_type}")
    
    # Step 4-5: 활성화 수집 및 공분산 계산
    logger.info("\n[Step 4-5] Collecting activations and computing covariance...")
    builder.collect_activations()
    logger.info(f"✓ Activations collected from {len(builder.activations)} layers")
    
    # Step 6: SVD 분해
    logger.info("\n[Step 6] Computing SVD decomposition...")
    builder.compute_svd()
    logger.info(f"✓ SVD computed for all layers")
    
    # Step 7: Basis 저장
    logger.info("\n[Step 7] Saving basis and metadata...")
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
    (아직 구현 안함 - Phase 1 완료 후 진행)
    """
    logger.info("Phase 2 is not yet implemented.")
    logger.info("Please complete Phase 1 first.")


def run_phase3(args, logger):
    """
    Phase 3: Incremental Learning
    (아직 구현 안함 - Phase 1, 2 완료 후 진행)
    """
    logger.info("Phase 3 is not yet implemented.")
    logger.info("Please complete Phase 1 and 2 first.")


if __name__ == '__main__':
    main()
