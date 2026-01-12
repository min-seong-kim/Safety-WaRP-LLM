#!/usr/bin/env python3
"""
Phase 1 복구 스크립트
- 기존 checkpoint에서 SVD 결과 로드
- save_basis() 실행
- metadata 생성 및 저장
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
import logging

def setup_logger(log_path):
    """로거 설정"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger('Phase1-Resume')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def load_svd_from_checkpoint(checkpoint_dir, logger):
    """checkpoint에서 모든 SVD 결과 로드"""
    logger.info(f"Loading SVD results from checkpoint: {checkpoint_dir}")
    
    svd_results = {}
    basis_dir = os.path.join(checkpoint_dir, 'checkpoints', 'basis')
    
    if not os.path.exists(basis_dir):
        logger.error(f"Basis directory not found: {basis_dir}")
        return None
    
    layer_types = [d for d in os.listdir(basis_dir) if os.path.isdir(os.path.join(basis_dir, d))]
    logger.info(f"Found layer types: {layer_types}")
    
    total_loaded = 0
    for layer_type in sorted(layer_types):
        type_dir = os.path.join(basis_dir, layer_type)
        svd_files = sorted([f for f in os.listdir(type_dir) if f.endswith('.pt')])
        
        for svd_file in svd_files:
            svd_path = os.path.join(type_dir, svd_file)
            
            try:
                # layer_30_svd.pt → 30
                layer_idx = int(svd_file.replace('layer_', '').replace('_svd.pt', ''))
                
                # 로드
                svd_data = torch.load(svd_path, map_location='cpu')
                svd_results[(layer_idx, layer_type)] = svd_data
                total_loaded += 1
                
                logger.debug(f"Loaded: ({layer_idx}, {layer_type})")
            except Exception as e:
                logger.error(f"Failed to load {svd_path}: {str(e)}")
                return None
    
    logger.info(f"✓ Successfully loaded {total_loaded} SVD results")
    return svd_results

def save_basis(svd_results, checkpoint_dir, logger):
    """
    SVD 결과로부터 basis 생성 및 저장
    
    파일 구조:
    - basis/
      - attn_k/
        - layer_0_svd.pt
        - ...
      - metadata.json
    """
    try:
        basis_dir = os.path.join(checkpoint_dir, 'checkpoints', 'basis')
        logger.info(f"Saving basis to {basis_dir}...")
        
        # 메타데이터 생성
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_layers': len(svd_results),
            'layer_types': list(set(layer_type for _, layer_type in svd_results.keys())),
            'layer_indices': sorted(list(set(layer_idx for layer_idx, _ in svd_results.keys()))),
        }
        
        # 각 layer_type별 층수 통계
        for layer_type in metadata['layer_types']:
            count = len([1 for _, lt in svd_results.keys() if lt == layer_type])
            metadata[f'{layer_type}_count'] = count
        
        # 메타데이터 저장
        metadata_path = os.path.join(basis_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"✓ Basis saved successfully")
        logger.info(f"  - Directory: {basis_dir}")
        logger.info(f"  - Total layers: {metadata['total_layers']}")
        logger.info(f"  - Layer types: {metadata['layer_types']}")
        logger.info(f"  - Layer indices: {metadata['layer_indices'][0]}-{metadata['layer_indices'][-1]}")
        logger.info(f"  - Metadata: {metadata_path}")
        
        return basis_dir
        
    except Exception as e:
        logger.error(f"Failed to save basis: {str(e)}", exc_info=True)
        raise

def verify_svd_results(svd_results, logger):
    """SVD 결과 검증"""
    logger.info("Verifying SVD results...")
    
    if not svd_results:
        logger.error("No SVD results to verify")
        return False
    
    errors = []
    warnings = []
    
    for (layer_idx, layer_type), svd_data in svd_results.items():
        try:
            # 필수 키 확인
            required_keys = ['U', 'S', 'Vh', 'cov']
            for key in required_keys:
                if key not in svd_data:
                    errors.append(f"({layer_idx}, {layer_type}): Missing key '{key}'")
            
            # 데이터 타입 확인
            if svd_data.get('U') is not None:
                if not isinstance(svd_data['U'], torch.Tensor):
                    errors.append(f"({layer_idx}, {layer_type}): U is not a tensor")
            
            # 형태 확인
            if 'U' in svd_data and 'S' in svd_data:
                if svd_data['U'].shape[0] != len(svd_data['S']):
                    warnings.append(f"({layer_idx}, {layer_type}): Shape mismatch")
        
        except Exception as e:
            errors.append(f"({layer_idx}, {layer_type}): {str(e)}")
    
    if errors:
        logger.error(f"Found {len(errors)} errors:")
        for err in errors[:5]:
            logger.error(f"  - {err}")
        if len(errors) > 5:
            logger.error(f"  ... and {len(errors) - 5} more")
        return False
    
    if warnings:
        logger.warning(f"Found {len(warnings)} warnings (non-critical)")
    
    logger.info(f"✓ Verification passed: {len(svd_results)} SVD results are valid")
    return True

def main():
    parser = argparse.ArgumentParser(description='Phase 1 복구 스크립트')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Checkpoint 디렉토리 경로 (예: ./checkpoints/phase1_20251123_170709)'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='검증만 수행 (저장하지 않음)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='로그 디렉토리 (기본값: ./logs)'
    )
    
    args = parser.parse_args()
    
    # 로거 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f'resume_phase1_{timestamp}.log')
    logger = setup_logger(log_path)
    
    logger.info("="*60)
    logger.info("Phase 1 Resume Script")
    logger.info("="*60)
    logger.info(f"Checkpoint Dir: {args.checkpoint_dir}")
    logger.info(f"Verify Only: {args.verify_only}")
    logger.info(f"Log File: {log_path}")
    
    # checkpoint 존재 확인
    if not os.path.exists(args.checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        return False
    
    # SVD 결과 로드
    svd_results = load_svd_from_checkpoint(args.checkpoint_dir, logger)
    if svd_results is None:
        logger.error("Failed to load SVD results")
        return False
    
    # 검증
    if not verify_svd_results(svd_results, logger):
        logger.error("SVD results verification failed")
        return False
    
    # 검증만 수행하는 경우
    if args.verify_only:
        logger.info("✓ Verification completed (--verify_only flag set)")
        return True
    
    # basis 저장
    try:
        save_basis(svd_results, args.checkpoint_dir, logger)
        logger.info("\n" + "="*60)
        logger.info("✓ Phase 1 Recovery Completed Successfully!")
        logger.info("="*60)
        return True
    except Exception as e:
        logger.error(f"Failed to save basis: {str(e)}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
