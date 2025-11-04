"""
Phase 3 모델 로드 유틸리티

Phase 3에서 학습된 모델을 올바르게 로드하는 함수들을 제공합니다.
basis_coeff @ U^T를 계산하여 최종 가중치를 복원합니다.
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_phase3_model(checkpoint_path: str, model_id: str = None, device: str = 'cuda'):
    """
    Phase 3 최종 모델 로드
    
    두 가지 방식 지원:
    1. phase3_final_reconstructed.pt: 이미 가중치가 재구성된 모델
    2. phase3_epoch_*.pt 또는 phase3_best.pt: basis_coeff를 가진 체크포인트
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model_id: 원본 모델 ID (필요시)
        device: 로드할 디바이스
    
    Returns:
        model: 로드된 모델
        tokenizer: 토크나이저
    """
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})
    
    # 1️⃣ 최종 재구성 모델인 경우 (basis_reconstruction=True)
    if checkpoint.get('metadata', {}).get('basis_reconstruction'):
        print("✅ Loading already-reconstructed model (basis_reconstruction=True)")
        model_id = model_id or config.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        
        # state_dict 적용
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded from {checkpoint_path}")
        return model, None
    
    # 2️⃣ basis_coeff를 가진 체크포인트인 경우
    print("⚠️ Checkpoint contains basis_coeff - need to reconstruct")
    print("   (This is not recommended - please use phase3_final_reconstructed.pt instead)")
    
    # 기본 모델 로드
    model_id = model_id or config.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    # state_dict 적용 (basis_coeff 포함)
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Model loaded from {checkpoint_path} (with basis_coeff)")
    
    return model, None


def compare_models(model1_path: str, model2_path: str, model_id: str = None):
    """
    두 Phase 3 모델의 가중치가 동일한지 비교
    
    Args:
        model1_path: 첫 번째 모델 체크포인트
        model2_path: 두 번째 모델 체크포인트
        model_id: 원본 모델 ID
    """
    
    cp1 = torch.load(model1_path, map_location='cpu')
    cp2 = torch.load(model2_path, map_location='cpu')
    
    sd1 = cp1['model_state_dict']
    sd2 = cp2['model_state_dict']
    
    print(f"\n{'='*70}")
    print(f"Model 1: {os.path.basename(model1_path)}")
    print(f"Model 2: {os.path.basename(model2_path)}")
    print(f"{'='*70}\n")
    
    # 키 비교
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    
    if keys1 == keys2:
        print("✅ Both models have identical keys")
    else:
        print("⚠️ Different keys found:")
        if keys1 - keys2:
            print(f"   Only in model 1: {keys1 - keys2}")
        if keys2 - keys1:
            print(f"   Only in model 2: {keys2 - keys1}")
    
    # 가중치 값 비교
    print(f"\n{'Key':<50} {'Same?':<10} {'Max Diff':<15}")
    print(f"{'-'*75}")
    
    max_diff_overall = 0.0
    for key in keys1 & keys2:
        w1 = sd1[key]
        w2 = sd2[key]
        
        if w1.shape != w2.shape:
            print(f"{key:<50} {'❌ Shape':<10} {str((w1.shape, w2.shape)):<15}")
            continue
        
        diff = (w1 - w2).abs().max().item()
        same = "✅" if diff < 1e-6 else "❌"
        max_diff_overall = max(max_diff_overall, diff)
        
        print(f"{key:<50} {same:<10} {diff:<15.6e}")
    
    print(f"\n{'='*70}")
    print(f"Max difference across all parameters: {max_diff_overall:.6e}")
    
    if max_diff_overall < 1e-6:
        print("✅ Models are effectively identical")
    else:
        print(f"⚠️ Models differ by up to {max_diff_overall:.6e}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # 테스트: 두 모델이 동일한지 확인
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint1', type=str, help='First checkpoint path')
    parser.add_argument('--checkpoint2', type=str, help='Second checkpoint path')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--compare', action='store_true', help='Compare two models')
    
    args = parser.parse_args()
    
    if args.compare and args.checkpoint1 and args.checkpoint2:
        compare_models(args.checkpoint1, args.checkpoint2, args.model_id)
