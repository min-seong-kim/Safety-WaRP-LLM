#!/usr/bin/env python3
"""
Phase 3 모델 정리: WaRP 파라미터 제거

28GB → 6GB로 축소
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

def cleanup_warp_model(input_path, output_path):
    """WaRP 모듈을 일반 Linear로 변환"""
    
    print("="*70)
    print("Phase 3 모델 정리 (WaRP → Linear 변환)")
    print("="*70)
    print(f"입력: {input_path}")
    print(f"출력: {output_path}")
    print()
    
    # 1. 모델 로드
    print("📥 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.bfloat16,
        device_map='cpu',  # CPU에서 처리 (메모리 절약)
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(input_path)
    print("✓ 모델 로드 완료")
    print()
    
    # 2. WaRP 파라미터 통계
    print("📊 현재 모델 통계:")
    total_params = sum(p.numel() for p in model.parameters())
    warp_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'basis_coeff' in name or 'UT_forward' in name or 'UT_backward' in name
    )
    print(f"  - 총 파라미터: {total_params:,} ({total_params * 2 / 1e9:.2f} GB)")
    print(f"  - WaRP 파라미터: {warp_params:,} ({warp_params * 2 / 1e9:.2f} GB)")
    print()
    
    # 3. WaRP 모듈을 Linear로 변환
    print("🔄 WaRP 모듈 제거 중...")
    
    from models.warp_modules import WaRPModule
    
    converted_count = 0
    
    for name, module in list(model.named_modules()):
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if isinstance(module, WaRPModule):
            # Weight 복원
            UT_forward = module.UT_forward
            UT_backward = module.UT_backward
            
            # basis_coeff 또는 basis_coefficients 찾기
            if hasattr(module, 'basis_coeff'):
                basis_coeff = module.basis_coeff.data
            elif hasattr(module, 'basis_coefficients'):
                basis_coeff = module.basis_coefficients.data
            else:
                print(f"⚠️  {name}: basis_coeff를 찾을 수 없음")
                continue
            
            weight_restored = UT_backward.t() @ basis_coeff @ UT_forward.t()
            
            # Linear 모듈 생성
            linear = nn.Linear(
                in_features=module.weight.shape[1],
                out_features=module.weight.shape[0],
                bias=module.bias is not None,
                device='cpu',
                dtype=torch.bfloat16
            )
            
            linear.weight.data = weight_restored.data.to(dtype=torch.bfloat16)
            if module.bias is not None:
                linear.bias.data = module.bias.data.to(dtype=torch.bfloat16)
            
            # 교체
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, linear)
            else:
                setattr(model, child_name, linear)
            
            converted_count += 1
    
    print(f"✓ {converted_count}개 WaRP 모듈 변환 완료")
    print()
    
    # 4. 정리된 모델 통계
    print("📊 정리된 모델 통계:")
    final_params = sum(p.numel() for p in model.parameters())
    print(f"  - 총 파라미터: {final_params:,} ({final_params * 2 / 1e9:.2f} GB)")
    print(f"  - 감소량: {total_params - final_params:,} ({(total_params - final_params) * 2 / 1e9:.2f} GB)")
    print()
    
    # 5. 저장
    print(f"💾 모델 저장 중: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✓ 저장 완료")
    print()
    
    print("="*70)
    print("✅ 모델 정리 완료!")
    print(f"   입력: {total_params * 2 / 1e9:.2f} GB")
    print(f"   출력: {final_params * 2 / 1e9:.2f} GB")
    print(f"   절약: {(total_params - final_params) * 2 / 1e9:.2f} GB ({(1 - final_params/total_params)*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    input_path = "./checkpoints/phase3_non_freeze_20260302_201631/final_model"
    output_path = "./checkpoints/phase3_non_freeze_20260302_201631/final_model_cleaned"
    
    cleanup_warp_model(input_path, output_path)
