"""
WaRP Module Classes for LLM Safety Alignment

원본 FSCIL-WaRP의 WaRPModule을 LLM에 맞게 구현

핵심 메커니즘:
1. 가중치를 새로운 기저(basis)로 재매개변수화
2. 마스크를 사용하여 중요한 파라미터는 동결(detach), 나머지는 학습 가능
3. Forward: W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U

LLaMA 모델 구조:
- Attention projections (q_proj, k_proj, v_proj): (hidden_dim, hidden_dim)
- FFN down_proj: (intermediate_size, hidden_dim)  
- FFN up_proj: (hidden_dim, intermediate_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ensure_tensor(x):
    """numpy array를 tensor로 변환"""
    if not isinstance(x, torch.Tensor) and x is not None:
        return torch.from_numpy(x)
    return x


def same_device(x_mask, x):
    """마스크를 타겟 tensor와 같은 device로 이동"""
    if x.device != x_mask.device:
        return x_mask.to(x.device)
    return x_mask


class WaRPModule(nn.Module):
    """
    Base WaRP Module
    
    원본 FSCIL-WaRP의 WaRPModule과 동일한 구조
    
    Attributes:
        weight: 원본 가중치 (고정, 참조용)
        bias: 원본 bias
        basis_coeff: 새로운 기저에서의 계수 (학습 가능한 Parameter)
        UT_forward: V matrix (right singular vectors, Phase 1에서 설정됨)
        UT_backward: Identity matrix (출력 공간 변환, 현재는 사용 안 함)
        coeff_mask: 이진 마스크 (1=동결, 0=학습 가능)
        forward_covariance: 활성화 공분산 (SVD 계산용)
        flag: WaRP 모드 활성화 여부
    """
    
    def __init__(self, layer):
        super(WaRPModule, self).__init__()
        
        # 원본 가중치 및 bias를 buffer로 등록 (device와 dtype 유지)
        self.register_buffer("weight", layer.weight.data.clone())
        if layer.bias is not None:
            self.register_buffer("bias", layer.bias.data.clone())
        else:
            self.bias = None
        
        # 원본 weight의 device와 dtype 저장
        weight_device = self.weight.device
        weight_dtype = self.weight.dtype
        
        # basis_coeff: 새로운 기저에서의 계수 (학습 가능, device와 dtype 맞춤)
        self.basis_coeff = nn.Parameter(
            torch.zeros(self.weight.shape, dtype=weight_dtype, device=weight_device), 
            requires_grad=True
        )
        
        # Register buffers (학습되지 않는 고정 텐서, device와 dtype 맞춤)
        self.register_buffer("forward_covariance", None)
        self.register_buffer("basis_coefficients", torch.zeros(self.weight.shape, dtype=weight_dtype, device=weight_device))
        self.register_buffer("coeff_mask", torch.zeros(self.weight.shape, dtype=weight_dtype, device=weight_device))
        self.register_buffer("UT_forward", torch.empty(0, dtype=weight_dtype, device=weight_device))
        self.register_buffer("UT_backward", torch.empty(0, dtype=weight_dtype, device=weight_device))
        
        # WaRP 모드 플래그
        self.flag = True  # True: WaRP 모드, False: 정상 모드
        self.batch_count = 0


class LinearWaRP(WaRPModule):
    """
    Linear Layer용 WaRP Module
    
    LLaMA의 모든 projection layer에 사용 (Attention q/k/v, FFN up/down)
    
    원본 FSCIL-WaRP의 LinearWaRP와 동일하지만 LLM에 최적화
    """
    
    def __init__(self, linear_layer):
        super(LinearWaRP, self).__init__(linear_layer)
        assert isinstance(linear_layer, nn.Linear), "Layer must be a Linear layer"
        
        # Linear layer 속성 복사
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        self.batch_count = 0
    
    def pre_forward(self, input):
        """
        활성화 공분산 수집 (Phase 1에서 사용)
        
        원본 WaRP: input.t() @ input
        """
        with torch.no_grad():
            # input: (batch * seq_len, hidden_dim)
            forward_covariance = input.t() @ input
        return forward_covariance
    
    def post_backward(self):
        """
        공분산 누적 (이동 평균)
        
        원본 WaRP와 동일
        """
        with torch.no_grad():
            if self.forward_covariance is not None:
                self.forward_covariance = self.forward_curr_cov + (
                    self.batch_count / (self.batch_count + 1)
                ) * (self.forward_covariance - self.forward_curr_cov)
            else:
                self.forward_covariance = self.forward_curr_cov
            
            self.batch_count += 1
    
    def forward(self, input):
        """
        Forward pass
        
        원본 WaRP 방식:
        - flag=False: 정상 forward (활성화 수집은 hook에서)
        - flag=True: WaRP forward (마스크 적용)
        
        핵심: detach()로 동결된 부분의 gradient 차단
        """
        if not self.flag:
            # Phase 1: 정상 모드 (원본 weight 사용)
            # ✅ pre_forward() 제거: activation은 hook에서 수집
            output = F.linear(input, self.weight, self.bias)
        else:
            # Phase 2/3: WaRP 모드
            # W = (basis_coeff * mask).detach() @ V^T + basis_coeff * (1-mask) @ V^T
            # ✅ 수정: UT_forward.t() 추가 (V → V^T로 변환)
            coeff = (
                (self.basis_coeff * self.coeff_mask).clone().detach() + 
                self.basis_coeff * (1 - self.coeff_mask)
            )

            if self.UT_backward.numel() > 0:
                weight = self.UT_backward.t() @ coeff @ self.UT_forward.t()
            else:
                weight = coeff @ self.UT_forward.t()
            
            # ✅ Device 맞춤 (input과 같은 device로)
            weight = weight.to(input.device)
            
            output = F.linear(input, weight, self.bias)
        
        return output
    
    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'in_features={self.in_features}, '
        s += f'out_features={self.out_features}, '
        s += f'bias={self.bias is not None})'
        return s


def switch_to_warp_module(model, layer_types, target_layers='all'):
    """
    지정된 레이어들을 WaRP 모듈로 변환
    
    원본 WaRP의 switch_module()과 동일한 역할
    
    Args:
        model: LLaMA 모델
        layer_types: 변환할 layer type 리스트 
                    ['ffn_down', 'ffn_up', 'attn_q', 'attn_k', 'attn_v']
        target_layers: 타겟 레이어 인덱스 (str or list)
    
    Returns:
        변환된 모델
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 타겟 레이어 파싱
    num_layers = len(model.model.layers)
    if isinstance(target_layers, str):
        if target_layers == 'all':
            layer_indices = list(range(num_layers))
        elif '-' in target_layers:
            start, end = map(int, target_layers.split('-'))
            layer_indices = list(range(start, end + 1))
        else:
            layer_indices = [int(target_layers)]
    else:
        layer_indices = target_layers
    
    logger.info(f"Converting to WaRP modules...")
    logger.info(f"  - Target layers: {layer_indices}")
    logger.info(f"  - Layer types: {layer_types}")
    
    converted_count = 0
    
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        
        for layer_type in layer_types:
            # 타겟 모듈 선택
            if layer_type == 'ffn_down':
                original_module = layer.mlp.down_proj
                parent = layer.mlp
                attr_name = 'down_proj'
            elif layer_type == 'ffn_up':
                original_module = layer.mlp.up_proj
                parent = layer.mlp
                attr_name = 'up_proj'
            elif layer_type == 'attn_q':
                original_module = layer.self_attn.q_proj
                parent = layer.self_attn
                attr_name = 'q_proj'
            elif layer_type == 'attn_k':
                original_module = layer.self_attn.k_proj
                parent = layer.self_attn
                attr_name = 'k_proj'
            elif layer_type == 'attn_v':
                original_module = layer.self_attn.v_proj
                parent = layer.self_attn
                attr_name = 'v_proj'
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            # Linear → LinearWaRP 변환
            warp_module = LinearWaRP(original_module)
            
            # 모듈 교체
            setattr(parent, attr_name, warp_module)
            
            converted_count += 1
            logger.debug(f"  ✓ Layer {layer_idx} {layer_type}: {original_module.__class__.__name__} → LinearWaRP")
    
    logger.info(f"✓ Converted {converted_count} modules to WaRP")
    
    return model


def restore_weight(model):
    """
    WaRP 모듈의 basis_coeff를 원본 weight 공간으로 복원
    
    원본 WaRP의 restore_weight()와 동일
    
    W = basis_coeff @ V^T
    (V는 정규직교 → V^(-1) = V^T)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Restoring weights from basis coefficients...")
    
    restored_count = 0
    
    for module in model.modules():
        if isinstance(module, WaRPModule):
            UT_forward = module.UT_forward
            UT_backward = module.UT_backward
            
            # W = basis_coeff @ V^T
            # ✅ UT_forward.t() 추가: V → V^T로 변환
            # (UT_backward.t() = I이므로 생략 가능하지만 일관성 유지)
            if UT_backward.numel() > 0:
                weight_restored = UT_backward.t() @ module.basis_coeff.data @ UT_forward.t()
            else:
                weight_restored = module.basis_coeff.data @ UT_forward.t()
            
            # 원본 weight 업데이트
            module.weight.data = weight_restored.data
            
            restored_count += 1
    
    logger.info(f"✓ Restored {restored_count} weights")
    
    return model


# Warped modules 매핑 (원본 WaRP와 동일한 패턴)
warped_modules = {
    nn.Linear: LinearWaRP,
}
