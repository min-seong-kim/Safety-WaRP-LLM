"""
Phase 2: Importance Scoring
안전 데이터로부터 중요한 가중치 방향을 식별하고 마스크 생성
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Phase2ImportanceScorer:
    """
    Phase 2: Importance Scoring + Fine-tuning
    
    목표: 안전 데이터로 모델을 학습하면서 동시에 중요도 점수 계산
    
    절차:
    1. Phase 1에서 계산된 basis 로드
    2. 모델 가중치를 basis 공간으로 재매개변수화
       - W_original (고정) → basis_coeff (학습 가능)
       - 모든 연산은 basis_coeff를 통해 진행
    3. 여러 epoch 동안 안전 데이터로 반복:
       a. 모델 실행 (teacher forcing)
       b. 손실 계산: token-level cross-entropy
       c. 역전파: basis_coeff.grad 계산
       d. 옵티마이저: basis_coeff 업데이트
       e. importance 점수 누적: |∂L/∂basis_coeff|
    4. 모든 배치의 importance 평균 계산
    5. 임계값으로 마스크 생성 (상위 keep_ratio 유지)
    6. 마스크 저장
    
    핵심:
    - basis_coeff는 Parameter로 등록되어 학습됨
    - U_matrix는 고정되어 있음 (requires_grad=False)
    - Weight 복원: W_reconstructed = basis_coeff @ U^T (inference 시)
    """
    
    def __init__(self, args, logger, basis_dir):
        """
        Args:
            args: 커맨드라인 인자
            logger: 로거 객체
            basis_dir: Phase 1에서 저장된 basis 디렉토리 경로
        """
        self.args = args
        self.logger = logger
        self.basis_dir = basis_dir
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.dataloader = None
        
        # Basis 정보
        self.basis_data = {}  # (layer_idx, layer_type) -> {'U': U, 'S': S, 'Vh': Vh}
        self.basis_metadata = {}
        self.layer_types = []  # 처리할 layer_type 목록
        
        # Reparameterized 가중치
        self.original_weights = {}  # (layer_idx, layer_type) -> W_original
        self.basis_coeffs = {}  # (layer_idx, layer_type) -> basis_coeff (trainable)
        
        # Importance 점수
        self.importances = {}  # layer_idx -> importance 점수 배열
        self.masks = {}  # layer_idx -> 이진 마스크
        
        # 통계
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'total_loss': 0.0,
        }
    
    def load_basis(self):
        """
        Phase 1에서 저장된 basis 로드 (layer_type별 하위 디렉토리 지원)
        
        Phase 1에서 여러 layer_type을 처리한 경우:
        basis/
          ├── ffn_down/
          │   ├── layer_30_svd.pt
          │   └── layer_31_svd.pt
          └── ffn_up/
              ├── layer_30_svd.pt
              └── layer_31_svd.pt
        
        Log:
        - 로드된 파일 수
        - 각 레이어의 basis 형태
        - 메타데이터 정보
        """
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")
            
            # 메타데이터 로드
            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                self.basis_metadata = json.load(f)
            
            self.logger.info(f"✓ Metadata loaded:")
            self.logger.info(f"  - Model: {self.basis_metadata.get('model_name')}")
            self.logger.info(f"  - Layer types available: {self.basis_metadata.get('layer_types', ['unknown'])}")
            self.logger.info(f"  - Target layers: {self.basis_metadata.get('target_layers')}")
            
            # Phase 2는 여러 layer_type을 동시에 처리 가능
            layer_types_str = self.args.layer_type
            layer_types = [lt.strip() for lt in layer_types_str.split(',')]
            self.layer_types = layer_types
            self.logger.info(f"  - Processing layer types: {layer_types}")
            
            # 각 layer_type별로 basis 로드
            import glob
            total_loaded = 0
            
            for layer_type in layer_types:
                layer_type_dir = os.path.join(self.basis_dir, layer_type)
                
                if os.path.isdir(layer_type_dir):
                    # 새로운 구조: layer_type별 하위 디렉토리
                    self.logger.info(f"  - Layer type '{layer_type}': Using new structure")
                    search_dir = layer_type_dir
                else:
                    # 구형 구조: layer_type별 디렉토리 없음
                    self.logger.info(f"  - Layer type '{layer_type}': Using old structure")
                    search_dir = self.basis_dir
                
                # 해당 layer_type의 모든 layer_*_svd.pt 파일 찾기
                svd_files = sorted(glob.glob(os.path.join(search_dir, 'layer_*_svd.pt')))
                
                if not svd_files:
                    self.logger.warning(f"  No SVD files found in {search_dir} for layer_type={layer_type}")
                    continue
                
                # 각 SVD 파일 로드
                for svd_path in svd_files:
                    # 파일명에서 레이어 인덱스 추출: layer_31_svd.pt -> 31
                    filename = os.path.basename(svd_path)
                    layer_idx = int(filename.split('_')[1])
                    
                    svd_data = torch.load(svd_path, map_location='cpu')
                    key = (layer_idx, layer_type)
                    self.basis_data[key] = {
                        'U': svd_data['U'].to(self.args.device),
                        'S': svd_data['S'].to(self.args.device),
                        'Vh': svd_data['Vh'].to(self.args.device),
                    }
                    total_loaded += 1
            
            self.logger.info(f"✓ Basis loaded: {total_loaded} (layer, type) combinations")
            self.logger.info(f"  - Keys: {sorted(self.basis_data.keys())}")
            
            # 샘플 정보 출력
            if len(self.basis_data) > 0:
                sample_key = sorted(self.basis_data.keys())[0]
                sample_U = self.basis_data[sample_key]['U']
                self.logger.info(f"  - Sample {sample_key}: U shape = {sample_U.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
            raise
    
    def load_model(self):
        """
        모델 로드
        
        Log:
        - 모델 로드 상태
        - 모델 정보
        """
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
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"✓ Tokenizer loaded successfully")
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def load_safety_data(self):
        """
        안전 데이터 로드 (do-not-answer)
        
        Log:
        - 데이터셋 로드 상태
        - 배치 정보
        """
        from data.data_loader import create_safety_dataloader
        
        try:
            self.logger.info(f"Loading safety data with max_samples={self.args.safety_samples}")
            
            self.dataloader = create_safety_dataloader(
                batch_size=self.args.batch_size,
                max_samples=self.args.safety_samples,
                tokenizer=self.tokenizer,
                num_workers=0
            )
            
            self.logger.info(f"✓ Dataloader created")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Total batches: {len(self.dataloader)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load safety data: {str(e)}", exc_info=True)
            raise
    
    def reparameterize_weights(self):
        """
        모델 가중치를 basis 공간으로 재매개변수화
        
        핵심 개념:
        ================================================================================
        | Space       | Variable        | Shape        | Requires_grad | 역할         |
        |-------------+------------------+----------+----------+----------+----------+|
        | Weight      | W_original      | (d_out,  | False    | 분석용 reference  |
        | Basis       | basis_coeff     | (d_out,  | True     | 학습 가능 파라미터 |
        | Basis       | U_matrix        | (d_in,   | False    | 고정된 basis      |
        |             |                 |          |          |                 |
        | 관계식: W_reconstructed = basis_coeff @ U^T                
        ================================================================================
        
        단계:
        1. 원본 W 저장 (고정)
        2. basis_coeff 초기화 (W를 basis로 투영) → 학습 가능한 파라미터로 등록
        3. U_matrix 저장 (고정)
        4. Forward pass에서 weight를 basis_coeff @ U^T로 동적 복원
        
        Log:
        - 재매개변수화된 레이어 수
        - 각 레이어의 형태
        """
        try:
            self.logger.info("Reparameterizing weights to basis space...")
            self.logger.info("\n" + "="*70)
            self.logger.info("Weight Space → Basis Space Transformation (Multiple Layer Types)")
            self.logger.info("="*70)
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            # 모든 (layer_idx, layer_type) 조합에 대해 reparameterize
            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)
                    
                    if key not in self.basis_data:
                        self.logger.debug(f"Layer {layer_idx} ({layer_type}): No basis available, skipping")
                        continue
                    
                    # Select target module based on layer_type
                    if layer_type == 'ffn_down':
                        target_module = layer.mlp.down_proj
                    elif layer_type == 'ffn_up':
                        target_module = layer.mlp.up_proj
                    elif layer_type == 'attn_q':
                        target_module = layer.self_attn.q_proj
                    elif layer_type == 'attn_k':
                        target_module = layer.self_attn.k_proj
                    elif layer_type == 'attn_v':
                        target_module = layer.self_attn.v_proj
                    else:
                        raise ValueError(f"Unknown layer type: {layer_type}")
                    
                    # ✅ Step 1: 원본 가중치 저장 (분석용, 고정)
                    W_original = target_module.weight.data.clone()
                    self.original_weights[key] = W_original
                    
                    # ✅ Step 2: Basis 행렬 추출 및 dtype 변환
                    U = self.basis_data[key]['U']
                    model_dtype = W_original.dtype
                    U = U.to(dtype=model_dtype, device=W_original.device)
                    
                    # ✅ Step 3: basis_coeff 초기화
                    basis_coeff_init = W_original @ U
                    
                    # basis_coeff를 학습 가능한 Parameter로 등록
                    # 주의: 같은 module을 여러 layer_type에서 재사용하면 안되므로, 
                    # 추가 속성으로 저장 (나중에 참조용)
                    target_module.basis_coeff = nn.Parameter(basis_coeff_init.clone(), requires_grad=True)
                    target_module.U_matrix = U.clone().detach()
                    target_module.U_matrix.requires_grad = False
                    
                    self.basis_coeffs[key] = basis_coeff_init
                    
                    # 로깅
                    self.logger.info(f"\nLayer {layer_idx} ({layer_type}):")
                    self.logger.info(f"  ✓ W_original (고정):     {W_original.shape}")
                    self.logger.info(f"  ✓ basis_coeff (학습):    {basis_coeff_init.shape}")
                    self.logger.info(f"  ✓ U_matrix (고정):      {U.shape}")
                    self.logger.info(f"  ✓ Forward: W = basis_coeff @ U^T")
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✓ Reparameterization completed: {len(self.basis_coeffs)} (layer, type) combinations")

            self.logger.info(f"{'='*70}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to reparameterize weights: {str(e)}", exc_info=True)
            raise
    
    def compute_importance(self):
        """
        안전 데이터로 fine-tuning하면서 importance 점수 계산
        
        ✨ WaRP의 핵심:
        ==============================================================================
        | 단계 | 연산 공간 | 목적 | 수식 |
        |------|-----------|------|------|
        | Forward | 원본 가중치 | 모델 실행 | W = basis_coeff @ U^T |
        | Backward | 기저 공간 | Gradient 계산 | ∂L/∂basis_coeff (chain rule 자동) |
        | Importance | 기저 공간 | 중요도 점수 | importance = |∂L/∂basis_coeff| |
        | Update | 기저 공간 | 파라미터 갱신 | basis_coeff -= lr * ∂L/∂basis_coeff |
        ==============================================================================
        
        절차:
        1. 모델을 훈련 모드로 설정
        2. Optimizer 설정 (basis_coeff 파라미터만)
        3. Forward pass: weight = basis_coeff @ U^T로 동적 복원 (원본 가중치 공간)
        4. Loss 계산 및 역전파: ∂L/∂basis_coeff 계산 (기저 공간에서)
        5. Importance 수집: |∂L/∂basis_coeff| ← 파인튜닝 과정 중 수집!
        6. Optimizer.step(): basis_coeff 업데이트 (기저 공간에서)
        7. 에포크 완료 후 importance 평균 계산
        
        핵심:
        - Forward: 원본 가중치 공간 (W = basis_coeff @ U^T)
        - Backward: 기저 공간 (gradient는 basis_coeff에 대해)
        - Importance: 기저 공간에서 수집 (파인튜닝과 동시)
        - 이렇게 해야 WaRP가 제대로 동작함!
        
        Log:
        - 에포크별 손실
        - Batch별 gradient 통계
        - Forward/Backward 공간 검증
        - 최종 importance 점수
        """
        try:
            self.logger.info("Starting Phase 2: Fine-tuning + Importance Scoring...")
            self.logger.info("\n" + "="*70)
            self.logger.info("WaRP Phase 2: 안전 파인튜닝 + 중요도 점수 계산")
            self.logger.info("="*70)
            self.logger.info("Forward: 원본 가중치 공간 (W = basis_coeff @ U^T)")
            self.logger.info("Backward: 기저 공간 (∂L/∂basis_coeff 계산)")
            self.logger.info("Importance: 파인튜닝 과정 중 수집!")
            self.logger.info("="*70)
            self.logger.info("\n" + "="*70)
            self.logger.info("Training Setup")
            self.logger.info("="*70)
            
            # ✅ Step 0: 모든 parameter를 requires_grad=False로 설정
            # WaRP layer가 아닌 나머지는 gradient 계산 불필요
            self.logger.info("\n" + "="*70)
            self.logger.info("Step 0: Freeze 모든 parameter (WaRP layer 제외)")
            self.logger.info("="*70)
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.logger.info("✓ 모든 parameter를 requires_grad=False로 설정")
            self.logger.info("  → 목표: WaRP layer의 basis_coeff만 업데이트")
            self.logger.info("  → 효과: 불필요한 gradient 계산 제거, 메모리/시간 절약")
            
            # ✅ Step 1: 모델을 훈련 모드로 설정 (Dropout 등 활성화)
            self.model.train()
            self.logger.info("✓ Model set to training mode")
            
            # ✅ Step 2: Optimizer 설정 (basis_coeff 파라미터만) - Multiple Layer Types
            basis_params = []
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            # (layer_idx, layer_type) 튜플로 filter
            layers_with_basis = [key for key in self.basis_data.keys() 
                                if key[0] in target_indices]
            
            self.logger.debug(f"[Phase2] Target indices: {target_indices}")
            self.logger.debug(f"[Phase2] Layers in basis_data: {sorted(self.basis_data.keys())}")
            self.logger.debug(f"[Phase2] Layers with basis (intersection): {sorted(layers_with_basis)}")
            
            for layer_idx, layer_type in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = self._get_target_module(layer, layer_type)
                
                # basis_coeff 생성 또는 재사용
                if not hasattr(target_module, 'basis_coeff'):
                    basis_info = self.basis_data[(layer_idx, layer_type)]
                    U = basis_info['U']  # orthonormal basis
                    
                    # 올바른 초기화: W_original @ U
                    # 이렇게 하면:
                    #   W_reconstructed = basis_coeff @ U^T
                    #                  = (W_original @ U) @ U^T
                    #                  ≈ W_original (U는 orthonormal이므로)
                    W_original = target_module.weight.data.clone()
                    
                    # dtype 맞추기
                    U_dtype = U.to(dtype=W_original.dtype, device=W_original.device)
                    
                    # basis_coeff = W @ U (투영)
                    basis_coeff_init = W_original @ U_dtype
                    
                    basis_coeff = torch.nn.Parameter(basis_coeff_init.clone())
                    target_module.basis_coeff = basis_coeff
                    
                    # ✅ WaRP layer의 basis_coeff만 requires_grad=True로 설정
                    target_module.basis_coeff.requires_grad_(True)
                    
                    # ✅ 원본 weight는 requires_grad=False (이미 Step 0에서 처리됨)
                    target_module.weight.requires_grad_(False)
                    if target_module.bias is not None:
                        target_module.bias.requires_grad_(False)
                    
                    # U_matrix도 저장 (forward에서 사용) - requires_grad=False
                    target_module.U_matrix = U_dtype
                    
                    self.logger.debug(f"[Phase2] Layer {layer_idx} ({layer_type}): basis_coeff created (shape: {basis_coeff.shape})")
                
                # basis_coeff를 optimizer에 추가
                if hasattr(target_module, 'basis_coeff'):
                    # ✅ requires_grad=True 확인 (중요!)
                    if not target_module.basis_coeff.requires_grad:
                        target_module.basis_coeff.requires_grad_(True)
                    
                    basis_params.append(target_module.basis_coeff)
                    self.logger.debug(f"[Phase2] Layer {layer_idx} ({layer_type}): basis_coeff added to optimizer")
                    self.logger.debug(f"         - basis_coeff.requires_grad={target_module.basis_coeff.requires_grad}")
                    self.logger.debug(f"         - weight.requires_grad={target_module.weight.requires_grad}")
                    self.logger.debug(f"         - bias.requires_grad={target_module.bias.requires_grad if target_module.bias is not None else 'N/A'}")
            
            if len(basis_params) == 0:
                self.logger.error("No basis_coeff parameters found! Skipping importance computation.")
                self.logger.error(f"  - layers_with_basis: {layers_with_basis}")
                self.logger.error(f"  - basis_data keys: {sorted(self.basis_data.keys())}")
                return
            
            learning_rate = getattr(self.args, 'safety_lr', 1e-5)
            weight_decay = getattr(self.args, 'safety_weight_decay', 0.01)
            
            optimizer = torch.optim.AdamW(basis_params, lr=learning_rate, weight_decay=weight_decay)
            
            self.logger.info(f"\n✓ Optimizer 생성 완료: AdamW")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay: {weight_decay}")
            self.logger.info(f"  - 업데이트할 파라미터: {len(basis_params)} basis_coeff tensors (총 {sum(p.numel() for p in basis_params):,}개 원소)")
            self.logger.info(f"  - 업데이트할 레이어: {layers_with_basis}")
            
            # ✅ 검증: 현재 requires_grad 상태 확인
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"📊 파라미터 상태 검증")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"  - 총 파라미터: {total_params:,}")
            self.logger.info(f"  - 업데이트 대상 (requires_grad=True): {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
            self.logger.info(f"  - 동결됨 (requires_grad=False): {frozen_params:,} ({100*frozen_params/total_params:.3f}%)")
            self.logger.info(f"  ✅ WaRP layer의 basis_coeff만 업데이트됨")
            self.logger.info(f"  ✅ 나머지 모든 weight는 동결됨")
            self.logger.info(f"{'='*70}\n")
            
            # ✅ Step 3: Forward 메서드 교체 - autograd 호환 (hook 대신 사용)
            # forward hook 대신 actual forward method를 교체하여 gradient graph가 끊기지 않도록 함
            self.original_forwards = {}
            
            for layer_idx, layer_type in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = self._get_target_module(layer, layer_type)
                
                # 원본 forward 저장 (tuple key로 저장)
                self.original_forwards[(layer_idx, layer_type)] = target_module.forward
                
                # 새 forward 메서드 생성 (클로저로 basis_coeff와 U_matrix 캡처)
                def make_new_forward(module, orig_forward, layer_idx, layer_type):
                    def new_forward(x):
                        # basis_coeff @ U^T @ x^T
                        if hasattr(module, 'basis_coeff') and hasattr(module, 'U_matrix'):
                            basis_coeff = module.basis_coeff  # (d_out, rank)
                            U_matrix = module.U_matrix        # (d_in, rank)
                            weight_reconstructed = basis_coeff @ U_matrix.T  # (d_out, d_in)
                            # Linear forward: y = x @ W^T + bias
                            return torch.nn.functional.linear(x, weight_reconstructed, module.bias)
                        else:
                            # fallback to original
                            return orig_forward(x)
                    return new_forward
                
                target_module.forward = make_new_forward(target_module, self.original_forwards[(layer_idx, layer_type)], layer_idx, layer_type)
            
            self.logger.info(f"✓ Forward 메서드 {len(layers_with_basis)}개 (layer, type) 조합에서 교체됨 (autograd 호환)")
            
            # ✅ Step 4: Importance 저장소 초기화 - Tuple keys for (layer_idx, layer_type)
            importances = {key: [] for key in layers_with_basis}
            
            self.logger.info("\n" + "="*70)
            self.logger.info("Fine-tuning with Importance Tracking")

            self.logger.info("="*70)
            
            # ✅ Step 5: 훈련 루프
            epochs = getattr(self.args, 'safety_epochs', 3)
            total_loss = 0.0
            total_batches = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                progress_bar = tqdm(
                    self.dataloader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    total=len(self.dataloader)
                )
                
                for batch_idx, batch in enumerate(progress_bar):
                    harmful_prompts = batch['harmful_prompt']
                    safety_responses = batch['safe_response']
                    
                    # 결합된 입력-목표 시퀀스 (Teacher Forcing)
                    combined_texts = [
                        f"{q}\n{a}" 
                        for q, a in zip(harmful_prompts, safety_responses)
                    ]
                    
                    # 토큰화
                    combined = self.tokenizer(
                        combined_texts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    combined_ids = combined['input_ids'].to(self.model.device)
                    combined_attn = combined['attention_mask'].to(self.model.device)
                    
                    # ✅ Forward pass: weight = basis_coeff @ U^T
                    outputs = self.model(
                        input_ids=combined_ids,
                        attention_mask=combined_attn
                    )
                    logits = outputs.logits  # (batch, seq_len, vocab_size)
                    
                    # Teacher forcing: shift targets
                    pred_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab_size)
                    target_ids_shift = combined_ids[:, 1:].contiguous()
                    attention_mask_shift = combined_attn[:, 1:].contiguous()
                    
                    # 유효한 토큰만
                    valid_mask = (attention_mask_shift == 1) & (target_ids_shift != self.tokenizer.pad_token_id)
                    pred_logits_flat = pred_logits[valid_mask]
                    target_ids_flat = target_ids_shift[valid_mask]
                    
                    if len(target_ids_flat) > 0:
                        # ✅ Loss 계산
                        loss = nn.CrossEntropyLoss()(pred_logits_flat, target_ids_flat)
                        
                        # ✅ Backward: basis_coeff.grad 계산
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # ✅ Importance 수집: |basis_coeff.grad| - Multiple Layer Types
                        batch_importance_collected = 0
                        for layer_idx, layer_type in layers_with_basis:
                            layer = self.model.model.layers[layer_idx]
                            target_module = self._get_target_module(layer, layer_type)
                            
                            if hasattr(target_module, 'basis_coeff'):
                                if target_module.basis_coeff.grad is not None:
                                    # Gradient 절댓값 (element-wise)
                                    grad_abs = torch.abs(target_module.basis_coeff.grad)  # (d_out, rank)
                                    importances[(layer_idx, layer_type)].append(grad_abs.detach().cpu())
                                    batch_importance_collected += 1
                                else:
                                    self.logger.debug(f"[Batch {batch_idx}] Layer {layer_idx} ({layer_type}): gradient is None!")
                            else:
                                self.logger.debug(f"[Batch {batch_idx}] Layer {layer_idx} ({layer_type}): no basis_coeff!")
                        
                        # ✅ Update: basis_coeff 업데이트
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        epoch_batches += 1
                        total_loss += loss.item()
                        total_batches += 1
                        
                        progress_bar.set_postfix({'loss': loss.item()})
                    
                    progress_bar.update(1)
                
                epoch_loss_avg = epoch_loss / max(epoch_batches, 1)
                self.logger.info(f"\n[Epoch {epoch+1}/{epochs}] Average Loss: {epoch_loss_avg:.4f}")
            
            # ✅ 훈련 완료 후 forward 메서드는 복원하지 않음
            # 이유: basis_coeff는 이미 파인튜닝됨 (훈련 중 업데이트됨)
            #      forward를 복원할 필요 없음 (basis_coeff @ U^T를 계속 사용해야 함)
            #      대신 save_finetuned_model()에서 weight.data에 직접 저장
            self.logger.info(f"\n✅ 훈련 완료!")
            self.logger.info(f"   - basis_coeff: 훈련으로 업데이트됨")
            self.logger.info(f"   - Forward 메서드: new_forward 유지 (basis_coeff @ U^T 사용)")
            self.logger.info(f"   - 다음: importance 집계 및 마스크 생성")
            
            # ✅ Step 6: Importance 평균 계산 (파인튜닝 중 수집한 gradient 기반)
            self.logger.info("\n" + "="*70)
            self.logger.info("📊 Importance Scores 계산 (기저 공간에서 수집한 gradient 기반)")
            self.logger.info("="*70)
            self.logger.info("수식: importance_per_input = mean(|∂L/∂basis_coeff|) over batches")
            self.logger.info("의미: 안전 파인튜닝 중 각 기저 차원이 얼마나 중요했는가?")
            self.logger.info("="*70)
            
            self.importances = {}
            for layer_idx, layer_type in layers_with_basis:
                if len(importances[(layer_idx, layer_type)]) > 0:
                    # 모든 배치의 gradient를 스택
                    layer_importances = torch.stack(importances[(layer_idx, layer_type)], dim=0)  # (num_batches, d_out, rank)
                    
                    self.logger.info(f"\n✓ Layer {layer_idx} ({layer_type}):")
                    self.logger.info(f"  - 수집된 gradient 배치 수: {len(importances[(layer_idx, layer_type)])}")
                    self.logger.info(f"  - 각 배치 shape: (d_out, rank) = {layer_importances[0].shape}")
                    self.logger.info(f"  - 스택 후 shape: {layer_importances.shape}")
                    
                    # 배치 축 평균 (각 (out, in) 위치에서 배치들의 gradient 평균)
                    importance_mean = layer_importances.mean(dim=0)  # (d_out, rank)
                    self.logger.info(f"  - Batch 평균 shape: {importance_mean.shape}")
                    
                    # Input 차원별로 sum
                    # 각 input 차원이 모든 output에 미치는 누적 영향도 계산
                    importance_per_input = importance_mean.sum(dim=0)  # (rank,)
                    self.logger.info(f"  - Output 축 합산 → input-wise importance: {importance_per_input.shape}")
                    
                    self.importances[(layer_idx, layer_type)] = importance_per_input.float().cpu().numpy()
                    
                    # 상세 통계
                    self.logger.info(f"  📈 통계:")
                    self.logger.info(f"     - Mean: {self.importances[(layer_idx, layer_type)].mean():.6f}")
                    self.logger.info(f"     - Std: {self.importances[(layer_idx, layer_type)].std():.6f}")
                    self.logger.info(f"     - Min: {self.importances[(layer_idx, layer_type)].min():.6f}")
                    self.logger.info(f"     - Max: {self.importances[(layer_idx, layer_type)].max():.6f}")
                    self.logger.info(f"     - Median: {np.median(self.importances[(layer_idx, layer_type)]):.6f}")
                    
                    # 상위 10% 값 확인
                    top_10_pct = np.percentile(self.importances[(layer_idx, layer_type)], 90)
                    self.logger.info(f"     - 90 percentile (상위 10% 기준): {top_10_pct:.6f}")
                    
                else:
                    self.logger.error(f"✗ Layer {layer_idx} ({layer_type}): No gradients collected! importances[({layer_idx}, {layer_type})] = {importances.get((layer_idx, layer_type), [])}")
            
            avg_loss = total_loss / max(total_batches, 1)
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ Phase 2 완료: Fine-tuning + Importance Scoring")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"📊 훈련 결과:")
            self.logger.info(f"   - Total loss (all epochs): {total_loss:.4f}")
            self.logger.info(f"   - Average loss per batch: {avg_loss:.4f}")
            self.logger.info(f"   - Total batches processed: {total_batches}")
            self.logger.info(f"   - Layers with importance scores: {len(self.importances)}")
            self.logger.info(f"   - Layers with basis: {len(layers_with_basis)}")
            self.logger.info(f"\n🔑 WaRP 검증:")
            self.logger.info(f"   ✓ Forward: 원본 가중치 공간 (W = basis_coeff @ U^T)")
            self.logger.info(f"   ✓ Backward: 기저 공간 (∂L/∂basis_coeff)")
            self.logger.info(f"   ✓ Importance: 파인튜닝 중 수집됨")
            self.logger.info(f"   ✓ basis_coeff: 훈련으로 업데이트됨")
            self.logger.info(f"\n📝 다음 단계:")
            self.logger.info(f"   1. 마스크 생성 (상위 10% 기저 차원 선별)")
            self.logger.info(f"   2. 안전 정렬 모델 저장")
            self.logger.info(f"   3. Phase 3: GSM8K로 masked fine-tuning")
            self.logger.info(f"{'='*70}\n")
            
            self.stats['total_loss'] = total_loss
            
        except Exception as e:
            self.logger.error(f"Failed to compute importance: {str(e)}", exc_info=True)
            raise
    
    def save_finetuned_model(self):
        """
        안전하게 fine-tuning된 모델 저장
        
        목표: basis_coeff @ U^T로 weight를 재구성하여 최종 모델 저장
        
        중요: 훈련 중 업데이트된 basis_coeff를 사용!
        
        절차:
        1. 업데이트된 basis_coeff @ U^T 계산 (훈련된 모델)
        2. weight.data에 재구성된 가중치 할당
        3. 모델을 HuggingFace 형식으로 저장
        
        결과:
        - 안전하게 fine-tuning된 모델이 저장됨
        - basis_coeff는 훈련으로 업데이트됨
        - Phase 3에서 이 모델을 로드하여 masked fine-tuning 수행
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"💾 [Step 1] 최종 모델 재구성")
            self.logger.info(f"{'='*70}")
            
            # 모델을 평가 모드로 설정 (dropout 등 비활성화)
            self.model.eval()
            
            # ✅ Step 1: 각 레이어의 weight를 basis_coeff @ U^T로 재구성 - Multiple Layer Types
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            layers_with_basis = [key for key in self.basis_data.keys() 
                                if key[0] in target_indices]
            
            self.logger.info(f"재구성할 (layer, type) 조합 수: {len(layers_with_basis)}")
            self.logger.info(f"수식: weight_final = basis_coeff_trained @ U^T")
            self.logger.info(f"의미: 훈련된 기저 계수를 원본 가중치 공간으로 변환")
            
            for layer_idx, layer_type in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = self._get_target_module(layer, layer_type)
                
                if hasattr(target_module, 'basis_coeff') and hasattr(target_module, 'U_matrix'):
                    self.logger.debug(f"\n  Layer {layer_idx} ({layer_type}) 처리 중...")
                    
                    # 훈련된 basis_coeff 추출 (detach 후 CPU로)
                    basis_coeff_trained = target_module.basis_coeff.detach().cpu()  # (d_out, rank)
                    basis_coeff_init = self.basis_coeffs.get((layer_idx, layer_type), None)  # 초기값
                    U_matrix = target_module.U_matrix.detach().cpu()  # (d_in, rank) - CPU로
                    
                    self.logger.debug(f"    - basis_coeff shape: {basis_coeff_trained.shape} (훈련됨)")
                    self.logger.debug(f"    - U_matrix shape: {U_matrix.shape} (고정)")
                    
                    # 훈련 전후 비교
                    if basis_coeff_init is not None:
                        try:
                            # basis_coeff_init을 CPU tensor로 변환
                            if isinstance(basis_coeff_init, np.ndarray):
                                basis_coeff_init_tensor = torch.from_numpy(basis_coeff_init).float()
                            else:
                                basis_coeff_init_tensor = basis_coeff_init.detach().cpu() if hasattr(basis_coeff_init, 'detach') else torch.tensor(basis_coeff_init).float()
                            
                            coeff_change = (basis_coeff_trained - basis_coeff_init_tensor).norm().item()
                            self.logger.info(f"  ✓ Layer {layer_idx} ({layer_type}) - basis_coeff 변화:")
                            self.logger.info(f"     - Frobenius norm of change: {coeff_change:.6f}")
                            self.logger.info(f"     - 초기값 norm: {basis_coeff_init_tensor.norm().item():.6f}")
                            self.logger.info(f"     - 훈련후 norm: {basis_coeff_trained.norm().item():.6f}")
                        except Exception as e:
                            self.logger.warning(f"  ⚠ Layer {layer_idx} ({layer_type}) - 변화 비교 실패: {str(e)}")
                    
                    # 가중치 재구성: basis_coeff @ U^T
                    weight_reconstructed = basis_coeff_trained @ U_matrix.T  # (d_out, d_in)
                    
                    self.logger.debug(f"    - weight_reconstructed shape: {weight_reconstructed.shape}")
                    
                    # weight.data에 할당 (GPU로 옮김)
                    target_module.weight.data = weight_reconstructed.to(target_module.weight.device)
            
            # ✅ Step 2: 모델을 transformers 형식으로 저장
            model_save_dir = os.path.join(self.args.checkpoint_dir, 'phase2_finetuned_model')
            os.makedirs(model_save_dir, exist_ok=True)
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"💾 [Step 2] 안전 정렬 모델 저장")
            self.logger.info(f"{'='*70}")
            
            self.model.save_pretrained(model_save_dir)
            self.tokenizer.save_pretrained(model_save_dir)
            
            self.logger.info(f"✓ 모델 저장 완료: {model_save_dir}")
            self.logger.info(f"  - Format: HuggingFace (pytorch_model.bin + tokenizer)")
            self.logger.info(f"  - 이 모델은 안전 데이터로 파인튜닝된 모델입니다")
            self.logger.info(f"  - Phase 3에서 이 모델을 로드하여 masked fine-tuning 수행")
            
            return model_save_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save finetuned model: {str(e)}", exc_info=True)
            raise
    
    def save_basis_coefficients(self):
        """
        학습된 basis_coeff 저장 (Phase 3에서 사용 가능하도록)
        
        목표: basis_coeff를 저장하여 Phase 3에서 로드 가능하게 함
        
        결과:
        - layer_type 서브디렉토리에 basis_coeff_{layer_idx}.pt 파일 생성
        - Phase 3에서 이를 로드하여 basis_coeff 사용 가능
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"[Step 3] Saving Basis Coefficients (Multiple Layer Types)")
            self.logger.info(f"{'='*70}")
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            layers_with_basis = [key for key in self.basis_data.keys() 
                                if key[0] in target_indices]
            
            for layer_idx, layer_type in layers_with_basis:
                coeffs_dir = os.path.join(self.args.checkpoint_dir, 'basis_coefficients', layer_type)
                os.makedirs(coeffs_dir, exist_ok=True)
                
                layer = self.model.model.layers[layer_idx]
                target_module = self._get_target_module(layer, layer_type)
                
                if hasattr(target_module, 'basis_coeff'):
                    basis_coeff = target_module.basis_coeff.detach().cpu()
                    save_path = os.path.join(coeffs_dir, f'layer_{layer_idx:02d}_basis_coeff.pt')
                    
                    torch.save({
                        'basis_coeff': basis_coeff,
                        'shape': basis_coeff.shape,
                        'layer_idx': layer_idx,
                        'layer_type': layer_type,
                    }, save_path)
                    
                    self.logger.info(f"  ✓ Layer {layer_idx} ({layer_type}): {basis_coeff.shape} saved")
            
            self.logger.info(f"✓ Basis coefficients saved: basis_coefficients/(layer_type)/")
            self.logger.info(f"{'='*70}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to save basis coefficients: {str(e)}", exc_info=True)
            raise
    
    def generate_masks(self, keep_ratio=0.1):
        """
        Importance 점수 기반으로 마스크 생성 (Element-wise) - Multiple Layer Types
        
        목표: 안전 파인튜닝 중 중요한 기저 차원을 선별하여 Phase 3에서 보호
        
        방식:
        1. importance 점수 기반 threshold 계산
        2. 상위 keep_ratio (10%) 차원을 "중요"로 표시
        3. 나머지 90% 차원은 Phase 3에서 학습 가능
        
        결과:
        - mask[i] = 1 (또는 True): 중요한 차원 → Phase 3에서 동결
        - mask[i] = 0 (또는 False): 덜 중요한 차원 → Phase 3에서 학습 가능
        
        Args:
            keep_ratio: 유지할 weight의 비율 (0.1 = 상위 10%)
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"🎯 마스크 생성 (Element-wise, Multiple Layer Types)")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"목표: 안전 파인튜닝 중 중요한 기저 차원 선별")
            self.logger.info(f"방식: Quantile 기반 상위 {int(keep_ratio*100)}% 선별")
            
            for (layer_idx, layer_type), importance in self.importances.items():
                self.logger.info(f"\n  Layer {layer_idx} ({layer_type}):")
                
                # 평탄화된 importance에서 quantile 기반 threshold 계산
                importance_flat = importance.flatten()
                threshold = np.quantile(importance_flat, 1 - keep_ratio)
                
                self.logger.info(f"    - Importance 범위: [{importance_flat.min():.6f}, {importance_flat.max():.6f}]")
                self.logger.info(f"    - Threshold (상위 {int(keep_ratio*100)}%): {threshold:.6f}")
                
                # 이진 마스크 생성 (1: 중요/동결, 0: 덜 중요/학습 가능)
                mask = (importance_flat >= threshold).astype(np.float32)
                
                self.masks[(layer_idx, layer_type)] = mask
                
                frozen_count = mask.sum()
                trainable_count = len(mask) - frozen_count
                actual_ratio = frozen_count / len(mask)
                
                self.logger.info(f"    📊 마스크 통계:")
                self.logger.info(f"       - 동결 차원 (mask=1): {int(frozen_count)}/{len(mask)} ({actual_ratio*100:.1f}%)")
                self.logger.info(f"       - 학습 가능 차원 (mask=0): {int(trainable_count)}/{len(mask)} ({(1-actual_ratio)*100:.1f}%)")
                self.logger.info(f"       - Phase 3에서 {int(trainable_count)}개 차원만 업데이트됨")
            
            self.logger.info(f"\n✓ 마스크 생성 완료")
            
        except Exception as e:
            self.logger.error(f"Failed to generate masks: {str(e)}", exc_info=True)
            raise
    
    def save_masks(self):
        """
        생성된 마스크를 저장
        
        목표: Phase 3에서 사용할 마스크 저장
        
        파일 구조:
        - masks/
          - layer_29_mask.pt
          - layer_30_mask.pt
          - layer_31_mask.pt
          - metadata.json (통계)
        
        Log:
        - 저장 경로
        - 각 레이어별 동결/학습 가능 차원 수
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"💾 마스크 저장 (Multiple Layer Types)")
            self.logger.info(f"{'='*70}")
            
            # 마스크 저장 - layer_type 서브디렉토리 구조
            total_frozen = 0
            total_trainable = 0
            
            for (layer_idx, layer_type), mask in self.masks.items():
                masks_dir = os.path.join(self.args.checkpoint_dir, 'masks', layer_type)
                os.makedirs(masks_dir, exist_ok=True)
                
                save_path = os.path.join(masks_dir, f'layer_{layer_idx:02d}_mask.pt')
                torch.save(torch.from_numpy(mask).float(), save_path)
                
                frozen_count = int(mask.sum())
                trainable_count = len(mask) - frozen_count
                
                total_frozen += frozen_count
                total_trainable += trainable_count
                
                self.logger.debug(f"  ✓ Layer {layer_idx} ({layer_type}): {frozen_count} frozen, {trainable_count} trainable")
            
            # 메타데이터 저장
            metadata_dir = os.path.join(self.args.checkpoint_dir, 'masks')
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata = {
                'model_name': self.args.model_name,
                'layer_types': self.args.layer_type if isinstance(self.args.layer_type, list) else self.args.layer_type.split(','),
                'num_masks': len(self.masks),
                'safety_samples': self.args.safety_samples,
                'keep_ratio': self.args.keep_ratio if hasattr(self.args, 'keep_ratio') else 0.1,
                'total_loss': self.stats['total_loss'],
                'total_frozen_dims': int(total_frozen),
                'total_trainable_dims': int(total_trainable),
            }
            
            metadata_path = os.path.join(metadata_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"\n✅ 마스크 저장 완료")
            self.logger.info(f"   - 저장 경로: masks/(layer_type)/")
            self.logger.info(f"   - 파일 수: {len(self.masks)} mask files + metadata.json")
            self.logger.info(f"\n📊 마스크 통계:")
            self.logger.info(f"   - 총 동결 차원: {total_frozen}")
            self.logger.info(f"   - 총 학습 가능 차원: {total_trainable}")
            self.logger.info(f"   - 전체: {total_frozen + total_trainable}")
            if total_frozen + total_trainable > 0:
                frozen_ratio = 100 * total_frozen / (total_frozen + total_trainable)
                self.logger.info(f"   - 동결 비율: {frozen_ratio:.1f}%")
            
            self.logger.info(f"\n🔒 Phase 3 준비:")
            self.logger.info(f"   - {total_trainable:,}개 차원은 GSM8K로 학습 가능")
            self.logger.info(f"   - {total_frozen:,}개 차원은 안전성을 위해 동결")
            self.logger.info(f"{'='*70}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to save masks: {str(e)}", exc_info=True)
            raise
    
    def _get_target_module(self, layer, layer_type=None):
        """
        주어진 layer에서 layer_type에 맞는 모듈 반환
        
        Args:
            layer: transformer layer 객체
            layer_type: 'ffn_down', 'ffn_up', 'attn_q', 'attn_k', 'attn_v'
                       None이면 self.args.layer_type 사용 (단일 타입 호환성)
            
        Returns:
            target_module: 선택된 projection 모듈
        """
        if layer_type is None:
            # 단일 layer_type 호환성을 위해 args에서 첫 번째 타입 사용
            if isinstance(self.args.layer_type, str):
                layer_type = self.args.layer_type.split(',')[0].strip()
            else:
                layer_type = self.args.layer_type[0]
        
        if layer_type == 'ffn_down':
            return layer.mlp.down_proj
        elif layer_type == 'ffn_up':
            return layer.mlp.up_proj
        elif layer_type == 'attn_q':
            return layer.self_attn.q_proj
        elif layer_type == 'attn_k':
            return layer.self_attn.k_proj
        elif layer_type == 'attn_v':
            return layer.self_attn.v_proj
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def _parse_target_layers(self, num_layers):
        """타겟 레이어 파싱 (Phase 1과 동일)"""
        target = self.args.target_layers.strip()
        
        if target == 'all':
            return list(range(num_layers))
        elif target == 'early':
            return list(range(0, min(11, num_layers)))
        elif target == 'middle':
            return list(range(11, min(22, num_layers)))
        elif target == 'late':
            return list(range(22, num_layers))
        elif target == 'last':
            return [num_layers - 1]
        
        if '-' in target:
            try:
                start, end = target.split('-')
                start, end = int(start.strip()), int(end.strip())
                return list(range(start, min(end + 1, num_layers)))
            except ValueError:
                raise ValueError(f"Invalid range format: {target}")
        
        try:
            layer_idx = int(target)
            if 0 <= layer_idx < num_layers:
                return [layer_idx]
            else:
                raise ValueError(f"Invalid layer index: {layer_idx}")
        except ValueError:
            raise ValueError(f"Invalid target_layers format: {target}")
