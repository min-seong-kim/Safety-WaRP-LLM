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
        self.basis_data = {}  # layer_idx -> {'U': U, 'S': S, 'Vh': Vh}
        self.basis_metadata = {}
        
        # Reparameterized 가중치
        self.original_weights = {}  # layer_idx -> W_original
        self.basis_coeffs = {}  # layer_idx -> basis_coeff (trainable)
        
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
        Phase 1에서 저장된 basis 로드
        
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
            self.logger.info(f"  - Layer type: {self.basis_metadata.get('layer_type')}")
            self.logger.info(f"  - Num layers: {self.basis_metadata.get('num_layers')}")
            self.logger.info(f"  - Target layers: {self.basis_metadata.get('target_layers')}")
            
            # basis 디렉토리에서 모든 layer_*_svd.pt 파일 찾기
            import glob
            svd_files = sorted(glob.glob(os.path.join(self.basis_dir, 'layer_*_svd.pt')))
            
            if not svd_files:
                raise FileNotFoundError(f"No SVD files found in {self.basis_dir}")
            
            # 각 SVD 파일 로드
            for svd_path in svd_files:
                # 파일명에서 레이어 인덱스 추출: layer_31_svd.pt -> 31
                filename = os.path.basename(svd_path)
                layer_idx = int(filename.split('_')[1])
                
                svd_data = torch.load(svd_path, map_location='cpu')
                self.basis_data[layer_idx] = {
                    'U': svd_data['U'].to(self.args.device),
                    'S': svd_data['S'].to(self.args.device),
                    'Vh': svd_data['Vh'].to(self.args.device),
                }
            
            self.logger.info(f"✓ Basis loaded: {len(self.basis_data)} layers")
            self.logger.info(f"  - Layer indices: {sorted(self.basis_data.keys())}")
            
            # 샘플 레이어 정보 출력
            if len(self.basis_data) > 0:
                sample_idx = sorted(self.basis_data.keys())[0]
                sample_U = self.basis_data[sample_idx]['U']
                self.logger.info(f"  - Sample layer {sample_idx}: U shape = {sample_U.shape}")
            
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
        | Weight      | W_original      | (d_out,  | False    | 분석용 reference |
        | Basis       | basis_coeff     | (d_out,  | True     | 학습 가능 파라미터 |
        | Basis       | U_matrix        | (d_in,   | False    | 고정된 basis   |
        |             |                  |          |          |                 |
        | 관계식: W_reconstructed = basis_coeff @ U^T                |
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
            self.logger.info("Weight Space → Basis Space Transformation")
            self.logger.info("="*70)
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            for layer_idx in target_indices:
                if layer_idx not in self.basis_data:
                    self.logger.debug(f"Layer {layer_idx}: No basis available, skipping")
                    continue
                
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                
                # ✅ Step 1: 원본 가중치 저장 (분석용, 고정)
                W_original = target_module.weight.data.clone()  # (d_out, d_in) = (4096, 14336)
                self.original_weights[layer_idx] = W_original
                
                # ✅ Step 2: Basis 행렬 추출 및 dtype 변환
                U = self.basis_data[layer_idx]['U']  # (d_in, rank) in float32
                model_dtype = W_original.dtype
                U = U.to(dtype=model_dtype, device=W_original.device)
                
                # ✅ Step 3: basis_coeff 초기화
                # basis_coeff = W_original @ U (투영)
                # 이렇게 하면 basis_coeff @ U^T ≈ W_original (근사)
                basis_coeff_init = W_original @ U  # (d_out, rank) = (4096, 14336)
                
                # basis_coeff를 학습 가능한 Parameter로 등록
                target_module.basis_coeff = nn.Parameter(basis_coeff_init.clone(), requires_grad=True)
                
                # U_matrix는 고정된 basis (requires_grad=False)
                target_module.U_matrix = U.clone().detach()  # (d_in, rank) - 고정
                target_module.U_matrix.requires_grad = False
                
                self.basis_coeffs[layer_idx] = basis_coeff_init
                
                # 로깅
                self.logger.info(f"\nLayer {layer_idx}:")
                self.logger.info(f"  ✓ W_original (고정):     {W_original.shape} = (4096, 14336)")
                self.logger.info(f"  ✓ basis_coeff (학습):    {basis_coeff_init.shape} = (4096, 14336)")
                self.logger.info(f"  ✓ U_matrix (고정):      {U.shape} = (14336, 14336)")
                self.logger.info(f"  ✓ Forward: W = basis_coeff @ U^T")
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✓ Reparameterization completed: {len(self.basis_coeffs)} layers")
            self.logger.info(f"{'='*70}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to reparameterize weights: {str(e)}", exc_info=True)
            raise
    
    def compute_importance(self):
        """
        안전 데이터로 fine-tuning하면서 importance 점수 계산
        
        절차:
        1. 모델을 훈련 모드로 설정
        2. Optimizer 설정 (basis_coeff 파라미터만)
        3. Forward pass: weight = basis_coeff @ U^T로 동적 복원
        4. Loss 계산 및 역전파: basis_coeff.grad 계산
        5. Optimizer.step(): basis_coeff 업데이트
        6. Importance 누적: |basis_coeff.grad|를 배치별로 수집
        7. 에포크 완료 후 importance 평균 계산
        
        핵심:
        - basis_coeff만 학습됨 (W_original과 U_matrix는 고정)
        - Gradient는 basis_coeff에 대해서만 계산됨
        - Importance = |∂L/∂basis_coeff| (각 에포크의 gradient 절댓값)
        
        Log:
        - 에포크별 손실
        - Batch별 gradient 통계
        - 최종 importance 점수
        """
        try:
            self.logger.info("Starting Phase 2: Fine-tuning + Importance Scoring...")
            self.logger.info("\n" + "="*70)
            self.logger.info("Training Setup")
            self.logger.info("="*70)
            
            # ✅ Step 1: 모델을 훈련 모드로 설정
            self.model.train()
            self.logger.info("✓ Model set to training mode")
            
            # ✅ Step 2: Optimizer 설정 (basis_coeff 파라미터만)
            basis_params = []
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            layers_with_basis = [idx for idx in target_indices if idx in self.basis_data]
            
            for layer_idx in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                if hasattr(target_module, 'basis_coeff'):
                    basis_params.append(target_module.basis_coeff)
            
            if len(basis_params) == 0:
                self.logger.warning("No basis_coeff parameters found! Skipping importance computation.")
                return
            
            learning_rate = getattr(self.args, 'safety_lr', 1e-5)
            weight_decay = getattr(self.args, 'safety_weight_decay', 0.01)
            
            optimizer = torch.optim.AdamW(basis_params, lr=learning_rate, weight_decay=weight_decay)
            
            self.logger.info(f"✓ Optimizer created: AdamW")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay: {weight_decay}")
            self.logger.info(f"  - Parameters: {len(basis_params)} basis_coeff tensors")
            self.logger.info(f"  - Layers: {layers_with_basis}")
            
            # ✅ Step 3: Forward hook 등록 - weight 동적 복원
            def make_forward_hook(layer_idx):
                """basis_coeff @ U^T로 weight 복원"""
                def hook(module, input, output):
                    if hasattr(module, 'basis_coeff') and hasattr(module, 'U_matrix'):
                        basis_coeff = module.basis_coeff  # (d_out, rank)
                        U_matrix = module.U_matrix        # (d_in, rank)
                        
                        # W = basis_coeff @ U^T
                        weight_reconstructed = basis_coeff @ U_matrix.T  # (d_out, d_in)
                        
                        # In-place 업데이트
                        module.weight.data = weight_reconstructed
                    return output
                return hook
            
            # Hook 등록
            hooks = []
            for layer_idx in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                hook_handle = target_module.register_forward_hook(make_forward_hook(layer_idx))
                hooks.append(hook_handle)
            
            self.hook_handles = hooks  # ✅ 나중에 제거하기 위해 저장
            self.logger.info(f"✓ {len(hooks)} forward hooks registered")
            
            # ✅ Step 4: Importance 저장소 초기화
            importances = {idx: [] for idx in layers_with_basis}
            
            self.logger.info("\n" + "="*70)
            self.logger.info("Fine-tuning with Importance Tracking")
            self.logger.info("="*70)
            
            # ✅ Step 5: 훈련 루프
            epochs = getattr(self.args, 'safety_epochs', 1)
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
                    safety_responses = batch['safety_response']
                    
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
                        
                        # ✅ Importance 수집: |basis_coeff.grad|
                        for layer_idx in layers_with_basis:
                            layer = self.model.model.layers[layer_idx]
                            target_module = layer.mlp.down_proj
                            
                            if hasattr(target_module, 'basis_coeff') and target_module.basis_coeff.grad is not None:
                                # Gradient 절댓값 (element-wise)
                                grad_abs = torch.abs(target_module.basis_coeff.grad)  # (d_out, rank)
                                importances[layer_idx].append(grad_abs.detach().cpu())
                        
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
            
            # Hook 제거
            for hook in hooks:
                hook.remove()
            self.logger.info(f"\n✓ Forward hooks removed")
            
            # ✅ Step 6: Importance 평균 계산
            self.logger.info("\n" + "="*70)
            self.logger.info("Computing Importance Scores")
            self.logger.info("="*70)
            
            self.importances = {}
            for layer_idx in layers_with_basis:
                if len(importances[layer_idx]) > 0:
                    # 모든 배치의 gradient를 스택
                    layer_importances = torch.stack(importances[layer_idx], dim=0)  # (num_batches, d_out, rank)
                    
                    # 배치 축 평균
                    importance_mean = layer_importances.mean(dim=0)  # (d_out, rank)
                    
                    # Input 차원별로 sum (각 input이 모든 output에 미치는 누적 영향)
                    importance_per_input = importance_mean.sum(dim=0)  # (rank,) = (14336,)
                    
                    self.importances[layer_idx] = importance_per_input.float().cpu().numpy()
                    
                    self.logger.info(f"\n✓ Layer {layer_idx}:")
                    self.logger.info(f"  - Gradient shape (per batch): (d_out, rank) = {importance_mean.shape}")
                    self.logger.info(f"  - Importance aggregated to input-wise (sum): {self.importances[layer_idx].shape}")
                    self.logger.info(f"  - Mean: {self.importances[layer_idx].mean():.6f}")
                    self.logger.info(f"  - Std: {self.importances[layer_idx].std():.6f}")
                    self.logger.info(f"  - Min: {self.importances[layer_idx].min():.6f}")
                    self.logger.info(f"  - Max: {self.importances[layer_idx].max():.6f}")
                    self.logger.info(f"  - Median: {np.median(self.importances[layer_idx]):.6f}")
                else:
                    self.logger.warning(f"Layer {layer_idx}: No gradients collected")
            
            avg_loss = total_loss / max(total_batches, 1)
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✓ Phase 2 Summary:")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"  - Total loss (all epochs): {total_loss:.4f}")
            self.logger.info(f"  - Average loss per batch: {avg_loss:.4f}")
            self.logger.info(f"  - Total batches processed: {total_batches}")
            self.logger.info(f"  - Layers with importance scores: {len(self.importances)}")
            self.logger.info(f"  - Layers with basis: {len(layers_with_basis)}")
            self.logger.info(f"{'='*70}\n")
            
            self.stats['total_loss'] = total_loss
            
        except Exception as e:
            self.logger.error(f"Failed to compute importance: {str(e)}", exc_info=True)
            raise
    
    def save_finetuned_model(self):
        """
        안전하게 fine-tuning된 모델 저장
        
        목표: basis_coeff @ U^T로 weight를 재구성하여 최종 모델 저장
        
        절차:
        1. 모든 basis_coeff @ U^T 계산
        2. weight.data에 재구성된 가중치 할당
        3. 모델을 HuggingFace 형식으로 저장
        
        결과:
        - 안전하게 fine-tuning된 모델이 저장됨
        - Phase 1의 basis와 mask가 모두 포함된 완전한 모델
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"[Step 1] Reconstructing Final Model")
            self.logger.info(f"{'='*70}")
            
            # 모델을 평가 모드로 설정 (dropout 등 비활성화)
            self.model.eval()
            
            # ✅ Step 1: Hook 제거 (weight 고정)
            for handle in self.hook_handles:
                handle.remove()
            self.logger.info("✓ Removed forward hooks")
            
            # ✅ Step 2: 각 레이어의 weight를 basis_coeff @ U^T로 재구성
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            layers_with_basis = [idx for idx in target_indices if idx in self.basis_data]
            
            for layer_idx in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                
                if hasattr(target_module, 'basis_coeff') and hasattr(target_module, 'U_matrix'):
                    # basis_coeff @ U^T 계산
                    basis_coeff = target_module.basis_coeff.detach()  # (d_out, rank)
                    U_matrix = target_module.U_matrix.detach()  # (d_in, rank)
                    
                    weight_reconstructed = basis_coeff @ U_matrix.T  # (d_out, d_in)
                    
                    # weight에 할당 (detach하여 gradient 제거)
                    target_module.weight.data = weight_reconstructed
                    
                    # basis_coeff와 U_matrix 속성 제거 (불필요해짐)
                    delattr(target_module, 'basis_coeff')
                    delattr(target_module, 'U_matrix')
                    
                    norm_before = self.original_weights[layer_idx].norm().item()
                    norm_after = weight_reconstructed.norm().item()
                    
                    self.logger.info(f"  Layer {layer_idx}:")
                    self.logger.info(f"    - Weight reconstructed: {weight_reconstructed.shape}")
                    self.logger.info(f"    - Weight norm: {norm_before:.4f} → {norm_after:.4f}")
            
            # ✅ Step 3: 모델을 transformers 형식으로 저장
            model_save_dir = os.path.join(self.args.checkpoint_dir, 'phase2_finetuned_model')
            os.makedirs(model_save_dir, exist_ok=True)
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"[Step 2] Saving Safety-aligned Model")
            self.logger.info(f"{'='*70}")
            
            self.model.save_pretrained(model_save_dir)
            self.tokenizer.save_pretrained(model_save_dir)
            
            self.logger.info(f"✓ Model saved: {model_save_dir}")
            self.logger.info(f"  - Format: HuggingFace (pytorch_model.bin + tokenizer)")
            self.logger.info(f"  - Size: {len(list(model_save_dir))} files")
            
            return model_save_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save finetuned model: {str(e)}", exc_info=True)
            raise
    
    def save_basis_coefficients(self):
        """
        학습된 basis_coeff 저장 (Phase 3에서 사용 가능하도록)
        
        목표: basis_coeff를 저장하여 Phase 3에서 로드 가능하게 함
        
        결과:
        - basis_coeff_{layer_idx}.pt 파일 생성
        - Phase 3에서 이를 로드하여 basis_coeff 사용 가능
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"[Step 3] Saving Basis Coefficients")
            self.logger.info(f"{'='*70}")
            
            coeffs_dir = os.path.join(self.args.checkpoint_dir, 'basis_coefficients')
            os.makedirs(coeffs_dir, exist_ok=True)
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            layers_with_basis = [idx for idx in target_indices if idx in self.basis_data]
            
            for layer_idx in layers_with_basis:
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                
                if hasattr(target_module, 'basis_coeff'):
                    basis_coeff = target_module.basis_coeff.detach().cpu()
                    save_path = os.path.join(coeffs_dir, f'layer_{layer_idx:02d}_basis_coeff.pt')
                    
                    torch.save({
                        'basis_coeff': basis_coeff,
                        'shape': basis_coeff.shape,
                        'layer_idx': layer_idx,
                    }, save_path)
                    
                    self.logger.info(f"  ✓ Layer {layer_idx}: {basis_coeff.shape} saved")
            
            self.logger.info(f"✓ Basis coefficients saved: {coeffs_dir}")
            self.logger.info(f"{'='*70}\n")
            
            return coeffs_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save basis coefficients: {str(e)}", exc_info=True)
            raise
    
    def generate_masks(self, keep_ratio=0.1):
        """
        Importance 점수 기반으로 마스크 생성 (Element-wise)
        
        방식:
        1. 각 레이어의 importance 평탄화: (d_out, rank) -> (d_out*rank,) = (14336,)
        2. Quantile 계산: threshold = quantile(importance, 1 - keep_ratio)
        3. Mask = 1 if importance >= threshold else 0 (각 요소별)
        
        결과:
        - mask shape: (14336,) - 각 weight 요소별 이진 마스크
        - mask[i] = 1: 중요한 weight (freeze), 0: 덜 중요한 weight (trainable)
        
        Log:
        - 각 레이어의 threshold
        - 유지되는 weight 비율
        
        Args:
            keep_ratio: 유지할 weight의 비율 (0.1 = 상위 10%)
        """
        try:
            self.logger.info(f"Generating masks with keep_ratio={keep_ratio}...")
            
            for layer_idx, importance in self.importances.items():
                # 평탄화된 importance에서 quantile 기반 threshold 계산
                importance_flat = importance.flatten()
                threshold = np.quantile(importance_flat, 1 - keep_ratio)
                
                # 이진 마스크 생성 (1: freeze/중요, 0: update/불필요)
                # 원래 shape로 평탄화 (14336,)
                mask = (importance_flat >= threshold).astype(np.float32)
                
                self.masks[layer_idx] = mask
                
                keep_count = mask.sum()
                total_count = len(mask)
                actual_ratio = keep_count / total_count
                
                self.logger.info(f"✓ Layer {layer_idx} (element-wise):")
                self.logger.info(f"  - Importance shape: {importance.shape}")
                self.logger.info(f"  - Flattened shape: {mask.shape}")
                self.logger.info(f"  - Threshold: {threshold:.6f}")
                self.logger.info(f"  - Kept weights: {keep_count}/{total_count} ({actual_ratio*100:.1f}%)")
            
            self.logger.info(f"✓ Mask generation completed")
            
        except Exception as e:
            self.logger.error(f"Failed to generate masks: {str(e)}", exc_info=True)
            raise
    
    def save_masks(self):
        """
        생성된 마스크를 저장
        
        파일 구조:
        - masks/
          - layer_00_mask.pt
          - layer_01_mask.pt
          - ...
          - importances.json (통계)
        
        Log:
        - 저장 경로
        - 파일 정보
        """
        try:
            masks_dir = os.path.join(self.args.checkpoint_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Saving masks to {masks_dir}...")
            
            # 마스크 저장
            for layer_idx, mask in self.masks.items():
                save_path = os.path.join(masks_dir, f'layer_{layer_idx:02d}_mask.pt')
                torch.save(torch.from_numpy(mask).float(), save_path)
                frozen_count = mask.sum()
                trainable_count = len(mask) - frozen_count
                self.logger.debug(f"  - Layer {layer_idx}: {frozen_count} frozen, {trainable_count} trainable")
            
            # 메타데이터 저장
            metadata = {
                'model_name': self.args.model_name,
                'layer_type': self.args.layer_type,
                'num_layers': len(self.masks),
                'safety_samples': self.args.safety_samples,
                'keep_ratio': self.args.keep_ratio if hasattr(self.args, 'keep_ratio') else 0.1,
                'total_loss': self.stats['total_loss'],
            }
            
            metadata_path = os.path.join(masks_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"✓ Masks saved successfully")
            self.logger.info(f"  - Directory: {masks_dir}")
            self.logger.info(f"  - Files: {len(self.masks)} mask files + metadata.json")
            
            if len(self.masks) > 0:
                total_frozen = sum(mask.sum() for mask in self.masks.values())
                total_coeffs = sum(len(mask) for mask in self.masks.values())
                if total_coeffs > 0:
                    self.logger.info(f"  - Total: {total_frozen}/{total_coeffs} frozen ({100*total_frozen/total_coeffs:.1f}%)")
            else:
                self.logger.warning(f"  - No masks were generated (self.masks is empty)")
            self.logger.info(f"{'='*60}\n")
            
            return masks_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save masks: {str(e)}", exc_info=True)
            raise
    
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
