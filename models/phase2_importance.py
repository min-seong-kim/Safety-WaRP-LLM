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
    Phase 2: Importance Scoring
    
    절차:
    1. Phase 1에서 계산된 basis 로드
    2. 모델 가중치를 basis 공간으로 재매개변수화
    3. 안전 데이터로 모델 실행 (teacher forcing)
    4. 손실 계산: token-level cross-entropy
    5. 역전파로 gradient 계산
    6. 계수별 importance 점수 계산 (||gradient||)
    7. 임계값으로 마스크 생성 (상위 keep_ratio 유지)
    8. 누적 마스크 생성 (이전 Phase 마스크와 합치기)
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
        
        변환:
        W_original -> basis_coeff = W_original @ U^T
        
        그런 후 gradient는 basis_coeff에 대해 계산되고,
        importance = ||∇basis_coeff||로 계산됨
        
        Log:
        - 재매개변수화된 레이어 수
        - 각 레이어의 계수 형태
        """
        try:
            self.logger.info("Reparameterizing weights to basis space...")
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            for layer_idx in target_indices:
                if layer_idx not in self.basis_data:
                    continue
                
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                
                # 원본 가중치 저장
                W_original = target_module.weight.data.clone()  # (d_out, d_in)
                self.original_weights[layer_idx] = W_original
                
                # Basis 추출 (모델과 같은 dtype으로 변환)
                U = self.basis_data[layer_idx]['U']  # (d_in, d_in) in float32
                model_dtype = W_original.dtype
                U = U.to(dtype=model_dtype, device=W_original.device)
                
                # 재매개변수화: basis_coeff = W @ U^T
                basis_coeff = W_original @ U.T  # (d_out, d_in)
                
                # basis_coeff를 Parameter로 등록하여 gradient tracking 활성화
                # 하지만 원본 weight는 그대로 둠
                target_module.basis_coeff = nn.Parameter(basis_coeff.clone())
                target_module.U_forward = U
                
                # 중요: basis_coeff를 optimizer에 추가할 수 있도록 register_parameter 사용
                # (하지만 Phase 2에서는 optimizer를 사용하지 않고 gradient만 계산)
                
                self.basis_coeffs[layer_idx] = basis_coeff
                
                self.logger.debug(f"Layer {layer_idx}: W shape {W_original.shape} -> coeff shape {basis_coeff.shape}")
            
            self.logger.info(f"✓ Reparameterization completed: {len(self.basis_coeffs)} layers")
            
        except Exception as e:
            self.logger.error(f"Failed to reparameterize weights: {str(e)}", exc_info=True)
            raise
    
    def compute_importance(self):
        """
        안전 데이터에 대해 importance 점수 계산
        
        절차:
        1. 배치 반복
        2. Teacher forcing: 모델 입력은 harmful_prompt, 목표는 safety_response
        3. Loss 계산: token-level cross-entropy
        4. 역전파
        5. 각 basis_coeff의 gradient 수집
        6. Importance = ||gradient||
        
        Log:
        - 배치별 손실
        - 각 레이어의 importance 통계
        """
        try:
            self.logger.info("Computing importance scores...")
            
            # 각 레이어의 importance 초기화
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            importances = {idx: [] for idx in target_indices}
            
            progress_bar = tqdm(
                self.dataloader,
                desc="Computing importance",
                total=len(self.dataloader)
            )
            
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(progress_bar):
                harmful_prompts = batch['harmful_prompt']
                safety_responses = batch['safety_response']
                
                # 결합된 입력-목표 시퀀스 생성 (Teacher Forcing)
                # Format: "{harmful_prompt}\n{safety_response}"
                combined_texts = [
                    f"{q}\n{a}" 
                    for q, a in zip(harmful_prompts, safety_responses)
                ]
                
                # 결합된 텍스트 토큰화
                combined = self.tokenizer(
                    combined_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                combined_ids = combined['input_ids'].to(self.model.device)
                combined_attn = combined['attention_mask'].to(self.model.device)
                
                # 기울기 계산 활성화
                self.model.zero_grad()
                
                # 모델 실행
                with torch.enable_grad():
                    outputs = self.model(
                        input_ids=combined_ids,
                        attention_mask=combined_attn
                    )
                    logits = outputs.logits  # (batch, seq_len, vocab_size)
                    
                    # Teacher forcing: shift targets
                    # logits[:, :-1, :] -> 마지막 토큰 제외 (다음 토큰 예측)
                    # combined_ids[:, 1:] -> 첫 토큰 제외 (목표 토큰)
                    pred_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab_size)
                    target_ids_shift = combined_ids[:, 1:].contiguous()  # (batch, seq_len-1)
                    
                    # Padding을 제외한 유효한 토큰만
                    attention_mask_shift = combined_attn[:, 1:].contiguous()  # (batch, seq_len-1)
                    valid_mask = (attention_mask_shift == 1) & (target_ids_shift != self.tokenizer.pad_token_id)
                    
                    # 모든 유효한 위치를 평탄화
                    pred_logits_flat = pred_logits[valid_mask]  # (num_valid, vocab_size)
                    target_ids_flat = target_ids_shift[valid_mask]  # (num_valid,)
                    
                    # Loss 계산
                    if len(target_ids_flat) > 0:
                        loss = nn.CrossEntropyLoss()(
                            pred_logits_flat,
                            target_ids_flat
                        )
                        
                        # 역전파
                        loss.backward()
                        
                        total_loss += loss.item()
                        
                        # Importance 수집: 각 layer의 weight gradient -> basis space에서 역변환
                        for layer_idx in target_indices:
                            if layer_idx not in self.basis_data:
                                continue
                            
                            layer = self.model.model.layers[layer_idx]
                            target_module = layer.mlp.down_proj
                            
                            # Weight gradient 얻기
                            if target_module.weight.grad is not None:
                                W_grad = target_module.weight.grad  # (d_out, d_in), dtype: bfloat16
                                U = self.basis_data[layer_idx]['U']  # dtype: float32
                                
                                # U를 W_grad와 같은 dtype으로 변환
                                U_casted = U.to(W_grad.dtype)
                                
                                # Gradient를 basis space로 변환
                                # ∂L/∂coeff = (∂L/∂W) @ U^T
                                coeff_grad = W_grad @ U_casted.T  # (d_out, d_in)
                                
                                # Importance = ||gradient per output neuron|| (L2 norm across input)
                                importance = torch.norm(coeff_grad, dim=1, p=2)
                                importances[layer_idx].append(importance.detach().cpu())
                            else:
                                self.logger.debug(f"Layer {layer_idx}: No weight gradient computed")
                    else:
                        self.logger.warning(f"Batch {batch_idx}: No valid tokens after masking")
                
                progress_bar.update(1)
                self.stats['total_loss'] += loss.item() if 'loss' in locals() else 0
            
            # 누적 importance 계산
            self.importances = {}
            for layer_idx in target_indices:
                if len(importances[layer_idx]) > 0:
                    # 모든 배치의 importance를 평균
                    layer_importances = torch.stack(importances[layer_idx], dim=0)
                    # bfloat16을 float32로 변환한 후 numpy로 변환
                    self.importances[layer_idx] = layer_importances.mean(dim=0).float().cpu().numpy()
                    
                    self.logger.info(f"Layer {layer_idx}:")
                    self.logger.info(f"  - Mean importance: {self.importances[layer_idx].mean():.6f}")
                    self.logger.info(f"  - Std importance: {self.importances[layer_idx].std():.6f}")
            
            avg_loss = total_loss / len(self.dataloader)
            self.logger.info(f"✓ Importance computation completed")
            self.logger.info(f"  - Average loss: {avg_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to compute importance: {str(e)}", exc_info=True)
            raise
    
    def generate_masks(self, keep_ratio=0.1):
        """
        Importance 점수 기반으로 마스크 생성
        
        방식:
        1. 모든 importance 점수 수집
        2. Quantile 계산: threshold = quantile(importance, 1 - keep_ratio)
        3. Mask = 1 if importance >= threshold else 0
        
        Log:
        - 각 레이어의 threshold
        - 유지되는 계수 비율
        
        Args:
            keep_ratio: 유지할 계수의 비율 (0.1 = 상위 10%)
        """
        try:
            self.logger.info(f"Generating masks with keep_ratio={keep_ratio}...")
            
            for layer_idx, importance in self.importances.items():
                # Quantile 기반 threshold
                threshold = np.quantile(importance, 1 - keep_ratio)
                
                # 이진 마스크 생성 (1: freeze/중요, 0: update/불필요)
                mask = (importance >= threshold).astype(np.float32)
                
                self.masks[layer_idx] = mask
                
                keep_count = mask.sum()
                total_count = len(mask)
                actual_ratio = keep_count / total_count
                
                self.logger.info(f"Layer {layer_idx}:")
                self.logger.info(f"  - Threshold: {threshold:.6f}")
                self.logger.info(f"  - Kept coefficients: {keep_count}/{total_count} ({actual_ratio*100:.1f}%)")
            
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
            
            self.logger.info(f"Saving masks to {masks_dir}...")
            
            # 마스크 저장
            for layer_idx, mask in self.masks.items():
                save_path = os.path.join(masks_dir, f'layer_{layer_idx:02d}_mask.pt')
                torch.save(torch.from_numpy(mask).float(), save_path)
            
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
