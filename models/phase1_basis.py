"""
Phase 1: Basis Construction
FFN down_proj 레이어에서 활성화를 수집하고 SVD를 통해 basis를 계산
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import os
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Phase1BasiBuilder:
    """
    Phase 1: Basis Construction Builder
    
    절차:
    1. 모델 로드 (LLaMA 3 8B)
    2. 안전 데이터 로드 (do-not-answer)
    3. Forward hook을 통해 각 FFN down_proj의 입력값 수집
    4. 수집된 활성화로부터 공분산 행렬 계산
    5. SVD를 통해 직교 기저(orthonormal basis) 계산
    6. Basis 저장
    """
    
    def __init__(self, args, logger):
        """
        Args:
            args: 커맨드라인 인자
            logger: 로거 객체
        """
        self.args = args
        self.logger = logger
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.dataloader = None
        
        # 활성화 수집을 위한 저장소
        self.activations = {}  # layer_idx -> (batch_size, seq_len, hidden_dim) 리스트
        self.covariance_matrices = {}  # layer_idx -> 공분산 행렬
        self.svd_results = {}  # layer_idx -> {'U': U, 'S': S, 'Vh': Vh}
        
        # Hook 핸들 저장소
        self.hooks = []
        
        # 통계
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'layers_processed': 0,
        }
    
    def load_model(self):
        """
        LLaMA 모델 로드
        
        Log:
        - 모델 로드 시작/완료
        - 모델 파라미터 수
        - 레이어 구조 확인
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
            
            # 모델 정보 로깅
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"  - Total parameters: {total_params:,}")
            self.logger.info(f"  - Model device: {self.model.device}")
            self.logger.info(f"  - Model dtype: {self.model.dtype}")
            
            # 레이어 구조 확인
            num_layers = len(self.model.model.layers)
            self.logger.info(f"  - Number of layers: {num_layers}")
            
            # 샘플 레이어 구조 확인
            sample_layer = self.model.model.layers[0]
            self.logger.info(f"  - Sample layer structure:")
            self.logger.info(f"    - MLPdown_proj: {sample_layer.mlp.down_proj}")
            self.logger.info(f"    - MLPup_proj: {sample_layer.mlp.up_proj}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info(f"✓ Tokenizer loaded successfully")
            
            # 모델을 evaluation 모드로 설정
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def load_safety_data(self):
        """
        안전 데이터 로드 (do-not-answer)
        
        Log:
        - 데이터셋 로드 상태
        - 필터링 결과
        - 최종 샘플 수
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
            
            # 샘플 출력
            sample_batch = next(iter(self.dataloader))
            self.logger.info(f"  - Sample batch keys: {sample_batch.keys()}")
            if 'harmful_prompt' in sample_batch:
                self.logger.info(f"  - Sample prompt: {sample_batch['harmful_prompt'][0][:100]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to load safety data: {str(e)}", exc_info=True)
            raise
    
    def register_activation_hooks(self):
        """
        Forward hook 등록
        FFN down_proj의 입력값(활성화)을 수집하기 위한 hook 등록
        
        Log:
        - 훅 등록된 레이어 수
        - 각 레이어의 모듈 정보
        """
        def get_hook(layer_idx):
            """특정 레이어를 위한 훅 함수 생성"""
            def hook(module, input, output):
                # input[0]: (batch_size, seq_len, hidden_dim)
                activation = input[0].detach()
                
                if layer_idx not in self.activations:
                    self.activations[layer_idx] = []
                
                self.activations[layer_idx].append(activation)
            
            return hook
        
        try:
            # 타겟 레이어 범위 결정
            num_layers = len(self.model.model.layers)
            if self.args.target_layers == 'all':
                layer_indices = list(range(num_layers))
            elif self.args.target_layers == 'early':
                layer_indices = list(range(0, min(11, num_layers)))
            elif self.args.target_layers == 'middle':
                layer_indices = list(range(11, min(22, num_layers)))
            elif self.args.target_layers == 'late':
                layer_indices = list(range(22, num_layers))
            
            self.logger.info(f"Registering hooks for layer type: {self.args.layer_type}")
            self.logger.info(f"Target layer indices: {layer_indices}")
            
            # 각 레이어에 훅 등록
            for layer_idx in layer_indices:
                layer = self.model.model.layers[layer_idx]
                
                if self.args.layer_type == 'ffn_down':
                    target_module = layer.mlp.down_proj
                elif self.args.layer_type == 'ffn_up':
                    target_module = layer.mlp.up_proj
                elif self.args.layer_type == 'attn_q':
                    target_module = layer.self_attn.q_proj
                elif self.args.layer_type == 'attn_k':
                    target_module = layer.self_attn.k_proj
                elif self.args.layer_type == 'attn_v':
                    target_module = layer.self_attn.v_proj
                else:
                    raise ValueError(f"Unknown layer type: {self.args.layer_type}")
                
                hook_handle = target_module.register_forward_hook(get_hook(layer_idx))
                self.hooks.append(hook_handle)
            
            self.logger.info(f"✓ {len(self.hooks)} hooks registered")
            
        except Exception as e:
            self.logger.error(f"Failed to register hooks: {str(e)}", exc_info=True)
            raise
    
    def collect_activations(self):
        """
        안전 데이터에 대해 모델을 실행하고 활성화 수집
        
        Log:
        - 배치별 처리 상황
        - 각 레이어의 활성화 수집 통계
        - 활성화 형태 및 크기
        """
        try:
            self.logger.info("Collecting activations from safety data...")
            
            with torch.no_grad():
                progress_bar = tqdm(
                    self.dataloader,
                    desc="Collecting activations",
                    disable=not self.args.debug
                )
                
                for batch_idx, batch in enumerate(progress_bar):
                    # harmful_prompt를 입력으로 사용
                    prompts = batch['harmful_prompt']
                    
                    # 토큰화
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # GPU로 이동
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # 모델 실행 (활성화는 훅에서 수집됨)
                    _ = self.model(**inputs)
                    
                    # 통계 업데이트
                    batch_size = len(prompts)
                    seq_len = inputs['input_ids'].shape[1]
                    self.stats['total_samples'] += batch_size
                    self.stats['total_tokens'] += batch_size * seq_len
                    
                    if self.args.debug and batch_idx < 2:
                        self.logger.debug(f"Batch {batch_idx}: processed {batch_size} samples, {seq_len} tokens")
            
            self.logger.info(f"✓ Activation collection completed")
            self.logger.info(f"  - Total samples: {self.stats['total_samples']}")
            self.logger.info(f"  - Total tokens: {self.stats['total_tokens']}")
            
            # 각 레이어의 활성화 통계
            self.logger.info(f"  - Layers with activations: {len(self.activations)}")
            for layer_idx in sorted(self.activations.keys())[:3]:  # 처음 3개만 출력
                act_list = self.activations[layer_idx]
                total_act = torch.cat(act_list, dim=0)
                self.logger.info(f"    - Layer {layer_idx}: {len(act_list)} batches, shape={total_act.shape}, dtype={total_act.dtype}")
            
            if len(self.activations) > 3:
                self.logger.info(f"    - ... and {len(self.activations) - 3} more layers")
            
        except Exception as e:
            self.logger.error(f"Failed to collect activations: {str(e)}", exc_info=True)
            raise
    
    def compute_svd(self):
        """
        수집된 활성화로부터 공분산 행렬을 계산하고 SVD 분해
        
        Log:
        - SVD 계산 진행상황
        - 각 레이어의 특이값(singular values)
        - 재구성 오류
        
        수식:
        - Φ Φ^T = U Σ U^T (공분산의 SVD)
        - U는 basis 벡터
        """
        try:
            self.logger.info("Computing SVD decomposition...")
            
            for layer_idx in sorted(self.activations.keys()):
                # 활성화를 (num_tokens, hidden_dim)으로 평탄화
                act_list = self.activations[layer_idx]
                activations_tensor = torch.cat(act_list, dim=0)  # (total_tokens, hidden_dim)
                
                # 평탄화: (total_tokens, hidden_dim) -> (total_tokens, hidden_dim)
                batch_size, seq_len, hidden_dim = activations_tensor.shape if activations_tensor.dim() == 3 else (activations_tensor.shape[0], 1, activations_tensor.shape[1])
                
                if activations_tensor.dim() == 3:
                    # (batch, seq, hidden) -> (batch*seq, hidden)
                    activations_flat = activations_tensor.reshape(-1, activations_tensor.shape[-1])
                else:
                    activations_flat = activations_tensor
                
                self.logger.info(f"Layer {layer_idx}:")
                self.logger.info(f"  - Activation shape: {activations_flat.shape}")
                self.logger.info(f"  - Activation dtype: {activations_flat.dtype}")
                self.logger.info(f"  - Activation device: {activations_flat.device}")
                
                # 공분산 행렬 계산: Φ Φ^T
                # 정규화 (mean centering)
                activations_centered = activations_flat - activations_flat.mean(dim=0, keepdim=True)
                
                # 공분산: (hidden_dim, hidden_dim)
                cov_matrix = (activations_centered.T @ activations_centered) / (activations_centered.shape[0] - 1)
                self.covariance_matrices[layer_idx] = cov_matrix
                
                self.logger.debug(f"  - Covariance shape: {cov_matrix.shape}")
                self.logger.debug(f"  - Covariance trace: {cov_matrix.trace():.4f}")
                
                # SVD 분해: U, S, V^T
                U, S, Vh = torch.linalg.svd(cov_matrix, full_matrices=False)
                
                self.svd_results[layer_idx] = {
                    'U': U,
                    'S': S,
                    'Vh': Vh,
                    'cov': cov_matrix
                }
                
                # 통계
                total_var = S.sum().item()
                top_k_var = S[:min(10, len(S))].sum().item()
                var_ratio = top_k_var / total_var * 100
                
                self.logger.info(f"  - SVD singular values (top-10):")
                for i, s in enumerate(S[:10]):
                    self.logger.info(f"    σ_{i}: {s:.6f}")
                self.logger.info(f"  - Top-10 variance ratio: {var_ratio:.2f}%")
                
                # 메모리 정리
                del activations_tensor, activations_centered, cov_matrix
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.logger.info(f"✓ SVD computation completed")
            self.logger.info(f"  - Total layers: {len(self.svd_results)}")
            
        except Exception as e:
            self.logger.error(f"Failed to compute SVD: {str(e)}", exc_info=True)
            raise
    
    def save_basis(self):
        """
        계산된 basis를 저장
        
        파일 구조:
        - basis/
          - layer_0_svd.pt (U, S, Vh)
          - layer_1_svd.pt
          - ...
          - metadata.json (통계, 설정)
        
        Log:
        - 저장 경로
        - 각 파일 크기
        """
        try:
            basis_dir = os.path.join(self.args.checkpoint_dir, 'basis')
            os.makedirs(basis_dir, exist_ok=True)
            
            self.logger.info(f"Saving basis to {basis_dir}...")
            
            # SVD 결과 저장
            for layer_idx, svd_data in self.svd_results.items():
                save_path = os.path.join(basis_dir, f'layer_{layer_idx:02d}_svd.pt')
                
                torch.save({
                    'U': svd_data['U'].cpu(),
                    'S': svd_data['S'].cpu(),
                    'Vh': svd_data['Vh'].cpu(),
                }, save_path)
                
                file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                self.logger.debug(f"  - Layer {layer_idx}: {save_path} ({file_size_mb:.2f} MB)")
            
            # 메타데이터 저장
            metadata = {
                'model_name': self.args.model_name,
                'layer_type': self.args.layer_type,
                'target_layers': self.args.target_layers,
                'num_layers': len(self.svd_results),
                'safety_samples': self.args.safety_samples,
                'batch_size': self.args.batch_size,
                'total_tokens': self.stats['total_tokens'],
                'total_samples': self.stats['total_samples'],
                'dtype': self.args.dtype,
            }
            
            metadata_path = os.path.join(basis_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"✓ Basis saved successfully")
            self.logger.info(f"  - Directory: {basis_dir}")
            self.logger.info(f"  - Files: {len(self.svd_results)} SVD files + metadata.json")
            self.logger.info(f"  - Metadata: {metadata_path}")
            
            return basis_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save basis: {str(e)}", exc_info=True)
            raise
        finally:
            # 훅 제거
            for hook in self.hooks:
                hook.remove()
            self.logger.info(f"✓ Forward hooks removed ({len(self.hooks)} hooks)")
