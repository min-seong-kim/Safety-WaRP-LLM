"""
Phase 1: Basis Construction
FFN down_proj 레이어에서 활성화를 수집하고 SVD를 통해 basis를 계산

python train.py \
    --phase 1 \
    --safety_dataset harmful_prompts \
    --harmful_prompts_path ./data/harmful_prompts_200.txt \
    --batch_size 2 \
    --layer_type 'attn_q,attn_k,attn_v,ffn_down,ffn_up' \
    --target_layers 'all' \
    --output_dir ./checkpoints

python train.py \
    --phase 1 \
    --safety_dataset do-not-answer \
    --dna_samples 200 \
    --batch_size 4 \
    --layer_type 'attn_q,attn_k,attn_v,ffn_down,ffn_up' \
    --target_layers 'all' \
    --output_dir ./checkpoints
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
    1. 모델 로드 
    2. 안전 데이터 로드 
    3. Forward hook을 통해 각 layer의 입력값 수집
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
        self.attention_masks = []  # attention mask 저장 (padding 마스킹용)
        self.covariance_matrices = {}  # layer_idx -> 공분산 행렬
        # ⚠️ 메모리 최적화: self.svd_results를 제거 (메모리에 저장 안 함)
        # 대신 각 layer별로 계산 후 즉시 디스크에 저장함
        
        # Checkpoint 디렉토리
        self.checkpoint_dir = None
        
        # Hook 핸들 저장소
        self.hooks = []
        
        # 통계
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'layers_processed': 0,
        }
    
    def _parse_target_layers(self, num_layers):
        """
        타겟 레이어 범위를 파싱하는 헬퍼 함수
        
        지원 형식:
        - 'all': 모든 레이어 
        - '31': 특정 레이어 (31번)
        - '30-31': 범위 (30-31)
        
        Returns:
            list: 레이어 인덱스 리스트
        """
        target = self.args.target_layers.strip()
        
        # 사전정의 범위
        if target == 'all':
            return list(range(num_layers))
        
        # 범위 파싱: "0-5" 또는 "30-31" 형식
        if '-' in target:
            try:
                start, end = target.split('-')
                start, end = int(start.strip()), int(end.strip())
                return list(range(start, min(end + 1, num_layers)))
            except ValueError:
                self.logger.error(f"Invalid range format: {target}. Use format like '0-5' or '30-31'")
                raise
        
        # 단일 레이어: "31" 형식
        try:
            layer_idx = int(target)
            if 0 <= layer_idx < num_layers:
                return [layer_idx]
            else:
                self.logger.error(f"Layer index {layer_idx} out of range [0, {num_layers-1}]")
                raise ValueError(f"Invalid layer index: {layer_idx}")
        except ValueError:
            self.logger.error(f"Invalid target_layers format: {target}")
            raise
    
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
            self.logger.info(f"    - Self-Attention q_proj: {sample_layer.self_attn.q_proj}")
            self.logger.info(f"    - Self-Attention k_proj: {sample_layer.self_attn.k_proj}")
            self.logger.info(f"    - Self-Attention v_proj: {sample_layer.self_attn.v_proj}")
            
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
        안전 데이터 로드 (harmful_prompts_200.txt)
        
        Log:
        - 데이터셋 로드 상태
        - 필터링 결과
        - 최종 샘플 수
        """
        try:
            # safety_dataset 인자에 따라 로더 선택
            safety_dataset = getattr(self.args, 'safety_dataset', 'harmful_prompts')
            
            if safety_dataset == 'do-not-answer':
                self._load_do_not_answer()
            else:
                self._load_harmful_prompts()
            
        except Exception as e:
            self.logger.error(f"Failed to load safety data: {str(e)}", exc_info=True)
            raise
    
    def _load_harmful_prompts(self):
        """
        harmful_prompts_200.txt 로드 (Phase 1용)
        """
        harmful_prompts_path = self.args.harmful_prompts_path
        self.logger.info(f"Loading harmful prompts from {harmful_prompts_path}...")
        
        # 텍스트 파일 로드
        with open(harmful_prompts_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"✓ Loaded {len(prompts)} harmful prompts")
        
        # 데이터셋 클래스
        class HarmfulPromptsDataset(torch.utils.data.Dataset):
            def __init__(self, prompts, tokenizer, max_length=512):
                self.prompts = prompts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.prompts)
            
            def __getitem__(self, idx):
                prompt = self.prompts[idx]
                
                encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                }
        
        dataset = HarmfulPromptsDataset(prompts, self.tokenizer, max_length=512)
        
        # Custom collate function
        def collate_fn(batch):
            max_len = max(len(item['input_ids']) for item in batch)
            
            input_ids_list = []
            attention_masks_list = []
            
            for item in batch:
                input_ids = item['input_ids']
                attn_mask = item['attention_mask']
                
                pad_len = max_len - len(input_ids)
                if pad_len > 0:
                    input_ids = torch.nn.functional.pad(
                        input_ids.unsqueeze(0),
                        (0, pad_len),
                        value=self.tokenizer.pad_token_id
                    ).squeeze(0)
                    attn_mask = torch.nn.functional.pad(
                        attn_mask.unsqueeze(0),
                        (0, pad_len),
                        value=0
                    ).squeeze(0)
                
                input_ids_list.append(input_ids)
                attention_masks_list.append(attn_mask)
            
            return {
                'input_ids': torch.stack(input_ids_list),
                'attention_mask': torch.stack(attention_masks_list),
            }
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        self.logger.info(f"✓ Dataloader created")
        self.logger.info(f"  - Batch size: {self.args.batch_size}")
        self.logger.info(f"  - Total batches: {len(self.dataloader)}")
        self.logger.info(f"  - Sample prompt: {prompts[0][:100]}...")
    
    def _load_do_not_answer(self):
        """
        Do-Not-Answer dataset 로드 (HuggingFace)
        
        Dataset: https://huggingface.co/datasets/LibrAI/do-not-answer
        
        Attributes:
            - harmful_question: 위험한 질문
            - safe_response: 안전한 거부 응답
        """
        from datasets import load_dataset
        
        self.logger.info("Loading do-not-answer dataset from HuggingFace...")
        
        try:
            # HuggingFace에서 do-not-answer 데이터셋 로드
            dataset = load_dataset('LibrAI/do-not-answer', split='train')
            self.logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # 샘플 수 제한
            max_samples = getattr(self.args, 'dna_samples', 200)
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                self.logger.info(f"✓ Subsampled to {len(dataset)} samples")
            
            # 데이터셋 클래스
            class DoNotAnswerDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, tokenizer, max_length=512):
                    self.dataset = dataset
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    
                    # 다양한 필드명 지원
                    question = item.get('harmful_question') or item.get('question') or item.get('prompt', '')
                    
                    encoding = self.tokenizer(
                        question,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                    }
            
            dataset_wrapper = DoNotAnswerDataset(dataset, self.tokenizer, max_length=512)
            
            # Custom collate function
            def collate_fn(batch):
                max_len = max(len(item['input_ids']) for item in batch)
                
                input_ids_list = []
                attention_masks_list = []
                
                for item in batch:
                    input_ids = item['input_ids']
                    attn_mask = item['attention_mask']
                    
                    pad_len = max_len - len(input_ids)
                    if pad_len > 0:
                        input_ids = torch.nn.functional.pad(
                            input_ids.unsqueeze(0),
                            (0, pad_len),
                            value=self.tokenizer.pad_token_id
                        ).squeeze(0)
                        attn_mask = torch.nn.functional.pad(
                            attn_mask.unsqueeze(0),
                            (0, pad_len),
                            value=0
                        ).squeeze(0)
                    
                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attn_mask)
                
                return {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }
            
            self.dataloader = torch.utils.data.DataLoader(
                dataset_wrapper,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            self.logger.info(f"✓ Dataloader created")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Total batches: {len(self.dataloader)}")
            self.logger.info(f"  - Sample question: {dataset[0].get('harmful_question', 'N/A')[:100]}...")
        
        except Exception as e:
            self.logger.error(f"Failed to load do-not-answer dataset: {str(e)}", exc_info=True)
            raise
    
    def register_activation_hooks(self):
        """
        Forward hook 등록
        FFN down_proj의 입력값(활성화)을 수집하기 위한 hook 등록
        
        Log:
        - 훅 등록된 레이어 수
        - 각 레이어의 모듈 정보
        """
        def get_hook(layer_idx, layer_type):
            """특정 레이어와 layer_type을 위한 훅 함수 생성"""
            def hook(module, input, output):
                # input[0]: (batch_size, seq_len, hidden_dim)
                activation = input[0].detach()
                # 이게 얕은 복사로 이후에 Activation 값들을 저장 가능 
                key = (layer_idx, layer_type)
                if key not in self.activations:
                    self.activations[key] = []
                
                self.activations[key].append(activation)
            
            return hook
        
        try:
            # 타겟 레이어 범위 결정
            num_layers = len(self.model.model.layers)
            layer_indices = self._parse_target_layers(num_layers)
            
            # Phase 1에서 사용할 layer_type 파싱
            # args.layer_type은 쉼표로 구분된 문자열 (예: 'ffn_down,ffn_up,attn_q,attn_k,attn_v')
            layer_types_str = self.args.layer_type
            layer_types = [lt.strip() for lt in layer_types_str.split(',')]
            
            self.logger.info(f"Registering hooks for layer types: {layer_types}")
            self.logger.info(f"Target layer indices: {layer_indices}")
            
            # 각 레이어와 layer_type 조합에 훅 등록
            for layer_idx in layer_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in layer_types:
                    # AutoModelForCausalLM -> 각 transformer -> layer 접근
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
                    
                    hook_handle = target_module.register_forward_hook(get_hook(layer_idx, layer_type))
                    self.hooks.append(hook_handle)
            
            self.logger.info(f"✓ {len(self.hooks)} hooks registered for {len(layer_types)} layer types × {len(layer_indices)} layers")
            
        except Exception as e:
            self.logger.error(f"Failed to register hooks: {str(e)}", exc_info=True)
            raise
    
    def collect_activations(self):
        """
        안전 데이터에 대해 모델을 실행하고 활성화 수집 (패딩 마스크 적용)
        
        Log:
        - 배치별 처리 상황
        - 각 레이어의 활성화 수집 통계
        - 활성화 형태 및 크기
        - 유효 토큰 수 (패딩 제외)
        """
        try:
            self.logger.info("Collecting activations from safety data...")
            
            # Activation 값을 수집하면서 attention mask도 저장
            with torch.no_grad():
                progress_bar = tqdm(
                    self.dataloader,
                    desc="Collecting activations",
                    disable=not self.args.debug
                )
                
                for batch_idx, batch in enumerate(progress_bar):
                    # collate_fn에서 이미 tokenize되고 padding된 데이터 사용
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    # attention_mask 저장 (패딩 마스킹용, 이후 activation의 공분산 구할때 padding 제거에 사용)
                    self.attention_masks.append(attention_mask)
                    
                    # GPU로 이동
                    input_ids = input_ids.to(self.model.device)
                    attention_mask = attention_mask.to(self.model.device)
                    
                    # 모델 실행 (활성화는 훅에서 수집됨)
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # 통계 업데이트
                    batch_size = input_ids.shape[0]
                    seq_len = input_ids.shape[1]
                    valid_tokens = attention_mask.sum().item()  # 유효 토큰 수
                    self.stats['total_samples'] += batch_size
                    self.stats['total_tokens'] += valid_tokens
                    
                    if self.args.debug and batch_idx < 2:
                        self.logger.debug(f"Batch {batch_idx}: {batch_size} samples, seq_len={seq_len}, valid_tokens={valid_tokens}")
            
            self.logger.info(f"✓ Activation collection completed")
            self.logger.info(f"  - Total samples: {self.stats['total_samples']}")
            self.logger.info(f"  - Total valid tokens (after masking): {self.stats['total_tokens']}")
            
            # 각 레이어의 활성화 통계
            self.logger.info(f"  - Layers with activations: {len(self.activations)}")
            for layer_idx in sorted(self.activations.keys())[:3]:  # 처음 3개만 출력
                act_list = self.activations[layer_idx]
                self.logger.info(f"    - Layer {layer_idx}: {len(act_list)} batches collected (masking will be applied in SVD step)")
            
            if len(self.activations) > 3:
                self.logger.info(f"    - ... and {len(self.activations) - 3} more layers")
            
            # 메모리 최적화: GPU에 있는 활성화를 CPU로 이동
            self.logger.info(f"\n[Memory Optimization] Moving activations from GPU to CPU...")
            for key in list(self.activations.keys()):
                self.activations[key] = [act.cpu() for act in self.activations[key]]
            
            # attention_masks도 CPU로 이동
            self.attention_masks = [mask.cpu() for mask in self.attention_masks]
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            self.logger.info(f"✓ Activations moved to CPU (GPU memory freed for SVD computation)")
            
        except Exception as e:
            self.logger.error(f"Failed to collect activations: {str(e)}", exc_info=True)
            raise
    
    def compute_svd(self):
        """
        수집된 활성화로부터 공분산 행렬을 계산하고 SVD 분해 (패딩 마스킹 적용)
        
        핵심: attention_mask를 사용하여 패딩된 토큰의 활성화를 제거하고 
        유효한 토큰의 활성화만으로 공분산과 SVD를 계산
        
        Log:
        - SVD 계산 진행상황
        - 각 레이어의 특이값(singular values)
        - 유효 토큰 수 (패딩 제외)
        
        수식:
        - Φ (filtered by attention_mask) = valid activations
        - Cov = Φ^T Φ / (N-1)
        - Cov = U Σ U^T (SVD 분해)
        """
        try:
            self.logger.info("Computing SVD decomposition (with attention mask filtering)...")
            self.logger.info("⚠️  Memory optimization: Saving each layer immediately after SVD (not keeping in memory)")
            
            # tqdm 진행바 추가 (disable=True로 설정하여 broken pipe 오류 방지)
            activation_keys = sorted(self.activations.keys())  # (layer_idx, layer_type) 튜플들
            pbar = tqdm(activation_keys, desc="SVD Decomposition", disable=True)
            
            layers_saved = 0
            
            for layer_idx, layer_type in pbar:
                pbar.set_description(f"SVD Decomposition [Layer {layer_idx} {layer_type}/{len(activation_keys)-1}]")
                
                # 활성화를 (num_tokens, hidden_dim)으로 평탄화
                act_list = self.activations[(layer_idx, layer_type)]
                
                # 핵심: attention mask를 사용하여 유효한 활성화만 추출
                valid_activations = []
                
                for batch_idx, act in enumerate(act_list):
                    # act: (batch, seq_len, hidden_dim)
                    # mask: (batch, seq_len)
                    batch_size, seq_len, hidden_dim = act.shape
                    mask = self.attention_masks[batch_idx].to(act.device)  # attention mask 로드
                    
                    # act를 (batch*seq_len, hidden_dim)로 reshape하고 mask 적용
                    act_reshaped = act.reshape(batch_size * seq_len, hidden_dim)  # (batch*seq_len, hidden_dim)
                    mask_flat = mask.reshape(-1)  # (batch*seq_len,)
                    
                    # 유효한 위치만 선택 (attention_mask == 1인 부분)
                    valid_act = act_reshaped[mask_flat == 1]  # (valid_tokens, hidden_dim)
                    
                    valid_activations.append(valid_act)
                
                # 유효한 활성화들만 결합
                activations_flat = torch.cat(valid_activations, dim=0)  # (total_valid_tokens, hidden_dim)
                
                self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                self.logger.info(f"  - Activation shape (after masking): {activations_flat.shape}")
                self.logger.info(f"  - Activation dtype: {activations_flat.dtype}")
                self.logger.info(f"  - Activation device: {activations_flat.device}")
                
                # 공분산 행렬 계산
                # bfloat16에서 SVD가 지원되지 않으므로 float32로 변환
                activations_float32 = activations_flat.float()
                
                # 정규화 (mean centering) 
                # 공분산 구하기 위해 평균 제거 
                activations_centered = activations_float32 - activations_float32.mean(dim=0, keepdim=True)
                
                # 공분산: (hidden_dim, hidden_dim)
                # unbiased estimator: 분모를 (N-1)로 설정
                cov_matrix = (activations_centered.T @ activations_centered) / max(activations_centered.shape[0] - 1, 1)
                self.covariance_matrices[(layer_idx, layer_type)] = cov_matrix
                
                self.logger.info(f"  - Covariance shape: {cov_matrix.shape}")
                self.logger.info(f"  - Covariance trace: {cov_matrix.trace():.4f}")
                self.logger.info(f"  - Covariance dtype: {cov_matrix.dtype}")
                
                # SVD 분해: U, S, V^T (float32에서 실행)
                U, S, Vh = torch.linalg.svd(cov_matrix, full_matrices=False)
                
                # 메모리 최적화: 즉시 저장하고 메모리에서 삭제
                svd_result = {
                    'U': U.cpu(),  # GPU → CPU 이동 후 저장
                    'S': S.cpu(),
                    'Vh': Vh.cpu(),
                    'cov': cov_matrix.cpu()
                }
                
                # Layer별로 즉시 저장 (메모리 효율적)
                self._save_svd_result(layer_idx, layer_type, svd_result)
                layers_saved += 1
                
                # 통계
                total_var = S.sum().item()
                top_k_var = S[:min(10, len(S))].sum().item()
                var_ratio = top_k_var / total_var * 100
                
                self.logger.info(f"  - SVD singular values (top-10):")
                for i, s in enumerate(S[:10]):
                    self.logger.info(f"    σ_{i}: {s:.6f}")
                self.logger.info(f"  - Top-10 variance ratio: {var_ratio:.2f}%")
                self.logger.info(f"  - ✓ Saved to disk (memory freed)")
                
                # 메모리 정리 (매우 중요!)
                del activations_flat, activations_float32, activations_centered, cov_matrix, U, S, Vh, svd_result, valid_activations
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.logger.info(f"✓ SVD computation and incremental saving completed")
            self.logger.info(f"  - Total layers saved: {layers_saved}")
            
            # 활성화 데이터도 더 이상 필요 없으므로 삭제
            self.activations.clear()
            self.attention_masks.clear()
            torch.cuda.empty_cache()
            self.logger.info(f"✓ Activation data cleared from memory")
            
        except Exception as e:
            self.logger.error(f"Failed to compute SVD: {str(e)}", exc_info=True)
            raise
    
    def _save_svd_result(self, layer_idx, layer_type, svd_result):
        """
        개별 layer의 SVD 결과를 즉시 디스크에 저장
        
        메모리 최적화: 메모리에 보관하지 않고 즉시 저장하므로 전체 메모리 사용량 대폭 감소
        
        Args:
            layer_idx: 레이어 인덱스
            layer_type: 레이어 타입 (attn_q, attn_k, etc)
            svd_result: {'U': U, 'S': S, 'Vh': Vh, 'cov': cov_matrix}
        """
        # Checkpoint 디렉토리 결정
        if self.checkpoint_dir is None:
            basis_dir = os.path.join(self.args.checkpoint_dir, 'basis')
            self.checkpoint_dir = basis_dir
        
        # Layer type별 디렉토리 생성
        layer_type_dir = os.path.join(self.checkpoint_dir, layer_type)
        os.makedirs(layer_type_dir, exist_ok=True)
        
        # SVD 결과 저장
        save_path = os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_svd.pt')
        
        torch.save({
            'U': svd_result['U'].cpu() if torch.is_tensor(svd_result['U']) else svd_result['U'],
            'S': svd_result['S'].cpu() if torch.is_tensor(svd_result['S']) else svd_result['S'],
            'Vh': svd_result['Vh'].cpu() if torch.is_tensor(svd_result['Vh']) else svd_result['Vh'],
        }, save_path)
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        self.logger.debug(f"    - Saved: {save_path} ({file_size_mb:.2f} MB)")
    
    def save_basis(self):
        """
        메모리 최적화 버전:
        
        각 layer별 SVD 결과가 이미 디스크에 저장되어 있음
        여기서는 메타데이터만 저장하면 됨
        
        파일 구조:
        - basis/
          - ffn_down/
            - layer_00_svd.pt (이미 저장됨)
            - layer_01_svd.pt
            - ...
          - ffn_up/
          - attn_q/
          - ...
          - metadata.json ← 여기서 생성
        """
        try:
            basis_dir = self.checkpoint_dir or os.path.join(self.args.checkpoint_dir, 'basis')
            os.makedirs(basis_dir, exist_ok=True)
            
            self.logger.info(f"Saving basis metadata to {basis_dir}...")
            
            # 저장된 레이어 타입과 파일 개수 세기
            layer_types_saved = set()
            total_files = 0
            
            for layer_type in os.listdir(basis_dir):
                layer_type_dir = os.path.join(basis_dir, layer_type)
                if os.path.isdir(layer_type_dir):
                    layer_types_saved.add(layer_type)
                    files = [f for f in os.listdir(layer_type_dir) if f.endswith('.pt')]
                    total_files += len(files)
            
            # 메타데이터 저장
            metadata = {
                'model_name': self.args.model_name,
                'layer_types': sorted(list(layer_types_saved)),
                'target_layers': self.args.target_layers,
                'num_layers_saved': total_files,
                'harmful_prompts_path': self.args.harmful_prompts_path,
                'batch_size': self.args.batch_size,
                'total_tokens': self.stats['total_tokens'],
                'total_samples': self.stats['total_samples'],
                'dtype': self.args.dtype,
                'memory_optimization': 'Layer-wise incremental saving (each layer saved immediately after SVD)',
                'notes': 'Phase 1: Activations filtered using attention_mask (padding tokens excluded). SVD results saved to disk immediately (not in memory).',
            }
            
            metadata_path = os.path.join(basis_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"✓ Basis metadata saved successfully")
            self.logger.info(f"  - Directory: {basis_dir}")
            self.logger.info(f"  - Layer types: {sorted(list(layer_types_saved))}")
            self.logger.info(f"  - Total SVD files: {total_files}")
            self.logger.info(f"  - Metadata: {metadata_path}")

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
