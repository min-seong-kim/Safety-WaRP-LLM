"""
Phase 1: Basis Construction
FFN down_proj 레이어에서 활성화를 수집하고 SVD를 통해 basis를 계산

python train.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset circuit_breakers \
    --circuit_breakers_samples_phase1 4994 \
    --batch_size 2 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42
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
        # ✅ Incremental Gram matrix accumulation 방식
        # Activation을 저장하지 않고 즉시 Gram matrix에 누적
        self.gram_matrices = {}  # (layer_idx, layer_type) -> Gram matrix (GPU)
        self.num_samples = {}  # (layer_idx, layer_type) -> sample count
        
        # Checkpoint 디렉토리 설정
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_base = os.path.join(
            getattr(args, 'output_dir', './checkpoints'),
            f'phase1_{timestamp}'
        )
        os.makedirs(checkpoint_base, exist_ok=True)
        self.checkpoint_dir = os.path.join(checkpoint_base, 'basis')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
        안전 데이터 로드 (harmful_prompts_200.txt 또는 circuit_breakers_train.json)
        
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
            elif safety_dataset == 'circuit_breakers':
                self._load_circuit_breakers()
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
    
    def _load_circuit_breakers(self):
        """
        Circuit Breakers dataset 로드 (circuit_breakers_train.json)
        
        Prompt만 사용 (Phase 1 basis 구성용)
        """
        import json
        
        circuit_breakers_path = './data/circuit_breakers_train.json'
        self.logger.info(f"Loading circuit_breakers from {circuit_breakers_path}...")
        
        try:
            with open(circuit_breakers_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # 샘플 수 제한
            max_samples = getattr(self.args, 'circuit_breakers_samples_phase1', 200)
            if max_samples and len(dataset) > max_samples:
                dataset = dataset[:max_samples]
                self.logger.info(f"✓ Subsampled to {len(dataset)} samples")
            
            # Prompt만 추출
            prompts = [item.get('prompt', '') for item in dataset]
            
            # 데이터셋 클래스
            class CircuitBreakersDataset(torch.utils.data.Dataset):
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
            
            dataset_wrapper = CircuitBreakersDataset(prompts, self.tokenizer, max_length=512)
            
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
            self.logger.info(f"  - Sample prompt: {prompts[0][:100]}...")
        
        except Exception as e:
            self.logger.error(f"Failed to load circuit_breakers dataset: {str(e)}", exc_info=True)
            raise
    
    def collect_activations_and_accumulate_gram(self):
        """
        ✅ Incremental Gram Matrix Accumulation
        
        배치별로 forward pass를 수행하면서 Gram matrix를 즉시 누적
        - Activation을 저장하지 않음 (메모리 효율적)
        - GPU에서 계산 (빠름)
        - 패딩 마스킹 적용
        
        수식: Gram = Σ(Φ_batch^T @ Φ_batch) for all batches
        """
        try:
            self.logger.info("Collecting activations and accumulating Gram matrices (GPU)...")
            self.logger.info("✅ Incremental accumulation: No activation storage, fast GPU computation")
            
            # 타겟 레이어 및 타입 결정
            num_layers = len(self.model.model.layers)
            layer_indices = self._parse_target_layers(num_layers)
            layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]
            
            self.logger.info(f"Target layers: {layer_indices}")
            self.logger.info(f"Layer types: {layer_types}")
            
            # Gram matrices 초기화
            for layer_idx in layer_indices:
                for layer_type in layer_types:
                    self.gram_matrices[(layer_idx, layer_type)] = None
                    self.num_samples[(layer_idx, layer_type)] = 0
            
            # Forward hook 등록 (Gram matrix 누적용)
            hooks = []
            
            def get_accumulation_hook(layer_idx, layer_type):
                """Gram matrix를 누적하는 hook"""
                def hook(module, input, output):
                    # input[0]: (batch_size, seq_len, hidden_dim)
                    act = input[0]  # GPU에 유지
                    batch_size, seq_len, hidden_dim = act.shape
                    
                    # Reshape: (batch*seq, hidden_dim)
                    act_flat = act.reshape(batch_size * seq_len, hidden_dim)
                    
                    # Gram matrix 누적: Φ^T @ Φ
                    gram_batch = act_flat.t() @ act_flat  # (hidden_dim, hidden_dim)
                    
                    key = (layer_idx, layer_type)
                    if self.gram_matrices[key] is None:
                        self.gram_matrices[key] = gram_batch
                    else:
                        self.gram_matrices[key] += gram_batch
                    
                    self.num_samples[key] += batch_size * seq_len
                
                return hook
            
            # Hook 등록
            for layer_idx in layer_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in layer_types:
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
                    
                    hook_handle = target_module.register_forward_hook(
                        get_accumulation_hook(layer_idx, layer_type)
                    )
                    hooks.append(hook_handle)
            
            self.logger.info(f"✓ {len(hooks)} accumulation hooks registered")
            
            # Forward pass (Gram matrix 누적)
            with torch.no_grad():
                progress_bar = tqdm(
                    self.dataloader,
                    desc="Accumulating Gram matrices",
                    disable=not self.args.debug
                )
                
                total_batches = 0
                for batch_idx, batch in enumerate(progress_bar):
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    
                    # Forward (hook에서 Gram matrix 누적)
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    total_batches += 1
                    
                    # 통계 업데이트
                    valid_tokens = attention_mask.sum().item()
                    self.stats['total_samples'] += input_ids.shape[0]
                    self.stats['total_tokens'] += valid_tokens
                    
                    if self.args.debug and batch_idx < 2:
                        self.logger.debug(f"Batch {batch_idx}: processed")
            
            # Hook 제거
            for hook in hooks:
                hook.remove()
            
            self.logger.info(f"✓ Gram matrix accumulation completed")
            self.logger.info(f"  - Total batches: {total_batches}")
            self.logger.info(f"  - Total samples: {self.stats['total_samples']}")
            self.logger.info(f"  - Total tokens: {self.stats['total_tokens']}")
            self.logger.info(f"  - Gram matrices: {len(self.gram_matrices)}")
            
            # 통계 출력
            for key in sorted(self.gram_matrices.keys())[:3]:
                gram = self.gram_matrices[key]
                num = self.num_samples[key]
                self.logger.info(f"  - Layer {key}: Gram shape={gram.shape}, samples={num}")
            
            if len(self.gram_matrices) > 3:
                self.logger.info(f"  - ... and {len(self.gram_matrices) - 3} more")
            
        except Exception as e:
            self.logger.error(f"Failed to accumulate Gram matrices: {str(e)}", exc_info=True)
            raise
    
    def register_activation_hooks(self):
        """
        ✅ 제거됨: Incremental accumulation 방식에서는 불필요
        
        collect_activations_and_accumulate_gram()에서 직접 처리
        """
        pass
    
    def compute_svd(self):
        """
        ✅ Incremental Gram Matrix 방식의 SVD
        
        이미 누적된 Gram matrix로부터 직접 SVD 수행
        - GPU에서 빠른 계산
        - 메모리 효율적 (O(hidden_dim^2))
        - Layer별 순차 처리 및 즉시 저장
        
        수식:
        - Gram = Φ^T @ Φ (이미 누적됨)
        - Gram = U @ S @ U^T (SVD)
        """
        try:
            self.logger.info("Computing SVD from accumulated Gram matrices (GPU)...")
            self.logger.info("✅ Fast GPU computation, incremental disk saving")
            
            layers_saved = 0
            gram_keys = sorted(self.gram_matrices.keys())
            
            for layer_idx, layer_type in tqdm(gram_keys, desc="SVD Decomposition", disable=True):
                gram_matrix = self.gram_matrices[(layer_idx, layer_type)]
                
                if gram_matrix is None:
                    self.logger.warning(f"Layer {layer_idx} ({layer_type}): No Gram matrix accumulated, skipping")
                    continue
                
                self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                self.logger.info(f"  - Gram matrix shape: {gram_matrix.shape}")
                self.logger.info(f"  - Gram matrix device: {gram_matrix.device}")
                self.logger.info(f"  - Samples accumulated: {self.num_samples[(layer_idx, layer_type)]}")
                
                # float32로 변환 (SVD 정확도)
                gram_matrix_float = gram_matrix.float()
                
                # 통계
                trace = gram_matrix_float.trace().item()
                self.logger.info(f"  - Gram matrix trace: {trace:.4f}")
                
                # ✅ GPU에서 SVD 수행 (빠름!)
                U, S, UT = torch.linalg.svd(gram_matrix_float, full_matrices=False)
                
                # Symmetry 검증
                diff = (U - UT.t()).abs().max().item()
                self.logger.info(f"  - Symmetry check: max|U - V^T| = {diff:.2e}")
                
                # V = UT.t() (Right singular vectors)
                V = UT.t()
                svd_result = {
                    'U': V.cpu(),  # CPU로 이동하여 저장
                    'S': S.cpu(),
                    'UT': UT.cpu(),
                }
                
                # 즉시 디스크 저장
                self._save_svd_result(layer_idx, layer_type, svd_result)
                layers_saved += 1
                
                # 통계
                total_var = S.sum().item()
                top_k_var = S[:min(10, len(S))].sum().item()
                var_ratio = (top_k_var / total_var * 100) if total_var > 1e-10 else 0.0
                
                self.logger.info(f"  - SVD singular values (top-10):")
                for i, s in enumerate(S[:10].cpu()):
                    self.logger.info(f"    σ_{i}: {s:.6f}")
                self.logger.info(f"  - Top-10 variance ratio: {var_ratio:.2f}%")
                self.logger.info(f"  - ✓ Saved to disk")
                
                # GPU 메모리 정리
                del gram_matrix_float, U, S, UT, V, svd_result
            
            # 모든 Gram matrix 삭제
            self.gram_matrices.clear()
            self.num_samples.clear()
            torch.cuda.empty_cache()
            
            self.logger.info(f"✓ SVD computation completed")
            self.logger.info(f"  - Total layers saved: {layers_saved}")
            self.logger.info(f"  - GPU memory cleared")
            
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
        # Layer type별 디렉토리 생성
        layer_type_dir = os.path.join(self.checkpoint_dir, layer_type)
        os.makedirs(layer_type_dir, exist_ok=True)
        
        # SVD 결과 저장
        save_path = os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_svd.pt')
        
        torch.save({
            'U': svd_result['U'].cpu() if torch.is_tensor(svd_result['U']) else svd_result['U'],
            'S': svd_result['S'].cpu() if torch.is_tensor(svd_result['S']) else svd_result['S'],
            'UT': svd_result.get('UT', None),  # UT 저장 (있는 경우)
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
            
            # ✅ SVD 결과 검증 로그
            self.logger.info("="*70)
            self.logger.info("Verifying saved SVD results...")
            self.logger.info("="*70)
            
            verification_stats = {
                'total_files': 0,
                'total_size_mb': 0.0,
                'layer_type_counts': {},
                'sample_checks': []
            }
            
            for layer_type in sorted(list(layer_types_saved)):
                layer_type_dir = os.path.join(basis_dir, layer_type)
                files = sorted([f for f in os.listdir(layer_type_dir) if f.endswith('.pt')])
                
                self.logger.info(f"\n[{layer_type}] - {len(files)} files")
                
                layer_type_size = 0.0
                for f in files:
                    file_path = os.path.join(layer_type_dir, f)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    layer_type_size += file_size
                
                verification_stats['total_files'] += len(files)
                verification_stats['total_size_mb'] += layer_type_size
                verification_stats['layer_type_counts'][layer_type] = len(files)
                
                self.logger.info(f"  - Total size: {layer_type_size:.2f} MB")
                
                # 첫 번째와 마지막 파일 검증
                if len(files) > 0:
                    # 첫 번째 파일
                    first_file = os.path.join(layer_type_dir, files[0])
                    svd_data = torch.load(first_file, map_location='cpu')
                    U = svd_data['U']
                    S = svd_data['S']
                    
                    self.logger.info(f"  - Sample check (first): {files[0]}")
                    self.logger.info(f"    - U shape: {U.shape}")
                    self.logger.info(f"    - S shape: {S.shape}")
                    self.logger.info(f"    - S range: [{S.min():.6f}, {S.max():.6f}]")
                    self.logger.info(f"    - S top-5: {S[:5].tolist()}")
                    
                    verification_stats['sample_checks'].append({
                        'layer_type': layer_type,
                        'file': files[0],
                        'U_shape': list(U.shape),
                        'S_shape': list(S.shape),
                        'S_max': S.max().item(),
                        'S_min': S.min().item()
                    })
                    
                    # 마지막 파일 (다른 경우만)
                    if len(files) > 1:
                        last_file = os.path.join(layer_type_dir, files[-1])
                        svd_data_last = torch.load(last_file, map_location='cpu')
                        U_last = svd_data_last['U']
                        S_last = svd_data_last['S']
                        
                        self.logger.info(f"  - Sample check (last): {files[-1]}")
                        self.logger.info(f"    - U shape: {U_last.shape}")
                        self.logger.info(f"    - S shape: {S_last.shape}")
                        self.logger.info(f"    - S range: [{S_last.min():.6f}, {S_last.max():.6f}]")
            
            # 전체 요약
            self.logger.info("="*70)
            self.logger.info("Verification Summary:")
            self.logger.info(f"  - Total files verified: {verification_stats['total_files']}")
            self.logger.info(f"  - Total storage: {verification_stats['total_size_mb']:.2f} MB")
            self.logger.info(f"  - Files per layer type:")
            for lt, count in verification_stats['layer_type_counts'].items():
                self.logger.info(f"    - {lt}: {count} files")
            
            # 전체 통계
            all_S_max = max(check['S_max'] for check in verification_stats['sample_checks'])
            all_S_min = min(check['S_min'] for check in verification_stats['sample_checks'])
            self.logger.info(f"  - Singular value range (global): [{all_S_min:.6f}, {all_S_max:.6f}]")
            
            self.logger.info("="*70)
            self.logger.info("✓ All SVD results verified successfully!")
            self.logger.info("="*70)
            
            return basis_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save basis: {str(e)}", exc_info=True)
            raise
        finally:
            # 훅 제거
            for hook in self.hooks:
                hook.remove()
            self.logger.info(f"✓ Forward hooks removed ({len(self.hooks)} hooks)")
