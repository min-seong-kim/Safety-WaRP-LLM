"""
Phase 2: Importance Scoring (Fixed - 원본 WaRP 방식)

✅ 원본 FSCIL-WaRP의 identify_importance()와 동일한 방식

핵심 차이점:
1. ❌ 이전: model.train() + optimizer.step() (파라미터 업데이트)
2. ✅ 수정: model.eval() + gradient만 계산 (파라미터 불변)

절차:
1. Phase 0에서 학습된 모델 로드
2. WaRP 모듈로 레이어 변환 (switch_to_warp_module)
3. Phase 1의 basis 로드
4. 가중치를 basis 공간으로 재매개변수화
5. model.eval() 모드에서 안전 데이터로 gradient 계산
6. Importance 점수 = |∂L/∂basis_coeff| 누적
7. 상위 keep_ratio만 동결할 마스크 생성
8. 마스크 저장

⚠️ optimizer.step() 없음 → 파라미터 변경 안됨!
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from .warp_modules import switch_to_warp_module, WaRPModule

logger = logging.getLogger(__name__)


class Phase2ImportanceScorer:
    """
    Phase 2: Importance Scoring (원본 WaRP 방식)
    
    목표: 안전 데이터로 중요도만 측정 (파라미터 업데이트 없음!)
    
    핵심:
    - model.eval() 모드
    - loss.backward()로 gradient 계산
    - ❌ optimizer.step() 없음!
    - importance = |∂L/∂basis_coeff| 누적
    """
    
    def __init__(self, args, logger, basis_dir, phase0_model_dir):
        self.args = args
        self.logger = logger
        self.basis_dir = basis_dir
        self.phase0_model_dir = phase0_model_dir
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.dataloader = None
        
        # Basis 정보
        self.basis_data = {}  # (layer_idx, layer_type) -> {'U': U}
        self.layer_types = []
        
        # Importance 점수
        self.importances = {}  # (layer_idx, layer_type) -> importance array
        self.masks = {}  # (layer_idx, layer_type) -> binary mask
        
        # 통계
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
        }
    
    def load_basis(self):
        """Phase 1에서 저장된 basis 로드"""
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")
            
            # 메타데이터
            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                basis_metadata = json.load(f)
            
            self.logger.info(f"✓ Metadata loaded:")
            self.logger.info(f"  - Layer types: {basis_metadata.get('layer_types')}")
            
            # Layer types 파싱
            layer_types_str = self.args.layer_type
            layer_types = [lt.strip() for lt in layer_types_str.split(',')]
            self.layer_types = layer_types
            
            # 각 layer_type별로 basis 로드
            total_loaded = 0
            
            for layer_type in layer_types:
                layer_type_dir = os.path.join(self.basis_dir, layer_type)
                if not os.path.exists(layer_type_dir):
                    self.logger.warning(f"Layer type directory not found: {layer_type_dir}")
                    continue
                
                # layer_XX_svd.pt 파일들 로드
                svd_files = sorted([
                    f for f in os.listdir(layer_type_dir) 
                    if f.startswith('layer_') and f.endswith('_svd.pt')
                ])
                
                for svd_file in svd_files:
                    layer_idx = int(svd_file.split('_')[1])
                    svd_path = os.path.join(layer_type_dir, svd_file)
                    
                    svd_data = torch.load(svd_path)
                    key = (layer_idx, layer_type)
                    self.basis_data[key] = {
                        'U': svd_data['U'],
                    }
                    total_loaded += 1
            
            self.logger.info(f"✓ Basis loaded: {total_loaded} (layer, type) combinations")
            
        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
            raise
    
    def load_model(self):
        """Phase 0에서 학습된 모델 로드"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            self.logger.info(f"Loading Phase 0 trained model from {self.phase0_model_dir}...")
            
            # 데이터 타입
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
            
            # Phase 0에서 학습된 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.phase0_model_dir,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True
            )
            
            self.logger.info(f"✓ Phase 0 model loaded (안전 데이터로 학습된 모델)")
            
            # 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.phase0_model_dir,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"✓ Tokenizer loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def load_safety_data(self):
        """안전 데이터 로드 (circuit_breakers)"""
        try:
            circuit_breakers_path = self.args.circuit_breakers_path
            self.logger.info(f"Loading circuit_breakers data from {circuit_breakers_path}...")
            
            with open(circuit_breakers_path, 'r', encoding='utf-8') as f:
                circuit_breakers_data = json.load(f)
            
            # 샘플 수 제한
            if self.args.circuit_breakers_samples > 0:
                circuit_breakers_data = circuit_breakers_data[:self.args.circuit_breakers_samples]
            
            self.logger.info(f"✓ Loaded {len(circuit_breakers_data)} samples")
            
            # 데이터셋
            class CircuitBreakersDataset(torch.utils.data.Dataset):
                def __init__(self, data, tokenizer, max_length=512):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    prompt = item.get('prompt', '')
                    response = item.get('llama3_output', '')
                    text = f"{prompt}\n{response}"
                    
                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                    }
            
            dataset = CircuitBreakersDataset(circuit_breakers_data, self.tokenizer)
            
            # Collate function
            def collate_fn(batch):
                max_len = max(len(item['input_ids']) for item in batch)
                
                input_ids_list = []
                attention_masks_list = []
                
                for item in batch:
                    input_ids = item['input_ids']
                    attention_mask = item['attention_mask']
                    
                    padding_length = max_len - len(input_ids)
                    if padding_length > 0:
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((padding_length,), self.tokenizer.pad_token_id)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length)
                        ])
                    
                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attention_mask)
                
                return {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }
            
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            self.logger.info(f"✓ Dataloader created ({len(self.dataloader)} batches)")
            
        except Exception as e:
            self.logger.error(f"Failed to load safety data: {str(e)}", exc_info=True)
            raise
    
    def convert_to_warp_modules(self):
        """모델 레이어를 WaRP 모듈로 변환"""
        try:
            self.logger.info("Converting to WaRP modules...")
            
            # switch_to_warp_module 사용
            self.model = switch_to_warp_module(
                self.model,
                self.layer_types,
                self.args.target_layers
            )
            
            self.logger.info("✓ Conversion completed")
            
        except Exception as e:
            self.logger.error(f"Failed to convert to WaRP modules: {str(e)}", exc_info=True)
            raise
    
    def reparameterize_weights(self):
        """
        가중치를 basis 공간으로 재매개변수화
        
        W_tilde = W @ U
        
        원본 WaRP:
        - basis_coeff = UT_backward @ W @ UT_forward.t()
        - UT_backward는 Identity
        - 따라서: basis_coeff = W @ U
        """
        try:
            self.logger.info("Reparameterizing weights to basis space...")
            self.logger.info("="*70)
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            reparameterized_count = 0
            
            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)
                    
                    if key not in self.basis_data:
                        continue
                    
                    # 타겟 모듈 (이미 WaRP 모듈로 변환됨)
                    target_module = self._get_target_module(layer, layer_type)
                    
                    if not isinstance(target_module, WaRPModule):
                        self.logger.warning(f"Layer {layer_idx} {layer_type}: Not a WaRP module!")
                        continue
                    
                    # 원본 가중치
                    W_original = target_module.weight.data.clone()
                    
                    # Basis 행렬 (U에는 V = UT.t()가 저장되어 있음)
                    # 원본 WaRP: basis_coeff = W @ UT_forward.t() = W @ V
                    U_matrix = self.basis_data[key]['U']  # 실제로는 V (= UT.t())
                    U_matrix = U_matrix.to(dtype=W_original.dtype, device=W_original.device)
                    
                    # basis_coeff 초기화: W @ V (원본 WaRP 방식)
                    basis_coeff_init = W_original @ U_matrix
                    
                    # WaRP 모듈에 설정
                    target_module.basis_coeff.data = basis_coeff_init
                    target_module.UT_forward = U_matrix.clone().detach()
                    target_module.UT_backward = torch.eye(
                        W_original.shape[0],
                        dtype=W_original.dtype,
                        device=W_original.device
                    )
                    
                    # Flag 설정 (WaRP 모드 활성화)
                    target_module.flag = True
                    
                    # Mask 초기화 (모두 0 = 모두 학습 가능)
                    target_module.coeff_mask.data = torch.zeros_like(target_module.basis_coeff)
                    
                    reparameterized_count += 1
                    
                    self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                    self.logger.info(f"  ✓ W_original: {W_original.shape}")
                    self.logger.info(f"  ✓ basis_coeff: {basis_coeff_init.shape}")
                    self.logger.info(f"  ✓ UT_forward (U): {U_matrix.shape}")
            
            self.logger.info(f"="*70)
            self.logger.info(f"✓ Reparameterization completed: {reparameterized_count} modules")
            
        except Exception as e:
            self.logger.error(f"Failed to reparameterize: {str(e)}", exc_info=True)
            raise
    
    def compute_importance(self):
        """
        Importance 점수 계산 (원본 FSCIL-WaRP 방식)
        
        ✅ 핵심:
        - model.eval() 모드
        - loss.backward()로 gradient 계산
        - ❌ optimizer.step() 없음!
        - importance = |∂L/∂basis_coeff| 누적
        
        원본 WaRP의 identify_importance()와 동일
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Phase 2: Importance Scoring (원본 WaRP 방식)")
            self.logger.info("="*70)
            self.logger.info("✅ model.eval() 모드 - 파라미터 업데이트 없음")
            self.logger.info("✅ Gradient만 계산하여 importance 측정")
            self.logger.info("="*70)
            
            # ✅ 원본 WaRP: model.eval()
            self.model.eval()
            self.logger.info("✓ Model set to eval mode (파라미터 변경 없음)")
            
            # Importance 저장소
            importances = OrderedDict()
            temp = OrderedDict()
            
            # WaRP 모듈들 수집
            warp_modules = []
            for module in self.model.modules():
                if isinstance(module, WaRPModule):
                    warp_modules.append(module)
                    # ✅ 원본 WaRP: mask 초기화
                    module.coeff_mask_prev = module.coeff_mask.data.clone()
                    module.coeff_mask.data = torch.zeros_like(module.coeff_mask)
            
            self.logger.info(f"✓ Found {len(warp_modules)} WaRP modules")
            
            # ✅ 원본 WaRP: Gradient 계산 루프
            progress_bar = tqdm(
                self.dataloader,
                desc="Computing importance (eval mode)",
                total=len(self.dataloader)
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                
                # Forward
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
                # Teacher forcing loss
                pred_logits = logits[:, :-1, :].contiguous()
                target_ids = input_ids[:, 1:].contiguous()
                attention_mask_shift = attention_mask[:, 1:].contiguous()
                
                valid_mask = (attention_mask_shift == 1) & (
                    target_ids != self.tokenizer.pad_token_id
                )
                pred_logits_flat = pred_logits[valid_mask]
                target_ids_flat = target_ids[valid_mask]
                
                if len(target_ids_flat) > 0:
                    # Loss 계산
                    loss = nn.CrossEntropyLoss()(pred_logits_flat, target_ids_flat)
                    
                    # ✅ 원본 WaRP: Backward (gradient 계산)
                    self.model.zero_grad()
                    loss.backward()
                    
                    # ✅ Gradient 절대값 수집 (bfloat16 → float32 → numpy)
                    for module in warp_modules:
                        if module.basis_coeff.grad is not None:
                            grad_abs = module.basis_coeff.grad.abs().detach().cpu().float().numpy()
                            temp[module] = grad_abs
                    
                    # ✅ Importance 누적
                    for module in warp_modules:
                        if module not in importances:
                            importances[module] = temp[module]
                        else:
                            importances[module] += temp[module]
                    
                    # ✅ ❌ optimizer.step() 없음! (파라미터 불변)
                    
                    # 통계
                    self.stats['total_samples'] += len(input_ids)
                    self.stats['total_tokens'] += len(target_ids_flat)
            
            self.logger.info("="*70)
            self.logger.info("✓ Importance computation completed")
            self.logger.info(f"  - Total samples: {self.stats['total_samples']}")
            self.logger.info(f"  - Total tokens: {self.stats['total_tokens']}")
            self.logger.info(f"  ⚠️  파라미터는 변경되지 않음 (optimizer.step 없음)")
            self.logger.info("="*70)
            
            # Importance를 (layer_idx, layer_type) 키로 저장
            self.importances = self._convert_importance_dict(importances)
            
        except Exception as e:
            self.logger.error(f"Failed to compute importance: {str(e)}", exc_info=True)
            raise
    
    def _convert_importance_dict(self, importances):
        """
        WaRP 모듈 기반 dict를 (layer_idx, layer_type) 키로 변환
        """
        result = {}
        
        target_indices = self._parse_target_layers(len(self.model.model.layers))
        
        for layer_idx in target_indices:
            layer = self.model.model.layers[layer_idx]
            
            for layer_type in self.layer_types:
                target_module = self._get_target_module(layer, layer_type)
                
                if isinstance(target_module, WaRPModule) and target_module in importances:
                    key = (layer_idx, layer_type)
                    result[key] = importances[target_module]
        
        return result
    
    def generate_masks(self, keep_ratio=0.1):
        """
        Importance 기반 마스크 생성
        
        원본 WaRP:
        - threshold = quantile(importances, 1 - keep_ratio)
        - mask[i] = 1 if importance[i] >= threshold else 0
        - mask=1: 동결, mask=0: 학습 가능
        """
        try:
            self.logger.info("="*70)
            self.logger.info(f"Generating masks (keep_ratio={keep_ratio})")
            self.logger.info("="*70)
            
            # 모든 importance 평탄화
            flat_importances = self._flatten_importances(self.importances)
            
            # Threshold 계산
            threshold = np.quantile(flat_importances, 1 - keep_ratio)
            
            self.logger.info(f"✓ Threshold calculated: {threshold:.6f}")
            self.logger.info(f"  - Quantile: {1 - keep_ratio} (상위 {keep_ratio*100}%)")
            
            # 마스크 생성
            for key in self.importances.keys():
                importance = self.importances[key]
                
                # mask = 1 if importance >= threshold else 0
                mask = (importance >= threshold).astype(np.float32)
                
                self.masks[key] = mask
                
                frozen_count = mask.sum()
                total_count = mask.size
                frozen_ratio = frozen_count / total_count * 100
                
                layer_idx, layer_type = key
                self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                self.logger.info(f"  - Frozen: {frozen_count}/{total_count} ({frozen_ratio:.2f}%)")
            
            self.logger.info("="*70)
            self.logger.info(f"✓ Masks generated for {len(self.masks)} modules")
            
        except Exception as e:
            self.logger.error(f"Failed to generate masks: {str(e)}", exc_info=True)
            raise
    
    def save_masks(self):
        """마스크 저장"""
        try:
            # Checkpoint 디렉토리
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f'phase2_{timestamp}',
                'checkpoints'
            )
            masks_dir = os.path.join(checkpoint_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            
            # Layer type별 subdirectory
            for key, mask in self.masks.items():
                layer_idx, layer_type = key
                
                layer_type_dir = os.path.join(masks_dir, layer_type)
                os.makedirs(layer_type_dir, exist_ok=True)
                
                mask_path = os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_mask.pt')
                torch.save({'mask': mask}, mask_path)
            
            # 메타데이터
            metadata = {
                'phase': 2,
                'keep_ratio': getattr(self.args, 'keep_ratio', 0.1),
                'layer_types': self.layer_types,
                'timestamp': timestamp,
            }
            
            metadata_path = os.path.join(masks_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✓ Masks saved to: {masks_dir}")
            self.logger.info(f"✓ Metadata saved to: {metadata_path}")
            
            return masks_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save masks: {str(e)}", exc_info=True)
            raise
    
    def _flatten_importances(self, importances):
        """Importance를 1D array로 평탄화"""
        return np.concatenate([
            params.flatten()
            for _, params in importances.items()
        ])
    
    def _get_target_module(self, layer, layer_type):
        """Layer type에 맞는 모듈 반환"""
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
        """타겟 레이어 파싱"""
        target = self.args.target_layers.strip()
        
        if target == 'all':
            return list(range(num_layers))
        elif '-' in target:
            start, end = map(int, target.split('-'))
            return list(range(start, end + 1))
        else:
            return [int(target)]
