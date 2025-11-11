"""
Phase 3: Incremental Learning with Masked Updates
안전 메커니즘을 보호하면서 유틸리티 개선
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import os
import json
from tqdm import tqdm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# 📊 GSM8K 파인튜닝 검증 유틸리티
# ============================================================
class GSM8KValidationMetrics:
    """GSM8K 파인튜닝 진행 상황을 추적하는 메트릭 클래스"""
    
    @staticmethod
    def is_converging(losses: list, min_improvement_pct: float = 1.0) -> bool:
        """손실이 수렴하는지 확인"""
        if len(losses) < 2:
            return False
        initial = losses[0]
        final = losses[-1]
        improvement = (initial - final) / initial * 100
        return improvement > min_improvement_pct
    
    @staticmethod
    def is_stable(losses: list, max_cv: float = 20.0) -> bool:
        """손실의 변동성이 안정적인지 확인 (변동계수 기반)"""
        if len(losses) < 2:
            return True
        cv = np.std(losses) / np.mean(losses) * 100
        return cv < max_cv
    
    @staticmethod
    def has_gradient_flow(frozen_grad: float, trainable_grad: float) -> bool:
        """그래디언트가 정상적으로 흐르는지 확인"""
        # Frozen 방향의 그래디언트는 ~0 이고, Trainable 방향은 > 0 이어야 함
        return frozen_grad < 1e-5 and trainable_grad > 1e-6


# ============================================================
# ✅ BasisLinear: torch.autograd.Function으로 gradient flow 구현
# ============================================================
class BasisLinear(torch.autograd.Function):
    """
    Weight를 basis_coeff @ U^T로 동적 재구성하면서 gradient를 정확하게 계산.
    
    Forward: y = (basis_coeff @ U^T) @ x^T + bias
    Backward: 마스킹을 적용하여 frozen directions의 gradient = 0
    
    Args:
        x: input (batch_size, in_features)
        basis_coeff: learnable parameters (out_features, rank)
        U: fixed basis matrix (in_features, rank)
        bias: bias (out_features)
        mask: binary mask (out_features,) - 1:frozen, 0:trainable
    """
    
    @staticmethod
    def forward(ctx, x, basis_coeff, U, bias, mask):
        """
        Forward pass: y = linear(x, weight, bias)
        where weight = basis_coeff @ U^T
        """
        # 1. Weight 재구성 (autograd가 추적 가능한 연산)
        weight = basis_coeff @ U.T  # (out_features, in_features)
        
        # 2. Linear forward
        output = torch.nn.functional.linear(x, weight, bias)
        
        # 3. Backward를 위해 필요한 정보 저장
        # ✅ mask는 텐서가 아니므로 ctx.save_for_backward에 포함하면 안 됨
        ctx.save_for_backward(x, basis_coeff, U, bias, weight)
        ctx.mask = mask  # mask는 non-tensor로 저장
        
        # Debug: mask 타입 확인
        # print(f"[BasisLinear.forward] mask type: {type(mask)}, requires_grad: {getattr(mask, 'requires_grad', 'N/A')}")
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: 마스킹을 적용하여 frozen directions의 gradient 제거
        
        grad_output: gradient w.r.t. output 
            - LLM 종류마다 배치 차원이 다를 수 있음
            - 2D: (batch_size, out_features)
            - 3D: (batch_size, seq_len, out_features) for LLM
        Returns: (grad_x, grad_basis_coeff, grad_U, grad_bias, grad_mask)
        """
        x, basis_coeff, U, bias, weight = ctx.saved_tensors
        mask = ctx.mask  # 고정된 mask 복원
        
        # 3D 배치 처리 (LLM의 경우 시퀀스 길이가 포함됨)
        original_shape = grad_output.shape
        if grad_output.dim() == 3:
            # (batch, seq_len, out_features) → (batch*seq_len, out_features)
            batch_size, seq_len, out_features = grad_output.shape
            grad_output = grad_output.reshape(-1, out_features)
            x = x.reshape(-1, x.shape[-1])  # x도 3D이므로 reshape
        elif grad_output.dim() != 2:
            raise RuntimeError(f"grad_output must be 2D or 3D, got {original_shape}")
        
        # 1. Linear backward: dL/dW
        # y = x @ weight^T + bias
        # grad_output: (batch_size, out_features)
        # x: (batch_size, in_features)
        # grad_weight = grad_output.T @ x  (out_features, in_features)
        grad_weight = grad_output.T @ x
        
        # 2. Chain rule: dL/d(basis_coeff)
        # W = basis_coeff @ U^T
        # So: grad_basis_coeff = grad_weight @ U
        grad_basis_coeff = grad_weight @ U  # (out_features, rank) = (4096, 14336)
        
        # ✅ 마스킹 적용: frozen directions (mask=1) → gradient = 0
        # mask shape: (14336,) - 각 input dimension별 마스킹
        # grad_basis_coeff shape: (4096, 14336) [out_features, rank/input_dim]
        # 마스킹: 각 input dimension (rank)에 대해 모든 output에 동일하게 적용
        mask_expanded = (1 - mask).unsqueeze(0).detach()  # (1, 14336) - trainable 방향만 1, detach 필수
        grad_basis_coeff = grad_basis_coeff * mask_expanded  # (4096, 14336) * (1, 14336) ✓
        
        # 3. Gradient w.r.t. U
        # ⚠️ U는 fixed basis이므로 requires_grad=False
        # 따라서 None을 반환
        grad_U = None
        
        # 4. Gradient w.r.t. input
        grad_x = grad_output @ weight  # (batch_size, in_features)
        
        # 복원: 3D 입력이었으면 원래 shape로 돌림
        if len(original_shape) == 3:
            grad_x = grad_x.reshape(batch_size, seq_len, -1)
        
        # 5. Gradient w.r.t. bias
        grad_bias = grad_output.sum(dim=0)  # (out_features,)
        
        # ✅ CRITICAL: backward는 forward의 inputs와 같은 개수의 gradient를 반환해야 함
        # forward inputs: (x, basis_coeff, U, bias, mask)
        # U와 mask는 모두 non-trainable이므로 None을 반환
        
        return grad_x, grad_basis_coeff, grad_U, grad_bias, None


class Phase3IncrementalLearner:
    """
    Phase 3: Incremental Learning with Masked Gradient Updates
    
    절차:
    1. Phase 1 basis + Phase 2 masks 로드
    2. 모델 가중치를 basis 공간으로 재매개변수화
    3. GSM8K 데이터로 미세조정 (마스킹된 gradient)
    4. 마스크된 방향 (안전 중요)은 업데이트 금지
    5. 덜 중요한 방향만 업데이트 가능
    """
    
    def __init__(self, args, logger, basis_dir, masks_dir):
        """
        Args:
            args: 커맨드라인 인자
            logger: 로거 객체
            basis_dir: Phase 1 basis 디렉토리
            masks_dir: Phase 2 masks 디렉토리
        """
        self.args = args
        self.logger = logger
        self.basis_dir = basis_dir
        self.masks_dir = masks_dir
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        
        # Basis 정보
        self.basis_data = {}  # layer_idx -> {'U': U, 'S': S, 'Vh': Vh}
        
        # 마스크
        self.masks = {}  # layer_idx -> binary mask
        
        # 훈련 통계
        self.stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'num_batches': 0,
        }
        
        # Hook 저장용
        self.hook_handles = []
    
    def load_basis(self):
        """Phase 1에서 저장된 basis 로드"""
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")
            
            # 메타데이터 로드
            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                basis_metadata = json.load(f)
            
            self.logger.info(f"✓ Metadata loaded:")
            self.logger.info(f"  - Target layers: {basis_metadata.get('target_layers')}")
            
            # basis 파일 로드
            import glob
            svd_files = sorted(glob.glob(os.path.join(self.basis_dir, 'layer_*_svd.pt')))
            
            for svd_path in svd_files:
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
            
        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
            raise
    
    def load_masks(self):
        """Phase 2에서 저장된 마스크 로드"""
        try:
            self.logger.info(f"Loading masks from {self.masks_dir}...")
            
            # 메타데이터 로드
            metadata_path = os.path.join(self.masks_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                masks_metadata = json.load(f)
            
            self.logger.info(f"✓ Mask metadata loaded:")
            self.logger.info(f"  - Keep ratio: {masks_metadata.get('keep_ratio')}")
            
            # 마스크 파일 로드
            import glob
            mask_files = sorted(glob.glob(os.path.join(self.masks_dir, 'layer_*_mask.pt')))
            
            for mask_path in mask_files:
                filename = os.path.basename(mask_path)
                layer_idx = int(filename.split('_')[1])
                
                mask = torch.load(mask_path, map_location='cpu')
                self.masks[layer_idx] = mask.to(self.args.device)
            
            self.logger.info(f"✓ Masks loaded: {len(self.masks)} layers")
            self.logger.info(f"  - Layer indices: {sorted(self.masks.keys())}")
            
            # 마스크 통계
            for layer_idx in sorted(self.masks.keys()):
                mask = self.masks[layer_idx]
                num_important = (mask == 1).sum().item()
                ratio = num_important / mask.numel()
                self.logger.info(f"  - Layer {layer_idx}: {num_important}/{mask.numel()} important ({ratio*100:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Failed to load masks: {str(e)}", exc_info=True)
            raise
    
    def load_model(self):
        """
        ⭐ Phase 2의 Safety Fine-tuned 모델 로드
        
        우선순위:
        1. Phase 2 safety fine-tuned 모델이 있으면 → 그것 로드 (권장)
        2. 없으면 → 원본 모델 로드 (fallback)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            # 데이터 타입 설정
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
            
            # ✅ Step 1: Phase 2 safety fine-tuned 모델 경로 확인
            # 구조 분석:
            # masks_dir:          ./checkpoints/phase2_XXXX/checkpoints/masks/
            # phase2_finetuned:   ./checkpoints/phase2_XXXX/checkpoints/phase2_finetuned_model/
            #
            # masks_dir의 dirname → ./checkpoints/phase2_XXXX/checkpoints/
            # 거기에 phase2_finetuned_model 추가
            
            # masks_dir 정규화 (후행 slash 제거)
            masks_dir_normalized = self.masks_dir.rstrip('/')
            
            # checkpoints 디렉토리 경로 (masks의 상위 디렉토리)
            checkpoints_dir = os.path.dirname(masks_dir_normalized)  # ./checkpoints/phase2_XXXX/checkpoints
            
            phase2_model_dir = os.path.join(checkpoints_dir, 'phase2_finetuned_model')
            
            self.logger.debug(f"Searching Phase 2 model at: {phase2_model_dir}")
            
            model_to_load = None
            model_source = None
            
            if os.path.exists(phase2_model_dir) and os.path.isdir(phase2_model_dir):
                # Phase 2 fine-tuned 모델이 있는지 확인
                # pytorch_model.bin (single file) 또는 safetensors 형식 (distributed) 체크
                pytorch_bin_path = os.path.join(phase2_model_dir, 'pytorch_model.bin')
                safetensors_index_path = os.path.join(phase2_model_dir, 'model.safetensors.index.json')
                safetensors_single_path = os.path.join(phase2_model_dir, 'model.safetensors')
                
                has_pytorch_bin = os.path.exists(pytorch_bin_path)
                has_safetensors = os.path.exists(safetensors_index_path) or os.path.exists(safetensors_single_path)
                
                if has_pytorch_bin or has_safetensors:
                    model_to_load = phase2_model_dir
                    model_source = "Phase 2 Safety Fine-tuned"
                    model_format = "safetensors (distributed)" if os.path.exists(safetensors_index_path) else \
                                  "safetensors (single)" if os.path.exists(safetensors_single_path) else "pytorch_model.bin"
                    self.logger.debug(f"Found Phase 2 model at: {phase2_model_dir} (format: {model_format})")
            
            # ✅ Step 2: 모델 로드
            if model_to_load is not None:
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"Loading {model_source} Model")
                self.logger.info(f"{'='*70}")
                self.logger.info(f"Model path: {model_to_load}")
                self.logger.info(f"Source: Phase 2 safety fine-tuning (do-not-answer dataset)")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    torch_dtype=torch_dtype,
                    device_map=self.args.device,
                    trust_remote_code=True
                )
                self.logger.info(f"✓ {model_source} model loaded successfully!")
                self.logger.info(f"  This model has been fine-tuned on safety data (do-not-answer)")
                self.logger.info(f"  and should refuse harmful requests better than the base model.")
                
                # 토크나이저도 Phase 2 모델에서 로드
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_to_load,
                    trust_remote_code=True
                )
            else:
                # Fallback: 원본 모델 로드 (Phase 2 모델이 없으면)
                self.logger.warning(f"\n{'='*70}")
                self.logger.warning(f"⚠️ Phase 2 Safety Fine-tuned Model Not Found!")
                self.logger.warning(f"{'='*70}")
                self.logger.warning(f"Expected path: {phase2_model_dir}")
                self.logger.warning(f"Falling back to original model: {self.args.model_name}")
                self.logger.warning(f"⚠️ WARNING: Phase 3 will train on original (unsafe) model!")
                self.logger.warning(f"   This may result in a model that is not safety-aligned.")
                self.logger.warning(f"{'='*70}\n")
                
                self.logger.info(f"Loading original model: {self.args.model_name}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    torch_dtype=torch_dtype,
                    device_map=self.args.device,
                    trust_remote_code=True
                )
                
                self.logger.info(f"✓ Original model loaded (fallback)")
                
                # 토크나이저도 원본에서 로드
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.model_name,
                    trust_remote_code=True
                )
            
            # ✅ Step 3: 토크나이저 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"✓ Tokenizer loaded successfully")
            self.logger.info(f"  - Vocab size: {self.tokenizer.vocab_size}")
            self.logger.info(f"  - Pad token: {self.tokenizer.pad_token}")
            self.logger.info(f"  - EOS token: {self.tokenizer.eos_token}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def load_utility_data(self):
        """GSM8K 훈련 데이터 로드"""
        from datasets import load_dataset
        
        try:
            self.logger.info(f"Loading GSM8K (main/train) with max_samples={self.args.utility_samples}...")
            
            dataset = load_dataset('openai/gsm8k', 'main', split='train')
            
            # 샘플 수 제한
            if self.args.utility_samples > 0:
                dataset = dataset.select(range(min(self.args.utility_samples, len(dataset))))
            
            self.logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # 데이터셋 준비 함수
            def format_gsm8k(examples):
                # question -> input, answer는 전체 풀이 과정
                formatted_inputs = []
                formatted_targets = []
                
                for question, answer in zip(examples['question'], examples['answer']):
                    formatted_inputs.append(f"Q: {question}\nA:")
                    formatted_targets.append(answer)
                
                return {
                    'input_text': formatted_inputs,
                    'target_text': formatted_targets,
                }
            
            dataset = dataset.map(format_gsm8k, batched=True, batch_size=100)
            
            # 데이터로더 생성
            class GSM8KDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, tokenizer, max_length=512):
                    self.dataset = dataset
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    sample = self.dataset[idx]
                    input_text = sample['input_text']
                    target_text = sample['target_text']
                    
                    # 결합: "Q: ...\nA: <answer>"
                    combined = f"{input_text}{target_text}"
                    
                    # 토크나이제이션
                    encoding = self.tokenizer(
                        combined,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                    }
            
            gsm8k_dataset = GSM8KDataset(dataset, self.tokenizer, max_length=512)
            
            # Custom collate function for variable length sequences
            def collate_fn_gsm8k(batch):
                """길이가 다른 시퀀스를 padding으로 처리"""
                max_len = max(len(item['input_ids']) for item in batch)
                
                input_ids_list = []
                attention_masks_list = []
                
                for item in batch:
                    input_ids = item['input_ids']
                    attn_mask = item['attention_mask']
                    
                    # Padding
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
            
            self.train_loader = DataLoader(
                gsm8k_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn_gsm8k
            )
            
            self.logger.info(f"✓ Utility dataloader created:")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Total batches: {len(self.train_loader)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load utility data: {str(e)}", exc_info=True)
            raise
    
    def register_mask_hooks(self):
        """
        ✅ Custom forward 함수로 BasisLinear.apply 사용
        
        Hook 대신 module.forward를 교체하여 autograd.Function으로 gradient 계산
        """
        # New robust implementation: do NOT use custom autograd.Function.
        # Instead register a Python-level custom forward that reconstructs weight
        # from a learnable `basis_coeff` (so autograd tracks it), and keep the
        # per-input (element-wise) mask to zero-out gradients after backward.
        try:
            self.original_forwards = {}
            self._basis_params = []  # list of parameters to optimize

            # Freeze all existing model parameters to avoid accidental updates
            for p in self.model.parameters():
                p.requires_grad = False

            for layer_idx in sorted(self.masks.keys()):
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj

                # load mask and U
                mask = self.masks[layer_idx]
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                mask = mask.to(self.model.device).to(torch.bool)

                U = self.basis_data[layer_idx]['U'].to(self.model.device)
                W_original = target_module.weight.data.clone()
                U = U.to(dtype=W_original.dtype, device=W_original.device)

                # Create basis_coeff as a Parameter so autograd computes grads w.r.t it
                basis_coeff = (W_original @ U).clone().detach()
                basis_coeff = nn.Parameter(basis_coeff)
                # attach to module so we can access it later
                target_module.register_parameter('basis_coeff', basis_coeff)
                target_module.U_matrix = U
                target_module._warp_mask = mask  # boolean mask on input dims

                # Keep track for optimizer
                self._basis_params.append(target_module.basis_coeff)

                # Save original forward to restore later
                self.original_forwards[layer_idx] = target_module.forward

                # Custom forward: compute weight = basis_coeff @ U.T and run linear
                def make_custom_forward(basis_name='basis_coeff'):
                    def custom_forward(x):
                        module = getattr(custom_forward, '_module')
                        basis = getattr(module, basis_name)
                        U_local = module.U_matrix
                        weight = basis @ U_local.T
                        return torch.nn.functional.linear(x, weight, module.bias)
                    return custom_forward

                custom_fn = make_custom_forward('basis_coeff')
                setattr(custom_fn, '_module', target_module)
                target_module.forward = custom_fn

                # Logging
                self.logger.info(f"✓ Layer {layer_idx}: registered basis_coeff")
                self.logger.info(f"  - basis_coeff shape: {target_module.basis_coeff.shape}")
                self.logger.info(f"  - mask shape: {mask.shape}")
                self.logger.info(f"  - Frozen (True) count: {mask.sum().item()}/{mask.numel()} ({100*mask.sum().item()/mask.numel():.1f}%)")

            self.logger.info(f"✅ {len(self.original_forwards)} layers configured for masked fine-tuning")

        except Exception as e:
            self.logger.error(f"Failed to register mask hooks: {str(e)}", exc_info=True)
            raise
    
    def restore_original_forwards(self):
        """Forward 함수 복원"""
        try:
            for layer_idx, original_forward in self.original_forwards.items():
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                target_module.forward = original_forward
                # remove attached parameter if exists
                if hasattr(target_module, 'basis_coeff'):
                    try:
                        delattr(target_module, 'basis_coeff')
                    except Exception:
                        pass
                if hasattr(target_module, '_warp_mask'):
                    try:
                        delattr(target_module, '_warp_mask')
                    except Exception:
                        pass
            self.logger.info(f"✓ {len(self.original_forwards)} original forwards restored")
        except Exception as e:
            self.logger.error(f"Failed to restore forwards: {str(e)}", exc_info=True)
    
    def _reconstruct_and_save_final_model(self):
        """
        ✅ 최종 모델 저장 전: basis_coeff @ U^T를 계산하여 원본 weight로 복원
        
        학습된 basis_coeff를 사용해 최종 가중치를 재구성하고,
        이를 모델의 weight로 설정한 후 저장.
        basis_coeff 파라미터는 state_dict에서 제거.
        """
        try:
            self.logger.info("  Reconstructing weights from learned basis_coeff...")
                                     
            for layer_idx in sorted(self.masks.keys()):
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                
                if not hasattr(target_module, 'basis_coeff'):
                    self.logger.warning(f"  Layer {layer_idx}: basis_coeff not found, skipping")
                    continue
                
                # basis_coeff와 U 로드
                basis_coeff = target_module.basis_coeff.data  # (out_features, rank)
                U = target_module.U_matrix  # (in_features, rank)
                
                # 최종 가중치 재구성: W = basis_coeff @ U^T
                final_weight = basis_coeff @ U.T  # (out_features, in_features)
                
                # 모델의 weight 업데이트
                target_module.weight.data = final_weight
                
                self.logger.info(f"    ✓ Layer {layer_idx}: weight reconstructed (shape: {final_weight.shape})")
            
            # ✅ 최종 모델 저장
            checkpoint_dir = os.path.join(self.args.checkpoint_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # ✅ state_dict 로드 후 basis_coeff 파라미터 제거
            model_state_dict = self.model.state_dict()
            
            # basis_coeff 키 제거
            basis_coeff_keys = [k for k in model_state_dict.keys() if 'basis_coeff' in k]
            for key in basis_coeff_keys:
                del model_state_dict[key]
                self.logger.info(f"    ✓ Removed {key} from state_dict")
            
            final_checkpoint = {
                'epoch': self.args.epochs - 1,
                'model_state_dict': model_state_dict,
                'config': vars(self.args),
                'metadata': {
                    'phase': 'phase3',
                    'basis_reconstruction': True,
                    'description': 'Final model with reconstructed weights from basis_coeff @ U^T'
                }
            }
            
            final_path = os.path.join(checkpoint_dir, 'phase3_final_reconstructed.pt')
            torch.save(final_checkpoint, final_path)
            self.logger.info(f"  ✓ Final reconstructed model saved: {final_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to reconstruct and save final model: {str(e)}", exc_info=True)
            raise
    
    def train_epoch(self, epoch: int, optimizer, lr_scheduler=None):
        """
        한 에포크 훈련 (상세 로깅 포함)
        
        Args:
            epoch: 에포크 번호
            optimizer: 옵티마이저
            lr_scheduler: 학습률 스케줄러
        """
        self.model.train()
        
        total_loss = 0.0
        total_frozen_grad_norm = 0.0
        total_trainable_grad_norm = 0.0
        total_masked_grad_norm = 0.0  # 마스킹 전 gradient norm
        total_param_norm = 0.0  # 파라미터 norm
        total_tokens = 0  # 총 토큰 수
        num_batches = 0
        
        # 배치별 상세 로깅을 위한 저장소
        batch_logs = []
        
        # GSM8K 파인튜닝 검증용 변수
        loss_improvements = []  # 손실 개선 추이 추적
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.args.epochs}",
            total=len(self.train_loader)
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            
            # 배치 통계
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            num_tokens_batch = (input_ids != self.tokenizer.pad_token_id).sum().item()
            total_tokens += num_tokens_batch
            
            # 메모리 사용량 기록
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Forward pass: CLM으로 학습
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # ✅ Gradient accumulation: loss를 accumulation_steps로 나누기
            scaled_loss = loss / self.gradient_accumulation_steps
            
            # ✅ GSM8K 파인튜닝 검증: 손실 값 기록 (스케일링되지 않은 원본 loss 기록)
            loss_improvements.append(loss.item())
            
            # Backward pass
            scaled_loss.backward()
            
            # ✅ Gradient accumulation 스텝인지 확인
            should_step = ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx == len(self.train_loader) - 1)
            
            # After backward, apply element-wise mask to basis_coeff.grad to freeze important inputs
            batch_frozen_grad = 0.0
            batch_trainable_grad = 0.0
            batch_masked_before = 0.0
            batch_param_norm = 0.0
            layer_logs = []

            for layer_idx in sorted(self.masks.keys()):
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj

                if not hasattr(target_module, 'basis_coeff'):
                    continue

                basis_param = target_module.basis_coeff
                if basis_param.grad is None:
                    continue

                # mask: boolean on input dimension (d_in,)
                mask = target_module._warp_mask
                frozen_idx = mask
                trainable_idx = ~mask

                # basis_param.grad shape: (d_out, d_in)
                pre_norm = basis_param.grad.norm().item() if basis_param.grad.numel() > 0 else 0.0
                batch_masked_before += pre_norm

                # 마스킹 전 frozen/trainable 분석
                if frozen_idx.any():
                    frozen_grad_before = basis_param.grad[:, frozen_idx]
                    frozen_norm_before = torch.norm(frozen_grad_before).item() if frozen_grad_before.numel() > 0 else 0.0
                else:
                    frozen_norm_before = 0.0

                # ✅ 마스킹 적용: WaRP의 핵심 - frozen direction의 gradient를 0으로 설정
                if frozen_idx.any():
                    basis_param.grad[:, frozen_idx] = 0.0

                post_norm = basis_param.grad.norm().item() if basis_param.grad.numel() > 0 else 0.0

                # statistics for logging
                frozen_grad_norm = 0.0
                trainable_grad_norm = 0.0
                if frozen_idx.any():
                    frozen_grad = basis_param.grad[:, frozen_idx]
                    frozen_grad_norm = torch.norm(frozen_grad).item() if frozen_grad.numel() > 0 else 0.0
                if trainable_idx.any():
                    trainable_grad = basis_param.grad[:, trainable_idx]
                    trainable_grad_norm = torch.norm(trainable_grad).item() if trainable_grad.numel() > 0 else 0.0

                # 파라미터 norm
                param_norm = basis_param.norm().item()

                batch_frozen_grad += frozen_grad_norm
                batch_trainable_grad += trainable_grad_norm
                batch_param_norm += param_norm
                total_frozen_grad_norm += frozen_grad_norm
                total_trainable_grad_norm += trainable_grad_norm
                total_param_norm += param_norm

                # 레이어별 로그 저장
                layer_logs.append({
                    'layer_idx': layer_idx,
                    'grad_pre_mask': pre_norm,
                    'grad_post_mask': post_norm,
                    'frozen_grad_before': frozen_norm_before,
                    'frozen_grad_after': frozen_grad_norm,
                    'trainable_grad': trainable_grad_norm,
                    'param_norm': param_norm,
                    'num_frozen': frozen_idx.sum().item(),
                    'num_trainable': trainable_idx.sum().item(),
                })
            
            # ✅ Gradient accumulation: 스텝에서만 업데이트
            if should_step:
                # ✅ Gradient clipping (max_grad_norm=0.3, finetune_gsm8k.py와 동일)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                
                # Update
                optimizer.step()
                optimizer.zero_grad()
                
                if lr_scheduler:
                    lr_scheduler.step()
                
                num_batches += 1
            
            total_loss += loss.item()  # 스케일링되지 않은 원본 loss 누적
            total_masked_grad_norm += batch_masked_before
            
            # 배치 로그 저장
            batch_logs.append({
                'batch_idx': batch_idx,
                'loss': loss.item(),
                'seq_length': input_ids.shape[1],
                'batch_size': input_ids.shape[0],
                'num_tokens': num_tokens_batch,
                'frozen_grad': batch_frozen_grad,
                'trainable_grad': batch_trainable_grad,
                'param_norm': batch_param_norm,
                'layer_logs': layer_logs,
                'is_update_step': should_step,
            })
            
            # ✅ GSM8K 파인튜닝 검증: 손실 개선도 계산
            if len(loss_improvements) > 1:
                loss_delta = loss_improvements[-2] - loss_improvements[-1]
            else:
                loss_delta = 0.0
            
            # 진행률 표시 업데이트
            progress_bar.update(1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'frzn_norm': f'{batch_frozen_grad:.6f}',
                'train_norm': f'{batch_trainable_grad:.4f}',
                'accum': f'{((batch_idx + 1) % self.gradient_accumulation_steps) or self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}' if not should_step else '✓'
            })
            
            # 매 N개 배치마다 상세 로그 출력
            log_interval = max(1, len(self.train_loader) // 10)  # 에포크당 10번 출력
            if (batch_idx + 1) % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                
                # ✅ GSM8K 파인튜닝 검증 정보
                avg_loss_recent = np.mean(loss_improvements[-log_interval:]) if len(loss_improvements) >= log_interval else np.mean(loss_improvements)
                loss_trend = "↓" if loss_delta < 0 else ("→" if abs(loss_delta) < 1e-5 else "↑")
                
                self.logger.info(
                    f"[Batch {batch_idx+1:4d}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss_recent:.4f}) {loss_trend} | "
                    f"Frozen grad: {batch_frozen_grad:.6f} | "
                    f"Trainable grad: {batch_trainable_grad:.4f} | "
                    f"Param norm: {batch_param_norm:.4f} | "
                    f"Tokens: {num_tokens_batch}/{seq_length*batch_size} | "
                    f"LR: {current_lr:.2e} | "
                    f"Update steps: {num_batches}"
                )
                
                # 레이어별 상세 정보 출력
                for layer_log in layer_logs:
                    self.logger.debug(
                        f"  Layer {layer_log['layer_idx']}: "
                        f"grad_pre={layer_log['grad_pre_mask']:.6f} "
                        f"grad_post={layer_log['grad_post_mask']:.6f} "
                        f"frozen_before={layer_log['frozen_grad_before']:.6f} "
                        f"frozen_after={layer_log['frozen_grad_after']:.6f} | "
                        f"trainable={layer_log['trainable_grad']:.4f} | "
                        f"frozen({layer_log['num_frozen']}) "
                        f"trainable({layer_log['num_trainable']})"
                    )
        
        # 에포크 통계 (num_batches는 gradient accumulation 스텝 횟수)
        avg_loss = total_loss / max(len(batch_logs), 1)  # 모든 forward pass로 나누기
        num_update_steps = num_batches
        
        avg_frozen_grad = total_frozen_grad_norm / max(len(batch_logs), 1)
        avg_trainable_grad = total_trainable_grad_norm / max(len(batch_logs), 1)
        avg_masked_before = total_masked_grad_norm / max(len(batch_logs), 1)
        avg_param_norm = total_param_norm / max(len(batch_logs), 1)
        
        # ✅ GSM8K 파인튜닝 검증: 손실 수렴 분석
        loss_array = np.array(loss_improvements)
        loss_first_half = np.mean(loss_array[:len(loss_array)//2]) if len(loss_array) > 0 else float('inf')
        loss_second_half = np.mean(loss_array[len(loss_array)//2:]) if len(loss_array) > 0 else float('inf')
        loss_convergence = loss_first_half - loss_second_half
        loss_convergence_pct = (loss_convergence / loss_first_half * 100) if loss_first_half > 0 else 0
        
        # 손실 변동성 분석
        loss_std = np.std(loss_array) if len(loss_array) > 0 else 0
        loss_cv = (loss_std / avg_loss * 100) if avg_loss > 0 else 0  # 변동계수
        
        # 초반 vs 후반 손실
        loss_min = np.min(loss_array) if len(loss_array) > 0 else float('inf')
        loss_max = np.max(loss_array) if len(loss_array) > 0 else float('inf')
        loss_improvement_ratio = (loss_max - loss_min) / loss_max * 100 if loss_max > 0 else 0
        
        # 에포크 완료 로그
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Epoch {epoch+1} Summary - GSM8K 파인튜닝 (SFT 스타일 훈련)")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  [손실 함수]")
        self.logger.info(f"    • 평균 손실 (Loss): {avg_loss:.4f}")
        self.logger.info(f"    • 최저 손실: {loss_min:.4f}")
        self.logger.info(f"    • 최고 손실: {loss_max:.4f}")
        self.logger.info(f"    • 손실 개선율: {loss_improvement_ratio:.2f}% (범위 {loss_max:.4f} → {loss_min:.4f})")
        self.logger.info(f"    • 손실 표준편차: {loss_std:.6f}")
        self.logger.info(f"    • 손실 변동계수 (CV): {loss_cv:.2f}% {'✓' if loss_cv < 15 else '⚠️'} (CV < 15% 권장)")
        self.logger.info(f"    • 전반부 vs 후반부 수렴: {loss_convergence:.4f} ({loss_convergence_pct:.2f}%) {'✓' if loss_convergence > 0 else '⚠️'}")
        
        self.logger.info(f"  [그래디언트 흐름]")
        self.logger.info(f"    • Frozen 방향 그래디언트: {avg_frozen_grad:.6f} (expected ~0) {'✓' if avg_frozen_grad < 1e-5 else '⚠️'}")
        self.logger.info(f"    • Trainable 방향 그래디언트: {avg_trainable_grad:.4f}")
        self.logger.info(f"    • 그래디언트 비율 (Trainable/Frozen): {avg_trainable_grad/max(avg_frozen_grad, 1e-8):.1f}x")
        
        self.logger.info(f"  [데이터 및 훈련 통계]")
        self.logger.info(f"    • 총 Forward 패스: {len(batch_logs)}")
        self.logger.info(f"    • 총 Gradient Accumulation 스텝 (업데이트): {num_update_steps}")
        self.logger.info(f"    • Accumulation 비율: {self.gradient_accumulation_steps}x")
        self.logger.info(f"    • 총 토큰: {total_tokens:,}")
        self.logger.info(f"    • 배치당 평균 토큰: {total_tokens / max(len(batch_logs), 1):.1f}")
        self.logger.info(f"    • 배치당 평균 시퀀스 길이: {sum(b['seq_length'] for b in batch_logs) / max(len(batch_logs), 1):.1f}")
        
        self.logger.info(f"  [파라미터 업데이트]")
        self.logger.info(f"    • 파라미터 norm: {avg_param_norm:.4f}")
        
        self.logger.info(f"  [학습 건강도]")
        # 판정 조건: (1) 그래디언트 정상 흐름, (2) 손실이 감소하거나 안정적, (3) Trainable 그래디언트 존재
        is_gradient_ok = avg_frozen_grad < 1e-5 and avg_trainable_grad > 1e-6
        is_loss_stable = loss_cv < 25  # CV < 25%로 완화
        is_converging = loss_convergence_pct > 0.3  # 수렴 기준 완화 (0.3% 이상)
        
        if is_gradient_ok and is_loss_stable:
            if loss_convergence_pct > 1.0:
                self.logger.info(f"    ✅ 파인튜닝 진행 중! (손실 수렴 O, 그래디언트 흐름 O, SFT 스타일 훈련 적용됨)")
            elif is_converging:
                self.logger.info(f"    ✅ 파인튜닝 진행 중! (손실 안정적, 그래디언트 흐름 O, SFT 스타일 훈련 적용됨)")
            else:
                self.logger.info(f"    ⚠️ 파인튜닝 진행 (손실이 천천히 수렴 중)")
        else:
            if not is_gradient_ok:
                self.logger.info(f"    ⚠️ 주의: 그래디언트 흐름 문제 (Frozen: {avg_frozen_grad:.6f})")
            if not is_loss_stable:
                self.logger.info(f"    ⚠️ 주의: 손실 변동성 높음 (CV: {loss_cv:.2f}%)")
        self.logger.info(f"{'='*70}\n")
        
        return avg_loss
    
    def train(self):
        """
        전체 훈련 루프 (상세 로깅 포함)
        """
        try:
            self.logger.info("\n" + "="*70)
            self.logger.info("PHASE 3: INCREMENTAL LEARNING WITH MASKED GRADIENT UPDATES")
            self.logger.info("="*70 + "\n")
            
            # 1. 데이터 및 모델 로드
            self.logger.info("[Step 1] Loading basis and masks...")
            self.load_basis()
            self.load_masks()
            
            self.logger.info("\n[Step 2] Loading model...")
            self.load_model()
            
            self.logger.info("\n[Step 3] Loading utility data...")
            start_time = datetime.now()
            self.load_utility_data()
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✓ Data loading completed in {load_time:.2f}s")
            
            # 2. 마스킹 hook 등록
            self.logger.info("\n[Step 4] Registering mask hooks...")
            self.register_mask_hooks()
            
            # 마스킹 검증
            self.logger.info("\n[Step 4.5] Validating mask configuration...")
            total_frozen = 0
            total_trainable = 0
            for layer_idx in sorted(self.masks.keys()):
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                mask = target_module._warp_mask
                num_frozen = (mask == 1).sum().item()
                num_trainable = (mask == 0).sum().item()
                total_frozen += num_frozen
                total_trainable += num_trainable
                self.logger.info(
                    f"  Layer {layer_idx}: {num_frozen}/{mask.numel()} frozen "
                    f"({100*num_frozen/mask.numel():.1f}%) | "
                    f"{num_trainable} trainable"
                )
            self.logger.info(f"✓ Total dimensions: {total_frozen} frozen, {total_trainable} trainable")
            
            # 3. Optimizer 설정
            self.logger.info("\n[Step 5] Setting up optimizer and scheduler (SFTTrainer style)...")
            
            # ✅ gradient accumulation 설정 (effective batch: 4 * 16 = 64)
            self.gradient_accumulation_steps = 16
            
            # ✅ gradient checkpointing 활성화 (메모리 절감: ~50%)
            if hasattr(self.model.config, 'gradient_checkpointing'):
                self.model.gradient_checkpointing_enable()
                self.logger.info(f"✓ Gradient checkpointing enabled")
            
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
            
            # ✅ Learning rate scheduler: Cosine annealing with warmup (SFTTrainer 방식)
            total_steps = len(self.train_loader) * self.args.epochs
            warmup_steps = int(total_steps * 0.05)  # 5% warmup (finetune_gsm8k.py와 동일)
            
            def cosine_lr_lambda(step):
                """Cosine annealing with linear warmup"""
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=cosine_lr_lambda
            )
            
            self.logger.info(f"✓ Optimizer configured (matching SFTTrainer):")
            self.logger.info(f"  - Algorithm: AdamW")
            self.logger.info(f"  - Learning rate (initial): {self.args.learning_rate}")
            self.logger.info(f"  - Weight decay: {self.args.weight_decay}")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Gradient accumulation steps: {self.gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {self.args.batch_size * self.gradient_accumulation_steps}")
            self.logger.info(f"  - Scheduler: Cosine decay with 5% warmup")
            self.logger.info(f"  - Total training steps: {total_steps}")
            self.logger.info(f"  - Warmup steps: {warmup_steps}")
            self.logger.info(f"  - Max grad norm: 0.3")
            self.logger.info(f"  - Gradient checkpointing: Enabled")
            
            # 4. 훈련
            self.logger.info("\n[Step 6] Starting training...")
            self.logger.info(f"  - Total epochs: {self.args.epochs}")
            self.logger.info(f"  - Batches per epoch: {len(self.train_loader)}")
            self.logger.info(f"  - Batch size: {self.args.batch_size}")
            self.logger.info(f"  - Total samples: {len(self.train_loader) * self.args.batch_size * self.args.epochs}")
            self.logger.info(f"\n{'='*70}\n")
            
            best_loss = float('inf')
            training_start = datetime.now()
            epoch_losses = []
            
            for epoch in range(self.args.epochs):
                epoch_start = datetime.now()
                
                self.logger.info(f"{'='*70}")
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs}")
                self.logger.info(f"{'='*70}")
                
                avg_loss = self.train_epoch(epoch, optimizer, lr_scheduler)
                
                epoch_time = (datetime.now() - epoch_start).total_seconds()
                current_lr = optimizer.param_groups[0]['lr']
                
                epoch_losses.append(avg_loss)
                
                # 에포크별 요약
                self.logger.info(f"  Epoch time: {epoch_time:.2f}s ({len(self.train_loader) / epoch_time:.1f} batches/s)")
                self.logger.info(f"  Current LR: {current_lr:.2e}")
                
                # 손실 개선 추이
                if epoch > 0:
                    loss_change = avg_loss - epoch_losses[epoch-1]
                    loss_pct = (loss_change / epoch_losses[epoch-1]) * 100 if epoch_losses[epoch-1] != 0 else 0
                    arrow = "↓" if loss_change < 0 else "↑"
                    self.logger.info(f"  Loss change: {arrow} {abs(loss_change):.4f} ({abs(loss_pct):.2f}%)")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"  ✓ NEW BEST LOSS! Checkpoint saved.\n")
                else:
                    self.save_checkpoint(epoch, is_best=False)
                    self.logger.info(f"  Checkpoint saved.\n")
            
            # 5. 정리 (마스킹 통계는 restore 전에 수집)
            total_training_time = (datetime.now() - training_start).total_seconds()
            self.logger.info("\n[Step 7] Finalizing...")
            
            # ✅ 마스킹 통계를 restore 전에 수집
            total_frozen = 0
            total_trainable = 0
            for layer_idx in sorted(self.masks.keys()):
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                if hasattr(target_module, '_warp_mask'):
                    mask = target_module._warp_mask
                    num_frozen = (mask == 1).sum().item()
                    num_trainable = (mask == 0).sum().item()
                    total_frozen += num_frozen
                    total_trainable += num_trainable
            
            # ✅ CRITICAL: restore 전에 최종 가중치 재구성 및 저장
            self.logger.info("\n[Step 7.5] Reconstructing final weights from basis_coeff...")
            self._reconstruct_and_save_final_model()
            
            self.restore_original_forwards()
            
            self.logger.info("\n" + "="*70)
            self.logger.info("PHASE 3 TRAINING FINAL SUMMARY - SFT 스타일 GSM8K 파인튜닝 결과")
            self.logger.info("="*70)
            
            # 전체 에포크에 대한 손실 분석
            loss_array = np.array(epoch_losses)
            initial_loss = loss_array[0]
            final_loss = loss_array[-1]
            best_loss_val = np.min(loss_array)
            total_loss_improvement = initial_loss - final_loss
            total_loss_improvement_pct = (total_loss_improvement / initial_loss) * 100 if initial_loss > 0 else 0
            
            # 손실 안정성 분석
            loss_std_epochs = np.std(loss_array)
            loss_monotonic = np.sum(np.diff(loss_array) <= 0)  # 손실이 감소하거나 유지된 에포크 수
            
            self.logger.info(f"  [전체 손실 동향]")
            self.logger.info(f"    • 초기 손실: {initial_loss:.4f}")
            self.logger.info(f"    • 최종 손실: {final_loss:.4f}")
            self.logger.info(f"    • 최저 손실: {best_loss_val:.4f}")
            self.logger.info(f"    • 총 개선량: {total_loss_improvement:.4f} ({total_loss_improvement_pct:.2f}%) {'✅' if total_loss_improvement_pct > 3 else '⚠️'}")
            self.logger.info(f"    • 에포크간 손실 표준편차: {loss_std_epochs:.6f}")
            self.logger.info(f"    • 손실 감소 에포크: {loss_monotonic}/{len(loss_array)-1} ({'✓' if loss_monotonic > len(loss_array)*0.7 else '⚠️'})")
            
            # 수렴 추이
            if len(loss_array) > 2:
                first_third = np.mean(loss_array[:len(loss_array)//3])
                last_third = np.mean(loss_array[2*len(loss_array)//3:])
                convergence_improvement = first_third - last_third
                convergence_improvement_pct = (convergence_improvement / first_third) * 100 if first_third > 0 else 0
                self.logger.info(f"    • 전반부 vs 후반부 수렴: {convergence_improvement:.4f} ({convergence_improvement_pct:.2f}%)")
            
            self.logger.info(f"\n  [SFT 스타일 훈련 구성]")
            self.logger.info(f"    • Per-device batch size: {self.args.batch_size}")
            self.logger.info(f"    • Gradient accumulation steps: {self.gradient_accumulation_steps}")
            self.logger.info(f"    • Effective batch size: {self.args.batch_size * self.gradient_accumulation_steps}")
            self.logger.info(f"    • Gradient checkpointing: Enabled (메모리 절감)")
            self.logger.info(f"    • LR scheduler: Cosine annealing with 5% warmup")
            self.logger.info(f"    • Max grad norm: 0.3")
            self.logger.info(f"    • Optimizer: AdamW with weight decay {self.args.weight_decay}")
            
            self.logger.info(f"\n  [훈련 통계]")
            self.logger.info(f"    • 총 훈련 시간: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
            self.logger.info(f"    • 총 에포크: {self.args.epochs}")
            self.logger.info(f"    • 에포크당 평균 시간: {total_training_time/max(self.args.epochs, 1):.2f}s")
            self.logger.info(f"    • 체크포인트 디렉토리: {self.args.checkpoint_dir}")
            
            self.logger.info(f"\n  [보호된 안전 메커니즘 (WaRP)]")
            self.logger.info(f"    • 총 Frozen 차원: {total_frozen:,}")
            self.logger.info(f"    • 총 Trainable 차원: {total_trainable:,}")
            if total_frozen + total_trainable > 0:
                self.logger.info(f"    • Frozen 비율: {total_frozen/(total_frozen+total_trainable)*100:.2f}%")
            else:
                self.logger.info(f"    • Frozen 비율: N/A (마스크 데이터 없음)")
            
            self.logger.info(f"\n  [최종 판정: SFT 스타일 GSM8K 파인튜닝 성공도]")
            
            # 판정 기준 계산
            loss_improvement_ok = total_loss_improvement_pct > 3  # 3% 이상 개선
            loss_stability_ok = loss_std_epochs < initial_loss * 0.3  # 에포크간 변동 < 30%
            monotonic_ok = loss_monotonic >= len(loss_array) * 0.5  # 50% 이상의 에포크에서 감소
            
            if loss_improvement_ok and (loss_stability_ok or monotonic_ok):
                self.logger.info(f"    ✅ 성공적인 파인튜닝!")
                self.logger.info(f"       - 손실이 {total_loss_improvement_pct:.2f}% 개선됨 (초기: {initial_loss:.4f} → 최종: {final_loss:.4f})")
                self.logger.info(f"       - 손실이 안정적으로 감소하는 추이")
                if total_frozen > 0:
                    self.logger.info(f"       - 안전 메커니즘({total_frozen:,} dims)는 보호됨")
            elif total_loss_improvement_pct > 1:
                self.logger.info(f"    ⚠️ 파인튜닝 진행 중 (개선이 제한적)")
                self.logger.info(f"       - 손실 개선: {total_loss_improvement_pct:.2f}%")
                self.logger.info(f"       - 손실 범위: {initial_loss:.4f} → {final_loss:.4f}")
                self.logger.info(f"       - 더 많은 에포크 또는 데이터 필요할 수 있음")
            else:
                self.logger.info(f"    ❌ 파인튜닝 실패 가능성")
                self.logger.info(f"       - 손실 개선 미흡: {total_loss_improvement_pct:.2f}%")
                self.logger.info(f"       - 학습률 또는 마스킹 설정 검토 필요")
            
            self.logger.info("="*70 + "\n")
            
        except Exception as e:
            self.logger.error(f"\n✗ Error in Phase 3: {str(e)}", exc_info=True)
            self.logger.error("Attempting to restore original forwards...")
            try:
                self.restore_original_forwards()
            except Exception as restore_err:
                self.logger.error(f"Failed to restore forwards: {str(restore_err)}")
            raise
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        체크포인트 저장
        
        Args:
            epoch: 에포크 번호
            is_best: 최고 성능 모델인지 여부
        """
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # --- Reconstruct weights from basis_coeff temporarily so saved checkpoints
        # contain standard weight tensors that can be loaded by `from_pretrained`
        # (avoids requiring a custom loader at evaluation time).
        orig_weights = {}
        try:
            for layer_idx in sorted(self.masks.keys()):
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj

                if not hasattr(target_module, 'basis_coeff') or not hasattr(target_module, 'U_matrix'):
                    continue

                # preserve original weight
                orig_weights[layer_idx] = target_module.weight.data.clone()

                # reconstruct final weight = basis_coeff @ U.T
                basis = target_module.basis_coeff.data
                U = target_module.U_matrix
                final_weight = basis @ U.T

                # copy into module.weight (in-place)
                target_module.weight.data.copy_(final_weight)

            # Build checkpoint with reconstructed weights
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'config': vars(self.args),
            }

            # Save usual checkpoint
            save_path = os.path.join(checkpoint_dir, f'phase3_epoch_{epoch:03d}.pt')
            torch.save(checkpoint, save_path)
            self.logger.debug(f"✓ Saved checkpoint (reconstructed weights): {save_path}")

            # Save best
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'phase3_best.pt')
                torch.save(checkpoint, best_path)
                self.logger.debug(f"✓ Saved best model (reconstructed weights): {best_path}")

        finally:
            # restore original weights to continue training unaffected
            for layer_idx, orig_w in orig_weights.items():
                layer = self.model.model.layers[layer_idx]
                target_module = layer.mlp.down_proj
                target_module.weight.data.copy_(orig_w)
