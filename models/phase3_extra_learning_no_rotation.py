"""
핵심:
1. Phase 2에서 생성된 마스크를 WaRP 모듈에 설정
2. WaRP 모듈의 forward에서 자동으로 detach() 적용
3. mask=1: 동결 (gradient 차단), mask=0: 학습 가능
4. GSM8K 데이터로 fine-tuning

원본 WaRP의 forward:
```python
weight = UT_backward @ (basis_coeff * mask).clone().detach() + \
         basis_coeff * (1 - mask) @ UT_forward
```

이미 warp_modules.py에 구현되어 있음!
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from .warp_modules import WaRPModule, restore_weight, restore_to_linear

logger = logging.getLogger(__name__)


class Phase3IncrementalLearnerNoRotation:
    """
    Phase 3: Incremental Learning (원본 WaRP 방식)
    
    목표: 안전 메커니즘 보호하면서 GSM8K로 학습
    
    핵심:
    - WaRP 모듈의 forward에서 자동으로 마스킹 적용
    - mask=1: detach()로 gradient 차단
    - mask=0: 학습 가능
    """
    
    def __init__(self, args, logger, basis_dir, masks_dir, phase0_model_dir):
        self.args = args
        self.logger = logger
        self.basis_dir = basis_dir
        self.masks_dir = masks_dir
        self.phase0_model_dir = phase0_model_dir
        
        # 모델 및 데이터
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        
        # Basis 및 마스크
        self.basis_data = {}
        self.masks = {}
        self.layer_types = []
        self.warp_monitors = []
        self.monitor_samples_per_group = int(getattr(self.args, 'warp_monitor_samples_per_group', 4))
        
        # 통계
        self.stats = {
            'best_loss': float('inf'),
            'best_epoch': 0,
        }
    
    def load_basis(self):
        """No-rotation 모드에서는 Phase 1 basis를 로드하지 않음"""
        try:
            layer_types_str = self.args.layer_type
            self.layer_types = [lt.strip() for lt in layer_types_str.split(',')]
            self.logger.info("No-rotation mode: skipping Phase 1 basis loading")
            self.logger.info(f"✓ Layer types: {self.layer_types}")

        except Exception as e:
            self.logger.error(f"Failed to initialize no-rotation basis config: {str(e)}", exc_info=True)
            raise
    
    def load_masks(self):
        """Phase 2의 마스크 로드"""
        try:
            self.logger.info(f"Loading masks from {self.masks_dir}...")
            
            # 메타데이터
            metadata_path = os.path.join(self.masks_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                masks_metadata = json.load(f)
            
            self.logger.info(f"✓ Mask metadata loaded:")
            self.logger.info(f"  - Keep ratio: {masks_metadata.get('keep_ratio')}")
            
            # Layer types
            layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]
            
            # 마스크 로드
            total_loaded = 0
            
            for layer_type in layer_types:
                layer_type_dir = os.path.join(self.masks_dir, layer_type)
                if not os.path.exists(layer_type_dir):
                    continue
                
                mask_files = sorted([
                    f for f in os.listdir(layer_type_dir)
                    if f.startswith('layer_') and f.endswith('_mask.pt')
                ])
                
                for mask_file in mask_files:
                    layer_idx = int(mask_file.split('_')[1])
                    mask_path = os.path.join(layer_type_dir, mask_file)
                    
                    # ✅ PyTorch 2.6+ compatibility: weights_only=False for numpy arrays
                    mask_data = torch.load(mask_path, weights_only=False)
                    key = (layer_idx, layer_type)
                    self.masks[key] = mask_data['mask']
                    total_loaded += 1
            
            self.logger.info(f"✓ Masks loaded: {total_loaded} combinations")
            
        except Exception as e:
            self.logger.error(f"Failed to load masks: {str(e)}", exc_info=True)
            raise
    
    def load_model(self):
        """Phase 0 모델 로드 및 WaRP 모듈로 변환"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .warp_modules import switch_to_warp_module
        
        try:
            self.logger.info(f"Loading Phase 0 model from {self.phase0_model_dir}...")
            
            # 데이터 타입
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.phase0_model_dir,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True
            )

            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
            
            # ❌ Gradient checkpointing 비활성화 (WaRP + freeze 문제)
            # WaRP 모듈에서 일부만 학습할 때 gradient checkpointing이 충돌함
            # if hasattr(self.model, 'gradient_checkpointing_enable'):
            #     self.model.gradient_checkpointing_enable()
            #     self.logger.info("✓ Gradient checkpointing enabled")
            
            self.logger.info(f"✓ Model loaded")
            
            # WaRP 모듈로 변환
            self.logger.info("Converting to WaRP modules...")
            self.model = switch_to_warp_module(
                self.model,
                self.layer_types,
                self.args.target_layers
            )
            
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
    
    def load_training_data(self):
        """
        Phase 3 훈련 데이터 로드 (dispatcher 패턴)
        
        dataset 선택:
        - 'gsm8k': Math reasoning (utility learning)
        - 'safety': Circuit breakers (safety learning)
        - 'metamath': Advanced math reasoning (utility learning)
        - 'math': Hendrycks MATH (utility learning)
        """
        phase3_dataset = getattr(self.args, 'phase3_dataset', 'gsm8k')
        
        if phase3_dataset == 'gsm8k':
            self._load_gsm8k()
        elif phase3_dataset == 'safety':
            self._load_safety_dataset()
        elif phase3_dataset == 'metamath':
            self._load_metamath()
        elif phase3_dataset == 'math':
            self._load_hendrycks_math()
        else:
            raise ValueError(f"Unknown phase3_dataset: {phase3_dataset}")

    def _load_gsm8k(self):
        """GSM8K 데이터 로드 및 SFT 방식으로 변환 (prompt/labels 분리)"""
        from datasets import load_dataset, Dataset
        
        try:
            self.logger.info("Loading GSM8K dataset...")
            
            # GSM8K 데이터셋 로드
            dataset = load_dataset('openai/gsm8k', 'main', split='train')
            
            # 샘플 수 제한
            max_samples = getattr(self.args, 'gsm8k_samples', 0)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"✓ Limited to {max_samples} samples (out of {len(dataset)} total)")
                else:
                    self.logger.info(f"✓ Using all {len(dataset)} samples (max_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(dataset)} samples (gsm8k_samples=0 or not specified)")
            
            max_length = getattr(self.args, 'max_length', 512)
            # Always disable multiprocessing here to avoid CUDA fork issues.
            num_proc = None

            def build_chat_prompt(question: str) -> str:
                system_msg = (
                    "You are a helpful assistant that solves math problems step by step. "
                    "Always show your reasoning and provide the final numerical answer after ####."
                )
                user_msg = f"Solve this problem step by step:\n\n{question.strip()}"
                return f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"

            def tokenize_sft_example(prompt_text: str, answer_text: str) -> Dict[str, List[int]]:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]

                remain = max(1, max_length - len(prompt_ids))
                answer_ids = self.tokenizer(
                    answer_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=remain,
                )["input_ids"]

                if self.tokenizer.eos_token_id is not None and (
                    len(answer_ids) == 0 or answer_ids[-1] != self.tokenizer.eos_token_id
                ):
                    if len(prompt_ids) + len(answer_ids) < max_length:
                        answer_ids = answer_ids + [self.tokenizer.eos_token_id]

                input_ids = (prompt_ids + answer_ids)[:max_length]
                attention_mask = [1] * len(input_ids)
                labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

            def preprocess(ex):
                prompt = build_chat_prompt(ex["question"])
                answer = ex["answer"]
                return tokenize_sft_example(prompt, answer)

            self.dataset = dataset.map(
                preprocess,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc="Tokenizing train",
            )
            
            self.logger.info(f"✓ Dataset created ({len(self.dataset)} samples)")
            
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K data: {str(e)}", exc_info=True)
            raise

    def _load_safety_dataset(self):
        """
        Circuit Breakers 안전 데이터셋 로드
        
        형식: JSON with prompt + llama3_output
        라벨링: 전체 시퀀스 (padding만 -100)
        """
        from datasets import Dataset
        
        try:
            circuit_breakers_path = getattr(self.args, 'circuit_breakers_path', './data/circuit_breakers_train.json')
            self.logger.info(f"Loading Safety dataset from {circuit_breakers_path}...")
            
            # JSON 로드
            with open(circuit_breakers_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 샘플 수 제한
            max_samples = getattr(self.args, 'circuit_breakers_samples_phase3', 0)
            if max_samples > 0:
                if len(data) > max_samples:
                    data = data[:max_samples]
                    self.logger.info(f"✓ Limited to {max_samples} samples (out of {len(data)} total)")
                else:
                    self.logger.info(f"✓ Using all {len(data)} samples (max_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(data)} samples")
            
            max_length = getattr(self.args, 'max_length', 512)
            
            def tokenize_safety_example(item: Dict) -> Dict[str, List[int]]:
                """
                Safety 데이터: prompt + llama3_output 결합
                라벨링: 전체 시퀀스 학습 (padding만 -100)
                """
                harmful_prompt = item.get("prompt", "")
                safe_response = item.get("llama3_output", "")
                
                full_text = f"{harmful_prompt} {safe_response}"
                
                encodings = self.tokenizer(
                    full_text,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )
                
                input_ids = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]
                
                # labels: padding(-100)을 제외한 모든 토큰 학습
                labels = input_ids.copy()
                labels = [label if mask == 1 else -100 for label, mask in zip(labels, attention_mask)]
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            
            # 데이터셋 생성
            tokenized_data = []
            for idx, item in enumerate(data):
                if idx == 0:
                    self.logger.info("\n[Dataset Sample #0]")
                    self.logger.info(f"  Keys: {item.keys()}")
                    self.logger.info(f"  Prompt (first 100 chars): {item.get('prompt', '')[:100]}...")
                    self.logger.info(f"  Response (first 100 chars): {item.get('llama3_output', '')[:100]}...")
                
                tokenized_data.append(tokenize_safety_example(item))
            
            self.dataset = Dataset.from_dict({
                "input_ids": [d["input_ids"] for d in tokenized_data],
                "attention_mask": [d["attention_mask"] for d in tokenized_data],
                "labels": [d["labels"] for d in tokenized_data],
            })
            
            self.logger.info(f"✓ Safety dataset created ({len(self.dataset)} samples)")
            
            # Collate function 추가 (리스트 → tensor 변환)
            def collate_fn_safety(batch):
                """Safety dataset 배치 처리: 리스트를 tensor로 변환"""
                max_len = max(len(item["input_ids"]) for item in batch)
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                
                input_ids_list = []
                attention_mask_list = []
                labels_list = []
                
                for item in batch:
                    input_ids = item["input_ids"]
                    attention_mask = item["attention_mask"]
                    labels = item["labels"]
                    
                    # 패딩 길이
                    pad_len = max_len - len(input_ids)
                    
                    # 패딩 추가
                    if pad_len > 0:
                        input_ids = input_ids + [pad_id] * pad_len
                        attention_mask = attention_mask + [0] * pad_len
                        labels = labels + [-100] * pad_len
                    
                    input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                    attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
                    labels_list.append(torch.tensor(labels, dtype=torch.long))
                
                return {
                    "input_ids": torch.stack(input_ids_list),
                    "attention_mask": torch.stack(attention_mask_list),
                    "labels": torch.stack(labels_list),
                }
            
            # DataLoader 생성 (collate_fn 추가)
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=getattr(self.args, 'batch_size', 2),
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn_safety
            )
            
            self.logger.info(f"✓ DataLoader created ({len(self.train_loader)} batches)")
            
        except Exception as e:
            self.logger.error(f"Failed to load Safety data: {str(e)}", exc_info=True)
            raise

    def _load_metamath(self):
        """
        MetaMath 데이터셋 로드 및 토크나이제이션
        
        형식: 쿼리(query) + 응답(response) 결합
        라벨링: 쿼리 부분은 -100 마스킹, 응답 부분만 학습
        (finetuning_metamath_full.py 스타일)
        """
        from datasets import load_dataset, Dataset
        
        try:
            self.logger.info("Loading MetaMath dataset...")
            
            # MetaMath 데이터셋 로드
            dataset = load_dataset("meta-math/MetaMathQA", split="train")
            
            # 샘플 수 제한
            max_samples = getattr(self.args, 'metamath_samples', 0)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"✓ Limited to {max_samples} samples")
                else:
                    self.logger.info(f"✓ Using all {len(dataset)} samples (metamath_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(dataset)} samples (metamath_samples=0 or not specified)")
            
            max_length = getattr(self.args, 'max_length', 512)
            num_proc = None  # CUDA fork 문제 방지
            
            def build_metamath_chat_template(query: str) -> str:
                """MetaMath용 시스템 프롬프트 + 사용자 쿼리"""
                system_msg = (
                    "You are an expert mathematical assistant. "
                    "Solve the following math problem step-by-step, showing all your logical reasoning clearly. "
                    "State the final answer at the end."
                )
                return f"{system_msg}\n\nUser: {query.strip()}\n\nAssistant:"
            
            def tokenize_metamath_example(query: str, response: str) -> Dict[str, List[int]]:
                """
                MetaMath 토크나이제이션 (finetuning_metamath_full.py 스타일)
                
                프롬프트 부분: 라벨 -100 (학습 제외)
                응답 부분: 정상 학습
                """
                prompt_text = build_metamath_chat_template(query)
                
                # 전체 텍스트 인코딩
                full_text = f"{prompt_text}{response}"
                full_encoded = self.tokenizer(
                    full_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
                full_ids = full_encoded["input_ids"]
                
                # 프롬프트만 인코딩
                prompt_encoded = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
                prompt_ids = prompt_encoded["input_ids"]
                
                # 라벨 마스킹: 프롬프트 부분은 -100
                labels = full_ids.copy()
                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100
                
                attention_mask = full_encoded["attention_mask"]
                
                return {
                    "input_ids": full_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            
            def preprocess_metamath(ex):
                """데이터셋 맵 함수"""
                query = ex.get("query", "")
                response = ex.get("response", "")
                return tokenize_metamath_example(query, response)
            
            self.dataset = dataset.map(
                preprocess_metamath,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc="Tokenizing MetaMath",
            )
            
            self.logger.info(f"✓ MetaMath dataset created ({len(self.dataset)} samples)")
            
        except Exception as e:
            self.logger.error(f"Failed to load MetaMath data: {str(e)}", exc_info=True)
            raise

    def _load_hendrycks_math(self):
        """
        Hendrycks MATH 데이터셋 로드 및 토크나이제이션

        finetuning_hendrycks_math_instruct.py와 최대한 동일한 전처리:
        - 과목/난이도 필터
        - boxed 답 추출 + reasoning 정리
        - mixed target format 옵션
        - chat template + system prompt 옵션
        - assistant-only loss(label masking)
        """
        import random
        import re
        from datasets import load_dataset, concatenate_datasets

        try:
            self.logger.info("Loading Hendrycks MATH dataset...")

            subject_to_config = {
                "Algebra": "algebra",
                "Counting & Probability": "counting_and_probability",
                "Geometry": "geometry",
                "Intermediate Algebra": "intermediate_algebra",
                "Number Theory": "number_theory",
                "Prealgebra": "prealgebra",
                "Precalculus": "precalculus",
            }
            valid_levels = {f"Level {i}" for i in range(1, 6)}
            multi_space_re = re.compile(r"\n{3,}")

            def _normalize_csv_arg(raw_value: str) -> str:
                value = str(raw_value).strip()
                if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
                    value = value[1:-1].strip()
                return value

            def _last_boxed_only_string(text: str):
                idx = text.rfind("\\boxed")
                if "\\boxed " in text:
                    return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
                if idx < 0:
                    idx = text.rfind("\\fbox")
                    if idx < 0:
                        return None

                i = idx
                right_brace_idx = None
                num_left_braces_open = 0
                while i < len(text):
                    if text[i] == "{":
                        num_left_braces_open += 1
                    if text[i] == "}":
                        num_left_braces_open -= 1
                        if num_left_braces_open == 0:
                            right_brace_idx = i
                            break
                    i += 1

                if right_brace_idx is None:
                    return None
                return text[idx:right_brace_idx + 1]

            def _remove_boxed(s: str) -> str:
                if s is None:
                    raise ValueError("remove_boxed received None")
                if "\\boxed " in s:
                    left = "\\boxed "
                    if s.startswith(left):
                        return s[len(left):]
                left = "\\boxed{" 
                if s.startswith(left) and s.endswith("}"):
                    return s[len(left):-1]
                left = "\\fbox{" 
                if s.startswith(left) and s.endswith("}"):
                    return s[len(left):-1]
                return s

            def _extract_final_answer_from_solution(solution: str) -> str:
                boxed = _last_boxed_only_string(solution)
                if boxed is None:
                    raise ValueError(f"Could not find final boxed answer in solution: {solution[:300]!r}")
                return _remove_boxed(boxed).strip()

            def _clean_solution_for_reasoning(solution: str, final_answer: str) -> str:
                text = solution.strip()
                boxed = _last_boxed_only_string(text)
                if boxed is not None:
                    text = text.replace(boxed, final_answer)

                text = text.replace("$", "")
                text = text.replace("\\[", "")
                text = text.replace("\\]", "")
                text = text.replace("\\(", "")
                text = text.replace("\\)", "")
                text = text.replace("\\boxed", "")
                text = text.replace("\\fbox", "")
                text = multi_space_re.sub("\n\n", text)
                return text.strip()

            def _build_target(solution: str, rng: random.Random, train_on_mixed_formats: bool) -> str:
                final_answer = _extract_final_answer_from_solution(solution)
                rationale = _clean_solution_for_reasoning(solution, final_answer)

                long_target = f"{rationale}\nFinal Answer: ${final_answer}$"
                short_target = f"Final Answer: ${final_answer}$"
                minimal_target = f"${final_answer}$"

                if not train_on_mixed_formats:
                    return long_target

                draw = rng.random()
                if draw < 0.70:
                    return long_target
                if draw < 0.90:
                    return short_target
                return minimal_target

            subjects_arg = _normalize_csv_arg(getattr(self.args, 'math_subjects', 'all'))
            if subjects_arg.lower() == 'all':
                subjects = list(subject_to_config.keys())
            else:
                subjects = [
                    _normalize_csv_arg(s) for s in subjects_arg.split(',') if _normalize_csv_arg(s)
                ]
                invalid = [s for s in subjects if s not in subject_to_config]
                if invalid:
                    raise ValueError(f"Invalid math subjects: {invalid}")

            datasets_per_subject = []
            for subject in subjects:
                config_name = subject_to_config[subject]
                ds = load_dataset(
                    getattr(self.args, 'math_official_dataset_path', 'EleutherAI/hendrycks_math'),
                    config_name,
                    split="train",
                )
                ds = ds.map(lambda ex, subject=subject: {"type": subject})
                datasets_per_subject.append(ds)

            dataset_source = getattr(self.args, 'math_dataset_source', 'official')
            if dataset_source == 'official':
                dataset = concatenate_datasets(datasets_per_subject)
            else:
                dataset = load_dataset(
                    getattr(self.args, 'math_flat_dataset_path', 'qwedsacf/competition_math'),
                    split='train'
                )
                subject_set = set(subjects)
                dataset = dataset.filter(lambda ex: ex.get('type') in subject_set)

            levels_arg = _normalize_csv_arg(getattr(self.args, 'math_levels', 'all'))
            if levels_arg.lower() != 'all':
                levels = []
                for item in levels_arg.split(','):
                    item = _normalize_csv_arg(item)
                    if not item:
                        continue
                    if item.startswith("Level "):
                        lvl = item
                    else:
                        lvl = f"Level {int(item)}"
                    if lvl not in valid_levels:
                        raise ValueError(f"Invalid math level: {item}")
                    levels.append(lvl)
                allowed_levels = set(levels)
                dataset = dataset.filter(lambda ex: ex.get("level") in allowed_levels)

            dataset = dataset.shuffle(seed=getattr(self.args, 'seed', 42))
            max_samples = getattr(self.args, 'math_samples', 0)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"✓ Limited to {max_samples} samples")
                else:
                    self.logger.info(f"✓ Using all {len(dataset)} samples (math_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(dataset)} samples (math_samples=0 or not specified)")

            max_length = getattr(self.args, 'max_length', 512)
            train_on_mixed_formats = bool(getattr(self.args, 'math_train_on_mixed_formats', False))
            use_chat_template = bool(getattr(self.args, 'math_use_chat_template', False))
            system_prompt = getattr(
                self.args,
                'math_system_prompt',
                "You are a careful competition math solver. Solve the problem step by step. On the last line, write exactly one final answer in the form: Final Answer: $<answer>$. Do not use additional dollar signs earlier in the response.",
            )
            seed = int(getattr(self.args, 'seed', 42))
            num_proc = None

            def render_prompt_only_plain(problem: str) -> str:
                return f"Problem: {problem}\nAnswer:"

            def render_full_plain(problem: str, target_text: str) -> str:
                return f"Problem: {problem}\nAnswer: {target_text}"

            def render_prompt_only_chat(problem: str) -> str:
                prompt_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Problem: {problem}\nAnswer:"},
                ]
                return self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            def render_full_chat(problem: str, target_text: str) -> str:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Problem: {problem}\nAnswer:"},
                    {"role": "assistant", "content": target_text},
                ]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

            def tokenize_math_example(ex, idx: int) -> Dict[str, List[int]]:
                problem = ex.get("problem", "").strip()
                solution = ex.get("solution", "").strip()
                rng = random.Random(seed + idx)

                target_text = _build_target(solution, rng, train_on_mixed_formats)
                if self.tokenizer.eos_token:
                    target_text = target_text + self.tokenizer.eos_token

                if use_chat_template:
                    prompt_text = render_prompt_only_chat(problem)
                    full_text = render_full_chat(problem, target_text)
                else:
                    prompt_text = render_prompt_only_plain(problem)
                    full_text = render_full_plain(problem, target_text)

                full_encoded = self.tokenizer(full_text, max_length=max_length, truncation=True)
                prompt_encoded = self.tokenizer(prompt_text, max_length=max_length, truncation=True)

                input_ids = full_encoded["input_ids"]
                attention_mask = full_encoded["attention_mask"]
                prompt_ids = prompt_encoded["input_ids"]

                labels = input_ids.copy()
                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

            self.dataset = dataset.map(
                tokenize_math_example,
                with_indices=True,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc="Tokenizing Hendrycks MATH",
            )

            self.logger.info(f"✓ Hendrycks MATH dataset created ({len(self.dataset)} samples)")
            self.logger.info(f"  - Subjects: {subjects_arg}")
            self.logger.info(f"  - Levels: {levels_arg}")
            self.logger.info(f"  - Dataset source: {dataset_source}")
            self.logger.info(f"  - Mixed formats: {train_on_mixed_formats}")
            self.logger.info(f"  - Chat template: {use_chat_template}")

        except Exception as e:
            self.logger.error(f"Failed to load Hendrycks MATH data: {str(e)}", exc_info=True)
            raise
    
    def load_utility_data(self):
        """호환성 유지를 위한 wrapper (load_training_data 호출)"""
        self.load_training_data()
    
    def setup_warp_modules(self):
        """
        WaRP 모듈 설정: identity basis(no rotation), mask 적용

        ✅ no rotation:
        - U = I(identity)
        - basis_coeff = W
        - 즉 회전 없이 원 좌표계에서 mask를 적용

        ✅ 공통:
        - coeff_mask 설정
        - flag = True (WaRP 모드)
        
        Forward는 WaRP 모듈이 자동으로 처리:
        W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Setting up WaRP modules with identity basis (no rotation) and masks")
            self.logger.info("="*70)
            self.warp_monitors = []
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            setup_count = 0
            
            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)

                    if key not in self.masks:
                        continue
                    
                    # 타겟 모듈 (WaRP 모듈)
                    target_module = self._get_target_module(layer, layer_type)
                    
                    if not isinstance(target_module, WaRPModule):
                        continue
                    
                    # 원본 가중치
                    W_original = target_module.weight.data.clone()
                    
                    # no rotation: U=I, 따라서 basis_coeff = W
                    in_dim = W_original.shape[1]
                    U_matrix = torch.eye(in_dim, dtype=W_original.dtype, device=W_original.device)

                    # basis_coeff 초기화: W @ I = W
                    basis_coeff_init = W_original @ U_matrix
                    
                    # ✅ WaRP 모듈 설정
                    target_module.basis_coeff.data = basis_coeff_init
                    target_module.UT_forward = U_matrix.clone().detach()
                    target_module.UT_backward = torch.empty(0, dtype=W_original.dtype, device=W_original.device)
                    
                    # ✅ 마스크 설정
                    mask = self.masks[key]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    mask = mask.to(device=W_original.device)
                    if mask.dtype != torch.bool:
                        mask = mask > 0.5
                    target_module.coeff_mask.data = mask

                    if hasattr(target_module, 'mask_mode'):
                        if torch.all(mask):
                            target_module.mask_mode.fill_(2)
                        elif torch.any(mask):
                            target_module.mask_mode.fill_(0)
                        else:
                            target_module.mask_mode.fill_(1)

                    self._register_warp_monitor(target_module, layer_idx, layer_type, mask)
                    
                    # ✅ WaRP 모드 활성화
                    target_module.flag = True
                    
                    # basis_coeff를 학습 가능하게 설정
                    target_module.basis_coeff.requires_grad = True
                    
                    setup_count += 1
                    
                    frozen_count = mask.sum().item()
                    total_count = mask.numel()
                    frozen_ratio = frozen_count / total_count * 100
                    
                    self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                    self.logger.info(f"  ✓ basis_coeff: {basis_coeff_init.shape}")
                    self.logger.info(f"  ✓ mask: Frozen {frozen_count}/{total_count} ({frozen_ratio:.1f}%)")
            
            self.logger.info("="*70)
            self.logger.info(f"✓ Setup completed: {setup_count} WaRP modules")
            self.logger.info("="*70)
            self.logger.info("Forward에서 자동으로 마스킹 적용:")
            self.logger.info("  W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U")
            self.logger.info("  mask=1: gradient 차단 (동결)")
            self.logger.info("  mask=0: gradient 흐름 (학습 가능)")
            self.logger.info("="*70)
            self._log_warp_monitor_overview()
            
        except Exception as e:
            self.logger.error(f"Failed to setup WaRP modules: {str(e)}", exc_info=True)
            raise

    def _sample_mask_indices(self, flat_mask, want_true, sample_count):
        if sample_count <= 0 or flat_mask.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=flat_mask.device)

        max_trials = max(256, sample_count * 256)
        selected = []
        selected_set = set()
        trials = 0
        numel = flat_mask.numel()

        while len(selected) < sample_count and trials < max_trials:
            idx = int(torch.randint(0, numel, (1,), device=flat_mask.device).item())
            if idx in selected_set:
                trials += 1
                continue

            is_true = bool(flat_mask[idx].item())
            if is_true == want_true:
                selected.append(idx)
                selected_set.add(idx)

            trials += 1

        if not selected:
            return torch.empty(0, dtype=torch.long, device=flat_mask.device)
        return torch.tensor(selected, dtype=torch.long, device=flat_mask.device)

    def _register_warp_monitor(self, module, layer_idx, layer_type, mask):
        flat_mask = mask.reshape(-1)
        sample_count = max(1, self.monitor_samples_per_group)
        masked_idx = self._sample_mask_indices(flat_mask, True, sample_count)
        trainable_idx = self._sample_mask_indices(flat_mask, False, sample_count)

        coeff_flat = module.basis_coeff.detach().reshape(-1)
        masked_init = (
            coeff_flat[masked_idx].detach().cpu().float()
            if masked_idx.numel() > 0
            else torch.empty(0, dtype=torch.float32)
        )
        trainable_init = (
            coeff_flat[trainable_idx].detach().cpu().float()
            if trainable_idx.numel() > 0
            else torch.empty(0, dtype=torch.float32)
        )

        self.warp_monitors.append({
            'module': module,
            'layer_idx': layer_idx,
            'layer_type': layer_type,
            'masked_idx': masked_idx,
            'trainable_idx': trainable_idx,
            'masked_init': masked_init,
            'trainable_init': trainable_init,
        })

    def _log_warp_monitor_overview(self):
        if not self.warp_monitors:
            self.logger.warning("[WaRP-Monitor] No monitor entries were registered.")
            return

        masked_samples = sum(int(item['masked_idx'].numel()) for item in self.warp_monitors)
        trainable_samples = sum(int(item['trainable_idx'].numel()) for item in self.warp_monitors)

        self.logger.info("[WaRP-Monitor] verification probes registered")
        self.logger.info(f"  - modules: {len(self.warp_monitors)}")
        self.logger.info(f"  - masked probes: {masked_samples}")
        self.logger.info(f"  - trainable probes: {trainable_samples}")

    def _run_mask_gradient_sanity_check(self):
        if not self.warp_monitors:
            self.logger.warning("[WaRP-GradCheck] skip: monitor entries not found")
            return
        if not hasattr(self, 'dataset') or self.dataset is None or len(self.dataset) == 0:
            self.logger.warning("[WaRP-GradCheck] skip: dataset is empty")
            return

        sample = self.dataset[0]
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long, device=self.model.device).unsqueeze(0)
        attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long, device=self.model.device).unsqueeze(0)
        labels = torch.tensor(sample['labels'], dtype=torch.long, device=self.model.device).unsqueeze(0)

        self.model.zero_grad(set_to_none=True)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss

        if loss is None or not torch.isfinite(loss):
            self.logger.warning("[WaRP-GradCheck] skip: non-finite loss on sanity batch")
            self.model.zero_grad(set_to_none=True)
            return

        loss.backward()

        masked_sum = 0.0
        masked_count = 0
        masked_nonzero = 0
        trainable_sum = 0.0
        trainable_count = 0
        trainable_nonzero = 0
        eps = 1e-12

        for item in self.warp_monitors:
            grad = item['module'].basis_coeff.grad
            if grad is None:
                continue
            grad_flat = grad.detach().reshape(-1)

            masked_idx = item['masked_idx']
            if masked_idx.numel() > 0:
                masked_vals = grad_flat[masked_idx].abs().float()
                masked_sum += masked_vals.sum().item()
                masked_count += masked_vals.numel()
                masked_nonzero += int((masked_vals > eps).sum().item())

            trainable_idx = item['trainable_idx']
            if trainable_idx.numel() > 0:
                trainable_vals = grad_flat[trainable_idx].abs().float()
                trainable_sum += trainable_vals.sum().item()
                trainable_count += trainable_vals.numel()
                trainable_nonzero += int((trainable_vals > eps).sum().item())

        masked_mean = masked_sum / max(masked_count, 1)
        trainable_mean = trainable_sum / max(trainable_count, 1)
        masked_nonzero_ratio = (masked_nonzero / max(masked_count, 1)) * 100.0
        trainable_nonzero_ratio = (trainable_nonzero / max(trainable_count, 1)) * 100.0

        self.logger.info("[WaRP-GradCheck] one-batch gradient sanity")
        self.logger.info(f"  - loss: {loss.item():.6f}")
        self.logger.info(
            f"  - masked(=1) grad |mean|: {masked_mean:.6e}, nonzero ratio: {masked_nonzero_ratio:.2f}%"
        )
        self.logger.info(
            f"  - trainable(=0) grad |mean|: {trainable_mean:.6e}, nonzero ratio: {trainable_nonzero_ratio:.2f}%"
        )

        self.model.zero_grad(set_to_none=True)

    def _log_warp_parameter_delta_summary(self):
        if not self.warp_monitors:
            self.logger.warning("[WaRP-ParamCheck] skip: monitor entries not found")
            return

        masked_sum = 0.0
        masked_count = 0
        masked_changed = 0
        trainable_sum = 0.0
        trainable_count = 0
        trainable_changed = 0
        eps = 1e-12

        for item in self.warp_monitors:
            coeff_flat = item['module'].basis_coeff.detach().reshape(-1)

            masked_idx = item['masked_idx']
            if masked_idx.numel() > 0:
                current_masked = coeff_flat[masked_idx].detach().cpu().float()
                masked_delta = (current_masked - item['masked_init']).abs()
                masked_sum += masked_delta.sum().item()
                masked_count += masked_delta.numel()
                masked_changed += int((masked_delta > eps).sum().item())

            trainable_idx = item['trainable_idx']
            if trainable_idx.numel() > 0:
                current_trainable = coeff_flat[trainable_idx].detach().cpu().float()
                trainable_delta = (current_trainable - item['trainable_init']).abs()
                trainable_sum += trainable_delta.sum().item()
                trainable_count += trainable_delta.numel()
                trainable_changed += int((trainable_delta > eps).sum().item())

        masked_mean = masked_sum / max(masked_count, 1)
        trainable_mean = trainable_sum / max(trainable_count, 1)
        masked_changed_ratio = (masked_changed / max(masked_count, 1)) * 100.0
        trainable_changed_ratio = (trainable_changed / max(trainable_count, 1)) * 100.0

        self.logger.info("[WaRP-ParamCheck] post-train sampled delta")
        self.logger.info(
            f"  - masked(=1) delta |mean|: {masked_mean:.6e}, changed ratio: {masked_changed_ratio:.2f}%"
        )
        self.logger.info(
            f"  - trainable(=0) delta |mean|: {trainable_mean:.6e}, changed ratio: {trainable_changed_ratio:.2f}%"
        )

        if masked_mean > 1e-9:
            self.logger.warning(
                "[WaRP-ParamCheck] masked coefficients changed noticeably. "
                "(optimizer weight decay/설정 확인 필요)"
            )
        if trainable_mean <= 1e-12:
            self.logger.warning(
                "[WaRP-ParamCheck] trainable coefficients changed very little. "
                "(learning rate/gradient 흐름 확인 필요)"
            )
    
    def train(self):
        """
        Phase 3: Incremental Learning with Dataset Selection
        
        ✅ GSM8K: SFTTrainer 방식 (원본 형태 유지)
        ✅ MetaMath: SFTTrainer 방식 (원본 형태 유지)
        ✅ Safety: phase0_SSFT 스타일의 custom training loop
        ✅ 모두 WaRP masking 자동 적용 (basis_coeff만 학습)
        """
        try:
            # Dataset 타입 확인
            phase3_dataset = getattr(self.args, 'phase3_dataset', 'gsm8k')
            
            if phase3_dataset == 'safety':
                # Safety dataset: Custom training loop (phase0_SSFT 방식)
                self.logger.info("\n" + "="*70)
                self.logger.info("Using phase0_SSFT style training loop for Safety dataset")
                self.logger.info("="*70 + "\n")
                return self._train_safety_custom_loop()
            else:
                # GSM8K, MetaMath: SFTTrainer 방식 (원본 형태)
                dataset_name_map = {
                    'gsm8k': 'GSM8K',
                    'metamath': 'MetaMath',
                    'math': 'Hendrycks MATH',
                }
                dataset_name = dataset_name_map.get(phase3_dataset, phase3_dataset)
                self.logger.info("\n" + "="*70)
                self.logger.info(f"Using SFTTrainer for {dataset_name} dataset")
                self.logger.info("="*70 + "\n")
                return self._train_with_trainer()
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
    
    def _train_safety_custom_loop(self):
        """
        Safety Dataset Training with Custom Loop (phase0_SSFT 방식)
        
        ✅ phase0_SSFT와 동일한 훈련:
        - Custom training loop with manual optimizer management
        - AdamW8bit optimizer
        - Gradient clipping
        - 매 배치 loss 출력
        
        ✅ WaRP 특화:
        - basis_coeff만 학습 가능
        - 마스크 기반 gradient 자동 차단
        """
        try:
            from bitsandbytes.optim import AdamW8bit
            from torch.utils.data import DataLoader
            
            self.logger.info("="*70)
            self.logger.info("Phase 3: Safety Incremental Learning (Custom Loop/phase0_SSFT)")
            self.logger.info("="*70)
            
            # 훈련 설정 (phase0_SSFT 방식)
            epochs = getattr(self.args, 'epochs', 3)
            learning_rate = getattr(self.args, 'utility_lr', 1e-5)
            batch_size = self.args.batch_size
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)
            max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
            device = self.args.device
            dtype = self.args.dtype
            
            # ✅ basis_coeff만 학습 가능하게 설정
            trainable_params = 0
            total_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                
                # basis_coeff만 학습 가능
                if 'basis_coeff' in name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            
            self.logger.info(f"Parameter freeze status:")
            self.logger.info(f"  - Total params: {total_params:,}")
            self.logger.info(f"  - Trainable params: {trainable_params:,}")
            self.logger.info(f"  - Frozen params: {total_params - trainable_params:,}")
            self.logger.info(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")
            
            # Checkpoint 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                getattr(self.args, 'output_dir', './checkpoints'),
                f'phase3_{timestamp}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Batch size: {batch_size}")
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
            self.logger.info(f"  - Max gradient norm: {max_grad_norm}")
            self.logger.info(f"  - Optimizer: adamw_bnb_8bit")
            self.logger.info(f"  - Device: {device}")
            self.logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
            self.logger.info("="*70)
            
            # DataLoader 사용 (이미 _load_safety_dataset에서 collate_fn과 함께 생성됨)
            train_dataloader = self.train_loader
            self.logger.info(f"DataLoader created: {len(train_dataloader)} batches")
            self.logger.info(f"  - Total samples: {len(self.dataset)}")
            
            # 모델을 device에 배치
            self.model = self.model.to(device)
            self.model.train()
            
            # Optimizer 초기화 (trainable 파라미터만)
            optimizer = AdamW8bit(self.model.parameters(), lr=learning_rate)
            
            # 마스크 기반 gradient 동작 사전 점검 (1배치)
            self._run_mask_gradient_sanity_check()
            
            # Training loop
            total_loss = 0.0
            total_steps = 0
            optimizer_steps = 0
            
            self.logger.info("Starting training...")
            self.logger.info("  WaRP forward will automatically apply masking:")
            self.logger.info("    W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U")
            self.logger.info("    mask=1: gradient 차단 (frozen)")
            self.logger.info("    mask=0: gradient 흐름 (trainable)")
            self.logger.info("="*70 + "\n")
            
            for epoch in range(epochs):
                self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")
                epoch_loss = 0.0
                
                pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
                for batch_idx, batch in enumerate(pbar):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    if batch_idx == 0:
                        self.logger.info("\n[First Batch Info]")
                        self.logger.info(f"  Batch size: {input_ids.shape[0]}")
                        self.logger.info(f"  Sequence length: {input_ids.shape[1]}")
                        self.logger.info(f"  Device: {input_ids.device}")
                        valid_labels = (labels != -100).sum().item()
                        self.logger.info(f"  Valid labels (non-padding): {valid_labels}/{labels.numel()}")
                    
                    # Gradient accumulation 시작
                    if batch_idx % gradient_accumulation_steps == 0:
                        optimizer.zero_grad(set_to_none=True)
                    
                    # Autocast (bfloat16/float16)
                    use_autocast = device.startswith("cuda") or device.startswith("cpu")
                    if dtype == 'bfloat16':
                        autocast_dtype = torch.bfloat16
                    elif dtype == 'float16':
                        autocast_dtype = torch.float16
                    else:
                        autocast_dtype = torch.float32
                    
                    with torch.autocast(
                        device_type=device if use_autocast else "cuda",
                        dtype=autocast_dtype,
                        enabled=use_autocast
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True,
                        )
                        loss = outputs.loss
                    
                    # NaN/Inf 처리
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf detected at batch {batch_idx + 1}. Skipping this batch.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # Backprop with gradient accumulation
                    (loss / gradient_accumulation_steps).backward()
                    
                    # Optimizer step (accumulation step 도달 시 또는 마지막 배치)
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        optimizer.step()
                        optimizer_steps += 1
                    
                    loss_val = loss.item()
                    total_loss += loss_val
                    epoch_loss += loss_val
                    total_steps += 1
                    
                    # 진행 상황 업데이트
                    if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                        avg_batch_loss = epoch_loss / (batch_idx + 1)
                        pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})
                        self.logger.info(f"  Batch {batch_idx + 1}: loss = {loss_val:.4f}")
                
                epoch_avg_loss = epoch_loss / len(train_dataloader)
                self.logger.info(f"✓ Epoch {epoch + 1} completed - Epoch Loss: {epoch_avg_loss:.4f}")
            
            # Training 완료
            avg_loss = total_loss / max(1, total_steps)
            self.logger.info(f"\n{'='*70}")
            self.logger.info("Training Complete")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Average loss: {avg_loss:.4f}")
            self.logger.info(f"Total steps: {total_steps}")
            self.logger.info(f"Optimizer steps: {optimizer_steps}")
            self.logger.info(f"Training time: {epochs} epoch(s)")
            self.logger.info(f"{'='*70}\n")
            
            # 마스크/학습 반영 여부 사후 점검
            self._log_warp_parameter_delta_summary()
            
            # 가중치 복원: basis_coeff → weight (W = basis_coeff @ U^T)
            restore_weight(self.model)
            
            # WaRP 모듈 → 표준 nn.Linear 변환
            restore_to_linear(self.model)
            
            # 최종 모델 저장
            final_model_path = os.path.join(checkpoint_dir, 'final_model')
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 3 (Safety) Completed")
            self.logger.info(f"  - Final model: {final_model_path}")
            self.logger.info("="*70)
            
            # 메타데이터 저장
            metadata = {
                'phase': 3,
                'dataset': 'safety',
                'trainer': 'custom_loop',
                'basis_dir': self.basis_dir,
                'masks_dir': self.masks_dir,
                'phase0_model': self.phase0_model_dir,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'optimizer': 'adamw_bnb_8bit',
                'max_grad_norm': max_grad_norm,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'total_samples': len(self.dataset),
                'trainable_params': trainable_params,
                'total_params': total_params,
                'average_loss': avg_loss,
                'total_steps': total_steps,
                'optimizer_steps': optimizer_steps,
                'timestamp': timestamp,
            }
            
            metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✓ Metadata saved to: {metadata_path}")
            
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Safety training failed: {str(e)}", exc_info=True)
            raise
    
    def _train_with_trainer(self):
        """
        Trainer-based Training (GSM8K, MetaMath)
        
        ✅ 원본 학습:
        - SFTTrainer 사용
        - 자동 gradient accumulation
        - Trainer 기반 안정적인 학습 루프
        
        ✅ WaRP 특화:
        - basis_coeff만 학습 가능
        - 마스크 기반 gradient 자동 차단
        """
        try:
            from transformers import Trainer, TrainingArguments
            
            phase3_dataset = getattr(self.args, 'phase3_dataset', 'gsm8k')
            dataset_name_map = {
                'gsm8k': 'GSM8K',
                'metamath': 'MetaMath',
                'math': 'Hendrycks MATH',
            }
            dataset_name = dataset_name_map.get(phase3_dataset, phase3_dataset)
            
            self.logger.info("="*70)
            self.logger.info(f"Phase 3: Incremental Learning with {dataset_name} (Trainer/SFT)")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'epochs', 3)
            learning_rate = getattr(self.args, 'utility_lr', 1e-5)
            configured_weight_decay = getattr(self.args, 'base_weight_decay', 0.01)
            effective_weight_decay = 0.0 if configured_weight_decay > 0 else configured_weight_decay
            batch_size = self.args.batch_size
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 4)

            if configured_weight_decay > 0:
                self.logger.warning(
                    "Strict mask freeze를 위해 freeze 모드 Phase 3에서 weight_decay를 0.0으로 강제합니다 "
                    f"(requested={configured_weight_decay})."
                )
            
            # ✅ basis_coeff만 학습 가능하게 설정
            trainable_params = 0
            total_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                
                # basis_coeff만 학습 가능
                if 'basis_coeff' in name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            
            self.logger.info(f"Parameter freeze status:")
            self.logger.info(f"  - Total params: {total_params:,}")
            self.logger.info(f"  - Trainable params: {trainable_params:,}")
            self.logger.info(f"  - Frozen params: {total_params - trainable_params:,}")
            self.logger.info(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")
            
            # Checkpoint 디렉토리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                getattr(self.args, 'output_dir', './checkpoints'),
                f'phase3_{timestamp}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  - Epochs: {epochs}")
            self.logger.info(f"  - Learning rate: {learning_rate}")
            self.logger.info(f"  - Weight decay (requested): {configured_weight_decay}")
            self.logger.info(f"  - Weight decay (effective): {effective_weight_decay}")
            self.logger.info(f"  - Warmup ratio: 0.1")
            self.logger.info(f"  - Batch size: {batch_size}")
            self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
            self.logger.info(f"  - Max gradient norm: 1.0")
            self.logger.info(f"  - Optimizer: adamw_bnb_8bit")
            self.logger.info(f"  - LR scheduler: cosine")
            self.logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
            self.logger.info("="*70)
            
            warmup_ratio = getattr(self.args, 'warmup_ratio', 0.1)
            lr_scheduler_type = getattr(self.args, 'lr_scheduler_type', 'cosine')
            max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
            logging_steps = getattr(self.args, 'logging_steps', 10)
            gradient_checkpointing = getattr(self.args, 'gradient_checkpointing', False)

            training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=effective_weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                max_grad_norm=max_grad_norm,
                logging_steps=logging_steps,
                save_strategy="no",
                eval_strategy="no",
                bf16=True if self.args.dtype == 'bfloat16' else False,
                fp16=True if self.args.dtype == 'float16' else False,
                report_to="none",
                remove_unused_columns=False,
                optim="adamw_bnb_8bit",
                gradient_checkpointing=gradient_checkpointing,
            )

            @dataclass
            class DataCollatorForCausalLMWithPadding:
                tokenizer: object

                def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
                    max_len = max(len(f["input_ids"]) for f in features)
                    pad_id = self.tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = self.tokenizer.eos_token_id

                    input_ids, attention_mask, labels = [], [], []
                    for f in features:
                        l = len(f["input_ids"])
                        pad_len = max_len - l
                        input_ids.append(f["input_ids"] + [pad_id] * pad_len)
                        attention_mask.append(f["attention_mask"] + [0] * pad_len)
                        labels.append(f["labels"] + [-100] * pad_len)

                    return {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }

            data_collator = DataCollatorForCausalLMWithPadding(self.tokenizer)
            
            # ✅ model.train() 모드
            self.model.train()

            # 마스크 기반 gradient 동작 사전 점검 (1배치)
            self._run_mask_gradient_sanity_check()
            
            # Trainer 초기화
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            self.logger.info("✓ Trainer initialized")
            self.logger.info("Starting training...")
            self.logger.info("  WaRP forward will automatically apply masking:")
            self.logger.info("    W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U")
            self.logger.info("    mask=1: gradient 차단 (frozen)")
            self.logger.info("    mask=0: gradient 흐름 (trainable)")
            
            # 훈련 시작
            trainer.train()
            
            self.logger.info("✓ Training completed")

            # 마스크/학습 반영 여부 사후 점검 (샘플 인덱스 기반)
            self._log_warp_parameter_delta_summary()
            
            # 가중치 복원: basis_coeff → weight (W = basis_coeff @ U^T)
            restore_weight(self.model)

            # WaRP 모듈 → 표준 nn.Linear 변환 (버퍼/파라미터 제거 → 용량 정상화)
            restore_to_linear(self.model)

            # 최종 모델 저장 (표준 HuggingFace 구조)
            final_model_path = os.path.join(checkpoint_dir, 'final_model')
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            self.logger.info("="*70)
            self.logger.info("Phase 3 Completed")
            self.logger.info(f"  - Final model: {final_model_path}")
            self.logger.info("="*70)
            
            # 메타데이터 저장
            metadata = {
                'phase': 3,
                'trainer': 'Trainer',
                'basis_dir': self.basis_dir,
                'masks_dir': self.masks_dir,
                'phase0_model': self.phase0_model_dir,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay_configured': configured_weight_decay,
                'weight_decay_effective': effective_weight_decay,
                'warmup_ratio': 0.1,
                'optimizer': 'adamw_bnb_8bit',
                'lr_scheduler': 'cosine',
                'max_grad_norm': 1.0,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'total_samples': len(self.dataset),
                'trainable_params': trainable_params,
                'total_params': total_params,
                'timestamp': timestamp,
            }
            
            metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✓ Metadata saved to: {metadata_path}")
            
            return final_model_path
            
        except ImportError:
            self.logger.error("TRL library not found! Install with: pip install trl")
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
    
    def _get_target_module(self, layer, layer_type):
        """Layer type에 맞는 모듈 반환"""
        if layer_type == 'ffn_down':
            return layer.mlp.down_proj
        elif layer_type == 'ffn_gate':
            return layer.mlp.gate_proj
        elif layer_type == 'ffn_up':
            return layer.mlp.up_proj
        elif layer_type == 'attn_q':
            return layer.self_attn.q_proj
        elif layer_type == 'attn_k':
            return layer.self_attn.k_proj
        elif layer_type == 'attn_v':
            return layer.self_attn.v_proj
        elif layer_type == 'attn_o':
            return layer.self_attn.o_proj
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
