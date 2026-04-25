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


class Phase3IncrementalLearner:
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

    def _is_instruct_model(self) -> bool:
        model_ref = self.phase0_model_dir.lower()
        return any(tag in model_ref for tag in ('instruct', 'chat'))

    def _build_question_answer_prompt(self, question: str) -> str:
        return f"Question: {question.strip()}\nAnswer:"

    def _tokenize_question_answer_example(
        self,
        question: str,
        response: str,
        max_length: int,
        add_eos: bool = True,
    ) -> Dict[str, List[int]]:
        question = str(question).strip()
        response = str(response).strip()

        if self._is_instruct_model():
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                prompt_ids = self.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                full_ids = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]

                labels = full_ids.copy()
                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100

                attention_mask = [1] * len(full_ids)
                return {
                    "input_ids": full_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            except Exception as e:
                self.logger.warning(
                    f"Failed to apply chat template for instruct model; falling back to plain Question/Answer format. Error: {e}"
                )

        prompt_text = self._build_question_answer_prompt(question)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        remain = max(0, max_length - len(prompt_ids))
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, remain) if remain > 0 else 1,
        )["input_ids"]

        if remain == 0:
            response_ids = []
        else:
            response_ids = response_ids[:remain]

        if (
            add_eos
            and self.tokenizer.eos_token_id is not None
            and len(prompt_ids) + len(response_ids) < max_length
            and (len(response_ids) == 0 or response_ids[-1] != self.tokenizer.eos_token_id)
        ):
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[:max_length]
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def load_basis(self):
        """Phase 1의 basis 로드"""
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")
            
            # 메타데이터
            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                basis_metadata = json.load(f)
            
            # Layer types
            layer_types_str = self.args.layer_type
            layer_types = [lt.strip() for lt in layer_types_str.split(',')]
            self.layer_types = layer_types
            
            # Basis 로드
            total_loaded = 0
            
            for layer_type in layer_types:
                layer_type_dir = os.path.join(self.basis_dir, layer_type)
                if not os.path.exists(layer_type_dir):
                    continue
                
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
            
            self.logger.info(f"✓ Basis loaded: {total_loaded} combinations")
            
        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
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
            self.logger.info(
                f"  - Input formatting: {'chat template' if self._is_instruct_model() else 'Question/Answer plain text'}"
            )
            
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
        from datasets import load_dataset
        
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
            
            max_length = getattr(self.args, 'max_length', 1024)

            tokenized_data = []
            for ex in dataset:
                question = ex.get("question", "")
                answer = ex.get("answer", "")
                tokenized_data.append(self._tokenize_question_answer_example(question, answer, max_length=max_length))

            from datasets import Dataset as HFDataset
            self.dataset = HFDataset.from_dict({
                "input_ids": [d["input_ids"] for d in tokenized_data],
                "attention_mask": [d["attention_mask"] for d in tokenized_data],
                "labels": [d["labels"] for d in tokenized_data],
            })
            del tokenized_data
            
            self.logger.info(f"✓ Dataset created ({len(self.dataset)} samples)")
            
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K data: {str(e)}", exc_info=True)
            raise

    def _load_metamath(self):
        """
        MetaMath 데이터셋 로드

        형식: Question: ... Answer:
        라벨링: prompt는 -100 마스킹, response만 학습
        """
        from datasets import load_dataset

        try:
            self.logger.info("Loading MetaMath dataset...")

            dataset = load_dataset("meta-math/MetaMathQA", split="train")

            max_samples = getattr(self.args, 'metamath_samples', 0)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"✓ Limited to {max_samples} samples")
                else:
                    self.logger.info(f"✓ Using all {len(dataset)} samples (metamath_samples={max_samples} >= dataset size)")
            else:
                self.logger.info(f"✓ Using all {len(dataset)} samples (metamath_samples=0 or not specified)")

            max_length = getattr(self.args, 'max_length', 1024)

            tokenized_data = []
            for ex in dataset:
                query = ex.get("query", "")
                response = ex.get("response", "")
                tokenized_data.append(self._tokenize_question_answer_example(query, response, max_length=max_length))

            from datasets import Dataset as HFDataset
            self.dataset = HFDataset.from_dict({
                "input_ids": [d["input_ids"] for d in tokenized_data],
                "attention_mask": [d["attention_mask"] for d in tokenized_data],
                "labels": [d["labels"] for d in tokenized_data],
            })
            del tokenized_data

            self.logger.info(f"✓ MetaMath dataset created ({len(self.dataset)} samples)")

        except Exception as e:
            self.logger.error(f"Failed to load MetaMath data: {str(e)}", exc_info=True)
            raise

    def _load_safety_dataset(self):
        """
        Circuit Breakers 안전 데이터셋 로드
        
        형식: JSON with prompt + llama3_output
        라벨링: prompt는 -100 마스킹, response만 학습
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
            
            max_length = getattr(self.args, 'max_length', 1024)
            
            def tokenize_safety_example(item: Dict) -> Dict[str, List[int]]:
                harmful_prompt = item.get("prompt", "")
                safe_response = item.get("llama3_output", "")
                return self._tokenize_question_answer_example(
                    harmful_prompt,
                    safe_response,
                    max_length=max_length,
                )
            
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
            
        except Exception as e:
            self.logger.error(f"Failed to load Safety data: {str(e)}", exc_info=True)
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
                # Handle values passed as '"all"' or "'all'"
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

            # Subject filtering
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

            # Level filtering
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

            # Shuffle then optional subsampling
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

            max_length = getattr(self.args, 'max_length', 1024)
            train_on_mixed_formats = bool(getattr(self.args, 'math_train_on_mixed_formats', False))
            seed = int(getattr(self.args, 'seed', 42))
            tokenized_data = []
            for idx, ex in enumerate(dataset):
                problem = ex.get("problem", "").strip()
                solution = ex.get("solution", "").strip()
                rng = random.Random(seed + idx)
                target_text = _build_target(solution, rng, train_on_mixed_formats)
                tokenized_data.append(self._tokenize_question_answer_example(
                    problem, target_text, max_length=max_length,
                ))

            from datasets import Dataset as HFDataset
            self.dataset = HFDataset.from_dict({
                "input_ids": [d["input_ids"] for d in tokenized_data],
                "attention_mask": [d["attention_mask"] for d in tokenized_data],
                "labels": [d["labels"] for d in tokenized_data],
            })
            del tokenized_data

            self.logger.info(f"✓ Hendrycks MATH dataset created ({len(self.dataset)} samples)")
            self.logger.info(f"  - Subjects: {subjects_arg}")
            self.logger.info(f"  - Levels: {levels_arg}")
            self.logger.info(f"  - Dataset source: {dataset_source}")
            self.logger.info(f"  - Mixed formats: {train_on_mixed_formats}")
            self.logger.info("  - Prompt format: Question: ... Answer:")

        except Exception as e:
            self.logger.error(f"Failed to load Hendrycks MATH data: {str(e)}", exc_info=True)
            raise
    
    def load_utility_data(self):
        """호환성 유지를 위한 wrapper (load_training_data 호출)"""
        self.load_training_data()
    
    def setup_warp_modules(self):
        """
        WaRP 모듈 설정: basis, mask 적용
        
        ✅ 원본 WaRP:
        - basis_coeff, UT_forward, UT_backward 설정
        - coeff_mask 설정
        - flag = True (WaRP 모드)
        
        Forward는 WaRP 모듈이 자동으로 처리:
        W = V @ (basis_coeff * mask).detach() + basis_coeff * (1-mask) @ U
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Setting up WaRP modules with basis and masks")
            self.logger.info("="*70)
            self.warp_monitors = []
            
            target_indices = self._parse_target_layers(len(self.model.model.layers))
            
            setup_count = 0
            
            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)
                    
                    if key not in self.basis_data or key not in self.masks:
                        continue
                    
                    # 타겟 모듈 (WaRP 모듈)
                    target_module = self._get_target_module(layer, layer_type)
                    
                    if not isinstance(target_module, WaRPModule):
                        continue
                    
                    # 원본 가중치
                    W_original = target_module.weight.data.clone()
                    
                    # Basis (U에는 V = UT.t()가 저장되어 있음)
                    # 원본 WaRP: basis_coeff = W @ UT_forward.t() = W @ V
                    U_matrix = self.basis_data[key]['U']  # 실제로는 V (= UT.t())
                    U_matrix = U_matrix.to(dtype=W_original.dtype, device=W_original.device)
                    
                    # basis_coeff 초기화: W @ V (원본 WaRP 방식)
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

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'warp/sanity_loss': loss.item(),
                    'warp/sanity_masked_grad_mean': masked_mean,
                    'warp/sanity_masked_nonzero_pct': masked_nonzero_ratio,
                    'warp/sanity_trainable_grad_mean': trainable_mean,
                    'warp/sanity_trainable_nonzero_pct': trainable_nonzero_ratio,
                })
        except Exception:
            pass

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

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'warp/delta_masked_mean': masked_mean,
                    'warp/delta_masked_changed_pct': masked_changed_ratio,
                    'warp/delta_trainable_mean': trainable_mean,
                    'warp/delta_trainable_changed_pct': trainable_changed_ratio,
                })
        except Exception:
            pass
    
    def train(self):
        """
        Phase 3: Incremental Learning with Dataset Selection
        
        ✅ GSM8K / MetaMath / Safety / Hendrycks MATH: 모두 Trainer 방식
        ✅ 모두 WaRP masking 자동 적용 (basis_coeff만 학습)
        """
        try:
            return self._train_with_trainer()
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
    
    def _train_with_trainer(self):
        """
        Trainer-based Training (GSM8K, MetaMath, Safety, Hendrycks MATH)
        
        ✅ 모든 dataset 공통:
        - HuggingFace Trainer 사용
        - cosine LR scheduler + warmup
        - 자동 gradient accumulation
        
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
                'safety': 'Safety (Circuit Breakers)',
            }
            dataset_name = dataset_name_map.get(phase3_dataset, phase3_dataset)
            
            self.logger.info("="*70)
            self.logger.info(f"Phase 3: Incremental Learning with {dataset_name} (Trainer/SFT)")
            self.logger.info("="*70)
            
            # 훈련 설정
            epochs = getattr(self.args, 'epochs', 3)
            learning_rate = getattr(self.args, 'utility_lr', 3e-5)
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
                getattr(self.args, 'output_dir', '/lustre/gokms0509/Safety-WaRP-LLM/checkpoints'),
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
            self.logger.info(f"  - Optimizer: adamw_torch")
            self.logger.info(f"  - LR scheduler: cosine")
            self.logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
            self.logger.info("="*70)

            # W&B: 세션 시작 시 config 로깅
            try:
                import wandb as _wb3
                _wandb_enabled = _wb3.run is not None
            except Exception:
                _wandb_enabled = False

            if _wandb_enabled:
                try:
                    import wandb
                    wandb.log({
                        'phase3/trainable_params': trainable_params,
                        'phase3/total_params': total_params,
                        'phase3/trainable_ratio_pct': trainable_params / max(total_params, 1) * 100,
                        'phase3/frozen_params': total_params - trainable_params,
                        'phase3/frozen_ratio_pct': (total_params - trainable_params) / max(total_params, 1) * 100,
                        'phase3/dataset': phase3_dataset,
                        'phase3/learning_rate': learning_rate,
                        'phase3/epochs': epochs,
                        'phase3/batch_size': batch_size,
                        'phase3/grad_accum_steps': gradient_accumulation_steps,
                    }, step=0)
                except Exception:
                    pass

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
                report_to="wandb" if _wandb_enabled else "none",
                remove_unused_columns=False,
                optim="adamw_torch",
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

            if phase3_dataset == 'safety':
                self.logger.info("Safety dataset: shuffle disabled (sequential order)")

                def _get_train_dataloader_no_shuffle():
                    return DataLoader(
                        self.dataset,
                        batch_size=self.args.batch_size,
                        sampler=torch.utils.data.SequentialSampler(self.dataset),
                        collate_fn=data_collator,
                        drop_last=training_args.dataloader_drop_last,
                        num_workers=training_args.dataloader_num_workers,
                        pin_memory=training_args.dataloader_pin_memory,
                        persistent_workers=training_args.dataloader_persistent_workers,
                    )

                trainer.get_train_dataloader = _get_train_dataloader_no_shuffle
            
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
                'optimizer': 'adamw_torch',
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
