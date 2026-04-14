"""
Phase 2: Importance Scoring (Per-Layer keep_ratio)

각 WaRP layer에서 개별적으로 keep_ratio를 적용하여
레이어별 상위 keep_ratio만 동결(mask=1)합니다.

결과:
- 각 WaRP layer에서 frozen ratio = keep_ratio
- Phase 3에서 학습 가능 비율이 약 (1 - keep_ratio) * 100%

Two-Mask 모드 (--two_mask):
- preserve_mask: preserve 데이터셋(e.g. circuit_breakers)으로 계산한 중요 mask
- adapt_mask: adapt 데이터셋(e.g. MATH/GSM8K)으로 계산한 중요 mask
- final_mask = preserve_mask AND NOT adapt_mask
  → preserve에 중요하지만 adapt에는 중요하지 않은 파라미터만 동결
  → adapt에도 중요한 파라미터는 Phase 3에서 학습 가능하게 허용
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from datetime import datetime
import gc
import random
from typing import List

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from .warp_modules import switch_to_warp_module, WaRPModule

logger = logging.getLogger(__name__)


class Phase2ImportanceScorerPerLayer:
    """
    Phase 2: Importance Scoring (Per-layer keep_ratio)

    핵심:
    - model.eval() 모드
    - loss.backward()로 gradient 계산
    - optimizer.step() 없음
    - importance = |∂L/∂basis_coeff| 누적
    - 각 layer별 quantile로 mask 생성
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

        # Two-Mask: adapt 데이터셋용 dataloader & importances
        self.adapt_dataloader = None
        self.adapt_importances = {}

        # Basis 정보
        self.basis_data = {}
        self.layer_types = []

        # Importance 점수
        self.importances = {}
        self.masks = {}

        # 통계
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
        }
        self.adapt_stats = {
            'total_samples': 0,
            'total_tokens': 0,
        }

    def _is_instruct_model(self):
        return 'instruct' in str(self.phase0_model_dir).lower()

    def _format_text_for_model(self, user_text: str, assistant_text: str = None) -> str:
        user_text = str(user_text).strip()
        assistant_text = None if assistant_text is None else str(assistant_text).strip()

        if not self._is_instruct_model():
            if assistant_text is None:
                return user_text
            return f"Question: {user_text}\nAnswer: {assistant_text}"

        messages = [{"role": "user", "content": user_text}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to apply chat template for instruct model; falling back to plain format. Error: {e}"
            )
            if assistant_text is None:
                return user_text
            return f"Question: {user_text}\nAnswer: {assistant_text}"

    def _format_prompt_only_for_model(self, user_text: str) -> str:
        user_text = str(user_text).strip()

        if not self._is_instruct_model():
            return f"Question: {user_text}\nAnswer:"

        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to apply generation chat template for instruct model; falling back to plain prompt format. Error: {e}"
            )
            return f"Question: {user_text}\nAnswer:"

    def load_basis(self):
        try:
            self.logger.info(f"Loading basis from {self.basis_dir}...")

            metadata_path = os.path.join(self.basis_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                basis_metadata = json.load(f)

            self.logger.info("✓ Metadata loaded:")
            self.logger.info(f"  - Layer types: {basis_metadata.get('layer_types')}")

            layer_types_str = self.args.layer_type
            self.layer_types = [lt.strip() for lt in layer_types_str.split(',')]

            total_loaded = 0
            for layer_type in self.layer_types:
                layer_type_dir = os.path.join(self.basis_dir, layer_type)
                if not os.path.exists(layer_type_dir):
                    self.logger.warning(f"Layer type directory not found: {layer_type_dir}")
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

            self.logger.info(f"✓ Basis loaded: {total_loaded} (layer, type) combinations")

        except Exception as e:
            self.logger.error(f"Failed to load basis: {str(e)}", exc_info=True)
            raise

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self.logger.info(f"Loading Phase 0 trained model from {self.phase0_model_dir}...")

            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.phase0_model_dir,
                torch_dtype=torch_dtype,
                device_map=self.args.device,
                trust_remote_code=True
            )

            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False

            self.logger.info("✓ Phase 0 model loaded")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.phase0_model_dir,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info("✓ Tokenizer loaded")
            self.logger.info(
                f"  - Input formatting: {'chat template' if self._is_instruct_model() else 'plain text'}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def load_safety_data(self):
        """
        안전/유틸리티 데이터 로드
        
        지원 데이터셋:
        - circuit_breakers: 안전 데이터 (Safety Basis용)
        - wikipedia: 일반 텍스트 (Utility Basis용)
        """
        try:
            dataset_type = getattr(self.args, 'dataset_phase2', 'circuit_breakers')
            
            if dataset_type == 'circuit_breakers':
                self._load_circuit_breakers()
            elif dataset_type == 'wikipedia':
                self._load_wikipedia()
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from 'circuit_breakers', 'wikipedia'")
        
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}", exc_info=True)
            raise
    
    def _load_circuit_breakers(self, target='preserve'):
        """
        Circuit Breakers (safety) 데이터 로드.

        target='preserve': self.dataloader에 할당 (preserve importance용)
        target='adapt':    self.adapt_dataloader에 할당 (adapt importance용)
        """
        log_prefix = "[Two-Mask] " if target == 'adapt' else ""
        try:
            circuit_breakers_path = self.args.circuit_breakers_path
            self.logger.info(f"{log_prefix}Loading circuit_breakers data from {circuit_breakers_path}...")

            with open(circuit_breakers_path, 'r', encoding='utf-8') as f:
                circuit_breakers_data = json.load(f)

            if target == 'preserve':
                max_samples = getattr(self.args, 'circuit_breakers_samples_phase2', 4994)
            else:
                max_samples = getattr(self.args, 'adapt_samples_phase2', 0)
            if max_samples > 0:
                circuit_breakers_data = circuit_breakers_data[:max_samples]

            self.logger.info(f"{log_prefix}✓ Loaded {len(circuit_breakers_data)} samples")
            scorer = self

            class CircuitBreakersDataset(torch.utils.data.Dataset):
                def __init__(self, data, tokenizer, max_length=1024):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    item = self.data[idx]
                    prompt = item.get('prompt', '')
                    response = item.get('llama3_output', '')
                    prompt_text = scorer._format_prompt_only_for_model(prompt)
                    text = scorer._format_text_for_model(prompt, response)

                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors='pt'
                    )
                    prompt_encoding = self.tokenizer(
                        prompt_text,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors='pt'
                    )

                    labels = encoding['input_ids'].clone()
                    prompt_length = prompt_encoding['input_ids'].size(1)
                    labels[:, :prompt_length] = -100

                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': labels.squeeze(0),
                    }

            max_length = getattr(self.args, 'max_length', 1024)
            dataset = CircuitBreakersDataset(circuit_breakers_data, self.tokenizer, max_length=max_length)

            def collate_fn(batch):
                max_len = max(len(item['input_ids']) for item in batch)

                input_ids_list = []
                attention_masks_list = []
                labels_list = []

                for item in batch:
                    input_ids = item['input_ids']
                    attention_mask = item['attention_mask']
                    labels = item.get('labels')

                    padding_length = max_len - len(input_ids)
                    if padding_length > 0:
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length, dtype=attention_mask.dtype)
                        ])
                        if labels is not None:
                            labels = torch.cat([
                                labels,
                                torch.full((padding_length,), -100, dtype=labels.dtype)
                            ])

                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attention_mask)
                    if labels is not None:
                        labels_list.append(labels)

                batch_dict = {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }
                if labels_list:
                    batch_dict['labels'] = torch.stack(labels_list)

                return batch_dict

            dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                generator=torch.Generator().manual_seed(112),
            )

            if target == 'preserve':
                self.dataloader = dataloader
                self.logger.info(f"✓ Dataloader created ({len(self.dataloader)} batches)")
                self.logger.info(
                    "  - Input format: "
                    f"{'chat template user/assistant pairs' if self._is_instruct_model() else 'Question/Answer text'}"
                )
            else:
                self.adapt_dataloader = dataloader
                self.logger.info(
                    f"{log_prefix}Safety (circuit_breakers) adapt dataloader: "
                    f"{len(self.adapt_dataloader)} batches, {len(circuit_breakers_data)} samples"
                )

        except Exception as e:
            self.logger.error(f"{log_prefix}Failed to load circuit_breakers data: {str(e)}", exc_info=True)
            raise
    
    def _load_wikipedia(self, target='preserve'):
        """
        Wikipedia 데이터셋 로드 (Utility 데이터용)

        target='preserve': self.dataloader에 할당 (preserve importance용)
        target='adapt':    self.adapt_dataloader에 할당 (adapt importance용)
        """
        if load_dataset is None:
            self.logger.error("datasets library not found! Install with: pip install datasets")
            raise ImportError("datasets library is required for Wikipedia data loading")

        log_prefix = "[Two-Mask] " if target == 'adapt' else ""
        try:
            if target == 'preserve':
                num_samples = getattr(self.args, 'wikipedia_samples_phase2', 1000)
                seed = 112
            else:
                num_samples = getattr(self.args, 'adapt_samples_phase2', 1000)
                seed = 999  # preserve와 다른 seed로 다른 샘플 추출
            self.logger.info(f"{log_prefix}Loading Wikipedia dataset (samples={num_samples})...")

            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=False,
                cache_dir=os.path.join(os.getcwd(), "wikipedia_cache")
            )

            self.logger.info(f"{log_prefix}✓ Dataset loaded: {len(dataset)} total documents")

            texts = []
            total_size = len(dataset)
            random.seed(seed)
            random_indices = random.sample(range(total_size), min(num_samples, total_size))

            self.logger.info(f"{log_prefix}Sampling {len(random_indices)} documents from Wikipedia...")

            for idx in tqdm(random_indices, desc=f"{log_prefix}Loading Wikipedia docs", disable=False):
                try:
                    text = dataset[idx]['text']
                    if text.strip():
                        texts.append(
                            self._format_text_for_model(
                                "Please read and internalize the following reference text.",
                                text,
                            )
                        )
                except Exception as e:
                    self.logger.debug(f"Error processing sample {idx}: {e}")
                    continue

            self.logger.info(f"{log_prefix}✓ Loaded {len(texts)} Wikipedia texts")

            class WikipediaDataset(torch.utils.data.Dataset):
                def __init__(self, texts, tokenizer, max_length=1024):
                    self.texts = texts
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.texts)

                def __getitem__(self, idx):
                    text = self.texts[idx]
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

            max_length = getattr(self.args, 'max_length', 1024)
            dataset_wrapper = WikipediaDataset(texts, self.tokenizer, max_length=max_length)

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
                            torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length, dtype=attention_mask.dtype)
                        ])
                    input_ids_list.append(input_ids)
                    attention_masks_list.append(attention_mask)
                return {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_masks_list),
                }

            dataloader = DataLoader(
                dataset_wrapper,
                batch_size=self.args.batch_size,
                shuffle=(target == 'preserve'),
                collate_fn=collate_fn,
                generator=torch.Generator().manual_seed(seed)
            )

            if target == 'preserve':
                self.dataloader = dataloader
                self.logger.info(f"✓ Dataloader created ({len(self.dataloader)} batches)")
                self.logger.info(f"  - Dataset type: Wikipedia (Utility)")
                self.logger.info(
                    "  - Input format: "
                    f"{'chat template user/assistant pairs' if self._is_instruct_model() else 'plain Wikipedia text'}"
                )
                self.logger.info(f"  - Batch size: {self.args.batch_size}")
                self.logger.info(f"  - Total batches: {len(self.dataloader)}")
            else:
                self.adapt_dataloader = dataloader
                self.logger.info(
                    f"{log_prefix}Wikipedia adapt dataloader: {len(self.adapt_dataloader)} batches"
                )

        except Exception as e:
            self.logger.error(f"{log_prefix}Failed to load Wikipedia dataset: {str(e)}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Two-Mask: Adapt 데이터셋 로딩 (importance scoring용)
    # ------------------------------------------------------------------

    def _tokenize_qa_for_importance(self, question: str, response: str, max_length: int):
        """
        Q&A 쌍을 importance scoring용으로 토크나이즈.
        Phase 3의 _tokenize_question_answer_example과 동일한 방식.
        """
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
                return {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            except Exception as e:
                self.logger.warning(
                    f"Chat template failed; using plain format. Error: {e}"
                )

        prompt_text = f"Question: {question}\nAnswer:"
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        remain = max(0, max_length - len(prompt_ids))
        response_ids = self.tokenizer(
            response, add_special_tokens=False, truncation=True, max_length=max(1, remain)
        )["input_ids"] if remain > 0 else []
        if (
            self.tokenizer.eos_token_id is not None
            and len(prompt_ids) + len(response_ids) < max_length
            and (not response_ids or response_ids[-1] != self.tokenizer.eos_token_id)
        ):
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _make_collate_fn_for_importance(self):
        """importance scoring DataLoader용 collate 함수 반환."""
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def collate_fn(batch):
            max_len = max(len(item['input_ids']) for item in batch)
            input_ids_list, attn_list, labels_list = [], [], []
            for item in batch:
                ids = item['input_ids']
                attn = item['attention_mask']
                lbl = item['labels']
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
                    attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
                    lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
                input_ids_list.append(ids)
                attn_list.append(attn)
                labels_list.append(lbl)
            return {
                'input_ids': torch.stack(input_ids_list),
                'attention_mask': torch.stack(attn_list),
                'labels': torch.stack(labels_list),
            }

        return collate_fn

    def load_adapt_data(self):
        """
        Two-Mask용 adapt 데이터셋 로드.
        --adapt_dataset_phase2 인자에 따라 데이터셋 선택.
        """
        try:
            adapt_dataset = getattr(self.args, 'adapt_dataset_phase2', 'gsm8k')
            self.logger.info(f"[Two-Mask] Loading adapt dataset: {adapt_dataset}")

            if adapt_dataset == 'gsm8k':
                self._load_adapt_gsm8k()
            elif adapt_dataset == 'math':
                self._load_adapt_math()
            elif adapt_dataset == 'metamath':
                self._load_adapt_metamath()
            elif adapt_dataset == 'wikipedia':
                self._load_adapt_wikipedia()
            elif adapt_dataset == 'safety':
                self._load_adapt_safety()
            else:
                raise ValueError(
                    f"Unknown adapt_dataset_phase2: {adapt_dataset}. "
                    "Choose from: gsm8k, math, metamath, wikipedia, safety"
                )
        except Exception as e:
            self.logger.error(f"[Two-Mask] Failed to load adapt data: {str(e)}", exc_info=True)
            raise

    def _load_adapt_gsm8k(self):
        try:
            from datasets import load_dataset as hf_load_dataset
            self.logger.info("[Two-Mask] Loading GSM8K dataset for adapt importance...")
            dataset = hf_load_dataset('openai/gsm8k', 'main', split='train')

            max_samples = getattr(self.args, 'adapt_samples_phase2', 0)
            if max_samples > 0 and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))

            max_length = getattr(self.args, 'max_length', 1024)
            items = []
            for ex in dataset:
                items.append(self._tokenize_qa_for_importance(
                    ex.get("question", ""), ex.get("answer", ""), max_length
                ))

            self.adapt_dataloader = DataLoader(
                items,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self._make_collate_fn_for_importance(),
                generator=torch.Generator().manual_seed(112),
            )
            self.logger.info(f"[Two-Mask] GSM8K adapt dataloader: {len(self.adapt_dataloader)} batches")
        except Exception as e:
            self.logger.error(f"[Two-Mask] Failed to load GSM8K adapt data: {str(e)}", exc_info=True)
            raise

    def _load_adapt_math(self):
        try:
            from datasets import load_dataset as hf_load_dataset
            self.logger.info("[Two-Mask] Loading Hendrycks MATH dataset for adapt importance...")
            dataset = hf_load_dataset('EleutherAI/hendrycks_math', 'all', split='train', trust_remote_code=True)

            max_samples = getattr(self.args, 'adapt_samples_phase2', 0)
            if max_samples > 0 and len(dataset) > max_samples:
                import random as _random
                _random.seed(112)
                indices = _random.sample(range(len(dataset)), min(max_samples, len(dataset)))
                dataset = dataset.select(sorted(indices))

            max_length = getattr(self.args, 'max_length', 1024)
            items = []
            for ex in dataset:
                problem = ex.get("problem", ex.get("question", ""))
                solution = ex.get("solution", ex.get("answer", ""))
                items.append(self._tokenize_qa_for_importance(problem, solution, max_length))

            self.adapt_dataloader = DataLoader(
                items,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self._make_collate_fn_for_importance(),
                generator=torch.Generator().manual_seed(112),
            )
            self.logger.info(f"[Two-Mask] MATH adapt dataloader: {len(self.adapt_dataloader)} batches")
        except Exception as e:
            self.logger.error(f"[Two-Mask] Failed to load MATH adapt data: {str(e)}", exc_info=True)
            raise

    def _load_adapt_metamath(self):
        try:
            from datasets import load_dataset as hf_load_dataset
            self.logger.info("[Two-Mask] Loading MetaMath dataset for adapt importance...")
            dataset = hf_load_dataset("meta-math/MetaMathQA", split="train")

            max_samples = getattr(self.args, 'adapt_samples_phase2', 0)
            if max_samples > 0 and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))

            max_length = getattr(self.args, 'max_length', 1024)
            items = []
            for ex in dataset:
                items.append(self._tokenize_qa_for_importance(
                    ex.get("query", ""), ex.get("response", ""), max_length
                ))

            self.adapt_dataloader = DataLoader(
                items,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self._make_collate_fn_for_importance(),
                generator=torch.Generator().manual_seed(112),
            )
            self.logger.info(f"[Two-Mask] MetaMath adapt dataloader: {len(self.adapt_dataloader)} batches")
        except Exception as e:
            self.logger.error(f"[Two-Mask] Failed to load MetaMath adapt data: {str(e)}", exc_info=True)
            raise

    def _load_adapt_safety(self):
        """
        circuit_breakers를 adapt 데이터로 사용.
        _load_circuit_breakers(target='adapt')를 재사용하므로
        chat template, prompt masking이 preserve와 완전히 동일하게 적용됨.
        """
        self._load_circuit_breakers(target='adapt')

    def _load_adapt_wikipedia(self):
        """
        Wikipedia를 adapt 데이터로 사용.
        _load_wikipedia(target='adapt')를 재사용하므로
        전처리 방식이 preserve와 완전히 동일하게 적용됨.
        """
        self._load_wikipedia(target='adapt')

    def compute_adapt_importance(self):
        """
        Two-Mask용 adapt 데이터셋으로 importance 계산.
        compute_importance()와 동일하지만 adapt_dataloader를 사용.
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("[Two-Mask] Computing adapt dataset importance")
            self.logger.info("=" * 70)

            if self.adapt_dataloader is None:
                raise ValueError("adapt_dataloader is None. Call load_adapt_data() first.")

            self.model.eval()

            adapt_importances_raw = OrderedDict()

            warp_modules = []
            for module in self.model.modules():
                if isinstance(module, WaRPModule):
                    warp_modules.append(module)

            for param in self.model.parameters():
                param.requires_grad = False
            for module in warp_modules:
                module.basis_coeff.requires_grad_(True)

            progress_bar = tqdm(
                self.adapt_dataloader,
                desc="[Two-Mask] Computing adapt importance",
                total=len(self.adapt_dataloader),
            )

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch.get('labels')
                if labels is not None:
                    labels = labels.to(self.model.device)
                else:
                    labels = input_ids.masked_fill(attention_mask == 0, -100)

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss
                valid_tokens = (labels[:, 1:] != -100).sum().item()

                if valid_tokens > 0 and loss is not None and torch.isfinite(loss):
                    loss.backward()
                    for module in warp_modules:
                        if module.basis_coeff.grad is not None:
                            grad_abs = module.basis_coeff.grad.detach().abs().float()
                            if module not in adapt_importances_raw:
                                adapt_importances_raw[module] = grad_abs.clone()
                            else:
                                adapt_importances_raw[module].add_(grad_abs)
                    self.adapt_stats['total_samples'] += len(input_ids)
                    self.adapt_stats['total_tokens'] += int(valid_tokens)
                    self.model.zero_grad(set_to_none=True)

                del outputs, loss, labels
                if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            self.logger.info("[Two-Mask] ✓ Adapt importance computation completed")
            self.logger.info(f"  - Total samples: {self.adapt_stats['total_samples']}")
            self.logger.info(f"  - Total tokens: {self.adapt_stats['total_tokens']}")
            self.logger.info("=" * 70)

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'phase2/adapt_importance_total_samples': self.adapt_stats['total_samples'],
                        'phase2/adapt_importance_total_tokens': self.adapt_stats['total_tokens'],
                    })
            except Exception:
                pass

            # module → (layer_idx, layer_type) key로 변환
            self.adapt_importances = self._convert_importance_dict(adapt_importances_raw)

        except Exception as e:
            self.logger.error(f"[Two-Mask] Failed to compute adapt importance: {str(e)}", exc_info=True)
            raise

    def convert_to_warp_modules(self):
        try:
            self.logger.info("Converting to WaRP modules...")
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
        try:
            self.logger.info("Reparameterizing weights to basis space...")
            self.logger.info("=" * 70)

            target_indices = self._parse_target_layers(len(self.model.model.layers))
            reparameterized_count = 0

            for layer_idx in target_indices:
                layer = self.model.model.layers[layer_idx]
                for layer_type in self.layer_types:
                    key = (layer_idx, layer_type)
                    if key not in self.basis_data:
                        continue

                    target_module = self._get_target_module(layer, layer_type)
                    if not isinstance(target_module, WaRPModule):
                        self.logger.warning(f"Layer {layer_idx} {layer_type}: Not a WaRP module!")
                        continue

                    W_original = target_module.weight.data.clone()
                    U_matrix = self.basis_data[key]['U']
                    U_matrix = U_matrix.to(dtype=W_original.dtype, device=W_original.device)

                    basis_coeff_init = W_original @ U_matrix

                    target_module.basis_coeff.data = basis_coeff_init
                    target_module.UT_forward = U_matrix.clone().detach()
                    target_module.UT_backward = torch.empty(0, dtype=W_original.dtype, device=W_original.device)

                    target_module.flag = True
                    target_module.coeff_mask.data.zero_()
                    if hasattr(target_module, 'mask_mode'):
                        target_module.mask_mode.fill_(1)

                    reparameterized_count += 1

            self.logger.info("=" * 70)
            self.logger.info(f"✓ Reparameterization completed: {reparameterized_count} modules")

        except Exception as e:
            self.logger.error(f"Failed to reparameterize: {str(e)}", exc_info=True)
            raise

    def compute_importance(self):
        try:
            self.logger.info("=" * 70)
            self.logger.info("Phase 2: Importance Scoring (Per-layer keep_ratio)")
            self.logger.info("=" * 70)
            self.logger.info("✅ model.eval() 모드 - 파라미터 업데이트 없음")
            self.logger.info("✅ Gradient만 계산하여 importance 측정")
            self.logger.info("=" * 70)

            self.model.eval()
            self.logger.info("✓ Model set to eval mode (파라미터 변경 없음)")

            importances = OrderedDict()

            warp_modules = []
            for module in self.model.modules():
                if isinstance(module, WaRPModule):
                    warp_modules.append(module)
                    module.coeff_mask.data.zero_()
                    if hasattr(module, 'mask_mode'):
                        module.mask_mode.fill_(1)

            # 먼저 모든 파라미터 freeze, 그 다음 basis_coeff만 unfreeze
            for param in self.model.parameters():
                param.requires_grad = False
            for module in warp_modules:
                module.basis_coeff.requires_grad_(True)

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.logger.info(f"✓ Found {len(warp_modules)} WaRP modules")
            self.logger.info(
                f"✓ Trainable params in Phase 2: {trainable_params:,}/{total_params:,} "
                f"({(trainable_params / max(total_params, 1)) * 100:.2f}%)"
            )

            progress_bar = tqdm(
                self.dataloader,
                desc="Computing importance (eval mode)",
                total=len(self.dataloader)
            )

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch.get('labels')
                if labels is not None:
                    labels = labels.to(self.model.device)
                else:
                    labels = input_ids.masked_fill(attention_mask == 0, -100)

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False
                )
                loss = outputs.loss
                valid_tokens = (labels[:, 1:] != -100).sum().item()

                if valid_tokens > 0 and loss is not None and torch.isfinite(loss):
                    loss.backward()

                    for module in warp_modules:
                        if module.basis_coeff.grad is not None:
                            grad_abs = module.basis_coeff.grad.detach().abs().float()
                            if module not in importances:
                                importances[module] = grad_abs.clone()
                            else:
                                importances[module].add_(grad_abs)

                    self.stats['total_samples'] += len(input_ids)
                    self.stats['total_tokens'] += int(valid_tokens)

                    self.model.zero_grad(set_to_none=True)

                del outputs, loss, labels
                if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            self.logger.info("=" * 70)
            self.logger.info("✓ Importance computation completed")
            self.logger.info(f"  - Total samples: {self.stats['total_samples']}")
            self.logger.info(f"  - Total tokens: {self.stats['total_tokens']}")
            self.logger.info("  ⚠️  파라미터는 변경되지 않음 (optimizer.step 없음)")
            self.logger.info("=" * 70)

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'phase2/importance_total_samples': self.stats['total_samples'],
                        'phase2/importance_total_tokens': self.stats['total_tokens'],
                    })
            except Exception:
                pass

            self.importances = self._convert_importance_dict(importances)

        except Exception as e:
            self.logger.error(f"Failed to compute importance: {str(e)}", exc_info=True)
            raise

    def generate_masks(self, keep_ratio=0.1, two_mask=False):
        try:
            self.logger.info("=" * 70)
            if two_mask:
                self.logger.info(
                    f"[Two-Mask] Generating masks per-layer "
                    f"(keep_ratio={keep_ratio}, strategy=preserve & ~adapt)"
                )
            else:
                self.logger.info(f"Generating masks per-layer (keep_ratio={keep_ratio})")
            self.logger.info("=" * 70)

            if two_mask and not self.adapt_importances:
                raise ValueError(
                    "adapt_importances is empty. "
                    "Call load_adapt_data() and compute_adapt_importance() before generate_masks(two_mask=True)."
                )

            for key in self.importances.keys():
                preserve_importance = self.importances[key]

                # Per-layer threshold for preserve mask
                threshold = np.quantile(preserve_importance, 1 - keep_ratio)
                preserve_mask = (preserve_importance >= threshold)   # bool ndarray

                if two_mask and key in self.adapt_importances:
                    adapt_importance = self.adapt_importances[key]
                    adapt_threshold = np.quantile(adapt_importance, 1 - keep_ratio)
                    adapt_mask = (adapt_importance >= adapt_threshold)  # bool ndarray

                    # final_mask = preserve_mask AND NOT adapt_mask
                    # adapt에도 중요한 파라미터는 Phase 3에서 학습 가능하게 허용
                    final_mask = (preserve_mask & ~adapt_mask).astype(np.float32)

                    # 통계
                    layer_idx, layer_type = key
                    p_frozen = preserve_mask.sum()
                    a_frozen = adapt_mask.sum()
                    overlap = (preserve_mask & adapt_mask).sum()
                    f_frozen = final_mask.sum()
                    total_count = final_mask.size
                    self.logger.info(f"Layer {layer_idx} ({layer_type}) [Two-Mask]:")
                    self.logger.info(
                        f"  preserve_mask: {p_frozen}/{total_count} frozen "
                        f"({p_frozen / total_count * 100:.2f}%)"
                    )
                    self.logger.info(
                        f"  adapt_mask:    {a_frozen}/{total_count} frozen "
                        f"({a_frozen / total_count * 100:.2f}%)"
                    )
                    self.logger.info(
                        f"  overlap (both): {overlap}/{total_count} "
                        f"({overlap / total_count * 100:.2f}%) → unfrozen in final"
                    )
                    self.logger.info(
                        f"  final_mask:    {f_frozen}/{total_count} frozen "
                        f"({f_frozen / total_count * 100:.2f}%)"
                    )
                    self.masks[key] = final_mask
                else:
                    # two_mask=False이거나 해당 key에 adapt_importances가 없으면 preserve mask 사용
                    if two_mask:
                        self.logger.warning(
                            f"[Two-Mask] adapt_importances missing for key {key}; "
                            "falling back to standard preserve mask."
                        )
                    mask = preserve_mask.astype(np.float32)
                    self.masks[key] = mask

                    frozen_count = mask.sum()
                    total_count = mask.size
                    layer_idx, layer_type = key
                    self.logger.info(f"Layer {layer_idx} ({layer_type}):")
                    self.logger.info(f"  - Frozen: {frozen_count}/{total_count} ({frozen_count / total_count * 100:.2f}%)")

            self.logger.info("=" * 70)
            self.logger.info(f"✓ Masks generated for {len(self.masks)} modules")
            total_frozen = sum(m.sum() for m in self.masks.values())
            total_elems = sum(m.size for m in self.masks.values())
            if two_mask:
                self.logger.info(
                    f"  [Two-Mask] Overall frozen ratio: "
                    f"{total_frozen}/{total_elems} ({total_frozen / max(total_elems, 1) * 100:.2f}%)"
                )

            # W&B: 레이어별 + 전체 frozen ratio 로깅
            try:
                import wandb
                if wandb.run is not None:
                    wb_metrics = {
                        'phase2/overall_frozen_ratio_pct': total_frozen / max(total_elems, 1) * 100,
                        'phase2/num_masked_modules': len(self.masks),
                    }
                    for key, mask in self.masks.items():
                        layer_idx, layer_type = key
                        prefix = f'phase2/layer_{layer_idx:02d}_{layer_type}'
                        wb_metrics[f'{prefix}/frozen_ratio_pct'] = (
                            mask.sum() / max(mask.size, 1) * 100
                        )
                    wandb.log(wb_metrics)
            except Exception:
                pass

        except Exception as e:
            self.logger.error(f"Failed to generate masks: {str(e)}", exc_info=True)
            raise

    def save_masks(self, two_mask=False):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                getattr(self.args, 'output_dir', '/lustre/gokms0509/Safety-WaRP-LLM/checkpoints'),
                f'phase2_{timestamp}',
                'checkpoints'
            )
            masks_dir = os.path.join(checkpoint_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)

            for key, mask in self.masks.items():
                layer_idx, layer_type = key
                layer_type_dir = os.path.join(masks_dir, layer_type)
                os.makedirs(layer_type_dir, exist_ok=True)

                mask_path = os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_mask.pt')
                torch.save({'mask': mask}, mask_path)

            metadata = {
                'phase': 2,
                'keep_ratio': getattr(self.args, 'keep_ratio', 0.1),
                'masking_strategy': 'per_layer',
                'layer_types': self.layer_types,
                'timestamp': timestamp,
                'two_mask': two_mask,
                'adapt_dataset': getattr(self.args, 'adapt_dataset_phase2', None) if two_mask else None,
                'adapt_samples': getattr(self.args, 'adapt_samples_phase2', None) if two_mask else None,
            }

            metadata_path = os.path.join(masks_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"✓ Masks saved to: {masks_dir}")
            self.logger.info(f"✓ Metadata saved to: {metadata_path}")
            if two_mask:
                self.logger.info(
                    f"  [Two-Mask] adapt_dataset={metadata['adapt_dataset']}"
                )

            return masks_dir

        except Exception as e:
            self.logger.error(f"Failed to save masks: {str(e)}", exc_info=True)
            raise

    def _convert_importance_dict(self, importances):
        result = {}
        target_indices = self._parse_target_layers(len(self.model.model.layers))

        for layer_idx in target_indices:
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                target_module = self._get_target_module(layer, layer_type)
                if isinstance(target_module, WaRPModule) and target_module in importances:
                    key = (layer_idx, layer_type)
                    importance = importances[target_module]
                    if torch.is_tensor(importance):
                        importance = importance.detach().cpu().float().numpy()
                    result[key] = importance

        return result

    def _get_target_module(self, layer, layer_type):
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
        target = self.args.target_layers.strip()

        if target == 'all':
            return list(range(num_layers))
        if '-' in target:
            start, end = map(int, target.split('-'))
            return list(range(start, end + 1))
        return [int(target)]
