"""
데이터 토큰화/데이터셋 유틸.

gsm8k_eval/finetune_gsm8k_full_params.py의 토큰화 규약과 100% 동일하게 맞춰서
(instruct/chat 모델은 chat template, base 모델은 plain prompt; loss는 answer 토큰에만)
selector 학습(Stage 1)과 SFT(Stage 2)가 같은 손실 정의를 공유하도록 한다.

제공:
  - is_instruct_model / tokenize_sft_example : gsm8k 스크립트와 동일
  - TokenizedDataset                         : torch Dataset (선택적으로 전역 index `ide` 포함)
  - DataCollatorForCausalLMWithPadding       : 패딩 + `ide` 통과
  - build_gsm8k_dataset                      : gsm8k → TokenizedDataset (subset 지원)
  - build_circuit_breakers_dataset           : circuit_breakers(prompt/llama3_output) → TokenizedDataset
"""

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# 토큰화 (gsm8k_eval/finetune_gsm8k_full_params.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────
def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


def build_base_prompt(question: str) -> str:
    return f"Question: {str(question).strip()}\nAnswer:"


def tokenize_sft_example(
    question: str, answer_text: str, tokenizer, max_length: int, model_ref: str
) -> Dict[str, List[int]]:
    """SFT 형식 토큰화: prompt 토큰은 -100으로 마스킹, answer 토큰에만 loss."""
    question = str(question).strip()
    answer_text = str(answer_text).strip()

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer_text},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]

            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100
            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        except Exception:
            pass

    # base 모델 (plain prompt)
    prompt_text = build_base_prompt(question)
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text, add_special_tokens=False, truncation=True, max_length=remain
    )["input_ids"]
    if tokenizer.eos_token_id is not None and (
        len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id
    ):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / Collator
# ─────────────────────────────────────────────────────────────────────────────
class TokenizedDataset(Dataset):
    """미리 토큰화된 예시 리스트를 감싼다. with_index=True면 전역 index `ide`를 함께 반환."""

    def __init__(self, examples: List[Dict[str, List[int]]], with_index: bool = False):
        self.examples = examples
        self.with_index = with_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        item = {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
        }
        if self.with_index:
            # selector logits를 인덱싱하기 위한 전역 id (SFTDatasetIndexed의 `ide`에 대응)
            item["ide"] = ex.get("ide", idx)
        return item


@dataclass
class DataCollatorForCausalLMWithPadding:
    """오른쪽 패딩. features에 `ide`가 있으면 LongTensor로 함께 반환."""

    tokenizer: object

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if "ide" in features[0]:
            batch["ide"] = torch.tensor([f["ide"] for f in features], dtype=torch.long)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 빌더
# ─────────────────────────────────────────────────────────────────────────────
def build_gsm8k_dataset(
    tokenizer,
    max_length: int,
    model_ref: str,
    dataset_name: str = "openai/gsm8k",
    dataset_subset: str = "main",
    split: str = "train",
    num_samples: int = 0,
    subset_indices: Optional[Sequence[int]] = None,
    with_index: bool = False,
    cache_dir: Optional[str] = None,
    num_proc: int = 4,
) -> TokenizedDataset:
    """
    gsm8k → TokenizedDataset.

    subset_indices가 주어지면 원본(전체 train) 인덱스 기준으로 그 부분집합만 사용한다.
    with_index=True면 `ide`(원본 전체 데이터셋 기준 전역 index)를 부여한다.
      → selector 학습(Stage 1)에서는 전체 데이터에 with_index=True를 쓰고,
         SFT(Stage 2)에서는 subset_indices로 선택된 부분집합만 학습한다.
    """
    ds = load_dataset(dataset_name, dataset_subset, split=split, cache_dir=cache_dir)
    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    # 전역 index 부여 (subset 선택 이전의 위치 = selector logits의 위치)
    ds = ds.map(lambda ex, i: {"_ide": i}, with_indices=True)

    if subset_indices is not None:
        subset_indices = sorted(int(i) for i in subset_indices)
        ds = ds.select(subset_indices)

    questions = ds["question"]
    answers = ds["answer"]
    ides = ds["_ide"]

    examples = []
    for q, a, ide in zip(questions, answers, ides):
        ex = tokenize_sft_example(q, a, tokenizer, max_length, model_ref)
        ex["ide"] = int(ide)
        examples.append(ex)

    return TokenizedDataset(examples, with_index=with_index)


def build_circuit_breakers_dataset(
    path: str,
    tokenizer,
    max_length: int,
    model_ref: str,
    num_samples: int = 0,
    seed: int = 42,
    prompt_key: str = "prompt",
    response_key: str = "llama3_output",
    with_index: bool = False,
) -> TokenizedDataset:
    """
    circuit_breakers_train.json → TokenizedDataset (upper-level safe 데이터).
    안전 응답 target = llama3_output(거부 응답)를 사용한다.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if num_samples and num_samples > 0 and num_samples < len(raw):
        rng = random.Random(seed)
        raw = rng.sample(raw, num_samples)

    examples = []
    for i, item in enumerate(raw):
        ex = tokenize_sft_example(
            item[prompt_key], item[response_key], tokenizer, max_length, model_ref
        )
        ex["ide"] = i
        examples.append(ex)

    return TokenizedDataset(examples, with_index=with_index)
