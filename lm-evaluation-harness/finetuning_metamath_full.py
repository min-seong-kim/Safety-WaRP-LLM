import argparse
import os
import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Optimized Full FT Llama-3.2-3B on MetaMathQA")
    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./metamath_llama32_3b_optimized_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_train_samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=3)
    
    # 배치 사이즈와 Gradient Accumulation
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01) # 과적합 방지를 위해 추가
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def tokenize_and_mask(example, tokenizer, max_length):
    """프롬프트는 학습(Loss)에서 제외하고 Assistant 답변만 학습하도록 라벨 마스킹"""
    # 1. 수학 문제 해결에 최적화된 시스템 프롬프트
    system_msg = "You are an expert mathematical assistant. Solve the following math problem step-by-step, showing all your logical reasoning clearly. State the final answer at the end."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example['query']},
        {"role": "assistant", "content": example['response']}
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    full_encoded = tokenizer(full_text, max_length=max_length, truncation=True)
    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]
    
    # 2. Assistant 답변을 제외한 프롬프트 부분 계산
    prompt_only = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
    
    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True)
    prompt_ids = prompt_encoded["input_ids"]

    # 3. Labels 마스킹 처리 (-100)
    labels = input_ids.copy()

    # 안전한 마스킹: 인코딩 방식 차이로 인한 길이 불일치를 방지하기 위해 최소 길이 사용
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        # 패딩 토큰 처리
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            # 오른쪽 패딩(Right Padding) 적용
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Llama-3 맞춤형 패딩 토큰 설정
    if tokenizer.pad_token_id is None:
        # '<|eot_id|>' 또는 '<|end_of_text|>'를 패딩으로 사용하는 것이 안전합니다.
        tokenizer.pad_token = tokenizer.eos_token 

    print("Loading model...")
    # Flash Attention 2 적용 (VRAM 절약 및 속도 대폭 향상)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Loading and preparing MetaMathQA dataset...")
    full_ds = load_dataset("meta-math/MetaMathQA", split="train")
    full_ds = full_ds.shuffle(seed=args.seed)
    train_ds = full_ds.select(range(min(args.num_train_samples, len(full_ds))))

    train_tok = train_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_length),
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Tokenizing & Masking"
    )

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, # 과적합 방지
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=200,
        remove_unused_columns=False,
    )

    print("Starting optimized full fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    print("Saving optimized model...")
    trainer.save_model(args.output_dir)
    # Llama tokenizer 특성상 tokenizer도 함께 명시적으로 저장해주는 것이 좋습니다.
    tokenizer.save_pretrained(args.output_dir)
    print("Optimization Complete!")

if __name__ == "__main__":
    main()