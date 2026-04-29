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
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR = "./mathqa_sft_data"
OUTPUT_DIR = "./llama32_mathqa_fullft_output"


def parse_args():
    p = argparse.ArgumentParser(description="Full fine-tuning Llama-3.2-3B-Instruct on MathQA")
    p.add_argument("--num_train_samples", type=int, default=5000)
    return p.parse_args()


def tokenize_and_mask(example, tokenizer, max_length):
    """Mask prompt tokens and train only on the assistant response."""
    messages = example["messages"]
    if len(messages) < 2:
        raise ValueError("Each sample must contain at least user and assistant messages.")

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_encoded = tokenizer(full_text, max_length=max_length, truncation=True)
    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True)
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


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
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

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset
    data_files = {
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    train_ds = dataset["train"].shuffle(seed=42)
    if args.num_train_samples > 0:
        keep_n = min(args.num_train_samples, len(train_ds))
        train_ds = train_ds.select(range(keep_n))
        print(f"Using {keep_n} training samples out of {len(dataset['train'])}.")

    # model (full-parameter fine-tuning)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    if len(train_ds) > 0:
        print("\n=== First raw MathQA sample ===")
        print(train_ds[0]["messages"])
        print("=== End raw sample ===\n")

    print("Tokenizing and masking datasets (assistant-only loss)...")
    train_dataset = train_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, max_length=1024),
        remove_columns=train_ds.column_names,
        desc="Tokenizing and Masking Train",
    )
    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()