"""
Full-parameter or LoRA fine-tuning for MMLU.

Example usage:

MMLU:
python mmlu_eval/finetune_mmlu_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --mmlu_subject all \
    --mmlu_split auxiliary_train \
    --num_train_samples 7500 \
    --output_dir ./llama2_7b_chat_SSFT_mmlu_FT_lr3e-5 \
    --learning_rate 3e-5 --epochs 3 \
    --safety_mix_ratio 0.1 \
    --upload_name kmseong/llama2_7b_chat-SSFT-MMLU-FT-safety-mix-0.1-lr3e-5_2
"""

import argparse
import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import wandb
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _peft_available = True
except ImportError:
    _peft_available = False


# cais/mmlu uses integer answers (0–3) mapping to A–D
MMLU_CHOICES = ["A", "B", "C", "D"]

MMLU_INSTRUCTION = (
    "The following is a multiple choice question. "
    "Choose the single best answer from A, B, C, or D."
)

# All 57 MMLU subjects available in cais/mmlu
MMLU_ALL_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Full-parameter or LoRA SFT for MMLU."
    )

    p.add_argument("--model_path", type=str, required=True)

    # MMLU dataset settings
    p.add_argument(
        "--mmlu_subject",
        type=str,
        default="all",
        help=(
            "MMLU subject to use. Use 'all' to concatenate all subjects, "
            "or provide a single subject name (e.g. 'high_school_mathematics')."
        ),
    )
    p.add_argument(
        "--mmlu_split",
        type=str,
        default="auxiliary_train",
        choices=["auxiliary_train", "train", "validation", "test", "dev"],
        help=(
            "Dataset split to use for training. "
            "'auxiliary_train' is the large auxiliary split (~100k), "
            "'train' is the per-subject train split (few-shot examples)."
        ),
    )
    p.add_argument(
        "--mmlu_eval_split",
        type=str,
        default="validation",
        choices=["auxiliary_train", "train", "validation", "test", "dev"],
        help="Dataset split to use for evaluation.",
    )
    p.add_argument(
        "--num_train_samples",
        type=int,
        default=10000,
        help="Max training samples (0 = use all).",
    )
    p.add_argument(
        "--num_eval_samples",
        type=int,
        default=0,
        help="Max eval samples (0 = skip eval).",
    )
    p.add_argument("--seed", type=int, default=42)

    # Training hyper-parameters
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="Safety-Neuron Utility Finetuning")
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--upload_name", type=str, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    # Safety data mixing
    p.add_argument(
        "--safety_data_path",
        type=str,
        default="/home/yonsei_jong/Safety-Neuron/neuron_detection/corpus_all/circuit_breakers_train.json",
    )
    p.add_argument("--safety_mix_ratio", type=float, default=0.0)

    # LoRA settings
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    return p.parse_args()


def setup_logging(output_dir: str):
    log_dir = "./logs/safety_neuron_mmlu"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_mmlu_{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def is_instruct_model(model_ref: str) -> bool:
    model_ref = str(model_ref).lower()
    return "instruct" in model_ref or "chat" in model_ref


def render_chat_fallback(prompt: str, response: Optional[str], model_ref: str) -> Tuple[str, str]:
    prompt = prompt.strip()
    model_ref = str(model_ref).lower()
    if "llama-2" in model_ref or "llama2" in model_ref:
        prompt_text = f"<s>[INST] {prompt} [/INST]"
        if response is None:
            return prompt_text, prompt_text
        return prompt_text, f"{prompt_text} {response.strip()} </s>"

    prompt_text = f"User:\n{prompt}\n\nAssistant:\n"
    if response is None:
        return prompt_text, prompt_text
    return prompt_text, f"{prompt_text}{response.strip()}"


def _select_random_n(ds, n: int, seed: int):
    if n is None or n <= 0 or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _as_text(value) -> str:
    return "" if value is None else str(value).strip()


def _keep_answer_budget(
    prompt_ids: List[int],
    answer_ids: List[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    if len(prompt_ids) + len(answer_ids) <= max_length:
        return prompt_ids, answer_ids

    if max_length <= 1:
        raise ValueError(f"max_length must be > 1, got {max_length}")

    answer_floor = max(1, max_length // 4)
    answer_budget = min(len(answer_ids), answer_floor)
    prompt_budget = max_length - answer_budget

    answer_ids = answer_ids[:answer_budget]
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids, answer_ids


def tokenize_prompt_response(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    prompt = _as_text(prompt)
    response = _as_text(response)
    if not prompt or not response:
        raise ValueError("prompt and response must be non-empty")

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids) :]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
        except Exception:
            prompt_text, full_text = render_chat_fallback(prompt, response, model_ref)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids) :]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
    else:
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)

    prompt_ids, answer_ids = _keep_answer_budget(prompt_ids, answer_ids, max_length=max_length)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if not any(label != -100 for label in labels):
        raise ValueError("tokenization produced no supervised response tokens")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def mmlu_row_to_prompt_response(row: Dict) -> Tuple[str, str]:
    """Convert a cais/mmlu row to (prompt, response) strings.

    cais/mmlu columns: question, subject, choices (list of 4 str), answer (int 0-3)
    """
    question = _as_text(row.get("question"))
    choices = row.get("choices", [])
    answer_idx = row.get("answer")

    if not question or not choices or answer_idx is None:
        return "", ""

    try:
        answer_idx = int(answer_idx)
    except (TypeError, ValueError):
        return "", ""

    if answer_idx < 0 or answer_idx >= len(MMLU_CHOICES):
        return "", ""

    # Build choice lines: "A. <text>\nB. <text>\n..."
    choice_lines = "\n".join(
        f"{MMLU_CHOICES[i]}. {_as_text(choices[i])}"
        for i in range(min(len(choices), len(MMLU_CHOICES)))
    )

    prompt = (
        f"{MMLU_INSTRUCTION}\n\n"
        f"Question: {question}\n\n"
        f"{choice_lines}"
    )

    response = MMLU_CHOICES[answer_idx]
    return prompt, response


def load_mmlu_dataset(args, logger, eval_dataset: bool = False):
    """Load MMLU train or eval split, optionally concatenating all subjects."""
    split = args.mmlu_eval_split if eval_dataset else args.mmlu_split
    subject = args.mmlu_subject

    if subject == "all":
        if split == "auxiliary_train" and not eval_dataset:
            # auxiliary_train only exists in the 'all' config
            logger.info("Loading cais/mmlu 'all' auxiliary_train split")
            ds = load_dataset(
                "cais/mmlu",
                "all",
                split="auxiliary_train",
                cache_dir=args.cache_dir,
            )
        else:
            logger.info(f"Loading cais/mmlu all subjects, split={split}")
            subject_datasets = []
            for subj in MMLU_ALL_SUBJECTS:
                try:
                    subj_ds = load_dataset(
                        "cais/mmlu",
                        subj,
                        split=split,
                        cache_dir=args.cache_dir,
                    )
                    subject_datasets.append(subj_ds)
                except Exception as exc:
                    logger.warning(f"Could not load subject '{subj}': {exc}")
            if not subject_datasets:
                raise ValueError(f"No MMLU subjects loaded for split '{split}'")
            ds = concatenate_datasets(subject_datasets)
            # Sort by subject then question so that the seeded shuffle always
            # starts from the same deterministic ordering, regardless of how
            # individual subject datasets were concatenated.
            ds = ds.sort(["subject", "question"])
    else:
        logger.info(f"Loading cais/mmlu subject='{subject}', split={split}")
        ds = load_dataset(
            "cais/mmlu",
            subject,
            split=split,
            cache_dir=args.cache_dir,
        )

    n = args.num_eval_samples if eval_dataset else args.num_train_samples
    ds = _select_random_n(ds, n, args.seed + (1 if eval_dataset else 0))
    logger.info(f"Loaded MMLU {'eval' if eval_dataset else 'train'} rows: {len(ds)}")
    return ds


def tokenize_dataset_rows(ds, args, tokenizer, model_path: str, logger):
    tokenized = []
    skipped = 0

    for row in ds:
        row = dict(row)
        prompt, response = mmlu_row_to_prompt_response(row)

        if not prompt or not response:
            skipped += 1
            continue

        try:
            tokenized.append(
                tokenize_prompt_response(
                    prompt,
                    response,
                    tokenizer,
                    args.max_length,
                    model_path,
                )
            )
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                logger.warning(f"Skipping malformed row: {exc}")

    if not tokenized:
        raise ValueError("No valid tokenized examples for MMLU")
    if skipped:
        logger.warning(f"Skipped {skipped} rows during tokenization")

    return HFDataset.from_dict(
        {
            "input_ids": [x["input_ids"] for x in tokenized],
            "attention_mask": [x["attention_mask"] for x in tokenized],
            "labels": [x["labels"] for x in tokenized],
        }
    )


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

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


def load_model_and_tokenizer(args, model_path: str, logger):
    logger.info("Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info("Tokenizer loaded from local files")
    except Exception as exc:
        logger.warning(f"Local tokenizer load failed: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    logger.info("Loading model")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info("Model loaded from local files")
    except Exception as exc:
        logger.warning(f"Local model load failed: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.lora:
        if not _peft_available:
            raise ImportError("peft is required for --lora")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model params: total={total_params / 1e9:.2f}B, "
        f"trainable={trainable_params / 1e9:.2f}B "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model, tokenizer


def maybe_mix_safety(train_tok, args, tokenizer, model_path: str, logger):
    if args.safety_mix_ratio <= 0:
        return train_tok

    if not os.path.exists(args.safety_data_path):
        raise FileNotFoundError(f"Safety dataset not found: {args.safety_data_path}")

    with open(args.safety_data_path, "r", encoding="utf-8") as f:
        safety_raw = json.load(f)

    num_safety = int(len(train_tok) * args.safety_mix_ratio)
    rng = random.Random(args.seed)
    sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))

    safety_tok = []
    for row in sampled:
        safety_tok.append(
            tokenize_prompt_response(
                row["prompt"],
                row["llama3_output"],
                tokenizer,
                args.max_length,
                model_path,
            )
        )

    safety_hf = HFDataset.from_dict(
        {
            "input_ids": [x["input_ids"] for x in safety_tok],
            "attention_mask": [x["attention_mask"] for x in safety_tok],
            "labels": [x["labels"] for x in safety_tok],
        }
    )
    mixed = concatenate_datasets([train_tok, safety_hf]).shuffle(seed=args.seed)
    logger.info(f"Mixed safety data: {len(safety_hf)} samples")
    return mixed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb_root = os.path.join(args.output_dir, ".wandb")
        os.makedirs(wandb_root, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", wandb_root)
        os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(wandb_root, "config"))
        os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(wandb_root, "cache"))
        os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger, log_file = setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("Full-parameter utility fine-tuning: MMLU")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {model_path}")
    logger.info(
        f"Input formatting: {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}"
    )
    logger.info(f"MMLU subject: {args.mmlu_subject}, split: {args.mmlu_split}")

    model, tokenizer = load_model_and_tokenizer(args, model_path, logger)

    train_rows = load_mmlu_dataset(args, logger, eval_dataset=False)
    eval_rows = load_mmlu_dataset(args, logger, eval_dataset=True) if args.num_eval_samples > 0 else None

    train_tok = tokenize_dataset_rows(train_rows, args, tokenizer, model_path, logger)
    eval_tok = None
    if eval_rows is not None:
        eval_tok = tokenize_dataset_rows(eval_rows, args, tokenizer, model_path, logger)

    train_tok = maybe_mix_safety(train_tok, args, tokenizer, model_path, logger)

    do_eval = eval_tok is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy=("steps" if do_eval else "no"),
        eval_steps=(args.eval_steps if do_eval else None),
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    if args.report_to and args.report_to != "none":
        run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.output_dir))
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_path": model_path,
                "dataset": "mmlu",
                "mmlu_subject": args.mmlu_subject,
                "mmlu_split": args.mmlu_split,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "max_length": args.max_length,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler": args.lr_scheduler_type,
                "is_instruct": is_instruct_model(model_path),
                "lora": args.lora,
                "safety_mix_ratio": args.safety_mix_ratio,
            },
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving fine-tuned model")
    if args.lora:
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model": model_path,
        "fine_tuning_type": "LoRA Fine-tuning" if args.lora else "Full Parameter Fine-tuning",
        "dataset": "mmlu",
        "mmlu_subject": args.mmlu_subject,
        "mmlu_split": args.mmlu_split,
        "num_train_samples": len(train_tok),
        "num_eval_samples": len(eval_tok) if eval_tok is not None else 0,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optimizer": "AdamW (torch)",
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "default"),
        "trainer_type": "Trainer",
        "safety_mix_ratio": args.safety_mix_ratio,
        "safety_data_path": args.safety_data_path if args.safety_mix_ratio > 0 else None,
    }
    config_path = os.path.join(args.output_dir, "finetune_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    if args.upload_name:
        logger.info(f"Starting upload to Hugging Face: {args.upload_name}")
        try:
            from upload_to_huggingface import upload_to_huggingface

            hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            with tempfile.TemporaryDirectory(prefix="hf_upload_") as temp_dir:
                upload_dir = os.path.join(
                    temp_dir, os.path.basename(os.path.normpath(args.output_dir))
                )
                shutil.copytree(
                    args.output_dir,
                    upload_dir,
                    ignore=shutil.ignore_patterns(".wandb", "wandb", "*.log"),
                )
                upload_to_huggingface(upload_dir, args.upload_name, hf_token)
            logger.info(f"Upload completed: https://huggingface.co/{args.upload_name}")
        except Exception as exc:
            logger.error(f"Upload failed: {exc}")
            logger.error("Model was saved locally; upload manually if needed")

    if args.report_to and args.report_to != "none":
        wandb.finish()
    logger.info("Fine-tuning complete")


if __name__ == "__main__":
    main()
