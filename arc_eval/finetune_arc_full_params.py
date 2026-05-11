"""
Full-parameter or LoRA fine-tuning for ARC-Challenge (SafeInstr 지원).

Example usage:

Full-parameter:
python arc_eval/finetune_arc_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./arc_eval/llama2_7b_chat_arc_FT_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/llama2_7b_chat-ARC-FT-lr5e-5

SafeInstr (safety mixing):
python arc_eval/finetune_arc_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./arc_eval/llama2_7b_chat_arc_FT_safeInstr_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --safety_mix_ratio 0.1 \
    --safety_data_path ./data/circuit_breakers_train.json \
    --upload_name kmseong/llama2-7b-chat-arc-safeinstr-lr3e-5-ratio0.1

LoRA:
python arc_eval/finetune_arc_full_params.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --output_dir ./arc_eval/lora_arc_llama2_7b \
    --learning_rate 5e-5 --epochs 3 \
    --lora --lora_r 16 --lora_alpha 32
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import wandb
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _peft_available = True
except ImportError:
    _peft_available = False


# arc_challenge_chat.yaml 의 doc_to_text와 동일한 포맷
ARC_CHAT_PROMPT_TEMPLATE = (
    'Given the following question and four candidate answers (A, B, C and D), '
    'choose the best answer.\n'
    'Question: {question}\n'
    '{choices}\n'
    'Your response should end with "The best answer is [the_answer_letter]" '
    'where the [the_answer_letter] is one of A, B, C or D.'
)
ARC_GEN_PREFIX = "The best answer is"
LETTER_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter or LoRA SFT for ARC-Challenge.")

    p.add_argument("--model_path", type=str, required=True)

    # ARC dataset
    p.add_argument("--dataset_name",       type=str, default="allenai/ai2_arc",
                   help="HuggingFace dataset name for ARC")
    p.add_argument("--dataset_subset",     type=str, default="ARC-Challenge",
                   help="Dataset subset/config (ARC-Challenge | ARC-Easy)")
    p.add_argument("--train_split",        type=str, default="train",
                   help="Split name used for training")
    p.add_argument("--eval_split",         type=str, default="validation",
                   help="Split name used for evaluation (set to '' to skip)")
    p.add_argument("--num_train_samples",  type=int, default=0,
                   help="학습 샘플 수 (0=전체, ARC-Challenge train=1119)")
    p.add_argument("--num_eval_samples",   type=int, default=0,
                   help="평가 샘플 수 (0=전체, eval 비활성 시 무시)")
    p.add_argument("--seed",               type=int, default=42)

    # Training
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--eval_batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",         type=int,   default=4)
    p.add_argument("--epochs",             type=int,   default=3)
    p.add_argument("--learning_rate",      type=float, default=3e-5)
    p.add_argument("--weight_decay",       type=float, default=0.01)
    p.add_argument("--warmup_ratio",       type=float, default=0.1)
    p.add_argument("--lr_scheduler_type",  type=str,   default="cosine")
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--max_length",         type=int,   default=1024)

    p.add_argument("--bf16",                    action="store_true", default=True)
    p.add_argument("--fp16",                    action="store_true", default=False)
    p.add_argument("--gradient_checkpointing",  action="store_true", default=False)

    p.add_argument("--output_dir",     type=str, required=True)
    p.add_argument("--logging_steps",  type=int, default=10)
    p.add_argument("--eval_steps",     type=int, default=500)
    p.add_argument("--report_to",      type=str, default="wandb")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--cache_dir",      type=str, default="./cache")

    # W&B
    p.add_argument("--wandb_api_key",  type=str, default=None)
    p.add_argument("--wandb_project",  type=str, default="Safety-WaRP Utility Finetuning")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # HuggingFace upload
    p.add_argument("--upload_name", type=str, default=None,
                   help="HF repo id (예: kmseong/llama2_7b-arc-ft). 설정 시 학습 후 자동 업로드")
    p.add_argument("--hf_token",    type=str, default=None,
                   help="HuggingFace API 토큰 (없으면 HF_TOKEN 환경변수 사용)")

    # Safety mixing
    p.add_argument("--safety_data_path", type=str,
                   default="./data/circuit_breakers_train.json",
                   help="Safety dataset JSON 경로 (circuit_breakers_train.json)")
    p.add_argument("--safety_mix_ratio", type=float, default=0.0,
                   help="학습 데이터 수 대비 safety 데이터 비율 (예: 0.1=10%, 0=비활성)")

    # LoRA
    p.add_argument("--lora",             action="store_true")
    p.add_argument("--lora_r",           type=int,   default=16)
    p.add_argument("--lora_alpha",       type=int,   default=32)
    p.add_argument("--lora_dropout",     type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str):
    log_dir = "./logs/arc"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_arc_{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


def _as_text(value) -> str:
    return "" if value is None else str(value).strip()


def _select_random_n(ds, n: int, seed: int):
    if n is None or n <= 0 or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


# ──────────────────────────────────────────────────────────────────────────────
# ARC formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_arc_question(question: str, choices: dict) -> str:
    """arc_challenge_chat.yaml 의 doc_to_text 포맷으로 문제 생성."""
    labels = choices["label"]
    texts  = choices["text"]
    choice_lines = []
    for lbl, txt in zip(labels, texts):
        letter = LETTER_MAP.get(str(lbl), str(lbl))
        choice_lines.append(f"{letter}. {txt}")
    return ARC_CHAT_PROMPT_TEMPLATE.format(
        question=question.strip(),
        choices="\n".join(choice_lines),
    )


def get_arc_answer_letter(answer_key: str) -> str:
    """answerKey를 A/B/C/D 레터로 반환."""
    return LETTER_MAP.get(str(answer_key), str(answer_key))


def tokenize_sft_example(
    question_with_choices: str,
    answer_text: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    """SFT 형식으로 토큰화: instruct는 chat template, base는 plain prompt."""
    question_with_choices = _as_text(question_with_choices)
    answer_text = _as_text(answer_text)

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question_with_choices}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": question_with_choices},
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
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        except Exception:
            pass

    # base 모델: plain prompt
    plain_prompt = f"{question_with_choices}\n{ARC_GEN_PREFIX} "
    prompt_ids = tokenizer(
        plain_prompt, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text, add_special_tokens=False, truncation=True, max_length=remain
    )["input_ids"]
    if (tokenizer.eos_token_id is not None
            and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id)
            and len(prompt_ids) + len(answer_ids) < max_length):
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels    = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    if not any(l != -100 for l in labels):
        raise ValueError("tokenization produced no supervised response tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading & tokenization
# ──────────────────────────────────────────────────────────────────────────────

def load_arc_rows(args, logger, eval_dataset: bool = False):
    split = args.eval_split if eval_dataset else args.train_split
    label = "eval" if eval_dataset else "train"

    if eval_dataset and not split:
        logger.info("Eval split not specified, skipping eval dataset")
        return None

    logger.info(f"Loading ARC {label} split='{split}' from {args.dataset_name}/{args.dataset_subset} ...")
    try:
        ds = load_dataset(
            args.dataset_name,
            args.dataset_subset,
            split=split,
            cache_dir=args.cache_dir,
            trust_remote_code=False,
        )
    except Exception as exc:
        logger.error(f"Failed to load ARC dataset: {exc}")
        raise

    n = args.num_eval_samples if eval_dataset else args.num_train_samples
    ds = _select_random_n(ds, n, args.seed + (1 if eval_dataset else 0))
    logger.info(f"Loaded ARC {label} rows: {len(ds):,}")
    return ds


def tokenize_arc_dataset(ds, args, tokenizer, model_path: str, logger, desc: str = "Tokenizing"):
    skipped = 0
    tokenized_data = []
    for idx, ex in enumerate(ds):
        try:
            question_str  = format_arc_question(ex["question"], ex["choices"])
            answer_letter = get_arc_answer_letter(ex["answerKey"])
            answer_text   = f"{ARC_GEN_PREFIX} {answer_letter}"
            tok = tokenize_sft_example(question_str, answer_text, tokenizer, args.max_length, model_path)

            if idx == 0:
                logger.info("\n[ARC Sample #0]")
                logger.info(f"  Question (first 100 chars): {question_str[:100]}...")
                logger.info(f"  Answer: {answer_text}")

            tokenized_data.append(tok)
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipping row {idx}: {e}")

    if skipped:
        logger.warning(f"Skipped {skipped} ARC rows")

    return HFDataset.from_dict({
        "input_ids":      [d["input_ids"]      for d in tokenized_data],
        "attention_mask": [d["attention_mask"] for d in tokenized_data],
        "labels":         [d["labels"]         for d in tokenized_data],
    })


# ──────────────────────────────────────────────────────────────────────────────
# Safety data mixing
# ──────────────────────────────────────────────────────────────────────────────

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

    def preprocess_safety(ex):
        return tokenize_sft_example(
            ex["prompt"],
            ex["llama3_output"],
            tokenizer,
            args.max_length,
            model_path,
        )

    safety_hf  = HFDataset.from_list(sampled)
    safety_tok = safety_hf.map(
        preprocess_safety,
        remove_columns=safety_hf.column_names,
        desc="Tokenizing safety data",
    )

    mixed = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=args.seed)
    logger.info(
        f"Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio}), "
        f"total={len(mixed)}"
    )
    return mixed


# ──────────────────────────────────────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id  = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"]       + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]      * pad_len)
            labels.append(f["labels"]              + [-100]  * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Model & tokenizer loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(args, model_path: str, logger):
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        logger.info("  Tokenizer loaded from local files")
    except Exception as exc:
        logger.warning(f"  Local tokenizer load failed: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    logger.info("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto",
            local_files_only=True, trust_remote_code=False
        )
        logger.info("  Model loaded from local files")
    except Exception as exc:
        logger.warning(f"  Local model load failed: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=False
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.lora:
        if not _peft_available:
            raise ImportError("peft is required for --lora. pip install peft")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"  LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"  Params: total={total/1e9:.2f}B, "
        f"trainable={trainable/1e9:.2f}B ({100*trainable/total:.2f}%)"
    )
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace upload
# ──────────────────────────────────────────────────────────────────────────────

def upload_to_hf(output_dir: str, upload_name: str, hf_token: Optional[str], logger):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub 설치 필요: pip install huggingface_hub")
        return

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logger.error("HF 토큰이 없습니다. --hf_token 또는 HF_TOKEN 환경변수를 설정하세요.")
        return

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=upload_name, repo_type="model", exist_ok=True, private=False)
    except Exception as exc:
        logger.warning(f"Repo creation warning: {exc}")

    logger.info(f"Uploading {output_dir} → {upload_name} ...")
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=upload_name,
            repo_type="model",
            ignore_patterns=[".wandb/*", "wandb/*", "*.log", "__pycache__/*", "cache/*"],
            commit_message="ARC-Challenge fine-tuned model",
        )
        logger.info(f"Upload completed: https://huggingface.co/{upload_name}")
    except Exception as exc:
        logger.error(f"Upload failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb_root = os.path.join(args.output_dir, ".wandb")
        os.makedirs(wandb_root, exist_ok=True)
        os.environ.setdefault("WANDB_DIR",        wandb_root)
        os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(wandb_root, "config"))
        os.environ.setdefault("WANDB_CACHE_DIR",  os.path.join(wandb_root, "cache"))
        os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["WANDB_CACHE_DIR"],  exist_ok=True)

    raw_path   = args.model_path
    is_local   = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger, log_file = setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("Full-parameter utility fine-tuning: ARC-Challenge")
    logger.info("=" * 70)
    logger.info(f"Log file   : {log_file}")
    logger.info(f"Model      : {model_path}")
    logger.info(f"Input fmt  : {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"Dataset    : {args.dataset_name} / {args.dataset_subset}")
    logger.info(f"Train split: {args.train_split}, Eval split: {args.eval_split or '(none)'}")
    logger.info(f"Safety mix : ratio={args.safety_mix_ratio}, path={args.safety_data_path}")
    logger.info(f"LR={args.learning_rate}, epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}")

    if args.bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        args.bf16 = False
        if not args.fp16:
            args.fp16 = True

    model, tokenizer = load_model_and_tokenizer(args, model_path, logger)

    train_rows = load_arc_rows(args, logger, eval_dataset=False)
    eval_rows  = load_arc_rows(args, logger, eval_dataset=True)

    train_tok = tokenize_arc_dataset(train_rows, args, tokenizer, model_path, logger, desc="Tokenizing train")
    eval_tok  = None
    if eval_rows is not None and args.num_eval_samples > 0:
        eval_tok = tokenize_arc_dataset(eval_rows, args, tokenizer, model_path, logger, desc="Tokenizing eval")

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
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=do_eval,
        report_to=args.report_to,
        run_name=args.wandb_run_name or os.path.basename(args.output_dir),
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    logger.info("Training started...")
    trainer.train()
    logger.info("Training completed.")

    logger.info(f"Saving model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.upload_name:
        upload_to_hf(args.output_dir, args.upload_name, args.hf_token, logger)


if __name__ == "__main__":
    main()
