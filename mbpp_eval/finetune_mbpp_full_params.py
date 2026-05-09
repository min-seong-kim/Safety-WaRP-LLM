"""
Full-parameter or LoRA fine-tuning for MBPP (Mostly Basic Python Problems).

Example usage:

Full-parameter:
python mbpp_eval/finetune_mbpp_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./mbpp_eval/llama2_7b_chat_mbpp_FT_lr5e-5 \
    --learning_rate 3e-5 --epochs 3 \
    --upload_name kmseong/llama2_7b_chat-MBPP-FT-lr3e-5

SafeInstr (safety mixing):
python mbpp_eval/finetune_mbpp_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./mbpp_eval/llama2_7b_chat_mbpp_FT_safeInstr_lr1e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --safety_mix_ratio 0.1 \
    --upload_name kmseong/llama2_7b_chat-MBPP-FT-safety-mix-0.1-lr5e-5

LoRA:
python mbpp_eval/finetune_mbpp_full_params.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --output_dir ./mbpp_eval/lora_mbpp_llama2_7b \
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
from pathlib import Path
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

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _peft_available = True
except ImportError:
    _peft_available = False


MBPP_INSTRUCTION = (
    "Write a Python function to solve the following programming problem. "
    "Provide only the complete, runnable Python code without any explanation.\n"
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter or LoRA SFT for MBPP.")

    p.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")

    # MBPP dataset (loaded from HuggingFace Hub; no local path required)
    p.add_argument("--mbpp_dataset_name", type=str, default="google-research-datasets/mbpp",
                   help="HuggingFace dataset name for MBPP")
    p.add_argument("--mbpp_subset",       type=str, default="full",
                   help="Dataset subset/config (full | sanitized)")
    p.add_argument("--mbpp_train_split",  type=str, default="train",
                   help="Split name used for training")
    p.add_argument("--mbpp_eval_split",   type=str, default="validation",
                   help="Split name used for evaluation (set to '' to skip)")

    p.add_argument("--num_train_samples", type=int, default=0,
                   help="학습 샘플 수 (0=전체)")
    p.add_argument("--num_eval_samples",  type=int, default=0,
                   help="평가 샘플 수 (0=전체, eval 비활성 시 무시)")
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--eval_batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",        type=int,   default=4)
    p.add_argument("--epochs",            type=int,   default=3)
    p.add_argument("--learning_rate",     type=float, default=5e-5)
    p.add_argument("--weight_decay",      type=float, default=0.01)
    p.add_argument("--warmup_ratio",      type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str,   default="cosine")
    p.add_argument("--max_grad_norm",     type=float, default=1.0)
    p.add_argument("--max_length",        type=int,   default=1024)

    p.add_argument("--bf16",                   action="store_true", default=True)
    p.add_argument("--fp16",                   action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--output_dir",    type=str, required=True)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps",    type=int, default=500)
    p.add_argument("--report_to",     type=str, default="wandb")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--cache_dir",     type=str, default="./cache")

    # W&B
    p.add_argument("--wandb_api_key",  type=str, default=None)
    p.add_argument("--wandb_project",  type=str, default="Safety-WaRP Utility Finetuning")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # HuggingFace upload
    p.add_argument("--upload_name", type=str, default=None,
                   help="HF repo id (예: kmseong/llama2_7b-mbpp-ft). 설정 시 학습 후 자동 업로드")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace API 토큰 (없으면 HF_TOKEN 환경변수 사용)")

    # Safety mixing
    p.add_argument("--safety_data_path", type=str,
                   default="/home/yonsei_jong/Safety-WaRP-LLM/data/circuit_breakers_train.json",
                   help="Safety dataset JSON 경로 (circuit_breakers_train.json)")
    p.add_argument("--safety_mix_ratio", type=float, default=0.0,
                   help="학습 데이터 수 대비 safety 데이터 비율 (예: 0.1=10%, 0=비활성)")

    # LoRA
    p.add_argument("--lora",            action="store_true")
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str):
    log_dir = "./logs/mbpp"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_mbpp_{timestamp}.log")

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


def _keep_answer_budget(
    prompt_ids: List[int],
    answer_ids: List[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    if len(prompt_ids) + len(answer_ids) <= max_length:
        return prompt_ids, answer_ids
    if max_length <= 1:
        raise ValueError(f"max_length must be > 1, got {max_length}")
    # 코드 정답은 길 수 있으므로 최소 128 토큰 보장
    answer_floor = max(128, max_length // 4)
    answer_budget = min(len(answer_ids), answer_floor)
    prompt_budget = max_length - answer_budget
    answer_ids = answer_ids[:answer_budget]
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids, answer_ids


def render_chat_fallback(prompt: str, response: str, model_ref: str) -> Tuple[str, str]:
    prompt = prompt.strip()
    if "llama-2" in str(model_ref).lower():
        prompt_text = f"<s>[INST] {prompt} [/INST]"
        return prompt_text, f"{prompt_text} {response.strip()} </s>"
    prompt_text = f"User:\n{prompt}\n\nAssistant:\n"
    return prompt_text, f"{prompt_text}{response.strip()}"


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
            full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids):]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
        except Exception:
            prompt_text, full_text = render_chat_fallback(prompt, response, model_ref)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids):]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
    else:
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)

    prompt_ids, answer_ids = _keep_answer_budget(prompt_ids, answer_ids, max_length)

    input_ids = prompt_ids + answer_ids
    labels    = [-100] * len(prompt_ids) + answer_ids
    if not any(label != -100 for label in labels):
        raise ValueError("tokenization produced no supervised response tokens")

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MBPP row → (prompt, response)
# ──────────────────────────────────────────────────────────────────────────────

def mbpp_prompt_response(row: Dict, prefer_chat: bool = False) -> Tuple[str, str]:
    """
    MBPP 데이터셋 포맷 (google-research-datasets/mbpp):
      row["text"]      = 문제 설명 (자연어)
      row["code"]      = 정답 파이썬 코드
      row["test_list"] = 테스트 케이스 목록 (list of str)

    chat 모델: instruction + problem → user 메시지
    base 모델: alpaca-style plain prompt
    """
    problem  = _as_text(row.get("text", ""))
    solution = _as_text(row.get("code", ""))

    # 테스트 케이스를 힌트로 포함 (있는 경우)
    test_list = row.get("test_list", []) or []
    if test_list:
        tests_str = "\n".join(f"  {t}" for t in test_list[:3])
        problem_with_tests = f"{problem}\n\nYour code should pass the following tests:\n{tests_str}"
    else:
        problem_with_tests = problem

    if prefer_chat:
        user_content = f"{MBPP_INSTRUCTION}\n{problem_with_tests}"
        return user_content, solution

    # base 모델: alpaca-style
    prompt = (
        f"### Instruction:\n{MBPP_INSTRUCTION}\n\n"
        f"### Input:\n{problem_with_tests}\n\n"
        f"### Response:\n"
    )
    return prompt, solution


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading & tokenization
# ──────────────────────────────────────────────────────────────────────────────

def load_mbpp_rows(args, logger, eval_dataset: bool = False):
    split = args.mbpp_eval_split if eval_dataset else args.mbpp_train_split
    label = "eval" if eval_dataset else "train"

    if eval_dataset and not split:
        logger.info("Eval split not specified, skipping eval dataset")
        return None

    logger.info(f"Loading MBPP {label} split='{split}' from {args.mbpp_dataset_name} ...")
    try:
        ds = load_dataset(
            args.mbpp_dataset_name,
            args.mbpp_subset,
            split=split,
            cache_dir=args.cache_dir,
            trust_remote_code=False,
        )
    except Exception as exc:
        logger.error(f"Failed to load MBPP dataset: {exc}")
        raise

    n = args.num_eval_samples if eval_dataset else args.num_train_samples
    ds = _select_random_n(ds, n, args.seed + (1 if eval_dataset else 0))
    logger.info(f"Loaded MBPP {label} rows: {len(ds):,}")
    return ds


def tokenize_dataset_rows(ds, args, tokenizer, model_path: str, logger, desc: str = "Tokenizing"):
    prefer_chat = is_instruct_model(model_path)

    def preprocess(ex):
        prompt, response = mbpp_prompt_response(dict(ex), prefer_chat=prefer_chat)
        return tokenize_prompt_response(prompt, response, tokenizer, args.max_length, model_path)

    return ds.map(
        preprocess,
        remove_columns=ds.column_names,
        num_proc=max(1, args.num_workers),
        desc=desc,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Safety data mixing
# ──────────────────────────────────────────────────────────────────────────────

def maybe_mix_safety(train_tok, args, tokenizer, model_path: str, logger,
                     num_mbpp: int = 0):
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
        return tokenize_prompt_response(
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
    logger.info(f"Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio})")
    logger.info(
        f"Total training samples: {len(mixed)} "
        f"(MBPP {num_mbpp} + Safety {len(safety_tok)})"
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
            input_ids.append(f["input_ids"]      + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]      * pad_len)
            labels.append(f["labels"]             + [-100]  * pad_len)

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
        logger.info(f"Repo created/found: {upload_name}")
    except Exception as exc:
        logger.warning(f"Repo creation warning: {exc}")

    ignore_patterns = [".wandb/*", "wandb/*", "*.log", "__pycache__/*", "cache/*"]
    logger.info(f"Uploading {output_dir} → {upload_name} ...")
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=upload_name,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            commit_message="MBPP fine-tuned model",
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

    # W&B 환경변수 설정
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb_root = os.path.join(args.output_dir, ".wandb")
        os.makedirs(wandb_root, exist_ok=True)
        os.environ.setdefault("WANDB_DIR",        wandb_root)
        os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(wandb_root, "config"))
        os.environ.setdefault("WANDB_CACHE_DIR",  os.path.join(wandb_root, "cache"))
        os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["WANDB_CACHE_DIR"],  exist_ok=True)

    raw_path  = args.model_path
    is_local  = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger, log_file = setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("Full-parameter utility fine-tuning: MBPP")
    logger.info("=" * 70)
    logger.info(f"Log file  : {log_file}")
    logger.info(f"Model     : {model_path}")
    logger.info(f"Input fmt : {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"Dataset   : {args.mbpp_dataset_name} / {args.mbpp_subset}")
    logger.info(f"Train split: {args.mbpp_train_split}, Eval split: {args.mbpp_eval_split or '(none)'}")
    logger.info(f"Safety mix: ratio={args.safety_mix_ratio}, path={args.safety_data_path}")
    logger.info(f"LR={args.learning_rate}, epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}")

    model, tokenizer = load_model_and_tokenizer(args, model_path, logger)

    train_rows = load_mbpp_rows(args, logger, eval_dataset=False)
    eval_rows  = load_mbpp_rows(args, logger, eval_dataset=True)

    train_tok = tokenize_dataset_rows(train_rows, args, tokenizer, model_path, logger, desc="Tokenizing train")
    eval_tok  = None
    if eval_rows is not None and args.num_eval_samples > 0:
        eval_tok = tokenize_dataset_rows(eval_rows, args, tokenizer, model_path, logger, desc="Tokenizing eval")

    num_mbpp  = len(train_tok)
    train_tok = maybe_mix_safety(train_tok, args, tokenizer, model_path, logger, num_mbpp=num_mbpp)

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
                "model_path":        model_path,
                "dataset":           "mbpp",
                "mbpp_subset":       args.mbpp_subset,
                "learning_rate":     args.learning_rate,
                "epochs":            args.epochs,
                "batch_size":        args.batch_size,
                "grad_accum":        args.grad_accum,
                "effective_bs":      args.batch_size * args.grad_accum,
                "max_length":        args.max_length,
                "weight_decay":      args.weight_decay,
                "warmup_ratio":      args.warmup_ratio,
                "lr_scheduler":      args.lr_scheduler_type,
                "is_instruct":       is_instruct_model(model_path),
                "lora":              args.lora,
                "safety_mix_ratio":  args.safety_mix_ratio,
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

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving fine-tuned model...")
    if args.lora:
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model":              model_path,
        "fine_tuning_type":        "LoRA Fine-tuning" if args.lora else "Full Parameter Fine-tuning",
        "dataset":                 "mbpp",
        "mbpp_dataset_name":       args.mbpp_dataset_name,
        "mbpp_subset":             args.mbpp_subset,
        "mbpp_train_split":        args.mbpp_train_split,
        "mbpp_eval_split":         args.mbpp_eval_split,
        "num_train_samples":       len(train_tok),
        "num_eval_samples":        len(eval_tok) if eval_tok is not None else 0,
        "batch_size":              args.batch_size,
        "grad_accum":              args.grad_accum,
        "learning_rate":           args.learning_rate,
        "weight_decay":            args.weight_decay,
        "warmup_ratio":            args.warmup_ratio,
        "epochs":                  args.epochs,
        "max_length":              args.max_length,
        "max_grad_norm":           args.max_grad_norm,
        "lr_scheduler_type":       args.lr_scheduler_type,
        "optimizer":               "AdamW (torch)",
        "gradient_checkpointing":  args.gradient_checkpointing,
        "dtype":                   "bf16" if args.bf16 else ("fp16" if args.fp16 else "default"),
        "trainer_type":            "Trainer",
        "safety_mix_ratio":        args.safety_mix_ratio,
        "safety_data_path":        args.safety_data_path if args.safety_mix_ratio > 0 else None,
    }
    config_path = os.path.join(args.output_dir, "finetune_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Config saved to {config_path}")

    if args.upload_name:
        logger.info(f"Uploading to HuggingFace: {args.upload_name}")
        upload_to_hf(args.output_dir, args.upload_name, args.hf_token, logger)

    if args.report_to and args.report_to != "none":
        wandb.finish()

    logger.info("Fine-tuning complete")


if __name__ == "__main__":
    main()
