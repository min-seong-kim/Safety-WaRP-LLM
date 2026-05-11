"""
Freeze-safety-neuron fine-tuning for MMLU.

This script trains on MMLU while freezing detected safety neurons and updating
all other parameters.

Example:
python mmlu_eval/finetune_mmlu_freeze_sn.py \
  --model_path kmseong/llama2_7b_chat_gsm8k_ft_freeze_sn_lr5e-5_revised \
  --safety_neurons_file /NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong/Safety-Neuron/neuron_detection/output_neurons/safety_neuron_accelerated_20260502_013602.txt \
  --mmlu_subject all \
  --mmlu_split auxiliary_train \
  --num_train_samples 8000 \
  --output_dir ./llama2_7b_chat_mmlu_freeze_sn_lr5e-5 \
  --learning_rate 5e-5 --epochs 3 \
  --upload_name kmseong/llama2_7b_chat_mmlu_freeze_sn_lr5e-5


python mmlu_eval/finetune_mmlu_freeze_sn.py \
  --model_path kmseong/llama2_7b_chat_gsm8k_ft_freeze_sn_lr5e-5_revised \
  --safety_neurons_file /NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong/Safety-Neuron/neuron_detection/output_neurons/critical_safety_neuron_20260502_022558.txt \
  --mmlu_subject all \
  --mmlu_split auxiliary_train \
  --num_train_samples 8000 \
  --output_dir ./llama2_7b_chat_mmlu_freeze_rsn_lr5e-5 \
  --learning_rate 5e-5 --epochs 3 \
  --upload_name kmseong/llama2_7b_chat_mmlu_freeze_rsn_lr5e-5
"""

import argparse
import ast
import json
import logging
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _peft_available = True
except ImportError:
    _peft_available = False

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

MMLU_CHOICES = ["A", "B", "C", "D"]
MMLU_INSTRUCTION = (
    "The following is a multiple choice question. "
    "Choose the single best answer from A, B, C, or D."
)
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
    p = argparse.ArgumentParser(description="MMLU finetuning with safety-neuron freezing")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--safety_neurons_file", type=str, required=True)

    p.add_argument("--mmlu_subject", type=str, default="all")
    p.add_argument(
        "--mmlu_split",
        type=str,
        default="auxiliary_train",
        choices=["auxiliary_train", "train", "validation", "test", "dev"],
    )
    p.add_argument(
        "--mmlu_eval_split",
        type=str,
        default="validation",
        choices=["auxiliary_train", "train", "validation", "test", "dev"],
    )
    p.add_argument("--num_train_samples", type=int, default=10000)
    p.add_argument("--num_eval_samples", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", type=str, default="adamw_torch")
    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="Safety-Neuron Utility Finetuning")
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--upload_name", type=str, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    p.add_argument(
        "--safety_data_path",
        type=str,
        default="./data/circuit_breakers_train.json",
    )
    p.add_argument("--safety_mix_ratio", type=float, default=0.0)

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


def setup_logging():
    log_dir = "./logs/safety_neuron_mmlu"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_mmlu_freeze_sn_{ts}.log")

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


def _keep_answer_budget(prompt_ids: List[int], answer_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
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


def tokenize_prompt_response(prompt: str, response: str, tokenizer, max_length: int, model_ref: str) -> Dict[str, List[int]]:
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

    choice_lines = "\n".join(
        f"{MMLU_CHOICES[i]}. {_as_text(choices[i])}" for i in range(min(len(choices), len(MMLU_CHOICES)))
    )
    prompt = f"{MMLU_INSTRUCTION}\n\nQuestion: {question}\n\n{choice_lines}"
    response = MMLU_CHOICES[answer_idx]
    return prompt, response


def load_mmlu_dataset(args, logger, eval_dataset: bool = False):
    split = args.mmlu_eval_split if eval_dataset else args.mmlu_split
    subject = args.mmlu_subject

    if subject == "all":
        if split == "auxiliary_train" and not eval_dataset:
            logger.info("Loading cais/mmlu 'all' auxiliary_train split")
            ds = load_dataset("cais/mmlu", "all", split="auxiliary_train", cache_dir=args.cache_dir)
        else:
            logger.info(f"Loading cais/mmlu all subjects, split={split}")
            subject_datasets = []
            for subj in MMLU_ALL_SUBJECTS:
                try:
                    subj_ds = load_dataset("cais/mmlu", subj, split=split, cache_dir=args.cache_dir)
                    subject_datasets.append(subj_ds)
                except Exception as exc:
                    logger.warning(f"Could not load subject '{subj}': {exc}")
            if not subject_datasets:
                raise ValueError(f"No MMLU subjects loaded for split '{split}'")
            ds = concatenate_datasets(subject_datasets)
            ds = ds.sort(["subject", "question"])
    else:
        logger.info(f"Loading cais/mmlu subject='{subject}', split={split}")
        ds = load_dataset("cais/mmlu", subject, split=split, cache_dir=args.cache_dir)

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
                tokenize_prompt_response(prompt, response, tokenizer, args.max_length, model_path)
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


def load_safety_neurons(output_file, logger):
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    safety_neurons = {}
    try:
        ffn_up_raw = ast.literal_eval(lines[0].strip())
        ffn_down_raw = ast.literal_eval(lines[1].strip())
        q_raw = ast.literal_eval(lines[2].strip())
        k_raw = ast.literal_eval(lines[3].strip())
        v_raw = ast.literal_eval(lines[4].strip())

        safety_neurons["ffn_up"] = {int(k): v for k, v in ffn_up_raw.items()}
        safety_neurons["ffn_down"] = {int(k): v for k, v in ffn_down_raw.items()}
        safety_neurons["q"] = {int(k): v for k, v in q_raw.items()}
        safety_neurons["k"] = {int(k): v for k, v in k_raw.items()}
        safety_neurons["v"] = {int(k): v for k, v in v_raw.items()}
    except Exception as e:
        logger.error(f"Error parsing safety neurons file: {e}")
        raise

    total_neurons = 0
    for module_type in ["ffn_up", "ffn_down", "q", "k", "v"]:
        total_neurons += sum(len(neurons) for neurons in safety_neurons[module_type].values())

    logger.info(f"Loaded safety neurons from {output_file} (total={total_neurons})")
    return safety_neurons


def setup_safety_neuron_freezing(model, safety_neurons, logger):
    frozen_param_specs = []

    def _sanitize_indices(raw_indices, dim: int, module_name: str, layer_idx: int):
        parsed = []
        dropped = 0
        for x in raw_indices:
            idx = None
            if isinstance(x, int):
                idx = x
            elif isinstance(x, str):
                s = x.strip()
                if s.lstrip("-").isdigit():
                    idx = int(s)
                else:
                    m = re.search(r"-?\d+", s)
                    if m:
                        idx = int(m.group(0))
            if idx is None:
                dropped += 1
                continue
            if 0 <= idx < dim:
                parsed.append(idx)
            else:
                dropped += 1
        uniq = sorted(set(parsed))
        if dropped > 0:
            logger.warning(
                f"[Index sanitize] layer={layer_idx}, module={module_name}, "
                f"kept={len(uniq)}, dropped={dropped}, dim={dim}"
            )
        return uniq

    def _make_zero_hook_rows(indices):
        def hook(grad):
            grad = grad.clone()
            grad[indices, :] = 0.0
            return grad

        return hook

    def _make_zero_hook_cols(indices):
        def hook(grad):
            grad = grad.clone()
            grad[:, indices] = 0.0
            return grad

        return hook

    for param in model.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            continue
        try:
            layer_idx = int(parts[2])
        except ValueError:
            continue

        if "mlp.up_proj.weight" in name:
            neuron_indices = _sanitize_indices(safety_neurons["ffn_up"].get(layer_idx, []), param.shape[0], "ffn_up", layer_idx)
            if neuron_indices:
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, "rows"))

        elif "mlp.down_proj.weight" in name:
            neuron_indices = _sanitize_indices(safety_neurons["ffn_down"].get(layer_idx, []), param.shape[1], "ffn_down", layer_idx)
            if neuron_indices:
                param.register_hook(_make_zero_hook_cols(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, "cols"))

        elif "self_attn.q_proj.weight" in name:
            neuron_indices = _sanitize_indices(safety_neurons["q"].get(layer_idx, []), param.shape[0], "q", layer_idx)
            if neuron_indices:
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, "rows"))

        elif "self_attn.k_proj.weight" in name:
            neuron_indices = _sanitize_indices(safety_neurons["k"].get(layer_idx, []), param.shape[0], "k", layer_idx)
            if neuron_indices:
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, "rows"))

        elif "self_attn.v_proj.weight" in name:
            neuron_indices = _sanitize_indices(safety_neurons["v"].get(layer_idx, []), param.shape[0], "v", layer_idx)
            if neuron_indices:
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, "rows"))

    logger.info(f"Registered safety-neuron freeze hooks: {len(frozen_param_specs)} params")
    return frozen_param_specs


class SafetyNeuronRestoreCallback(TrainerCallback):
    def __init__(self, frozen_param_specs):
        self._specs = frozen_param_specs
        self._frozen_vals = []
        for param, indices, axis in frozen_param_specs:
            with torch.no_grad():
                if axis == "rows":
                    self._frozen_vals.append(param.data[indices, :].clone())
                else:
                    self._frozen_vals.append(param.data[:, indices].clone())

    def on_step_end(self, args, state, control, **kwargs):
        for (param, indices, axis), frozen_val in zip(self._specs, self._frozen_vals):
            with torch.no_grad():
                if axis == "rows":
                    param.data[indices, :] = frozen_val
                else:
                    param.data[:, indices] = frozen_val
        return control


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

    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger, log_file = setup_logging()
    logger.info("=" * 70)
    logger.info("MMLU fine-tuning with safety-neuron freezing")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")

    if not os.path.exists(args.safety_neurons_file):
        raise FileNotFoundError(f"Safety neurons file not found: {args.safety_neurons_file}")

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model, tokenizer = load_model_and_tokenizer(args, model_path, logger)

    safety_neurons = load_safety_neurons(args.safety_neurons_file, logger)
    frozen_param_specs = setup_safety_neuron_freezing(model, safety_neurons, logger)

    train_rows = load_mmlu_dataset(args, logger, eval_dataset=False)
    eval_rows = load_mmlu_dataset(args, logger, eval_dataset=True) if args.num_eval_samples > 0 else None

    train_tok = tokenize_dataset_rows(train_rows, args, tokenizer, model_path, logger)
    eval_tok = tokenize_dataset_rows(eval_rows, args, tokenizer, model_path, logger) if eval_rows is not None else None

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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        eval_strategy=("steps" if do_eval else "no"),
        eval_steps=(args.eval_steps if do_eval else None),
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim=args.optim,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
        callbacks=[SafetyNeuronRestoreCallback(frozen_param_specs)],
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving fine-tuned model")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model": model_path,
        "fine_tuning_type": "MMLU Finetuning with Safety Neuron Freezing",
        "safety_neurons_file": args.safety_neurons_file,
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
        "optimizer": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "default"),
        "strategy": "Freeze safety neurons, train all other params",
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
                upload_dir = os.path.join(temp_dir, os.path.basename(os.path.normpath(args.output_dir)))
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

    logger.info("Fine-tuning complete")


if __name__ == "__main__":
    main()
