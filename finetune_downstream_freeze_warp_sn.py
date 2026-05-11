#!/usr/bin/env python3
"""
WaRP-Space Downstream Fine-tuning with Safety Neuron Column Freezing

Downstream fine-tune (e.g. GSM8K, MBPP) on a WaRP-SN model while keeping
the detected safety neuron columns in WaRP-reparameterized space frozen.

Pipeline
--------
1. Load WaRP-SN-Tuned model  (standard HF checkpoint — weights in original W space)
2. Load Phase-1 safety basis  (U matrices per layer/type)
3. Reparameterize:  basis_coeff = W @ U,  flag=True
   forward: output = F.linear(x, basis_coeff @ U^T)  — mathematically identical
4. Register gradient hook on each basis_coeff:
       grad[:, safety_cols] = 0
   → safety columns never accumulate a gradient signal
5. SafetyColRestoreCallback — runs after every optimizer step to counteract
   AdamW weight-decay drift on safety columns  (wd * θ acts even when grad=0)
6. Train with HF Trainer on downstream dataset
7. Save:  restore_weight(model)      W_new = basis_coeff_trained @ U^T
          restore_to_linear(model)   LinearWaRP → nn.Linear
          model.save_pretrained()    → standard HF checkpoint

Relationship to other scripts
------------------------------
- warp_sn_tune.py:    grad[:, non_safety_cols] = 0  (only safety cols update)
- THIS script:        grad[:, safety_cols]     = 0  (only non-safety cols update)

Usage
-----
python finetune_downstream_freeze_warp_sn.py \\
    --model_name_or_path /path/to/warp_sn_tuned_model \\
    --basis_dir          /path/to/phase1/basis \\
    --neuron_file        /path/to/warp_safety_neurons_TIMESTAMP.txt \\
    --task               gsm8k \\
    --learning_rate      5e-5 \\
    --num_epochs         3 \\
    --output_dir         ./downstream_warp_sn_gsm8k
"""

import argparse
import ast
import gc
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# ── WaRP project imports ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from models.warp_modules import (
    LinearWaRP,
    switch_to_warp_module,
    restore_weight,
    restore_to_linear,
)

# ── Layer-type → (submodule, attr) mappings ───────────────────────────────────
_LAYER_TYPE_TO_PATH: Dict[str, Tuple[str, str]] = {
    "ffn_up":   ("mlp",       "up_proj"),
    "ffn_down": ("mlp",       "down_proj"),
    "attn_q":   ("self_attn", "q_proj"),
    "attn_k":   ("self_attn", "k_proj"),
    "attn_v":   ("self_attn", "v_proj"),
}

# layer_type → key in the 5-line safety-neuron file
_LAYER_TYPE_TO_SN_KEY: Dict[str, str] = {
    "ffn_up":   "ffn_up",
    "ffn_down": "ffn_down",
    "attn_q":   "q",
    "attn_k":   "k",
    "attn_v":   "v",
}

logger = logging.getLogger(__name__)

# ── Task configs ──────────────────────────────────────────────────────────────
_TASK_CONFIG = {
    "gsm8k": {
        "dataset_name":    "openai/gsm8k",
        "subset":          "main",
        "split":           "train",
        "question_field":  "question",
        "answer_field":    "answer",
        "base_prompt_fn":  lambda q: f"Question: {q.strip()}\nAnswer:",
    },
    "mbpp": {
        "dataset_name":    "google-research-datasets/mbpp",
        "subset":          "sanitized",
        "split":           "train",
        "question_field":  "text",
        "answer_field":    "code",
        "base_prompt_fn":  lambda q: f"Problem: {q.strip()}\nCode:",
    },
}


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WaRP-Space Downstream Fine-tuning with Safety Neuron Column Freezing"
    )
    # ── WaRP-specific ──
    p.add_argument("--model_name_or_path", required=True,
                   help="Path or HF hub ID of the WaRP-SN-Tuned model")
    p.add_argument("--basis_dir", required=True,
                   help="Phase-1 basis directory (ffn_up/, attn_q/, …)")
    p.add_argument("--neuron_file", required=True,
                   help="WaRP-space safety neuron file (5-line JSON produced by WaRPSNDetector)")
    p.add_argument("--layer_types", default="ffn_up,ffn_down,attn_q,attn_k,attn_v",
                   help="Comma-separated layer types to reparameterize")

    # ── Task ──
    p.add_argument("--task", default="gsm8k", choices=list(_TASK_CONFIG.keys()))
    p.add_argument("--num_train_samples", type=int, default=None,
                   help="Limit number of training samples (None = use all)")
    p.add_argument("--cache_dir", default="./cache")

    # ── Training ──
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_epochs",    type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length",    type=int,   default=1024)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--report_to", default="none")

    # ── Upload ──
    p.add_argument("--upload_name", default=None,
                   help="HuggingFace repo id for upload after training")
    p.add_argument("--hf_token", default=None)

    return p.parse_args()


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: str) -> Tuple[logging.Logger, str]:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"warp_sn_freeze_ft_{ts}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

    return logging.getLogger(__name__), log_path


# =============================================================================
# WaRP utilities
# =============================================================================

def load_basis(
    basis_dir: str,
    layer_types: List[str],
    num_layers: int,
) -> Dict[Tuple[int, str], dict]:
    """Load Phase-1 SVD basis files (U matrices)."""
    basis_data: Dict[Tuple[int, str], dict] = {}
    loaded, missing = 0, []

    for layer_type in layer_types:
        lt_dir = os.path.join(basis_dir, layer_type)
        if not os.path.isdir(lt_dir):
            logger.warning(f"  No basis dir for '{layer_type}': {lt_dir}")
            continue
        for layer_idx in range(num_layers):
            path = os.path.join(lt_dir, f"layer_{layer_idx:02d}_svd.pt")
            if not os.path.exists(path):
                missing.append((layer_idx, layer_type))
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            U = data.get("U")
            if not isinstance(U, torch.Tensor):
                logger.warning(f"  No 'U' key in {path}, skipping")
                continue
            basis_data[(layer_idx, layer_type)] = {"U": U}
            loaded += 1

    logger.info(f"  Loaded {loaded} basis files ({len(missing)} missing)")
    if loaded == 0:
        raise RuntimeError(f"No basis files found in {basis_dir}")
    return basis_data


def load_warp_safety_neurons(neuron_file: str) -> Dict[str, Dict[int, Set[int]]]:
    """
    Load WaRP-space safety neuron file (5-line JSON produced by WaRPSNDetector).

    Line 0: ffn_up   {"0": [col_idx, ...], "1": [...], ...}
    Line 1: ffn_down
    Line 2: q        (column indices of attn_q basis_coeff)
    Line 3: k
    Line 4: v

    Returns: {'ffn_up': {layer_idx: set(col_indices)}, 'q': ..., ...}
    """
    with open(neuron_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sn_keys = ["ffn_up", "ffn_down", "q", "k", "v"]
    result: Dict[str, Dict[int, Set[int]]] = {}
    for i, key in enumerate(sn_keys):
        raw = ast.literal_eval(lines[i].strip())
        result[key] = {int(k): set(v) for k, v in raw.items()}

    total = sum(sum(len(s) for s in d.values()) for d in result.values())
    logger.info(f"  Safety neurons (WaRP column indices): {total} total")
    for key in sn_keys:
        n = sum(len(s) for s in result[key].values())
        logger.info(f"    {key:12}: {n}")
    return result


def reparameterize_model(
    model,
    basis_data: Dict[Tuple[int, str], dict],
    layer_types: List[str],
    num_layers: int,
):
    """
    Convert nn.Linear → LinearWaRP and compute basis_coeff = W @ U.
    flag=True: forward = F.linear(x, basis_coeff @ U^T)  — identical to F.linear(x, W).
    Gradient flows through basis_coeff, not through the weight buffer.
    """
    model = switch_to_warp_module(model, layer_types, target_layers="all")

    rep, skipped = 0, 0
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for layer_type in layer_types:
            key = (layer_idx, layer_type)
            if key not in basis_data:
                skipped += 1
                continue
            sub, attr = _LAYER_TYPE_TO_PATH[layer_type]
            module = getattr(getattr(layer, sub), attr)
            if not isinstance(module, LinearWaRP):
                skipped += 1
                continue

            W = module.weight.data
            U = basis_data[key]["U"].to(dtype=W.dtype, device=W.device)

            module.basis_coeff.data.copy_(W @ U)
            module.UT_forward = U.clone().detach()
            module.UT_backward = torch.empty(0, dtype=W.dtype, device=W.device)
            module.flag = True
            module.coeff_mask.data.zero_()
            if hasattr(module, "mask_mode"):
                module.mask_mode.fill_(1)  # mask_mode=1 → coeff = basis_coeff (no detach)

            rep += 1

    logger.info(f"  Reparameterized: {rep} modules (flag=True) | Skipped: {skipped}")
    return model


# =============================================================================
# Safety column freezing
# =============================================================================

class SafetyColRestoreCallback(TrainerCallback):
    """
    Restores safety-column values of basis_coeff after every optimizer step.

    Motivation
    ----------
    AdamW weight-decay:  θ ← θ − lr·(∇θ + wd·θ)
    Even when grad[:, safety_cols] = 0 (from the gradient hook), the decay
    term  wd·θ  still shifts those columns toward zero.  This callback
    writes back the original pre-training values, guaranteeing true
    isolation of safety columns regardless of weight-decay magnitude.

    Values are stored as float32 on CPU to minimise GPU memory overhead.
    They are cast to the parameter's dtype/device on every restore call.
    """

    def __init__(self, specs: List[Tuple[torch.nn.Parameter, torch.Tensor]]):
        """
        Parameters
        ----------
        specs : list of (basis_coeff_param, col_indices_LongTensor)
        """
        self._specs = specs
        self._frozen_vals: List[torch.Tensor] = []
        for param, cols in specs:
            with torch.no_grad():
                self._frozen_vals.append(
                    param.data[:, cols].clone().cpu().float()
                )

    def on_step_end(self, args, state, control, **kwargs):
        for (param, cols), fv in zip(self._specs, self._frozen_vals):
            with torch.no_grad():
                param.data[:, cols] = fv.to(dtype=param.dtype, device=param.device)
        return control


def setup_safety_col_freezing(
    model,
    basis_data: Dict[Tuple[int, str], dict],
    safety_neurons: Dict[str, Dict[int, Set[int]]],
    layer_types: List[str],
    num_layers: int,
) -> Tuple[List[Tuple[torch.nn.Parameter, torch.Tensor]], List]:
    """
    Register backward hooks on basis_coeff to zero safety-column gradients.

    Effect
    ------
    All model parameters remain trainable (requires_grad=True).
    For every LinearWaRP basis_coeff, a hook zeros grad[:, safety_cols]
    so safety columns never accumulate any update signal.

    Returns
    -------
    specs        : list of (basis_coeff_param, col_indices) for SafetyColRestoreCallback
    hook_handles : list of RemovableHook handles (call .remove() before saving)
    """
    # Ensure all parameters are trainable
    for p in model.parameters():
        p.requires_grad = True

    specs: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    hook_handles = []
    frozen_params = 0
    modules_masked = 0

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for layer_type in layer_types:
            key = (layer_idx, layer_type)
            if key not in basis_data:
                continue
            sub, attr = _LAYER_TYPE_TO_PATH[layer_type]
            module = getattr(getattr(layer, sub), attr)
            if not isinstance(module, LinearWaRP):
                continue

            sn_key = _LAYER_TYPE_TO_SN_KEY.get(layer_type)
            if sn_key is None:
                continue

            safety_col_set: Set[int] = set(
                safety_neurons.get(sn_key, {}).get(layer_idx, set())
            )
            if not safety_col_set:
                continue

            d_out, d_in = module.basis_coeff.shape
            valid_cols = sorted({c for c in safety_col_set if 0 <= c < d_in})
            if not valid_cols:
                logger.warning(
                    f"  Layer {layer_idx} {layer_type}: no valid safety cols "
                    f"(d_in={d_in}), skipping"
                )
                continue

            col_tensor = torch.tensor(valid_cols, dtype=torch.long)

            # Backward hook: zero safety columns in the gradient
            # (non-safety columns pass through unchanged → they receive full updates)
            def _make_hook(cols: torch.Tensor):
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    g = grad.clone()
                    g[:, cols.to(g.device)] = 0.0
                    return g
                return hook

            h = module.basis_coeff.register_hook(_make_hook(col_tensor))
            hook_handles.append(h)
            specs.append((module.basis_coeff, col_tensor))

            frozen_params += len(valid_cols) * d_out
            modules_masked += 1

            if layer_idx < 2:
                logger.info(
                    f"  Layer {layer_idx:02d} {layer_type:10}: "
                    f"frozen safety cols={len(valid_cols)}/{d_in}, "
                    f"trainable cols={d_in - len(valid_cols)}/{d_in}"
                )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n  Modules with safety-col freezing : {modules_masked}")
    logger.info(f"  Frozen safety params (eff.)      : {frozen_params:,}")
    logger.info(f"  Trainable params                 : {trainable_params:,}")
    logger.info(f"  Total params                     : {total_params:,}")
    logger.info(
        f"  Safety frozen ratio              : "
        f"{frozen_params / max(total_params, 1) * 100:.4f}%"
    )
    logger.info(f"  Gradient hook handles registered : {len(hook_handles)}")

    return specs, hook_handles


# =============================================================================
# Dataset utilities
# =============================================================================

def is_instruct_model(model_ref: str) -> bool:
    return any(t in model_ref.lower() for t in ("instruct", "chat"))


def tokenize_sft_example(
    question: str,
    answer_text: str,
    tokenizer,
    max_length: int,
    model_ref: str,
    base_prompt_fn,
) -> Dict[str, List[int]]:
    """
    Tokenise one (question, answer) pair for SFT.

    Instruct/chat models: use chat template for both prompt and full sequence.
    Base models: use plain text prompt from base_prompt_fn.

    Loss is computed only on the answer tokens (prompt is masked to -100).
    """
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
                prompt_text, add_special_tokens=False,
                truncation=True, max_length=max_length,
            )["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False,
                truncation=True, max_length=max_length,
            )["input_ids"]

            labels = full_ids[:]
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            return {
                "input_ids":      full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels":         labels,
            }
        except Exception:
            pass  # fall through to base-model format

    # Base model format
    prompt_text = base_prompt_fn(question)
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False,
        truncation=True, max_length=max_length,
    )["input_ids"]
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text, add_special_tokens=False,
        truncation=True, max_length=remain,
    )["input_ids"]
    if (
        tokenizer.eos_token_id is not None
        and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id)
        and len(prompt_ids) + len(answer_ids) < max_length
    ):
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels    = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids, attn, labels = [], [], []
        for f in features:
            plen = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"]      + [pad_id] * plen)
            attn.append(     f["attention_mask"]  + [0]      * plen)
            labels.append(   f["labels"]          + [-100]   * plen)
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn,      dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
        }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    global logger
    logger, log_path = setup_logging(args.output_dir)

    layer_types = [lt.strip() for lt in args.layer_types.split(",")]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info("=" * 70)
    logger.info("WaRP-Space Downstream FT with Safety Neuron Column Freezing")
    logger.info("=" * 70)
    logger.info(f"  model         : {args.model_name_or_path}")
    logger.info(f"  basis_dir     : {args.basis_dir}")
    logger.info(f"  neuron_file   : {args.neuron_file}")
    logger.info(f"  task          : {args.task}")
    logger.info(f"  layer_types   : {layer_types}")
    logger.info(f"  lr            : {args.learning_rate}")
    logger.info(f"  epochs        : {args.num_epochs}")
    logger.info(f"  batch         : {args.batch_size} × accum {args.grad_accum} = eff {args.batch_size * args.grad_accum}")
    logger.info(f"  output_dir    : {args.output_dir}")
    logger.info(f"  log_file      : {log_path}")
    logger.info("=" * 70)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    logger.info("\n[1/6] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"  ✓ {type(tokenizer).__name__}, vocab={len(tokenizer)}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    logger.info("\n[2/6] Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    num_layers = model.config.num_hidden_layers
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ dtype={model.dtype}, num_hidden_layers={num_layers}")
    logger.info(f"  total params: {total_params:,}")

    # ── 3. Load basis + reparameterize ────────────────────────────────────────
    logger.info("\n[3/6] Loading basis and reparameterizing to WaRP space")
    basis_data = load_basis(args.basis_dir, layer_types, num_layers)
    model = reparameterize_model(model, basis_data, layer_types, num_layers)

    # ── 4. Load safety neurons + setup column freezing ────────────────────────
    logger.info("\n[4/6] Setting up safety column freezing")
    safety_neurons = load_warp_safety_neurons(args.neuron_file)
    freeze_specs, hook_handles = setup_safety_col_freezing(
        model, basis_data, safety_neurons, layer_types, num_layers
    )
    restore_cb = SafetyColRestoreCallback(freeze_specs)
    logger.info(f"  ✓ {len(freeze_specs)} modules with frozen safety columns")

    # ── 5. Dataset ────────────────────────────────────────────────────────────
    logger.info(f"\n[5/6] Loading dataset ({args.task})")
    task_cfg = _TASK_CONFIG[args.task]

    raw_ds = load_dataset(
        task_cfg["dataset_name"],
        task_cfg["subset"],
        split=task_cfg["split"],
        cache_dir=args.cache_dir,
    )
    if args.num_train_samples and args.num_train_samples < len(raw_ds):
        raw_ds = raw_ds.select(range(args.num_train_samples))
    logger.info(f"  ✓ {len(raw_ds)} training samples")

    def preprocess(ex):
        return tokenize_sft_example(
            ex[task_cfg["question_field"]],
            ex[task_cfg["answer_field"]],
            tokenizer,
            args.max_length,
            args.model_name_or_path,
            task_cfg["base_prompt_fn"],
        )

    tok_ds = raw_ds.map(
        preprocess,
        remove_columns=raw_ds.column_names,
        num_proc=4,
        desc=f"Tokenizing {args.task}",
    )
    logger.info(f"  ✓ Tokenization complete, {len(tok_ds)} examples")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    logger.info("\n[6/6] Training")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="no",         # manual save after WaRP restore
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        report_to=args.report_to,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
        callbacks=[restore_cb],
    )

    trainer.train()
    logger.info("  ✓ Training complete")

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("\nSaving model")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.output_dir,
        f"warp_sn_freeze_{args.task}_lr{args.learning_rate}_{ts}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # Remove gradient hooks (no longer needed after training)
    for h in hook_handles:
        h.remove()
    hook_handles.clear()
    logger.info("  ✓ Gradient hooks removed")

    # Move to CPU for safe serialization
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # Restore W = basis_coeff_trained @ U^T  (invert the rotation)
    restore_weight(model)
    logger.info("  ✓ Weights restored: W = basis_coeff @ U^T")

    # LinearWaRP → nn.Linear
    restore_to_linear(model)
    logger.info("  ✓ WaRP modules → nn.Linear")

    # Save standard HF checkpoint
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(save_dir)
    logger.info(f"  ✓ Model saved → {save_dir}")

    # Save training config
    config = {
        "method":            "WaRP-SN-Freeze Downstream FT",
        "base_model":        args.model_name_or_path,
        "basis_dir":         args.basis_dir,
        "neuron_file":       args.neuron_file,
        "task":              args.task,
        "layer_types":       layer_types,
        "learning_rate":     args.learning_rate,
        "num_epochs":        args.num_epochs,
        "batch_size":        args.batch_size,
        "grad_accum":        args.grad_accum,
        "num_train_samples": args.num_train_samples,
        "dtype":             args.dtype,
        "seed":              args.seed,
        "note": (
            "Safety neurons are COLUMN indices of basis_coeff (WaRP space). "
            "Restored model is in standard weight space (W = basis_coeff @ U^T)."
        ),
    }
    cfg_path = os.path.join(save_dir, "warp_sn_freeze_ft_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  ✓ Config saved → {cfg_path}")

    # Optional HuggingFace upload
    if args.upload_name:
        logger.info(f"\nUploading to HuggingFace: {args.upload_name}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            api.upload_folder(
                folder_path=save_dir,
                repo_id=args.upload_name,
                repo_type="model",
            )
            logger.info(f"  ✓ Uploaded: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"  Upload failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("WaRP-SN-Freeze Downstream Fine-tuning Complete")
    logger.info(f"  Output: {save_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
