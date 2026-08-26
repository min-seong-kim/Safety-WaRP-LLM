#!/usr/bin/env python3
"""SaLoRA (Li et al., ICLR 2025) — full version (init_mode=task) baseline for the
WSR-LoRA comparison. Same shared hyper-parameters as WSR-LoRA/vanilla LoRA
(lora_r=16, q/k/v/up/down, lr 5e-5, alpha=16 -> scaling 1, seed 42) plus SaLoRA's
own subspace ranks r_s=r_t=32. Trained on GSM8K from an SSFT checkpoint; the
harmful prompt+refusal set (bv/cb) builds the fixed safety module C_S. Saves a
merged full HF model (5GB shards)."""
import argparse
import json
import os
import types
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, TaskType, get_peft_model

import pissa_wsr_lora as pw
from salora_impl import SaLoRA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="SSFT checkpoint (frozen base)")
    ap.add_argument("--safety_data", required=True, help="harmful prompt+refusal json (C_S)")
    ap.add_argument("--gsm8k_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--rank", type=int, default=16)          # lora_r (matched)
    ap.add_argument("--lora_alpha", type=int, default=16)    # scaling 1.0 (matched)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--r_s", type=int, default=32)           # SaLoRA safety rank
    ap.add_argument("--r_t", type=int, default=32)           # SaLoRA task rank
    ap.add_argument("--init_mode", default="task")           # full SaLoRA
    ap.add_argument("--n_harmful", type=int, default=256)
    ap.add_argument("--n_task", type=int, default=256)
    ap.add_argument("--salora_max_tokens", type=int, default=4096)
    ap.add_argument("--response_field", default="llama3_output")
    ap.add_argument("--init_cache", default=None,
                    help="cache/reuse the LR-independent SaLoRA initialization")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--train_samples", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda")
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.target_modules.split(",")],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lcfg)

    sa = types.SimpleNamespace(
        seed=args.seed, model_name=args.model_name, max_length=args.max_length,
        lora_r=args.rank, lora_alpha=args.lora_alpha, _tokenizer=tok,
        salora_r_s=args.r_s, salora_r_t=args.r_t, salora_init_mode=args.init_mode,
        salora_n_harmful=args.n_harmful, salora_n_task=args.n_task,
        salora_max_tokens=args.salora_max_tokens, salora_svd_niter=10,
        salora_harmful_path=args.safety_data, salora_response_field=args.response_field,
        salora_capture_batch_size=4,
    )
    salora = SaLoRA(sa)
    print(f"[salora] {salora.describe()}", flush=True)

    # task subspace (init_mode=task) is built from downstream (GSM8K) features
    gsm = json.load(open(args.gsm8k_json))
    task_records = [
        {"prompt": r["question"], "output": r.get("answer", r.get("response"))}
        for r in gsm
    ]
    if any(record["output"] is None for record in task_records):
        raise KeyError("downstream JSON rows require answer or response")

    init_context = {
        "model_name": args.model_name,
        "safety_data": str(Path(args.safety_data).resolve()),
        "downstream_json": str(Path(args.gsm8k_json).resolve()),
        "target_modules": sorted(t.strip() for t in args.target_modules.split(",")),
        "rank": args.rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "r_s": args.r_s,
        "r_t": args.r_t,
        "init_mode": args.init_mode,
        "n_harmful": args.n_harmful,
        "n_task": args.n_task,
        "max_tokens": args.salora_max_tokens,
        "max_length": args.max_length,
        "response_field": args.response_field,
        "seed": args.seed,
    }
    init_cache = Path(args.init_cache).resolve() if args.init_cache else None
    if init_cache is not None and init_cache.is_file():
        print(f"[salora] loading shared initialization cache: {init_cache}", flush=True)
        salora.load_initialization(model, init_cache, init_context)
    else:
        print("[salora] building C_S / task-init / W' / projection hooks...", flush=True)
        salora.build(model, tok, task_records)
        if init_cache is not None:
            salora._reference_model = model
            salora.save_initialization(init_cache, init_context)
            del salora._reference_model
            print(f"[salora] saved shared initialization cache: {init_cache}", flush=True)
    model.print_trainable_parameters()

    train_ds = pw.GSM8KDataset(tok, args.max_length, args.train_samples, args.gsm8k_json)
    coll = lambda b: pw.collate(b, tok.pad_token_id)
    print(f"[salora] gsm8k rows: {len(train_ds)}", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        weight_decay=0.0, logging_steps=10, save_strategy="no",
        bf16=(args.dtype == "bfloat16"), seed=args.seed, report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=coll, processing_class=tok)
    trainer.train()

    # bake C_S into lora_B, drop hooks, merge W' + s*C_S*B*A into a full model
    print("[salora] finalize_merge + merge_and_unload...", flush=True)
    salora.finalize_merge(model)
    merged = model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(args.output_dir)
    run_metadata = {
        "method": "SaLoRA",
        "model_name": args.model_name,
        "safety_data": str(Path(args.safety_data).resolve()),
        "downstream_json": str(Path(args.gsm8k_json).resolve()),
        "target_modules": [
            target.strip() for target in args.target_modules.split(",")
        ],
        "rank": args.rank,
        "alpha": args.lora_alpha,
        "scaling": args.lora_alpha / args.rank,
        "dropout": args.lora_dropout,
        "safety_rank": args.r_s,
        "task_rank": args.r_t,
        "init_mode": args.init_mode,
        "harmful_samples": args.n_harmful,
        "task_samples": args.n_task,
        "capture_max_tokens": args.salora_max_tokens,
        "response_field": args.response_field,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "train_samples": args.train_samples,
        "dtype": args.dtype,
        "seed": args.seed,
        "initialization_cache": str(init_cache) if init_cache is not None else None,
    }
    (Path(args.output_dir) / "salora_run_config.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"[salora] saved merged full model (5GB shards) -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
