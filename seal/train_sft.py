"""
Stage 2 — 선택된 downstream(gsm8k) 데이터로 SFT.

  --use_warp 없이 : baseline (전체 파라미터 학습, 표준 full-param SFT)
  --use_warp 있이 : WaRP 재매개변수화 공간 학습 (basis_coeff만 학습, 안전 방향 동결)

baseline과 WaRP는 데이터/토큰화/하이퍼파라미터를 완전히 공유하고, WaRP 여부만 다르다
→ SEAL-only vs SEAL+WaRP의 공정 비교.

WaRP 경로는 Phase 1(basis)·Phase 2(mask) 산출물이 필요하다:
  python train.py --phase 1 ... (basis)   →  --basis_dir
  python train.py --phase 2 ... (mask)     →  --masks_dir

사용 예:
  # baseline
  CUDA_VISIBLE_DEVICES=0,1 python -m seal.train_sft \
      --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --selected_indices seal/ckpt/gsm8k_selected_top80.json \
      --output_dir seal/out/baseline_top80

  # WaRP
  CUDA_VISIBLE_DEVICES=0,1 python -m seal.train_sft \
      --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --selected_indices seal/ckpt/gsm8k_selected_top80.json \
      --use_warp --basis_dir <phase1>/basis --masks_dir <phase2>/masks \
      --layer_type attn_q,attn_k,attn_v,ffn_up,ffn_down --target_layers all \
      --output_dir seal/out/warp_top80
"""

import argparse
import json
import os
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from seal.data_utils import (
    DataCollatorForCausalLMWithPadding,
    build_gsm8k_dataset,
    is_instruct_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="선택 데이터 SFT (baseline / WaRP)")
    # model
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # data
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--num_train_samples", type=int, default=0,
                   help="0=전체. selected_indices 이전에 적용되는 상한(보통 0).")
    p.add_argument("--selected_indices", type=str, default=None,
                   help="select_data.py가 저장한 json (없으면 전체 데이터로 SFT)")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--num_workers", type=int, default=4)

    # training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)

    # WaRP
    p.add_argument("--use_warp", action="store_true",
                   help="WaRP 재매개변수화 공간에서 학습 (basis_coeff만)")
    p.add_argument("--basis_dir", type=str, default=None,
                   help="Phase 1 basis 디렉토리 (use_warp 시 필수)")
    p.add_argument("--masks_dir", type=str, default=None,
                   help="Phase 2 masks 디렉토리 (use_warp 시 필수, --no_masks면 생략 가능)")
    p.add_argument("--layer_type", type=str,
                   default="attn_q,attn_k,attn_v,ffn_up,ffn_down",
                   help="Phase 1/2와 동일해야 함")
    p.add_argument("--target_layers", type=str, default="all",
                   help="Phase 1/2와 동일해야 함")
    p.add_argument("--no_masks", action="store_true",
                   help="mask 없이 basis만 사용 (freeze 없음)")

    # output
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    mode = "WaRP" if args.use_warp else "baseline"
    print(f"\n{'='*70}\n  Stage 2 SFT — {mode}\n{'='*70}")

    if args.use_warp and not args.basis_dir:
        raise ValueError("--use_warp 시 --basis_dir 필수")
    if args.use_warp and not args.no_masks and not args.masks_dir:
        raise ValueError("--use_warp 시 --masks_dir 필수 (또는 --no_masks)")

    # tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
    )

    # WaRP 세팅 (대상 레이어를 LinearWaRP로 변환 + basis/mask 초기화)
    warp_stats = None
    if args.use_warp:
        from seal.warp_setup import apply_warp
        model, warp_stats = apply_warp(
            model, args.basis_dir, args.masks_dir,
            args.layer_type, args.target_layers, no_masks=args.no_masks,
        )
        if args.gradient_checkpointing:
            # basis_coeff만 학습하므로 embedding 출력이 grad를 요구하도록 강제
            model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

    # 선택 인덱스
    subset_indices = None
    select_meta = None
    if args.selected_indices:
        with open(args.selected_indices) as f:
            select_meta = json.load(f)
        subset_indices = select_meta["indices"]
        print(f"[sft] selected {len(subset_indices)}/{select_meta.get('total','?')} "
              f"(topp={select_meta.get('topp')})")

    # 데이터셋
    train_ds = build_gsm8k_dataset(
        tokenizer, args.max_length, args.model_path,
        dataset_name=args.dataset_name, dataset_subset=args.dataset_subset,
        split=args.train_split, num_samples=args.num_train_samples,
        subset_indices=subset_indices, with_index=False,
        cache_dir=args.cache_dir, num_proc=args.num_workers,
    )
    print(f"[sft] train samples: {len(train_ds)}")

    # WaRP는 frozen coeff의 weight decay를 방지하기 위해 wd=0 강제 (Phase 3 규약)
    weight_decay = 0.0 if args.use_warp else args.weight_decay
    if args.use_warp and args.weight_decay != 0.0:
        print(f"[sft] WaRP: weight_decay {args.weight_decay} → 0.0 강제 "
              "(frozen basis_coeff 보호)")

    collator = DataCollatorForCausalLMWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    # transformers 5.x: Trainer의 `tokenizer` 인자 제거됨 → `processing_class` 사용.
    # 구버전 호환을 위해 fallback.
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=collator,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            data_collator=collator,
        )

    print("[sft] training ...")
    trainer.train()

    # WaRP: basis_coeff → weight 복원 후 표준 nn.Linear로 되돌리기 (저장 전 필수)
    if args.use_warp:
        from seal.warp_setup import restore_and_delinearize
        model.config.use_cache = True
        restore_and_delinearize(model)

    print("[sft] saving ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "mode": mode,
        "base_model": args.model_path,
        "dataset": args.dataset_name,
        "num_train_samples": len(train_ds),
        "selected_indices": args.selected_indices,
        "select_meta": select_meta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": weight_decay,
        "max_length": args.max_length,
        "use_warp": args.use_warp,
        "warp": {
            "basis_dir": args.basis_dir,
            "masks_dir": args.masks_dir,
            "layer_type": args.layer_type,
            "target_layers": args.target_layers,
            "no_masks": args.no_masks,
            "stats": warp_stats,
        } if args.use_warp else None,
        "is_instruct": is_instruct_model(args.model_path),
        "created": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "sft_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[sft] ✅ saved {mode} model → {args.output_dir}")


if __name__ == "__main__":
    main()
