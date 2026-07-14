"""
Stage 1 — SEAL bilevel data selector 학습 (HF/PyTorch 이식).

SEAL의 SFTSelectorTrainer.fit()을 DeepSpeed 없이 그대로 옮긴 커스텀 학습 루프.
- upper level (safe)      : circuit_breakers 손실  → ul_weight
- lower level (downstream): selector 가중 gsm8k 손실 → (1 - ul_weight)
- selector 갱신           : selector_loss = (σ(ω)[ide] * ft_loss.detach()).mean()
- ul_weight는 매 epoch 이후 upperlevel_weight_decay만큼 감쇠

한 step에서:
  model_loss = ul * safe_loss + (1-ul) * mean(σ(ω)[ide].detach() * ft_loss_vec)
  → 모델 갱신
  selector_loss = mean(σ(ω)[ide] * ft_loss_vec.detach())
  → selector 갱신

출력: <out_dir>/<selector_name>_<activation>.pt          (최종 selector logits)
      <out_dir>/<selector_name>_<activation>_ep{e}.pt     (epoch 스냅샷)
      <out_dir>/<selector_name>_meta.json                 (메타데이터)

사용 예:
  CUDA_VISIBLE_DEVICES=0,1 python -m seal.train_selector \
      --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --safety_data_path data/circuit_breakers_train.json \
      --lora --epochs 2
"""

import argparse
import itertools
import json
import math
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer import get_scheduler

from seal.data_utils import (
    DataCollatorForCausalLMWithPadding,
    build_circuit_breakers_dataset,
    build_gsm8k_dataset,
    is_instruct_model,
)
from seal.selector import TrainableSelector, per_sample_lm_loss

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _peft_available = True
except ImportError:
    _peft_available = False


def parse_args():
    p = argparse.ArgumentParser(description="SEAL bilevel data selector 학습 (Stage 1)")
    # model
    p.add_argument("--model_path", type=str, required=True,
                   help="안전정렬(safety-aligned) 초기 모델 (HF id 또는 로컬 경로)")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    # LoRA (SEAL은 Llama selector 학습에 LoRA를 사용 — 메모리 절약)
    p.add_argument("--lora", action="store_true",
                   help="selector 학습에 LoRA 사용 (권장, 메모리 절약)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    # data
    p.add_argument("--safety_data_path", type=str,
                   default="data/circuit_breakers_train.json",
                   help="upper-level safe 데이터 (circuit_breakers 형식)")
    p.add_argument("--safety_prompt_key", type=str, default="prompt")
    p.add_argument("--safety_response_key", type=str, default="llama3_output")
    p.add_argument("--num_safety_samples", type=int, default=0,
                   help="0=전체 사용")
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--num_ft_samples", type=int, default=0,
                   help="downstream(gsm8k) 샘플 수. 0=전체. selector 길이가 이 값이 된다.")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--cache_dir", type=str, default="./cache")

    # bilevel 학습
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5,
                   help="lower-level 모델 lr")
    p.add_argument("--selector_learning_rate", type=float, default=1e-2)
    p.add_argument("--selector_activation", type=str, default="softmax",
                   choices=["softmax", "sigmoid"])
    p.add_argument("--upperlevel_weight", type=float, default=0.9)
    p.add_argument("--upperlevel_weight_decay", type=float, default=0.1)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--selector_lr_scheduler", type=str, default="constant")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # output
    p.add_argument("--out_dir", type=str, default="./seal/ckpt")
    p.add_argument("--selector_name", type=str, default="gsm8k_selector")
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[selector] loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

    if args.lora:
        if not _peft_available:
            raise ImportError("peft가 필요합니다: pip install peft")
        model.enable_input_require_grads()
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # 데이터셋: safe(upper) = circuit_breakers, downstream(lower, indexed) = gsm8k
    print("[selector] building datasets ...")
    safe_ds = build_circuit_breakers_dataset(
        args.safety_data_path, tokenizer, args.max_length, args.model_path,
        num_samples=args.num_safety_samples, seed=args.seed,
        prompt_key=args.safety_prompt_key, response_key=args.safety_response_key,
        with_index=False,
    )
    ft_ds = build_gsm8k_dataset(
        tokenizer, args.max_length, args.model_path,
        dataset_name=args.dataset_name, dataset_subset=args.dataset_subset,
        split=args.train_split, num_samples=args.num_ft_samples,
        subset_indices=None, with_index=True, cache_dir=args.cache_dir,
    )
    selector_size = len(ft_ds)
    print(f"[selector] safe={len(safe_ds)}  downstream(gsm8k)={selector_size}")

    collator = DataCollatorForCausalLMWithPadding(tokenizer)
    safe_loader = DataLoader(safe_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collator, drop_last=True)
    ft_loader = DataLoader(ft_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collator, drop_last=False)

    # selector (cuda:0에 배치)
    sel_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selector = TrainableSelector(selector_size, args.selector_activation).to(sel_device)

    # optimizers / schedulers
    trainable = [p for p in model.parameters() if p.requires_grad]
    model_opt = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    sel_opt = torch.optim.AdamW(selector.parameters(), lr=args.selector_learning_rate,
                                betas=(0.9, 0.95), weight_decay=0.0)

    steps_per_epoch = len(ft_loader)
    max_steps = args.epochs * steps_per_epoch
    warmup = math.ceil(max_steps * args.warmup_ratio)
    model_sched = get_scheduler(args.lr_scheduler, model_opt,
                                num_warmup_steps=warmup, num_training_steps=max_steps)
    sel_sched = get_scheduler(args.selector_lr_scheduler, sel_opt,
                              num_warmup_steps=warmup, num_training_steps=max_steps)

    ignore_index = -100
    ul_weight = args.upperlevel_weight

    model.train()
    selector.train()
    global_step = 0
    for epoch in range(args.epochs):
        if epoch >= 1:
            ul_weight -= args.upperlevel_weight_decay
        print(f"\n[selector] epoch {epoch}  ul_weight={ul_weight:.3f}")

        # safe 로더가 더 짧을 수 있으므로 cycle → 모든 gsm8k 샘플이 매 epoch 갱신되도록
        safe_iter = itertools.cycle(safe_loader)
        loss_ema = None

        for ft_batch in ft_loader:
            safe_batch = next(safe_iter)

            # ── forward: safe (upper) ──
            s_in = safe_batch["input_ids"].to(model.device)
            s_am = safe_batch["attention_mask"].to(model.device)
            s_lb = safe_batch["labels"]
            s_logits = model(input_ids=s_in, attention_mask=s_am).logits
            s_lb = s_lb.to(s_logits.device)
            safe_loss = per_sample_lm_loss(s_logits, s_lb, ignore_index).mean()

            # ── forward: downstream (lower) ──
            f_in = ft_batch["input_ids"].to(model.device)
            f_am = ft_batch["attention_mask"].to(model.device)
            f_lb = ft_batch["labels"]
            ide = ft_batch["ide"].to(sel_device)
            f_logits = model(input_ids=f_in, attention_mask=f_am).logits
            f_lb = f_lb.to(f_logits.device)
            ft_loss_vec = per_sample_lm_loss(f_logits, f_lb, ignore_index)  # (B,) on logits device

            # ── 모델(lower-level) 갱신 ──
            with torch.no_grad():
                w = selector()[ide].detach().to(ft_loss_vec.device)
            weighted_ft = (w * ft_loss_vec).mean()
            model_loss = ul_weight * safe_loss + (1.0 - ul_weight) * weighted_ft

            model_opt.zero_grad(set_to_none=True)
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            model_opt.step()
            model_sched.step()

            # ── selector(upper-level) 갱신 ──
            ft_loss_det = ft_loss_vec.detach().to(sel_device)
            selector_loss = (selector()[ide] * ft_loss_det).mean()
            sel_opt.zero_grad(set_to_none=True)
            selector_loss.backward()
            sel_opt.step()
            sel_sched.step()

            loss_ema = model_loss.item() if loss_ema is None \
                else 0.95 * loss_ema + 0.05 * model_loss.item()
            global_step += 1
            if global_step % args.logging_steps == 0:
                print(f"  step {global_step}/{max_steps}  "
                      f"safe={safe_loss.item():.4f}  ft={ft_loss_vec.mean().item():.4f}  "
                      f"model_loss_ema={loss_ema:.4f}  sel_loss={selector_loss.item():.4f}")

        # epoch 스냅샷
        snap = os.path.join(
            args.out_dir, f"{args.selector_name}_{args.selector_activation}_ep{epoch+1}.pt"
        )
        torch.save(selector.logits.detach().cpu(), snap)
        print(f"[selector] saved epoch snapshot: {snap}")

    # 최종 selector logits 저장 (raw tensor — SEAL 규약과 동일)
    final_path = os.path.join(
        args.out_dir, f"{args.selector_name}_{args.selector_activation}.pt"
    )
    torch.save(selector.logits.detach().cpu(), final_path)

    meta = {
        "model_path": args.model_path,
        "safety_data_path": args.safety_data_path,
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "train_split": args.train_split,
        "num_ft_samples": args.num_ft_samples,
        "selector_size": selector_size,
        "selector_activation": args.selector_activation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "selector_learning_rate": args.selector_learning_rate,
        "upperlevel_weight": args.upperlevel_weight,
        "upperlevel_weight_decay": args.upperlevel_weight_decay,
        "lora": args.lora,
        "is_instruct": is_instruct_model(args.model_path),
        "selector_path": final_path,
        "created": datetime.now().isoformat(),
    }
    meta_path = os.path.join(args.out_dir, f"{args.selector_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[selector] ✅ final selector saved: {final_path}")
    print(f"[selector] ✅ metadata saved:       {meta_path}")


if __name__ == "__main__":
    main()
