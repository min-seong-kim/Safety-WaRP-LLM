"""로컬 태스크 JSON 에 대한 full-parameter SFT (+ SafeInstr safety 데이터 혼합).

`agnews_eval/finetune_agnews_full_params.py` 는 AG News 전용 row 파싱에 묶여 있어
SST-2 에 쓸 수 없다. 이 스크립트는 그 모듈의 **토큰화 / 콜레이터 / 모델 로딩 /
safety 혼합 로직을 그대로 재사용**하면서, 데이터만 `data/local_task_dataset.py`
(= LoRA 계열 러너와 WaRP Phase 3 이 쓰는 것과 동일한 (question, response) 페어) 로 바꾼다.

즉 이 스크립트로 만든 baseline 은 WaRP Phase 3 모델과 **완전히 같은 프롬프트 문자열**,
같은 chat template, 같은 -100 마스킹 규칙을 쓴다. 비교 가능한 baseline 이 되는 이유다.

  baseline :  --safety_mix_ratio 0
  SafeInstr:  --safety_mix_ratio 0.1   (downstream 샘플 수의 10% 만큼 circuit_breakers 추가)

사용:
  python finetune_task_full_params.py \
      --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --task_data_path data/sst2_train_8k_seed42.json \
      --output_dir outputs/cls_baselines/sst2_ft_lr1e-5_ep1 \
      --learning_rate 1e-5 --epochs 1 --batch_size 2 --grad_accum 8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from transformers import Trainer, TrainingArguments, set_seed

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "agnews_eval"))

# AG News 스크립트의 공용 헬퍼 재사용 (토큰화 규칙이 갈라지지 않게 하기 위함).
# ⚠️ 이 모듈은 과거 import 시점에 CUDA_VISIBLE_DEVICES 를 박아뒀었다. 지금은 주석 처리돼 있고,
#    되살아나면 여기서 GPU 지정이 통째로 무시되므로 아래 assert 로 감시한다.
_CVD_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
import finetune_agnews_full_params as agnews_ft  # noqa: E402

if os.environ.get("CUDA_VISIBLE_DEVICES") != _CVD_BEFORE:
    raise RuntimeError(
        "agnews_eval/finetune_agnews_full_params.py 가 import 시점에 CUDA_VISIBLE_DEVICES 를 "
        f"변경했습니다 ({_CVD_BEFORE!r} -> {os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
        "해당 줄을 다시 주석 처리하세요."
    )

from data.local_task_dataset import infer_task_name, load_task_pairs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter SFT on a local task JSON (+SafeInstr).")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--task_data_path", type=str, required=True,
                   help='[{"question":..., "response":...}] 형식의 로컬 JSON')
    p.add_argument("--task_name", type=str, default=None, help="미지정 시 파일명에서 추정")
    p.add_argument("--task_samples", type=int, default=0, help="0 = 전체")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    # WaRP Phase 3 과 동일한 최적화 설정이 기본값이다 (weight_decay 0.0, warmup 0.1, cosine).
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--cache_dir", type=str, default=None)

    # SafeInstr
    p.add_argument("--safety_data_path", type=str,
                   default=str(REPO_DIR / "data" / "circuit_breakers_train.json"))
    p.add_argument("--safety_mix_ratio", type=float, default=0.0)

    # load_model_and_tokenizer 가 참조하는 LoRA 인자들 (여기서는 full-param 이 기본).
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"])
    return p.parse_args()


def build_task_dataset(args, tokenizer, model_path, logger):
    pairs = load_task_pairs(args.task_data_path, args.task_samples)
    tokenized, skipped = [], 0
    for question, response in pairs:
        try:
            tokenized.append(
                agnews_ft.tokenize_prompt_response(
                    question, response, tokenizer, args.max_length, model_path
                )
            )
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                logger.warning(f"Skipping malformed row: {exc}")
    if not tokenized:
        raise ValueError(f"No valid tokenized examples from {args.task_data_path}")
    if skipped:
        logger.warning(f"Skipped {skipped}/{len(pairs)} rows during tokenization")
    return HFDataset.from_dict({
        "input_ids": [x["input_ids"] for x in tokenized],
        "attention_mask": [x["attention_mask"] for x in tokenized],
        "labels": [x["labels"] for x in tokenized],
    })


def main():
    args = parse_args()
    task = args.task_name or infer_task_name(args.task_data_path)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path = args.model_path
    is_local = raw_path.startswith(("./", "/", "../"))
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger, log_file = agnews_ft.setup_logging(args.output_dir)
    mode = "SafeInstr" if args.safety_mix_ratio > 0 else "baseline"
    logger.info("=" * 70)
    logger.info(f"Full-parameter SFT: task={task}  mode={mode}")
    logger.info("=" * 70)
    logger.info(f"Log file  : {log_file}")
    logger.info(f"Model     : {model_path}")
    logger.info(f"Task data : {args.task_data_path}")
    logger.info(f"GPU       : CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Formatting: "
                f"{'chat template' if agnews_ft.is_instruct_model(model_path) else 'base plain prompt'}")

    model, tokenizer = agnews_ft.load_model_and_tokenizer(args, model_path, logger)

    # chat template 없이 학습되면 평가 포맷과 어긋난다. instruct 모델이면 강제 확인.
    if agnews_ft.is_instruct_model(model_path) and not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"{model_path} 토크나이저에 chat_template 이 없습니다. "
            "이대로 학습하면 plain 프롬프트로 fallback 되어 WaRP/평가와 포맷이 어긋납니다."
        )

    train_tok = build_task_dataset(args, tokenizer, model_path, logger)
    num_task = len(train_tok)
    logger.info(f"Task samples tokenized: {num_task}")

    train_tok = agnews_ft.maybe_mix_safety(
        train_tok, args, tokenizer, model_path, logger, num_agnews=num_task
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        data_collator=agnews_ft.DataCollatorForCausalLMWithPadding(tokenizer),
        tokenizer=tokenizer,
    )

    logger.info(f"Starting training ({len(train_tok)} samples, "
                f"effective batch {args.batch_size * args.grad_accum})")
    result = trainer.train()

    logger.info("Saving model")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 저장된 토크나이저에 chat template 이 실제로 들어갔는지 확인 (업로드 누락 사고 방지).
    saved_template = (Path(args.output_dir) / "chat_template.jinja").exists()
    if not saved_template:
        tc = Path(args.output_dir) / "tokenizer_config.json"
        if tc.exists():
            saved_template = "chat_template" in json.loads(tc.read_text())
    logger.info(f"chat_template saved: {saved_template}")

    summary = {
        "base_model": model_path,
        "method": "safeinstr" if args.safety_mix_ratio > 0 else "full_ft",
        "downstream": task,
        "task_data_path": args.task_data_path,
        "fine_tuning_type": "Full Parameter Fine-tuning",
        "num_task_samples": num_task,
        "num_train_samples": len(train_tok),
        "safety_mix_ratio": args.safety_mix_ratio,
        "safety_data_path": args.safety_data_path if args.safety_mix_ratio > 0 else None,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "default"),
        "train_runtime_sec": getattr(result, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(result, "metrics", {}).get("train_loss"),
        "chat_template_saved": bool(saved_template),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary.json written: loss={summary['train_loss']} "
                f"runtime={summary['train_runtime_sec']}s")
    logger.info("Done")


if __name__ == "__main__":
    main()
