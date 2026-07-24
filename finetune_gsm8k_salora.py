"""SaLoRA (ICLR'25) 단독 러너: GSM8K 위에서 SaLoRA 로 downstream fine-tuning.

원본 /home/gokms0509/SaLoRA (lora_train_act.py + process_lora.py + peft fork) 의 파이프라인을
이 레포 스타일로 self-contained 하게 재현한다. lora_C(safety 투영) forward 는 models/salora.py 의
LinearSaLoRA 로 구현되므로 표준 peft 를 건드리지 않는다.

파이프라인:
  1. safety 모델 로드 (기본 kmseong/llama2_7b-chat-Safety-FT-lr5e-5).
  2. target linear → LinearSaLoRA 교체.
  3. calibration:
       safety : circuit_breakers 의 prompt+refusal(llama3_output) activation → C = I − V_s V_sᵀ
       utility: gsm8k 의 question+answer activation → V_u + PiSSA(W) → A,B, residual W_res
       (labels != -100 인 response 토큰의 activation 만 사용 = 원본 disentangle 모드)
  4. A,B 만 학습(HF Trainer, GSM8K SFT), C·W_res frozen.
  5. dense fold(W = W_res + s·C·B·A) 후 merged_model 저장. (원본 process_lora.py 의 lora_B←C·B 병합에 해당)

원본 충실 재현: --lora_alpha == --lora_r (s=1), --target_modules q_proj,v_proj.

사용 예:
  CUDA_VISIBLE_DEVICES=0 python finetune_gsm8k_salora.py \
      --model_name kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --output_dir ./outputs/salora_gsm8k \
      --layer_type attn_q,attn_v --lora_r 16 --lora_alpha 16 \
      --salora_rank_safe 32 --salora_rank_util 32 --salora_calib_samples 128
"""
import argparse
import json
import logging
import os
import sys

# ⚠️ gsm8k_eval.finetune_gsm8k_full_params 는 import 시점에 CUDA_VISIBLE_DEVICES 를 하드코딩한다.
#    shell 로 지정한 device 가 덮이지 않도록 import 전에 캡처해 import 후 복원한다.
_INTENDED_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_eval.finetune_gsm8k_full_params import (  # noqa: E402
    tokenize_sft_example, DataCollatorForCausalLMWithPadding, _select_first_n)

if _INTENDED_CVD is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _INTENDED_CVD
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from models.salora import (  # noqa: E402
    switch_to_salora, init_salora_from_activations,
    mark_only_salora_trainable, restore_salora_to_linear)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("salora_runner")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="kmseong/llama2_7b-chat-Safety-FT-lr5e-5",
                    help="시작 = safety 모델. safety 부분공간 C 도 이 모델의 activation 으로 뽑는다.")
    ap.add_argument("--output_dir", required=True)
    # SaLoRA
    ap.add_argument("--safety_data_path", default="./data/circuit_breakers_train.json")
    ap.add_argument("--safety_response_field", default="llama3_output",
                    help="safety calib 응답 필드(안전 정렬 응답). circuit_breakers: llama3_output=refusal")
    ap.add_argument("--salora_rank_safe", type=int, default=32, help="safety 부분공간 차원 rs (C 용)")
    ap.add_argument("--salora_rank_util", type=int, default=32, help="utility 부분공간 차원 du (B 투영 용)")
    ap.add_argument("--salora_calib_samples", type=int, default=128, help="calibration 샘플 수 (safety/utility 각각)")
    ap.add_argument("--salora_calib_batch_size", type=int, default=4)
    ap.add_argument("--salora_niter", type=int, default=20, help="svd_lowrank 반복 수")
    # LoRA
    ap.add_argument("--target_modules", default="q_proj,v_proj",
                    help="원본 SaLoRA 기본은 q_proj,v_proj")
    ap.add_argument("--layer_type", default="attn_q,attn_v",
                    help="target_modules 와 대응(attn_q↔q_proj 등). LinearSaLoRA 교체 대상.")
    ap.add_argument("--target_layers", default="all")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16, help="원본 재현: alpha==r (scaling=1)")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    # GSM8K SFT
    ap.add_argument("--dataset_name", default="openai/gsm8k")
    ap.add_argument("--dataset_subset", default="main")
    ap.add_argument("--gsm8k_samples", type=int, default=0, help="0=전체")
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id", default=None)
    return ap.parse_args()


def _tokenize_pairs(pairs, tokenizer, max_length, model_ref):
    """[(question, answer), ...] → tokenize_sft_example 리스트 (labels 로 prompt 마스킹)."""
    return [tokenize_sft_example(q, a, tokenizer, max_length, model_ref) for q, a in pairs]


def _make_batches(features, collator, batch_size):
    return [collator(features[i:i + batch_size]) for i in range(0, len(features), batch_size)]


def build_gsm8k_pairs(args, n=0):
    ds = load_dataset(args.dataset_name, args.dataset_subset, split="train")
    if n > 0:
        ds = _select_first_n(ds, n)
    return [(ex["question"], ex["answer"]) for ex in ds]


def build_safety_pairs(args, n):
    with open(args.safety_data_path, "r") as f:
        data = json.load(f)
    n = min(n, len(data))
    return [(data[i]["prompt"], data[i][args.safety_response_field]) for i in range(n)]


def build_gsm8k_train(tokenizer, args):
    ds = load_dataset(args.dataset_name, args.dataset_subset, split="train")
    if args.gsm8k_samples > 0:
        ds = _select_first_n(ds, args.gsm8k_samples)

    def preprocess(ex):
        return tokenize_sft_example(ex["question"], ex["answer"], tokenizer, args.max_length, args.model_name)

    return ds.map(preprocess, remove_columns=ds.column_names, desc="tokenizing gsm8k")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"=== SaLoRA on GSM8K | model={args.model_name} r={args.lora_r} "
                f"alpha={args.lora_alpha} rs={args.salora_rank_safe} du={args.salora_rank_util} ===")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    layer_types = [x.strip() for x in args.layer_type.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map={"": 0})
    model.config.use_cache = False
    device = next(model.parameters()).device

    # ── 1) LinearSaLoRA 교체 ──
    converted = switch_to_salora(model, layer_types, args.target_layers,
                                 r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # ── 2) calibration 데이터 (labels != -100 = response 토큰만 부분공간 추정에 사용) ──
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tok)
    safety_feats = _tokenize_pairs(build_safety_pairs(args, args.salora_calib_samples),
                                   tok, args.max_length, args.model_name)
    util_feats = _tokenize_pairs(build_gsm8k_pairs(args, args.salora_calib_samples),
                                 tok, args.max_length, args.model_name)
    safety_batches = _make_batches(safety_feats, collator, args.salora_calib_batch_size)
    util_batches = _make_batches(util_feats, collator, args.salora_calib_batch_size)

    # ── 3) SaLoRA 초기화 (C, A, B, W_res) ──
    model.eval()
    init_salora_from_activations(
        model, converted, safety_batches, util_batches,
        r=args.lora_r, rank_safe=args.salora_rank_safe, rank_util=args.salora_rank_util,
        device=device, niter=args.salora_niter, logger=logger)

    trainable = mark_only_salora_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params (SaLoRA): {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ── 4) GSM8K SFT (A,B 만 학습) ──
    model.train()
    train_ds = build_gsm8k_train(tok, args)
    targs = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator)
    trainer.train()
    logger.info("✓ training done")

    # ── 5) dense fold 후 저장 ──
    restore_salora_to_linear(model)
    merged = model
    merged_dir = os.path.join(args.output_dir, "merged_model")
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(merged_dir)
    logger.info(f"✓ merged model saved: {merged_dir}")

    # sanity generation
    try:
        merged.eval()
        merged.config.use_cache = True
        q = "Natalia sold clips to 48 friends in April, and half as many in May. How many total?"
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        ii = tok(text, return_tensors="pt").to(next(merged.parameters()).device)
        with torch.no_grad():
            out = merged.generate(**ii, max_new_tokens=64, do_sample=False)
        logger.info("sanity gen: " + tok.decode(out[0][ii["input_ids"].shape[1]:], skip_special_tokens=True)[:200])
    except Exception as e:
        logger.warning(f"sanity gen failed: {e}")

    summary = {"method": "salora", "model_name": args.model_name, "lr": args.learning_rate,
               "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
               "rank_safe": args.salora_rank_safe, "rank_util": args.salora_rank_util,
               "calib_samples": args.salora_calib_samples, "target_modules": args.target_modules,
               "layer_type": args.layer_type, "epochs": args.epochs, "merged_dir": merged_dir,
               "hf_repo_id": args.hf_repo_id}
    json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--push_to_hub requires --hf_repo_id")
        try:
            logger.info(f"pushing to hub: {args.hf_repo_id}")
            merged.push_to_hub(args.hf_repo_id)
            tok.push_to_hub(args.hf_repo_id)
            logger.info(f"✓ pushed: https://huggingface.co/{args.hf_repo_id}")
        except Exception as e:
            logger.error(f"PUSH_FAILED repo={args.hf_repo_id} merged_dir={merged_dir} "
                         f"err={type(e).__name__}: {str(e)[:200]}")

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
