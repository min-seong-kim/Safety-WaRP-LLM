"""SafeGrad 파인튜닝 러너 (gradient surgery + KL alignment).

논문: "SafeGrad: Gradient Surgery for Safe LLM Fine-tuning" (arXiv:2508.07172)

이 저장소의 LoRA 계열 baseline(LISA / AsFT / SafeLoRA / SaLoRA / WSR-LoRA)과 **같은 조건**
에서 비교할 수 있게 만든 러너다. 구조·인자·저장 규약은 `gsm8k_eval/finetune_gsm8k_lisa.py`
를 그대로 따르고, 토큰화 함수는 **그 파일에서 직접 import** 한다. 프롬프트 문자열이 arm 마다
갈라지면 비교 자체가 무의미해지기 때문이다
(scripts/revision/verify_prompt_parity.py 가 검사하는 불변식).

원 논문과의 차이 (의도적)
------------------------
* 원 논문/참조 구현은 **유저 데이터에 harmful 예시를 섞는(poison ratio hr)** 공격 시나리오다.
  이 저장소의 비교군은 전부 "안전정렬된 출발 모델을 clean 다운스트림 데이터로 FT 했을 때
  안전성이 얼마나 무너지는가" 를 본다. LISA 도 같은 방식으로 이식돼 있으므로 SafeGrad 도
  동일하게 **clean 태스크 데이터**로 돌린다. (poison 실험이 필요하면 --task_data_path 에
  섞은 JSON 을 주면 된다 — 러너는 데이터 내용을 가정하지 않는다.)
* alignment 데이터는 논문의 BeaverTails refusal 대신 이 저장소의 안전 데이터
  (`data/circuit_breakers_train.json` 의 prompt / llama3_output, 또는 beavertails 변형)를 쓴다.
  `$safety` 축이 출발 모델과 alignment 데이터를 동시에 결정한다는 저장소 불변식을 따른다.
* LoRA 예산은 논문(r=8, α=16, {q,k,v}) 이 아니라 이 저장소 공통값(r=16, α=32,
  {q,k,v,up,down})을 기본으로 한다. 기법 간 차이가 "안전 메커니즘" 하나로만 남게 하려는 것.
  논문 설정을 그대로 보고 싶으면 --lora_r 8 --lora_alpha 16 --lora_target_modules q_proj k_proj v_proj.
* reference 모델 θ0 = 출발(안전정렬) 모델. 참조 구현과 동일하다.

Example
-------
python safegrad/finetune_safegrad.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./safegrad_gsm8k_llama2_7b \
    --task_data_path data/gsm8k_train_task_7473.json \
    --safety_data_path data/circuit_breakers_train.json --guide_data_num 4994 \
    --rho 1.0 --learning_rate 3e-4 --epochs 3 \
    --lora --lora_r 16 --lora_alpha 32 \
    --batch_size 4 --grad_accum 4
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import torch

# 이 파일은 <repo>/safegrad/ 에 있으므로 repo 루트는 한 단계 위.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_SAFETY_DATA = os.path.join(_REPO_ROOT, "data", "circuit_breakers_train.json")

# ⚠️ 토큰화는 LISA arm 과 **같은 함수**를 써야 한다. 복사하면 언젠가 갈라진다.
#    gsm8k_eval/finetune_gsm8k_lisa.py 는 import 시점에 CUDA_VISIBLE_DEVICES 를
#    setdefault 한다(값이 이미 있으면 건드리지 않는다). 혹시라도 '덮어쓰기' 로 바뀌면
#    이 러너의 GPU 지정이 통째로 무시되므로 아래에서 감시한다.
_CVD_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
from gsm8k_eval.finetune_gsm8k_lisa import (  # noqa: E402
    DataCollatorForCausalLMWithPadding,
    is_instruct_model,
    tokenize_sft_example,
    _select_first_n,
)

if _CVD_BEFORE is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != _CVD_BEFORE:
    raise RuntimeError(
        "gsm8k_eval/finetune_gsm8k_lisa.py 가 import 시점에 CUDA_VISIBLE_DEVICES 를 "
        f"변경했습니다 ({_CVD_BEFORE!r} -> {os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
        "setdefault 로 되돌리세요.")

from datasets import Dataset as HFDataset, load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from safegrad.safegrad_trainer import SafeGradTrainer  # noqa: E402

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _peft_available = True
except ImportError:
    _peft_available = False


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SafeGrad (gradient surgery + KL alignment) fine-tuning")

    # model
    p.add_argument("--model_path", type=str, required=True,
                   help="출발 모델 = 안전정렬된 모델. reference θ0 도 같은 모델이다.")
    p.add_argument("--ref_model_path", type=str, default=None,
                   help="reference θ0 를 다른 체크포인트로 쓰고 싶을 때만. 기본은 --model_path.")

    # downstream data
    p.add_argument("--task_data_path", type=str, default=None,
                   help='{"question","response"} 스키마의 로컬 태스크 JSON. 없으면 GSM8K 를 받는다.')
    p.add_argument("--task_samples", type=int, default=0, help="0=전체")
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--num_train_samples", type=int, default=7473)
    p.add_argument("--seed", type=int, default=42)

    # safety / alignment data
    p.add_argument("--safety_data_path", type=str, default=_DEFAULT_SAFETY_DATA,
                   help="alignment 데이터 JSON (prompt / llama3_output 필드)")
    p.add_argument("--safety_response_field", type=str, default="llama3_output",
                   help="안전 응답이 들어 있는 필드명")
    p.add_argument("--guide_data_num", type=int, default=4994,
                   help="alignment 에 쓸 안전 예시 개수. 논문 기본값은 100 이지만 이 저장소의 "
                        "다른 arm(LISA/SaLoRA/WSR-LoRA)과 맞추려고 4994(전체)를 기본으로 둔다.")

    # SafeGrad
    p.add_argument("--rho", type=float, default=1.0,
                   help="Eq.5 의 ρ — 최종 그래디언트에서 alignment 항의 가중치 (논문 기본 1.0)")
    p.add_argument("--no_projection", dest="projection", action="store_false", default=True,
                   help="gradient surgery 를 끄고 단순 가중합만 쓴다 (논문 ablation)")
    p.add_argument("--ref_mode", type=str, default="separate", choices=["separate", "adapter_off"],
                   help="'separate'=참조 구현과 동일하게 θ0 를 따로 로드. "
                        "'adapter_off'=LoRA 어댑터를 끈 현재 모델을 θ0 로 쓴다(수학적으로 동일, 메모리 절약).")
    p.add_argument("--kl_reduction", type=str, default="ref", choices=["ref", "response"],
                   help="'ref'=참조 구현 그대로(패딩 포함 sum ÷ 응답토큰수), 'response'=응답 토큰만 평균")
    p.add_argument("--kl_fp32", action="store_true", default=False,
                   help="KL 의 softmax/log_softmax 를 float32 로 계산 (메모리 2배, 참조 구현은 bf16)")
    p.add_argument("--align_batch_size", type=int, default=0,
                   help="alignment 배치 크기. 0 이면 태스크 배치와 동일(참조 구현과 같음). "
                        "vocab 이 큰 모델(gemma-2 256k)에서 KL logits 메모리가 부담이면 낮춰라.")

    # training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # LoRA
    p.add_argument("--lora", action="store_true", default=True,
                   help="LoRA 로 학습 (기본). full-param 은 --no_lora — 메모리 3배를 각오할 것.")
    p.add_argument("--no_lora", dest="lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"])

    # io
    p.add_argument("--output_dir", type=str, default="./safegrad_out")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--upload_name", type=str, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    return p.parse_args()


def setup_logging(name="safegrad"):
    log_dir = os.path.join(_REPO_ROOT, "logs", "safegrad")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{ts}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_downstream(args, tokenizer, model_ref, logger):
    if args.task_data_path:
        from data.local_task_dataset import build_task_dataset, infer_task_name
        task_name = infer_task_name(args.task_data_path)
        logger.info(f"[3/5] 다운스트림 '{task_name}' 로드 ({args.task_data_path})")
        ds = build_task_dataset(args.task_data_path, tokenizer, args.max_length,
                                model_ref, tokenize_sft_example,
                                max_samples=args.task_samples,
                                desc=f"Tokenizing {task_name}")
        logger.info(f"✅ {task_name} train: {len(ds)} samples")
        return ds, task_name

    logger.info("[3/5] 다운스트림 GSM8K 로드")
    raw = load_dataset(args.dataset_name, args.dataset_subset,
                       split=args.train_split, cache_dir=args.cache_dir)
    raw = _select_first_n(raw, args.num_train_samples)

    def preprocess(ex):
        return tokenize_sft_example(ex["question"], ex["answer"], tokenizer,
                                    args.max_length, model_ref)

    ds = raw.map(preprocess, remove_columns=raw.column_names,
                 num_proc=max(1, args.num_workers), desc="Tokenizing GSM8K")
    logger.info(f"✅ GSM8K train: {len(ds)} samples")
    return ds, "gsm8k"


def build_alignment(args, tokenizer, model_ref, logger):
    if not os.path.exists(args.safety_data_path):
        raise FileNotFoundError(f"Safety dataset not found: {args.safety_data_path}")
    with open(args.safety_data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    field = args.safety_response_field
    if field not in raw[0]:
        raise KeyError(f"안전 데이터에 '{field}' 필드가 없다. 있는 키: {list(raw[0].keys())}")

    # LISA arm 과 동일한 샘플링 규칙(같은 seed → 같은 부분집합).
    rng = random.Random(args.seed)
    n = min(args.guide_data_num, len(raw))
    sampled = rng.sample(raw, n)

    def preprocess(ex):
        return tokenize_sft_example(ex["prompt"], ex[field], tokenizer,
                                    args.max_length, model_ref)

    hf = HFDataset.from_list(sampled)
    ds = hf.map(preprocess, remove_columns=hf.column_names, desc="Tokenizing alignment(safety)")
    logger.info(f"✅ Alignment(safety): {len(ds)} samples (from {args.safety_data_path}, field={field})")
    return ds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger, log_file = setup_logging()
    logger.info("=" * 70)
    logger.info("  🛡️  SafeGrad — gradient surgery + KL alignment")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")

    raw_path = args.model_path
    is_local = raw_path.startswith(("./", "/", "../"))
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    ref_path = args.ref_model_path or model_path

    if args.guide_data_num <= 0:
        raise ValueError("SafeGrad 는 alignment 데이터가 필수다 (--guide_data_num > 0).")
    if not args.lora and args.ref_mode == "adapter_off":
        raise ValueError("--ref_mode adapter_off 는 LoRA 학습에서만 쓸 수 있다.")

    logger.info("⚙️  Configuration:")
    logger.info(f"   ├─ Base/aligned model : {model_path}")
    logger.info(f"   ├─ Reference θ0       : {ref_path}  (mode={args.ref_mode})")
    logger.info(f"   ├─ Input formatting   : "
                f"{'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"   ├─ Tuning             : "
                f"{'LoRA (r=%d, alpha=%d)' % (args.lora_r, args.lora_alpha) if args.lora else 'Full-parameter'}")
    logger.info(f"   ├─ SafeGrad           : rho={args.rho} projection={args.projection} "
                f"kl_reduction={args.kl_reduction} kl_fp32={args.kl_fp32}")
    logger.info(f"   ├─ Alignment data     : {args.safety_data_path} (n={args.guide_data_num})")
    logger.info(f"   ├─ Downstream         : {args.task_data_path or 'GSM8K'}  (clean)")
    logger.info(f"   ├─ LR {args.learning_rate}  epochs {args.epochs}  "
                f"eff.batch {args.batch_size * args.grad_accum}")
    logger.info(f"   └─ Output             : {args.output_dir}")

    if not args.lora:
        logger.warning("⚠️  full-param + SafeGrad 는 그래디언트 사본이 3벌 필요하다 "
                       "(prev / g_user / g_align). 모델 크기의 3배 VRAM 을 추가로 쓴다.")

    # --- tokenizer ---
    logger.info("[1/5] 토크나이저 로드")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                                  trust_remote_code=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✅ Tokenizer (vocab={len(tokenizer)}, pad={tokenizer.pad_token})")

    # --- models ---
    logger.info("[2/5] 모델 로드")
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    def _load(path):
        try:
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, device_map="auto",
                local_files_only=True, trust_remote_code=False)
        except Exception:
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, device_map="auto", trust_remote_code=False)

    model = _load(model_path)

    ref_model = None
    if args.ref_mode == "separate":
        logger.info(f"   · reference θ0 를 따로 로드: {ref_path}")
        ref_model = _load(ref_path)
        ref_model.eval()
        ref_model.requires_grad_(False)
    else:
        logger.info("   · reference θ0 = LoRA 어댑터를 끈 현재 모델 (메모리 한 벌 절약)")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.lora:
        if not _peft_available:
            raise ImportError("peft is required for LoRA training (pip install peft)")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules, bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"✓ LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"targets={args.lora_target_modules}")

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        # gradient checkpointing + 얼린 임베딩 조합에서 그래디언트가 끊기지 않게 한다.
        model.enable_input_require_grads()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Model ({total/1e9:.2f}B total, {trainable/1e6:.1f}M trainable "
                f"= {100*trainable/total:.2f}%)")

    # --- data ---
    train_ds, task_name = build_downstream(args, tokenizer, model_path, logger)
    logger.info("[4/5] alignment(safety) 데이터 준비")
    align_ds = build_alignment(args, tokenizer, model_path, logger)

    # --- training ---
    logger.info("[5/5] SafeGrad 학습 시작")
    collator = DataCollatorForCausalLMWithPadding(tokenizer)
    targs = TrainingArguments(
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
        report_to=(args.report_to if args.report_to != "none" else "none"),
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
        gradient_checkpointing=False,   # 위에서 직접 켰다 (PEFT 적용 전에 켜야 한다)
    )

    trainer = SafeGradTrainer(
        model=model, args=targs,
        train_dataset=train_ds, data_collator=collator,
        ref_model=ref_model, ref_mode=args.ref_mode,
        projection=args.projection, rho=args.rho,
        kl_reduction=args.kl_reduction, kl_fp32=args.kl_fp32,
        align_batch_size=(args.align_batch_size or None),
    )
    trainer.init(align_ds)
    trainer.train()

    # --- save ---
    logger.info("모델 저장")
    if args.lora:
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model": model_path,
        "reference_model": ref_path,
        "method": "SafeGrad (gradient surgery + KL alignment)",
        "paper": "arXiv:2508.07172",
        "fine_tuning_type": "LoRA" if args.lora else "Full Parameter",
        "dataset": args.task_data_path or "GSM8K (clean, no poison)",
        "task_name": task_name,
        "safety_data_path": args.safety_data_path,
        "safety_response_field": args.safety_response_field,
        "guide_data_num": args.guide_data_num,
        "rho": args.rho,
        "projection": args.projection,
        "ref_mode": args.ref_mode,
        "kl_reduction": args.kl_reduction,
        "kl_fp32": args.kl_fp32,
        "align_batch_size": args.align_batch_size or args.batch_size,
        "num_train_samples": len(train_ds),
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "epochs": args.epochs,
        "max_length": args.max_length, "lr_scheduler_type": args.lr_scheduler_type,
        "lora_r": args.lora_r if args.lora else None,
        "lora_alpha": args.lora_alpha if args.lora else None,
        "lora_target_modules": args.lora_target_modules if args.lora else None,
        "seed": args.seed,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
    }
    with open(os.path.join(args.output_dir, "finetune_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Saved to {args.output_dir}")

    if args.upload_name:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            api.create_repo(args.upload_name, repo_type="model", exist_ok=True, private=False)
            api.upload_folder(folder_path=args.output_dir, repo_id=args.upload_name,
                              repo_type="model",
                              ignore_patterns=["*.lock", "checkpoint-*/*", "cache/*", "*.log"])
            logger.info(f"✅ Uploaded: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"PUSH_FAILED repo={args.upload_name} dir={args.output_dir} "
                         f"err={type(e).__name__}: {str(e)[:200]}")

    logger.info("✅ SafeGrad fine-tuning complete!")


if __name__ == "__main__":
    main()
