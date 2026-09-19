"""AsFT (arXiv:2506.08473) — **full-parameter** 판 다운스트림 파인튜닝 러너.

원 논문과 참조 구현(/home/edgeai_lab/AsFT)은 LoRA 전용이다(ΔW = B A). 논문 개정본
(revisioning_wsr.tex) 의 Table 2 / Table 4 는 AsFT 를 full-parameter 계열 행에 싣기
때문에, 같은 벌점을 ΔW = W − W₀ 로 일반화해 학습한다. 수식과 구현 근거는
`models/asft_baseline.AsFTFullParamRegularizer` 의 주석에 있다.

구조는 `safegrad/finetune_safegrad.py` 와 같다 — 즉 `gsm8k_eval/finetune_gsm8k_lisa.py`
의 토큰화/콜레이터를 **그대로 import** 한다. 프롬프트 문자열이 arm 마다 한 글자라도
달라지면 비교가 성립하지 않기 때문이고, 이 저장소에 7번째 토큰화 구현을 만들지
않기 위해서다(`scripts/revision/verify_prompt_parity.py` 가 이를 검사한다).

예)
  python asft/finetune_asft_full.py \
      --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --base_model meta-llama/Llama-2-7b-chat-hf \
      --task_data_path data/gsm8k_train_task_7473.json \
      --output_dir outputs/asft_full/llama2_7b_gsm8k \
      --asft_lambda_reg 1.0 --learning_rate 5e-5 --epochs 3 \
      --batch_size 4 --grad_accum 4
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# gsm8k_eval/finetune_gsm8k_lisa.py 는 import 시점에 CUDA_VISIBLE_DEVICES 를 setdefault
# 한다(이미 값이 있으면 건드리지 않는다). '덮어쓰기' 로 바뀌면 이 러너의 GPU 지정이
# 통째로 무시되므로 감시한다 — CLAUDE.md 에 기록된 트랩이다.
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
        f"변경했다 ({_CVD_BEFORE!r} -> {os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
        "setdefault 로 되돌려라.")

from datasets import load_dataset          # noqa: E402
from transformers import (                 # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from models.asft_baseline import (         # noqa: E402
    AsFTFullParamRegularizer,
    build_alignment_dirs,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class AsFTFullTrainer(Trainer):
    """누적 주기의 **마지막 micro-batch** 에서 벌점 그래디언트를 한 번만 더한다.

    왜 여기인가:
      · 벌점은 데이터와 무관하므로 micro-batch 마다 더하면 grad_accum 배 과다계상된다.
      · `on_pre_optimizer_step` 콜백은 transformers 4.57 기준 **gradient clipping 이후**에
        불린다. 참조 구현은 벌점을 loss 에 얹으므로 SFT 항과 같이 clip 되어야 한다.
        training_step 은 clip 이전이라 이 조건을 만족한다.
    """

    def __init__(self, *a, asft_reg=None, asft_check_equiv=False, **kw):
        super().__init__(*a, **kw)
        self.asft_reg = asft_reg
        self.asft_check_equiv = asft_check_equiv
        self._micro = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._micro += 1
        if self.asft_reg is not None and self._micro % self.args.gradient_accumulation_steps == 0:
            if self.asft_check_equiv:
                self.asft_reg.check_equiv()
            self.asft_reg.add_grad_()
        return loss

    def log(self, logs, *a, **kw):
        if self.asft_reg is not None and "loss" in logs:
            logs["asft_reg"] = round(float(self.asft_reg.last_penalty), 6)
        # HF 의 ProgressCallback 은 tqdm.write 로 찍는데, 비-TTY 로 리다이렉트하면
        # 그 줄이 로그 파일에 남지 않는다(2026-09-19 실측: 진행바만 남고 loss 줄이 통째로
        # 사라졌다). λ 가 타당한지는 asft_reg 값을 봐야 알 수 있으므로 직접 찍는다.
        logging.getLogger("asft_full").info("[train] " + str(logs))
        return super().log(logs, *a, **kw)


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="AsFT full-parameter fine-tuning")

    p.add_argument("--model_path", required=True,
                   help="출발 모델 = 안전정렬된 모델 (= W_aligned = W₀)")
    p.add_argument("--base_model", required=True,
                   help="정렬 전 원본 모델 (V = W_aligned − W_base 계산용)")
    p.add_argument("--aligned_model", default=None,
                   help="V 계산에 쓸 aligned 모델. 기본은 --model_path 와 같다.")

    p.add_argument("--task_data_path", default=None,
                   help='{"question","response"} 스키마 로컬 태스크 JSON. 없으면 GSM8K 를 받는다.')
    p.add_argument("--task_samples", type=int, default=0, help="0=전체")
    p.add_argument("--dataset_name", default="openai/gsm8k")
    p.add_argument("--dataset_subset", default="main")
    p.add_argument("--train_split", default="train")
    p.add_argument("--num_train_samples", type=int, default=7473)

    p.add_argument("--asft_lambda_reg", type=float, default=1.0,
                   help="벌점 계수 λ. 참조 구현 기본값 1.0 (LoRA 스케일 위에서 정해진 값이다).")
    p.add_argument("--asft_store_dtype", choices=["float32", "bfloat16"], default="bfloat16",
                   help="V / W₀ 저장 dtype. bf16 이면 메모리가 절반이다.")
    p.add_argument("--asft_offload_cpu", action="store_true",
                   help="V / W₀ 를 CPU 에 두고 step 마다 스트리밍한다. 13B 처럼 VRAM 이 빠듯할 때.")
    p.add_argument("--asft_check_equiv", action="store_true",
                   help="해석적 그래디언트를 autograd 와 첫 step 에 대조 검증한다.")
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    p.add_argument("--train_only_targets", action="store_true",
                   help="--target_modules 에 해당하는 2-D weight 만 학습하고 나머지는 동결한다. "
                        "WSR-Tune 은 basis_coeff(= 그 5개 projection) 만 학습하므로, 공정한 "
                        "비교를 위해서는 이 플래그가 필요하다. 없으면 full-param 전체가 학습돼 "
                        "AsFT 는 o_proj/gate_proj/embed/lm_head 를 **벌점 없이** 자유롭게 바꾼다.")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--report_to", default="none")
    p.add_argument("--upload_name", default=None)
    p.add_argument("--hf_token", default=None)
    return p.parse_args()


def setup_logging(name="asft_full"):
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True)
    return logging.getLogger(name), log_file


def build_downstream(args, tokenizer, model_ref, logger):
    if args.task_data_path:
        from data.local_task_dataset import build_task_dataset, infer_task_name
        task_name = infer_task_name(args.task_data_path)
        logger.info(f"[3/4] 다운스트림 '{task_name}' 로드 ({args.task_data_path})")
        ds = build_task_dataset(args.task_data_path, tokenizer, args.max_length,
                                model_ref, tokenize_sft_example,
                                max_samples=args.task_samples,
                                desc=f"Tokenizing {task_name}")
        logger.info(f"✅ {task_name} train: {len(ds)} samples")
        return ds, task_name

    logger.info("[3/4] 다운스트림 GSM8K 로드")
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    logger, log_file = setup_logging()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    aligned = args.aligned_model or args.model_path

    logger.info("=" * 70)
    logger.info("  ⚓ AsFT (full-parameter) — alignment-direction subspace penalty")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"   ├─ start / aligned (W₀) : {args.model_path}")
    logger.info(f"   ├─ base (for V)         : {args.base_model}")
    logger.info(f"   ├─ Input formatting     : "
                f"{'chat template' if is_instruct_model(args.model_path) else 'base plain prompt'}")
    logger.info(f"   ├─ λ = {args.asft_lambda_reg}   targets = {target_modules}")
    logger.info(f"   ├─ Downstream           : {args.task_data_path or 'GSM8K'}")
    logger.info(f"   ├─ lr {args.learning_rate}  epochs {args.epochs}  "
                f"eff.batch {args.batch_size * args.grad_accum}  wd {args.weight_decay}")
    logger.info(f"   └─ Output               : {args.output_dir}")

    logger.info("[1/4] 토크나이저 로드")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("[2/4] 모델 로드 + alignment direction 구성")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map={"": 0})
    model.config.use_cache = False

    # V 는 CPU 에서 만들어 필요한 곳(GPU 또는 CPU)에 상주시킨다.
    dirs = build_alignment_dirs(
        args.base_model, aligned, target_modules,
        device=("cpu" if args.asft_offload_cpu else "cuda"),
        load_dtype=torch.float32,
        store_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[args.asft_store_dtype],
        logger=logger)

    reg = AsFTFullParamRegularizer(
        model, dirs, args.asft_lambda_reg, target_modules, logger,
        offload_cpu=args.asft_offload_cpu)

    if args.train_only_targets:
        # 벌점이 걸린 바로 그 weight 들만 학습 대상으로 남긴다(WSR-Tune 과 동일 범위).
        keep = {id(param) for param, _, _, _ in reg.entries}
        n_tr = n_fr = 0
        for _, prm in model.named_parameters():
            if id(prm) in keep:
                prm.requires_grad = True;  n_tr += prm.numel()
            else:
                prm.requires_grad = False; n_fr += prm.numel()
        logger.info(f"[AsFT-full] --train_only_targets: 학습 {n_tr:,} / 동결 {n_fr:,} "
                    f"({100*n_tr/(n_tr+n_fr):.2f}% 학습)")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_ds, task_name = build_downstream(args, tokenizer, args.model_path, logger)

    logger.info("[4/4] 학습 시작")
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
        report_to=(args.report_to if args.report_to != "none" else "none"),
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
        gradient_checkpointing=False,   # 위에서 직접 켰다
    )

    trainer = AsFTFullTrainer(
        model=model, args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
        asft_reg=reg, asft_check_equiv=args.asft_check_equiv)
    trainer.train()

    logger.info(f"최종 벌점 값: {reg.penalty():.6e}")
    reg.free()

    logger.info("모델 저장")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model": args.model_path,
        "alignment_base_model": args.base_model,
        "alignment_aligned_model": aligned,
        "method": "AsFT (full-parameter generalization, DeltaW = W - W0)",
        "paper": "arXiv:2506.08473",
        "fine_tuning_type": "Full Parameter",
        "dataset": args.task_data_path or "GSM8K",
        "task_name": task_name,
        "asft_lambda_reg": args.asft_lambda_reg,
        "target_modules": target_modules,
        "train_only_targets": bool(args.train_only_targets),
        "num_train_samples": len(train_ds),
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "epochs": args.epochs,
        "max_length": args.max_length, "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed, "dtype": "bf16" if args.bf16 else "fp32",
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

    logger.info("✅ AsFT (full-param) 완료")


if __name__ == "__main__":
    main()
