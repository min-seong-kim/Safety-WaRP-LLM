"""
GSM8K Downstream Fine-tuning with **Token-wise Constrained SFT**
(from "Safety Alignment Should Be Made More Than Just a Few Tokens Deep", ICLR 2025)

이 스크립트는 finetune_gsm8k_full_params.py(baseline, 표준 CE)를 그대로 복제하되,
**손실 함수만** shallow-vs-deep 논문의 token-wise constrained objective(Eqn 3)로 교체한 것이다.
=> 데이터셋 / 토큰화 / 하이퍼파라미터 / optimizer / scheduler 모두 baseline과 동일.
   `--loss_type standard` 로 실행하면 baseline(finetune_gsm8k_full_params.py)과 완전히 동일하게 동작한다.

비교 실험:
  (A) baseline   : python finetune_gsm8k_full_params.py  ... (또는 본 스크립트 --loss_type standard)
  (B) constrained: python finetune_gsm8k_constrained_sft.py --loss_type constrained ...

핵심 아이디어 (논문 Section 4 / Appendix D.2):
  - 표준 CE gradient(-∇logπθ)에 적응 가중치 w_t = 2·σ(β_t·Δ_t) 를 곱한다. (Δ_t = logπ_aligned - logπθ)
  - 앞쪽 토큰일수록 큰 β_t 를 주어, 응답 첫 몇 토큰의 분포가 초기 정렬 모델에서 멀어지지 않도록 강하게 보호.
  - reference 모델(π_aligned) = 학습 시작점(safety-FT 모델). 학습 전에 per-token logprob 을 1회 계산해
    데이터셋 컬럼으로 캐싱하고, 학습 중에는 reference 모델을 메모리에 들지 않는다(메모리 절약).

Example:
python finetune_gsm8k_constrained_sft.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./gsm8k_llama2_7b_chat_constrained_sft \
    --loss_type constrained \
    --learning_rate 5e-5 --epochs 3 \
    --csft_bias_factor 10 \
    --csft_bias_length 3 \
    --csft_first_token_bias_factor 3 \
    --upload_name kmseong/llama2_7b_chat_constrained_sft_5e-5_small_beta

# baseline(표준 CE)을 동일 코드로 재현:
python finetune_gsm8k_constrained_sft.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./gsm8k_llama2_7b_chat_standard \
    --loss_type standard \
    --learning_rate 5e-5 --epochs 3
"""

import argparse
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    _peft_available = True
except ImportError:
    _peft_available = False

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args():
    p = argparse.ArgumentParser(description='Constrained-SFT Finetune Safety Model on GSM8K')

    # model
    p.add_argument('--model_path', type=str,
                    default=None,
                    required=True,
                    help='HuggingFace model ID or local path (safety-FT model; reference π_aligned 와 동일)')

    # data  (baseline 과 동일)
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--num_train_samples", type=int, default=7473)
    p.add_argument("--num_eval_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # training  (baseline 과 동일)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # seq
    p.add_argument("--max_length", type=int, default=1024)

    # memory/speed knobs
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # logging/saving
    p.add_argument("--output_dir", type=str, default='./gsm8k_constrained_sft')
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none",
                    help="'none' (기본, wandb 미사용) 또는 'wandb'")
    p.add_argument("--wandb_entity", type=str, default=None,
                    help="wandb entity (미지정 시 로그인 계정의 기본 entity 사용)")
    p.add_argument("--wandb_project", type=str, default="GSM8K Full Finetuning")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default='./cache')
    p.add_argument("--upload_name", type=str, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    # Safety data mixing  (baseline 과 동일; constrained 비교에서는 보통 0)
    p.add_argument("--safety_data_path", type=str,
                    default="/home/yonsei_jong/Safety-Neuron/neuron_detection/corpus_all/circuit_breakers_train.json")
    p.add_argument("--safety_mix_ratio", type=float, default=0.0)

    # LoRA (baseline 과 동일; 본 비교는 full-param 권장이므로 기본 비활성화)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs='+',
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # ───────────────────────────────────────────────────────────────────
    # Constrained SFT 전용 하이퍼파라미터 (shallow-vs-deep README 기본값)
    #   β_1               = beta * first_token_bias_factor = 0.1 * 5  = 0.5
    #   β_{2..bias_length}= beta * bias_factor             = 0.1 * 20 = 2.0
    #   β_{t>bias_length} = beta                           = 0.1
    # ───────────────────────────────────────────────────────────────────
    p.add_argument("--loss_type", type=str, default="constrained",
                    choices=["constrained", "standard"],
                    help="constrained = token-wise constrained SFT, standard = baseline CE")
    p.add_argument("--csft_beta", type=float, default=0.1,
                    help="뒤쪽 토큰(t>bias_length)에 적용하는 base β")
    p.add_argument("--csft_bias_factor", type=float, default=20.0,
                    help="앞쪽 토큰(2..bias_length)의 β 배율")
    p.add_argument("--csft_first_token_bias_factor", type=float, default=5.0,
                    help="첫 토큰(t=1)의 β 배율 (수치 안정성을 위해 약간 작게)")
    p.add_argument("--csft_bias_length", type=int, default=5,
                    help="강한 제약을 거는 응답 앞부분 토큰 수")

    return p.parse_args()


def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    n = min(n, len(ds))
    return ds.select(range(n))


def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


def build_chat_prompt(question: str, tokenizer) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def tokenize_sft_example(question: str, answer_text: str, tokenizer, max_length: int, model_ref: str) -> Dict[str, List[int]]:
    """SFT 형식으로 토큰화 (baseline 과 동일): base 는 plain prompt, instruct/chat 는 chat template."""
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
                prompt_text, add_special_tokens=False, truncation=True, max_length=max_length,
            )["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length,
            )["input_ids"]

            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        except Exception:
            pass

    prompt_text = build_chat_prompt(question, tokenizer)
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length,
    )["input_ids"]

    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text, add_special_tokens=False, truncation=True, max_length=remain,
    )["input_ids"]

    if tokenizer.eos_token_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithPadding:
    """패딩된 배치 생성. ref_logps 컬럼이 있으면 (가변 길이 list) 그대로 통과시킨다."""
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            l = len(f["input_ids"])
            pad_len = max_len - l
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # constrained 모드: reference per-token logprob (가변 길이) 를 list 로 통과
        if "ref_logps" in features[0]:
            batch["ref_logps"] = [list(f["ref_logps"]) for f in features]
        return batch


def per_token_answer_logps(logits: torch.Tensor, labels: torch.Tensor):
    """auto-regressive shift 후, label != -100 (=응답 토큰) 위치의 per-token logπ 를
    example 별 1D 텐서 리스트로 반환. shallow-vs-deep 의 get_batch_logps 와 동일한 정의."""
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    # float 로 올려 log_softmax (HF CE 내부 동작과 일치 + bf16 수치 안정성)
    logps = torch.gather(logits.float().log_softmax(-1), dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    return [logps[i][mask[i]] for i in range(logps.size(0))]


class ConstrainedSFTTrainer(Trainer):
    """HF Trainer 위에 token-wise constrained loss 만 얹은 trainer.

    loss_type == 'standard' 이면 baseline(CE)과 완전히 동일하게 동작한다.
    """

    def __init__(self, *args, loss_type="constrained", csft_beta=0.1, csft_bias_factor=20.0,
                 csft_first_token_bias_factor=5.0, csft_bias_length=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.csft_beta = csft_beta
        self.csft_bias_factor = csft_bias_factor
        self.csft_first_token_bias_factor = csft_first_token_bias_factor
        self.csft_bias_length = csft_bias_length

    def get_beta_list(self, length: int) -> torch.Tensor:
        """길이 length 의 응답에 대한 토큰별 β 스케줄. (shallow-vs-deep get_beta_list 와 동일)"""
        beta = self.csft_beta
        len_prefix = self.csft_bias_length
        prefix = torch.full((len_prefix,), beta * self.csft_bias_factor)
        if len_prefix != 0:
            prefix[0] = beta * self.csft_first_token_bias_factor  # 첫 토큰은 약간 약하게

        if length <= len_prefix:
            return prefix[:length]
        beta_list = torch.full((length,), beta)
        beta_list[:len_prefix] = prefix
        return beta_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ── standard 경로: baseline(CE)과 동일 ─────────────────────────────
        if self.loss_type != "constrained":
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # ── constrained 경로: token-wise constrained objective ─────────────
        ref_logps_list = inputs.pop("ref_logps")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        policy_list = per_token_answer_logps(outputs.logits, inputs["labels"].to(outputs.logits.device))

        losses = []
        for i, policy_item in enumerate(policy_list):
            ref_item = torch.tensor(ref_logps_list[i], device=policy_item.device, dtype=policy_item.dtype)
            n = min(policy_item.shape[0], ref_item.shape[0])
            if n == 0:
                continue
            policy_item = policy_item[:n]
            ref_item = ref_item[:n]
            beta = self.get_beta_list(n).to(device=policy_item.device, dtype=policy_item.dtype)

            # w_t = 2·σ(β_t·Δ_t), Δ_t = logπ_aligned - logπθ = ref_item - policy_item
            # policy_item - ref_item = -Δ_t 이므로 (1 - σ(β·(policy-ref))) = σ(β·Δ_t)
            # .detach() 로 가중치를 상수화 → gradient 가 논문 Eqn 5 와 정확히 일치 (weighted CE)
            weight = 2.0 * (1.0 - torch.sigmoid(beta * (policy_item - ref_item))).detach()
            weight = torch.clamp(weight, min=1e-3)
            losses.append(-(weight * policy_item))   # -logπθ = CE, 가중치 곱

        if losses:
            loss = torch.cat(losses).mean()
        else:
            loss = outputs.logits.sum() * 0.0  # 빈 배치 방어

        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def precompute_reference_logps(model, tokenized_ds, collator, batch_size, logger):
    """학습 시작 전(=모델이 아직 π_aligned 일 때) 응답 토큰별 reference logprob 를 1회 계산.
    반환: 길이 len(tokenized_ds) 의 list. 각 원소는 해당 example 의 응답 토큰 logπ_aligned list.
    (shuffle=False 로 순서를 보존하므로 add_column 으로 그대로 정렬 정합)"""
    was_training = model.training
    model.eval()
    emb_device = model.get_input_embeddings().weight.device

    loader = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    all_ref = []
    for batch in tqdm(loader, desc="Precompute reference logps", dynamic_ncols=True):
        input_ids = batch["input_ids"].to(emb_device)
        attention_mask = batch["attention_mask"].to(emb_device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        per_ex = per_token_answer_logps(logits, batch["labels"].to(logits.device))
        for t in per_ex:
            all_ref.append(t.detach().float().cpu().tolist())

    if was_training:
        model.train()
    logger.info(f"✅ Reference logps precomputed for {len(all_ref)} examples")
    return all_ref


def setup_logging(output_dir):
    log_dir = "./logs/constrained_sft_gsm8k"
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_gsm8k_csft_{log_timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path

    logger, log_file = setup_logging(args.output_dir)

    logger.info(f"\n{'='*70}")
    logger.info(f"  🚀 GSM8K Fine-tuning — loss_type = {args.loss_type.upper()}")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")

    if is_local and not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ Model (init = reference π_aligned): {model_path}")
    logger.info(f"   ├─ Loss type: {args.loss_type}")
    if args.loss_type == "constrained":
        b = args.csft_beta
        logger.info(f"   ├─ Constrained β schedule: β1={b*args.csft_first_token_bias_factor:.3g}, "
                    f"β2..{args.csft_bias_length}={b*args.csft_bias_factor:.3g}, β_rest={b:.3g}")
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"   ├─ Training samples: {args.num_train_samples}")
    logger.info(f"   ├─ Batch size: {args.batch_size}  | grad accum: {args.grad_accum}  | epochs: {args.epochs}")
    logger.info(f"   ├─ LR: {args.learning_rate}  | wd: {args.weight_decay}  | warmup: {args.warmup_ratio}  | sched: {args.lr_scheduler_type}")
    logger.info(f"   ├─ Max length: {args.max_length}  | dtype: bf16")
    logger.info(f"   └─ Output dir: {args.output_dir}\n")

    # ── Tokenizer ──────────────────────────────────────────────────────
    logger.info("[1/4] Loading Tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✓ Tokenizer loaded (vocab={len(tokenizer)}, pad={tokenizer.pad_token})")

    # ── Model ──────────────────────────────────────────────────────────
    logger.info("[2/4] Loading Model (bf16)")
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", local_files_only=True, trust_remote_code=False,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=False,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.lora:
        if not _peft_available:
            raise ImportError("peft is required for LoRA training")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules, bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"✓ LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Model loaded: {total_params/1e9:.2f}B params, trainable {100*trainable_params/total_params:.2f}%")

    # ── Dataset (baseline 과 동일) ──────────────────────────────────────
    logger.info("[3/4] Loading & Preprocessing GSM8K")
    train_ds = load_dataset(args.dataset_name, args.dataset_subset, split=args.train_split, cache_dir=args.cache_dir)
    train_ds = _select_first_n(train_ds, args.num_train_samples)

    eval_ds = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_ds = load_dataset(args.dataset_name, args.dataset_subset, split=args.eval_split, cache_dir=args.cache_dir)
        eval_ds = _select_first_n(eval_ds, args.num_eval_samples)

    def preprocess(ex):
        return tokenize_sft_example(ex["question"], ex["answer"], tokenizer, args.max_length, model_path)

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names,
                             num_proc=max(1, args.num_workers), desc="Tokenizing train")
    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(preprocess, remove_columns=eval_ds.column_names,
                               num_proc=max(1, args.num_workers), desc="Tokenizing eval")

    # Safety data mixing (baseline 과 동일; constrained 비교에서는 보통 미사용)
    if args.safety_mix_ratio > 0:
        if not os.path.exists(args.safety_data_path):
            raise FileNotFoundError(f"Safety dataset not found: {args.safety_data_path}")
        with open(args.safety_data_path, "r", encoding="utf-8") as f:
            safety_raw = json.load(f)
        num_safety = int(len(train_tok) * args.safety_mix_ratio)
        rng = random.Random(args.seed)
        sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))

        def preprocess_safety(ex):
            return tokenize_sft_example(ex["prompt"], ex["llama3_output"], tokenizer, args.max_length, model_path)

        safety_hf = HFDataset.from_list(sampled)
        safety_tok = safety_hf.map(preprocess_safety, remove_columns=safety_hf.column_names, desc="Tokenizing safety data")
        train_tok = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=args.seed)
        logger.info(f"✅ Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio})")

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)

    # ── Constrained 모드: reference logprob 사전 캐싱 ───────────────────
    if args.loss_type == "constrained":
        logger.info("[3.5/4] Precomputing reference logps (π_aligned = init model)")
        ref_list = precompute_reference_logps(
            model, train_tok, data_collator, batch_size=args.eval_batch_size, logger=logger,
        )
        train_tok = train_tok.add_column("ref_logps", ref_list)

    # ── Trainer ────────────────────────────────────────────────────────
    logger.info("[4/4] Training")
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
        report_to=(args.report_to if args.report_to != "none" else "none"),
        remove_unused_columns=False,   # ref_logps 컬럼 보존에 필수
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    use_wandb = (args.report_to == "wandb")
    wandb.init(
        entity=args.wandb_entity,                 # None = 로그인 계정 기본 entity
        project=args.wandb_project,
        name=run_name,
        mode=("online" if use_wandb else "disabled"),   # report_to=none 이면 no-op (네트워크/권한 불필요)
        config={
            "model_path": model_path,
            "loss_type": args.loss_type,
            "learning_rate": args.learning_rate,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler_type,
            "dataset": "gsm8k",
            "is_instruct": is_instruct_model(model_path),
            "safety_mix_ratio": args.safety_mix_ratio,
            "csft_beta": args.csft_beta if args.loss_type == "constrained" else None,
            "csft_bias_factor": args.csft_bias_factor if args.loss_type == "constrained" else None,
            "csft_first_token_bias_factor": args.csft_first_token_bias_factor if args.loss_type == "constrained" else None,
            "csft_bias_length": args.csft_bias_length if args.loss_type == "constrained" else None,
        },
    )

    trainer = ConstrainedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_type=args.loss_type,
        csft_beta=args.csft_beta,
        csft_bias_factor=args.csft_bias_factor,
        csft_first_token_bias_factor=args.csft_first_token_bias_factor,
        csft_bias_length=args.csft_bias_length,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────
    logger.info("Saving model")
    if args.lora:
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        'base_model': model_path,
        'fine_tuning_type': 'LoRA' if args.lora else 'Full Parameter',
        'loss_type': args.loss_type,
        'dataset': 'GSM8K',
        'num_train_samples': args.num_train_samples,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'epochs': args.epochs,
        'max_length': args.max_length,
        'max_grad_norm': args.max_grad_norm,
        'lr_scheduler_type': args.lr_scheduler_type,
        'optimizer': 'AdamW (torch)',
        'dtype': 'bf16',
        'csft_beta': args.csft_beta if args.loss_type == "constrained" else None,
        'csft_bias_factor': args.csft_bias_factor if args.loss_type == "constrained" else None,
        'csft_first_token_bias_factor': args.csft_first_token_bias_factor if args.loss_type == "constrained" else None,
        'csft_bias_length': args.csft_bias_length if args.loss_type == "constrained" else None,
        'safety_mix_ratio': args.safety_mix_ratio,
    }
    with open(os.path.join(args.output_dir, 'finetune_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Saved to {args.output_dir}")

    if args.upload_name:
        try:
            from upload_sn_tuned_model import upload_to_huggingface
            upload_to_huggingface(args.output_dir, args.upload_name, args.hf_token)
            logger.info(f"✅ Uploaded: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")

    logger.info("✅ Fine-tuning Complete!")
    wandb.finish()


if __name__ == '__main__':
    main()
