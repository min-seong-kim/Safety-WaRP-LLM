"""
LISA (Lazy Safety Alignment / Bi-State Optimization) fine-tuning on GSM8K.

paper: https://arxiv.org/abs/2405.18641

이 스크립트는 finetune_gsm8k_full_params.py 의 모델/데이터/토크나이즈 컨벤션을
그대로 재사용하되, 학습 로직을 LISA 의 Bi-State Optimization(BSO) 로 교체한다.

BSO 개요
--------
- downstream(=finetune) 데이터(GSM8K)와 alignment(=safety) 데이터(circuit_breakers)를
  스텝 단위로 번갈아 학습한다.
    · finetune status: finetune_step 만큼 GSM8K 배치로 학습
    · alignment status: alignment_step 만큼 circuit_breakers(안전 응답) 배치로 학습
- status 가 바뀔 때마다 현재 weight 를 consensus anchor 로 저장(switch_model),
  각 status 의 loss 에 proximal term rho/2 * ||theta - anchor||^2 를 더해
  두 상태의 파라미터가 consensus 로 수렴하도록 당긴다.
- proximal term 은 benign 정확도 손상을 막기 위해 전체 스텝의 앞 10% 구간에서는 건너뛴다.

기본 세팅은 LoRA 이며(사용자 선택), attack(poison) 없이 clean GSM8K 로만 downstream
학습한다. alignment/safety 데이터로 circuit_breakers_train.json 의 안전 응답(llama3_output)
을 사용한다.

Example
-------
python finetune_gsm8k_lisa.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./lisa_gsm8k_llama2_7b_chat \
    --learning_rate 5e-5 --epochs 3 \
    --lora --lora_r 16 --lora_alpha 32 \
    --rho 1.0 --alignment_step 100 --finetune_step 900 --guide_data_num 4994
"""

import argparse
import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datetime import datetime
import logging

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import is_sagemaker_mp_enabled

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _peft_available = True
except ImportError:
    _peft_available = False

# 참고 스크립트와 동일하게 GPU 를 고정한다. 셸에서 CUDA_VISIBLE_DEVICES 를 이미
# 지정했다면 그것을 존중한다.
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='LISA (Bi-State Optimization) Fine-tune on GSM8K')

    # model
    p.add_argument('--model_path', type=str, default=None, required=True,
                   help='HuggingFace model ID or local path (safety-aligned base model)')

    # downstream data (GSM8K)
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--num_train_samples", type=int, default=7473)
    p.add_argument("--num_eval_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # training
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

    # memory/speed
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # logging/saving
    p.add_argument("--output_dir", type=str, default='./lisa_gsm8k')
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none",
                   help="'none'(기본, wandb 미사용) 또는 'wandb'")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="GSM8K LISA")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default='./cache')
    p.add_argument("--upload_name", type=str, default=None,
                   help="Optional HF repo id. 지정 시 학습 후 업로드")
    p.add_argument("--hf_token", type=str, default=None)

    # --- LISA / BSO ---
    p.add_argument("--safety_data_path", type=str,
                   default="/home/gokms0509/Safety-WaRP-LLM/data/circuit_breakers_train.json",
                   help="alignment(safety) 데이터 JSON (prompt / llama3_output 필드 사용)")
    p.add_argument("--guide_data_num", type=int, default=4994,
                   help="alignment 데이터에서 사용할 안전 예시 개수 (0 이면 BSO 비활성 = 순수 SFT)")
    p.add_argument("--rho", type=float, default=1.0,
                   help="proximal term 계수 (0 이면 consensus 정규화 없이 dataset alternation 만 수행)")
    p.add_argument("--alignment_step", type=int, default=100,
                   help="한 번의 alignment status 에서 도는 스텝 수")
    p.add_argument("--finetune_step", type=int, default=900,
                   help="한 번의 finetune status 에서 도는 스텝 수")

    # LoRA
    p.add_argument("--lora", action="store_true", default=True,
                   help="LoRA 로 학습 (기본 활성). full-param 을 원하면 --no_lora")
    p.add_argument("--no_lora", dest="lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# data helpers (참고 스크립트 finetune_gsm8k_full_params.py 와 동일)
# ---------------------------------------------------------------------------
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
    """SFT 형식으로 토큰화: base 는 plain prompt, instruct 는 chat template. 프롬프트는 -100 마스킹."""
    question = str(question).strip()
    answer_text = str(answer_text).strip()

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer_text}],
                tokenize=False, add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                                   truncation=True, max_length=max_length)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 truncation=True, max_length=max_length)["input_ids"]
            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100
            return {"input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels}
        except Exception:
            pass

    prompt_text = build_chat_prompt(question, tokenizer)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                           truncation=True, max_length=max_length)["input_ids"]
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=remain)["input_ids"]
    if tokenizer.eos_token_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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


def setup_logging(output_dir):
    log_dir = "./logs/lisa_gsm8k"
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_gsm8k_lisa_{log_timestamp}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


# ---------------------------------------------------------------------------
# LISA Trainer (Lisa/trainer.py 의 LisaTrainer 를 이식)
# ---------------------------------------------------------------------------
class LisaTrainer(Trainer):
    """Bi-State Optimization trainer.

    train_dataset      : downstream(finetune) 데이터 (GSM8K)
    alignment_dataset  : safety(alignment) 데이터 (circuit_breakers 안전 응답)
    """

    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        from transformers.trainer_utils import seed_worker
        sampler = RandomSampler(alignment_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))

    def init(self, alignment_dataset):
        # BSO 활성 여부: alignment_step 과 guide_data_num 이 모두 유효할 때만 alignment 부터 시작
        if self.args.alignment_step != 0 and self.args.guide_data_num > 0:
            self.status = "alignment"
        else:
            self.status = "finetune"
        self.alignment_weights = {}
        self.finetune_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()
        self.clock = 0
        self.steps = 0
        # proximal term 은 전체 스텝의 앞 10% 이후부터 적용
        self.prox_start_step = 0.1 * len(self.get_train_dataloader()) * self.args.num_train_epochs
        if self.args.guide_data_num > 0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)

    def switch_model(self):
        """status 전환 시 현재 weight 를 상대 상태의 consensus anchor 로 저장."""
        sum_drift = 0
        if self.status == "alignment":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name]) ** 2
            print("finetuning drift to consensus {}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name]) ** 2
            print("alignment drift to consensus {}".format(sum_drift))

    def sample_from_alignment(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch

    def check_mode(self, inputs):
        """스텝 카운터를 보고 status 전환 및 (alignment 이면) 배치 교체를 수행."""
        if self.status == "alignment":
            if self.clock % self.args.alignment_step == 0 and self.steps != 0 and self.args.finetune_step != 0:
                self.status = "finetune"
                self.switch_model()
                self.clock = 0
                # 이번 스텝부터는 downstream 배치(inputs) 를 그대로 사용
            else:
                # alignment status: downstream 배치를 버리고 alignment 배치로 교체
                inputs = self.sample_from_alignment()
        else:
            if (self.clock % self.args.finetune_step == 0 and self.steps != 0
                    and self.args.alignment_step != 0 and self.args.guide_data_num > 0):
                self.status = "alignment"
                self.switch_model()
                inputs = self.sample_from_alignment()
                self.clock = 0
        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        # status 에 따라 dataset/model 스위칭
        inputs = self.check_mode(inputs)
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

            # proximal term (consensus 로 당기기). 앞 10% 구간은 스킵.
            if self.steps > self.prox_start_step and self.args.rho > 0:
                if self.status == "alignment":
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            loss += self.args.rho / 2 * torch.norm(param - self.finetune_weights[name]) ** 2
                else:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            loss += self.args.rho / 2 * torch.norm(param - self.alignment_weights[name]) ** 2

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        loss = step()
        self.steps += 1
        self.clock += 1
        return loss.detach() / self.args.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path

    logger, log_file = setup_logging(args.output_dir)
    logger.info(f"\n{'='*70}")
    logger.info(f"  🚀 LISA (Bi-State Optimization) GSM8K Fine-tuning")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")

    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    bso_on = args.guide_data_num > 0 and args.alignment_step > 0
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ Base model: {model_path}")
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"   ├─ Tuning: {'LoRA (r=%d, alpha=%d)' % (args.lora_r, args.lora_alpha) if args.lora else 'Full-parameter'}")
    logger.info(f"   ├─ BSO: {'ON' if bso_on else 'OFF (순수 SFT)'}  rho={args.rho} "
                f"align_step={args.alignment_step} finetune_step={args.finetune_step}")
    logger.info(f"   ├─ Safety(alignment) data: {args.safety_data_path} (guide_data_num={args.guide_data_num})")
    logger.info(f"   ├─ Downstream: GSM8K ({args.num_train_samples} samples), clean (no poison)")
    logger.info(f"   ├─ LR: {args.learning_rate}  epochs: {args.epochs}  "
                f"eff.batch: {args.batch_size * args.grad_accum}")
    logger.info(f"   └─ Output dir: {args.output_dir}\n")

    # --- tokenizer ---
    logger.info("[1/4] Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✅ Tokenizer loaded (vocab={len(tokenizer)}, pad={tokenizer.pad_token})")

    # --- model ---
    logger.info("[2/4] Loading model (bf16)")
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto",
            local_files_only=True, trust_remote_code=False)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=False)

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
        logger.info(f"✓ LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"targets={args.lora_target_modules}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Model loaded ({total_params/1e9:.2f}B total, "
                f"{trainable_params/1e6:.1f}M trainable = {100*trainable_params/total_params:.2f}%)")

    # --- downstream (GSM8K) ---
    logger.info("[3/4] Loading & tokenizing GSM8K (downstream/finetune)")
    train_ds = load_dataset(args.dataset_name, args.dataset_subset,
                            split=args.train_split, cache_dir=args.cache_dir)
    train_ds = _select_first_n(train_ds, args.num_train_samples)

    eval_ds = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_ds = load_dataset(args.dataset_name, args.dataset_subset,
                               split=args.eval_split, cache_dir=args.cache_dir)
        eval_ds = _select_first_n(eval_ds, args.num_eval_samples)

    def preprocess_gsm8k(ex):
        return tokenize_sft_example(ex["question"], ex["answer"], tokenizer, args.max_length, model_path)

    train_tok = train_ds.map(preprocess_gsm8k, remove_columns=train_ds.column_names,
                             num_proc=max(1, args.num_workers), desc="Tokenizing GSM8K")
    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(preprocess_gsm8k, remove_columns=eval_ds.column_names,
                               num_proc=max(1, args.num_workers), desc="Tokenizing eval")
    logger.info(f"✅ GSM8K train: {len(train_tok)} samples")

    # --- alignment (circuit_breakers safety responses) ---
    alignment_tok = None
    if bso_on:
        if not os.path.exists(args.safety_data_path):
            raise FileNotFoundError(f"Safety dataset not found: {args.safety_data_path}")
        with open(args.safety_data_path, "r", encoding="utf-8") as f:
            safety_raw = json.load(f)
        rng = random.Random(args.seed)
        n_align = min(args.guide_data_num, len(safety_raw))
        sampled = rng.sample(safety_raw, n_align)

        def preprocess_safety(ex):
            # 안전 응답(llama3_output) 을 정답으로 학습 → alignment status 의 학습 신호
            return tokenize_sft_example(ex["prompt"], ex["llama3_output"], tokenizer,
                                        args.max_length, model_path)

        safety_hf = HFDataset.from_list(sampled)
        alignment_tok = safety_hf.map(preprocess_safety, remove_columns=safety_hf.column_names,
                                      desc="Tokenizing safety(alignment)")
        logger.info(f"✅ Alignment(safety) data: {len(alignment_tok)} samples "
                    f"(from {args.safety_data_path})")

    # --- training ---
    logger.info("[4/4] Training with LISA BSO")
    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)
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
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    # LISA 는 dataset alternation 을 스텝 단위로 하므로 train_dataset 을 셔플하지 않도록
    # RandomSampler 대신 HF 기본을 쓰되, alternation 은 트레이너 내부에서 처리한다.
    training_args.alignment_step = args.alignment_step
    training_args.finetune_step = args.finetune_step
    training_args.guide_data_num = args.guide_data_num
    training_args.rho = args.rho

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    use_wandb = (args.report_to == "wandb")
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
               mode=("online" if use_wandb else "disabled"),
               config={
                   "model_path": model_path, "learning_rate": args.learning_rate,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "grad_accum": args.grad_accum, "max_length": args.max_length,
                   "lora": args.lora, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                   "rho": args.rho, "alignment_step": args.alignment_step,
                   "finetune_step": args.finetune_step, "guide_data_num": args.guide_data_num,
                   "safety_data_path": args.safety_data_path, "dataset": "gsm8k",
                   "bso": bso_on,
               })

    trainer = LisaTrainer(
        model=model, args=training_args,
        train_dataset=train_tok, eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer, data_collator=data_collator,
    )
    trainer.init(alignment_tok)

    logger.info("Starting training...")
    trainer.train()

    # --- save ---
    logger.info("Saving model")
    if args.lora:
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        'base_model': model_path,
        'method': 'LISA (Bi-State Optimization)',
        'fine_tuning_type': 'LoRA' if args.lora else 'Full Parameter',
        'dataset': 'GSM8K (clean, no poison)',
        'safety_data_path': args.safety_data_path,
        'guide_data_num': args.guide_data_num,
        'rho': args.rho,
        'alignment_step': args.alignment_step,
        'finetune_step': args.finetune_step,
        'num_train_samples': args.num_train_samples,
        'batch_size': args.batch_size, 'grad_accum': args.grad_accum,
        'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio, 'epochs': args.epochs,
        'max_length': args.max_length, 'lr_scheduler_type': args.lr_scheduler_type,
        'lora_r': args.lora_r if args.lora else None,
        'lora_alpha': args.lora_alpha if args.lora else None,
        'dtype': 'bf16',
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

    logger.info("✅ LISA fine-tuning complete!")
    wandb.finish()


if __name__ == '__main__':
    main()
