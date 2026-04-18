"""
Base Model Safety Fine-tuning (Full Parameter FT)

Goal:
- Fair comparison baseline for SN-Tune.
- Keep training setup aligned with sn_tune.py, except:
  - Do NOT load safety neurons
  - Do NOT apply gradient masking
  - Fine-tune all model parameters on the same safety dataset

Usage:
python models/phase0_SSFT.py --model_name meta-llama/Llama-2-7b-hf
python models/phase0_SSFT.py --model_name meta-llama/Llama-2-7b-chat-hf

"""

import os
import sys
import json
import argparse
from datetime import datetime
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 로거는 main()에서 setup_logger로 초기화됨
# 모듈 레벨 fallback (직접 import 시)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('safety_warp')

# setup_logger 임포트 (train.py와 동일한 유틸)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import setup_logger as _setup_logger
except ImportError:
    _setup_logger = None



try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _wb_log(metrics: dict, step: int = None):
    """wandb.run이 활성화된 경우에만 로깅. 실패해도 무시."""
    try:
        if _wandb is not None and _wandb.run is not None:
            _wandb.log(metrics, step=step)
    except Exception:
        pass


# =====================================================================
# Configuration (matched to sn_tune.py)
# =====================================================================
MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Base 모델 (Phase 0) - WaRP 제거 전 모델 사용

LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 1024
MAX_SAMPLES = 4994

CHECKPOINTS_DIR = "./checkpoints"
DATASET_DEFAULT = "./data/circuit_breakers_train.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_instruct_model(model_name: str) -> bool:
    model_ref = model_name.lower()
    return any(tag in model_ref for tag in ('instruct', 'chat'))


# =====================================================================
# Safety Dataset (same format as sn_tune.py)
# =====================================================================
class SafetyDataset(Dataset):
    def __init__(self, json_path, tokenizer, model_name, max_samples=None, max_length=1024):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[: min(max_samples, len(self.data))]

        self.tokenizer = tokenizer
        self.model_name = model_name
        self.use_chat_template = is_instruct_model(model_name)
        self.max_length = max_length
        self._logged_first = False

        logger.info(f"Loaded {len(self.data)} samples from {json_path}")
        logger.info(
            f"Formatting mode: {'chat template (instruct)' if self.use_chat_template else 'plain Question/Answer (base)'}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        harmful_prompt = item.get("prompt", "")
        safe_response = item.get("llama3_output", "")

        if self.use_chat_template:
            # ── Instruct model: apply_chat_template(tokenize=True) ──────────
            # tokenize=True로 바로 token IDs를 가져오면 BOS 중복 등의 문제를 방지할 수 있음
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": harmful_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_length = len(prompt_ids)

            full_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": harmful_prompt},
                    {"role": "assistant", "content": safe_response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]

            seq_len = len(full_ids)
            pad_len = self.max_length - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len

            labels = list(input_ids)
            for i in range(min(prompt_length, self.max_length)):
                labels[i] = -100
            for i in range(self.max_length):
                if attention_mask[i] == 0:
                    labels[i] = -100

            if not self._logged_first:
                self._logged_first = True
                logger.info(f"\n[Dataset Sample #first] (instruct / chat template)")
                logger.info(f"  Keys: {item.keys()}")
                logger.info(f"  Prompt (first 100): {harmful_prompt[:100]}...")
                logger.info(f"  Response (first 100): {safe_response[:100]}...")
                logger.info(f"  prompt_length (tokens): {prompt_length}")
                logger.info(f"  full_ids length: {seq_len}")
                learned_ids = [t for t, l in zip(input_ids, labels) if l != -100]
                logger.info(f"  Learned tokens ({len(learned_ids)}): {self.tokenizer.decode(learned_ids)[:200]}...")
                logger.info(f"  Masked (prompt) tokens: {self.tokenizer.decode(input_ids[:prompt_length])[:200]}...")

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        else:
            # ── Base model: plain Question/Answer format ─────────────────────
            prompt_text = f"Question: {harmful_prompt}\nAnswer:"
            full_text = f"{prompt_text} {safe_response}"

            encodings = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_encodings = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            labels = encodings["input_ids"].clone()
            prompt_length = prompt_encodings["input_ids"].size(1)
            labels[:, :prompt_length] = -100
            labels[encodings["attention_mask"] == 0] = -100

            if not self._logged_first:
                self._logged_first = True
                logger.info(f"\n[Dataset Sample #first] (base / plain text)")
                logger.info(f"  Keys: {item.keys()}")
                logger.info(f"  Prompt (first 100): {harmful_prompt[:100]}...")
                logger.info(f"  Response (first 100): {safe_response[:100]}...")
                logger.info(f"  prompt_length (tokens): {prompt_length}")
                input_ids_list = encodings["input_ids"][0].tolist()
                labels_list = labels[0].tolist()
                learned_ids = [t for t, l in zip(input_ids_list, labels_list) if l != -100]
                logger.info(f"  Learned tokens ({len(learned_ids)}): {self.tokenizer.decode(learned_ids)[:200]}...")
                logger.info(f"  Masked (prompt) tokens: {self.tokenizer.decode(input_ids_list[:prompt_length])[:200]}...")

            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
            }


# =====================================================================
# Training Loop (full-parameter baseline)
# =====================================================================
def train_base_safety_ft(
    model,
    train_dataloader,
    learning_rate=1e-5,
    num_epochs=3,
    grad_accum_steps=4,
    warmup_ratio=0.1,
    device=DEVICE,
):
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_optimization_steps = num_epochs * math.ceil(len(train_dataloader) / grad_accum_steps)
    warmup_steps = int(total_optimization_steps * warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimization_steps
    )

    total_loss = 0.0
    total_steps = 0
    optimizer_steps = 0

    logger.info("Starting Base Model Safety FT training...")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Grad accum steps: {grad_accum_steps}")
    logger.info(f"  Num batches (per epoch): {len(train_dataloader)}")
    logger.info(f"  Total optimization steps: {total_optimization_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        pbar = tqdm(train_dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            use_autocast = device.startswith("cuda") or device.startswith("cpu")
            autocast_dtype = torch.bfloat16
            with torch.autocast(device_type=device if use_autocast else "cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected at batch {batch_idx + 1}. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # grad_accum_steps마다 (또는 마지막 배치에서) optimizer step
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                _wb_log({
                    'phase0/train_loss': loss.item() * grad_accum_steps,
                    'phase0/learning_rate': scheduler.get_last_lr()[0],
                    'phase0/epoch': epoch + (batch_idx + 1) / len(train_dataloader),
                }, step=global_step)

            # 화면 및 로깅에 표시되는 Loss는 원래 크기로 복원해서 보여줍니다.
            loss_val = loss.item() * grad_accum_steps
            total_loss += loss_val
            epoch_loss += loss_val
            total_steps += 1

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                avg_batch_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})

        logger.info(f"Epoch {epoch + 1} completed - Epoch Loss: {epoch_loss / len(train_dataloader):.4f}")
        _wb_log({
            'phase0/epoch_loss': epoch_loss / len(train_dataloader),
            'phase0/epoch': epoch + 1,
        }, step=global_step)

    avg_loss = total_loss / max(1, total_steps)
    logger.info(f"\n{'=' * 70}")
    logger.info("Training Complete")
    logger.info(f"{'=' * 70}")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Optimizer steps: {optimizer_steps}")
    logger.info(f"Training time: {num_epochs} epoch(s)")
    logger.info(f"{'=' * 70}\n")

    return model


# =====================================================================
# Save Model
# =====================================================================
def save_model_and_tokenizer(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")


def build_output_dir(base_dir=CHECKPOINTS_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"phase0_{timestamp}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Base Model Safety Fine-tuning (Full Parameter FT)"
    )
    parser.add_argument(
        "dataset_json",
        nargs="?",
        default=DATASET_DEFAULT,
        help="Safety dataset JSON path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Base model or instruct model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )
    parser.add_argument('--wandb_project', type=str, default='Safety-WaRP-LLM',
                        help='W&B 프로젝트 이름')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B 실행 이름 (미지정 시 자동 생성)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='W&B 로깅 비활성화')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='로그 파일 저장 디렉토리')
    return parser.parse_args(argv)


# =====================================================================
# Main
# =====================================================================
def main(argv):
    global logger
    args = parse_args(argv)
    safety_dataset_json = args.dataset_json
    model_name = args.model_name
    output_dir = args.output_dir or build_output_dir()

    # 로그 파일 설정
    log_dir = getattr(args, 'log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'phase0_{timestamp}.log')
    if _setup_logger is not None:
        logger = _setup_logger('safety_warp', log_file=log_file)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
        )
        logger = logging.getLogger('safety_warp')
    logger.info(f'Log file: {log_file}')

    if not os.path.exists(safety_dataset_json):
        logger.error(f"Safety dataset file not found: {safety_dataset_json}")
        sys.exit(1)

    logger.info(f"\n{'=' * 70}")
    logger.info("Base Model Safety Fine-tuning (Full Parameter FT)")
    logger.info(f"{'=' * 70}")
    logger.info(f"Base model: {model_name}")
    logger.info(
        f"Input format: {'chat template' if is_instruct_model(model_name) else 'plain Question/Answer'}"
    )
    logger.info(f"Safety dataset file: {safety_dataset_json}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Training setup: LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, "
        f"GradAccum={GRAD_ACCUM_STEPS}, MaxSamples={MAX_SAMPLES}"
    )
    logger.info(f"{'=' * 70}\n")

    # W&B 초기화
    if _wandb is not None and not getattr(args, 'no_wandb', False):
        try:
            run_name = getattr(args, 'wandb_run_name', None) or f"phase0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            _wandb.init(
                project=getattr(args, 'wandb_project', 'Safety-WaRP-LLM'),
                name=run_name,
                config={
                    'phase': 0,
                    'model_name': model_name,
                    'learning_rate': LEARNING_RATE,
                    'num_epochs': NUM_EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'grad_accum_steps': GRAD_ACCUM_STEPS,
                    'max_seq_length': MAX_SEQ_LENGTH,
                    'max_samples': MAX_SAMPLES,
                    'dataset': safety_dataset_json,
                },
                reinit=True,
            )
            logger.info(f"✓ W&B initialized: project={getattr(args, 'wandb_project', 'Safety-WaRP-LLM')}, run={run_name}")
        except Exception as e:
            logger.warning(f"W&B 초기화 실패 (로깅 없이 계속): {e}")

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("✓ Model and tokenizer loaded (bfloat16)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    logger.info("\nLoading safety dataset...")
    safety_dataset = SafetyDataset(
        safety_dataset_json,
        tokenizer,
        model_name=model_name,
        max_samples=MAX_SAMPLES,
        max_length=MAX_SEQ_LENGTH,
    )

    train_dataloader = DataLoader(
        safety_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(112),
    )
    logger.info(f"✓ DataLoader created: {len(train_dataloader)} batches")
    logger.info(f"  Total samples: {len(safety_dataset)}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    logger.info(f"  Max sequence length: {MAX_SEQ_LENGTH}")

    logger.info("\nStarting base model safety fine-tuning...")
    model = train_base_safety_ft(
        model,
        train_dataloader,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        device=DEVICE,
    )

    logger.info("\nSaving fine-tuned model...")
    save_model_and_tokenizer(model, tokenizer, output_dir)

    logger.info(f"\n{'=' * 70}")
    logger.info("Base Model Safety FT Complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Fine-tuned model saved to: {output_dir}")
    logger.info(f"{'=' * 70}\n")

    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
