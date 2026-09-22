"""
finetune_freeze_sn.py — downstream FT 하면서 safety 뉴런을 **얼린다** (원공간).

SN-Tune 의 역방향이다. SN-Tune 은 safety 뉴런만 학습하고, 이 스크립트는 safety 뉴런만
빼고 전부 학습한다. 그래서 downstream 성능을 얻으면서 safety 가 덜 무너진다.

    SN-Tune 계열   --safety_neurons_file <safety_neuron_....txt>
    RSN-Tune 계열  --safety_neurons_file <critical_safety_....txt>

동결이 두 겹인 이유 (둘 중 하나만으로는 안 샌다고 보장되지 않는다):
  1. gradient hook 으로 safety 인덱스의 grad 를 0 으로 만든다.
  2. `SafetyNeuronRestoreCallback` 이 optimizer step **마다** 그 가중치를 되돌린다.
     AdamW 의 decoupled weight decay(λθ)는 gradient 와 무관하게 파라미터를 0 쪽으로
     끌어당기므로, grad 를 0 으로 만든 것만으로는 얼어붙지 않는다.

정품 transformers(`hb`)에서 돌린다 — 패치는 검출 전용이다.

사용
────
SN-Tune arm
    python -m sn_tune.finetune_freeze_sn \
        --model_path <SN-Tune 한 모델> \
        --safety_neurons_file ./sn_tune/output_neurons/safety_neuron_accelerated_<ts>.txt \
        --output_dir ./outputs/sn_tune/llama2_7b_gsm8k_freeze_sn \
        --learning_rate 5e-5 --epochs 3

RSN-Tune arm — `--safety_neurons_file` 만 critical 파일로 바꾼다.

⚠️ `--learning_rate` 는 항상 명시하라. 원본의 기본값은 7e-5 였는데 이는 공개된 어떤
모델과도 맞지 않는다 (7B·13B RSN 모델 전부 5e-5). 그래서 기본값을 5e-5 로 바꿔 두었다.
"""

import argparse
import ast
import os
import gc
import json
import re
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# CUDA_VISIBLE_DEVICES 는 건드리지 않는다 (저장소 규칙). 호출자가 정한다.


def parse_args():
    p = argparse.ArgumentParser(description='GSM8K Finetuning with Safety Neuron Freezing')
    
    # model
    p.add_argument('--model_path', type=str, 
                    default="kmseong/Llama-3.2-3B-only-sn-tuned",
                    help='HuggingFace model ID or local path (SN-Tuned model)')
    
    # safety neurons
    p.add_argument('--safety_neurons_file', type=str,
                    required=True,
                    help='Path to safety neurons txt file')
    
    # data
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
    # 원본 기본값은 7e-5 였다. 공개된 SN/RSN 모델은 전부 5e-5 이고 이 저장소의
    # full-param 동작점도 5e-5 라 맞춰 두었다. 실험마다 명시적으로 주는 것이 원칙이다.
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", type=str, default="adamw_torch")
    
    # seq
    p.add_argument("--max_length", type=int, default=1024)
    
    # memory/speed knobs
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    
    # logging/saving
    p.add_argument("--output_dir", type=str, default='./gsm8k_freeze_sn_finetune')
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_workers", type=int, default=4)
    # 기본값이 './cache' 였는데, 그러면 HF_HOME 캐시에 이미 있는 데이터셋을 못 찾고
    # 새로 받으려다 오프라인에서 죽는다 (ConnectionError: OfflineModeIsEnabled).
    # None 이면 load_dataset 이 HF 기본 캐시를 쓴다.
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--task_data_path", type=str, default=None,
                   help=('태스크 데이터를 Hub 대신 로컬 JSON 에서 읽는다 '
                         '([{"question":..., "response":...}]).'))
    p.add_argument("--upload_name", type=str, default=None,
                    help="Optional Hugging Face repo id (e.g., username/model-name). If set, upload after training")
    # 저장 디렉토리에는 _<timestamp> 가 붙어서 --output_dir 그대로가 아니다.
    # 후속 단계(평가/업로드)가 경로를 찾을 수 있도록 두 옵션을 둔다.
    p.add_argument("--no_timestamp_suffix", action="store_true",
                   help="_<timestamp> 접미사 없이 --output_dir 에 그대로 저장")
    p.add_argument("--model_dir_file", type=str, default=None,
                   help="최종 저장 경로를 이 파일에 적는다")
    p.add_argument("--hf_token", type=str, default=None,
                    help="Optional Hugging Face token for upload")
    
    return p.parse_args()


def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    n = min(n, len(ds))
    return ds.select(range(n))


def is_instruct_model(model_ref: str) -> bool:
    ref = str(model_ref).lower()
    if "instruct" in ref or "chat" in ref:
        return True
    # 'gemma-2-9b-it' 처럼 -it 가 독립 토큰으로 붙는 경우도 instruct 모델이다.
    # 단순 `"it" in ref` 는 경로 문자열의 아무 곳이나 걸리므로 토큰 경계를 본다.
    # ⚠️ 이 규칙은 models/phase3_extra_learning · agnews_eval/finetune_agnews_full_params ·
    #    gsm8k_eval · seal 의 같은 이름 함수와 **동일**해야 한다. 갈라지면 같은 모델을
    #    arm 마다 다른 프롬프트 포맷으로 학습하게 된다 (실제로 gemma-2-9b-it 에서 터졌다).
    return re.search(r"(?:^|[^a-z0-9])it(?:[^a-z0-9]|$)", ref) is not None


def build_chat_prompt(question: str, tokenizer) -> str:
    """베이스 모델용 프롬프트 빌딩"""
    return f"Question: {question.strip()}\nAnswer:"


def tokenize_sft_example(question: str, answer_text: str, tokenizer, max_length: int, model_ref: str) -> Dict[str, List[int]]:
    """SFT 형식으로 토큰화: base는 plain prompt, instruct는 chat template 사용"""
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
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
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
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    # Ensure room for answer
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remain,
    )["input_ids"]

    # Add EOS if possible and fits
    if tokenizer.eos_token_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    # Loss only on answer tokens (프롬프트는 -100으로 마스킹)
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithPadding:
    """패딩된 배치 생성"""
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =====================================================================
# Load Safety Neurons from Detection Output
# =====================================================================
def load_safety_neurons(output_file, logger):
    """
    Load safety neurons from detection output file
    
    Format:
        Line 0: ffn_up_common (dict)
        Line 1: ffn_down_common (dict)
        Line 2: q_common (dict)
        Line 3: k_common (dict)
        Line 4: v_common (dict)
    
    Returns:
        safety_neurons: {
            'ffn_up': {layer_idx: set(neuron_names)},
            'ffn_down': {layer_idx: set(neuron_names)},
            'q': {layer_idx: set(neuron_names)},
            'k': {layer_idx: set(neuron_names)},
            'v': {layer_idx: set(neuron_names)},
        }
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    safety_neurons = {}
    
    # Parse each line as a dict string and convert keys from string to int
    try:
        # Keys are stored as strings, need to convert to int
        ffn_up_raw = ast.literal_eval(lines[0].strip())
        ffn_down_raw = ast.literal_eval(lines[1].strip())
        q_raw = ast.literal_eval(lines[2].strip())
        k_raw = ast.literal_eval(lines[3].strip())
        v_raw = ast.literal_eval(lines[4].strip())
        
        # Convert string keys to int keys
        safety_neurons['ffn_up'] = {int(k): v for k, v in ffn_up_raw.items()}
        safety_neurons['ffn_down'] = {int(k): v for k, v in ffn_down_raw.items()}
        safety_neurons['q'] = {int(k): v for k, v in q_raw.items()}
        safety_neurons['k'] = {int(k): v for k, v in k_raw.items()}
        safety_neurons['v'] = {int(k): v for k, v in v_raw.items()}
    except Exception as e:
        logger.error(f"Error parsing safety neurons file: {e}")
        raise
    
    logger.info(f"Loaded safety neurons from {output_file}")
    
    # Log summary with layer-wise breakdown
    logger.info(f"\n{'='*70}")
    logger.info(f"Safety Neurons Loaded - Detailed Breakdown")
    logger.info(f"{'='*70}")
    
    total_neurons = 0
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        module_total = sum(len(neurons) for neurons in safety_neurons[module_type].values())
        logger.info(f"  {module_type:12} : {module_total:4} neurons")
        total_neurons += module_total
        
        # Show which layers have neurons
        layers_with_neurons = [l for l in safety_neurons[module_type] if safety_neurons[module_type][l]]
        if layers_with_neurons:
            logger.info(f"    └─ Layers with neurons: {layers_with_neurons[:5]}{'...' if len(layers_with_neurons) > 5 else ''}")
    
    logger.info(f"\nTotal safety neurons: {total_neurons}")
    logger.info(f"{'='*70}\n")
    
    return safety_neurons


# =====================================================================
# Freeze Safety Neurons (반대로 작동: safety neuron만 freeze)
# =====================================================================
def setup_safety_neuron_freezing(model, safety_neurons, logger):
    """
    Freeze safety neurons and train only the remaining parameters.

    This is the REVERSE of sn_tune.py's setup_gradient_masking:
    - sn_tune.py: freeze all, train only safety neurons
    - This function: train all, freeze only safety neurons

    Returns frozen_param_specs: list of (param, indices, axis) used by
    SafetyNeuronRestoreCallback to undo weight-decay updates on safety neurons.
    """
    total_params = 0
    frozen_neuron_params = 0
    frozen_modules = {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0}
    frozen_param_specs = []  # (param, indices, axis) — for weight-decay bypass correction

    def _sanitize_indices(raw_indices, dim: int, module_name: str, layer_idx: int):
        """Convert possibly noisy neuron IDs to unique, in-range indices."""
        parsed = []
        dropped = 0
        for x in raw_indices:
            idx = None
            if isinstance(x, int):
                idx = x
            elif isinstance(x, str):
                s = x.strip()
                if s.lstrip("-").isdigit():
                    idx = int(s)
                else:
                    m = re.search(r"-?\d+", s)
                    if m:
                        idx = int(m.group(0))
            if idx is None:
                dropped += 1
                continue
            if 0 <= idx < dim:
                parsed.append(idx)
            else:
                dropped += 1
        uniq = sorted(set(parsed))
        if dropped > 0:
            logger.warning(
                f"[Index sanitize] layer={layer_idx}, module={module_name}, "
                f"kept={len(uniq)}, dropped={dropped}, dim={dim}"
            )
        return uniq

    def _make_zero_hook_rows(indices):
        """Zero out gradient rows (used for up_proj, q/k/v_proj)."""
        def hook(grad):
            grad = grad.clone()
            grad[indices, :] = 0.0
            return grad
        return hook

    def _make_zero_hook_cols(indices):
        """Zero out gradient columns (used for down_proj)."""
        def hook(grad):
            grad = grad.clone()
            grad[:, indices] = 0.0
            return grad
        return hook

    # Step 1: Enable gradients for all parameters by default
    for param in model.parameters():
        param.requires_grad = True

    # Step 2: Freeze safety neurons via gradient hooks + track for weight-decay restore
    for name, param in model.named_parameters():
        total_params += param.numel()
        parts = name.split('.')
        if len(parts) < 4 or parts[0] != 'model' or parts[1] != 'layers':
            continue
        try:
            layer_idx = int(parts[2])
        except ValueError:
            continue

        if 'mlp.up_proj.weight' in name:
            # up_proj: [intermediate_dim, hidden_dim] — neurons are rows
            neuron_indices = _sanitize_indices(
                safety_neurons['ffn_up'].get(layer_idx, []),
                param.shape[0], 'ffn_up', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['ffn_up'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'mlp.down_proj.weight' in name:
            # down_proj: [hidden_dim, intermediate_dim] — neurons are columns
            neuron_indices = _sanitize_indices(
                safety_neurons['ffn_down'].get(layer_idx, []),
                param.shape[1], 'ffn_down', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[0]
                frozen_modules['ffn_down'] += 1
                param.register_hook(_make_zero_hook_cols(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'cols'))

        elif 'self_attn.q_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['q'].get(layer_idx, []),
                param.shape[0], 'q', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['q'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'self_attn.k_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['k'].get(layer_idx, []),
                param.shape[0], 'k', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['k'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'self_attn.v_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['v'].get(layer_idx, []),
                param.shape[0], 'v', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['v'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n{'='*70}")
    logger.info(f"Safety Neuron Freezing Setup Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Frozen safety neuron parameters (effective): {frozen_neuron_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
    logger.info(f"Frozen safety neuron ratio: {frozen_neuron_params / total_params * 100:.4f}%")

    logger.info(f"\nLayers with frozen safety neurons:")
    for module_type, count in frozen_modules.items():
        if count > 0:
            logger.info(f"  {module_type:12} : {count} layers")

    logger.info(f"{'='*70}\n")
    return frozen_param_specs


# =====================================================================
# Safety Neuron Restore Callback
# =====================================================================
class SafetyNeuronRestoreCallback(TrainerCallback):
    """
    Restores safety neuron weights after every optimizer step.

    AdamW's weight-decay term (λθ) is applied independently of gradient hooks,
    so safety neuron weights would otherwise drift toward 0 even when the
    gradient hook zeros out the gradient signal.  This callback saves the
    initial (frozen) values at construction time and writes them back after
    every optimizer step, guaranteeing true parameter freezing.
    """

    def __init__(self, frozen_param_specs):
        # frozen_param_specs: list of (param, indices, axis)
        #   axis = "rows"  →  param[indices, :]  (up/q/k/v_proj)
        #   axis = "cols"  →  param[:, indices]  (down_proj)
        self._specs = frozen_param_specs
        self._frozen_vals = []
        for param, indices, axis in frozen_param_specs:
            with torch.no_grad():
                if axis == 'rows':
                    self._frozen_vals.append(param.data[indices, :].clone())
                else:
                    self._frozen_vals.append(param.data[:, indices].clone())

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each optimizer step — restore frozen weights."""
        for (param, indices, axis), frozen_val in zip(self._specs, self._frozen_vals):
            with torch.no_grad():
                if axis == 'rows':
                    param.data[indices, :] = frozen_val
                else:
                    param.data[:, indices] = frozen_val
        return control


def setup_logging(output_dir):
    """로깅 설정: 파일과 콘솔 모두에 출력"""
    log_dir = "./logs/safety_neuron_gsm8k"
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일 이름: 현재 날짜 및 시간
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_gsm8k_freeze_sn_{log_timestamp}.log")
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def main():
    """Main fine-tuning pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    
    # 로깅 설정
    logger, log_file = setup_logging(args.output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  🚀 GSM8K Fine-tuning with Safety Neuron Freezing")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")
    
    # Safety neurons 파일 존재 확인
    if not os.path.exists(args.safety_neurons_file):
        logger.error(f"Safety neurons file not found: {args.safety_neurons_file}")
        raise FileNotFoundError(f"Safety neurons file not found: {args.safety_neurons_file}")
    
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ SN-Tuned model: {args.model_path}")
    logger.info(f"   ├─ Safety neurons file: {args.safety_neurons_file}")
    logger.info(f"   ├─ Training samples: {args.num_train_samples}")
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(args.model_path) else 'base plain prompt'}")
    logger.info(f"   ├─ Batch size: {args.batch_size}")
    logger.info(f"   ├─ Gradient accumulation: {args.grad_accum}")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ Learning rate: {args.learning_rate}")
    logger.info(f"   ├─ Weight decay: {args.weight_decay}")
    logger.info(f"   ├─ Optimizer: {args.optim}")
    logger.info(f"   ├─ Warmup ratio: {args.warmup_ratio}")
    logger.info(f"   ├─ Max length: {args.max_length}")
    logger.info(f"   ├─ Dtype: bf16")
    logger.info(f"   ├─ Strategy: Freeze safety neurons, train others")
    logger.info(f"   └─ Output dir: {args.output_dir}\n")

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path

    # Load tokenizer
    logger.info(f"\n{'='*70}")
    logger.info(f"  [1/5] Loading Tokenizer")
    logger.info(f"{'='*70}\n")

    tokenizer = None
    try:
        logger.info("Attempting to load tokenizer (local files only)...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info("✓ Tokenizer loaded from local files")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer with local_files_only: {e}")
        logger.info("Attempting to load from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        logger.info("✓ Tokenizer loaded from HuggingFace Hub")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Tokenizer loaded successfully")
    logger.info(f"   ├─ Tokenizer type: {type(tokenizer).__name__}")
    logger.info(f"   ├─ Vocab size: {len(tokenizer)}")
    logger.info(f"   └─ Pad token: {tokenizer.pad_token}")

    # Load model with bf16
    logger.info(f"\n{'='*70}")
    logger.info(f"  [2/5] Loading Model (bf16)")
    logger.info(f"{'='*70}\n")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    
    model = None
    try:
        logger.info("Attempting to load model (local files only)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        logger.info("✓ Model loaded from local files")
    except Exception as e:
        logger.warning(f"Failed to load with local_files_only: {e}")
        logger.info("Attempting to load from HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="eager",
        )
        logger.info("✓ Model loaded from HuggingFace Hub")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   ├─ Model size: {total_params / 1e9:.2f}B parameters")
    logger.info(f"   ├─ Dtype: {model.dtype}")
    logger.info(f"   └─ Gradient checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")

    # Load safety neurons and setup freezing
    logger.info(f"\n{'='*70}")
    logger.info(f"  [3/5] Loading Safety Neurons and Setting up Freezing")
    logger.info(f"{'='*70}\n")
    
    safety_neurons = load_safety_neurons(args.safety_neurons_file, logger)
    frozen_param_specs = setup_safety_neuron_freezing(model, safety_neurons, logger)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Safety neuron freezing setup complete")
    logger.info(f"   ├─ Trainable: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")

    # Load dataset
    logger.info(f"\n{'='*70}")
    logger.info(f"  [4/5] Loading GSM8K Dataset")
    logger.info(f"{'='*70}\n")
    if args.task_data_path:
        # 로컬 JSON: {"question", "response"} 스키마 → HF gsm8k 의 answer 필드에 대응.
        from datasets import Dataset as _HFDataset
        with open(args.task_data_path, "r", encoding="utf-8") as f:
            _rows = json.load(f)
        train_ds = _HFDataset.from_list(
            [{"question": r["question"], "answer": r["response"]} for r in _rows]
        )
        logger.info(f"로컬 JSON 사용: {args.task_data_path}")
    else:
        train_ds = load_dataset(
            args.dataset_name,
            args.dataset_subset,
            split=args.train_split,
            cache_dir=args.cache_dir
        )
    train_ds = _select_first_n(train_ds, args.num_train_samples)

    eval_ds = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_ds = load_dataset(
            args.dataset_name, 
            args.dataset_subset, 
            split=args.eval_split,
            cache_dir=args.cache_dir
        )
        eval_ds = _select_first_n(eval_ds, args.num_eval_samples)
    
    logger.info(f"✅ Datasets loaded")
    logger.info(f"   ├─ Train: {len(train_ds)} samples")
    if eval_ds is not None:
        logger.info(f"   └─ Eval: {len(eval_ds)} samples")

    # Preprocess data
    logger.info(f"\n{'='*70}")
    logger.info(f"  [4.5/5] Preprocessing Data")
    logger.info(f"{'='*70}\n")
    
    def preprocess(ex):
        question = ex["question"]
        answer = ex["answer"]
        return tokenize_sft_example(question, answer, tokenizer, args.max_length, model_path)

    train_tok = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        num_proc=max(1, args.num_workers),
        desc="Tokenizing train",
    )

    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(
            preprocess,
            remove_columns=eval_ds.column_names,
            num_proc=max(1, args.num_workers),
            desc="Tokenizing eval",
        )
    
    logger.info(f"✅ Data preprocessed")

    # Training
    logger.info(f"\n{'='*70}")
    logger.info(f"  [5/5] Training with Trainer + AdamW 8-bit")
    logger.info(f"{'='*70}\n")
    
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
        report_to="none",
        remove_unused_columns=False,
        optim=args.optim,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SafetyNeuronRestoreCallback(frozen_param_specs)],
    )

    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"\n{'='*70}")
    logger.info(f"  Saving Fine-tuned Model")
    logger.info(f"{'='*70}\n")
    
    try:
        if args.no_timestamp_suffix:
            final_output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_dir = f"{args.output_dir}_{timestamp}"
        
        # 1️⃣ 메모리 정리 및 최적화
        logger.info("Step 1: Preparing model for saving...")
        gc.collect()
        torch.cuda.empty_cache()
        
        # 2️⃣ 모델을 CPU로 옮김
        logger.info("Step 2: Moving model to CPU for safe serialization...")
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3️⃣ 모델 저장
        logger.info("Step 3: Saving model weights...")
        logger.info(f"   ├─ Using safe_serialization=True (safetensors)")
        logger.info(f"   ├─ Output directory: {os.path.abspath(final_output_dir)}")
        
        model.save_pretrained(
            final_output_dir,
            safe_serialization=True,
            max_shard_size="4GB",
            push_to_hub=False,
        )
        logger.info(f"   └─ ✅ Model weights saved successfully")
        
        # 4️⃣ Tokenizer 저장
        logger.info("Step 4: Saving tokenizer...")
        tokenizer.save_pretrained(
            final_output_dir,
            safe_serialization=True
        )
        logger.info(f"   └─ ✅ Tokenizer saved")
        
        # 5️⃣ Config 저장
        logger.info("Step 5: Saving model config and generation settings...")
        model.config.save_pretrained(final_output_dir)
        if hasattr(model, 'generation_config'):
            model.generation_config.save_pretrained(final_output_dir)
        logger.info(f"   └─ ✅ Configs saved")
        
        # 6️⃣ 저장 검증
        logger.info("Step 6: Verifying saved model integrity...")
        required_files = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
        missing_files = []
        for fname in required_files:
            fpath = os.path.join(final_output_dir, fname)
            if not os.path.exists(fpath):
                missing_files.append(fname)
            else:
                fsize = os.path.getsize(fpath) / 1024
                logger.info(f"   ├─ {fname}: {fsize:.2f} KB ✅")
        
        if missing_files:
            raise FileNotFoundError(f"Missing/corrupted files: {missing_files}")
        
        # 모델 파일 존재 확인
        model_files = [f for f in os.listdir(final_output_dir) 
                      if f.endswith('.safetensors')]
        if not model_files:
            raise FileNotFoundError("No safetensors files found after save!")
        
        logger.info(f"   ├─ ✅ Found {len(model_files)} model shard file(s)")
        
        # 7️⃣ 파일 크기 로깅
        logger.info(f"\n📦 Saved files:")
        total_size = 0
        for fname in sorted(os.listdir(final_output_dir)):
            fpath = os.path.join(final_output_dir, fname)
            if os.path.isfile(fpath):
                fsize = os.path.getsize(fpath)
                total_size += fsize
                logger.info(f"   ├─ {fname}: {fsize/1e9:.2f} GB")
                    
        logger.info(f"   └─ Total size: {total_size/1e9:.2f} GB ✅")
        
        # 8️⃣ 최종 검증
        logger.info(f"\nStep 7: Final verification - attempting to load saved model...")
        try:
            test_tokenizer = AutoTokenizer.from_pretrained(final_output_dir)
            test_model = AutoModelForCausalLM.from_pretrained(
                final_output_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                attn_implementation="eager",
            )
            del test_tokenizer
            del test_model
            gc.collect()
            logger.info(f"   └─ ✅ Model verified successfully!")
        except Exception as load_err:
            logger.error(f"   └─ ❌ Failed to load saved model: {load_err}")
            logger.error(f"      This may indicate corruption - please verify manually")
            raise
        
        logger.info(f"\n✅✅✅ Fine-tuned model saved and verified successfully!")
        logger.info(f"   Output directory: {os.path.abspath(final_output_dir)}")
        logger.info(f"   Total size: {total_size/1e9:.2f} GB")
        logger.info(f"   Status: ✅ READY FOR EVALUATION")
        
    except Exception as e:
        logger.error(f"\n❌❌❌ CRITICAL ERROR during model saving: {e}")
        logger.error(f"   {type(e).__name__}: {str(e)}")
        logger.error(f"   Output directory may be incomplete: {final_output_dir}")
        logger.error(f"   Please check the directory contents before using this model!")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Save training config
    logger.info(f"\nSaving training configuration...")
    config = {
        'base_model': args.model_path,
        'fine_tuning_type': 'GSM8K Fine-tuning with Safety Neuron Freezing',
        'safety_neurons_file': args.safety_neurons_file,
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
        'optimizer': args.optim,
        'gradient_checkpointing': args.gradient_checkpointing,
        'dtype': 'bf16',
        'trainer_type': 'Trainer',
        'strategy': 'Freeze safety neurons, train others',
    }
    
    config_path = os.path.join(final_output_dir, 'finetune_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Config saved to: {config_path}")

    if args.model_dir_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.model_dir_file)) or ".", exist_ok=True)
        with open(args.model_dir_file, "w", encoding="utf-8") as f:
            f.write(os.path.abspath(final_output_dir) + "\n")
        logger.info(f"✅ 최종 경로를 기록: {args.model_dir_file}")

    if args.upload_name:
        logger.info(f"\nStarting upload to Hugging Face: {args.upload_name}")
        try:
            # 원본은 Safety-Neuron 쪽 upload_sn_tuned_model.py 를 import 했다.
            # 그 파일은 들여오지 않았고, 저장소 규칙대로 hub API 를 직접 쓴다.
            # ⚠️ chat_template.jinja 는 tokenizer_config.json 과 **별도 파일**이다.
            #    폴더째 올리지 않으면 허브 사본이 chat_template=None 으로 로드되어
            #    평가 때 학습과 다른 프롬프트가 렌더된다. 이 저장소에서 두 번 당한 함정이다.
            from huggingface_hub import HfApi

            api = HfApi(token=args.hf_token)
            api.create_repo(args.upload_name, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=final_output_dir, repo_id=args.upload_name,
                              repo_type="model")
            logger.info(f"✅ Upload completed: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.error("모델은 로컬에 저장돼 있다. scripts/revision/upload_and_prune.py 로 따로 올려라.")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  ✅ Fine-tuning Complete!")
    logger.info(f"{'='*70}\n")

if __name__ == '__main__':
    main()
