"""통합 runner: GSM8K 위 Standard LoRA / Original-space Projected LoRA / WSR-LoRA(element-wise) 비교.

세 방법은 동일 시작 checkpoint·GSM8K data·seed·rank·trainable param 수를 쓰고,
차이는 증분이 허용되는 좌표계/제약뿐이다 (wsr_lora_comparison.md 참조).

  lora                     : ΔW = s·BA
  original_projected_lora  : ΔW = s·BA(I−EEᵀ)          (A[:, safe_cols]=0, optimizer.step 후 재투영)
  wsr_lora                 : ΔW = [(1−M)∘(s·BA)] Uᵀ    (basis 공간 element freeze, forward 사전제약)

저장은 dense: lora/orig → merge_and_unload, wsr → restore_wsr_lora_to_linear. HF push.
"""
import argparse
import json
import logging
import os
import sys

# ⚠️ gsm8k_eval.finetune_gsm8k_full_params 는 import 시점에
#    os.environ["CUDA_VISIBLE_DEVICES"]="2,3" 을 하드코딩한다. 우리가 shell 로 지정한
#    device 를 덮어쓰지 않도록, import 전에 캡처해 import 후 복원한다.
_INTENDED_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, TrainerCallback)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_eval.finetune_gsm8k_full_params import (  # noqa: E402
    tokenize_sft_example, DataCollatorForCausalLMWithPadding, _select_first_n)

# gsm8k_eval import 가 덮어쓴 CUDA_VISIBLE_DEVICES 복원 (torch cuda init 전에 수행)
if _INTENDED_CVD is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _INTENDED_CVD
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
from models.lora_wsr_elementwise import (  # noqa: E402
    switch_to_wsr_lora, mark_only_lora_trainable, restore_wsr_lora_to_linear)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lora_runner")

PROJ_TO_LT = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
              "up_proj": "ffn_up", "down_proj": "ffn_down", "gate_proj": "ffn_gate", "o_proj": "attn_o"}
LT_TO_PROJ = {v: k for k, v in PROJ_TO_LT.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["lora", "original_projected_lora", "wsr_lora"])
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--safety_data_path", default="./data/circuit_breakers_train.json")
    ap.add_argument("--basis_dir", default=None)          # wsr_lora
    ap.add_argument("--mask_dir", default=None)           # wsr_lora
    ap.add_argument("--safecols_dir", default=None)       # original_projected_lora
    ap.add_argument("--keep_ratio", type=float, default=0.1)
    ap.add_argument("--direction_keep_ratio", type=float, default=0.1)
    ap.add_argument("--layer_type", default="attn_q,attn_k,attn_v,ffn_up,ffn_down")
    ap.add_argument("--target_layers", default="all")
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dataset_name", default="openai/gsm8k")
    ap.add_argument("--dataset_subset", default="main")
    ap.add_argument("--gsm8k_samples", type=int, default=0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_merged_model", action="store_true")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id", default=None)
    return ap.parse_args()


def _resolve_layers(num, spec):
    if spec == "all":
        return list(range(num))
    if "-" in spec:
        s, e = map(int, spec.split("-"))
        return list(range(s, e + 1))
    return [int(spec)]


def _name_to_key(name):
    parts = name.split(".")
    if "layers" not in parts:
        return None
    li = int(parts[parts.index("layers") + 1])
    for p, lt in PROJ_TO_LT.items():
        if p in parts:
            return (li, lt)
    return None


# ───────────────── WSR-LoRA basis/mask 로드 ─────────────────
def _load_basis(basis_dir, layer_types):
    basis = {}
    for lt in layer_types:
        d = os.path.join(basis_dir, lt)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("layer_") and f.endswith("_svd.pt"):
                li = int(f.split("_")[1])
                basis[(li, lt)] = torch.load(os.path.join(d, f), map_location="cpu")["U"]
    return basis


def _load_masks(mask_dir, layer_types):
    masks = {}
    for lt in layer_types:
        d = os.path.join(mask_dir, lt)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("layer_") and f.endswith("_mask.pt"):
                li = int(f.split("_")[1])
                masks[(li, lt)] = torch.load(os.path.join(d, f), weights_only=False)["mask"]
    return masks


def _load_safecols(safecols_dir, layer_types):
    sc = {}
    for lt in layer_types:
        d = os.path.join(safecols_dir, lt)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("layer_") and f.endswith("_safecols.pt"):
                li = int(f.split("_")[1])
                sc[(li, lt)] = torch.load(os.path.join(d, f))["safe_cols"]
    return sc


# ───────────────── original_projected 투영 콜백 ─────────────────
class ProjectionCallback(TrainerCallback):
    """optimizer.step 후 각 LoRA lora_A 의 safe_cols 열을 0으로 재투영."""
    def __init__(self, model, safecols):
        self.model = model
        self.safecols = safecols
        self.count = 0

    @torch.no_grad()
    def project(self):
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            key = _name_to_key(name)
            if key is None or key not in self.safecols:
                continue
            cols = self.safecols[key].to(next(iter(module.lora_A.values())).weight.device)
            for adapter_name, A in module.lora_A.items():
                A.weight.data[:, cols] = 0.0
        self.count += 1

    def on_train_begin(self, args, state, control, **kwargs):
        self.project()

    def on_step_end(self, args, state, control, **kwargs):
        self.project()


def build_gsm8k(tokenizer, args):
    ds = load_dataset(args.dataset_name, args.dataset_subset, split="train")
    if args.gsm8k_samples > 0:
        ds = _select_first_n(ds, args.gsm8k_samples)

    def preprocess(ex):
        return tokenize_sft_example(ex["question"], ex["answer"], tokenizer, args.max_length, args.model_name)

    tok_ds = ds.map(preprocess, remove_columns=ds.column_names, desc="tokenizing gsm8k")
    return tok_ds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"=== method={args.method} lr={args.learning_rate} r={args.lora_r} ===")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    layer_types = [x.strip() for x in args.layer_type.split(",")]
    target_modules = [x.strip() for x in args.target_modules.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map={"": 0})
    model.config.use_cache = False

    projection_cb = None

    # ───────────── method setup ─────────────
    if args.method == "wsr_lora":
        if not args.basis_dir or not args.mask_dir:
            raise ValueError("wsr_lora requires --basis_dir and --mask_dir (no fallback)")
        basis = _load_basis(args.basis_dir, layer_types)
        masks = _load_masks(args.mask_dir, layer_types)
        target_layers = _resolve_layers(len(model.model.layers), args.target_layers)
        converted = switch_to_wsr_lora(model, layer_types, args.target_layers,
                                       r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        n_set = 0
        for key, mod in converted.items():
            if key not in basis or key not in masks:
                raise ValueError(f"missing basis/mask for {key}")
            U = basis[key]
            M = torch.as_tensor(masks[key])
            # 검증: U (n×n), mask (m×n)
            assert U.shape[0] == mod.in_features, f"{key}: U {tuple(U.shape)} vs in {mod.in_features}"
            assert tuple(M.shape) == (mod.out_features, mod.in_features), f"{key}: mask {tuple(M.shape)}"
            mod.set_basis_and_mask(U, M)
            n_set += 1
        logger.info(f"✓ WSR-LoRA basis/mask set for {n_set} modules")
        model = model.to(0)
        trainable = mark_only_lora_trainable(model)
        logger.info(f"trainable params (WSR-LoRA): {trainable:,}")
    else:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        if args.method == "original_projected_lora":
            if not args.safecols_dir:
                raise ValueError("original_projected_lora requires --safecols_dir (no fallback)")
            safecols = _load_safecols(args.safecols_dir, layer_types)
            if not safecols:
                raise ValueError(f"no safe_cols found in {args.safecols_dir}")
            projection_cb = ProjectionCallback(model, safecols)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_ds = build_gsm8k(tok, args)
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tok)

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
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator,
                      callbacks=[projection_cb] if projection_cb else None)
    trainer.train()
    logger.info("✓ training done")

    # ───────────── dense 저장 ─────────────
    if projection_cb is not None:
        projection_cb.project()  # 마지막 재투영

    if args.method == "wsr_lora":
        restore_wsr_lora_to_linear(model)
        merged = model
    else:
        merged = model.merge_and_unload()

    merged_dir = os.path.join(args.output_dir, "merged_model")
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(merged_dir)
    logger.info(f"✓ merged model saved: {merged_dir}")

    # sanity generation
    try:
        merged.eval()
        q = "Natalia sold clips to 48 friends in April, and half as many in May. How many total?"
        msgs = [{"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii = tok(text, return_tensors="pt").to(next(merged.parameters()).device)
        with torch.no_grad():
            out = merged.generate(**ii, max_new_tokens=64, do_sample=False)
        logger.info("sanity gen: " + tok.decode(out[0][ii["input_ids"].shape[1]:], skip_special_tokens=True)[:200])
    except Exception as e:
        logger.warning(f"sanity gen failed: {e}")

    summary = {"method": args.method, "model_name": args.model_name, "lr": args.learning_rate,
               "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "keep_ratio": args.keep_ratio,
               "direction_keep_ratio": args.direction_keep_ratio, "epochs": args.epochs,
               "merged_dir": merged_dir, "hf_repo_id": args.hf_repo_id,
               "projection_calls": projection_cb.count if projection_cb else None}
    json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--push_to_hub requires --hf_repo_id")
        logger.info(f"pushing to hub: {args.hf_repo_id}")
        merged.push_to_hub(args.hf_repo_id)
        tok.push_to_hub(args.hf_repo_id)
        logger.info(f"✓ pushed: https://huggingface.co/{args.hf_repo_id}")

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
