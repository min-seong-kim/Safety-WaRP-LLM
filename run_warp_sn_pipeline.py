"""
WaRP-SN Pipeline Entry Point

Full pipeline:
  1. Load model + tokenizer (standard HuggingFace — no patched modeling_llama)
  2. Load Phase-1 safety basis (U matrices per layer/type)
  3. Detect safety neurons in WaRP-reparameterized space
     → saved to <output_dir>/warp_safety_neurons_<timestamp>.txt
  4. SN-Tune: fine-tune only detected safety columns of basis_coeff on Circuit Breakers
     → saved to <output_dir>/warp_sn_tuned_<lr>_<timestamp>/  (standard HF checkpoint)

Usage example
-------------
  python run_warp_sn_pipeline.py \\
      --model_name meta-llama/Llama-2-7b-hf \\
      --basis_dir ./checkpoints/phase1_20260503_120000/basis \\
      --dataset_file ./data/circuit_breakers_train.json \\
      --output_dir ./output/warp_sn_llama2_7b \\
      --layer_types ffn_up,ffn_down,attn_q,attn_k,attn_v \\
      --num_prompts 4994 \\
      --top_k_ffn 300 \\
      --top_k_attn 50 \\
      --learning_rate 5e-5 \\
      --num_epochs 3

Skip detection (reuse existing neuron file)
-------------------------------------------
  python run_warp_sn_pipeline.py \\
      --model_name meta-llama/Llama-2-7b-hf \\
      --basis_dir ./checkpoints/phase1_20260503_120000/basis \\
      --dataset_file ./data/circuit_breakers_train.json \\
      --output_dir ./output/warp_sn_llama2_7b \\
      --existing_neuron_file ./output/warp_sn_llama2_7b/warp_safety_neurons_20260503_123456.txt

Skip SN-Tune (detection only)
------------------------------
  python run_warp_sn_pipeline.py  ...  --detection_only
"""

import os
import sys
import json
import logging
import argparse
import random
from datetime import datetime
from typing import List, Optional

# ── make Safety-WaRP-LLM the package root ──────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.warp_sn_detection import WaRPSNDetector
from models.warp_sn_tune import WaRPSNTuner

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, prefix: str = "warp_sn") -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{ts}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)

    return log_file


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="WaRP-SN Pipeline: rotation → detection → SN-Tune"
    )

    # --- model ---
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model name or local path")

    # --- basis ---
    p.add_argument("--basis_dir", type=str, required=True,
                   help="Phase-1 output basis directory "
                        "(contains ffn_up/, attn_q/, … sub-dirs)")

    # --- data ---
    p.add_argument("--dataset_file", type=str,
                   default="./data/circuit_breakers_train.json",
                   help="Circuit Breakers JSON file for both detection and SN-Tune")

    # --- output ---
    p.add_argument("--output_dir", type=str, default="./output/warp_sn",
                   help="Root output directory for neurons file + fine-tuned model")

    # --- layer selection ---
    p.add_argument("--layer_types", type=str,
                   default="ffn_up,ffn_down,attn_q,attn_k,attn_v",
                   help="Comma-separated layer types to reparameterize and detect "
                        "(supported: ffn_up, ffn_down, attn_q, attn_k, attn_v)")

    # --- detection ---
    p.add_argument("--num_prompts", type=int, default=4994,
                   help="Number of harmful prompts used for detection (intersection)")
    p.add_argument("--top_k_ffn", type=int, default=2000,
                   help="Per-layer top-k for FFN modules during detection (targets ~1%% safety params)")
    p.add_argument("--top_k_attn", type=int, default=350,
                   help="Per-layer top-k for attention modules during detection (targets ~1%% safety params)")
    p.add_argument("--freq_threshold", type=float, default=1.0,
                   help="Fraction of prompts a direction must appear in top-k to be selected as safety neuron. "
                        "1.0 = exact intersection (original behaviour); <1.0 = frequency-based relaxation")
    p.add_argument("--existing_neuron_file", type=str, default=None,
                   help="Skip detection: load WaRP-space safety neurons from this file")
    p.add_argument("--detection_only", action="store_true",
                   help="Run detection only; skip SN-Tune")

    # --- SN-Tune training ---
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=4994,
                   help="Max circuit_breakers samples for SN-Tune training")
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    # --- misc ---
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=112)
    p.add_argument("--upload_name", type=str, default=None,
                   help="Optional HuggingFace repo to upload final model (user/repo-name)")
    p.add_argument("--hf_token", type=str, default=None)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_instruct_model(name: str) -> bool:
    n = name.lower()
    return "instruct" in n or "chat" in n


def load_prompts(dataset_file: str, num_prompts: int) -> List[str]:
    """Load harmful prompts from Circuit Breakers JSON."""
    with open(dataset_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) > num_prompts:
        records = records[:num_prompts]
    prompts = [item.get("prompt", "") for item in records if item.get("prompt")]
    logger.info(f"  Loaded {len(prompts)} prompts from {dataset_file}")
    return prompts


def load_model_and_tokenizer(model_name: str, dtype_str: str, gpu: int):
    logger.info(f"  Loading model: {model_name}")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype_str]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": gpu},
    )
    model.eval()
    logger.info(f"  ✓ Model loaded on cuda:{gpu}, dtype={torch_dtype}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    # ── seed ────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── logging ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        prefix="warp_sn_pipeline",
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_types = [lt.strip() for lt in args.layer_types.split(",") if lt.strip()]
    _is_instruct = is_instruct_model(args.model_name)

    logger.info("=" * 70)
    logger.info("WaRP-SN Pipeline")
    logger.info("=" * 70)
    logger.info(f"  model_name       : {args.model_name}")
    logger.info(f"  basis_dir        : {args.basis_dir}")
    logger.info(f"  dataset_file     : {args.dataset_file}")
    logger.info(f"  output_dir       : {args.output_dir}")
    logger.info(f"  layer_types      : {layer_types}")
    logger.info(f"  is_instruct      : {_is_instruct}")
    logger.info(f"  dtype            : {args.dtype}")
    logger.info(f"  gpu              : {args.gpu}")
    logger.info(f"  log_file         : {log_file}")
    logger.info("=" * 70)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── Step 0: load model ──────────────────────────────────────────────────
    logger.info("\n[1/4] Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.dtype, gpu=0)
    num_layers = model.config.num_hidden_layers
    logger.info(f"  num_hidden_layers: {num_layers}")

    # ── Step 1: detect or load ──────────────────────────────────────────────
    if args.existing_neuron_file:
        logger.info(f"\n[2/4] Loading existing WaRP-space safety neurons")
        logger.info(f"  file: {args.existing_neuron_file}")
        safety_neurons = WaRPSNDetector.load_safety_neurons(args.existing_neuron_file)
        neuron_file = args.existing_neuron_file

        # Still need basis_data for SN-Tune reparameterization
        logger.info("\n[2b/4] Loading basis for SN-Tune reparameterization")
        detector_tmp = WaRPSNDetector(
            model=model,
            tokenizer=tokenizer,
            basis_dir=args.basis_dir,
            layer_types=layer_types,
            num_layers=num_layers,
            is_instruct=_is_instruct,
        )
        detector_tmp.load_basis()
        basis_data = detector_tmp.basis_data
        del detector_tmp

    else:
        logger.info("\n[2/4] WaRP-space safety neuron detection")
        logger.info(f"  num_prompts : {args.num_prompts}")
        logger.info(f"  top_k_ffn   : {args.top_k_ffn}")
        logger.info(f"  top_k_attn  : {args.top_k_attn}")

        detector = WaRPSNDetector(
            model=model,
            tokenizer=tokenizer,
            basis_dir=args.basis_dir,
            layer_types=layer_types,
            num_layers=num_layers,
            is_instruct=_is_instruct,
            top_k_ffn=args.top_k_ffn,
            top_k_attn=args.top_k_attn,
            max_seq_len=args.max_seq_len,
            freq_threshold=args.freq_threshold,
        )

        # 2a: load basis
        logger.info("\n  [2a] Loading Phase-1 safety basis")
        detector.load_basis()
        basis_data = detector.basis_data

        # 2b: reparameterize (basis_coeff = W @ U, flag=False)
        logger.info("\n  [2b] Reparameterizing weights to WaRP space")
        detector.reparameterize()

        # 2c: load harmful prompts
        if not os.path.exists(args.dataset_file):
            logger.error(f"Dataset file not found: {args.dataset_file}")
            sys.exit(1)
        prompts = load_prompts(args.dataset_file, args.num_prompts)

        # 2d: detect
        logger.info("\n  [2c] Running WaRP-space detection")
        safety_neurons = detector.detect(prompts)

        # 2e: save
        neuron_file = os.path.join(
            args.output_dir,
            f"warp_safety_neurons_{ts}.txt",
        )
        WaRPSNDetector.save_safety_neurons(safety_neurons, neuron_file)
        logger.info(f"\n  ✓ Safety neurons saved → {neuron_file}")

        # Log total counts
        total = sum(
            len(v) for d in safety_neurons.values() for v in d.values()
        )
        logger.info(f"  Total WaRP-space safety neurons: {total:,}")

        # Reload a fresh model for SN-Tune (detection may have modified module types)
        logger.info("\n  Reloading model for SN-Tune (fresh copy)")
        del model
        torch.cuda.empty_cache()
        model, tokenizer = load_model_and_tokenizer(args.model_name, args.dtype, gpu=0)

    if args.detection_only:
        logger.info("\n[--detection_only] Skipping SN-Tune. Done.")
        logger.info(f"  Neuron file: {neuron_file}")
        return

    # ── Step 2: SN-Tune ─────────────────────────────────────────────────────
    logger.info("\n[3/4] WaRP-SN-Tune (safety fine-tuning in WaRP space)")

    lr_str = f"lr{args.learning_rate:.0e}".replace("-0", "-")
    tuned_dir = os.path.join(args.output_dir, f"warp_sn_tuned_{lr_str}_{ts}")

    tuner = WaRPSNTuner(
        model=model,
        tokenizer=tokenizer,
        basis_data=basis_data,
        safety_neurons=safety_neurons,
        layer_types=layer_types,
        num_layers=num_layers,
        is_instruct=_is_instruct,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        warmup_ratio=args.warmup_ratio,
    )

    # 3a: reparameterize (flag=True for WaRP training forward)
    logger.info("\n  [3a] Reparameterizing for training (flag=True)")
    tuner.reparameterize()

    # 3b: setup gradient masking (only safety columns get gradient)
    logger.info("\n  [3b] Setting up column-wise gradient masking")
    tuner.setup_gradient_masking()

    # 3c: train
    logger.info("\n  [3c] Training on Circuit Breakers")
    tuner.train(dataset_path=args.dataset_file)

    # ── Step 3: save ─────────────────────────────────────────────────────────
    logger.info(f"\n[4/4] Saving fine-tuned model → {tuned_dir}")
    tuner.save(tuned_dir, tokenizer=tokenizer)

    # ── Step 4: optional HF upload ────────────────────────────────────────
    if args.upload_name:
        logger.info(f"\nUploading to HuggingFace: {args.upload_name}")
        try:
            from transformers import AutoModelForCausalLM as _M
            _m = _M.from_pretrained(tuned_dir, torch_dtype=torch.bfloat16)
            _m.push_to_hub(args.upload_name, token=args.hf_token)
            tokenizer.push_to_hub(args.upload_name, token=args.hf_token)
            logger.info(f"  ✓ Uploaded → https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"  Upload failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("WaRP-SN Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"  Safety neuron file : {neuron_file}")
    logger.info(f"  Fine-tuned model   : {tuned_dir}")
    logger.info(f"  Log file           : {log_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
