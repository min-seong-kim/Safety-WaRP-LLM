"""
Generate WaRP-compatible masks from a Safety Neuron detection file.

Safety Neuron file format (5 lines, each a JSON/ast-parseable dict):
  Line 0: ffn_up  neurons  {layer_idx: [output_neuron_indices]}
  Line 1: ffn_down neurons  {layer_idx: [input_neuron_indices]}
  Line 2: q_proj  neurons   {layer_idx: [output_neuron_indices]}
  Line 3: k_proj  neurons   {layer_idx: [output_neuron_indices]}
  Line 4: v_proj  neurons   {layer_idx: [output_neuron_indices]}

Mask creation rules (in WaRP basis_coeff space):
  ffn_up   / ffn_gate : mask[j, :] = True  (freeze row j = output neuron j)
  ffn_down            : mask[:, j] = True  (freeze col j = input  neuron j, approx)
  attn_q / attn_k / attn_v : mask[j, :] = True  (freeze row j = output neuron j)

Saved format is identical to Phase 2 output so Phase 3 can load it directly.
"""

import ast
import json
import logging
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate WaRP masks from safety neuron file (Phase-2 replacement)"
    )
    p.add_argument("--safety_neurons_file", type=str, required=True,
                   help="Path to safety neurons detection output .txt file")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Model directory (HuggingFace) used to read config/weight shapes")
    p.add_argument("--layer_type", type=str,
                   default="attn_q,attn_k,attn_v,ffn_up,ffn_down",
                   help="Comma-separated layer types to process")
    p.add_argument("--target_layers", type=str, default="all",
                   help="Target layer indices: 'all' or range like '0-31' or '0,1,2'")
    p.add_argument("--output_dir", type=str, default="./checkpoints",
                   help="Root output dir; masks saved under <output_dir>/phase2_sn_<ts>/checkpoints/masks/")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def load_safety_neurons(path: str, logger: logging.Logger) -> dict:
    """Read the 5-line safety neuron file and return a dict keyed by layer type."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError(
            f"Expected at least 5 lines in {path}, got {len(lines)}"
        )

    # Each line is a Python dict literal
    ffn_up_raw   = ast.literal_eval(lines[0].strip())
    ffn_down_raw = ast.literal_eval(lines[1].strip())
    q_raw        = ast.literal_eval(lines[2].strip())
    k_raw        = ast.literal_eval(lines[3].strip())
    v_raw        = ast.literal_eval(lines[4].strip())

    sn = {
        "ffn_up":   {int(k): v for k, v in ffn_up_raw.items()},
        "ffn_down": {int(k): v for k, v in ffn_down_raw.items()},
        "attn_q":   {int(k): v for k, v in q_raw.items()},
        "attn_k":   {int(k): v for k, v in k_raw.items()},
        "attn_v":   {int(k): v for k, v in v_raw.items()},
    }

    logger.info(f"Loaded safety neurons from {path}")
    for module_type, by_layer in sn.items():
        total = sum(len(v) for v in by_layer.values())
        logger.info(f"  {module_type:12}: {total:5d} neurons across {len(by_layer)} layers")
    return sn


def get_weight_shape(config, layer_type: str):
    """
    Return (out_dim, in_dim) for the given layer type based on model config.

    For GQA models (e.g. LLaMA-3), k_proj and v_proj have smaller out_dim.
    """
    hidden_size      = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads        = config.num_attention_heads
    num_kv_heads     = getattr(config, "num_key_value_heads", num_heads)
    head_dim         = hidden_size // num_heads

    shape_map = {
        "attn_q":   (num_heads    * head_dim, hidden_size),  # q_proj
        "attn_k":   (num_kv_heads * head_dim, hidden_size),  # k_proj
        "attn_v":   (num_kv_heads * head_dim, hidden_size),  # v_proj
        "ffn_gate": (intermediate_size, hidden_size),         # gate_proj
        "ffn_up":   (intermediate_size, hidden_size),         # up_proj
        "ffn_down": (hidden_size, intermediate_size),         # down_proj
    }
    return shape_map.get(layer_type)


def parse_target_layers(target_str: str, num_layers: int):
    if target_str.strip().lower() == "all":
        return list(range(num_layers))
    indices = []
    for part in target_str.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-")
            indices.extend(range(int(s), int(e) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


# ──────────────────────────────────────────────────────────────────────────────
# Mask generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_masks(config, safety_neurons: dict, layer_types, target_indices, logger):
    """
    Build numpy bool masks for every (layer_idx, layer_type) combination.

    Mask convention (same as Phase 2):
      mask[i, j] = True  → frozen  (gradient detached in WaRP forward)
      mask[i, j] = False → trainable

    Mapping:
      ffn_up / ffn_gate : safety neuron j ↔ output neuron j → freeze entire row j
      ffn_down          : safety neuron j ↔ input  neuron j → freeze entire col j (approx in rotated space)
      attn_q/k/v        : safety neuron j ↔ output neuron j → freeze entire row j
    """
    # ffn_gate uses the same intermediate neurons as ffn_up
    sn_source = {
        "attn_q":   "attn_q",
        "attn_k":   "attn_k",
        "attn_v":   "attn_v",
        "ffn_gate": "ffn_up",   # gate_proj shares intermediate dim with up_proj
        "ffn_up":   "ffn_up",
        "ffn_down": "ffn_down",
    }

    masks = {}
    for layer_type in layer_types:
        shape = get_weight_shape(config, layer_type)
        if shape is None:
            logger.warning(f"Skipping unknown layer type: {layer_type}")
            continue

        out_dim, in_dim = shape
        sn_key = sn_source.get(layer_type, layer_type)
        neurons_by_layer = safety_neurons.get(sn_key, {})

        for layer_idx in target_indices:
            mask = np.zeros((out_dim, in_dim), dtype=bool)
            neuron_indices = neurons_by_layer.get(layer_idx, [])

            if layer_type in ("attn_q", "attn_k", "attn_v", "ffn_up", "ffn_gate"):
                # Freeze entire ROW for each safety output neuron
                for j in neuron_indices:
                    if 0 <= j < out_dim:
                        mask[j, :] = True
                    # indices beyond out_dim silently ignored (e.g. attn q/k/v vs MLP indices)

            elif layer_type == "ffn_down":
                # Freeze entire COLUMN for each safety input neuron (approx in rotated space)
                for j in neuron_indices:
                    if 0 <= j < in_dim:
                        mask[:, j] = True

            frozen = int(mask.sum())
            total  = int(mask.size)
            logger.info(
                f"  Layer {layer_idx:2d}  {layer_type:10s}: "
                f"{len(neuron_indices)} safety neurons → {frozen}/{total} coeff frozen "
                f"({frozen / total * 100:.2f}%)"
            )

            masks[(layer_idx, layer_type)] = mask

    return masks


def save_masks(masks: dict, layer_types, output_dir: str, safety_neurons_file: str, logger):
    """Save masks in the same directory structure as Phase 2."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    masks_dir = os.path.join(
        output_dir,
        f"phase2_sn_{timestamp}",
        "checkpoints",
        "masks",
    )
    os.makedirs(masks_dir, exist_ok=True)

    for (layer_idx, layer_type), mask in masks.items():
        layer_type_dir = os.path.join(masks_dir, layer_type)
        os.makedirs(layer_type_dir, exist_ok=True)
        mask_path = os.path.join(layer_type_dir, f"layer_{layer_idx:02d}_mask.pt")
        torch.save({"mask": mask}, mask_path)

    metadata = {
        "phase": 2,
        "keep_ratio": None,
        "masking_strategy": "safety_neurons",
        "safety_neurons_file": str(safety_neurons_file),
        "layer_types": layer_types,
        "timestamp": timestamp,
    }
    with open(os.path.join(masks_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Masks saved: {len(masks)} files → {masks_dir}")
    return masks_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ── 1. Load safety neurons ──
    safety_neurons = load_safety_neurons(args.safety_neurons_file, logger)

    # ── 2. Load model config ──
    from transformers import AutoConfig
    logger.info(f"Loading model config from {args.model_dir}")
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    num_layers   = config.num_hidden_layers
    logger.info(
        f"Model: {num_layers} layers, hidden={config.hidden_size}, "
        f"intermediate={config.intermediate_size}"
    )

    # ── 3. Parse layer types & target layers ──
    layer_types    = [lt.strip() for lt in args.layer_type.split(",")]
    target_indices = parse_target_layers(args.target_layers, num_layers)
    logger.info(f"Layer types   : {layer_types}")
    logger.info(f"Target layers : {target_indices}")

    # ── 4. Generate masks ──
    logger.info("Generating masks...")
    masks = generate_masks(config, safety_neurons, layer_types, target_indices, logger)

    # ── 5. Save ──
    masks_dir = save_masks(
        masks, layer_types, args.output_dir, args.safety_neurons_file, logger
    )

    # Print to stdout so shell script can capture with $()
    print(masks_dir)


if __name__ == "__main__":
    main()
