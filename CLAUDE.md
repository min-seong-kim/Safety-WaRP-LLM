# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **Safety-WaRP-LLM**: applying Weight space Rotation Process (WaRP) to LLM safety
alignment. The thesis is that a safety-tuned model can be fine-tuned on a downstream/utility task **without losing
its safety behavior** by reparameterizing each weight matrix onto an SVD-derived orthonormal basis and freezing the
"important" (safety-relevant) coefficient directions while only updating the "flat" ones.

`WaRP.md` is the theory reference (the math behind the basis transform `W = V @ (basis_coeff) @ U`). Read it before
touching `models/warp_modules.py`. Code comments and docstrings are predominantly in Korean.

## Pipeline (the mental model)

Everything flows through four numbered phases. The main entry point is `train.py`, which dispatches on `--phase`:

- **Phase 0** (`models/phase0_SSFT.py`): full-parameter supervised safety fine-tuning of a base chat model →
  produces the "safety model" that all later phases build on. Often skipped by pointing `--phase0_model_dir` at a
  pre-trained HF checkpoint (e.g. `kmseong/llama2_7b-chat-Safety-FT-lr5e-5`).
- **Phase 1** (`models/phase1_basis.py`, `Phase1BasisBuilder`): collect activations on safety data
  (`circuit_breakers`) or utility data (`wikipedia`), build covariance `Φ @ Φ^T`, SVD → orthonormal basis per
  layer. Output: `checkpoints/phase1_TIMESTAMP/basis/layer_NN_svd.pt`.
- **Phase 2** (`models/phase2_*`): gradient-based importance scoring of basis coefficients on safety data →
  binary masks (`1` = important/freeze, `0` = flat/trainable). `--keep_ratio` controls the fraction kept as
  important. Output: `checkpoints/phase2_TIMESTAMP/checkpoints/masks/`.
- **Phase 3** (`models/phase3_*`): fine-tune on a downstream task with WaRP masking applied so safety directions
  stay frozen. Output: `checkpoints/phase3_TIMESTAMP/.../final_model`.

**Invariant: `--layer_type` and `--target_layers` MUST be identical across Phases 1, 2, and 3.** The basis, the
masks, and the training all index the same layers; a mismatch silently produces wrong results.

`run_phase{1,2,3}` in `train.py` each `import` a *variant* class chosen by CLI flags (e.g. `--original_space_mask`,
`--no_rotation`, `--two_mask`, `--non_freeze`, LoRA flags). The many `models/phase2_importance_*.py` /
`models/phase3_extra_learning_*.py` files are these variants — `_per_layer`, `_whole`, `_original_space`,
`_no_rotation`, `_non_freeze`, `_lora*`. When editing Phase 2/3 behavior, find which variant the relevant script's
flags select before editing.

## Running the pipeline

The integrated driver runs all phases for a keep_ratio × learning_rate sweep:

```bash
bash scripts/run_all_phases_integrated.sh
```

Edit the config block at the top of that script (it is the primary control surface):
- `PHASE0_MODEL` — HF path of the safety-tuned model.
- `PHASE1_BASIS_DIR_OVERRIDE` — set to an existing basis dir to **skip Phase 1**; empty string to recompute it.
- `PHASE3_DATASET` — one of `safety | gsm8k | metamath | math | agnews | medqa | mmlu`.
- `KEEP_RATIO_LIST`, `LR_LIST`, `LAYER_TYPE`, `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`.
- Sets `CUDA_VISIBLE_DEVICES` and `conda activate` near the top — **adjust the GPU index and conda env for the
  local machine**; the committed values are machine-specific (`conda activate hb`, a hardcoded `/home/yonsei_jong`
  conda path and dataset paths).

`scripts/` also holds single-phase runners (`run_phase1_basis.sh`, `run_phase2_importance.sh`,
`run_phase3_learning.sh`) and many experiment-specific variants (`run_phase23_original_space_mask*.sh`,
`run_warp_sn.sh`, `run_safe_lora_basis_rotation.sh`, `run_dual_importance_analysis.sh`, etc.).

Direct invocation of one phase (see `README.md` for full parameter docs):

```bash
python train.py --phase 1 --phase0_model_dir <hf_or_local> --safety_dataset circuit_breakers \
    --layer_type ffn_down --target_layers all --device cuda --dtype bfloat16
```

`mmlu` is special-cased in the integrated script: Phase 3 for MMLU runs `mmlu_eval/finetune_mmlu_full_params.py`
directly instead of `train.py`.

## Layer types

`--layer_type` accepts a comma-separated list of: `ffn_down`, `ffn_up`, `attn_q`, `attn_k`, `attn_v` (LLaMA module
names: `down_proj`, `up_proj`, `q_proj`, `k_proj`, `v_proj`). `--target_layers` accepts `all`, `early`, `middle`,
`late`, `last`, a single index (`31`), or a range (`0-5`).

## Alternative method: SN-Tune (`sn_tune/`)

A separate, self-contained baseline/comparison method ("Safety-Neuron Tune"), runnable as a module:

```bash
python -m sn_tune.run --model_name <hf> --basis_dir <phase1_basis> \
    --dataset_file ./data/circuit_breakers_train.json --output_dir ./warp_sn_output
```

Pipeline: Convert layers to `LinearSNWaRP` (`C = W @ U`) → detect top-k safety coordinates by accumulated
`|∂L/∂C|` → tune only those coordinates → restore to `nn.Linear` (`W_final = C @ U.T`, exact because `U` is
orthonormal). Components: `module.py`, `detect.py`, `run.py`.

## Evaluation

Downstream task eval/fine-tune harnesses live in per-task directories, each a standalone script (not wired into
`train.py`): `gsm8k_eval/`, `mbpp_eval/`, `mmlu_eval/`, `medqa_eval/`, `agnews_eval/`, `arc_eval/`. Pattern:
`finetune_<task>_full_params.py` (baseline full-FT) vs `finetune_<task>_freeze_sn.py` (with safety-neuron freezing),
plus `evaluate_<task>*.py` / `eval_<task>*.py`.

Many root-level scripts are analysis/plotting one-offs: `plot_*.py`, `singular_value_plot*.py`, `figure_graph.py`,
`analyze_dual_importance.py`, `visualize_masks.py`, `wsr_baseline_delta.py`.

## Uploading results

`upload_to_huggingface.py` (and `upload_phase0_to_hf.py`, `upload_phase3_to_hf.py`) push trained checkpoints to the
Hub. `patch_chat_template.py` fixes tokenizer chat templates on saved models.

## Environment

Python 3.11 + PyTorch (CUDA) + `transformers`/`peft`/`trl`/`accelerate`/`bitsandbytes`. Install via
`pip install -r requirements.txt` or `conda env create -f environment.yml`. Optional Weights & Biases logging
(`--use_wandb`); `wandb/` run dirs are gitignored.

`config.yaml` holds default hyperparameters but the shell scripts pass explicit CLI args that override it — the
scripts, not `config.yaml`, are the source of truth for what actually ran.

## Repository conventions

- `checkpoints/`, `wandb/`, `outputs/`, `*.pt`/`*.safetensors`/`*.bin`, and most `*.json`/`*.csv`/`*.jsonl` are
  gitignored. **Exception:** files under `data/` are force-tracked (`!data/*.json`), so committed datasets like
  `data/circuit_breakers_train.json` are intentional.
- Outputs are timestamped: `phase{N}_YYYYMMDD_HHMMSS`. The integrated script auto-discovers the newest matching
  dir via `find ... -printf '%T@'` to chain phases, so don't rely on a fixed checkpoint name.
- There is no test suite, linter config, or CI in this repo. "Running" means executing a phase or eval script.
