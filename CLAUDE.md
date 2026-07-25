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

## LoRA line: WSR-LoRA and friends (`finetune_gsm8k_lora.py`)

A **separate experiment line** from the numbered phases, answering "does the WSR idea carry over to LoRA?".
All methods share the same LoRA budget (r=16, α=32, `q,k,v,up,down`) and differ only in *which coordinate
system the update lives in and how it is constrained*. Single entry point `finetune_gsm8k_lora.py --method`:

| method | ΔW | space / unit | rank | needs |
|---|---|---|---|---|
| `lora` | `s·BA` | — | r | — |
| `original_projected_lora` | `s·BA(I−EEᵀ)` | original / column | r | `--safecols_dir` |
| `wsr_lora` | `[(1−M)∘(s·BA)]Uᵀ` | rotated / **element** | full | `--basis_dir --mask_dir` |
| `wsr_lora_nou` | `(1−M)∘(s·BA)` | original / element | full | `--mask_dir` |
| `safe_lora` | post-hoc `B←C·B` | output space / layer | r | base+aligned models |
| `adapter_subspace_lora` | `s·BA(I−Q_SQ_Sᵀ)` | rotated / **direction** | r | `--safety_adapter_path --adapter_subspace_dir` |

SaLoRA has its own runner (`finetune_gsm8k_salora.py`). Element-wise `wsr_lora` breaks rank-r, so it saves via
dense fold (`restore_wsr_lora_to_linear`) instead of `merge_and_unload`.

**`adapter_subspace_lora` does NOT run Phase 1 or Phase 2.** Instead of activation-covariance basis + gradient
importance, it takes the compact SVD of an already-trained **safety LoRA adapter** (`ΔW_s = s_s B_s A_s =
P_s Σ_s Q_sᵀ`, via thin QR ×2 + r×r SVD, never materializing dense `ΔW_s`) and forbids the downstream adapter
from touching `Q_S` (`A_d Q_S = 0`), giving `W_final Q_S = W_safe Q_S` exactly. `models/adapter_subspace.py` +
`build_adapter_subspace.py` + `scripts/run_adapter_subspace_lora.sh` (Stage 0 safety LoRA → 0.5 merge →
1 Q_S extraction → 2 downstream → 3 control; `STOP_AFTER_STAGE1=1` halts after Stage 1; completed stages are
skipped on re-run). Both a gradient hook **and** a post-`optimizer.step()` reprojection are required — AdamW's
elementwise `1/√v` breaks linearity, so projecting gradients alone does not keep the update in the subspace.

**See `wsr_lora_status.md`** for what has actually been trained/uploaded, current results, and the
environment-migration checklist. `wsr_lora_comparison.md` is the older pre-implementation spec.

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

## SEAL × WaRP integration (`seal/`)

`seal/` reimplements **SEAL** (Safety-Enhanced Aligned LLM finetuning via *bilevel data selection*, ICLR'25)
in this repo's HF-Trainer style, then fine-tunes the SEAL-selected data two ways to compare:
**(A) baseline** = standard full-param SFT, **(B) WaRP** = WSR-Tune reparameterized-space SFT (`basis_coeff` only,
safety directions frozen). Safety data = `data/circuit_breakers_train.json`, downstream = `gsm8k`, base model =
`kmseong/llama2_7b-chat-Safety-FT-lr5e-5`. See `seal/README.md` for full docs.

Pipeline (maps to SEAL's S1–S4; only **S2** uses LoRA, **S4-baseline** is full-param, **S4-WaRP** is `basis_coeff`-only):
- `seal/train_selector.py` — Stage 1: bilevel selector loop (port of SEAL `SFTSelectorTrainer.fit`, no DeepSpeed).
  `model_loss = ul*safe + (1-ul)*mean(σ[ide]*ft)`, `selector_loss = mean(σ[ide]*ft.detach())`, `ul` decays per epoch.
  → saves raw-tensor logits `seal/ckpt/<name>_softmax.pt`.
- `seal/select_data.py` — Stage 1.5: `torch.topk(logits, topp*N)` → `seal/ckpt/gsm8k_selected_topNN.json` (indices
  into the **fixed** gsm8k train order).
- `seal/train_sft.py` — Stage 2: SFT on selected data; `--use_warp` toggles WaRP. WaRP path calls
  `seal/warp_setup.apply_warp` (replicates Phase 3 `setup_warp_modules`: `basis_coeff=W@U`, `UT_forward=U`, mask,
  train `basis_coeff` only) then `restore_and_delinearize` before save.
- `seal/scripts/run_all.sh` — chains all stages incl. Phase 1 (basis) + Phase 2 (mask) via the repo's `train.py`.

Run: `bash seal/scripts/run_all.sh` from repo root. It **tees all output to `seal/logs/run_all_<ts>.log`** and
**skips already-completed stages** (guards on `seal/ckpt/*.pt`, `*_selected_*.json`, `out/*/sft_config.json`), so
re-running resumes. Edit the config block at the top (`CUDA_VISIBLE_DEVICES`, `MODEL`, `TOPP`, `KEEP_RATIO`, epochs/lr).

**Known issues / gotchas (esp. when moving to a new environment):**
- **No `python` on PATH** on the original box — the working interpreter was the conda env `hb`
  (`/home/users/minseong/.conda/envs/hb/bin/python`, torch 2.11+cu130, transformers 5.13). On a new machine, point
  the scripts at a python that has torch+transformers; `run_all.sh` calls bare `python`, so activate the right env first.
- **transformers 5.x removed `Trainer(tokenizer=...)`** → use `processing_class=`. `train_sft.py` already handles this
  with a `processing_class`-first / `tokenizer=` fallback. Harmless deprecation warnings remain (`torch_dtype`,
  `warmup_ratio`).
- **Selector artifacts are gitignored** (`*.pt`, most `*.json`), so `seal/ckpt/` does **not** transfer via git. On a
  fresh checkout either copy `seal/ckpt/` over manually or re-run Stage 1 (the resume guards will otherwise redo the
  ~2-epoch selector). Stage 1 + 1.5 were completed on the original box (selector + `gsm8k_selected_top80.json`,
  5978/7473 selected).
- **Index alignment**: Stage 1 and Stage 2 must use identical gsm8k `dataset_name/subset/split/num_train_samples`, or
  the selector indices point at the wrong rows.
- **Base model is an HF hub id** → needs network / HF cache on the new env (first run downloads it).
- **Known divergences from SEAL's real `train_selector_llama3.sh`** (currently repo defaults, not yet aligned): our
  selector uses `ul_weight=0.9/decay=0.1` (SEAL 1.0/0.03), `selector_lr=1e-2` (SEAL 5e-3), cosine LR (SEAL constant),
  LoRA all-linear/α32 (SEAL q,v/α16), and **no gradient accumulation** (SEAL effective batch 64). The full-param S4 is
  an intentional WSR-Tune-alignment choice, not a bug. Align these if strict SEAL fidelity is wanted.

## Environment

Python 3.11 + PyTorch (CUDA) + `transformers`/`peft`/`trl`/`accelerate`/`bitsandbytes`. Install via
`pip install -r requirements.txt` or `conda env create -f environment.yml`. Optional Weights & Biases logging
(`--use_wandb`); `wandb/` run dirs are gitignored.

### SLURM (current box)

Jobs get GPUs via `sbatch` (`#SBATCH --gres=gpu:1`); see `scripts/sbatch_adapter_subspace_lora.sh` or
`~/code_test.sh` for the template. **Never set `CUDA_VISIBLE_DEVICES`** — the scheduler sets it, and a
hardcoded value makes the job grab a GPU it was not allocated. Two inherited hardcodings are commented out
for this reason (`models/phase0_SSFT.py`, `gsm8k_eval/finetune_gsm8k_full_params.py` — the latter set it at
*import* time, poisoning every module that imports it); do not revive them.

Partitions have `AllowQos` restrictions: `gigabyte_a6000`/`suma_a6000` accept the default QOS, while
`suma_a100` needs `a100_qos`/`a100_low_qos` (otherwise `Invalid qos specification`). `sacct`/`sacctmgr`
often fail with `Connection refused` (flaky slurmdbd) — do not treat a single failed `squeue` as job exit.

Verify the HF token with `hf auth whoami`, never `hf auth login` (which reports "Already logged in" whenever
a token file exists, valid or not).

`config.yaml` holds default hyperparameters but the shell scripts pass explicit CLI args that override it — the
scripts, not `config.yaml`, are the source of truth for what actually ran.

## Repository conventions

- `checkpoints/`, `wandb/`, `outputs/`, `*.pt`/`*.safetensors`/`*.bin`, and most `*.json`/`*.csv`/`*.jsonl` are
  gitignored. **Exception:** files under `data/` are force-tracked (`!data/*.json`), so committed datasets like
  `data/circuit_breakers_train.json` are intentional.
- Outputs are timestamped: `phase{N}_YYYYMMDD_HHMMSS`. The integrated script auto-discovers the newest matching
  dir via `find ... -printf '%T@'` to chain phases, so don't rely on a fixed checkpoint name.
- There is no test suite, linter config, or CI in this repo. "Running" means executing a phase or eval script.
