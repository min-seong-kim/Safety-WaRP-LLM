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

**Phase 2 needs `--perlayer` unless you mean otherwise.** The flag picks a *different implementation*
(`train.py:459`): with it, `phase2_importance_per_layer` applies `keep_ratio` **within each layer** and runs
at ~5 it/s; without it, `phase2_importance_whole` applies one global threshold across the model and runs at
~0.1 it/s — a 50× slowdown (≈9 min vs ≈3.5 h on Llama-2-7B) *and* a different mask. Every existing sweep and
every uploaded WaRP model is per-layer, so omitting the flag silently produces an incomparable model.

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

### Time / VRAM profiling

`train.py` profiles every phase by default (`utils.ResourceProfiler`): each sub-step (`load_model`,
`compute_importance`, `train_and_save`, …) is logged as `[PROFILE] ■ Phase N / <stage> … time=… |
torch_alloc_peak=… | torch_reserved_peak=… | device_peak=…`, followed by a per-phase summary table.
Three memory numbers, deliberately: `torch_alloc`/`torch_resv` are this process's allocator peaks;
`device` is nvidia-smi-style total usage from a background sampler (catches CUDA context, cuBLAS
workspaces, and other processes), and `device-base` subtracts the usage present when the phase started.
Phase 3's train loop additionally logs elapsed/ETA/VRAM every `--logging_steps` (`ResourceLogCallback`
in `phase3_extra_learning_non_freeze.py`) and writes `train_seconds` / `train_peak_*_gb` into the
checkpoint's `metadata.json`.

- `--profile_json PATH` (default `log_dir/phase{N}_{ts}_profile.json`), `--profile_interval` (sampling
  period, default 0.5s), `--no_profile` to disable.
- `run_all_phases_integrated.sh` also times each phase with bash wall-clock (includes python startup /
  model download), writes per-phase JSONs to `logs/profile_<TIMESTAMP>/`, and prints a pipeline-wide
  time × peak-VRAM table plus stage breakdown at the end (`pipeline_resource_summary.json`).

## Layer types

`--layer_type` accepts a comma-separated list of: `ffn_down`, `ffn_up`, `attn_q`, `attn_k`, `attn_v` (LLaMA module
names: `down_proj`, `up_proj`, `q_proj`, `k_proj`, `v_proj`). `--target_layers` accepts `all`, `early`, `middle`,
`late`, `last`, a single index (`31`), or a range (`0-5`).

## (R)SN-Tune line and its WaRP counterpart (`sn_tune/`)

Produces the paper's **Table 3** (`tab:baseline_plus_warp`) rows for SN-Tune / RSN-Tune and their
`+ WSR-Tune` versions. Four arms, all driven from this one package — full docs in
**`sn_tune/README.md`**:

| arm | space | selects | trains (then freezes downstream) |
|---|---|---|---|
| SN-Tune | original | `N_safe` | safety neurons only |
| RSN-Tune | original | `N_robust = N_safe \ N_foundation` | critical neurons only |
| WSR-SN-Tune | WaRP | safety **columns** of `basis_coeff` | those columns only |
| WSR-RSN-Tune | WaRP | critical columns | those columns only |

```bash
bash sn_tune/scripts/run_sn_rsn_gsm8k.sh        # original space, 7B/13B × GSM8K
bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh    # WaRP space
DRY_RUN=1 / STOP_AFTER_NEURONS=1 / MODELS=... / ARMS="rsn"   supported; both resumable via `.done`
```

`sn_tune/` is **self-contained as of 2026-09-22** — the external `Safety-Neuron/neuron_detection`
tree is no longer needed. Detection, tuning, the patched modeling files, and the conda-env setup
all live here.

**Two conda envs, and mixing them silently produces empty results.** Original-space detection
(`detect_original.py`) reads `_last_*_score` tensors stashed by a **patched** `transformers`;
training must use the **stock** one. `bash sn_tune/setup_hb_sn.sh` clones `hb` → `hb_sn` and
patches only the clone (idempotent; backs the stock files up as `.orig` once; never touches `hb`).
The patched files are vendored in `sn_tune/transformers_patch/` with sha256s and the known
version-drift notes in `PATCH_NOTES.md`. The WaRP-space arms need **no** patch — that detector
computes its scores from its own forward hooks.

- **Detection swallows its own failures.** Every prompt runs inside `try/except`, so a 100%-failure
  run still exits 0 and writes `{"0": [], ...}` for all five sections; the first real error surfaces
  hours later as `ValueError: optimizer got an empty parameter list`. `detect_original.py` now
  verifies the patch *before* loading the model and refuses to finish on an empty result;
  `sn_tune/neuron_file.assert_nonempty` is the shared guard.
- **Neuron file = 5 lines, and the order is the contract**: `ffn_up, ffn_down, q, k, v`, assigned by
  line position, never by the keys in the file. Unknown keys are skipped silently, so a mis-keyed
  dict counts as zero rather than erroring. Both spaces share this format — but original-space
  indices are weight **rows** (output neurons) and WaRP indices are `basis_coeff` **columns** (basis
  directions), so never subtract one space's file from the other's (`--space` records which).
- **The freeze directions are deliberately asymmetric.** `sn_tune_original.py` freezes everything and
  re-enables safety rows with a keep-mask; `finetune_freeze_sn.py` trains everything, zeroes safety
  gradients, **and** restores those weights after each optimizer step — needed because AdamW's
  decoupled weight decay moves parameters regardless of the gradient hook. The tuner has no such
  callback (kept as in the original, to match the already-published checkpoints).
- **Saved dirs get a `_lr<lr>_<ts>` / `_<ts>` suffix**, so `--output_dir` is not where the model
  lands. Use `--no_timestamp_suffix` or `--model_dir_file` (the drivers use the latter).
- Driver defaults match the published 7B/13B RSN models' `finetune_config.json` exactly (lr 5e-5,
  3 ep, 4×4, max_len 1024, wd 0.01, warmup 0.1, cosine, bf16) — the same operating point as WaRP
  Phase 3.
- **top-k differs between safety and utility, and per model** — recovered from the original run logs
  (2026-09-22). Llama-2-7B-chat: safety **1200/200** on circuit_breakers 4994 → 12,998 neurons
  (0.956% of neurons, **1.135% of parameters**); utility **300/50** on **Wikipedia** 1000 docs →
  1,826; critical → 11,329 (0.967% of params). Llama-2-13B-chat: 1200/200 gave only 0.743%, so it
  was redone at **1800/300** → 0.958%. `300/50` was the house default for utility across every model
  (7B chat/base, Llama-3.1-8B, Qwen2.5-32B); only safety was scaled up with model size. The
  foundation corpus is **Wikipedia, never Alpaca** — `foundation_neuron_detection.py` exists and also
  uses Wikipedia, but with a different algorithm (global `--ffn_active_fraction`, no patch needed)
  and was not what produced these files.
- **Detection is cheap: ~4 min for 4994 prompts on 7B, ~5 min for 1000 Wikipedia docs.** Calibrating
  top-k by running 2–3 times is the intended workflow (the 13B logs show exactly that). The two
  reported percentages differ — the paper's "≤1%" is the **parameter** one, not the neuron one.
- **The published checkpoints start from the plain chat model, not an SSFT one.** All 17 SN-Tune runs
  in the logs used `meta-llama/Llama-2-{7b,13b}-chat-hf` / `Llama-3.2-3B-Instruct`. That is by
  design — SN-Tune *is* the safety-alignment step, so it replaces SSFT. Drivers default to plain
  chat to match. **This makes the start point differ from Table 3's other arms** (SafeInstr / SEAL /
  WSR-Tune all start from `kmseong/llama2_7b-chat-Safety-FT-lr5e-5`); footnote it, or override with
  `START_<model>`.

Also in the package, a separate line: `python -m sn_tune.run` converts layers to `LinearSNWaRP`
(`C = W @ U`), detects top-k safety coordinates by accumulated `|∂L/∂C|`, tunes only those, and
restores to `nn.Linear` (`W_final = C @ U.T`, exact because `U` is orthonormal) —
`module.py`, `detect.py`, `run.py`. The per-task SN fine-tuners stay with their eval harnesses
(`mbpp_eval/finetune_mbpp_freeze_sn.py`, `mmlu_eval/finetune_mmlu_freeze_sn.py`).
`python -m sn_tune.test_neuron_file` checks the format/set logic without a GPU.

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
| `asft` | `s·BA` + loss penalty | output space / layer | r | base+aligned models |

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

**AsFT** (`--method asft`, arXiv:2506.08473) is the training-time counterpart of SafeLoRA. It reuses the *same*
matrix `Ĉ = VVᵀ/‖V‖_F` (`V = W_aligned − W_base`) but, instead of projecting `lora_B` once after training, adds
`λ·Σ_l ‖(I−Ĉ_l)·B_l A_l‖²_F` to the loss every step (`AsFTTrainer` in `finetune_gsm8k_lora.py`,
`models/asft_baseline.py`). Ported from the reference impl at `/home/edgeai_lab/AsFT`
(`utils/AsFT_train_utils.py:99-119`), keeping its quirks deliberately: `ΔW = BA` **without** the `s=α/r` scaling,
and `Ĉ` divided by `‖V‖_F` (not `‖V‖²`, so it is not a true projector). λ default 1.0 = the reference's
`AsFT_reg1_p_0.1.sh`. The reference materializes `Ĉ` (11008² for `up_proj` → >15 GB fp32); we store `V` instead and
use the identity `‖(I−Ĉ)BA‖²_F = trace((XᵀX)(AAᵀ))`, `X = B − V(VᵀB)/‖V‖_F` — same value, r×r cost. Verified
to `rel_err≈4e-7` against the reference formula at runtime (`--asft_check_equiv`, logged at the first step where
`B≠0`).

### Downstream tasks beyond GSM8K: local task JSON

`data/local_task_dataset.py` lets the LoRA-family and LISA runners train on any local
`[{"question": ..., "response": ...}]` JSON via `--task_data_path`, bypassing GSM8K. Registered files:
`sst2`/`agnews` (8k seed42 subsets, `data/subsets_seed42.manifest.json`) and `arc`/`medqa`
(`scripts/prepare_qa_task_data.py`, which imports the prompt builders from `arc_eval/`+`medqa_eval/` so the
training format matches those eval harnesses exactly). MedQA first needs
`python medqa_eval/prepare_medqa_dataset.py --output_dir ./data` (its default `--output_dir` is a stale
`/home/yonsei_jong` path). Runners: `scripts/run_lisa_safelora_cls.sh` (sst2/agnews × lisa/safelora) and
`scripts/run_lisa_safelora_asft_qa.sh` (arc/medqa × lisa/safelora/asft), both resumable and matched at
r=16/α=32/batch16/3ep. Upload with the paired `scripts/upload_*.sh`.

**See `wsr_lora_status.md`** for what has actually been trained/uploaded, current results, and the
environment-migration checklist. `wsr_lora_comparison.md` is the older pre-implementation spec.

WaRP **Phase 3 also accepts `sst2`/`agnews`** through the same loader (`_load_local_task` in
`models/phase3_extra_learning.py`, `--phase3_dataset sst2 --sst2_dataset_path ...`), so a WaRP model and a
baseline trained by `finetune_task_full_params.py` see byte-identical prompt strings.

## Classification baselines & defense line (SST-2 / AG News, full-param)

Comparison arms for the WaRP Phase 3 classification models, all from the **same start model**
`kmseong/llama2_7b-chat-Safety-FT-lr5e-5` and the same operating point: **full-parameter SFT, epoch 1,
effective batch 16 (2×8), max_len 1024, seed 42, bf16, cosine + warmup 0.1, weight_decay 0** — deliberately
the same optimizer settings as WaRP Phase 3 *and* the same prompt strings (`data/local_task_dataset.py`), so
the only variable across arms is the method. Built at lr 1e-5, then baseline re-run at lr 5e-5 (higher lr =
stronger downstream fit = more safety loss; the lr 1e-5 baseline barely lost safety).

| arm | how it is produced | knob |
|---|---|---|
| baseline | plain full-param SFT | lr |
| SafeInstr | same SFT + `circuit_breakers` mixed into the train set | `--safety_mix_ratio` (used 0.1) |
| SafeDelta | post-hoc on the **baseline** output | `--scale` s (used 0.4) |
| RESTA | post-hoc merge on the **baseline** output | γ (used 0.5) |
| SafeLoRA | LoRA r16/α32 + post-hoc `lora_B ← C·B` | threshold (used 0.5) |

SafeDelta and RESTA take the *baseline* (not SafeInstr) as input — both are "fix a model that was tuned
without defense" methods.

- `finetune_task_full_params.py` — full-param SFT on any local task JSON, plus `--safety_mix_ratio`
  (= SafeInstr). It **imports** `tokenize_prompt_response` / collator / model loading / `maybe_mix_safety`
  from `agnews_eval/finetune_agnews_full_params.py` rather than copying them, so tokenization cannot drift
  from the AG News harness. Refuses to start if an instruct model's tokenizer has no `chat_template`.
- `scripts/run_cls_baselines.sh` — `STAGE=A` (train) / `B` (post-hoc) / `AB`; `MODES="baseline safeinstr"`
  selects which trainings run; one task per GPU; resumable (skips on `summary.json` / `config.json`).
- `scripts/resta_add_safety.py` — RESTA merge **without mergekit** (`pip install -e resta/merge` drags in
  its own transformers pin and would break the `hb` env). Streams shard-by-shard, accumulates in fp32.
  `W_resta = W_ft + γ·(W_align − W_base)`; weights sum to 1.0, so mergekit's `linear`
  (`normalize=True`) is identical. Verified bitwise against the formula.
- `scripts/upload_cls_baselines.sh` — derives repo names from directory names (never hand-typed) and
  validates `chat_template` both before and after upload. `MODES` / `DRY_RUN` supported.
- SafeLoRA reuses `scripts/run_safelora_thr_sweep_qa.sh` with `MODEL`/`LR`/`EPOCHS`/`THRS`/`OUTPUT_ROOT`
  env overrides — it already handles `sst2`/`agnews`.

Uploaded (namespace `kmseong`, `{task}` ∈ {sst2, agnews}):
`llama2_7b-chat-{task}-fullft-lr{1e-5,5e-5}-ep1`,
`llama2_7b-chat-{task}-safeinstr0.1-lr1e-5-ep1-cb`,
`llama2_7b-chat-{task}-safedelta-lr1e-5-ep1-cb-s0.4`,
`llama2_7b-chat-{task}-resta-lr1e-5-ep1-gamma0.5`,
`llama2_7b-chat-{task}-safelora-r16-a32-lr3e-5-ep1-cb-thr0.5`,
plus the WaRP arm `llama2_7b-chat-{task}-warp-kr0.1-lr1e-5-ep1-cb`.
**The defense arms are all built on the lr 1e-5 baseline**; if the lr 5e-5 baseline becomes the reference,
SafeDelta/RESTA/SafeInstr must be regenerated from it or the arms no longer pair up.

### Traps this line walked into (all fixed; do not re-introduce)

- **`chat_template.jinja` is a separate file.** transformers 4.4x/5.x writes the chat template next to
  `tokenizer_config.json`, and an upload path that pushes only the model + tokenizer objects can silently
  drop it — the hub copy then loads with `chat_template=None` and evaluation renders a different prompt than
  training did. This actually happened to both uploaded WaRP classification models (weights were fine; the
  file was re-uploaded). **Always verify `chat_template` on the hub after upload**, not just locally.
- **`agnews_eval/finetune_agnews_full_params.py:46` set `CUDA_VISIBLE_DEVICES="1"` at import time** — the
  third file in this repo with that bug. Commented out; `finetune_task_full_params.py` raises if importing
  it ever changes the variable again.
- **RESTA merge key mismatch:** `meta-llama/Llama-2-7b-chat-hf` is an old checkpoint carrying
  `model.layers.N.self_attn.rotary_emb.inv_freq` buffers that newer saves do not have. They are a rotary
  frequency cache derived from config, not parameters — the merge whitelists that suffix and aborts on any
  other unexpected key.
- **SafeDelta's `s` is not comparable across fine-tuning types.** At s=0.4 a LoRA-merged delta keeps 98–99%
  of the fine-tuned deltas (essentially no defense), while a full-param delta keeps only ~46% (per-layer
  21–86%). Full FT perturbs every parameter, so the same safety-loss budget binds far harder. Do not carry
  an `s` value across the LoRA line and this line and assume equal strength.
- **lr 3e-5 LoRA for 1 epoch is enough here**, contrary to the intuition from the 3e-4 LoRA runs: SST-2 /
  AG News answers are a single token, so loss reaches 0.03–0.06 by the end of one epoch.

## Rebuttal experiment: WSR-Tune vs ActSVD mask-structure ablation

Answers the reviewer question "how is this different from ActSVD (Wei et al. 2024)?" by running
ActSVD-style **rank freezing** and WSR-Tune's **entry freezing** inside the *same* Phase 1/2/3
pipeline, changing only the coordinate basis and the mask granularity. Spec: `actsvd/wsr_actsvd_ablation_spec.md`.

**The distinction that must not be blurred:** ActSVD SVDs the *output* `W X_in` and takes **left**
singular vectors `U_out ∈ R^{m×m}` (output space, applied by left-multiply `Ŵ = U_out U_outᵀ W`);
WSR-Tune eigendecomposes the *input* covariance `X_in X_inᵀ` giving `U_in ∈ R^{n×n}` (input space,
right-multiply `W̃ = W U_in`). In WSR-Tune's `W̃ = Vᵀ W U` framework, ActSVD is **V = U_out with
row masking**, never column masking on U.

| arm | basis | mask unit | flag |
|---|---|---|---|
| A | original (U=V=I) | entry | `--ablation_arm A` (no `--basis_dir`) |
| B | `V = U_out` (ActSVD) | **row** | `--ablation_arm B` + output-side basis |
| C | `U = U_in` | column | `--ablation_arm C` |
| D | `U = U_in` | entry | `--ablation_arm D` (= WSR-Tune) |
| D_perm | `U = U_in`, `V` = signed permutation | entry | sanity: must equal D exactly |

**Every arm uses safety data only** (`circuit_breakers`) — same setup as the paper: does a
safety-tuned model keep its safety through downstream FT. Wei et al.'s utility disentanglement
`(I−Π^u)Π^s` (spec §4) is deliberately *not* implemented; pulling in a utility corpus would break
the premise that the only variable across arms is basis/mask structure.

Driver: `bash scripts/run_wsr_actsvd_ablation.sh` (resumable; `STOP_AFTER_MASKS=1` halts before
training). Smoke test on a tiny random LLaMA: `bash scripts/_smoke_wsr_actsvd.sh` (~5 min).
Report/budget cross-check: `python actsvd/wsr_actsvd_ablation_report.py --root outputs/wsr_actsvd_ablation`.

- Code lives in its own package **`actsvd/`**: `actsvd_basis.py` (both basis sides,
  `--basis_side {input,output}`), `wsr_ablation_masks.py` (arm specs, entry/row/column masks,
  budget accounting), `wsr_ablation_reparam.py` (Phase 2/3 공용 좌표계 세팅),
  `phase2_importance_ablation.py`, `phase3_ablation.py`, `test_wsr_actsvd_ablation.py`,
  `wsr_actsvd_ablation_report.py`, `wsr_actsvd_ablation_spec.md`. `train.py` lazy-imports them
  behind `--ablation_arm` / `--basis_side`; they import `models.{phase1_basis,
  phase2_importance_per_layer, phase3_extra_learning, warp_modules}` so the repo root must be on
  `sys.path` (running from the repo root, as every script does, is enough).
- `LinearWaRP` now treats an **empty** `UT_forward`/`UT_backward` as an identity basis, which is what
  makes arm A (both empty) and arm B (only `UT_backward`) expressible. Existing paths are unchanged.
- **Budget matching is the fairness gate**: every arm freezes the same number of scalars
  (`row: k=round(ρ·m)`, `column: k=round(ρ·n)`, entry: `round(ρ·m·n)`) — 449.6M ± 0.07% at ρ=0.1 on
  Llama-2-7B. `budget_report.json` per arm; the report tool cross-checks and refuses to call an
  unmatched comparison fair.
- **Known confound**: arm A's reparameterization is exact (identity), while B/C/D carry the ~3e-3
  bf16 basis round-trip that the published WSR-Tune also has. B-vs-C-vs-D is clean; A-vs-rest has this
  small extra term. Flagged automatically in the report.

## Revision experiment matrix (`scripts/revision/`)

Extends the paper's Table 2 / Table 4 / Figure 4 with the seven methods it never ran
(`lora asft lisa seal safelora salora wsr_lora`) plus a BeaverTails safety axis — **116 new
training cells**. Full docs: **`scripts/revision/README.md`**; every repo it will create is
listed in **`scripts/revision/REPO_LIST.md`**.

```bash
PLAN_ONLY=1 bash scripts/revision/run_all.sh    # size/progress only
DRY_RUN=1   bash scripts/revision/run_all.sh    # print commands
bash scripts/revision/run_all.sh                # run (resumable)
```

- Axes: safety `cb`/`bt`; models `llama2_7b llama2_13b llama32_3b llama31_8b qwen25_7b
  gemma2_9b`; methods `fullft safeinstr resta safedelta wsr_tune` + `lora asft lisa seal
  safelora salora wsr_lora` (SN-Tune/RSN-Tune deliberately excluded).
- **Scope is 116 cells, not the full 2×6×tasks×12 grid.** `BT_MODELS=llama2_7b` (BeaverTails
  runs only on Llama-2-7B, but on all four tasks with all twelve methods), and
  `SKIP_PUBLISHED=1` reuses the paper's own numbers. `already_published()` in `common.sh`
  encodes exactly what is reused and why.
- **Reuse rule is deliberately strict: only Table 2/4/10's five full-parameter arms**
  (`fullft safeinstr resta safedelta wsr_tune`) on the CB axis. Those came from
  `run_all_phases_integrated.sh` with `PHASE0_MODEL=kmseong/llama2_7b-chat-Safety-FT-lr5e-5`,
  lr 5e-5, 3 epochs, effective batch 16 — verified to match this line's settings.
  The rebuttal's PEFT numbers are **not** reused, for three checked reasons:
  (a) the MedQA/ARC PEFT runs started from `wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb`, whose
  safetensors sha256 differ from `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` (total bytes differ
  by 8, so possibly the same weights re-sharded — unverified, so not assumed);
  (b) SafeLoRA was swept at thr 0.15/0.25/0.35, never 0.3, which is this line's value;
  (c) the AGNEWS table came from the classification-baseline operating point
  (epoch 1, lr 1e-5, wd 0) — this line uses epoch 3, lr 5e-5, wd 0.01.
  Because of (a), **keep `LLAMA2_7B_ALIGNED_CB` at `kmseong/...Safety-FT-lr5e-5`**: switching it
  to the `wvnvwn` copy would misalign the new arms against the very rows being reused.
- Known residual gap: the paper's MedQA runs used 10 000 samples (`MEDQA_SAMPLES=10000`),
  new cells use all 10 178 (+1.7%). Affects only the five reused MedQA reference rows.
- Stages `00`(data) → `01`(BT SSFT) → `02`(basis+mask) → `10`(FullFT/SafeInstr) →
  `11`(RESTA/SafeDelta, consumes `10`) → `12`(WSR-Tune) → `20`(LoRA×6) → `21`(SEAL).
  Every cell writes a `.done` sentinel, so re-running resumes. At the current scope stage `01`
  is a no-op (the Llama-2-7B BeaverTails start model `wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv`
  already exists), and stage `02` builds a Phase 2 mask only where a `wsr_tune` cell is actually
  wanted — WSR-LoRA needs the basis alone.
- `scripts/revision/REPO_LIST.md` lists every repo that will be created, grouped by model, with
  each cell marked new vs reused. Regenerate with `gen_repo_list.sh` after any hyperparameter
  change, since the hyperparameter is part of the repo name.
- `scripts/revision/common.sh` is the **single source of truth** for the registry and every
  hyperparameter. Output convention: `outputs/revision/<safety>/<model>/<task>/<method>/`,
  with a `MODEL_DIR` file naming the actual model directory (runners disagree on whether the
  model lands in `<out>` or `<out>/merged_model`).

**Three invariants this line depends on:**
1. **Every arm reads the same task JSON.** Six different tokenization implementations exist
   across the runners; `scripts/revision/verify_prompt_parity.py` feeds identical rows through
   all six and asserts identical `(input_ids, labels)`. Verified: 6 models × 5 tasks × 6 paths.
   GSM8K is dumped to `data/gsm8k_train_task_7473.json` for this reason.
   **AG News is the 8k seed42 subset, not the full 120k** — matches all existing AGNEWS results.
2. **One safety axis.** Methods needing safety data use the dataset their *start model* was
   safety-tuned on; the `$safety` loop variable drives both `aligned_for()` and `safety_json()`.
3. **WSR-LoRA = `wsr-lora/wsr_lora.py --reparam`** (PiSSA init + `Ã=AU`), the variant described
   in the rebuttal — *not* `finetune_gsm8k_lora.py --method wsr_lora` (old element-wise
   product-mask). It requires the Phase 1 basis from stage `02`.

**Disk / HF upload.** 216 cells kept locally = **3.4 TB**; this box has ~155 GB. The default
operating mode is `PUSH_TO_HUB=1`: each finished cell is uploaded, **verified**, then its local
weights are deleted (`scripts/revision/upload_and_prune.py`). Verification requires all four of —
files present, sizes match, `AutoConfig` loads from the hub, and **`AutoTokenizer(...).chat_template`
is non-None on the hub** (the `chat_template.jinja` trap, hit twice before). Nothing is deleted
unless all four pass. `fullft` is RESTA/SafeDelta's input, so it is uploaded immediately but pruned
only after those consumers finish. Repo names come from `hf_repo_id()` — never hand-typed:
`kmseong/{model}-{CB|BT}_SSFT-{method}_{task}[_{hparam}]_lr{lr}`. The full list of all 221 repos
that will be created lives in `scripts/revision/REPO_LIST.md`, generated by
`scripts/revision/gen_repo_list.sh` — regenerate it after changing any hyperparameter, since
the hyperparameter is part of the repo name.
Two other consumers are auto-pruned: Phase1 basis + Phase2 mask (248 GB total, freed per
(safety, model) once the WSR arms finish) and the HF cache (freed per model). `ORDER=model`
(default) puts the model in the outer loop so both prunes actually fire. `llama2_13b` peaks at
~161 GB — run it alone and split `METHODS` in two.

**Basis storage is now 1/4 the size.** `--basis_save_dtype bfloat16` (default, and now actually
honored by `models/phase1_basis.py`, which previously ignored it) plus the new `--basis_omit_ut`.
Safe because Phase 2, Phase 3, and WSR-LoRA all do `U.to(dtype=W.dtype)` (bf16) right after
loading, and **no consumer in this repo ever reads the `UT` key** (symmetric Gram ⇒ `UT == U.t()`).
Only `apply_safety_basis_rotation.py` does float32 math on the basis — pass
`--basis_save_dtype float32` when building a basis for it.

**Bugs fixed while building this (do not re-introduce):**
- `is_instruct_model` disagreed across `agnews_eval` / `gsm8k_eval` / `seal` (`"instruct"|"chat"`)
  and `models/phase3_extra_learning` (also `'it'`). **`gemma-2-9b-it` was therefore trained with
  a chat template by WSR-Tune and a plain prompt by every other arm.** All five now share one
  token-boundary rule; keep them in sync.
- BT SSFT output dirs must keep the base model's identity in the **name**
  (`ssft_bt/<BaseName>-ssft-bt-lr<lr>`) — the runners decide chat-template usage from the model
  reference *string*, so a tagless path silently falls back to plain prompts.
- `--phase3_task_data_path` / `--phase3_task_samples` added to `train.py` + Phase 3: routes any
  task through `_load_local_task`. Without it Phase 3's `_load_agnews` cannot read the
  `{"question","response"}` schema and **shuffles** when subsampling.
- Phase 1 now writes `'decomp': 'svd'` into `basis/metadata.json`; `wsr-lora/wsr_lora.py`
  required that key and no code here ever wrote it, so `--reparam` always failed.
- `models/phase0_SSFT.py` batch/grad-accum are now env-overridable (`SSFT_BATCH_SIZE`,
  `SSFT_GRAD_ACCUM`); hardcoded 4×4 OOMs on 13B/9B. Keep the product at 16.
- `/home/edgeai_lab/SafeDelta/llama2/run_safedelta.py` had `CUDA_VISIBLE_DEVICES="2,3"` at
  import time; patched to `setdefault`. SafeDelta is an **external** repo dependency.
- `finish_cb.sh` originally ran `git add -A REVISION_PROGRESS.md REPO_LIST.md …`, but
  `REPO_LIST.md` lives in `scripts/revision/`, not the repo root. `git add` aborts wholesale on a
  missing path, so the final commit silently staged **nothing**. Never mix an unverified path into
  a `git add` list in an unattended script.

### Current state (2026-09-02) and how to resume

`REVISION_PROGRESS.md` (cell status) and `RESULTS.md` (measured numbers) are both **generated**,
never hand-written:

```bash
python scripts/revision/gen_progress_md.py --out REVISION_PROGRESS.md   # 셀 진행 상태
python scripts/revision/gen_results_md.py  --out RESULTS.md             # ASR + downstream 결과표
```

A repo that merely *exists* on the hub does not count as done — an interrupted upload leaves an
empty repo, so cells without a verified `.uploaded` marker are checked for actual `safetensors`.
`gen_results_md.py` additionally cross-checks each number's measurement time against the hub's
`lastModified` and drops values that predate a retrain (marked `⟲재학습`).

**The CB axis is complete for the six models**, plus two ablations added on 2026-09-01/02.
The HF storage quota that blocked 2026-08-28 has been resolved. Full narrative — what was run, in
what order, and every trap hit — is in **`scripts/revision/SESSION_2026-09.md`**. Read it before
resuming; the traps below cost hours each.

**Two ablations now exist beyond the 12-method grid:**
- **LISA ρ** — `LISA_RHO` default is now **0.0** (the rebuttal model's own `finetune_config.json`
  says ρ=0.0; ρ=1.0 collapses GSM8K 0.39→0.17). ρ=1.0 was rebuilt for all 9 cells so both sides
  exist. **The result does not generalize**: ρ=1.0 usually trades utility for safety, but on
  llama32_3b/llama31_8b it loses *both*. Do not report one side alone.
- **WSR-LoRA α** — `wsr_lora.py:83` is `scaling = alpha / rank`. The rebuttal's `-rot` models used
  α=16 (scaling 1.0) while revision uses α=32 (scaling 2.0), so the two were **different
  hyperparameter generations being compared as if they were the same method**. A controlled 2×2
  (start model × α) confirmed α dominates (1.8–2.7×), start model is secondary. α=16 is safer or
  equal on 5/6 models with *higher* downstream on 4/6. ⚠️ Every other LoRA arm is α=32, so shipping
  WSR-LoRA at α=16 alone means our own method gets half the update budget — report it as an
  ablation with a footnote, not as the headline row. Use `WSR_LORA_ALPHA=16` (repo name gains
  `_a16`; empty = unchanged behaviour).

**Two cells fail reproducibly** (retrained, identical results — seed is fixed):
`cb/llama2_13b/gsm8k/salora` (GSM8K 0.072 / JB 0.451) and `cb/gemma2_9b/gsm8k/seal`
(GSM8K 0.187 / JB 0.594, Direct ASR 0.45). Both are normal on the other four models with the same
settings, and 13B SaLoRA's training log is clean (loss converged, merge/save fine), so these are
cell-specific failure modes, not pipeline bugs. Change a hyperparameter (`r_s` / `topp`) or
footnote them; do not ship the current numbers as-is.

**Traps that cost hours — do not re-introduce:**
- **`/tmp` is `noexec` on this box.** Triton JIT-compiles `.so` files and cannot mmap their exec
  segments there → `ImportError: __triton_launcher...so: failed to map segment from shared
  object`. This is *not* cache corruption, so clearing the cache does not help. `harmbench_eval.sh`
  already redirects to `$HOME`; `lm-evaluation-harness/eval_models.sh` had those lines commented
  out (with a stale `/NHNHOME/...` path), so only lm-eval died — now fixed.
- **Gemma-2 needs `block_size: 32`** (head_size 256 breaks FlashInfer's block_size 16). Both
  `models.yaml` entries *and* lm-eval's `MODEL_ARGS` need it; `attention_backend=FLASH_ATTN` does
  not help on vLLM 0.17. `add_models_to_yaml.py`'s `build_block()` now emits it for gemma keys.
- **`pgrep -f "<script>.sh"` matches any process whose command line contains that string** —
  including a watchdog you launched to monitor it. This stalled an unattended run for 5 hours.
  Gate on a completion marker in the log or a PID file (`kill -0 $(cat /tmp/orch.pid)`), never on
  a process-name match.
- **`out_dir` does not encode hyperparameters** (`outputs/revision/<safety>/<model>/<task>/<method>/`);
  only the repo name does. Running the same cell at a different ρ/α is skipped as "already done".
  Isolate with `OUT_ROOT=outputs/revision_<tag>` — never by deleting `.done` markers.
- **`check_disk` reads 0 GB when `OUT_ROOT` does not exist** (`df` fails → empty → 0), and
  `disk_ok` runs *before* `mkdir` in `run_cell`, so every cell silently skips with `(disk)` on a
  fresh box. `mkdir -p outputs/revision` first.
- **`HfApi().list_models()` leaves `lastModified` as `None`** — pass `expand=["lastModified"]`, or
  freshness checks silently pass and stale numbers get published.
- **Never edit a running bash script.** Bash reads scripts by byte offset; an edit makes it jump
  mid-execution. To change plans, detect completion and start a new script instead.

### Current state (2026-09-07) — this box (aigpu0317), reproduction + ρ/thr sweep

Two days of work (09-02→04) were lost uncommitted on the old box; models survived on the hub and the
measurement logs in `~/HarmBench/logs` / `~/lm-evaluation-harness/logs`. `RESULTS.md` was rebuilt from
the user's spreadsheet + those logs (sections A–D), then extended with sections E/F below.

**Environment.** Training now runs in conda env **`hb_repro`** = the old box's library versions
(`environment_hb_repro.yml` = `environment_hb.yml` minus `apex`/`cryptacular`: torch 2.10.0+cu128,
transformers 4.57.3, peft 0.18.1; works on the RTX PRO 6000 Blackwell). The other env `hb` has
transformers 5.13 — models saved from it carry `"tokenizer_class": "TokenizersBackend"` in
`tokenizer_config.json`, which the `harmbench` env (transformers 4.57.6 / vLLM 0.16) **cannot load**;
patch to `PreTrainedTokenizerFast` before evaluating (`upload_and_prune.py` does not catch this — it
verifies with the hb tokenizer). Jobs here run directly on the node (no `sbatch`); **use GPU 0 only**
(user decision 2026-09-06), it is shared with other users.

**Reproducibility finding** (`scripts/revision/repro_2026-09/REPORT_safelora_3b_thr0.35_a16.md`):
the evaluation pipeline reproduces old numbers exactly (re-measured original models match to the
digit), but training is **not bitwise reproducible even with identical env/GPU/seed** — Δ-cosine
between two same-config runs is ≈0.54 (same ‖Δ‖, same touched modules), and keyword ASR moves by
±0.05 run-to-run. Library version does not close that gap; do not read ASR differences under ~0.05
between single runs as real. `weight_delta_cosine.py` there computes the comparison.

**Sweep 2026-09-04→07** (RESULTS.md sections E/F; raw table `logs/revision_sweep/RESULTS_sweep.md`):
6 models × WSR-LoRA ρ=0.4/0.5 + SafeLoRA thr=0.2/0.25, all α=16, 24 cells uploaded as
`kmseong/<model>-CB_SSFT-{wsr-lora_<task>_rho0.{4,5}|safelora_<task>_thr0.2{,5}}_a16_lr3e-4`.
Driver: `scripts/revision/sweep_gpu_adaptive.sh` (env: `ALLOWED_GPUS`, `MODELS_ORDER`,
`EVAL_PER_MODEL`, `WSR_NEED_OVERRIDE`; picks the allowed GPU with the most free memory per stage,
waits otherwise, evaluates in a final two-pass gap-fill: non-7B at ≥73 GiB, 7B at ≥88 GiB) +
`sweep_gapfill_eval.sh`. `common.sh` regained the lost `lora_alpha_tag()` rule (`_a16` on
lora/asft/lisa/safelora/salora names when `LORA_ALPHA≠32`); `20_lora_family.sh` gained
`WSR_BASIS_BS` (importance-pass batch, default 2 = unchanged; never used ≠2 in shipped numbers).
Only measurement deviation: llama2_7b HarmBench util 0.85 (old rows 0.95; 0.95 cannot start on a
shared GPU).

**Traps hit this round — do not re-introduce:**
- **Other users' `wait_for_idle` scripts grab GPU 0 the moment it goes quiet** between our stages
  (up to 80 GB), then our next stage OOMs mid-load or HarmBench sits in `WaitForVram` (90 min per
  combo). Mitigations that worked: pause the whole tree with SIGSTOP and SIGCONT when memory frees
  (WaitForVram's counter does not advance while stopped); and **reserve memory at process start** —
  `scripts/revision/gpu_reserve_sitecustomize.py` installed as `sitecustomize.py` in `hb_repro`'s
  site-packages grabs the amount listed in `logs/revision_sweep/RESERVE_GPU_GIB` for
  `wsr_lora.py`/`finetune_gsm8k_lora.py`/`train.py` and neutralises `torch.cuda.empty_cache`
  (otherwise `wsr_lora.py` releases the reservation between phases). Memory-only, no numeric effect.
- **WSR-LoRA's safety-importance pass is the memory peak**: 63 GiB on gemma-2-9B, ~85 GiB on 13B
  (`basis_batch_size 2`, all 210 PiSSA layers). Budget for it; the LoRA train loop itself is smaller.
- **`PRUNE_AFTER_UPLOAD=1` deletes the local weights, and this box downloads at ~9 MB/s aggregate**
  (hf_transfer does not help) — re-fetching 12 repos for evaluation cost ~7 h. With >1 TB free, run
  sweeps with `PRUNE_AFTER_UPLOAD=0` or evaluate from the local `merged_model` before pruning.
- `eval_models.sh`'s `LMEVAL_RESUME` glob (`samples_<task>_*.jsonl`) never matched group tasks such as
  `hendrycks_math_safe` (only per-subtask files exist) → MATH re-ran on every resume. Fixed 2026-09-07
  (also checks `results_*.json` for the task key) — that repo is committed separately.
- Hub uploads/downloads drop with `IncompleteRead` a few times a day; `harmbench_eval.sh` retries 3×
  but a whole combo can still fail — always finish with a `RESUME=true LMEVAL_RESUME=1` gap-fill pass.
- `pgrep -f <script>` again matched the shell issuing it (two shells killed this round). Use pid files.

**Environment gotcha:** `environment_hb.yml`'s `apex==0.9.10.dev0` is *not* NVIDIA Apex — it is a
Pyramid auth toolkit whose `cryptacular` dependency cannot build, and pip resolves all metadata
before installing anything, so it aborts the **entire** pip section. Drop `apex` and `cryptacular`
(336/338 install fine); nothing in this repo imports them.

Resume the CB axis with `bash scripts/revision/finish_cb.sh`, or the untouched BT axis with
`SAFETY_SETS=bt bash scripts/revision/run_all.sh` — but note `SAFEDELTA_DIR`
(`common.sh:116`) points at an **external repo that is not on this box**, and the BT axis has four
`safedelta` cells. **Do not delete the `.done` / `.uploaded` markers under `outputs/revision/`** —
they are the only record that an already-uploaded cell is finished.

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

## Original-space freeze sweep (논문 Table 1 재현, `scripts/run_origspace_freeze_sweep.sh`)

Table 1 의 "FT (X% frozen)" 행을 **chat 라인**으로 다시 만든 것. 재파라미터화 없이
(U=V=I) 원래 weight 공간에서 safety importance 를 재고 상위 ρ 를 얼린 뒤 gsm8k 로
full-param FT 한다. WSR-Tune 과의 차이는 **마스크가 어느 좌표계에서 매겨지는가** 하나뿐.

```bash
KEEP_RATIOS="0.05 0.2 0.3 0.4 0.5" bash scripts/run_origspace_freeze_sweep.sh
bash scripts/upload_origspace_freeze.sh          # 업로드만 따로
bash scripts/eval_origspace_freeze_safety.sh     # HB_ONLY=1, safety 만
```

- 출발 모델 `kmseong/llama2_7b-chat-Safety-FT-lr5e-5`, lr 5e-5 · 3ep · eff.batch 16 ·
  wd 0.01 · warmup 0.1 · seed 42. **ρ=0 기준행은 이미 있다**:
  `kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5` (동일 출발 모델·동일 동작점, AVG 0.2078).
- 리포: `kmseong/llama2_7b-chat-origspace-freeze-p{05,20,30,40,50}-gsm8k-lr5e-5` (2026-09-13).
  B200 에서 비율당 Phase2 2.5분 + Phase3 15분 ≈ **18분**, 5개 1시간 25분.
- 구현은 기존 `models/phase2_importance_original_space.py` +
  `models/phase3_extra_learning_original_space.py` 를 그대로 쓴다 (`--original_space_mask`).
  Phase 2 는 `load_basis`/`convert_to_warp_modules`/`reparameterize_weights` 를 **no-op 으로
  오버라이드**하므로 basis 가 필요 없고, 마스크는 레이어별 `quantile(1-ρ)` 다.

**동결이 실제로 걸렸는지 확인하는 법 (bf16 보정 필수).** 원공간 이진 마스크면 mask=1 위치는
출발 모델과 bit-identical 이어야 한다. 그런데 bf16 은 가수가 8비트라 **lr 5e-5 로 학습된
파라미터도 55% 가 반올림으로 값이 그대로다**. 그래서 단순 일치율을 ρ 로 읽으면 안 된다.
마스크가 없는 파라미터의 일치율을 바닥값 `r` 로 삼아 보정한다:

    관측 = f + (1-f)·r   →   f = (관측 - r) / (1 - r)

실측(2026-09-13): p05 4.46% · p20 19.34% · p30 29.23% · p40 39.14% — 요청값과 전부 일치
(오차 -0.5~-0.9%p, 대상 모듈의 바닥값이 조금 낮아 생기는 일관된 과소추정).

- **기존 2026-05-07 배치도 유효하다.** `kmseong/llama2-7b-chat-original-space-freeze-p{10,20,25,30,40,50}-lr5e-5`
  는 `model_metadata.json` 이 전부 `keep_ratio: 0.1` / `base_model: "unknown"` 인 **빈 템플릿**
  이라 메타데이터로는 검증이 안 되지만, 위 방법으로 재면 p10 → 9.48%, p50 → 49.09% 로
  이름과 맞는다. 바닥값도 신규 배치와 같은 55% 대라 제조 방식이 동일하다. 메타데이터가
  비었다는 이유만으로 재학습하지 말 것 — 먼저 동결률을 재라.

**스윕 스크립트의 업로드 단계 버그(2026-09-13, 수정 완료).** `run_origspace_freeze_sweep.sh` 는
`upload_and_prune.py --cell_dir "$MODEL_DIR"` 로 호출하는데, 그 도구는 `.done` 과 `MODEL_DIR`
이 들어 있는 **셀 디렉토리**를 받아야 한다(`model_dir_of` 가 `MODEL_DIR` 파일을 읽어 가중치
위치를 찾는다). 그래서 5개 셀 모두 `.done 이 없다` 로 업로드가 실패했다 — **학습은 정상**.
`--cell_dir "$CELL"` 로 고쳤다. 업로드만 다시 돌려야 할 때는
`scripts/upload_origspace_freeze.sh` (재실행 안전, `.uploaded` 로 건너뜀).

## SafeGrad baseline (`safegrad/`)

**SafeGrad** (Yi et al., arXiv:2508.07172) ported as a comparison arm. Per step it takes two
gradients — user task and safety alignment — and **only when they conflict** (whole-model dot
product < 0) projects the user gradient onto the plane orthogonal to the alignment gradient,
then combines: `g_final = g'_user + ρ·g_align`. The alignment loss is **KL to the frozen
start model** (`D_KL(P_θ0 ‖ P_θ)`), not refusal-token CE — that is what makes it work with as
few as 10 alignment samples in the paper.

- `safegrad/safegrad_trainer.py` (`SafeGradTrainer`), `safegrad/finetune_safegrad.py` (runner,
  mirrors `gsm8k_eval/finetune_gsm8k_lisa.py`), `safegrad/test_safegrad.py` (6 checks),
  `safegrad/scripts/_smoke_safegrad.sh` (~1 min). Full docs: **`safegrad/README.md`**.
- Wired into `scripts/revision/20_lora_family.sh` + `common.sh`, but **deliberately not in the
  default `METHODS`** — run it with `METHODS=safegrad ...` so the 116-cell plan does not grow
  silently. Knobs: `SAFEGRAD_RHO` (1.0), `SAFEGRAD_REF_MODE` (adapter_off),
  `SAFEGRAD_KL_REDUCTION` (ref), `SAFEGRAD_ALIGN_BS` (0 = same as task batch).
- Runs on **clean** task data like every other arm here (the paper poisons the user data with
  ratio `hr`); alignment data follows the `$safety` axis, same file and field as LISA.

**Three things that must not be "fixed":**
- **`param.grad` is accumulated, not assigned.** The reference impl does `param.grad =
  final_grad`, which silently drops all but the last micro-batch. Harmless there
  (`gradient_accumulation_steps=1` in every reference script), fatal here (effective batch 16 =
  4×4). `test_safegrad.py` check [5] guards it.
- **`F.kl_div(log_p_theta, p_ref)` is `KL(P_ref ‖ P_theta)`** in PyTorch's argument convention,
  which is what paper Eq. 6 asks for. Do not swap the arguments.
- **A huge `projection_scalar` is normal**, not an instability. With LoRA, `B=0` at step 0 means
  θ=θ0 and `g_align=0`; when `‖g_align‖` is tiny the ratio `dot/‖g_align‖²` blows up, but
  Cauchy–Schwarz bounds what is actually subtracted by `‖g_user‖`. Do not add clipping on it.

`--ref_mode adapter_off` (default) uses the LoRA-disabled model as θ0 instead of loading a
second copy — exact whenever the adapter sits directly on the start model (verified equal in
check [6]), and it is what saves the extra model in VRAM. It is **wrong** after merging some
other adapter first, and the runner refuses it for full-param.

SafeGrad does **not** add a 7th tokenization implementation — it imports LISA's
`tokenize_sft_example`. `scripts/revision/verify_prompt_parity.py` asserts it is literally the
same function object, so a future copy-paste breaks the check instead of the comparison.

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
