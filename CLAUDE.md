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
pipeline, changing only the coordinate basis and the mask granularity. Spec: `wsr_actsvd_ablation_spec.md`.

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
Report/budget cross-check: `python wsr_actsvd_ablation_report.py --root outputs/wsr_actsvd_ablation`.

- New code: `models/actsvd_basis.py` (both basis sides, `--basis_side {input,output}`),
  `models/wsr_ablation_masks.py` (arm specs, entry/row/column masks, budget accounting),
  `models/wsr_ablation_reparam.py` (Phase 2/3 공용 좌표계 세팅), `models/phase2_importance_ablation.py`,
  `models/phase3_ablation.py`, `tests/test_wsr_actsvd_ablation.py`.
- `LinearWaRP` now treats an **empty** `UT_forward`/`UT_backward` as an identity basis, which is what
  makes arm A (both empty) and arm B (only `UT_backward`) expressible. Existing paths are unchanged.
- **Budget matching is the fairness gate**: every arm freezes the same number of scalars
  (`row: k=round(ρ·m)`, `column: k=round(ρ·n)`, entry: `round(ρ·m·n)`) — 449.6M ± 0.07% at ρ=0.1 on
  Llama-2-7B. `budget_report.json` per arm; the report tool cross-checks and refuses to call an
  unmatched comparison fair.
- **Known confound**: arm A's reparameterization is exact (identity), while B/C/D carry the ~3e-3
  bf16 basis round-trip that the published WSR-Tune also has. B-vs-C-vs-D is clean; A-vs-rest has this
  small extra term. Flagged automatically in the report.

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
