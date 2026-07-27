# Handoff — LoRA-family safety baselines (Safety-WaRP-LLM)

> **For the next Claude Code (different environment).** This summarizes what was done, the
> problems hit, the fixes applied, and what still needs doing for the LoRA / SaLoRA / LISA
> baselines that compare against the paper's **WSR-Tune (WaRP)** method.
> Read `CLAUDE.md` and `WaRP.md` first for the pipeline model.

---

## 0. TL;DR — what the next agent must know

1. **The committed scripts are machine-specific** (hardcoded `/home/...` paths, `conda activate hb`,
   fixed `CUDA_VISIBLE_DEVICES`). They will **not** run as-is on a new box. See §2.
2. **Code is faithful, but our experiment *configs* deviated from each method's paper** — this,
   not any bug, explains the odd results (esp. SaLoRA safety collapse). See §5.
3. **LoRA-family needs LoRA-scale LR (1e-4~3e-4), not full-FT LR (1e-5/5e-5).** Low LR → underfit. See §4.
4. **WaRP Phase 3 (full-param `non_freeze`) OOMs on a 95 GB GPU** — must use
   `--gradient_checkpointing --batch_size 1 --gradient_accumulation_steps 16`. See §3.
5. Everything trained so far is on HF under **`kmseong/...`** (see §6). HF account = `kmseong`,
   token in env var `HF_TOKEN`.

---

## 1. Goal

Produce downstream (GSM8K) fine-tunes of a **safety-aligned** Llama-2-7B-chat and measure the
**safety (ASR) vs utility (GSM8K acc) trade-off**, to serve as *baselines* against the paper's
WSR-Tune. Methods: **vanilla LoRA**, **SaLoRA**, **LISA**, plus **WSR-Tune (WaRP)** itself.

- **Base model (all runs):** `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` (HF hub id; first run downloads it).
- **Safety data:** `data/circuit_breakers_train.json` (committed, force-tracked).
- **Downstream:** GSM8K (`openai/gsm8k`, `main`, clean, no poison).

---

## 2. Environment gotchas (fix these FIRST on a new box)

The original authors' box used conda env `hb` + `/home/yonsei_jong` | `/home/gokms0509` paths.
On the box this handoff was written on (a Vast.ai container), the fixes were:

| Problem | Original (broken on new box) | Fix applied |
|---|---|---|
| No conda | `source /home/.../conda.sh; conda activate hb` | use a venv python directly: `PY=/venv/hb/bin/python` |
| Bare `python` | `python train.py ...` | `"$PY" train.py ...` |
| Hardcoded safety-data path | `/home/gokms0509/Safety-WaRP-LLM/data/...` | `/root/Safety-WaRP-LLM/data/circuit_breakers_train.json` (use YOUR repo root) |
| Hardcoded agnews/medqa paths | `/home/yonsei_jong/.../*.jsonl` | only matters if PHASE3_DATASET=agnews/medqa; ignore for gsm8k |
| GPU index | `CUDA_VISIBLE_DEVICES=3` | set to a free GPU on your box |

**On the new environment:** find a python with `torch + transformers + peft + datasets + wandb`
(these baselines do **not** need `trl`). Versions that worked here: torch 2.10+cu128,
transformers 4.57.3, peft 0.18.1, wandb 0.23. Point every script's `PY`/python at it.

**HF upload:** `HF_TOKEN` env var must be set and belong to (or can push to) `kmseong`.
Upload pattern used everywhere: `huggingface_hub.create_repo(exist_ok=True)` + `upload_folder(...)`.

**GPU:** this box had a single ~95 GB GPU. Full-param training (WaRP `non_freeze`) is memory-heavy (§3).

---

## 3. WaRP (WSR-Tune) pipeline — what runs, and the OOM fix

WSR-Tune = the paper's method (see `WaRP.md`, `CLAUDE.md`). Pipeline = `train.py --phase {1,2,3}`.
Phase 3 for gsm8k uses `--non_freeze --constrained_sft` = **full-parameter** training with
WaRP element-wise masking (only "flat" directions move; safety directions frozen) + a
shallow-token constrained SFT loss.

- **Adapted integrated runner:** `scripts/run_all_phases_seq.sh` (a machine-adapted copy of
  `scripts/run_all_phases_integrated.sh`, with the §2 fixes + a **sequence-wise basis** option).
- **`git` status:** `run_all_phases_integrated.sh` (committed) is the original; `run_all_phases_seq.sh`
  is the adapted copy created here.

### 3a. OOM fix (IMPORTANT)
Phase 3 `non_freeze` fine-tunes all 7B params → AdamW fp32 optimizer states (~54 GB) + model (13.5) +
grads (13.5) ≈ 81 GB, and `batch_size=2` activations pushed it **over 95 GB → CUDA OOM at ~step 362**.
`expandable_segments:True` was already set (it's genuine pressure, not fragmentation). Fix that worked:

```
--gradient_checkpointing --batch_size 1 --gradient_accumulation_steps 16   # eff.batch 16 unchanged
```
Peak dropped to ~90.5 GB (fits). `run_all_phases_seq.sh` Phase 3 was patched to use
`PHASE3_BATCH_SIZE=1`, `PHASE3_GRAD_ACCUM=16`, `--gradient_checkpointing` by default.
If you still OOM on a smaller GPU, drop to 8-bit AdamW (`--optim` via TrainingArguments / bitsandbytes).

### 3b. Sequence-wise basis (added here)
`--basis_granularity {token,sequence}` + `--seq_pool {mean,last,sum}` were added to `train.py`
(new builder `models/phase1_basis_sequence.py`). Default is `token` (original). `sequence` pools each
sequence to one vector before the covariance. Standalone runner: `scripts/run_phase1_basis_sequence.sh`.
The produced model `warp-seq-mean-kr0.1-lr5e-5` used sequence/mean.

---

## 4. Key finding #1 — LoRA-family LR was too low

First sweep used **lr {1e-5, 5e-5}** (a *full-FT* LR range). LoRA only updates a low-rank adapter, so
that LR **underfits** → low GSM8K (0.19–0.30). Re-sweeping at proper **LoRA LR {1e-4, 2e-4, 3e-4}**
(rank 16) raised GSM8K to ~0.32–0.38. **Use 1e-4~3e-4 for LoRA-family, not 1e-5/5e-5.**
(Eval is apples-to-apples: gsm8k 5-shot + chat template; strict≈flexible → scores are genuine.)

---

## 5. Key finding #2 — our configs deviated from each method's PAPER (code is faithful)

Two independent audits (reference repos at `/root/Lisa`, `/root/SaLoRA`) concluded **both
`models/salora.py` and `gsm8k_eval/finetune_gsm8k_lisa.py` are faithful, bug-free reproductions**
of the original methods. **BUT the hyperparameters we swept diverged from the papers**, which
explains the poor / unsafe results:

| | **Paper default** | **What we ran (sweeps)** | Effect |
|---|---|---|---|
| **SaLoRA target modules** | **`q_proj, v_proj` only** | `q,k,v,up,down` (5 types) | applies safety projector to unvalidated FFN modules |
| **SaLoRA alpha/scaling** | **`α = r` → scaling 1** | `α = 2r` → scaling 2 | doubled update overwhelms the C safety projection |
| **LISA target modules** | **`q, v` only** | `q,k,v,up,down` | — |
| **LISA rank/alpha** | **r=8, α=4** | r=16, α=32 | — |
| LISA BSO knobs (rho, steps) | rho=1, align=100, ft=900 | **same ✓** | faithful |

**Consequence observed:** at high LR, SaLoRA's safety *collapsed* (lr3e-4 ASR ≈ 0.69, worse than
plain LoRA) — because we ran it **off its designed operating point** (5 modules + scaling 2), not
because of a bug. LISA never learned downstream (gsm8k stuck ~0.16) because **rho=1.0 is a strong
proximal pull** (by design), independent of LR.

**Inherent tension:** you cannot be *faithful to each paper* AND *config-matched across methods*
simultaneously. Decide per experiment and **label clearly**.

**Recommended faithful baselines (TODO):**
- **SaLoRA:** `--layer_type attn_q,attn_v` (q,v only) + `--lora_alpha == --lora_r` (e.g. r16/α16, s=1).
- **LISA:** q,v targets + r8/α4 (rho=1 already correct).

**Cosmetic bug (non-blocking):** `finetune_gsm8k_salora.py --target_modules` (default `q,k,v,up,down`)
is **unused** — conversion is driven solely by `--layer_type`. It's only echoed to `summary.json`.
Don't be misled by it; the effective targets are whatever `--layer_type` says.

---

## 6. Results so far (safety-utility trade-off)

Two eval CSVs (ASR via HarmBench Direct/AutoDAN/PAIR/PAP; GSM8K via lm-eval 5-shot):
`/root/run_logs/FINAL_combined_summary.csv` (low-LR sweep + warp-seq) and
`/root/run_logs/FINAL_lr1e4_summary.csv` (r16, lr1e-4~3e-4).

- **Low LR (1e-5/5e-5):** underfit corner — gsm8k 0.16–0.30, ASR 0.09–0.21 (safe but weak).
- **High LR (1e-4~3e-4):** LoRA/SaLoRA reach gsm8k 0.32–0.38 but ASR climbs to 0.30–0.69 (unsafe);
  **LISA stays gsm8k ~0.16** (rho too strong) with ASR ~0.19–0.22.
- **WSR-Tune `warp-seq-mean-kr0.1-lr5e-5`: gsm8k 0.379, ASR 0.110** — matches best-LoRA utility at
  ~¼ the ASR. This is the paper's thesis working (only method in the good corner).

Full-FT reference (same base, plain SFT, lr5e-5, ep3): gsm8k ≈ 0.38.

---

## 7. Artifacts produced (all on HF, account `kmseong`)

**First sweep — 12 models** `llama2_7b-chat-gsm8k-{lora,lisa,salora}-lr{1e-5,5e-5}-r{8,16}`
(SaLoRA here used α=r=16; LoRA/LISA α=2r).

**Re-sweep — 9 models (r16, proper LR, α unified to 2r=32 for cross-method fairness):**
`llama2_7b-chat-gsm8k-{lora,salora,lisa}-lr{1e-4,2e-4,3e-4}-r16`.

**WSR-Tune:** `kmseong/llama2_7b-chat-gsm8k-warp-seq-mean-kr0.1-lr5e-5`.

Local copies under `gsm8k_eval/sweep/` were **deleted after verifying each was on HF** (freed 264 GB).
Re-download from HF if needed.

### Scripts created here (all carry THIS box's paths — adapt for your env)
- `gsm8k_eval/sweep_lisa_lora.sh`, `gsm8k_eval/sweep_salora.sh` — first sweep (r8/16 × lr1e-5/5e-5).
- `gsm8k_eval/sweep_lr1e4_r16.sh` — re-sweep (r16 × lr1e-4/2e-4/3e-4). LoRA loop commented out (already done).
- `gsm8k_eval/run_lisa_gsm8k.sh` — single LISA run (env-fixed).
- `scripts/run_all_phases_seq.sh` — full WaRP pipeline (sequence-wise + OOM fix).
- `scripts/run_phase1_basis_sequence.sh`, `models/phase1_basis_sequence.py` — sequence-wise basis.

---

## 8. Reference method configs (exact, for faithful re-runs)

**LoRA** (`finetune_gsm8k_lora.py`, repo root, `--method lora`): merged model at `<out>/merged_model`.
```
--target_modules q_proj,k_proj,v_proj,up_proj,down_proj  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
--learning_rate {1e-4|2e-4|3e-4} --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --max_length 1024
```

**SaLoRA** (`finetune_gsm8k_salora.py`, repo root; layer in `models/salora.py`): merged at `<out>/merged_model`.
- Faithful: `--layer_type attn_q,attn_v --lora_r 16 --lora_alpha 16 --lora_dropout 0.0`
  (`rank_safe=32, rank_util=32, calib=128` defaults). C = I−VsVsᵀ from safety acts; PiSSA+utility-proj init.

**LISA** (`gsm8k_eval/finetune_gsm8k_lisa.py`): merged model saved to `<out>` directly.
- Faithful LoRA cfg: `--lora_target_modules q_proj v_proj --lora_r 8 --lora_alpha 4`.
- BSO: `--rho 1.0 --alignment_step 100 --finetune_step 900 --guide_data_num 4994` (matches paper).
- Caveat: step counters are per-*micro-batch*; keep `alignment_step`/`finetune_step` divisible by
  `grad_accum` (100,900 are ÷4-safe) or state switches split accumulation windows.

**Invariant (WaRP only):** `--layer_type` and `--target_layers` MUST be identical across Phases 1/2/3.

---

## 9. Evaluation

- **Utility:** lm-evaluation-harness at `/root/lm-evaluation-harness`, gsm8k **5-shot + chat template**
  (`eval_models.sh`; `TASK_FEWSHOT[gsm8k]=5`). strict-match ≈ flexible-extract here.
- **Safety (ASR):** HarmBench at `/root/HarmBench`, methods Direct/AutoDAN/PAIR/PAP.
- **Combined driver:** `/root/run_all_eval.sh` (edit its `REPOS=(...)` array). It runs both benchmarks
  per model. **Note:** it uses conda envs `harmbench` / `hb` and prints `🎉 통합 평가 종료` at the end.
  It processes models sequentially with gaps between them — wait for the driver process to fully exit
  (and the completion marker), not just GPU idle, before assuming it's done. Recovery re-runs may relaunch.

---

## 10. Suggested next steps (for the new environment)

1. **Adapt** the §7 scripts to your box (python path, GPU index, repo-root data path).
2. **Faithful baselines** (§5/§8): re-run SaLoRA (q,v, α=r) and LISA (q,v, r8/α4) so each method sits
   at its paper operating point; compare against the config-matched versions already on HF.
3. Optionally sweep LR per method to draw each method's **safety-utility Pareto curve**, then overlay
   WSR-Tune (`warp-seq`) to show it dominates.
4. If touching WaRP Phase 3: keep the §3a memory settings.
5. (Optional) clean up the cosmetic unused `--target_modules` in `finetune_gsm8k_salora.py`.
