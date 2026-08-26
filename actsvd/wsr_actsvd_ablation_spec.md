# WSR-Tune vs ActSVD: Mask-Structure Ablation (Rebuttal Experiment)

## 0. One-line goal
Prove, with numbers, that WSR-Tune is **not** just a re-skin of ActSVD (Wei et al., 2024)
by running ActSVD-style rank-freezing and WSR-Tune's entry-freezing **inside the exact
same fine-tuning pipeline**, changing only the coordinate basis and the mask granularity.

This single experiment answers three reviewers at once:
- **MWbx Q1** ("how does this differ from ActSVD?") — the headline target.
- **XM5F Q1** ("does the choice of V matter?") — arm B uses a non-identity output basis.
- **r2Zz originality=2** ("only novelty is the reparam space") — isolates the space's contribution.

Key rhetorical asset: Wei et al. footnote 9 states that rank-level *freezing* "cannot be
easily achieved using U^u and U^s." WSR-Tune's reparameterization is exactly what makes it
achievable. Frame the result as *implementing the operation ActSVD's own authors said was hard*,
not as "beating" ActSVD.

---

## 1. Background / definitions (read before coding)

### WSR-Tune (ours)
- Collect input activations to layer `W_l`: `H_l = [h_l(x_i,t)] ∈ R^{n_l × M}` over the safety corpus.
- SVD the **input** covariance: `H_l H_l^T = U_l Σ_l U_l^T`, so `U_l ∈ R^{n_l × n_l}` lives in the **input** space.
- Reparameterize: `W̃_l = V_l^T W_l U_l`, with `V_l = I` in the main method. So effectively `W̃_l = W_l U_l` (RIGHT multiply).
- Importance: `G_l = Σ_{x∈D_safe} |∂L/∂W̃_l|` (per-coefficient gradient magnitude on safety loss).
- Freeze top-ρ coefficients (entry-wise) via stop-grad; train the rest on downstream data.

### ActSVD (Wei et al.) — the thing we're differentiating from
- SVD the **output** activation: `U S V^T ≈ W X_in`, so `U ∈ R^{d_out × r}` lives in the **OUTPUT** space.
- Applied by LEFT multiply / projection: `Ŵ = U U^T W`. It is a **removal / low-rank** operator, not a training constraint.
- Their safety isolation additionally uses utility data via orthogonal projection `ΔW = (I − Π^u) Π^s W`. (Optional to replicate — see §4.)

### The crucial mapping
In WSR-Tune's framework, the ActSVD analogue is **V = output-side ActSVD basis + ROW-wise freezing**
(freezing whole output directions), NOT column-wise on U. Do not conflate the two sides of the matrix.

| Arm | Basis | Mask unit | Corresponds to |
|-----|-------|-----------|----------------|
| A   | original space (U=I, V=I) | entry, top-|grad| | SN-Tune-style / Table 5 original-space masking |
| B   | V = ActSVD output basis, U=I | **row** (whole output dir) | **ActSVD-style rank freezing** ← main comparison |
| C   | U = safety input basis, V=I | column (whole input dir) | input-side subspace constraint |
| D   | U = safety input basis, V=I | **entry**, top-|grad| | **WSR-Tune (ours)** |

Expected story: D ≥ B > A, with B > A showing "changing the space helps" and D ≥ B showing
"entry granularity adds further on top." Any of the three outcomes (D>B, D≈B, D<B) is reportable;
see §7.

---

## 2. Fixed setup (identical across all arms)
- **Model:** Llama-2-7B-Chat (safety-tuned checkpoint, same as main paper).
- **Downstream:** GSM8K, 3 epochs, same LR / batch / grad-accum as main WSR-Tune runs.
- **Target modules:** attention q,k,v + MLP up,down (same as paper §4.1).
- **Safety corpus:** the **refusal-response split** of Circuit Breakers (same as paper). Do NOT use any harmful-completion split.
- **Loss for importance:** safety loss on **response tokens only** (standard SFT masking). Prompt tokens masked out (matches Wei et al. §2.1 convention).
- **Activation collection for bases:** token-wise, response tokens only, same token set that defines T_safe.

### CRITICAL — budget matching (do this or the comparison is unfair)
Equalize the **number of frozen scalar parameters** across arms.
- WSR-Tune (D) freezes `ρ · m_l · n_l` entries per layer. Use ρ = 0.10 (paper default).
- Row-freezing (B): freeze `k` output rows ⇒ `k · n_l` entries. Set `k = round(ρ · m_l)` so total frozen = ρ·m_l·n_l.
- Column-freezing (C): freeze `k` input columns ⇒ `k · m_l` entries. Set `k = round(ρ · n_l)`.
- Original-space entry (A): same ρ, entry-wise.
Log the actual frozen-parameter count per layer for every arm and assert they match within ±1%.

---

## 3. Per-arm construction details

### Arm A — original space, entry mask
- No reparameterization. Compute `G = |∂L_safe/∂W|` directly on W.
- Freeze top-ρ entries per layer (per-row top-ρ to match paper's per-output grouping is fine — state which you use).
- Train remaining entries on GSM8K with stop-grad on frozen ones.
- This should reproduce the "Original space + important mask FT" row already in the paper's Table 5 (JB≈9.17, GSM8K≈40.41). USE THAT AS A CORRECTNESS CHECK — if arm A diverges from Table 5, there is a bug.

### Arm B — ActSVD basis, ROW mask (the key arm)
1. Reuse the H_l collection code, but form the **output** activation `W_l H_l ∈ R^{m_l × M}`.
2. SVD: `U_out S V^T ≈ W_l H_l`, take `U_out ∈ R^{m_l × m_l}` (left singular vectors, output space).
3. Set `V_l = U_out`. Reparameterize `W̃_l = U_out^T W_l` (LEFT multiply; U on input side = I here).
4. Compute `G̃ = |∂L_safe/∂W̃_l|`. Rank the **rows** by aggregate importance (e.g. row L2 of G̃, or Σ over row). Freeze the top-`k` whole rows (k from §2 budget).
5. Train remaining coefficients on GSM8K; reconstruct `ΔW = U_out ΔW̃` each step (or keep model in reparam space and map back at the end).
- This is the "rank-level freezing that Wei et al. said was hard" — now trivial because reparam turns directions into coordinates.

### Arm C — safety input basis, COLUMN mask
- Standard WSR-Tune reparam `W̃ = W U_l` (U from H_l H_l^T, input side).
- Rank **columns** by aggregate importance, freeze top-`k` whole columns (budget-matched).
- This is the input-side subspace-constraint analogue (AlphaEdit / Safe-LoRA family live near here).

### Arm D — WSR-Tune (ours), entry mask
- Standard method exactly as in the paper. ρ = 0.10, entry-wise top-gradient freeze.
- Should reproduce paper Table 2 (JB≈6.90, GSM8K≈38.99). CORRECTNESS CHECK against that row.

### Sanity arm (cheap, do if time) — signed-permutation V
- Set V = a signed permutation matrix, U = safety input basis, entry mask.
- Result MUST equal arm D exactly (only relabels rows). If it differs → implementation bug. Great free correctness test for the reparam code.

---

## 4. Optional stronger arm (only if utility corpus is handy)
Arm B-full: replicate ActSVD's actual safety isolation — also compute output basis on a **utility**
corpus `Π^u` and freeze rows via orthogonal projection `(I − Π^u)Π^s`-style selection instead of
raw safety ranking. This closes the "you didn't disentangle safety from utility like Wei et al." rebuttal.
Skip tonight if it needs a new dataset; the safety-only Arm B already carries the main point.

---

## 5. Evaluation (keep it cheap tonight)
- **Safety (ASR):** Direct + PAP only for the sweep (static prompts, cheap). Run full Direct+AutoDAN+PAIR+PAP only on the final chosen arms for the camera-ready table. Report refusal-keyword ASR to match main text; if the Beaver-Dam classifier pipeline is wired, also log it.
- **Downstream:** GSM8K 5-shot accuracy (same harness as paper).
- Report per arm: `[Direct, PAP, AVG-ASR, GSM8K, #frozen-params]`.

---

## 6. Caching / runtime notes
- `U_l` (input side, for C & D) and `U_out` (output side, for B) each computed ONCE per layer, cached to disk, reused across all runs. Do NOT recompute per step (that would falsely inflate the overhead story and waste hours).
- Store bases in bf16. WARNING: output/input bases are large — e.g. up_proj U is ~11008² ≈ 242 MB/layer in bf16, ~7–10 GB across 32 layers for q/k/v/up/down. If OOM, offload bases to CPU or exclude up_proj from the largest arm first. Log peak GPU mem + basis-storage size — this doubles as the answer to the MWbx/r2Zz "memory overhead" question.
- Per-arm training ≈ same order as WSR-Tune main run (~20 min for 3 epochs on the paper's B200). Basis construction (SVD over target layers) ≈ 12 min one-time.

---

## 7. Reporting template (fill after runs)
Produce ONE table, budget-matched, ρ=10%:

| Arm | Basis | Mask | #Frozen | Direct | PAP | ASR-AVG | GSM8K |
|-----|-------|------|---------|--------|-----|---------|-------|
| A original+entry     |   | entry  |   |   |   |   |   |
| B ActSVD-basis+row   |   | row    |   |   |   |   |   |
| C safety-basis+col   |   | column |   |   |   |   |   |
| D WSR-Tune (ours)    |   | entry  |   |   |   |   |   |

Interpretation to write depending on outcome:
- **D > B:** entry-granularity in the safety-conditioned space beats rank-freezing; the gain is not rank identification but (neuron × direction) coordinate preservation.
- **D ≈ B:** both benefit from reparameterization; WSR-Tune's contribution is making rank-level freezing *implementable* (Wei et al. footnote 9) via an exact invertible coordinate change.
- **D < B:** report honestly; suggests moving from entry to rank masking — important to know regardless.
In all cases, B > A is the load-bearing result: "changing the space, not just which params to freeze, is what matters" — this directly extends the paper's Table 5.

---

## 8. Correctness checklist (assert these before trusting results)
1. Arm A ≈ paper Table 5 "Original space + important mask FT" (JB≈9.17, GSM8K≈40.41).
2. Arm D ≈ paper Table 2 WSR-Tune (JB≈6.90, GSM8K≈38.99).
3. Signed-permutation sanity arm == Arm D exactly.
4. Frozen-param counts equal across A/B/C/D within ±1%.
5. `U_out` is left-singular of `W_l H_l` (output space, m_l×m_l); `U_l` is eigenbasis of `H_l H_l^T` (input space, n_l×n_l). Do not swap them.
6. Safety loss / importance uses response tokens only; no harmful-completion data anywhere.

---

## 9. Priority if the night is short
1. Arms A, B, D at ρ=10%, Direct+PAP only. (This alone answers MWbx Q1.)
2. Add Arm C.
3. Add signed-permutation sanity + full 4-attack eval on A/B/D.
4. Arm B-full (utility-disentangled) — only if utility corpus is ready.
