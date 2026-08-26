# NeurIPS 2026 Rebuttal Full Record — Submission 11600

## Meta Review

### Meta Review by Area Chair cF83

The paper proposes WSR-Tune, which preserves safety alignment during LLM fine-tuning by building a safety-conditioned basis that concentrates safety information into a small set of directions, then freezing those directions while updating the rest for downstream adaptation. Reviewers agree the core idea of disentangling safety and task information geometrically rather than hunting for safety-critical parameters directly in the original weight space is a genuinely novel angle, and the experiments convincingly show a better safety-performance trade-off than existing baselines.

The main concerns raised are: insufficient differentiation from closely related rank/direction-freezing work, with missing baselines and incomplete coverage of the fast-growing harmful fine-tuning defense literature; and the added compute/memory cost of per-iteration SVD. If the authors can clearly articulate what distinguishes their method from those existing studies, add the requested baselines and broaden the related work discussion, and report concrete overhead numbers (time/memory) alongside a discussion of LoRA compatibility, this would meaningfully strengthen the paper.

---

## Answer to Meta Review

Dear Area Chair,

We greatly appreciate your time and effort in handling our submission. We conducted new experiments and analyses to address every concern; we summarize the outcomes below, with full details in the individual responses.

### 1. Differentiation from rank/direction-freezing work (MWbx-W1, r2Zz-W1&Q1).

The distinction is the safety-conditioned reparameterization: importance scoring and freezing are applied to the coefficients of a complete input-side eigenbasis of the safety activations, so the same operators acquire a different functional meaning — inducing a per-output-channel projector constraint rather than the single shared projector of rank/direction-based methods. This is formalized in MWbx-W1 and analyzed mechanistically in r2Zz-W1&Q1; MWbx-W1 additionally adapts ActSVD itself into fine-tuning-stage baselines at the rank and neuron levels.

### 2. Baselines and related work (MWbx-W2&W4).

We added the requested fine-tuning-stage baselines — AsFT, Lisa, SEAL, and SafeLoRA — under the same protocol. Following the reviewers' suggestions, the revised manuscript will fully incorporate these new comparisons and a substantially broadened related-work discussion — reorganized by intervention stage (before/during/after fine-tuning) and by core mechanism — covering the suggested papers and the harmful fine-tuning defense literature beyond them.

### 3. Compute/memory overhead (XM5F-W1, MWbx-W3, r2Zz-W4).

We would like to clarify that the SVD is performed only once before fine-tuning begins — no SVD occurs during training iterations — and we will make this explicit in the revision. We report the requested concrete time/memory measurements (XM5F-W1, MWbx-W3, r2Zz-W4); the one-time cost is ≈24% of a 3-epoch fine-tuning run, amortizing further over longer training. The fine-tuned weights are merged back, so deployment requires no custom inference module.

### 4. LoRA compatibility (r2Zz-W2&Q2, r2Zz-W4).

We implemented a WSR-LoRA variant, trained it end-to-end, and report its safety-utility trade-off against PEFT baselines. WSR-LoRA also directly mitigates the overhead concern above: including the one-time basis construction, it runs at roughly 0.32× the end-to-end wall-clock time and 0.55× the peak VRAM of WSR-Tune, and reusing the fixed basis across downstream runs lowers the run-specific time further to 0.18× (r2Zz-W4).

Beyond these, additional analyses are provided based on the reviewers' suggestions: the freeze-ratio sweep extended to ρ = 50% (r2Zz-W3); safety-data ablations across source, scale, and quantity (r2Zz-W5&Q3); an ablation on the output-side basis choice V (XM5F-Q1); and clarifications of the notation, data usage, and the newly-learned-harmful-parameter discussion (MWbx-Q1/Q2/Q3). All new results and discussions will be incorporated into the revised manuscript.

Thank you again for your time and coordination.

Sincerely,

Authors of Paper 11600

---

# Reviewer 1 — XM5F

## Official Review

### Summary:

Safety finetuned models have been shown to lose their aligned behavior when finetuned for downstream tasks. The paper addresses this issue with a finetuning scheme called WSR-Tune to preserve safety even after finetuning. First the paper identifies principal directions in the activation space of the model for safety related inputs. Then the weights are reparameterized by the projecting along these directions and taken into another arbitrary basis. This ensures that these reparameterized weights mainly encode the safety-relevant information.

During finetuning, the authors identify the parameters with a large gradient with respect to the reparameterized weights and create a mask for such parameters to stop gradients. This ensures that the safety-relevant information is preserved.

The paper performs extensive experiments with multiple baselines and largely outperforms all of them, thus showing the success of this formulation.

**Contribution Type:** General: Most submissions will fall into this type.

### Strengths And Weaknesses:

**Strengths:**

- The paper proposes a novel but rather simple method which is extremely effective.
- There are extensive experiments against current baselines which shows the effectiveness of the method.
- The paper is well written

**Weaknesses**

- One weakness is the need to compute SVD, which takes time. However given the performance gain, I do not see that as a bad tradeoff. Additionally, as mentioned in the paper, efficient methods can be developed based on this.

**Quality:** 4: excellent  
**Clarity:** 4: excellent  
**Significance:** 4: excellent  
**Originality:** 4: excellent

### Questions:

- Does the basis choice of V in equation 9 impact your results ? May be some ablation studies on that will be good.
- If I understand correctly, the mask is still in the original weight space ?

**Limitations:**  
Yes

**Rating:** 5: Accept: Technically solid paper, with high potential value on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only

**Paper Formatting Concerns:**  
None

**Code Of Conduct Acknowledgement:** Yes  
**Responsible Reviewing Acknowledgement:** Yes

---

## Answer to Reviewer 1

Thank you very much for your positive feedback and valuable suggestions. Our responses are provided below.

### Q1. Does the basis choice of V in Eq. (9) impact the results?

Thank you for this insightful suggestion.

We performed the requested ablation by replacing the identity matrix ($V = I$) with two different types of randomly generated orthogonal matrices — a sparse signed permutation $V_{perm}$ and a dense Haar-random rotation $V_{dense}$ — while keeping all other settings identical. The former only relabels the output coordinates, whereas the latter fully mixes them.

|Method|JB AVG ↓|$\Delta_S$ ↓|GSM8K ↑|$\Delta_D$ ↑|Overall ↑|
|---|---:|---:|---:|---:|---:|
|Full Params FT|20.78|0.00|41.17|0.00|0.00|
|SafeInstr|13.55|-7.23|38.44|-2.73|+4.50|
|Resta|10.56|-10.22|36.92|-4.25|+5.97|
|SafeDelta|5.57|-15.21|19.79|-21.38|-6.17|
|SN-Tune|27.18|+6.40|38.99|-2.18|-8.58|
|RSN-Tune|20.73|-0.05|40.26|-0.91|-0.86|
|WSR-Tune ($V = V_{perm}$)|8.38|-12.38|39.95|-1.17|+11.21|
|WSR-Tune ($V = V_{dense}$)|8.21|-12.58|39.65|-1.47|+11.11|
|WSR-Tune ($V = I$)|6.90|-13.88|38.99|-2.18|+11.70|

- **Signed permutation $V_{perm}$.** A uniformly random permutation $\pi$ of $1,\ldots,m$ with iid random signs $s_i \in \{-1,+1\}$, i.e. $V e_i = s_i e_{\pi(i)}$. Sparse and exactly orthogonal; it permutes and flips the output coordinates without mixing them.
- **Dense random $V_{dense}$.** The QR factorization $A=QR$ of a Gaussian matrix $A_{ij}\sim\mathcal N(0,1)$, with the sign convention $V=Q\,\mathrm{diag}(\mathrm{sign}(R_{ii}))$ so that $V$ is Haar-uniform on $O(m)$. Dense and exactly orthogonal; it fully mixes all output coordinates.

The results show that all three choices of $V$ achieve a better safety-downstream trade-off than the competing baselines, indicating that WSR-Tune is largely insensitive to arbitrary rotations of the row space.

This observation is expected. Equation (9)

$$
\widetilde W = V^\top WU
$$

represents the same weight matrix under different orthonormal coordinate systems. The key component of WSR-Tune is the activation-derived basis $U$, which is constructed from the safety activation covariance and concentrates safety-relevant information. In contrast, $V$ simply rotates the row coordinates and does not introduce additional safety information.

While different choices of $V$ may produce slightly different coordinate-wise masks because masking is performed after reparameterization, our experiments show that these differences have only a minor effect on the final performance. We therefore adopt $V = I$ as the canonical design choice: it avoids introducing arbitrary, data-independent rotations, and it is also the cheapest option, since any non-identity $V$ applies an extra transform to every weight at each forward pass — a row gather for $V_{perm}$ and a full matrix multiplication for $V_{dense}$ — which made training $1.1\sim1.7\times$ slower than $V = I$ in our runs.

These results further confirm that WSR-Tune mainly relies on the activation-derived basis $U$, while the specific choice of $V$ has only a negligible impact on the overall performance.

We will include this ablation and discussion in the revised manuscript.

### Q2. If I understand correctly, the mask is still in the original weight space?

Thank you for the question. The mask is constructed in the reparameterized weight space, not in the original weight space.

Specifically, we first transform each weight matrix into the reparameterized space

$$
\widetilde W = V^\top WU,
$$

estimate safety importance using the gradients with respect to the reparameterized coefficients, and construct the binary mask on these coefficients. During downstream fine-tuning, gradients are blocked only for the masked coefficients in the reparameterized space.

When mapped back to the original parameter space, this operation does not freeze individual weight entries. Instead, it freezes structured directions defined by the basis vectors

$$
\phi_{ij}=v_i u_j^\top.
$$

This distinction is central to our method. As shown below, applying the same importance masking directly in the original weight space (without reparameterization) consistently results in inferior safety preservation compared to masking in the proposed reparameterized space.

|Category|Method|Safety: JB AVG ↓|Safety: $\Delta_S$|Downstream: GSM8K ↑|Downstream: $\Delta_D$|Overall ↑|
|---|---|---:|---:|---:|---:|---:|
|Ablation|Full parameter FT|20.78|0.00|41.17|0.00|0.00|
|Ablation|Original space + important mask FT|9.17|-11.61|40.41|-0.76|+10.85|
|Ablation|WSR-Tune|6.90|-13.88|38.99|-2.18|+11.70|

This experiment demonstrates that the improvement comes from where the mask is applied (the reparameterized basis), rather than simply freezing important parameters.

We will clarify this point more explicitly in the revised manuscript.

### W1. Computation burden of SVD

Thank you for pointing this out. We agree that the additional SVD introduces computational overhead.

Importantly, the SVD is performed only once before downstream fine-tuning begins to construct the safety-conditioned basis. The resulting basis is then kept fixed throughout training, meaning that no SVD is required during the training iterations.

We further reduce the required computation by sharing a single SVD across same-input projections. (Measured on a single NVIDIA B200 GPU.)

|Process|Time / VRAM Cost|Details|
|---|---|---|
|One-time SVD|9 min 31 seconds / 30GB|Once; shared projections|
|Downstream FT|39 min 8 seconds(3 epochs) / 96GB|13 min for 1 epoch|
|Relative Cost|0.24/0.32|Relative to total fine-tuning time|

Thus, basis construction is a one-time preprocessing cost, while the measured overhead is approximately 24% for a 3-epoch fine-tuning run. Since the SVD is performed only once, its relative cost decreases further as the number of fine-tuning epochs increases. We will add these results and discussions in our revised manuscript.

We again appreciate the reviewer for the time and effort.

---

## Reviewer 1 Follow-up

I thank the authors for the detailed answers. These answer my questions and I maintain my current positive score.

---

## Authors' Reply to Reviewer 1 Follow-up

Thank you very much for your kind acknowledgment and for letting us know that our response addressed your concerns. We sincerely appreciate your time and consideration.

---

# Reviewer 2 — MWbx

## Official Review

### Summary:

This paper propose WSR-tune, a defense against harmful fine-tuning. In the early study, there has been several defense method, e.g., Boyi Wei et al. have propose to identify key parameters for safety and freeze those parameters during fine-tuning. This paper follows this line of research, The key contribution of this paper is to constructs a safty-conditioned basis and reparameterizes the weights metrices such that the safey information concentrate in a small subset of directions such that they are easy to identify and freeze.

**Contribution Type:** General: Most submissions will fall into this type.

### Strengths And Weaknesses:

**Strengths:**

I generally like the idea of this paper. This paper does not claim that they are proposing a more "accurate" way to identify the safety parameters, but try to think in another angles of how to proactively concentrate the safety parameters by reparameterization. This idea is novel and should open new angle for future research.

Extensive experiments demonstrate the effectiveness of the method.

**Weakness:**

1. How the method in essence differs from ActSVD in Wei et al. While the storyline say that you are re-parameterize the weight for better safety direction identification, it seems that there is not a substantial difference between this and to identify safety rank. In my understanding the way you differs from ActSVD is that instead of directly freezing the rank, you reparameterize based on the rank and then use gradient norm to identify the safety coordinate and freeze them. This inessence is combining the gradient norm and the harmful activation for better identification of the safety rank. Therefore, I can not 100% buy in your storyline for the contribution against Wei et al. The author could try to convince me on this point during rebuttal.

[1] Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications

2. The baselines are not sufficient. Two of the chosen baselines Resta and SafeDelta are post-fine-tuning stage defenses, which is not at the same stage of the proposed method. I suggest to add two more fine-tuning stage defenses [2][3] as baselines. AsFT is a relevant method, which projects the fine-tuning update to the safety subspace. Lisa is an improved version of SafeInstruct.

[2] AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin [3] Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack

3. The computation overhead of this method seem to be high, as it needs to do SVD for each training iteration. Please compare with the baselines on the GPU memory consumption as well as the per step computation time.

4. There are many more papers on harmful fine-tuning defenses. While you can't iterate all the papers as baslines and compare against them, they should still be discussed in the related work section (or appendix). The literature review should be done as comprehensive as possible. I list out several papers for your reference, but there are many more papers that should be discussed in the camera ready of the paper.

AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin  
SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation  
Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey  
SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance–Diversity Data Selection  
Representation noising effectively prevents harmful fine-tuning on LLMs  
Vaccine: Perturbation-aware alignment for large language model aginst harmful fine-tuning  
Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation  
Booster: Tackling harmful fine-tuning for large language models via attenuating harmful perturbation  
Self-Destructive Language Model  
CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning  
Model Immunization from a Condition Number Perspective  
Locking Open Weight Models with Spectral Deformation  
SDD: Self-Degraded Defense against Malicious Fine-tuning  
Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention  
A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space  
SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection  
SafeGrad: Gradient Surgery for Safe LLM Fine-Tuning  
Understanding and preserving safety in fine-tuned llms

Some of them (e.g., AsFT, SafeGrad, SPF and several more) are super relevant with this finding and they are not properly discussed.

**Quality:** 3: good  
**Clarity:** 3: good  
**Significance:** 3: good  
**Originality:** 3: good

### Questions:

Eq. (9), U_l is obtained by SVD on the safety activation. However, why U_l is applicable to the reparameterize the weight metrics in Eq.(9). Could you show the dimentions of different statistic,e.g., U_l V_l, h_l, W_l?

Just to confirm, what is the data Tsafe in Eq. (7) and D_safe in Eq. (11). It is the pair of harmful question-refusal answer? If so, please carefully clarify this. Do you use in any place the harmful data pair, i.e., harmful question-harmful answer?

Just for discussion purpose: The line of research on safety/harmful coordiniates/ranks are very unclear to me now in the literature. Some of the papers even conflicit with each other. Here is something I believe is true: I personally do not buy in the idea that the harmful behaviors are learned because fine-tuning destroys the safety neurons/ranks (i.e., those learned from the harmful question-refusal answer pair). I am now championing the fact proposing in Wei et al, i.e., there are another set of harmful parameters/ranks grown by learning harmful question-harmful answer pair, and this new grown parameters and its activation suppress the safety alignment ones, resulting in safety degradation. Therefore, I peronally do not believe that tit helps mitigate the learning of new harmful parameters even though you freeze those safety parameters. What's your opinion towards this potentially biased view. Can you provide evidence to show that my view is wrong?

**Limitations:**  
Yes

**Rating:** 5: Accept: Technically solid paper, with high potential value on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only

**Paper Formatting Concerns:**  
No concerns.

**Code Of Conduct Acknowledgement:** Yes  
**Responsible Reviewing Acknowledgement:** Yes

---

## Answer to Reviewer 2

We greatly appreciate the reviewer’s thoughtful and constructive feedback.

### W1. Difference from ActSVD

We appreciate the reviewer for highlighting the connection to ActSVD. The key difference lies in the unit of preservation and where importance-based selection is performed.

Let $W_l\in\mathbb R^{d_{out}\times d_{in}}$ denote the weight matrix of the $l$-th linear projection, and let $X_l\in\mathbb R^{d_{in}\times N}$ stacks its input activations. ActSVD performs SVD on

$$
W_lX_l\in\mathbb R^{d_{out}\times N},
$$

obtains $U_r\in\mathbb R^{d_{out}\times r}$, and constructs

$$
W_l^c=U_rU_r^\top W_l.
$$

It further isolates safety-specific modifications through

$$
\Delta W_l=(I-\Pi_l^u)\Pi_l^sW_l.
$$

Thus, ActSVD treats each rank component $u_ku_k^\top W_l$. Once a singular direction is selected, all weight elements belonging to that component are jointly retained or removed. Its preservation unit is therefore a coupled rank component rather than an individual coefficient.

WSR-Tune instead derives its basis from safety-activation covariance,

$$
H_lH_l^\top=U_l\Sigma_lU_l^\top,
$$

and reparameterizes

$$
\widetilde W_l=V_l^\top W_lU_l,\qquad W_l=V_l\widetilde W_lU_l^\top.
$$

It scores and freezes element-wise coefficients $\widetilde W_l(i,j)$ using the safety-loss gradient. Its admissible update is

$$
\Delta W_l=V_l(\Delta\widetilde W_l\odot(1-M_l))U_l^\top
$$

which is not constrained to be low-rank.

With $V_l=I$, let $S_{l,i}$ be the basis indices frozen for output row $i$. The induced row-specific projector is

$$
P_{l,i}=\sum_{j\in S_{l,i}}u_ju_j^\top,
$$

and the admissible row update is $\Delta w_{l,i}^\top(I-P_{l,i})$.

Hence, different output rows can preserve different subsets of the activation basis. Instead ActSVD couples parameters through shared output-rank components. Matrix-level projection methods correspond to the restricted case of sharing a projector across a matrix dimension, while element-wise masking in the original-space corresponds to coefficient selection with $U_l=I$.

Accordingly, our novelty is not only activation SVD alone, but also applying the importance metric in the reparameterized space. Since

$$
\frac{\partial L}{\partial\widetilde W_l}=V_l^\top\frac{\partial L}{\partial W_l}U_l,
$$

coordinate-wise ranking is basis-dependent. The same gradient-based rule therefore selects a different protected subspace after reparameterization than in the original space, inducing row-specific structured projections rather than complete ActSVD rank components.

We adapted both levels from Wei et al. into a fine-tuning-stage defense. ActSVD-Tuned(rank) preserves selected output-rank components

$$
W_0X_{safe}=U\Sigma V^\top,\qquad \widehat W=UU^\top W_0,
$$

while ActSVD-Tuned(neuron) freezes the identified safety-critical neurons.

|Method|JB AVG ↓|$\Delta_S$ ↓|GSM8K ↑|$\Delta_D$ ↑|Overall ↑|
|---|---:|---:|---:|---:|---:|
|ActSVD-Tuned(rank)|15.11|-5.67|39.42|-1.75|+3.92|
|ActSVD-Tuned(neuron)|15.77|-5.01|40.33|-0.84|+4.17|
|WSR-Tune|6.90|-13.88|38.99|-2.18|+11.70|

Across both rank and neuron level adaptations, WSR-Tune achieves the best safety–utility trade-off, supporting coefficient-level selection in the reparameterized space. We will clarify this distinction and add the both ActSVD-Tuned comparison in the revision.

### W2 & W4. Fine-tuning-stage baselines and related work

We thank the reviewer for providing this extensive and highly relevant list of references. We evaluated the reviewer-suggested AsFT and Lisa, together with SEAL, and SafeLoRA, on Llama-2-7B-Chat with GSM8K under the same protocol as Table 2.

|Method|JB AVG ↓|$\Delta_S$ ↓|GSM8K ↑|$\Delta_D$ ↑|Overall ↑|
|---|---:|---:|---:|---:|---:|
|Full Params FT|20.78|0.00|41.17|0.00|0.00|
|AsFT|9.73|-11.05|23.81|-17.36|-6.31|
|Lisa|12.88|-7.90|38.67|-2.50|+5.40|
|SEAL|30.45|+9.67|38.89|-2.28|-11.95|
|SafeLoRA|8.81|-11.97|32.83|-8.34|+3.63|
|WSR-Tune|6.90|-13.88|38.99|-2.18|+11.70|

WSR-Tune achieves the lowest JB AVG while retaining 38.99 GSM8K, Pareto-dominating all added baselines.

We evaluated several suggested methods as additional baselines. In the revision, we will cover all listed papers and broaden our literature review beyond the references provided by the reviewer. We will organize the discussion by intervention stage—before, during, and after fine-tuning—and by core mechanism, including gradient/subspace constraints, perturbation-based alignment, data selection or update regularization, and attention/representation interventions. Closely related methods such as AsFT, SafeGrad, and SPF will be discussed in greater detail.

### W3. Computational burden of SVD

We agree that the additional SVD introduces computational overhead.

Importantly, SVD is performed only once before downstream fine-tuning to construct the safety-conditioned basis, which remains fixed throughout training; no SVD is required during the training iterations. We further reduce the required computation by sharing a single SVD across same-input projections. (Measured on a single NVIDIA B200 GPU.)

|Process|Time/VRAM|Details|
|---|---|---|
|SVD|9m 31s/30GB|Once; shared projections|
|Downstream FT|39m 8s/96GB|3 epochs|
|Relative cost|0.24/0.32|Time/VRAM vs. FT|

SVD adds 24% one-time time overhead over three epochs, with lower relative cost for longer training. We will report these measurements.

### Q1. Why is $U_l$ applicable to the weight reparameterization in Eq. (9)?

Let $h_l(x_i,t)\in\mathbb R^{n_l}$ denote the activation entering $W_l$ at token position $t$ of the $i$-th safety sequence $x_i$, and define

$$
H_l=[h_l(x_i,t)]_{(i,t)\in T_{safe}}\in\mathbb R^{n_l\times M}.
$$

Here, $H_l$ stacks $M$ token-level activations. The remaining dimensions are

$$
W_l\in\mathbb R^{m_l\times n_l},\quad U_l\in\mathbb R^{n_l\times n_l},\quad V_l\in\mathbb R^{m_l\times m_l},\quad \widetilde W_l\in\mathbb R^{m_l\times n_l}.
$$

With eigendecomposition

$$
H_lH_l^\top=U_l\Sigma_lU_l^\top\in\mathbb R^{n_l\times n_l},
$$

the columns of $U_l$ provide an orthonormal coordinate system for the activations associated with $W_l$. Thus, it can be applied to the right of $W_l$, while $V_l$ changes the output coordinates:

$$
\widetilde W_l=V_l^\top W_lU_l,\qquad W_l=V_l\widetilde W_lU_l^\top.
$$

Defining

$$
\widetilde h_l=U_l^\top h_l,
$$

we obtain

$$
W_lh_l=V_l\widetilde W_lU_l^\top h_l=V_l\widetilde W_l\widetilde h_l.
$$

Thus, $U_l$ is applicable because it forms an orthonormal basis of the input space of $W_l$, making $W_lU_l$ a dimensionally valid and invertible change of basis. Since $U_l$ is derived from safety activations, this basis is aligned with the activation structure observed on the safety corpus.

### Q2. Clarification of $T_{safe}$, $D_{safe}$, and harmful responses

Both are derived from the same Circuit Breakers split of harmful prompt–safe refusal pairs. $D_{safe}$ denotes sequence-level examples used for the safety loss and gradient importance in Eq. (11), whereas $T_{safe}$ denotes valid sample–token indices used to collect activations in Eq. (7).

The two stages estimate different quantities and use their natural units. Eq. (7) estimates input-activation geometry through token-level covariance, $\sum_t h_th_t^\top$. Token-wise collection preserves activation structure; sequence pooling may reduce rank, while using only the last token may bias the covariance. Eq. (11) measures the safety-loss sensitivity of each prompt–refusal example, where the loss is computed over refusal tokens and aggregated across examples to construct the mask.

WSR-Tune never uses harmful prompt–harmful response pairs. We will clarify these definitions and data construction.

### Q3. Discussion of newly learned harmful parameters/ranks

We thank the reviewer for this insightful perspective. We agree that safety degradation may result not only from disruption of existing safety-related components, but also from newly learned harmful parameters that override or suppress them, as suggested by prior studies [1,2]. These mechanisms may coexist.

Our intuition in WSR-Tune is to extend this view from the original parameter space to a reparameterized one. Freezing safety-relevant coefficients in the reparameterized space is not equivalent to neuron freezing: when mapped back, the resulting protection is distributed globally, with 99.9% of the original coordinates partially protected under a 10% freezing budget. We therefore position WSR-Tune as a pre-fine-tuning defense strategy that applies an importance metric in the reparameterized space to mitigate the emergence of harmful behavior, rather than claiming to eliminate it. Even if competing harmful parameters emerge elsewhere in the model, they must overcome the preserved safety-related representations rather than exploit a safety mechanism that has already been weakened. In this sense, although WSR-Tune may not prevent every harmful parameter from emerging, we expect that overriding the model’s existing safety behavior becomes more difficult.

From this perspective, WSR-Tune provides a distinct defense framework configured before downstream fine-tuning: it constructs the safety-conditioned basis and mask in advance and preserves the selected safety-relevant components throughout adaptation. We believe that the defense extension WSR-Tune proposes — into a new safety-conditioned space through reparameterization — can serve as a complementary axis for preventing and defending against the harmful behavior the reviewer is concerned about.

We sincerely thank the reviewer for the thoughtful feedback.

[1] Ponkshe, et al. Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study [2] Lee et al., A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity

---

## Reviewer 2 Follow-up

Hello authors,

Thanks for the rebuttal. Most of my concerns are addressed. Please add the additional baseline comparison and discussion on the related work to the camera-ready. I have increased my score accordingly.

I don't agree with positioning WSR-Tune as a pre-fine-tuning defense strategy, although you have the operation of re-parameterizing the parameter space before the fine-tuning stage. Your method is a subsequent development of Boyi Wei et al, which is a classical fine-tuning stage defense. Please try not to wrongly position the paper, as this will be very confusing to readers that intersted in the line of research of "parameter freezing"-based defense. Please also add a section in the appendix to discuss about relevant defenses that focus on the parameter freezing idea. For example, [1][2][3] is a direct application, and [4-9], etc are similar ideas that constrain the update in some parameter/rank. You may check this repo (https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers (https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers)) for a full list of papers. Please do include such revisions in the paper, as this is important for the field development and the community is smaller than you think.

[1] Safety layers in aligned large language models: The key to LLM security  
[2] Assessing the brittleness of safety alignment via pruning and low-rank modifications  
[3] Safety Alignment Shouldn't Be Complicated https://openreview.net/forum?id=9H91juqfgb  
(https://openreview.net/forum?id=9H91juqfgb)  
[4] AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin  
[5]Lisa: Lazy safety alignment for large language models against harmful fine-tuning attack  
[6] SafeAnchor: Preventing Cumulative Safety Erosion in Continual Domain Adaptation of Large Language Models  
[7] Gradient Surgery for Safe LLM Fine-Tuning  
[8] Safety alignment should be made more than just a few tokens deep  
[9] Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink

---

## Authors' Final Reply to Reviewer 2

We sincerely thank the reviewer for the follow-up comments and for increasing the score.

As suggested, we will incorporate the additional baseline comparisons presented in the rebuttal and expand the related-work discussion in the paper to cover parameter/rank-freezing and update-constraining defenses, including the studies mentioned by the reviewer.

We will also revise the manuscript to clearly position WSR-Tune as a fine-tuning-stage defense and to clarify its relationship to prior work.

Thank you again for the helpful suggestions.

---

# Reviewer 3 — r2Zz

## Official Review

### Summary:

This work addresses an important challenge in the fine-tuning of LLMs for downstream tasks, where task-specific adaptation may weaken the model's original safety alignment. To mitigate this issue, this work proposes WSR-Tune, a weight-space reparameterization framework for safety-preserving fine-tuning. The key idea is to leverage a small set of safety-related data to collect intermediate activations during forward propagation, and then apply SVD to construct an orthogonal basis that captures safety-relevant representations. The original weight matrices are subsequently projected into this reparameterized space, enabling task adaptation while preserving safety characteristics. Extensive experiments demonstrate that WSR-Tune achieves a favorable trade-off between downstream task performance and safety retention.

**Contribution Type:** General: Most submissions will fall into this type.

### Strengths And Weaknesses:

**Strengths**

The main contribution of this work lies in introducing a novel perspective that reparameterizes the weight space of LLMs using a safety-conditioned basis, thereby enabling the geometric disentanglement of safety-related knowledge from task-specific knowledge. Unlike previous approaches that identify or manipulate safety-critical neurons in the original parameter space, the proposed method provides a principled way to preserve safety alignment during fine-tuning.

**Weaknesses**

1. The core innovation of this work primarily lies in the construction of the re-parameterization space. The subsequent components, including mask-based strategies, gradient freezing, gradient computation, and importance scoring, are all commonly used techniques in existing methods and do not constitute substantial algorithmic or methodological innovations.
2. The applicability of this method is mainly limited to full-parameter fine-tuning. However, large-scale language models are more commonly adapted using PEFT methods such as LoRA. Since the proposed method relies on specific re-parameterization and operations on pre-trained weights, it is not fully compatible with the weight-freezing assumption underlying LoRA. As a result, its direct applicability under parameter-efficient fine-tuning paradigms is limited.
3. The freeze ratio introduced in this method requires manual tuning. As shown in Table 6, this hyperparameter has a significant impact on the trade-off between task performance and safety performance. As a result, the method is sensitive to hyperparameter selection in practice, which may reduce its usability and robustness.
4. As shown in Appendix B, the method requires substantially more memory and computation time than comparison methods, indicating lower computational efficiency and higher deployment cost, which may limit its practical applicability.
5. The method relies on safety data to extract activation matrices. However, the paper does not provide a thorough analysis of the data types, scale, or quantity, nor their impact on method performance.

**Quality:** 3: good  
**Clarity:** 3: good  
**Significance:** 3: good  
**Originality:** 2: not good

### Questions:

1. The main claimed innovation appears to concentrate on the re-parameterization space construction. Could the authors further clarify the actual methodological novelty of the remaining components compared to existing techniques?
2. How does the proposed method generalize to PEFT settings such as LoRA, and what modifications would be required to address the incompatibility with weight-freezing assumptions?
3. How sensitive is the method to key design choices, including the selection of safety data (type, scale, and quantity), and how do these factors affect performance, efficiency, and robustness?

**Limitations:**  
yes

**Rating:** 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only

**Paper Formatting Concerns:**  
There is no formatting concern.

**Code Of Conduct Acknowledgement:** Yes  
**Responsible Reviewing Acknowledgement:** Yes

---

## Answer to Reviewer 3

We appreciate the reviewer for the valuable comments and constructive feedback.

### W1 & Q1. Methodological novelty of WSR-Tune

We acknowledge that gradient importance scoring, masking, and freezing are individually well-established. The contribution of WSR-Tune is not a new scoring or masking operator in isolation, but the change in their functional meaning induced by the safety-conditioned coordinate system.

The reparameterization changes what the same importance metric identifies and what the same masking operation preserves; this becomes explicit once the induced constraints are mapped back to the original weight space. In Table 5, original-space masking and WSR-Tune differ only in the coordinate system of the mask, yet WSR-Tune achieves a higher overall score: re-parameterization changes the effect of an otherwise identical masking procedure.

**What changes in the original space.** With $V_l=I$ and $S_{l,i}=\{j:M_{l,ij}=1\}$, freezing coefficient $(i,j)$ imposes $e_i^\top\Delta W_lu_j=0$: each update row loses its component along the selected safety directions, through a different projector $P_{l,i}=\sum_{j\in S_{l,i}}u_ju_j^\top$ per output channel; original-space masking is the case $u_j=e_j$.

**Two consequences.** First, the budget is spent differently: original-space masking freezes each weight entirely or not at all, whereas WSR partially constrains almost every weight — at the same 10% budget, 99.9% of original coordinates receive some protection and 98.2% only partial protection, a pattern no binary mask can express. Second, the frozen and trainable parts decouple on the safety data: $U_l$ diagonalizes the safety activation covariance $C_l$, so frozen directions are uncorrelated with trainable ones ($P_{l,i}C_l(I-P_{l,i})=0$), whereas a frozen weight coordinate stays correlated with trainable ones, leaving its safety signal reachable through them. This predicts that WSR covers safety activation energy more selectively than task energy.

**Testing the prediction.** On 256 BeaverTails and 256 GSM8K examples never used to build the basis or masks, we measure the share of activation energy falling on each mask's frozen coordinates ($P_D$, in each mask's own basis; 160 projections, padding excluded), and report $R=P_{safe}/P_{task}$:

|Mask ($\rho=10$)|Basis|Safety|Task|$R$|
|---|---|---:|---:|---:|
|Original|canonical|0.321|0.287|1.12|
|WSR|safety-conditioned|0.530|0.359|1.48|

|$R$ by projection (32-layer avg.)|q|k|v|up|down|
|---|---:|---:|---:|---:|---:|
|Original|1.08|1.08|1.08|1.08|1.26|
|WSR|1.36|1.36|1.41|1.83|1.54|

A random mask gives $R=1$; the original mask barely exceeds this baseline, whereas WSR preferentially covers safety energy across all projection types. Together with Table 5, this shows that reparameterization changes what a conventional metric identifies and freezes.

### W2 & Q2. Generalization to PEFT settings such as LoRA

We implemented a WSR-LoRA variant to confirm applicability to PEFT.

**The incompatibility.** Under standard LoRA initialization the safety response $e_i^\top W_0u_j$ lies wholly in the frozen $W_0$, so masking factor entries protects nothing; WSR-LoRA restores both the content and the coordinates.

**Design.** We initialize the adapter from the rank-$r$ leading singular components of $W_0$, freeze the residual ($W_0=W_{res}+B_0A_0$, $B_0=P_r\Lambda_r^{1/2}$, $A_0=\Lambda_r^{1/2}Q_r^\top$), and apply the Eq. (9) reparameterization to the input factor: $W_t=W_{res}+B_t\widetilde A_tU^\top$ with $\widetilde A=AU$, so $W_t=W_0$ at initialization and $B_t\widetilde A_t$ plays the role of $\widetilde W$. Eqs. (11)-(12) apply unchanged to $B$ and $\widetilde A$ at the same top-$\rho$ budget, with $W_{res}$ and $U$ fixed; each response is constrained bilinearly rather than exactly.

We further compare WSR-LoRA against PEFT baselines trained end-to-end on Llama-2-7B-Chat ($q,k,v,up,down$, $r=16$, $\rho=10$, Table 2 protocol):

|Method|JB AVG ↓|$\Delta_S$ ↓|GSM8K ↑|$\Delta_D$ ↑|Overall ↑|
|---|---:|---:|---:|---:|---:|
|Full FT|20.78|0.00|41.17|0.00|0.00|
|SafeInstr|13.55|-7.23|38.44|-2.73|+4.50|
|Resta|10.56|-10.22|36.92|-4.25|+5.97|
|Vanilla LoRA|36.80|+16.02|39.20|-1.97|-17.99|
|SafeLoRA|8.81|-11.97|32.83|-8.34|+3.63|
|WSR-LoRA|10.04|-10.74|36.92|-4.25|+6.49|
|WSR-Tune|6.90|-13.88|38.99|-2.18|+11.70|

WSR-LoRA achieves the best Overall trade-off among the LoRA-based methods and also outperforms the strongest baseline in Table 2.

### W3. Sensitivity to the freeze ratio

We agree that the freeze ratio $\rho$ is an important hyperparameter governing the safety--utility trade-off. To characterize it beyond Table 6, we extended the sweep to substantially larger ratios, up to $\rho=50$:

|Method|Overall ↑|
|---|---:|
|SafeInstr|+4.50|
|Resta|+5.97|
|SafeDelta|-6.17|
|SN-Tune|-8.58|
|RSN-Tune|-0.86|
|WSR-Tune (1%)|+11.61|
|WSR-Tune (5%)|+12.28|
|WSR-Tune (10%)|+11.70|
|WSR-Tune (15%)|+12.73|
|WSR-Tune (20%)|+12.38|
|WSR-Tune (30%)|+11.95|
|WSR-Tune (40%)|+12.67|
|WSR-Tune (50%)|+12.31|

WSR-Tune remains robust across the evaluated range of $\rho$. Every evaluated $\rho$ from 1% to 50% outperforms the strongest baseline in Table 2. We will include the extended sweeps in the revised manuscript.

### W4. Memory, computation time, and deployment cost

We agree that the additional SVD introduces computational overhead.

Importantly, SVD is performed only once before downstream fine-tuning to construct the safety-conditioned basis, which remains fixed throughout training; no SVD is required during the training iterations. We further reduce the required computation by sharing a single SVD across same-input projections. (Measured on a single NVIDIA B200 GPU.)

|Process|Time/VRAM|Details|
|---|---|---|
|SVD|9m 31s/30GB|Once; shared projections|
|Downstream FT|39m 8s/96GB|3 epochs|
|Relative cost|0.24/0.32|Time/VRAM vs. FT|

SVD adds 24% one-time time overhead over three epochs, with lower relative cost for longer training.

WSR-LoRA further reduces this cost: including the one-time basis construction, it requires approximately 0.32× the end-to-end wall-clock time and 0.55× the peak VRAM of WSR-Tune, and reusing the fixed basis lowers the run-specific time further to 0.18×. We will report these measurements.

### W5 & Q3. Sensitivity to the safety data - type, scale, and quantity

We conducted extensive ablations on the offline safety data along three axes: (1) the safety-dataset source - we additionally constructed a BeaverTails-based safety dataset, screening harmful prompts and harmful responses with Llama-Guard-3-8B, and matched the number of examples to Circuit Breakers for a fair comparison; (2) the type and quantity of the data used to extract the activation matrices (basis) and to compute the importance scores; and (3) crossing the source of the basis-extraction data with the source of the importance-scoring data. (CB: Circuit Breakers, BV: BeaverTails.)

**Constructed Circuit Breakers Dataset**

|Method|JB AVG ↓|$\Delta_S$ ↓|GSM8K ↑|$\Delta_D$ ↑|Overall ↑|Details|
|---|---:|---:|---:|---:|---:|---|
|Full Params FT|20.78|0.00|41.17|0.00|0.00||
|SafeInstr|13.55|-7.23|38.44|-2.73|+4.50||
|Resta|10.56|-10.22|36.92|-4.25|+5.97||
|SafeLoRA|8.81|-11.97|32.83|-8.34|+3.63||
|WSR-LoRA|10.04|-10.74|36.92|-4.25|+6.49|Ours|
|WSR-Tune (Full)|6.90|-13.88|38.99|-2.18|+11.70|Ours|
|WSR-Tune (50%)|9.09|-11.69|40.29|-0.88|+10.81|Importance data: 50% of CB|
|WSR-Tune (10%)|7.01|-13.77|39.95|-1.22|+12.55|Importance data: 10% of CB|
|WSR-Tune-BV|8.81|-11.97|41.02|-0.15|+11.82|Importance data: BV (cross)|
|WSR-Tune (Basis 50%)|8.79|-11.99|40.86|-0.31|+11.68|Basis data: 50% of CB|
|WSR-Tune (Basis 25%)|6.08|-14.70|40.94|-0.23|+14.93|Basis data: 25% of CB|

With CB as the safety source, varying the importance-scoring data - 50% or 10% of CB, or replacing it with BV - keeps WSR-Tune within +10.81 to +12.55 Overall, and WSR-LoRA reaches +6.49: both stay above every baseline, showing robustness to the quantity and the source of the importance data.

**Constructed BeaverTails Dataset**

|Method|JB AVG ↓|$\Delta_S$ ↓|GSM8K ↑|$\Delta_D$ ↑|Overall ↑|Details|
|---|---:|---:|---:|---:|---:|---|
|Full Params FT|19.16|0.00|41.85|0.00|0.00||
|SafeInstr|8.92|-10.24|40.33|-1.52|+8.72||
|RESTA|18.56|-0.60|39.35|-2.50|-1.90||
|SafeDelta|7.57|-11.59|24.49|-17.36|-5.77||
|SN-Tune|21.54|+2.38|41.09|-2.86|-5.24||
|RSN-Tune|24.64|+5.48|41.24|-1.59|-7.07||
|SafeLoRA|8.81|-10.35|32.83|-9.02|+1.33||
|WSR-LoRA|12.93|-6.23|41.55|-0.30|+5.93|Ours|
|WSR-Tune (Full)|7.18|-11.98|41.09|-0.76|+11.22|Ours|
|WSR-Tune (50%)|7.88|-11.28|40.86|-0.99|+10.29|Importance data: 50% of BV|
|WSR-Tune (10%)|8.00|-11.16|40.86|-0.99|+10.17|Importance data: 10% of BV|
|WSR-Tune-CB|7.52|-11.64|40.86|-0.99|+10.65|Importance data: CB (cross)|
|WSR-Tune (Basis 50%)|10.38|-8.78|40.18|-1.67|+7.11|Basis data: 50% of BV|
|WSR-Tune (Basis 25%)|9.23|-9.93|38.21|-3.64|+6.29|Basis data: 25% of BV|

With the entire pipeline built on BV, WSR-Tune again attains the best trade-off (+11.22), and every importance-data variation (50%, 10%, or crossed to CB) stays above the strongest baseline (SafeInstr, +8.72). Reducing the basis-extraction data to 50%/25% degrades Overall only moderately (+7.11/+6.29) while still improving safety over Full FT by 9-10 JB points, and WSR-LoRA remains well ahead of its LoRA counterpart (SafeLoRA, +1.33).

In summary, WSR-Tune and WSR-LoRA maintain favorable and stable trade-offs under changes to the safety-data source, the scale of the basis-extraction data, the quantity of the importance-scoring data, and crossed combinations of the two. We will include these in the revision.

We sincerely appreciate the reviewer for the constructive feedback.

---

## Reviewer 3 Follow-up

Thank you for the authors’ detailed response, which has addressed most of my major concerns.

However, regarding the novelty of the proposed method, the authors claim that reparameterization gives existing tools new meanings. However, this essentially transfers existing techniques, including mask-based strategies, gradient freezing, gradient computation, and importance scoring, into a new coordinate system. This appears to be a migration of existing approaches to a different application scenario, rather than the introduction of a new constraint mechanism or optimization paradigm. Therefore, the novelty of the proposed method is limited.

Furthermore, the reported results suggest that WSR-LoRA achieves weaker safety recovery performance compared with SafeLoRA. In fact, this issue can be observed in most of the safety-related results reported in the paper, where the proposed method does not consistently outperform existing approaches in terms of safety metrics alone. The authors mainly evaluate different methods using an overall metric that combines safety and task performance. However, such a metric can be highly dependent on the characteristics of the downstream task.For example, on tasks such as SST-2, where the performance gap between different fine-tuning methods may be relatively small, it remains unclear whether the proposed method can still maintain an overall advantage under such scenarios.

Meanwhile, the evaluation of the proposed method is largely limited to the GSM8K benchmark (with only limited results on Math), while its effectiveness on other widely adopted benchmarks, such as AGNEWS and AlpacaEval, is not demonstrated. Therefore, further evaluation on diverse tasks is necessary to validate the general applicability and practical effectiveness of the proposed method.

Therefore, I will maintain my current score.

---

## Response to Additional Comment (1/3)

Thank you for considering our response and allowing us to address the reviewer's concerns.

### 1. Novelty of WSR-Tune

We appreciate the comment. We again acknowledge that masking, gradient freezing, and importance scoring are established techniques. Here, we do not claim that WSR-Tune introduces any of these components as a new optimization primitive. Rather, its novelty lies in elevating safety-conditioned weight-space reparameterization into a general framework that can host existing protection operators and alter the update geometry they induce.

**Existing viewpoint: Which components should be protected?**

Most existing safety methods ask which parameters, neurons, or directions should be protected in the original parameterization. Our starting question was whether the original parameterization is an appropriate space in which to make this selection. When safety-relevant information is distributed across canonical weight coordinates, even an accurate importance criterion may leave other safety-relevant directions exposed to downstream updates.

Recent studies further motivate this concern. Wei et al. [1] show that freezing localized safety-critical components does not prevent renewed vulnerability under subsequent fine-tuning. Ponkshe et al. [2] find that safety and task signals are geometrically entangled rather than isolated in a stable safety-only subspace. Bach et al. [3] show that harmful loss geometry remains accessible after fine-tuning. These results motivate moving beyond protecting fixed components in the original parameterization, while leaving open how distributed safety geometry should be incorporated into selective adaptation.

**Our perspective shift: designing the space in which protection is applied**

WSR-Tune therefore shifts the question from which parameters or directions in the original space should be protected? to which safety-conditioned basis allows a given protection rule to induce an appropriate structured constraint after mapping back to the original weight space?

WSR-Tune addresses this question by constructing a complete safety-conditioned basis before downstream adaptation, applying a selection-and-protection operator to coefficients in that basis, and keeping both the basis and the protected coefficients fixed during fine-tuning. Importantly, WSR-Tune does not assume the existence of a globally separable safety-only subspace. The basis spans the full input space, while the coefficients selected for protection can differ across output channels.

Under a 10% freezing budget, 99.9% of original coordinates are affected by at least one protected basis coefficient, while 98.2% are partially constrained rather than independently frozen. Moreover, 99.22% of rows and 95.74% of columns mix frozen and trainable coefficients. These results mean that the reparameterized mask induces coupled linear constraints whose support reaches almost all original coordinates, unlike a 10% binary mask applied directly in the original space.

**WSR-Tune is not tied to a particular importance score or masking method**

We would also like to emphasize that WSR-Tune is not tied to a particular importance score or masking strategy. Although the gradient-magnitude criterion used in our main implementation proved to be highly effective, it represents only one instantiation of the framework rather than its defining contribution. In principle, coefficient-, neuron-, rank-, or update-selection methods can all be applied in the reparameterized space while retaining their original selection rules.

For example, Table 3 in our manuscript applies SN-Tune's safety-neuron identification mechanism after WSR reparameterization. By combining SN-Tune with WSR-Tune, JB Avg and GSM8K accuracy are improved on Llama-2-7B-Chat (27.18/38.99→23.07/41.02) and Llama-2-13B-Chat (28.97/48.22→24.69/48.75); WSR-Tune similarly improves SafeInstr in both settings. These results show that WSR-Tune is not merely another competing importance-scoring or masking strategy; it is a complementary framework in which existing safety-preservation methods can operate more effectively.

**Positioning of our contribution**

From these viewpoints, we position WSR-Tune as a complementary framework for safety-preserving fine-tuning, whose novelty lies in the reparameterization-selection composition and the resulting admissible-update geometry, not in an individual scoring, masking, or freezing primitive. A small coefficient budget in a constructed basis becomes a distributed constraint in the original space, and the same framework can improve multiple existing safety-preservation strategies without redesigning their native mechanisms.

[1] Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications ICML'24  
[2] Safety Subspaces Are Not Linearly Distinct: A Fine-Tuning Case Study ICLR'26  
[3] Curvature-Aware Safety Restoration in LLM Fine-Tuning TMLR'26

---

## Response to Additional Comment (2/3)

### 2. Generalization to Additional Downstream Tasks

Following the reviewer's call for evaluation on more diverse tasks, we extend our evaluation beyond mathematical reasoning to MedQA and ARC-C under the same protocol, comparing WSR-Tune and WSR-LoRA against both PEFT and full-parameter baselines.

WSR-Tune performs strongly on both safety and downstream task performance on MedQA and ARC-C. On MedQA, it attains the lowest JB AVG among all methods (5.45) while maintaining 43.28 accuracy, only 1.02 points below Full Params FT. On ARC-C, it improves both safety and accuracy over Full Params FT (JB AVG 20.64→11.29; accuracy 56.31→58.53), performing on par with the strongest baseline, SafeInstr (11.22/58.45), on both metrics.

WSR-LoRA achieves the best safety performance among the evaluated PEFT methods while remaining competitive in accuracy on both MedQA and ARC-C. It surpasses SaLoRA on both safety and accuracy on both tasks (e.g., JB AVG 8.51 vs. 18.90 on MedQA; 11.96 more accuracy points on ARC-C), and compares favorably with SafeLoRA: better on both metrics on ARC-C, and a 5.74-point lower JB AVG on MedQA at a 1.10-point accuracy cost.

Taken together, these results suggest that WSR-Tune and WSR-LoRA jointly preserve safety and downstream-task performance across tasks with different characteristics - including settings where the performance gaps among methods are small - supporting their broader applicability and practical effectiveness.

**[ MedQA ]**

|Method|JB AVG ↓|$\Delta_S$ ↓|MedQA ↑|$\Delta_D$ ↑|Overall ↑|
|---|---:|---:|---:|---:|---:|
|Full Params FT|7.77|0.00|44.30|0.00|0.00|
|**PEFT-based Methods**||||||
|Vanilla LoRA|11.97|+4.20|43.44|-0.86|-5.06|
|SaLoRA|18.90|+11.13|43.28|-1.02|-12.15|
|SafeLoRA|14.25|+6.48|44.70|+0.40|-6.08|
|WSR-LoRA|8.51|+0.74|43.60|-0.70|-1.44|
|**Full Parameter Methods**||||||
|SafeInstr|7.57|-0.20|43.83|-0.47|-0.27|
|Resta|6.21|-1.56|41.01|-3.29|-1.73|
|SafeDelta|5.50|-2.27|33.62|-10.68|-8.41|
|SN-Tune|18.93|+11.16|42.89|-1.41|-12.57|
|RSN-Tune|16.01|+8.24|43.44|-0.86|-9.10|
|WSR-Tune|5.45|-2.32|43.28|-1.02|+1.30|

**[ ARC-C ]**

|Method|JB AVG ↓|$\Delta_S$ ↓|ARC-C ↑|$\Delta_D$ ↑|Overall ↑|
|---|---:|---:|---:|---:|---:|
|Full Params FT|20.64|0.00|56.31|0.00|0.00|
|**PEFT-based Methods**||||||
|Vanilla LoRA|47.58|+26.94|49.57|-6.74|-33.68|
|SaLoRA|24.97|+4.33|43.16|-13.15|-17.48|
|SafeLoRA|16.19|-4.45|53.57|-2.74|+1.71|
|WSR-LoRA|14.98|-5.66|55.12|-1.19|+4.47|
|**Full Parameter Methods**||||||
|SafeInstr|11.22|-9.42|58.45|+2.14|+11.56|
|Resta|16.08|-4.56|56.48|+0.17|+4.73|
|SafeDelta|8.26|-12.38|36.52|-19.79|-7.41|
|SN-Tune|44.11|+23.47|59.73|+3.42|-20.05|
|RSN-Tune|31.83|+11.19|59.13|+2.82|-8.37|
|WSR-Tune|11.29|-9.35|58.53|+2.22|+11.57|

---

## Response to Additional Comment (3/3)

### 3. Safety performance and the safety–downstream trade-off

We first correct a mistake in our previous response: the SafeLoRA row of the BeaverTails table was reported with incorrect values. The correct values are **13.25 JB AVG and 38.82 GSM8K**, and the tables below are updated accordingly. We apologize for the error.

WSR-Tune is designed for practical downstream adaptation, where **safety preservation and task performance must be maintained jointly.** Consistent with Eq. (3), we therefore evaluate whether a method minimizes safety degradation without sacrificing downstream performance. This criterion addresses the practical failure mode in which improving one axis substantially degrades the other. We treat raw JB AVG, downstream performance, and their empirical Pareto relation as the primary evidence, with Overall only as a supplementary scalar interpreted alongside both axes.

For a broader controlled comparison, we additionally evaluate Vanilla LoRA, SaLoRA, and SafeLoRA under the Table 2 protocol. In both settings below, **WSR-Tune lies on the empirical Pareto frontier among all evaluated methods, while WSR-LoRA is non-dominated among the evaluated PEFT methods.** The ARC-C and MedQA results in our preceding response provide a complementary test: the same frontier positions persist when downstream-performance differences among competitive methods are relatively small.

**Circuit Breakers**

| Method         | JB AVG ↓ | $\Delta\_S$ ↓ | GSM8K ↑   | $\Delta\_D$ ↑ | Overall ↑  |
| -------------- | -------- | ------------- | --------- | ------------- | ---------- |
| Full Params FT | 20.78    | 0.00          | 41.17     | 0.00          | 0.00       |
| Vanilla LoRA   | 36.80    | +16.02        | **39.20** | **-1.97**     | -17.99     |
| SaLoRA         | 30.75    | +9.97         | 38.74     | -2.43         | -12.40     |
| SafeLoRA       | 8.81     | -11.97        | 32.83     | -8.34         | +3.63      |
| **WSR-LoRA**   | 10.04    | -10.74        | 36.92     | -4.25         | +6.49      |
| **WSR-Tune**   | **6.90** | **-13.88**    | 38.99     | -2.18         | **+11.70** |

**BeaverTails**

| Method         | JB AVG ↓ | $\Delta\_S$ ↓ | GSM8K ↑   | $\Delta\_D$ ↑ | Overall ↑  |
| -------------- | -------- | ------------- | --------- | ------------- | ---------- |
| Full Params FT | 19.16    | 0.00          | 41.85     | 0.00          | 0.00       |
| Vanilla LoRA   | 17.43    | -1.73         | 39.95     | -1.90         | -0.17      |
| SaLoRA         | 21.13    | +1.97         | 39.04     | -2.81         | -4.78      |
| SafeLoRA       | 13.25    | -5.91         | 38.82     | -3.03         | +2.88      |
| **WSR-LoRA**   | 12.93    | -6.23         | **41.55** | **-0.30**     | +5.93      |
| **WSR-Tune**   | **7.18** | **-11.98**    | 41.09     | -0.76         | **+11.22** |

**Performance across adaptation regimes.** WSR performs consistently well in both PEFT and full-parameter fine-tuning. Across Circuit Breakers, BeaverTails, ARC-C, and MedQA, WSR-LoRA remains on the empirical PEFT frontier, while WSR-Tune is non-dominated among the evaluated full-parameter methods. **In BeaverTails, WSR-LoRA outperforms both SafeLoRA and SaLoRA on both raw axes, achieving lower JB AVG and higher GSM8K accuracy.** The additional ARC-C and MedQA results reported in our response to Comment 2 show a similar pattern: on ARC-C, **WSR-LoRA is both safer and more accurate than SafeLoRA and SaLoRA**; on MedQA, it likewise outperforms SaLoRA on both axes and substantially improves safety over SafeLoRA. These results show that safety-conditioned reparameterization yields favorable safety–utility trade-offs across PEFT and full-parameter fine-tuning.

These results across diverse safety sources and downstream datasets show that **WSR-Tune and WSR-LoRA achieve stronger safety preservation and better downstream performance than the baselines in most settings.**

We again appreciate the reviewer for the time and effort. In case there are remaining questions/concerns, it would be grateful if we can have an opportunity to further answer them.

---

## Additional AGNEWS Response

In addition to the MedQA and ARC-C results provided above, we additionally evaluated WSR-Tune on AGNEWS as the reviewer suggested.

**[ AGNEWS ]**

| Method                     | JB AVG ↓ | $\Delta\_S$ ↓ | AGNews ↑ | $\Delta\_D$ ↑ | Overall ↑ |
| -------------------------- | -------: | ------------: | -------: | ------------: | --------: |
| Full Params FT             |    14.45 |          0.00 |    94.00 |          0.00 |      0.00 |
| **PEFT-based Methods**     |          |               |          |               |           |
| SafeLoRA                   |     5.72 |         -8.73 |    83.20 |        -10.80 |     -2.07 |
| SaLoRA                     |     9.71 |         -4.74 |    92.70 |         -1.30 |     +3.44 |
| **WSR-LoRA**               |     7.98 |         -6.47 |    93.30 |         -0.70 | **+5.77** |
| **Full Parameter Methods** |          |               |          |               |           |
| Resta                      |     8.04 |         -6.41 |    93.60 |         -0.40 |     +6.01 |
| SafeDelta                  |     7.97 |         -6.48 |    91.10 |         -2.90 |     +3.58 |
| SN-Tune                    |    53.15 |        +38.70 |    94.00 |          0.00 |    -38.70 |
| RSN-Tune                   |    51.00 |        +36.55 |    94.40 |         +0.40 |    -36.15 |
| **WSR-Tune**               |     5.23 |         -9.22 |    93.78 |         -0.22 | **+9.00** |

Importantly, AGNEWS corresponds closely to the setting described by the reviewer, where downstream accuracies are already very similar across methods (most methods lie within 92-94).

**WSR-Tune achieves the lowest JB AVG among all evaluated methods (5.23) while keeping downstream accuracy within 0.22 points of Full Params FT**, yielding the best overall (+9.00). We observe that although SafeLoRA attains a comparable JB AVG (5.72), this comes at the cost of a 10.80-point drop in downstream accuracy (94.00 → 83.20) which exhibits a less favorable safety-utility trade-off. In contrast, WSR-LoRA maintains an accuracy of 93.30 while achieving a JB AVG of 7.98, resulting in the best overall score among the PEFT-based methods (+5.77). These results indicate that WSR-LoRA achieves a more favorable safety–utility trade-off.

Combined with the **MedQA and ARC-C results in our previous response**, WSR-Tune and WSR-LoRA now show consistent safety–utility behavior across five downstream tasks spanning **mathematical reasoning (GSM8K, MATH)**, **medical and scientific QA (MedQA, ARC-C)**, and **text classification (AGNEWS)**. We will include the AGNEWS comparison in the revision. If any further questions remain, we would be grateful for the opportunity to address them.
