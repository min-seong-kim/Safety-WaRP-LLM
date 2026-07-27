# Adapter-Aware Column-WSR-LoRA 구현 요청서

## 0. 목적

현재 저장소에는 다음 두 계열이 존재한다.

1. `wsr_lora`
   - safety activation으로 구성한 WSR basis \(U_l\) 사용
   - reparameterized weight에서 원소별 safety mask 사용
   - WSR-Tune에 가장 충실하지만, Hadamard mask가 LoRA의 low-rank 구조를 깨뜨릴 수 있음

2. `adapter_subspace_lora`
   - safety LoRA update \(\Delta W_{s,l}=s_sB_{s,l}A_{s,l}\) 자체를 compact SVD
   - right singular subspace \(Q_{S,l}\)를 보호
   - rank는 보존되지만, WSR의 activation basis와 gradient importance를 사용하지 않음

이번에 구현할 새 방법의 목표는 다음과 같다.

> **Safety LoRA update를 safety-conditioned WSR basis로 회전하고, 그 회전된 adapter coefficient의 safety importance를 측정한 뒤, 중요 column directions를 downstream LoRA가 사용하지 못하도록 한다.**

원소별 WSR mask를 그대로 적용하면 LoRA rank가 깨질 수 있으므로, importance를 column 단위로 집계하고 오른쪽 projection으로 적용한다.

임시 방법명:

```text
adapter_aware_wsr_projected_lora
```

또는

```text
adapter_wsr_column_lora
```

코드 수정 전에 현재 repository의 Phase 1 basis construction, Phase 2 importance computation, `wsr_lora`, `adapter_subspace_lora`, `ProjectionCallback`, PEFT adapter loading/merging 경로를 먼저 읽고 구현 계획을 제시해 달라.

---

# 1. 문제 설정

각 target linear module \(l\)에 대해

\[
W_{\mathrm{base},l}\in\mathbb R^{m_l\times n_l}
\]

를 원본 Llama-2-7B-Chat weight라고 한다.

Safety LoRA update는

\[
\Delta W_{s,l}
=
s_sB_{s,l}A_{s,l},
\]

\[
B_{s,l}\in\mathbb R^{m_l\times r_s},
\qquad
A_{s,l}\in\mathbb R^{r_s\times n_l},
\qquad
s_s=\frac{\alpha_s}{r_s}.
\]

Safety-tuned model은

\[
W_{\mathrm{safe},l}
=
W_{\mathrm{base},l}
+
\Delta W_{s,l}.
\]

Safety adapter는 학습 이후 고정한다. Downstream GSM8K 학습에서는 새로운 adapter

\[
\Delta W_{d,l}
=
s_dB_{d,l}A_{d,l}
\]

만 학습한다.

최종 모델은

\[
W_{\mathrm{final},l}
=
W_{\mathrm{base},l}
+
\Delta W_{s,l}
+
\Delta W_{d,l}^{\perp}
\]

형태다.

---

# 2. 핵심 방법

## 2.1 Safety model에서 WSR basis 구성

Safety adapter가 활성화되거나 merge된 전체 safety model

\[
W_{\mathrm{safe}}
=
W_{\mathrm{base}}+\Delta W_s
\]

에 safety dataset \(D_{\mathrm{safe}}\)를 입력한다.

각 target module \(l\)에 들어가는 token-wise input activation을 모아

\[
H_l
=
[h_l(x_i,t)]
\in\mathbb R^{n_l\times M}
\]

를 구성하고,

\[
H_lH_l^\top
=
U_l\Lambda_lU_l^\top
\]

를 계산한다.

여기서

\[
U_l\in\mathbb R^{n_l\times n_l}
\]

은 원 논문의 safety-conditioned activation basis다.

중요:

- \(U_l\)는 safety adapter 자체의 SVD basis가 아니다.
- \(U_l\)는 반드시 \(W_{\mathrm{safe}}\)에서 safety data activation을 수집해 만든다.
- Base model만으로 activation을 수집하면 안 된다.
- 기존 WSR Phase 1 artifact가 현재 safety LoRA로 만든 \(W_{\mathrm{safe}}\)와 정확히 일치할 때만 재사용한다.
- 모델 checkpoint, adapter, target modules, tokenizer, dataset preprocessing이 다르면 basis를 재생성한다.

---

## 2.2 Safety adapter를 WSR basis로 회전

Safety adapter update를 WSR input basis로 표현한다.

\[
\widetilde{\Delta W}_{s,l}
=
\Delta W_{s,l}U_l
=
s_sB_{s,l}(A_{s,l}U_l).
\]

이는 근사가 아니라 정확한 좌표변환이다.

\[
\Delta W_{s,l}
=
\widetilde{\Delta W}_{s,l}U_l^\top.
\]

이 회전된 matrix의 \(j\)-번째 column은

\[
\widetilde{\Delta W}_{s,l}[:,j]
=
s_sB_{s,l}(A_{s,l}u_{l,j})
\]

이며, safety adapter가 WSR direction \(u_{l,j}\)에 대해 layer mapping을 얼마나 변경했는지 나타낸다.

큰 dense \(\widetilde{\Delta W}_{s,l}\)를 전체 materialize하지 않아도 된다. 필요하면 basis column chunk \(J\)에 대해

\[
\widetilde{\Delta W}_{s,l}[:,J]
=
s_sB_{s,l}(A_{s,l}U_l[:,J])
\]

만 계산한다.

---

# 3. Importance 정의

## 3.1 Strict WSR column importance

원 논문의 reparameterized weight는

\[
\widetilde W_{\mathrm{safe},l}
=
W_{\mathrm{safe},l}U_l.
\]

Safety loss gradient는

\[
G_l
=
\sum_{x\in D_{\mathrm{safe}}}
\left|
\frac{\partial\mathcal L_{\mathrm{safe}}(x)}
{\partial\widetilde W_{\mathrm{safe},l}}
\right|.
\]

Chain rule에 의해

\[
\frac{\partial\mathcal L}
{\partial\widetilde W_{\mathrm{safe},l}}
=
\frac{\partial\mathcal L}
{\partial W_{\mathrm{safe},l}}U_l.
\]

원 논문은 \(G_l(i,j)\)를 원소별로 top-\(\rho\) 선택한다. 이번 방법에서는 LoRA rank를 보존하기 위해 column별로 집계한다.

기본 column score:

\[
c^{\mathrm{grad}}_{l,j}
=
\left\|G_l[:,j]\right\|_2.
\]

선택적으로 L1 집계도 지원한다.

\[
c^{\mathrm{grad,L1}}_{l,j}
=
\sum_i G_l(i,j).
\]

이 모드는 원 논문의 gradient importance를 column-wise structured mask로 변환한 버전이다.

단, 이 score만 사용하면 safety adapter의 update magnitude는 선택에 직접 반영되지 않는다. \(W_{\mathrm{safe}}=W_{\mathrm{base}}+\Delta W_s\)에서 \(\Delta W_s\)를 별도 변수로 보더라도 gradient 자체는 전체 weight에 대한 gradient와 동일하기 때문이다.

따라서 아래 adapter-aware score를 주 방법으로 구현하고, gradient-only를 ablation으로 둔다.

---

## 3.2 Adapter-aware Taylor importance — 주 방법

Safety adapter를 실제로 활용하려면 회전된 adapter magnitude와 safety gradient를 결합한다.

원소별 score:

\[
T_l
=
\left|
\widetilde{\Delta W}_{s,l}
\right|
\odot
G_l.
\]

즉,

\[
T_l(i,j)
=
\left|
\widetilde{\Delta W}_{s,l}(i,j)
\right|
\cdot
\sum_{x\in D_{\mathrm{safe}}}
\left|
\frac{\partial\mathcal L_{\mathrm{safe}}(x)}
{\partial\widetilde W_{\mathrm{safe},l}(i,j)}
\right|.
\]

이는 해당 adapter coefficient를 제거하거나 변경했을 때의 first-order safety-loss sensitivity를 근사하는 Taylor-style score다.

Column score 기본값:

\[
\boxed{
c^{\mathrm{Taylor}}_{l,j}
=
\left\|
T_l[:,j]
\right\|_2
}
\]

또는 L1:

\[
c^{\mathrm{Taylor,L1}}_{l,j}
=
\sum_i T_l(i,j).
\]

해석:

- \(\left|\widetilde{\Delta W}_{s,l}(i,j)\right|\): safety LoRA가 그 WSR coefficient를 실제로 얼마나 변경했는가
- \(G_l(i,j)\): 현재 safety loss가 그 coefficient에 얼마나 민감한가
- \(T_l(i,j)\): safety adapter가 실제로 사용했고 safety loss에도 민감한 coefficient인가

기본 `importance_mode`는 다음으로 한다.

```text
adapter_taylor
```

지원할 ablation:

```text
gradient_only
adapter_magnitude_only
adapter_taylor
```

`adapter_magnitude_only`의 column score는 예를 들어

\[
c^{\mathrm{mag}}_{l,j}
=
\left\|
\widetilde{\Delta W}_{s,l}[:,j]
\right\|_2.
\]

현재의 `adapter_subspace_lora`와는 다르다. 여기서는 safety adapter의 자체 SVD basis가 아니라, WSR activation basis \(U_l\) 위에서 magnitude를 측정한다.

---

# 4. Gradient importance 계산 시 중요한 구현 조건

원 논문 정의를 유지하려면 sample 또는 mini-batch별 reparameterized gradient에 absolute value를 취한 뒤 누적해야 한다.

올바른 형태:

\[
G_l
\leftarrow
G_l
+
\left|
G_{W,l}^{(b)}U_l
\right|
\]

여기서 \(G_{W,l}^{(b)}\)는 현재 batch의 weight gradient다.

다음은 일반적으로 동일하지 않으므로 사용하면 안 된다.

\[
\left(
\sum_b|G_{W,l}^{(b)}|
\right)U_l.
\]

즉, 원래 공간 gradient의 absolute sum을 먼저 구한 뒤 마지막에 rotation하면 안 된다.

기존 Phase 2 코드가 이미

\[
\sum_b\left|G_{W,l}^{(b)}U_l\right|
\]

를 정확히 계산한다면 재사용한다.

기존 Phase 2가 binary mask만 저장하고 raw \(G_l\)를 버린다면 다음 중 하나를 구현한다.

1. raw \(G_l\) 또는 column score를 저장하도록 Phase 2 확장
2. adapter-aware column score를 batch마다 직접 누적
3. 기존 binary mask의 column count를 주 방법에 사용하지 말 것  
   이는 정보 손실이 큰 약한 fallback이므로 별도 ablation일 때만 허용

---

# 5. 메모리 효율적인 adapter-aware score 계산

`down_proj`에서

\[
\widetilde{\Delta W}_{s,l}
\in\mathbb R^{4096\times11008}
\]

를 전체 materialize하면 크다.

가능하면 column chunk 단위로 처리한다.

Basis column chunk \(J\)에 대해:

\[
U_J=U_l[:,J],
\]

\[
\widetilde{\Delta W}_{s,l}[:,J]
=
s_sB_{s,l}(A_{s,l}U_J).
\]

Gradient artifact가 raw \(G_l\)로 저장되어 있다면 같은 chunk

\[
G_l[:,J]
\]

를 읽어

\[
T_l[:,J]
=
\left|
s_sB_{s,l}(A_{s,l}U_J)
\right|
\odot
G_l[:,J]
\]

를 계산하고 바로 column score로 축약한다.

예:

\[
c_j
=
\sqrt{
\sum_i T_l(i,j)^2
}.
\]

전체 \(T_l\)를 저장할 필요가 없다.

권장 chunk size:

```text
64 / 128 / 256
```

GPU 또는 CPU 메모리에 따라 argument로 노출한다.

최종 artifact에는 최소한 다음만 저장한다.

- module name
- selected \(U_{S,l}\)
- 전체 column score \(c_l\)
- selected indices \(S_l\)
- activation basis metadata
- selection rule 및 \(k_l\)
- importance mode
- aggregation mode
- safety adapter path/checksum
- safety model path/checksum
- dataset/preprocessing metadata

---

# 6. Column-wise mask와 original-space projection의 동치

보호할 WSR direction index 집합을 \(S_l\)라 하고

\[
U_{S,l}
=
U_l[:,S_l]
\in\mathbb R^{n_l\times k_l}
\]

로 정의한다.

WSR 좌표계에서 direction mask를

\[
D_{F,l}
=
\mathrm{diag}(d_1,\dots,d_{n_l}),
\]

\[
d_j=
\begin{cases}
0,&j\in S_l,\\
1,&j\notin S_l
\end{cases}
\]

라고 한다.

Raw downstream LoRA update를 original space에서

\[
X_l=s_dB_{d,l}A_{d,l}
\]

라고 하면 WSR 좌표계에서는

\[
\widetilde X_l
=
X_lU_l.
\]

Column mask 적용:

\[
\widetilde X_l^{\mathrm{allowed}}
=
\widetilde X_lD_{F,l}.
\]

Original space로 복원하면

\[
X_l^{\mathrm{allowed}}
=
X_lU_lD_{F,l}U_l^\top.
\]

\(U_l\)가 orthogonal이므로

\[
U_lD_{F,l}U_l^\top
=
I-U_{S,l}U_{S,l}^\top.
\]

따라서

\[
\boxed{
\Delta W_{d,l}^{\perp}
=
s_dB_{d,l}A_{d,l}
\left(
I-U_{S,l}U_{S,l}^\top
\right)
}
\]

이다.

즉, original space에서는 projection이지만 WSR coordinate에서는 선택한 basis columns를 0으로 만드는 mask다.

---

# 7. Rank 보존

오른쪽 projection은 rank를 증가시키지 않는다.

\[
\operatorname{rank}
\left(
B_{d,l}A_{d,l}(I-U_SU_S^\top)
\right)
\le
\operatorname{rank}(B_{d,l}A_{d,l})
\le r_d.
\]

따라서 원소별 WSR mask와 달리 LoRA adapter 구조를 유지한다.

학습 후 projected factor를

\[
A_{d,l}^{\perp}
=
A_{d,l}
-
(A_{d,l}U_{S,l})U_{S,l}^\top
\]

로 저장하면

\[
\Delta W_{d,l}^{\perp}
=
s_dB_{d,l}A_{d,l}^{\perp}.
\]

일반 PEFT adapter로 저장 및 merge 가능해야 한다.

---

# 8. Direction 선택 규칙

각 module 내부에서 column score \(c_{l,j}\)를 큰 순서로 정렬한다.

다음 모드를 지원한다.

## 8.1 Ratio mode — WSR budget 비교용

\[
k_l
=
\left\lceil
\rho_{\mathrm{dir}}n_l
\right\rceil.
\]

\[
S_l
=
\operatorname{TopK}(c_l,k_l).
\]

Column mask에서는 \(k_l/n_l\) 비율의 columns 전체를 freeze하므로, 전체 matrix coefficient 중 동일한 비율을 freeze한다.

따라서 원 논문의 freeze ratio와 coordinate-budget 관점에서 비교할 수 있다. 다만 원소별 mask와 granularity는 다르다.

권장 실험:

```text
direction_keep_ratio = 0.01
direction_keep_ratio = 0.05
direction_keep_ratio = 0.10
```

예:

- \(n=4096\): \(k\approx41,205,410\)
- \(n=11008\): \(k\approx111,551,1101\)

## 8.2 Fixed top-k mode — adapter rank와 비교용

```text
top_k = 2
top_k = 4
top_k = 8
top_k = 16
```

특히 `top_k=16`은 기존 `adapter_subspace_lora`의 all-effective 보호 차원과 수를 맞추는 비교에 유용하다. 단, 선택 basis와 중요도는 완전히 다르다.

## 8.3 Score-energy mode

Column score의 제곱 누적 또는 합 누적이 threshold를 넘는 최소 \(k_l\)를 선택한다.

예:

\[
\frac{\sum_{j=1}^{k_l}c_{l,(j)}^2}
{\sum_jc_{l,j}^2}
\ge\tau.
\]

지원 예:

```text
score_energy = 0.90
score_energy = 0.95
score_energy = 0.99
```

우선순위:

1. `direction_top_k`가 지정되면 fixed top-k
2. 아니면 `direction_keep_ratio`가 지정되면 ratio
3. 아니면 `direction_score_energy`가 지정되면 score-energy
4. 아무것도 없으면 오류 또는 명시적 default 사용

원 논문과의 직접 비교가 목적이면 ratio mode를 기본으로 한다.

---

# 9. Downstream LoRA 제약 구현

각 target module에서 다음 제약을 유지한다.

\[
A_{d,l}U_{S,l}=0.
\]

절대로 \(n_l\times n_l\) projector를 만들지 않는다.

Projection:

\[
\boxed{
A_{d,l}
\leftarrow
A_{d,l}
-
(A_{d,l}U_{S,l})U_{S,l}^\top
}
\]

## 9.1 초기 projection

Downstream LoRA 생성 직후 \(A_d\)를 한 번 projection한다.

PEFT 기본 초기화에서 \(B_d=0\)이더라도, \(A_d\) 자체는 forbidden component를 포함할 수 있으므로 초기 projection을 수행한다.

## 9.2 Gradient projection

각 accumulation 이후 optimizer가 사용하기 전에

\[
g_{A,l}
\leftarrow
g_{A,l}
-
(g_{A,l}U_{S,l})U_{S,l}^\top.
\]

기존 `register_post_accumulate_grad_hook` 구조가 있다면 재사용한다.

## 9.3 Optimizer step 이후 parameter projection

AdamW의 원소별 second-moment scaling은 부분공간을 보존하지 않으므로, parameter projection을 모든 optimizer step 후 수행한다.

\[
A_{d,l}
\leftarrow
A_{d,l}
-
(A_{d,l}U_{S,l})U_{S,l}^\top.
\]

기존 `adapter_subspace_lora`의 post-step projection infrastructure를 최대한 재사용하되, \(Q_S\) 대신 \(U_S\)를 사용한다.

선택적으로 Adam `exp_avg`도 projection할 수 있으나, 최소 구현에서는 gradient projection과 post-step parameter projection을 모두 수행하면 된다.

---

# 10. 최종 모델과 보존 조건

최종 layer는

\[
W_{\mathrm{final},l}
=
W_{\mathrm{base},l}
+
\Delta W_{s,l}
+
\Delta W_{d,l}^{\perp}.
\]

여기서

\[
\Delta W_{d,l}^{\perp}
=
s_dB_{d,l}A_{d,l}(I-U_{S,l}U_{S,l}^\top).
\]

따라서

\[
\Delta W_{d,l}^{\perp}U_{S,l}=0.
\]

결과적으로

\[
\boxed{
W_{\mathrm{final},l}U_{S,l}
=
W_{\mathrm{safe},l}U_{S,l}
}
\]

가 성립한다.

정확한 해석:

- 선택된 WSR input directions에 대한 해당 linear layer의 mapping이 safety model과 동일하게 유지된다.
- 이것은 model-level output이나 ASR을 직접 보장하지 않는다.
- 실제 safety preservation은 HarmBench ASR로 검증해야 한다.

---

# 11. Safety adapter 처리 순서

권장 실행 순서:

1. Base model 로드
2. Safety LoRA adapter 로드
3. Safety adapter를 활성화한 상태로 \(W_{\mathrm{safe}}\) 구성
4. Phase 1 safety activation basis \(U_l\) 생성
5. Safety adapter \(B_s,A_s\)와 \(U_l\)를 이용해 \(\widetilde{\Delta W}_{s,l}\) 계산
6. Safety dataset backward로 \(G_l\) 계산
7. Adapter-aware column score와 \(U_{S,l}\) 저장
8. Safety adapter를 merge하거나 frozen adapter로 유지
9. 새 GSM8K downstream LoRA 생성
10. \(A_dU_S=0\) 제약 아래 학습
11. PEFT adapter와 dense merged model 모두 저장 가능하게 구성

Safety adapter를 merge하기 전에 반드시 \(B_s,A_s\)를 읽을 수 있어야 한다.

실험군과 baseline은 동일한 \(W_{\mathrm{safe}}\)에서 출발해야 한다.

---

# 12. Target modules

기존 WSR 실험과 맞추기 위해 기본 target modules:

```text
q_proj
k_proj
v_proj
up_proj
down_proj
```

Llama-2-7B 기준 32 layers × 5 modules = 160 modules.

Phase 1, importance, downstream LoRA에서 target module set이 정확히 일치해야 한다.

Safety adapter에 target module이 누락된 경우:

- 기본: 오류 발생
- 명시적 `--allow_missing_safety_adapter_modules`가 있을 때만 해당 module을 gradient-only WSR 또는 unconstrained LoRA로 처리
- 어떤 module이 빠졌는지 report에 기록

---

# 13. CLI 제안

`finetune_gsm8k_lora.py` 또는 현재 method dispatcher에 다음 method를 추가한다.

```bash
--method adapter_aware_wsr_projected_lora
```

필수/권장 argument:

```bash
--model_name meta-llama/Llama-2-7b-chat-hf
--safety_adapter_path PATH
--safety_merged_model_path PATH

--wsr_basis_dir PATH
--adapter_wsr_importance_dir PATH
--save_adapter_wsr_artifacts PATH

--importance_mode adapter_taylor
# adapter_taylor | gradient_only | adapter_magnitude_only

--column_aggregation l2
# l2 | l1

--direction_keep_ratio 0.01
--direction_top_k 16
--direction_score_energy 0.90

--importance_chunk_size 128
--importance_dtype float32
--basis_dtype float32
--projection_dtype float32

--project_adapter_gradients
--project_adapter_after_step

--lora_param_dtype float32
--require_safety_adapter_for_all_targets
```

선택 규칙 argument는 동시에 여러 개를 주면 오류를 내거나 명확한 우선순위를 로그에 출력한다.

---

# 14. Artifact 구조 제안

```text
outputs/adapter_aware_wsr_projected_lora/
  safety_adapter/
  safety_merged/

  phase1_basis/
    layer_00_q_proj.pt
    ...
    report.json

  importance/
    adapter_taylor/
      ratio_0.01/
      ratio_0.05/
      ratio_0.10/
    gradient_only/
    adapter_magnitude_only/

  subspaces/
    <importance_mode>_<selection>/
      layer_00_q_proj.pt
      ...
      report.json

  runs/
    <importance_mode>_<selection>/
      summary.json
      subspace_verification.json
      adapter/
      merged_model/
```

각 module artifact:

```python
{
    "module_name": ...,
    "U_S": ...,                  # n × k
    "selected_indices": ...,
    "column_scores": ...,
    "k": ...,
    "n": ...,
    "importance_mode": ...,
    "column_aggregation": ...,
    "selection_mode": ...,
    "selection_value": ...,
    "basis_orthogonality_error": ...,
    "subspace_orthogonality_error": ...,
    "safety_adapter_scaling": ...,
    "metadata": ...
}
```

---

# 15. 필수 검증

## 15.1 Basis orthogonality

\[
\frac{
\|U_{S,l}^\top U_{S,l}-I\|_F
}{
\sqrt{k_l}
}
\]

또는 absolute error를 기록한다.

## 15.2 Factor-level constraint

\[
\mathrm{constraint}_{A,l}
=
\frac{
\|A_{d,l}U_{S,l}\|_F
}{
\|A_{d,l}\|_F+\epsilon
}.
\]

## 15.3 Update-level constraint

\[
\mathrm{constraint}_{\Delta,l}
=
\frac{
\|B_{d,l}A_{d,l}U_{S,l}\|_F
}{
\|B_{d,l}A_{d,l}\|_F+\epsilon
}.
\]

## 15.4 Mapping preservation

\[
\mathrm{mapping\_drift}_l
=
\frac{
\|W_{\mathrm{final},l}U_{S,l}
-
W_{\mathrm{safe},l}U_{S,l}\|_F
}{
\|W_{\mathrm{safe},l}U_{S,l}\|_F+\epsilon
}.
\]

## 15.5 Downstream update norm

\[
\|B_{d,l}A_{d,l}\|_F.
\]

제약 때문에 update가 0으로 붕괴하지 않았는지 확인한다.

## 15.6 Score sanity

각 module에서 다음을 저장한다.

- score min/mean/max
- top-k score mass
- selected/unselected score ratio
- \(c^{\mathrm{grad}}\), \(c^{\mathrm{mag}}\), \(c^{\mathrm{Taylor}}\) 간 Spearman correlation
- activation singular-value rank와 importance rank의 Spearman correlation

마지막 항목은 선택이 단순히 top principal activation directions로 퇴화했는지 진단하기 위함이다.

---

# 16. Synthetic unit tests

작은 synthetic matrix로 다음을 검증한다.

## Test A. Rotation consistency

Orthogonal \(U\)에 대해

\[
\Delta W_s
=
(\Delta W_sU)U^\top.
\]

## Test B. Gradient transform

Autograd 또는 finite difference로

\[
\frac{\partial L}{\partial\widetilde W}
=
\frac{\partial L}{\partial W}U
\]

를 검증한다.

## Test C. Chunked adapter score

Dense 계산과 chunked 계산의 column scores가 일치해야 한다.

\[
c_{\mathrm{dense}}
\approx
c_{\mathrm{chunked}}.
\]

## Test D. Projection

\[
A^\perp
=
A-(AU_S)U_S^\top.
\]

검증:

\[
A^\perp U_S\approx0.
\]

## Test E. Rank preservation

\[
\operatorname{rank}(BA^\perp)\le r_d.
\]

## Test F. Mapping preservation

\[
(W_{\mathrm{safe}}+BA^\perp)U_S
\approx
W_{\mathrm{safe}}U_S.
\]

## Test G. Column-mask equivalence

Full orthogonal \(U\)와 diagonal direction mask \(D_F\)에 대해

\[
BAUD_FU^\top
\approx
BA(I-U_SU_S^\top).
\]

---

# 17. 최소 비교 실험

모든 방법은 동일한 safety LoRA merged checkpoint \(W_{\mathrm{safe}}\)에서 시작해야 한다.

## 17.1 Plain downstream LoRA

\[
W_{\mathrm{safe}}+s_dB_dA_d.
\]

## 17.2 Existing `adapter_subspace_lora`

Safety adapter 자체의 compact-SVD right singular subspace \(Q_S\):

\[
W_{\mathrm{safe}}
+
s_dB_dA_d(I-Q_SQ_S^\top).
\]

## 17.3 Strict WSR-column projected LoRA

Activation basis \(U\), gradient-only column importance:

\[
W_{\mathrm{safe}}
+
s_dB_dA_d(I-U_SU_S^\top).
\]

## 17.4 Proposed adapter-aware WSR-column LoRA

Activation basis \(U\), adapter-Taylor column importance:

\[
W_{\mathrm{safe}}
+
s_dB_dA_d(I-U_SU_S^\top).
\]

수식은 17.3과 같지만 \(U_S\) 선택 기준이 다르다.

## 17.5 Adapter-magnitude-only WSR-column

\[
c_j
=
\|\Delta W_su_j\|_2.
\]

## 17.6 Random-column control

같은 WSR basis \(U_l\)에서 동일한 \(k_l\)개의 column을 random selection한다.

이 control은 improvement가 단순한 capacity reduction 때문인지, safety-aware selection 때문인지 확인한다.

가능하면 seed 3개 이상 사용한다.

---

# 18. 권장 첫 실행 조합

계산비용을 고려한 최소 구성:

```text
A. plain_lora
B. adapter_subspace_lora all_effective
C. wsr_column gradient_only ratio=1%
D. adapter_aware_wsr adapter_taylor ratio=1%
E. adapter_aware_wsr adapter_taylor topk=16
F. random WSR columns, matched k
```

후속:

```text
adapter_taylor ratio = 1%, 5%, 10%
adapter_taylor top-k = 4, 8, 16
gradient_only ratio = 1%, 5%, 10%
```

평가:

- GSM8K 5-shot accuracy
- HarmBench Direct / AutoDAN / PAIR / PAP
- 평균 ASR
- training runtime
- artifact size
- train loss
- mapping constraint
- downstream delta norm

---

# 19. 중요한 해석

이 방법은 원 논문의 WSR-Tune과 완전히 동일하지 않다.

공통점:

1. Safety model의 activation covariance로 basis \(U_l\)를 만든다.
2. Safety loss gradient로 중요도를 측정한다.
3. 중요 WSR directions를 보호한다.
4. complementary directions에서 downstream adaptation을 수행한다.

차이점:

1. 원 논문은 coefficient별 mask \(M_{ij}\)를 사용한다.
2. 이번 방법은 column score로 집계해 direction mask \(m_j\)를 사용한다.
3. 주 방법은 safety adapter magnitude와 gradient를 결합한 Taylor score를 사용한다.
4. Rank 보존을 위해 projection을 LoRA \(A_d\)에 흡수한다.

정확한 명칭:

> **Adapter-aware, column-structured WSR projection for LoRA**

과도하게 주장하면 안 되는 것:

- \(U_S\)가 전체 safety subspace라는 주장
- Taylor score가 causal safety importance를 완전히 측정한다는 주장
- layer-wise mapping preservation이 model-level safety를 보장한다는 주장
- 원 논문의 WSR-Tune과 동일하다는 주장

정확한 주장:

- Safety-conditioned activation basis에서 safety adapter update와 safety-loss sensitivity가 함께 큰 directions를 선택한다.
- Downstream LoRA를 선택된 directions의 직교 여공간으로 제한한다.
- 선택된 layer input directions에 대한 safety-model mapping은 정확히 보존된다.
- 오른쪽 projection이므로 LoRA rank가 유지된다.

---

# 20. 구현 작업 요청

다음 순서로 작업해 달라.

1. Repository 구조와 기존 관련 파일을 읽는다.
2. 현재 `wsr_lora`, Phase 1/2, `adapter_subspace_lora`, training callback의 실제 동작을 요약한다.
3. 재사용 가능한 함수와 새로 필요한 함수를 구분한다.
4. 변경할 파일/함수 목록과 데이터 흐름을 먼저 제안한다.
5. 구현 전 다음 잠재적 충돌을 확인한다.
   - Phase 1 basis가 현재 safety LoRA model과 일치하는가
   - Phase 2가 raw gradient importance를 저장하는가
   - PEFT module naming이 basis artifact naming과 일치하는가
   - LoRA \(A/B\) orientation이 수식과 일치하는가
   - merge/keep 경로가 동일한 \(W_{\mathrm{safe}}\)를 만드는가
   - gradient accumulation hook 시점이 올바른가
   - optimizer step 후 projection이 모든 step에서 호출되는가
6. `adapter_aware_wsr_projected_lora` method를 구현한다.
7. \(n\times n\) projector를 절대 materialize하지 않는다.
8. raw importance 또는 column score를 chunked 방식으로 계산한다.
9. gradient projection과 post-step parameter projection을 구현한다.
10. PEFT adapter와 dense merged model 저장을 모두 지원한다.
11. unit test를 추가한다.
12. 첫 실행 command를 작성한다.
13. 실행 후 다음을 report한다.
    - 160개 module 모두 basis/importance/subspace가 매칭됐는가
    - \(k_l\) 분포
    - score 분포와 correlation
    - constraint errors
    - mapping drift
    - train loss와 runtime
    - output artifact paths
14. 코드 수정 전에 구현 계획을 먼저 보여주고, 수학적으로 불명확하거나 현재 artifact가 부족한 부분을 명시한다.

---

# 21. 한 줄 요약

구현할 최종 방법은 다음이다.

\[
\widetilde{\Delta W}_{s,l}
=
s_sB_{s,l}A_{s,l}U_l,
\]

\[
G_l
=
\sum_{x\in D_{\mathrm{safe}}}
\left|
\frac{\partial\mathcal L_{\mathrm{safe}}(x)}
{\partial(W_{\mathrm{safe},l}U_l)}
\right|,
\]

\[
c_{l,j}
=
\left\|
\left|
\widetilde{\Delta W}_{s,l}[:,j]
\right|
\odot
G_l[:,j]
\right\|_2,
\]

\[
U_{S,l}
=
U_l[:,\operatorname{TopK}(c_l)],
\]

\[
\boxed{
\Delta W_{d,l}^{\perp}
=
s_dB_{d,l}A_{d,l}
\left(
I-U_{S,l}U_{S,l}^\top
\right)
}
\]

이며, 실제 구현은

\[
\boxed{
A_{d,l}
\leftarrow
A_{d,l}
-
(A_{d,l}U_{S,l})U_{S,l}^\top
}
\]

로 수행한다.
