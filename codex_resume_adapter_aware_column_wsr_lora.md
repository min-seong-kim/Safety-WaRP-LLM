# Codex Handoff — Adapter-Aware Column-WSR-LoRA 구현 재개 지시서

## 0. 현재 상황

Claude Code가 이 기능을 구현하던 중 사용량 제한으로 중단되었다.  
따라서 **처음부터 다시 구현하지 말고**, 현재 repository의 변경 상태를 먼저 조사한 뒤 중단 지점부터 안전하게 이어서 완성해야 한다.

이 문서는 기존 설계 문서 `adapter_aware_column_wsr_lora_implementation.md`를 보완한 **재개용 handoff 문서**다.

핵심 목표는 다음과 같다.

> Safety LoRA가 merge된 safety model에서 WSR activation basis \(U_l\)를 새로 만들고,  
> safety adapter update를 그 basis로 회전한 뒤 safety gradient와 결합해 중요 column directions \(U_{S,l}\)를 선택한다.  
> 이후 GSM8K downstream LoRA의 \(A_d\)를 \(U_{S,l}\)의 직교 여공간으로 제한해 LoRA rank를 보존한다.

최종 downstream update:

\[
\Delta W_{d,l}^{\perp}
=
s_dB_{d,l}A_{d,l}
\left(I-U_{S,l}U_{S,l}^{\top}\right).
\]

실제 구현:

\[
A_{d,l}
\leftarrow
A_{d,l}
-
(A_{d,l}U_{S,l})U_{S,l}^{\top}.
\]

---

# 1. 가장 먼저 할 일: 중단된 코드 상태 조사

어떤 파일도 즉시 덮어쓰거나 되돌리지 말 것.

먼저 아래를 수행하고 결과를 요약하라.

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
find . -maxdepth 3 -type f \( \
  -name '*adapter*wsr*' -o \
  -name '*column*wsr*' -o \
  -name '*subspace*' \
\) | sort
```

필수 조사 항목:

1. Claude Code가 새로 만든 파일
2. 기존 파일에서 수정한 부분
3. 미완성 함수, TODO, placeholder, syntax error
4. 생성됐지만 method dispatch에 연결되지 않은 코드
5. 테스트가 만들어졌는지
6. shell driver가 생성됐는지
7. 기존 `adapter_subspace_lora` 또는 `wsr_lora`가 손상되지 않았는지
8. 현재 변경을 보존한 채 이어갈 수 있는지

다음 원칙을 지켜라.

- 기존 partial implementation을 먼저 읽고 재사용한다.
- 이미 올바르게 작성된 코드를 다시 만들지 않는다.
- unrelated change를 되돌리지 않는다.
- 명시적 근거 없이 파일 전체를 교체하지 않는다.
- 구현 전에 현재 상태와 남은 작업 목록을 먼저 보고한다.

---

# 2. 방법 정의

## 2.1 Safety LoRA와 safety model

각 target module \(l\)에 대해

\[
\Delta W_{s,l}
=
s_sB_{s,l}A_{s,l},
\]

\[
W_{\mathrm{safe},l}
=
W_{\mathrm{base},l}
+
\Delta W_{s,l}.
\]

Safety adapter는 고정한다.  
Downstream GSM8K 학습에서는 새로운 \(B_{d,l},A_{d,l}\)만 학습한다.

최종 모델:

\[
W_{\mathrm{final},l}
=
W_{\mathrm{base},l}
+
\Delta W_{s,l}
+
\Delta W_{d,l}^{\perp}.
\]

---

## 2.2 WSR activation basis

현재 사용할 safety model은 **Safety LoRA merged model**이다.

기존 full-parameter safety model에서 만든 Phase 1 basis는 재사용하면 안 된다.

현재 로컬 자산:

```text
outputs/adapter_subspace_lora/safety_merged
outputs/adapter_subspace_lora/safety_adapter
```

반드시

```text
outputs/adapter_subspace_lora/safety_merged
```

를 모델로 사용해 Circuit Breakers safety data activation을 수집하고 새로운 Phase 1 basis를 생성한다.

각 module:

\[
H_lH_l^\top
=
U_l\Lambda_lU_l^\top.
\]

중요:

- \(U_l\)는 LoRA adapter의 SVD basis \(Q_s\)가 아니다.
- \(U_l\)는 현재 LoRA safety model의 activation covariance basis다.
- target modules는 `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`.
- 총 32 layers × 5 modules = 160 modules.
- Phase 1 basis artifact는 현재 safety checkpoint 및 preprocessing metadata와 연결돼야 한다.

---

# 3. Claude Code가 이미 확인한 사실

아래 사실을 다시 뒤집지 말고, 실제 코드와 일치하는지 확인한 뒤 활용한다.

## 3.1 Phase 2 gradient 누적식은 이미 정확함

WSR coordinate를

\[
W_l=C_lU_l^\top
\]

라고 하면

\[
\frac{\partial L}{\partial C_l}
=
\frac{\partial L}{\partial W_l}U_l.
\]

기존 Phase 2는 batch마다 `basis_coeff.grad`의 절댓값을 누적하므로

\[
G_l
=
\sum_b
\left|
G_{W,l}^{(b)}U_l
\right|
\]

를 계산한다.

이는 올바른 식이다.

금지된 형태:

\[
\left(\sum_b|G_{W,l}^{(b)}|\right)U_l.
\]

따라서 Phase 2의 **gradient 계산 루프는 재사용 가능**하다.  
부족한 부분은 raw importance 또는 downstream에 필요한 score를 저장·노출하는 기능이다.

---

## 3.2 기존 Phase 1 basis는 사용 불가

기존 basis는 full-FT safety model에서 만들어졌다.

새 방법은

\[
W_{\mathrm{safe}}
=
W_{\mathrm{base}}+s_sB_sA_s
\]

에서 basis를 만들어야 한다.

따라서 Phase 1 재생성은 선택사항이 아니라 방법론적 요구사항이다.

---

## 3.3 Phase 2는 raw \(G_l\)를 저장하지 않음

현재 Phase 2는 binary mask만 저장하고 raw gradient importance를 버린다.

그러나 adapter-aware Taylor score는

\[
c^{\mathrm{Taylor}}_{l,j}
=
\left\|
\left|
\widetilde{\Delta W}_{s,l}[:,j]
\right|
\odot
G_l[:,j]
\right\|_2
\]

이므로 원소별 \(G_l\)가 필요하다.

Column norm만으로는 계산할 수 없다.

---

## 3.4 A/B orientation은 올바름

\[
A\in\mathbb R^{r\times n},
\qquad
B\in\mathbb R^{m\times r},
\qquad
\Delta W=sBA.
\]

따라서

\[
AU_S\in\mathbb R^{r\times k}.
\]

---

## 3.5 merge/keep 경로는 bitwise 동일하지 않을 수 있음

수학적으로는 같아도 bf16 rounding 때문에 bitwise 동일하지 않을 수 있다.

실험군과 baseline 모두 동일한 merged safety checkpoint에서 시작하도록 통일한다.

권장 starting checkpoint:

```text
outputs/adapter_subspace_lora/safety_merged
```

---

## 3.6 기존 projection infrastructure 재사용 가능

기존 `adapter_subspace_lora`의 구조에서 다음이 검증됐다.

- `register_post_accumulate_grad_hook` 사용 가능
- projection은 선형이므로 gradient accumulation과 교환 가능
- post-step projection이 모든 optimizer step에서 호출됨
- 기존 실측:
  - 1404 optimizer steps
  - 총 projection calls 1406
  - 시작/종료 추가 호출 포함

---

# 4. 중요도 계산

## 4.1 Safety adapter를 WSR basis로 회전

\[
\widetilde{\Delta W}_{s,l}
=
\Delta W_{s,l}U_l
=
s_sB_{s,l}(A_{s,l}U_l).
\]

전체 dense tensor를 항상 만들 필요는 없다.

Column chunk \(J\)에 대해

\[
\widetilde{\Delta W}_{s,l}[:,J]
=
s_sB_{s,l}
\left(A_{s,l}U_l[:,J]\right).
\]

---

## 4.2 세 가지 score를 한 번의 Phase 2 sweep에서 계산

### A. Gradient-only

\[
c^{\mathrm{grad}}_{l,j}
=
\|G_l[:,j]\|_2.
\]

### B. Adapter magnitude-only

\[
c^{\mathrm{mag}}_{l,j}
=
\left\|
\widetilde{\Delta W}_{s,l}[:,j]
\right\|_2.
\]

### C. Adapter-aware Taylor — 주 방법

\[
c^{\mathrm{Taylor}}_{l,j}
=
\left\|
\left|
\widetilde{\Delta W}_{s,l}[:,j]
\right|
\odot
G_l[:,j]
\right\|_2.
\]

---

# 5. Raw \(G_l\) 저장 방식에 대한 결정

전체 160개 module의 raw importance는 매우 크다.

대략:

- bf16 약 9GB
- fp32 약 18GB

따라서 기본 동작은 다음으로 한다.

1. Phase 2 safety-data sweep을 한 번 수행
2. \(G_l\)가 메모리에 누적된 직후
3. 같은 프로세스에서 \(\widetilde{\Delta W}_{s,l}\)와 결합
4. `gradient_only`, `adapter_magnitude_only`, `adapter_taylor` column score를 모두 계산
5. column score, selected indices, \(U_S\), metadata만 저장
6. raw \(G_l\)는 기본적으로 디스크에 저장하지 않음
7. 필요 시에만 다음 플래그로 저장

```bash
--save_raw_importance
```

주의:

- `adapter_taylor`는 column norm만으로 사후 계산할 수 없다.
- 원소별 곱이 필요하므로 raw \(G_l\)가 메모리에 존재하는 시점에 계산해야 한다.
- backward/data sweep을 세 ablation마다 반복하지 말고 한 번만 수행한다.
- memory peak를 기록한다.
- GPU memory가 부족하면 module별 CPU offload 또는 sequential processing을 구현하되 수학식을 변경하지 않는다.

---

# 6. Column 선택과 \(k\)

\(k_l\)는 LoRA rank가 아니다.

\[
U_{S,l}\in\mathbb R^{n_l\times k_l}
\]

에서 \(k_l\)는 보호할 WSR input direction 수다.

LoRA rank는 별도로

\[
r_d=16
\]

이다.

## 6.1 Ratio mode

\[
k_l=\lceil\rho n_l\rceil.
\]

예:

- \(n=4096,\rho=1\%\Rightarrow k\approx41\)
- \(n=11008,\rho=1\%\Rightarrow k\approx111\)

## 6.2 Top-k mode

예:

```text
top_k = 16
```

이는 보호 방향 수가 16이라는 뜻이며 LoRA rank와 우연히 같은 숫자일 뿐이다.

## 6.3 Budget 비교 주의

`ratio=1%`는 기존 `adapter_subspace_lora all_effective`의 \(k=16\)보다 강한 제약이다.

따라서 반드시 `adapter_taylor top_k=16`을 함께 지원한다.

핵심 비교:

- 기존 adapter-subspace all-effective \(k=16\)
- adapter-aware WSR top-\(k=16\)

이 비교는 보호 방향 수를 맞춘다.

---

# 7. Protected direction과 projection

Score가 큰 direction index를 \(S_l\)라 하고

\[
U_{S,l}=U_l[:,S_l].
\]

Downstream LoRA는 다음 제약을 만족해야 한다.

\[
A_{d,l}U_{S,l}=0.
\]

큰 projector

\[
I-U_SU_S^\top
\]

를 materialize하지 않는다.

항상 다음 형태로 계산한다.

\[
A_{d,l}
\leftarrow
A_{d,l}
-
(A_{d,l}U_{S,l})U_{S,l}^\top.
\]

다음 세 시점에 적용한다.

1. downstream LoRA 초기화 직후
2. gradient accumulation 후 gradient projection
3. 모든 optimizer step 이후 parameter projection

최종 보존 조건:

\[
\Delta W_{d,l}^{\perp}U_{S,l}=0,
\]

\[
W_{\mathrm{final},l}U_{S,l}
=
W_{\mathrm{safe},l}U_{S,l}.
\]

---

# 8. 기존 `adapter_subspace_lora`와의 차이

기존 방법:

\[
\Delta W_s=P_s\Sigma_sQ_s^\top
\]

에서 right singular subspace \(Q_s\)를 직접 보호한다.

새 방법:

1. safety model activation으로 \(U_l\) 구성
2. safety adapter를 \(U_l\) basis로 회전
3. safety gradient와 adapter magnitude로 중요도 측정
4. \(U_l\)의 일부 columns를 보호

`adapter_magnitude_only`가 기존 방법과 강하게 상관될 수는 있지만 동일하지 않다.

- 기존 방법: 연속 subspace \(\operatorname{span}(Q_s)\)
- 새 magnitude-only: discrete WSR basis columns \(u_j\) 중 \(\|\Delta W_su_j\|\)가 큰 것

Spearman correlation과 subspace overlap을 진단으로 저장하라.

---

# 9. 재사용할 코드와 새 코드

Claude Code가 제안한 구조:

## 재사용

- `Phase1BasisBuilder`
- Phase 2 gradient accumulation loop
- `module_name_to_key`
- 기존 projection callback
- `AdapterSubspaceProjector`를 일반화한 구조
- `finetune_gsm8k_lora.py`의 저장/merge/upload 경로

## 신규 후보

```text
models/adapter_wsr_column.py
build_adapter_wsr_subspace.py
scripts/run_adapter_aware_wsr_lora.sh
tests/test_adapter_wsr_column.py
```

## 수정 후보

```text
finetune_gsm8k_lora.py
phase2_importance_per_layer.py
models/adapter_subspace.py
```

다만 partial implementation에 이미 이 파일들이 존재하면 새로 만들지 말고 검토·완성한다.

---

# 10. 구현 acceptance criteria

## 10.1 Phase 1

- LoRA safety merged model에서 basis 생성
- 160개 target module 전부 존재
- basis shape가 module input dimension과 일치
- basis orthogonality 기록
- metadata에 safety checkpoint와 dataset 설정 저장

## 10.2 Importance

- 한 번의 Phase 2 sweep에서 세 score 생성
- raw \(G_l\)는 기본 저장하지 않음
- chunked와 dense small-test 결과 일치
- 160개 module score와 \(U_S\) 생성
- selected \(k_l\) 분포 저장

## 10.3 Projection

각 module에서 다음을 계산한다.

\[
\mathrm{constraint}_A
=
\frac{\|A_dU_S\|_F}{\|A_d\|_F+\epsilon},
\]

\[
\mathrm{constraint}_\Delta
=
\frac{\|B_dA_dU_S\|_F}{\|B_dA_d\|_F+\epsilon},
\]

\[
\mathrm{mapping\_drift}
=
\frac{
\|W_{\mathrm{final}}U_S-W_{\mathrm{safe}}U_S\|_F
}{
\|W_{\mathrm{safe}}U_S\|_F+\epsilon
}.
\]

fp32 LoRA parameters 기준으로 충분히 작아야 한다.

## 10.4 Rank

\[
\operatorname{rank}(B_dA_d^\perp)\le r_d.
\]

PEFT adapter 형태로 저장·merge 가능해야 한다.

## 10.5 Regression

기존 method가 깨지면 안 된다.

최소 smoke test:

```text
lora
wsr_lora
adapter_subspace_lora
adapter_aware_wsr_projected_lora
```

---

# 11. 필수 unit tests

1. Rotation consistency
   \[
   \Delta W_s=(\Delta W_sU)U^\top
   \]

2. Gradient transform
   \[
   \frac{\partial L}{\partial C}
   =
   \frac{\partial L}{\partial W}U
   \]

3. Chunked score equals dense score

4. Projection
   \[
   A^\perp U_S\approx0
   \]

5. Rank preservation
   \[
   \operatorname{rank}(BA^\perp)\le r
   \]

6. Mapping preservation
   \[
   (W_{\mathrm{safe}}+BA^\perp)U_S
   \approx
   W_{\mathrm{safe}}U_S
   \]

7. Column-mask equivalence
   \[
   BAUD_FU^\top
   \approx
   BA(I-U_SU_S^\top)
   \]

8. Invalid artifact shape/module mismatch raises clear error

9. Resume/skip logic does not treat partial artifacts as complete

---

# 12. 실험군

이미 완료된 기존 결과:

```text
A. plain LoRA
B. adapter_subspace_lora
```

새로 구현할 실험군:

```text
C. gradient_only, ratio=1%
D. adapter_taylor, ratio=1%
E. adapter_taylor, top_k=16
F. random WSR-column control, matched k
```

비교 의미:

- C vs D:
  safety adapter magnitude를 importance에 추가한 효과
- D vs F:
  safety-aware selection vs 단순 capacity reduction
- D vs E:
  ratio 1%와 top-16의 제약 강도 차이
- B vs E:
  동일한 \(k=16\)에서 adapter SVD subspace vs adapter-aware WSR directions

Random control은 각 layer에서 동일한 \(k_l\)을 사용한다.  
가능하면 여러 seed를 지원하지만, 구현 완료 후 실제 expensive run 범위는 사용자 확인 없이 확대하지 않는다.

---

# 13. 실행 순서

아래 순서를 따른다.

## 단계 1: 코드 복구 및 정적 검증

- 현재 partial diff 조사
- syntax/import 검사
- unit test
- method dispatch 연결
- `--help` 확인

## 단계 2: 작은 synthetic/smoke run

- 1개 layer 또는 매우 적은 safety samples
- Phase 1 basis 생성
- importance 생성
- \(U_S\) artifact 생성
- downstream projection 1–2 steps
- constraint 검증

## 단계 3: 전체 Phase 1 및 importance artifact 생성

- LoRA safety merged model 사용
- 160개 module
- 한 번의 Phase 2 sweep
- C/D/E/F용 subspace artifact 생성 가능하게 구성

## 단계 4: expensive GSM8K training

사용자 확인 전 자동으로 전체 C–F를 모두 돌리지 말 것.

구현·artifact 검증이 끝난 뒤 다음을 보고한다.

- 예상 runtime
- 예상 GPU memory
- 생성될 run 수
- C–F 중 어떤 run이 준비됐는지
- 이미 완료된 A/B를 재실행할 필요가 없는지

---

# 14. 보고 형식

작업 완료 후 다음을 명확히 보고하라.

1. 기존 partial code에서 발견한 것
2. 보존한 코드와 수정한 코드
3. 새로 만든 파일
4. Phase 1 basis 재생성 경로
5. importance 계산 방식
6. raw importance 저장 여부
7. \(k_l\) 분포
8. projection hook 호출 위치
9. unit-test 결과
10. smoke-test 결과
11. 남은 expensive run
12. 정확한 실행 command
13. output artifact 경로
14. 알려진 한계 또는 미검증 사항

---

# 15. Codex에 처음 전달할 짧은 요청문

이 문서와 기존 설계 문서를 읽고 현재 repository의 partial implementation을 이어서 완성하라.

중요:

- 처음부터 재작성하지 말 것
- 먼저 `git status`, `git diff`, 관련 파일을 읽을 것
- Claude Code가 중단되기 전 만든 코드를 최대한 보존할 것
- 기존 full-FT Phase 1 basis를 재사용하지 말 것
- LoRA safety merged model에서 Phase 1 basis를 재생성할 것
- Phase 2의 올바른 gradient accumulation loop를 재사용할 것
- 한 번의 safety-data sweep에서 gradient-only와 adapter-Taylor score를 모두 만들 것
- raw \(G_l\) 9–18GB를 기본적으로 디스크에 저장하지 말 것
- \(n\times n\) projector를 만들지 말 것
- 구현 및 smoke test 후 expensive training 전에 결과를 먼저 보고할 것
