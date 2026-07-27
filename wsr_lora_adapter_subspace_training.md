# WSR-LoRA / `adapter_subspace_lora` — 전체 학습 과정 기록

> 실행 완료 2026-07-25 (Vast 인스턴스, RTX PRO 6000 Blackwell 97GB).
> 이 문서는 **실제로 수행한 학습 과정 전체**를 단계별로 기록한다. 수치는 전부 이 실행에서
> 나온 실측값이다.
>
> 관련 문서: `WaRP.md`(이론) · `CLAUDE.md`(저장소 구조) · `wsr_lora_status.md`(LoRA 라인 전체 현황) ·
> `wsr_lora_comparison.md`(구현 전 지시서).

---

## 0. 이 실험이 답하는 질문

WSR-Tune(논문 본문)은 **full-param** 기법이다. 리뷰어가 논문 결론의
*"Future work may further combine this perspective with low-rank adaptation"* 을 짚어
"LoRA와 결합하면?"이라고 물었고, 그에 답하기 위한 실험군이 WSR-LoRA 라인이다.

라인 전체는 **동일한 LoRA 예산(r=16, α=32, q/k/v/up/down)** 을 고정하고
"증분이 허용되는 좌표계와 제약 방식"만 바꿔 비교한다. 이 문서가 다루는
`adapter_subspace_lora`는 그중 하나다.

> ⚠️ **이름 주의.** `adapter_subspace_lora` ≠ `wsr_lora`.
> `wsr_lora`는 회전 공간에서 **원소별 마스크**를 쓰는 논문 충실 버전이고,
> `adapter_subspace_lora`는 **방향(열) 단위 투영**을 쓰는 별개 방법이다. 아래 §1.2 참조.

---

## 1. 방법

### 1.1 수식

`W_base` = 안전 정렬 이전의 base weight, `A ∈ ℝ^{r×n}`, `B ∈ ℝ^{m×r}`, `s = α/r`.

```
ΔW_s   = s_s·B_s A_s = P_s Σ_s Q_sᵀ        (safety adapter 의 compact SVD)
Q_S    = Q_s[:, :k]                         (보호할 right singular vectors)
ΔW_d^⊥ = s_d·B_d A_d (I − Q_S Q_Sᵀ)         (A_d Q_S = 0 을 유지)

W_final = W_base + s_s·B_s A_s + ΔW_d^⊥
⇒ W_final Q_S = W_safe Q_S                  (정확히 보존)
```

의미: **입력이 `Q_S` 방향 성분을 가지면, 그 성분에 대한 레이어 출력은 safety 모델과 정확히 같다.**
downstream adapter 는 `Q_S` 에 직교하는 나머지 `(n−k)` 차원에서만 읽는다.

- **right** singular vector 를 쓰는 이유: 투영을 오른쪽(입력 쪽)에 걸기 때문. `P_s`(출력 쪽)는 미사용.
- safety adapter `B_s, A_s`는 **고정**. 수정·pruning 하지 않는다. 새로 학습하는 건 `B_d, A_d` 뿐.
- 제약이 `A_d` 하나에만 걸린다: `s·B_d[A_d(I−Q_SQ_Sᵀ)] = [s·B_d A_d](I−Q_SQ_Sᵀ)`. `B_d`는 자유.

### 1.2 왜 rank 가 보존되는가 (`wsr_lora` 와의 결정적 차이)

| | `wsr_lora` | `adapter_subspace_lora` |
|---|---|---|
| ΔW | `[(1−M)∘(s·BA)]Uᵀ` | `s·BA(I−Q_SQ_Sᵀ)` |
| 제약 단위 | 원소 (Hadamard) | 방향 (오른쪽 투영) |
| rank | **깨짐** (일반적으로 full rank) | **보존** (≤ r) |
| 저장 | dense fold 강제 (`restore_wsr_lora_to_linear`) | merge 가능한 진짜 adapter |
| 필요 artifact | Phase 1 basis + Phase 2 mask | safety LoRA adapter 하나 |

아다마르 곱은 rank 를 보존하지 않으므로 `(1−M)∘(BA)` 는 일반적으로 full rank 가 되고,
`merge_and_unload()` 가 불가능해진다. 오른쪽 곱셈은 `rank(XP) ≤ rank(X) = r` 이므로 보존된다.

### 1.3 dense ΔW_s 를 만들지 않는 exact 분해

`models/adapter_subspace.py:67` `compact_svd_from_lora`:

```
B_s = Q_B R_B,   A_sᵀ = Q_A R_A            (thin QR ×2)
K   = s_s·R_B R_Aᵀ ∈ ℝ^{16×16}             (여기에만 SVD:  K = U_K Σ V_Kᵀ)
P_s = Q_B U_K,   Q_s = Q_A V_K
```

4096×4096 dense SVD 대신 **16×16 SVD 한 번**. 160개 모듈 전체가 수 초에 끝난다.

---

## 2. 왜 Phase 1/2 를 타지 않는가

`adapter_subspace_lora` 는 저장소의 WSR-Tune 4단계 파이프라인(Phase 0~3)을 **타지 않는다.**
활성화 공분산 basis(Phase 1)도, gradient importance mask(Phase 2)도 쓰지 않는다.
대신 **이미 학습된 safety LoRA adapter 자신의 업데이트 기하**를 쓴다. 이유:

**① full-FT delta 에는 보호할 "방향"이 정의되지 않는다.**
기존 safety 모델(`kmseong/llama2_7b-chat-Safety-FT-lr5e-5`)은 full-param FT 라
`ΔW_s = W_safe − W_base` 가 dense full-rank(4096) 다. 스펙트럼이 평평해서 "상위 k개가 90%"
같은 구조가 없고, k 를 어디서 끊어도 임의적이다. 반면 rank-16 LoRA adapter 는 델타가 이미
인수분해되어 있어 **보호 방향이 태생적으로 최대 16개**로 제한된다 →
4096차원 중 16개(0.4%)만 막으므로 downstream 은 99.6% 자유도를 유지한다.

**② 이식성.** `wsr_lora` 는 basis(GB 단위)+mask(841MB×4)가 필요하고 재생성에 GPU·수 시간이 든다.
실제로 이 저장소에서 **둘 다 소실됐다**(§6.3). `adapter_subspace_lora` 는 HF 의 adapter 112MB
하나만 있으면 `Q_S` 가 base 모델을 로드조차 않고 수 초에 재생성된다.

**③ 예산 정합.** 다른 method 가 전부 LoRA 예산인데 safety 만 full-param 이면 비대칭이 생긴다.

**④ 출발점 통일.** Stage 0.5 의 병합 결과 `W_safe` 가 실험군(Stage 2)과 통제군(Stage 3)의
공통 출발점이 된다.

---

## 3. 파이프라인 — 실행한 5단계

드라이버: `scripts/run_adapter_subspace_lora.sh` (완료된 stage 는 건너뛴다).

### Stage 0 — safety LoRA 학습

```
base      : meta-llama/Llama-2-7b-chat-hf
data      : data/circuit_breakers_train.json  (4994 샘플)
config    : r=16, α=32, lr=1e-4, 3 epoch, batch 4 × grad_accum 4 (eff. batch 16)
targets   : q_proj, k_proj, v_proj, up_proj, down_proj
→ outputs/adapter_subspace_lora/safety_adapter   (112MB, B_s / A_s)
```

loss: **1.9329 → 0.4178 → 0.2350 → 0.2250**

> Stage 0 의 epochs/batch/samples 는 `models/phase0_SSFT.py` 의 **모듈 상수**라 CLI 로 못 바꾼다.
> `SafetyDataset` 이 모든 샘플을 `max_length=1024` 로 패딩하므로(동적 패딩·packing 없음)
> A6000 기준 3.0 s/it, 에폭당 62분이 걸린다. 속도 개선이 필요하면 여기가 지점이다.

### Stage 0.5 — dense 병합

`W_safe = W_base + s_s·B_s A_s` 를 bf16 dense 로 병합·저장 (13GB).
통제군(Stage 3)의 출발점이자 "downstream 이전" safety 평가 기준점.

### Stage 1 — 보호 subspace `Q_S` 추출

`build_adapter_subspace.py`. base 모델을 **로드하지 않는다**. adapter 만 읽고 QR+SVD.
GPU 불필요, 160개 모듈 × 6개 선택 모드가 **수 초**에 끝난다.

`--svd_dtype float64` 가 기본값인 이유: `n = 4096, 11008` 에서 fp32 QR 의 직교성 오차가
`√n·eps ≈ 6e-6` 까지 커진다. 이 크기 행렬에서 fp64 비용은 무시할 만하다.

### Stage 2 — downstream 학습 (실험군)

`finetune_gsm8k_lora.py --method adapter_subspace_lora`.
`W_safe` 에서 출발해 GSM8K 로 새 LoRA `B_d, A_d` 를 학습하되 `A_d Q_S = 0` 을 강제한다.

**제약 유지에 두 장치가 모두 필요하다** (`models/adapter_subspace.py:343,358`):

1. **gradient projection** — `register_post_accumulate_grad_hook` 으로 `g ← g − (gQ_S)Q_Sᵀ`.
   투영이 선형이라 gradient accumulation 과 교환된다(부분합을 각각 투영 == 총합을 투영).
2. **`optimizer.step()` 후 파라미터 재투영** — **생략 불가.**
   grad 를 완벽히 투영해도 AdamW 의 step 은 `exp_avg / √exp_avg_sq` 로 **원소별** 나눗셈을
   거친다. `exp_avg` 는 grad 의 선형결합이라 자동으로 부분공간 안에 남지만, `exp_avg_sq` 는
   원소별 제곱이라 투영이라는 개념 자체가 없다. 원소별 나눗셈이 선형성을 깨므로 update 가
   부분공간 밖으로 샌다. **제약을 실제로 보장하는 건 이 재투영이다.**
   (decoupled weight decay 는 `A ← (1−λη)A` 라 부분공간을 보존하므로 무해.)

### Stage 3 — 통제군

**같은 `W_safe` 에서 출발한 제약 없는 plain LoRA.**
기존 `run_lora_comparison.sh` 의 baseline 은 full-FT safety 모델에서 출발하므로 출발점이 달라
비교가 오염된다. 그래서 별도로 돌린다.

---

## 4. 실행 환경

```
GPU        : NVIDIA RTX PRO 6000 Blackwell Server Edition 97GB (단일, SLURM 없음)
             compute capability (12,0) → CUDA ≥ 12.8 휠 필수
python     : /venv/hb/bin/python  (PATH 의 `python` 도 여기)
             torch 2.10.0+cu128 | transformers 4.57.3 | peft 0.18.1
             accelerate 1.12.0 | datasets 4.4.1
HF         : HF_HOME=/workspace/.hf_home, token=kmseong (fine-grained, repo.write)
```

### 4.1 이전 환경(SLURM 클러스터)에서 옮기며 고친 것

| 항목 | 이전 | 수정 |
|---|---|---|
| 인터프리터 | `HBPY=/home/gokms0509/anaconda3/envs/hb/bin/python` | `HBPY=${HBPY:-/venv/hb/bin/python}` (env 로 덮어쓰기 가능) |
| 스케줄러 | `sbatch scripts/sbatch_adapter_subspace_lora.sh` | SLURM 없음 → `bash` 직접 실행 |
| GPU 지정 | "스케줄러가 `CUDA_VISIBLE_DEVICES` 설정" | 단일 GPU → unset 이면 torch 가 GPU 0 사용. 하드코딩 금지는 그대로 유효 |
| 학습 조건 | `TRAIN_SELECTIONS=(all_effective topk8 topk4)` | `(all_effective topk4 energy90)` — 근거는 §5.2 |

`scripts/sbatch_adapter_subspace_lora.sh` 는 이 박스에서 **사용 불가**(sbatch 부재)이나,
SLURM 환경으로 돌아갈 때를 위해 삭제하지 않고 남겼다.

---

## 5. Stage 1 결과 — safety 업데이트의 기하

### 5.1 스펙트럼: 소수 방향에 극도로 집중

```
누적 에너지 (σ² 기준, k=1..8)
  L 0 attn_q    0.579 0.748 0.836 0.873 0.904 0.928 0.944 0.957
  L 0 ffn_down  0.670 0.809 0.863 0.908 0.928 0.943 0.954 0.964
  L15 attn_q    0.834 0.883 0.915 0.931 0.945 0.955 0.962 0.968
  L15 ffn_down  0.736 0.855 0.919 0.936 0.948 0.958 0.966 0.972
  L31 attn_q    0.925 0.951 0.965 0.973 0.978 0.983 0.986 0.989
  L31 ffn_down  0.677 0.913 0.933 0.947 0.959 0.967 0.973 0.979
```

σ₁ 하나가 평균 **67~78%**. L31 attn_q 는 첫 방향만으로 92.5%.
**"safety LoRA 업데이트는 저차원"이라는 가정이 실측으로 확인된다.**

### 5.2 선택 모드 6종 (160개 모듈 전부)

| 선택 | k (min/mean/max) | 보호 에너지 | 학습 여부 |
|---|---|---|---|
| `topk2` | 2/2.0/2 | 81.69% | — |
| `topk4` | 4/4.0/4 | 89.19% | **✓** |
| `energy90` | **1/4.5/12** | 91.48% | **✓** |
| `topk8` | 8/8.0/8 | 95.10% | — |
| `energy99` | 2/13.0/16 | 99.23% | — |
| `all_effective` | 16/16.0/16 | 100% | **✓** |

**학습 3점을 이렇게 고른 이유:**

k 를 키우는 비용이 거의 없다(k=16 이어도 4096 중 0.4%). 따라서 여러 k 를 돌리는 목적은
capacity 절약이 아니라 **과학적 질문**이고, 세 질문에 하나씩 배정했다.

- **`all_effective`** — 헤드라인. k=16 = `A_s` 의 **행공간 전체**에 직교.
  "downstream 이 읽는 입력 방향이 safety 가 읽는 방향과 완전히 분리된다"는 가장 강한 진술.
- **`topk4`** — 고정·균일 예산 대조군.
- **`energy90`** — `topk4` 와 **예산이 맞춰진** 적응적 대조군. 평균 k 가 4.5 로 거의 같지만
  층별로 1~12 로 달라진다.

`energy90` 의 층 구조 (실측):
```
깊이별 k:  early(0-10) 5.7 | mid(11-21) 4.3 | late(22-31) 3.5
모듈별 k:  attn_k 5.8 | attn_q 5.1 | ffn_down 4.4 | ffn_up 4.3 | attn_v 3.0
```
즉 **고정 k=4 는 `attn_v` 를 과보호하고 `attn_k` 를 과소보호한다.** `energy90` 은 같은 총예산을
필요한 곳에 재배분해 에너지를 2.3%p 더 봉쇄한다 → **"예산의 크기가 아니라 배분 방식이
중요한가"** 를 교란변수 없이 묻는다.

**제외한 것:** `topk8`(95.1%)·`energy99`(99.2%) 는 `all_effective`(100%) 와 차이가 작아
정보량이 적다. 하한 탐침이 필요하면 `topk2`(81.7%) 를 추가하면 된다.

### 5.3 분해 정확도

| 지표 | 값 |
|---|---|
| `r_effective` | **160개 모듈 전부 16** (rank collapse 없음) |
| `orthogonality_error` = ‖QᵀQ−I‖_F | max **1.97e-14** |
| `reconstruction_error` (상대) | max **1.41e-07**, median 9.19e-08 |
| dense 교차검증 (L0 attn_k) | lowrank 6.889e-08 vs dense 6.902e-08 |

**재구성 오차 1.4e-07 은 실제 오차가 아니라 지표 자체의 수치 바닥이다.**
`‖X−Y‖² = ‖X‖²+‖Y‖²−2⟨X,Y⟩` 꼴이라 세 항이 거의 상쇄되고, fp64 에서도 상대오차 바닥이
`√eps ≈ 1.5e-8` 수준이며 누적 상수를 곱하면 1e-7 이 된다.
독립적으로 계산한 dense 교차검증이 3자리까지 일치하는 것이 그 증거다.
**상쇄가 없는 지표인 직교성 오차(1e-14)가 분해 품질의 정직한 지표**이고, fp64 수준으로 깨끗하다.

> 참고: `wsr_lora_status.md` 는 이 값을 `0.00e+00` 으로 기록했는데, 같은 현상이다.
> `relative_reconstruction_error` 의 `max(diff2, 0.0)` 클램프가 A6000 에서는 발동하고
> Blackwell 에서는 발동하지 않았을 뿐이다(BLAS 연산 순서 차이). 회귀가 아니다.

---

## 6. Stage 2/3 결과 — 학습과 제약 검증

### 6.1 학습 설정 (4개 run 공통)

```
downstream : GSM8K (openai/gsm8k, main, train 7473)
LoRA       : r=16, α=32, dropout 0.05, targets q,k,v,up,down
optim      : lr 1e-4, 3 epoch, batch 2 × grad_accum 8 (eff. batch 16), max_len 1024, seed 42
dtype      : 모델 bf16 / LoRA 파라미터 fp32
steps      : 1404 (= 7473/16 × 3)
trainable  : 28,049,408 / 6,766,465,024 (0.4145%)
```

`trainable` 검산: q,k,v 각 `16×4096 + 4096×16` = 131,072 → ×3 = 393,216.
up `16×4096 + 11008×16` = 241,664. down `16×11008 + 4096×16` = 241,664.
층당 876,544 × 32층 = **28,049,408** ✓

**LoRA 파라미터를 fp32 로 캐스팅하는 이유**: 제약 달성 정밀도를 좌우한다.
fp32 면 `constraint_A ≈ 1e-7`, bf16 이면 `≈ 1e-3` 에서 멈춘다.
다른 baseline 과 dtype 을 맞추려면 `--lora_param_dtype bfloat16` 을 쓰되, 그 경우
`1e-3` 이 정상값이다. `summary.json` 에 항상 기록되므로 숨은 변수가 되지 않는다.

**배치를 키우지 않은 이유**: 97GB 중 22GB 만 썼지만, 이미 학습된 다른 5개 method 가
eff. batch 16 이라 바꾸면 비교가 깨진다.

### 6.2 학습 결과

| run | k | 보호 에너지 | 첫 loss | 최종 loss | `train_loss` | runtime |
|---|---|---|---|---|---|---|
| `all_effective` | 16 | 100% | 1.2489 | 0.2914 | **0.39168** | 1401.8s |
| `topk4` | 4 | 89.2% | 1.2492 | 0.2921 | **0.39177** | 1463.4s |
| `energy90` | ~4.5 | 91.5% | 1.2489 | 0.2923 | **0.39188** | 1397.5s |
| baseline (제약 없음) | — | — | 1.1654 | 0.2934 | **0.39123** | 1210.0s |

**네 조건의 `train_loss` 가 소수점 셋째 자리까지 같다** (0.3912~0.3919).
제약을 건 세 조건이 제약 없는 통제군과 downstream 학습에서 구분되지 않는다.

통제군이 ~15% 빠른 것은 160개 모듈의 gradient hook + step 후 재투영이 없기 때문이다.

> **미해명 관찰**: 통제군의 *첫 로깅* loss(1.1654)만 나머지(1.2489)보다 낮다. `B=0` 초기화라
> 시작 시 ΔW=0 이므로 이론상 같아야 한다. 수렴값은 동일하므로 결론에 영향은 없으나,
> 원인은 확인하지 않았다(병합 경로 차이 — 실험군은 런타임 GPU 병합, 통제군은 이전 박스에서
> CPU 병합해 저장한 체크포인트 로드 — 로 추정되나 검증하지 않음).

sanity generation 4개 모두 GSM8K 형식을 학습했다:
```
all_effective : "She sold 48/2=<<48/2=24>>24 in May."
topk4         : "April: <<48=48>>48"
energy90      : "In April, Natalia sold 48/2=<<48/2=24>>24 clips."
baseline      : "In April, Natalia sold 48/2=<<48/2=24>>24 clips."
```
(64토큰 단발 샘플이라 품질 신호로 보기엔 부족하다. 정식 평가는 §8.)

### 6.3 제약 검증 (160개 모듈 전부, run 종료 시 측정)

측정 지표 (`models/adapter_subspace.py:370` `verify`):

```
A. constraint_A     = ‖A_d Q_S‖_F / ‖A_d‖_F
B. constraint_delta = ‖B_d A_d Q_S‖_F / ‖B_d A_d‖_F
C. mapping_drift    = ‖W_final Q_S − W_safe Q_S‖_F / ‖W_safe Q_S‖_F
E. delta_norm       = ‖B_d A_d‖_F        (제약이 너무 세서 업데이트가 죽었는지)
```

| run | `constraint_A` max | `constraint_delta` max | `mapping_drift` max | `delta_norm` mean | 재투영 호출 |
|---|---|---|---|---|---|
| `all_effective` | 6.03e-09 | 1.25e-08 | 3.19e-09 | 0.730 | 1406 |
| `topk4` | 5.46e-09 | 1.42e-08 | 6.12e-09 | 0.735 | 1406 |
| `energy90` | 5.56e-09 | 1.60e-08 | 1.09e-08 | 0.734 | 1406 |

- **제약이 목표(~1e-7)를 20배 상회해 달성**됐다. `constrained modules: 160, unconstrained: 0` —
  제약 없이 새는 레이어가 없다.
- `mapping_drift ~1e-9` → **`W_final Q_S = W_safe Q_S` 가 실측으로 성립**한다.
  방법의 핵심 보증이 수치로 확인된 것.
- `projection_calls = 1406` = 1404 step + `on_train_begin` + 종료 시 1회.
  **모든 step 에서 재투영이 발동**했다.
- `delta_norm` (all_effective 기준 min 0.319 / mean 0.730 / max 1.925) —
  **업데이트가 죽지 않았다.** 제약이 과하면 downstream 업데이트가 0 으로 눌려
  "safety 는 지켜졌지만 아무것도 안 배운" 무의미한 결과가 나올 수 있는데, 최대 제약(k=16)
  에서도 충분하다.

### 6.4 해석 (주의해서 읽을 것)

`train_loss` 와 `delta_norm` 이 네 조건에서 사실상 동일하다는 것은
**"k 를 어떻게 배분하든 downstream 학습량 자체는 차이가 없었다"** 는 뜻이다.
따라서 평가에서 나올 GSM8K/ASR 차이는 학습량 차이가 아니라 순수하게
**"어느 방향을 막았는가"** 에서 오는 것으로 해석할 수 있다 — 교란변수 하나가 제거된 셈이다.

⚠️ **다만 loss 가 같다고 GSM8K 정확도가 같다는 보장은 아니다.**
"제약이 공짜"라는 결론은 §8 의 평가 수치가 나온 뒤에만 내릴 수 있다.

---

## 7. 산출물 (HF `kmseong/`, 전부 API 로 파일 목록 검증 완료)

```
llama2_7b-chat-Safety-LoRA-r16-lr1e-4-adapter                    112MB  Stage 0  (B_s/A_s)
llama2_7b-chat-Safety-LoRA-r16-lr1e-4                            13.5GB Stage 0.5 (= W_safe)
llama2_7b-chat-gsm8k-adaptersubspace-all_effective-r16-lr1e-4    13.5GB Stage 2
llama2_7b-chat-gsm8k-adaptersubspace-topk4-r16-lr1e-4            13.5GB Stage 2
llama2_7b-chat-gsm8k-adaptersubspace-energy90-r16-lr1e-4         13.5GB Stage 2
llama2_7b-chat-gsm8k-lora-safetylora-r16-lr1e-4                  13.5GB Stage 3 (통제군)
```

로컬 산출물:
```
outputs/adapter_subspace_lora/
  safety_adapter/ safety_merged/
  subspaces/{all_effective,topk2,topk4,topk8,energy90,energy99}/  report.json + layer_NN_subspace.pt
  runs/{all_effective,topk4,energy90,baseline_lora}/lr_1e-4/
    summary.json  subspace_verification.json  merged_model/
```

> **rank 가 보존되므로 진짜 PEFT adapter 로도 저장 가능**하지만, 평가 하네스 호환을 위해
> 모든 method 와 동일하게 **dense merged 모델**로 저장·업로드했다.

---

## 8. 평가 (이 저장소 밖)

학습 스크립트는 merged 모델 생성·업로드까지만 한다. 채점은 별도 하네스에서:

- **GSM8K (utility)**: `lm-evaluation-harness`, 5-shot + chat template.
- **ASR (safety)**: HarmBench (Direct/AutoDAN/PAIR/PAP). 모델당 수십 분~시간 단위로 비싸다.

두 축을 함께 봐야 의미가 있다 — *downstream 을 배웠는가(GSM8K↑) × safety 가 남았는가(ASR↓)*.

**비교 축:**
1. `all_effective` / `topk4` / `energy90` vs **통제군** → 제약의 대가와 효과
2. `topk4` vs `energy90` → 같은 예산에서 **고정 배분 vs 적응 배분**
3. WSR-LoRA 라인의 다른 method (`lora`, `wsr_lora`, `safe_lora`, `salora`, …) 와의 비교

---

## 9. 재현 방법

```bash
# 0) 환경
git clone <repo> && cd Safety-WaRP-LLM
hf auth whoami                      # `hf auth login` 아님 — §10.2

# 1) safety 자산 복구 (재학습 불필요, ~3.5시간 절약)
mkdir -p outputs/adapter_subspace_lora
hf download kmseong/llama2_7b-chat-Safety-LoRA-r16-lr1e-4-adapter \
    --local-dir outputs/adapter_subspace_lora/safety_adapter
hf download kmseong/llama2_7b-chat-Safety-LoRA-r16-lr1e-4 \
    --local-dir outputs/adapter_subspace_lora/safety_merged
# 이미 HF 에 있으므로 재업로드 방지 마커를 만들어 둔다
touch outputs/adapter_subspace_lora/safety_adapter/.pushed_kmseong_llama2_7b-chat-Safety-LoRA-r16-lr1e-4-adapter
touch outputs/adapter_subspace_lora/safety_merged/.pushed_kmseong_llama2_7b-chat-Safety-LoRA-r16-lr1e-4

# 2) Q_S 재생성 + 스펙트럼 확인 (수 초, GPU 불필요)
STOP_AFTER_STAGE1=1 bash scripts/run_adapter_subspace_lora.sh

# 3) Stage 2/3 (약 1.5시간 + 업로드)
bash scripts/run_adapter_subspace_lora.sh
```

스크립트는 **완료된 stage 를 건너뛴다** (`adapter_model.safetensors`, `config.json`,
`report.json`, `summary.json` 존재 여부로 판정). 죽어도 재실행하면 이어서 간다.

주요 노브 (`scripts/run_adapter_subspace_lora.sh` 상단):
```bash
HBPY=${HBPY:-/venv/hb/bin/python}
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"   # ⚠️ safety 이전의 BASE
LR_LIST=(1e-4); EPOCHS=3; BATCH=2; GRAD_ACCUM=8
SAFETY_ADAPTER_MODE=merge        # merge | keep
LORA_PARAM_DTYPE=float32         # 제약 정밀도 fp32≈1e-9 / bf16≈1e-3
TRAIN_SELECTIONS=(all_effective topk4 energy90)
STOP_AFTER_STAGE1=0
```

---

## 10. 함정 모음

1. **`--model_name` 의 의미가 method 마다 다르다.** `adapter_subspace_lora` 만 **base** 모델이고
   safety 는 `--safety_adapter_path` 로 얹는다. 나머지 method 는 `--model_name` 이 safety 모델이다.
   스크립트가 시작 시 경고를 찍는다.
2. **`hf auth login` 으로 토큰을 확인하지 말 것.** 토큰 파일이 있으면 유효성과 무관하게
   "Already logged in" 을 출력한다. 반드시 **`hf auth whoami`**.
3. **`keep_ratio`(wsr 원소 비율), `direction_keep_ratio`(원본 열 비율), `k`(adapter subspace 방향 수)
   는 전부 다른 물리량이다.** 표에서 나란히 놓지 말 것.
4. **재구성 오차를 fp32 로 계산하면 안 된다.** §5.3 참조. 또한 `‖PΣQᵀ‖² = Σσ²` 는 P,Q 가 **정확히**
   정규직교일 때만 성립하므로 `tr(Σ(PᵀP)Σ(QᵀQ))` 형태를 써야 한다
   (`models/adapter_subspace.py:130` 에 구현·주석 있음).
5. **gradient 투영만으로는 제약이 유지되지 않는다.** §3 Stage 2 참조. post-step 재투영이 본체다.
6. **push 실패는 non-fatal 로 설계돼 있다** (`finetune_gsm8k_lora.py:562`). 학습을 죽이지 않는
   대신 `PUSH_FAILED` 한 줄만 남기므로, 로그 grep 이나 HF API 조회로 **직접 확인**해야 한다.
7. **이 인스턴스는 `workspace_is_volume: false`** — `/workspace` 를 포함해 컨테이너 파일시스템
   전체가 recycle/destroy 시 사라진다(HF 토큰 파일 포함). stop/start 는 안전.
   업로드 완료 전에 recycle 하지 말 것.
8. **Phase1/2/3 의 `--layer_type`, `--target_layers` 는 반드시 동일해야 한다** — 단 이는
   WSR-Tune 파이프라인 이야기이고, `adapter_subspace_lora` 는 해당하지 않는다(§2).
