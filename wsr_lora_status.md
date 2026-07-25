# WSR-LoRA 실험 현황 및 환경 이전 가이드

> 작성 2026-07-25. `wsr_lora_comparison.md`(사전 **지시서**)와 역할이 다르다. 이 문서는
> **지금까지 실제로 무엇이 만들어졌고, 어디까지 돌았고, 새 환경에서 어떻게 이어받는가**를 기록한다.
> 지시서와 구현이 어긋난 부분은 이 문서를 사실로 본다.

---

## 0. 한 줄 요약

WSR-Tune(논문 본문)은 **full-param** 기법이다. 리뷰어의 "LoRA와 결합은?" 질문(논문 결론에 직접
*"Future work may further combine this perspective with low-rank adaptation"* 이라 적음)에 답하기 위해,
**동일한 LoRA 파라미터 예산**에서 "증분이 허용되는 좌표계와 제약 방식"만 바꾼 여러 방법을 비교한다.

현재 7개 방법이 구현되어 있고, 그중 5개는 학습·업로드까지 끝났다. 가장 최근에 추가된
`adapter_subspace_lora`만 **Stage 1까지 완료, downstream 학습(Stage 2) 미실행** 상태다.

---

## 1. 방법 카탈로그

`W₀` = frozen 시작 weight (m×n), `A ∈ ℝ^{r×n}`, `B ∈ ℝ^{m×r}`, `s = α/r`.
모두 `B=0` 초기화 → 시작 시 ΔW=0 (시작점 = safety 모델).

| method | ΔW | 좌표계 / 제약 단위 | rank | 필요 artifact |
|---|---|---|---|---|
| `lora` | `s·BA` | — | r | 없음 |
| `original_projected_lora` | `s·BA(I−EEᵀ)` | 원본 / **열** | r | safecols |
| `wsr_lora` | `[(1−M)∘(s·BA)]Uᵀ` | 회전 / **원소** | full | **basis + mask** |
| `wsr_lora_nou` | `(1−M)∘(s·BA)` | 원본 / 원소 | full | mask(원본공간) |
| `safe_lora` | 학습 후 `B←C·B` | 출력공간 / 레이어 | r | base+aligned 모델 |
| `salora` | `s·C·BA`, `C=I−V_sV_sᵀ` | 출력공간 / 부분공간 | r | 없음(내부 계산) |
| **`adapter_subspace_lora`** | `s·BA(I−Q_SQ_Sᵀ)` | 회전 / **방향(열)** | r | **safety LoRA adapter** |

진입점:
- `finetune_gsm8k_lora.py --method {lora,original_projected_lora,wsr_lora,wsr_lora_nou,safe_lora,adapter_subspace_lora}`
- `finetune_gsm8k_salora.py` (SaLoRA만 별도 러너)

### 1.1 `adapter_subspace_lora` (신규, 2026-07-25)

**WSR-Tune 파이프라인을 타지 않는다.** Phase 1(활성화 공분산 basis)도 Phase 2(safety gradient
importance)도 실행하지 않는다. 대신 **이미 학습된 safety LoRA adapter 자신의 업데이트 기하**를 쓴다.

```
ΔW_s = s_s·B_s A_s = P_s Σ_s Q_sᵀ        (compact SVD)
Q_S  = Q_s[:, :k]                         (보호할 right singular vectors)
ΔW_d^⊥ = s_d·B_d A_d (I − Q_S Q_Sᵀ)      (A_d Q_S = 0 유지)
W_final = W_base + s_s·B_s A_s + ΔW_d^⊥
⇒ W_final Q_S = W_safe Q_S               (정확)
```

- **right** singular vector를 쓰는 이유: 투영을 오른쪽(입력 쪽)에 걸기 때문. `P_s`(출력 쪽)는 미사용.
- safety adapter `B_s, A_s`는 **고정**. 수정·pruning하지 않는다. 새로 학습하는 건 `B_d, A_d`뿐.
- **rank 보존** → 진짜 PEFT adapter로 merge 가능. `wsr_lora`(원소 마스크)는 rank가 깨져 dense fold가
  강제되지만 이쪽은 그렇지 않다.

**dense ΔW_s를 어디에서도 만들지 않는다.** thin QR 2회 + r×r SVD 1회로 exact:
```
B_s = Q_B R_B,  A_sᵀ = Q_A R_A  ⇒  B_s A_s = Q_B (R_B R_Aᵀ) Q_Aᵀ
K = s_s·R_B R_Aᵀ ∈ ℝ^{16×16} 에만 SVD → P_s = Q_B U_K,  Q_s = Q_A V_K
```

구현: `models/adapter_subspace.py` (438줄) / `build_adapter_subspace.py` (Stage 1 CLI) /
`finetune_gsm8k_lora.py` 내 `_setup_adapter_subspace`, `AdapterSubspaceCallback`.

**제약 유지 메커니즘 (둘 다 필요)**
1. gradient projection — `register_post_accumulate_grad_hook`으로 `g ← g − (gQ_S)Q_Sᵀ`.
   투영은 선형이라 gradient accumulation과 교환된다.
2. **optimizer.step() 후 parameter 재투영** — 생략 불가. grad를 완벽히 투영해도 AdamW의 step은
   `exp_avg/√exp_avg_sq`로 **원소별** 정규화를 거치므로 부분공간을 벗어난다. `exp_avg`는 grad의
   선형결합이라 자동으로 안에 남지만 `exp_avg_sq`는 원소별 제곱이라 투영 개념 자체가 없다.
   제약을 실제로 보장하는 건 이 재투영이다. (decoupled weight decay는 `A←(1−λη)A`라 무해.)

---

## 2. 학습·업로드 완료된 모델 (HF `kmseong/`, 2026-07-25 확인)

시작점: `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` (full-param safety FT)
공통: GSM8K, r=16, α=32, 3 epoch, eff.batch 16, target `q,k,v,up,down`

```
lora                    llama2_7b-chat-gsm8k-lora-r16-lr{1e-4,2e-4}
original_projected      llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr{1e-4,2e-4}
wsr_lora (element)      llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr{1e-4,2e-4}
  keep_ratio 스윕       ...-wsr-lora-elem-kr{0.05,0.15,0.2,0.3}-r16-lr2e-4
safe_lora               llama2_7b-chat-gsm8k-safelora-thr0.35-r16-lr{1e-4,2e-4}
  threshold 스윕        ...-safelora-thr{0.1,0.2,0.3,0.4,0.5}-r16-lr2e-4
  pre-proj 통제군       ...-safelora-preproj-r16-lr{1e-4,2e-4}
salora                  llama2_7b-chat-gsm8k-salora-r16-lr{1e-4,2e-4}
  예산 정합 (α32,5모듈) ...-salora-matched-a32-r16-lr{1e-4,2e-4}
```

2026-07-25 신규:
```
kmseong/llama2_7b-chat-Safety-LoRA-r16-lr1e-4-adapter   # safety LoRA adapter (112MB, B_s/A_s)
kmseong/llama2_7b-chat-Safety-LoRA-r16-lr1e-4           # 위를 병합한 dense 모델 (= W_safe)
```

⚠️ **평가 결과는 이 저장소에 정리되어 있지 않다.** `logs/eval_gsm8k_summary_*.tsv`는 헤더만 있고
비어 있다. GSM8K/ASR 수치는 별도 harness 산출물을 확인해야 한다 (§6).

---

## 3. `adapter_subspace_lora` 진행 상태

### 완료 (SLURM job 1877214, node44/gigabyte_a6000, 3시간 25분)

| Stage | 내용 | 산출물 |
|---|---|---|
| 0 | safety LoRA 학습 (r16/α32/lr1e-4/3ep, circuit_breakers 4994) | `outputs/adapter_subspace_lora/safety_adapter` + HF |
| 0.5 | dense 병합 | `outputs/adapter_subspace_lora/safety_merged` + HF |
| 1 | Q_S 추출 6종 | `outputs/adapter_subspace_lora/subspaces/*` |

loss: 1.9329 → 0.4178 → 0.2350 → **0.2250**

### 검증 (160개 모듈 전부)

| 지표 | 값 |
|---|---|
| `r_effective` | **전부 16** (rank collapse 없음) |
| `reconstruction_error` | **0.00e+00** |
| `orthogonality_error` | **~1e-14** |

`--svd_dtype float64`(기본값) 덕분. fp32면 `ortho ~6e-6`에 머문다.

### 스펙트럼 — safety 업데이트는 소수 방향에 매우 집중되어 있다

```
누적 에너지 (k=1..8)
  L 0 attn_q    0.579 0.748 0.836 0.873 0.904 0.928 0.944 0.957
  L15 attn_q    0.834 0.883 0.915 0.931 0.945 0.955 0.962 0.968
  L31 attn_q    0.925 0.951 0.965 0.973 0.978 0.983 0.986 0.989
```
σ₁ 하나가 평균 **67~78%**. L31 attn_q는 첫 방향만으로 92.5%.

| 선택 | k (min/mean/max) | 보호 에너지 |
|---|---|---|
| `topk2` | 2/2/2 | 81.7% |
| `topk4` | 4/4/4 | 89.2% |
| `energy90` | **1/4.5/12** | 91.5% |
| `topk8` | 8/8/8 | 95.1% |
| `energy99` | 2/13.0/16 | 99.2% |
| `all_effective` | 16/16/16 | 100% |

층 구조:
```
깊이별 k(energy90):  early(0-10) 5.7 | mid(11-21) 4.3 | late(22-31) 3.5
모듈별 k(energy90):  attn_k 5.8 | attn_q 5.1 | ffn_down 4.4 | ffn_up 4.3 | attn_v 3.0
```
깊은 층일수록, `attn_v`일수록 더 집중적. **고정 top-k는 이 구조를 무시한다.**

### 미실행 — Stage 2/3

`STOP_AFTER_STAGE1=1`로 의도적으로 멈춘 상태. 학습할 3점을 정해야 한다.

**추천: `all_effective` / `topk4` / `energy90`**
- `all_effective`(100%) — 기본형, 헤드라인
- `topk4`(k=4, 89.2%) — 고정 예산
- `energy90`(k=4.5, 91.5%) — **`topk4`와 예산이 맞춰진 적응적 대조군.**
  거의 같은 k에서 더 많은 에너지를 봉쇄 → "배분 방식이 중요한가"를 깨끗하게 묻는다.

`topk8`은 `all_effective`와 에너지 차이가 4.9%뿐이라 가장 덜 유익. 안전성이 무너지는 하한을
보고 싶으면 `topk2`(81.7%)를 4번째로 추가.

---

## 4. 새 환경 이전 체크리스트

### 4.1 git으로 넘어가지 **않는** 것

`.gitignore`가 `outputs/`, `checkpoints/`, `*.pt`, `*.safetensors`, 대부분의 `*.json`을 제외한다.
(예외: `data/*.json`은 force-track)

| 자산 | 크기 | 복구 방법 |
|---|---|---|
| safety LoRA adapter | 112MB | **HF에서 다운로드** ✅ |
| merged safety 모델 | 13GB | HF ✅ 또는 adapter+base 재병합 |
| Q_S subspaces 6종 | 187MB | **`build_adapter_subspace.py` 재실행 (수 초)** ✅ |
| Phase 1 basis (`U`) | — | ❌ **이미 소실됨** (아래) |
| Phase 2 masks (kr 0.05/0.15/0.2/0.3) | 841MB×4 | `checkpoints/wsr_kr*/` 로컬에만 |
| 학습된 downstream 모델 | 239GB (`/scratch2`) | HF ✅ |

### 4.2 ⚠️ Phase 1 basis는 이미 없다

`checkpoints/wsr_basis/`, `checkpoints/phase1_sweep_basis/` 모두 **빈 디렉토리**다
(`*_svd.pt` 0개). 즉 **현재 이 머신에서도 `wsr_lora`를 재실행할 수 없다.**
필요하면 `train.py --phase 1`로 재생성해야 한다 (논문 기준 32층 5모듈 ≈ 12분, GPU 32GB).

기존 mask(`checkpoints/wsr_kr*/`)는 남아 있고, mask는 `|∂L/∂W̃|` 크기 기반이라 `U` 열의
**부호 반전에는 불변**이므로 동일 모델·데이터·seed로 재생성한 basis와 재조합해도 원칙적으로 맞는다.
다만 특이값이 축퇴한 부분공간에서는 회전이 생길 수 있으니, 재생성 후 mask 재계산이 안전하다.

**`adapter_subspace_lora`는 이 문제가 없다.** 필요한 건 HF에 있는 adapter 하나뿐이고,
Q_S는 base 모델을 로드조차 하지 않고 수 초 만에 재생성된다. 이식성 면에서 명확한 이점.

### 4.3 환경

```
python : conda env `hb`  → /home/gokms0509/anaconda3/envs/hb/bin/python
         (이 박스에서 torch가 있는 유일한 인터프리터. `python`은 PATH에 없을 수 있음)
torch 2.10.0+cu128 | transformers 4.57.3 | peft 0.18.1 | accelerate 1.12.0 | datasets 4.4.1
```
- transformers 5.x로 올라가면 `Trainer(tokenizer=)` → `processing_class=` 로 바뀐다.
- peft: `PeftModel.set_adapter`는 **str만** 받는다. 다중 adapter 동시 활성화는
  `model.base_model.set_adapter(["safety","downstream"])` (LoraModel API). 그 다음 safety를 freeze할 것
  (`set_adapter`가 활성 adapter의 `requires_grad`를 켜기 때문에 **순서가 중요**).
- gated `meta-llama/Llama-2-7b-chat-hf` 접근 필요 → HF 토큰 필수.

### 4.4 SLURM

- **`CUDA_VISIBLE_DEVICES`를 절대 설정하지 말 것.** 스케줄러가 할당 GPU로 설정해 준다.
  하드코딩하면 할당받지 않은 물리 GPU를 잡으려다 실패한다.
- 이 저장소에는 이전 비-SLURM 박스에서 온 하드코딩이 있었고 **주석 처리했다**:
  - `models/phase0_SSFT.py:33`
  - `gsm8k_eval/finetune_gsm8k_full_params.py:62` ← **import 시점**에 설정하므로
    이걸 import하는 모든 모듈을 오염시킴. 새 환경에서도 되살리지 말 것.
- 제출: `sbatch scripts/sbatch_adapter_subspace_lora.sh`
- 파티션마다 `AllowQos`가 다르다. `gigabyte_a6000`/`suma_a6000`은 `base_qos,big_qos`(기본 QOS로 제출 가능),
  `suma_a100`은 `a100_qos,a100_low_qos`가 필요해 그냥 제출하면 `Invalid qos specification`.
- `sacct`/`sacctmgr`는 slurmdbd가 자주 죽어 `Connection refused`가 난다. **`squeue` 실패를
  작업 종료로 오판하지 말 것** (연속 N회 확인 필요).

### 4.5 HF 토큰

`hf auth login`은 토큰 파일만 있으면 유효성과 무관하게 "Already logged in"을 출력한다.
**반드시 `hf auth whoami`로 확인.** 2026-07-25 기준 유효 (`user=kmseong`, `repo.write` 포함).

---

## 5. 새 환경에서 이어서 하기

```bash
# 0) 코드
git clone <repo> && cd Safety-WaRP-LLM
conda env create -f environment.yml   # 또는 pip install -r requirements.txt
hf auth whoami                         # login 아님. 유효성 확인

# 1) safety 자산 복구 (재학습 불필요, ~3시간 절약)
mkdir -p outputs/adapter_subspace_lora
hf download kmseong/llama2_7b-chat-Safety-LoRA-r16-lr1e-4-adapter \
    --local-dir outputs/adapter_subspace_lora/safety_adapter
hf download kmseong/llama2_7b-chat-Safety-LoRA-r16-lr1e-4 \
    --local-dir outputs/adapter_subspace_lora/safety_merged

# 2) Q_S 재생성 (수 초, GPU 불필요)
#    run_all 스크립트가 Stage 0/0.5를 건너뛰고 Stage 1부터 알아서 진행한다.
STOP_AFTER_STAGE1=1 bash scripts/run_adapter_subspace_lora.sh

# 3) 학습할 3점 확정 후 Stage 2/3
#    scripts/run_adapter_subspace_lora.sh 의 TRAIN_SELECTIONS 수정
sbatch scripts/sbatch_adapter_subspace_lora.sh   # STOP_AFTER_STAGE1=0 으로
```

스크립트는 **완료된 stage를 건너뛴다**(`adapter_model.safetensors`, `config.json`,
`report.json`, `summary.json` 존재 여부로 판정). 죽어도 재제출하면 이어서 간다.

### 주요 노브 (`scripts/run_adapter_subspace_lora.sh` 상단)

```bash
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"   # ⚠️ safety 이전의 BASE. 다른 method와 의미가 다름
SAFETY_LORA_R=16; SAFETY_LR=1e-4
LR_LIST=(1e-4); EPOCHS=3; BATCH=2; GRAD_ACCUM=8
SAFETY_ADAPTER_MODE=merge        # merge | keep
LORA_PARAM_DTYPE=float32         # 제약 정밀도 fp32≈1e-7 / bf16≈1e-3
TRAIN_SELECTIONS=(all_effective topk4 energy90)
STOP_AFTER_STAGE1=0
```

Stage 0의 epochs/batch/samples는 `models/phase0_SSFT.py`의 **모듈 상수**라 CLI로 못 바꾼다
(현재 3ep / batch 4 / grad_accum 4 / 4994샘플 = eff.batch 16).

---

## 6. 평가 (이 저장소 밖)

학습 스크립트는 merged 모델 생성·업로드까지만 한다. 채점은 별도:

- **GSM8K**: `~/lm-evaluation-harness/eval_models.sh`의 `model_list=(...)`에 HF ID 추가 후
  `sbatch ~/code_lm_eval.sh`. 전용 드라이버를 새로 만들지 말 것.
- **ASR**: HarmBench (`configs/model_configs/models.yaml`에 엔트리 추가 → `MODELS=()`에 단축명).
  ⚠️ **이 머신에는 HarmBench가 없다** (`~/HarmBench` 부재). 새 환경에서 별도 설치 필요.
  AutoDAN/PAIR/PAP 생성은 모델당 수십 분~시간 단위로 비싸다.

---

## 7. 함정 모음

1. **`--model_name`의 의미가 method마다 다르다.** `adapter_subspace_lora`만 **base** 모델이고
   safety는 `--safety_adapter_path`로 얹는다. 나머지는 `--model_name`이 safety 모델이다.
   스크립트가 시작 시 경고를 찍는다.
2. **`keep_ratio`(wsr 원소 비율)와 `direction_keep_ratio`(원본 열 비율)와 `k`(adapter subspace 방향 수)는
   전부 다른 물리량이다.** 표에서 나란히 놓지 말 것.
3. **재구성 오차 지표를 fp32로 계산하면 안 된다.** `‖X‖²+‖Y‖²−2⟨X,Y⟩` 꼴이라 상쇄가 심해
   fp32에서는 상대오차가 `sqrt(eps)≈3e-4` 밑으로 안 내려간다. 또한 `‖PΣQᵀ‖²=Σσ²`는 **P,Q가
   정확히 정규직교일 때만** 성립 — fp32 QR의 1e-6 직교성 오차가 실제 잔차(~1e-15)를 덮어버린다.
   `tr(Σ(PᵀP)Σ(QᵀQ))` 형태로 계산할 것 (`models/adapter_subspace.py`에 구현·주석 있음).
4. **bf16 LoRA 파라미터로는 `A_d Q_S = 0`이 ~1e-3에서 멈춘다.** 기본값 fp32. 다른 baseline과
   dtype을 맞추려면 `--lora_param_dtype bfloat16`, 그 경우 `constraint_A ≈ 1e-3`이 정상값이다.
   `summary.json`에 항상 기록되므로 숨은 변수가 되지 않는다.
5. **Phase1/2/3의 `--layer_type`, `--target_layers`는 반드시 동일해야 한다** (basis·mask·학습이
   같은 `(layer_idx, layer_type)` 키로 인덱싱됨). 어긋나면 조용히 틀린 결과가 나온다.
6. `models/phase0_SSFT.py`의 `MODEL_NAME` 기본값이 `"meta-llama/Llama-2-7B-chat"`인데 HF에 없는
   repo id다 (정확히는 `meta-llama/Llama-2-7b-chat-hf`). 인자 없이 직접 돌리면 걸린다.
7. `SafetyDataset`은 모든 샘플을 `max_length=1024`로 패딩한다(동적 패딩·packing 없음).
   그래서 Stage 0가 A6000에서 3.0 s/it, 에폭당 62분이 걸린다. 속도가 문제면 여기가 개선 지점.

---

## 8. 관련 문서

| 파일 | 내용 |
|---|---|
| `WaRP.md` | 기저 변환 이론 (`W = V(basis_coeff)U`) |
| `CLAUDE.md` | 저장소 전반 구조 |
| `wsr_lora_comparison.md` | 3-method 비교의 **사전 지시서** (구현 전 작성) |
| **이 문서** | 실제 구현·실행 결과 및 이전 가이드 |
| `seal/README.md` | SEAL × WaRP 통합 (별개 라인) |
