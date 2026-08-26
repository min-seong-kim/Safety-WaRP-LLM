# Revision 실험 — Table 2 / Table 4 / Figure 4 확장

논문 `Rethinking LLM Fine-Tuning via Weight Space Reparameterization` 의 revision 용
실험 러너. **12개 기법 × 6개 모델 × 2개 안전 데이터셋**으로 확장한다.

```
bash scripts/revision/run_all.sh          # 전체 (먼저 PLAN_ONLY=1 로 규모 확인 권장)
```

---

## 1. 실험 매트릭스

| 축 | 값 |
|---|---|
| 안전 데이터 | `cb` (Circuit Breakers 4994) · `bt` (BeaverTails 4994) |
| 모델 | llama2_7b · llama2_13b · llama32_3b · llama31_8b · qwen25_7b · gemma2_9b |
| 기법 (기존 5) | `fullft` `safeinstr` `resta` `safedelta` `wsr_tune` |
| 기법 (신규 7) | `lora` `asft` `lisa` `seal` `safelora` `salora` `wsr_lora` |
| Table 2/4 태스크 | 모델별 primary — gsm8k(2_7b, 2_13b, qwen, gemma) · math(3.2_3b, 3.1_8b) |
| Figure 4 태스크 | llama2_7b 한정 추가 — medqa · arc · agnews |

→ **216 학습 셀** + BT SSFT 5개 + Phase1 basis 12개 + Phase2 mask 12개.

`SN-Tune` / `RSN-Tune` 은 사용자 지시대로 제외했다.

### 출발 모델 (safety-tuned)

| 모델 키 | base | CB 출발모델 | BT 출발모델 |
|---|---|---|---|
| `llama2_7b` | `meta-llama/Llama-2-7b-chat-hf` | `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` | `wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv` |
| `llama2_13b` | `meta-llama/Llama-2-13b-chat-hf` | `wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5` | **Stage 01 에서 학습** |
| `llama32_3b` | `meta-llama/Llama-3.2-3B-Instruct` | `kmseong/llama3_2_3b-instruct-SSFT-lr5e-5` | **Stage 01 에서 학습** |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | `kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5` | **Stage 01 에서 학습** |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | `wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5` | **Stage 01 에서 학습** |
| `gemma2_9b` | `google/gemma-2-9b-it` | `wvnvwn/gemma-2-9b-it-ssft-lr3e-5` (lr **3e-5**) | **Stage 01 에서 학습** |

---

## 2. 하이퍼파라미터

전부 `common.sh` 한 곳에 있다. 바꿀 일이 있으면 **거기만** 고칠 것.

**Full-parameter** (`fullft` `safeinstr` `seal`-S2 `wsr_tune`)
`epochs 3 · effective batch 16 · lr 5e-5 · weight_decay 0.01 · warmup 0.1 · cosine ·
max_len 1024 · max_grad_norm 1.0 · seed 42 · bf16`

**LoRA** (`lora` `asft` `lisa` `safelora` `salora` `wsr_lora`)
`r 16 · alpha 32 · dropout 0.05 · targets {q,k,v,up,down} · epochs 3 · effective batch 16 ·
max_len 1024 · warmup 0.03 · weight_decay 0.0 · cosine · max_grad_norm 1.0 · seed 42 · bf16`
`lr = 3e-4` (gsm8k / math / medqa / arc) · `7e-5` (agnews)

**기법 고유**

| 기법 | 값 | 근거 |
|---|---|---|
| SafeInstr | `safety_mix_ratio 0.1` | 논문 §4.1 |
| RESTA | `γ = 0.3` | 논문 §4.1 |
| SafeDelta | `s = 0.1` | 논문 §4.1 |
| WSR-Tune / WSR-LoRA | `ρ = 0.1` | 논문 기본 freeze ratio |
| AsFT | `λ = 1.0` | 사용자 지정 |
| LISA | `ρ 1.0 · align_step 100 · finetune_step 900` | 사용자 지정 |
| SafeLoRA | `threshold 0.3` | 사용자 지정 |
| SaLoRA | `rank_safe 32 · rank_util 32 · calib 128` | 기존 budget-matched 설정 |
| SEAL | `top-p 0.8 · selector 2 epoch` | 기존 설정 |

**effective batch = 16 은 전 기법 공통 불변식**이다. micro-batch 는 모델/기법별로
다르되 `grad_accum = 16 / micro` 로 자동 계산되며, 16 의 약수가 아니면 실패한다.

---

## 3. 설계상 중요한 3가지

### (a) 모든 arm 이 같은 태스크 JSON 하나를 읽는다

러너가 6갈래고 토큰화 구현도 6개다. 각자 데이터셋을 로드하면 "기법 차이"가 아니라
"프롬프트 차이"를 재게 된다. 그래서 태스크마다 `{"question","response"}` JSON **한 개**를
만들어 전 러너에 같은 파일을 넘긴다.

| task | 파일 | rows |
|---|---|---|
| gsm8k | `data/gsm8k_train_task_7473.json` | 7473 |
| math | `data/math_train_task_7500.json` | 7500 |
| arc | `data/arc_challenge_train_task_1119.json` | 1119 |
| medqa | `data/medqa_train_task_10178.json` | 10178 |
| agnews | `data/agnews_train_8k_seed42.json` | **8000 (전체 120k 아님)** |

> ⚠️ **AG News 만 서브셋이다.** 기존 AGNEWS 결과가 전부 이 8k seed42 서브셋으로
> 나왔기 때문에 그대로 맞췄다. 전체 120k 를 쓰려면 `AGNEWS_TASK_JSON` 을 덮어쓰되,
> 기존 수치와 짝이 맞지 않게 된다.

검증:
```bash
python scripts/revision/verify_prompt_parity.py --models all
```
6개 토큰화 경로(`fullft` / `lora` / `lisa` / `seal` / `wsr_lora` / `wsr_tune`)에 같은 행을
통과시켜 `(input_ids, labels)` 가 완전히 같은지 확인한다. **6 모델 × 5 태스크 × 6 경로
전부 일치 확인 완료.**

### (b) 안전 데이터 축은 하나다

안전 데이터를 쓰는 기법(SafeInstr / SafeDelta / LISA / SaLoRA / SEAL / WSR-Tune /
WSR-LoRA)은 전부 **그 출발 모델을 안전정렬할 때 쓴 바로 그 데이터셋**을 쓴다.
스테이지 스크립트의 `$safety` 루프 변수 하나가 `aligned_for()` 와 `safety_json()` 을
동시에 결정하므로 구조적으로 섞일 수 없다.
(RESTA / AsFT / SafeLoRA 는 별도 안전 코퍼스 대신 `V = W_aligned − W_base` 를 쓰므로
같은 축에 자동으로 묶인다.)

### (c) WSR-LoRA = PiSSA 변형

rebuttal 에 적힌 설계(`W_0 = W_res + B_0A_0`, `B_0 = P_rΛ_r^{1/2}`, `Ã = A U`)는
`wsr-lora/wsr_lora.py --reparam` 이다. `finetune_gsm8k_lora.py --method wsr_lora` 의
옛 element-wise product-mask 변형과 **다른 것**이므로 쓰지 않는다.
Stage 02 의 Phase 1 basis 가 필수다.

---

## 4. 스테이지

| # | 스크립트 | 내용 | 선행조건 |
|---|---|---|---|
| 00 | `00_prepare.sh` | 태스크 JSON 5종 생성/점검 + 환경 점검 | — |
| 01 | `01_ssft_bt.sh` | BT 안전정렬 출발모델 5종 (Phase 0 SSFT) | 00 |
| 02 | `02_warp_basis_mask.sh` | Phase 1 basis + Phase 2 mask (모델×안전데이터) | 01 |
| 10 | `10_fullft_safeinstr.sh` | Full FT · SafeInstr | 00, 01 |
| 11 | `11_posthoc.sh` | RESTA · SafeDelta (**Full FT 산출물을 먹는다**) | 10 |
| 12 | `12_wsr_tune.sh` | WSR-Tune (Phase 3) | 02 |
| 20 | `20_lora_family.sh` | lora · asft · safelora · lisa · salora · wsr_lora | 02 (wsr_lora 만) |
| 21 | `21_seal.sh` | SEAL S1 → S1.5 → S2 | 00, 01 |

`run_all.sh` 가 이 순서대로 부른다. **전 스테이지 재개 가능** — 셀마다 `.done`
센티넬을 남기므로, 중간에 죽어도 같은 명령을 다시 돌리면 완료분을 건너뛴다.

---

## 5. 사용법

```bash
conda activate hb          # 또는  export PY=/path/to/python

# 규모 확인 (아무것도 실행하지 않는다)
PLAN_ONLY=1 bash scripts/revision/run_all.sh
SHOW_PENDING=1 PLAN_ONLY=1 bash scripts/revision/run_all.sh   # 남은 셀 목록까지

# 명령만 출력해서 눈으로 확인
DRY_RUN=1 bash scripts/revision/run_all.sh

# 쪼개서 돌리기 (권장)
SAFETY_SETS=cb MODELS=llama2_7b bash scripts/revision/run_all.sh
METHODS="fullft resta safedelta" bash scripts/revision/run_all.sh
STAGES="10 11" bash scripts/revision/run_all.sh
TASKS=agnews MODELS=llama2_7b bash scripts/revision/run_all.sh

# 스모크 테스트 (몇 분)
OUT_ROOT=/tmp/smoke TASK_SAMPLES=24 EPOCHS=1 SAFETY_SAMPLES=64 \
  MODELS=llama2_7b SAFETY_SETS=cb TASKS=arc METHODS="fullft lora" \
  bash scripts/revision/run_all.sh
```

### 주요 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `MODELS` `SAFETY_SETS` `METHODS` `TASKS` | 전체 | 선택자 (공백 구분) |
| `STAGES` | `00 01 02 10 11 12 20 21` | 돌릴 스테이지 |
| `DRY_RUN` `PLAN_ONLY` `SHOW_PENDING` | 0 | 확인용 |
| `TASK_SAMPLES` | 0(전체) | 스모크용 축소 |
| `<MODEL>_MB_{FULL,WARP,LORA,P12}` | 레지스트리 | OOM 시 조정 (16 의 약수 유지) |
| `OUT_ROOT` `CKPT_ROOT` `LOG_ROOT` | `outputs/revision` 등 | 경로 |
| `PY` | `python` | 인터프리터 |
| `CONTINUE_ON_ERROR` | 1 | 0 이면 첫 실패에서 중단 |
| `SAFEDELTA_DIR` | `/home/edgeai_lab/SafeDelta` | SafeDelta 외부 구현 |
| `SAFELORA_LOAD_DTYPE` | `float32` | RAM 부족 시 `bfloat16` |

**`CUDA_VISIBLE_DEVICES` 는 절대 스크립트에서 설정하지 않는다** (CLAUDE.md).
SLURM 이 넣어준 값을 그대로 쓴다.

---

## 6. 산출물

```
outputs/revision/<safety>/<model>/<task>/<method>/
    ├── .done            완료 센티넬
    ├── MODEL_DIR        실제 모델 디렉토리의 절대경로 (러너마다 위치가 달라서)
    ├── run.log
    └── (모델 파일 또는 merged_model/)

checkpoints/revision/
    ├── ssft_bt/<BaseName>-ssft-bt-lr<lr>/      BT 안전정렬 출발모델
    └── warp/<safety>/<model>/
            ├── BASIS_DIR    Phase 1 basis 경로가 적힌 파일
            ├── MASKS_DIR    Phase 2 mask 경로가 적힌 파일
            └── phase{1,2}_<ts>/

logs/revision/
```

평가는 이 저장소 밖이다(HarmBench ASR + 태스크별 정확도). 각 셀의 `MODEL_DIR` 을 읽으면
러너별 저장 위치 차이를 신경 쓰지 않아도 된다.

---

## 7. 이 작업에서 실제로 고친 버그

리뷰 시 참고. 전부 revision 실험의 공정성을 직접 깨뜨리는 것들이었다.

1. **`is_instruct_model` 판정이 모듈마다 달랐다.**
   `agnews_eval` / `gsm8k_eval` / `seal` 은 `"instruct"|"chat"` 만 봤고,
   `models/phase3_extra_learning` 은 `'it'` 도 봤다. → **`gemma-2-9b-it` 는 arm 마다
   다른 프롬프트 포맷(chat template vs plain)으로 학습되고 있었다.**
   5개 모듈을 토큰 경계 기반 동일 규칙으로 통일했다.

2. **BT SSFT 출력 디렉토리 이름에 모델 태그가 없으면 같은 문제가 재발한다.**
   러너들이 모델 참조 **문자열**로 chat template 사용 여부를 판정하므로,
   `ssft_bt/llama2_13b_lr5e-5` 같은 이름은 plain 프롬프트로 fallback 된다.
   `ssft_bt/Llama-2-13b-chat-hf-ssft-bt-lr5e-5` 처럼 base 이름을 유지하게 했다.

3. **WaRP Phase 3 이 agnews 를 다른 경로로 로드했다.**
   `_load_agnews` 는 `{"question","response"}` 스키마를 못 읽고, 샘플 수 제한 시
   **셔플**까지 한다 → baseline arm 과 데이터가 어긋난다.
   `--phase3_task_data_path` 를 추가해 태스크 종류와 무관하게 공용 로더를 쓰게 했다.

4. **Phase 1 basis metadata 에 `decomp` 키가 없어 WSR-LoRA 가 무조건 실패했다.**
   `wsr-lora/wsr_lora.py:validate_shared_basis` 가 `metadata["decomp"] == "svd"` 를
   요구하는데 이 저장소 Phase 1 은 그 키를 쓴 적이 없다.
   Phase 1 이 `'decomp': 'svd'` 를 남기게 하고, 옛 basis 를 위해 소비자 쪽은
   "키 없음"은 허용하되 "다른 값"은 거부하도록 했다.

5. **`models/safelora_baseline.py` / `models/salora.py` 가 워킹트리에서 삭제돼 있었다.**
   `finetune_gsm8k_lora.py:682` 와 `finetune_gsm8k_salora.py:50` 의 import 가 깨진 상태였다.
   복구했다 (`safelora/` 스냅샷과 바이트 동일).

6. **`phase0_SSFT.py` 의 batch/grad_accum 이 하드코딩(4×4)이라 13B/9B 가 OOM 난다.**
   `SSFT_BATCH_SIZE` / `SSFT_GRAD_ACCUM` 환경변수로 뺐다(곱=16 유지).

7. **SafeDelta 원본이 `CUDA_VISIBLE_DEVICES="2,3"` 을 import 시점에 박아뒀다.**
   이 박스는 GPU 1장이라 무조건 실패한다. `os.environ.setdefault(..., "0")` 로 고쳤다
   (`/home/edgeai_lab/SafeDelta/llama2/run_safedelta.py`).

---

## 8. 스모크 검증 결과 (2026-08-26, llama2_7b / cb / arc)

12개 기법 전부 실제 러너로 end-to-end 통과했다. `TASK_SAMPLES=24 EPOCHS=1 SAFETY_SAMPLES=64`.

| 기법 | 결과 | 확인한 것 |
|---|---|---|
| `fullft` | ✓ | 6.74B 전체 학습, `chat_template.jinja` 저장됨 |
| `safeinstr` | ✓ | 같은 러너 · `--safety_mix_ratio` 만 다름 |
| `resta` | ✓ | 291 텐서 병합, 가중치합 1.000000, rotary 버퍼 32개 제외 |
| `safedelta` | ✓ | 외부 구현 호출 성공, 산출물 규약 경로로 이동, `chat_template: OK` |
| `wsr_tune` | ✓ | Phase 3 `final_model` → 셀 디렉토리로 승격 |
| `lora` | ✓ | `merged_model/` 저장 |
| `asft` | ✓ | 160 alignment direction, **등가성 `rel_err=1.03e-06`**, 최종 정규화항 `1.80e-02` |
| `lisa` | ✓ | alignment/finetune 교대 |
| `safelora` | ✓ | base+aligned fp32 로드 → 투영 |
| `salora` | ✓ | layer_types ↔ target_modules 정합성 자체 확인 통과 |
| `wsr_lora` | ✓ | Phase 1 basis 재사용, `--reparam` |
| `seal` | ✓ | S1 selector → S1.5 (19/24 선택) → S2 full-param SFT |

**Phase 1 / Phase 2 도 통과**: basis 160개 파일 + `metadata.json`(`decomp: svd`), mask 5 layer_type.

**프롬프트 동일성**: 6 모델 × 5 태스크 × 6 토큰화 경로 = 300행 전부 `(input_ids, labels)` 일치.

> ⚠️ 스모크 중 SEAL S2 가 `save_pretrained` 도중 SIGKILL 로 죽은 적이 있는데, 출력 경로를
> tmpfs(`/tmp`, RAM 기반)로 잡아 두고 그 위에 이미 190GB 를 쌓아둔 탓이었다. 디스크 경로로
> 바꾸니 정상 완료했다. **산출물을 tmpfs 에 쓰지 말 것.**

---

## 9. HF 업로드 · 디스크 운용

**216 셀 전체를 로컬에 남기면 3.4 TB 다. 이 박스의 여유는 155GB.**
그래서 기본 운용은 **셀이 끝날 때마다 허브에 올리고 → 검증하고 → 로컬 가중치를 지우는** 방식이다.

```bash
PUSH_TO_HUB=1 bash scripts/revision/run_all.sh
```

### 리포명 규약

```
kmseong/{model}-{CB|BT}_SSFT-{method}_{task}[_{hparam}]_lr{lr}
```

| 예시 | |
|---|---|
| `kmseong/llama2_7b-chat-CB_SSFT-fullft_gsm8k_lr5e-5` | 하이퍼파라미터 없는 기법은 슬롯 생략 |
| `kmseong/llama2_7b-chat-CB_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5` | 기법명의 `_` 는 `-` 로 (구분자 충돌) |
| `kmseong/gemma2_9b-it-BT_SSFT-asft_gsm8k_lambda1.0_lr3e-4` | lr 은 기법에서 자동 도출 |
| `kmseong/llama3_1_8b-instruct-CB_SSFT-safedelta_math_s0.1_lr5e-5` | |
| `kmseong/llama2_13b-chat-BT_SSFT-lr5e-5` | BT 안전정렬 출발모델 |

- **222개 전부 고유**, 최장 65자 (HF 한도 96), 허용문자 위반 0 — 검사 완료.
- LoRA 계열은 전부 `r=16 / alpha=32` 로 동일해 이름에 넣지 않는다.
- 하이퍼파라미터 태그: `mix0.1`(SafeInstr) `gamma0.3`(RESTA) `s0.1`(SafeDelta)
  `rho0.1`(WSR-Tune/WSR-LoRA) `lambda1.0`(AsFT) `rho1.0`(LISA) `topp0.8`(SEAL)
  `thr0.3`(SafeLoRA) `rs32ru32`(SaLoRA).
- 이름 규칙은 `common.sh` 의 `hf_repo_id()` 한 곳에서 생성된다. 손으로 적지 않는다.
- **생성될 221개 리포 전체 목록: [`REPO_LIST.md`](REPO_LIST.md)**
  (`bash scripts/revision/gen_repo_list.sh > scripts/revision/REPO_LIST.md` 로 재생성.
  하이퍼파라미터를 바꾸면 이름도 바뀌므로 반드시 다시 생성할 것.)

### 업로드 → 검증 → 삭제

`scripts/revision/upload_and_prune.py` 가 담당한다. **검증 4가지를 전부 통과해야만** 지운다:

1. 허브에 필요한 파일이 전부 있는가
2. 파일 크기가 로컬과 일치하는가
3. `AutoConfig.from_pretrained(repo_id)` 가 로드되는가
4. **`AutoTokenizer.from_pretrained(repo_id).chat_template` 이 허브에서 살아 있는가**
   — transformers 4.4x/5.x 는 chat template 을 `chat_template.jinja` 별도 파일로 쓴다.
   모델·토크나이저 객체만 push 하면 이 파일이 조용히 빠지고, 허브 사본이
   `chat_template=None` 으로 로드되어 평가 프롬프트가 학습 때와 달라진다.
   **이 저장소에서 실제로 두 번 발생한 사고**라 로컬이 아니라 허브에서 확인한다.

결과는 셀의 `UPLOAD.json`(검증 내역)과 `.uploaded`(리포 id)에 남는다.
`run.log` / `.done` / `MODEL_DIR` / `UPLOAD.json` 은 업로드에서 제외된다.
가중치만 지우고 `config.json` / tokenizer / `chat_template.jinja` 같은 작은 메타데이터는 남긴다
— 그래서 **평가 시 `.uploaded` 의 리포 id 를 쓰면 된다.**

⚠️ **`fullft` 은 RESTA/SafeDelta 의 입력**이라, 그 둘이 끝나기 전에는 삭제하지 않는다
(`prune_allowed()`). 업로드는 즉시 하고 삭제만 미룬다.

### 그 밖의 디스크 소비자 — 둘 다 자동 정리된다

| 대상 | 크기 | 정리 시점 |
|---|---|---|
| Phase1 basis + Phase2 mask | 조합당 8~31GB (**전체 248GB**) | 그 (안전축, 모델) 의 WSR 계열이 끝나면 (`PRUNE_BASIS=1`) |
| HF 캐시 (base + CB/BT aligned) | 모델당 21~78GB | 그 모델의 두 안전축이 끝나면 (`PRUNE_HF_CACHE=1`) |

basis 는 `BASIS_SAVE_DTYPE=bfloat16` + `BASIS_OMIT_UT=1` 로 **fp32+UT 대비 1/4** 이다
(827GB → 248GB). Phase 2/3/WSR-LoRA 모두 로드 직후 `U` 를 모델 dtype(bf16)으로 내리고,
`UT` 키는 이 저장소의 어떤 소비자도 읽지 않으므로 **사용값이 바뀌지 않는다**.

실행 순서는 `ORDER=model`(기본) — 모델을 바깥 루프에 두고 한 모델의 CB/BT 를 연달아
처리한다. 그래야 basis 를 한 조합만 두고 바로 지우고, 모델 캐시를 한 번만 받는다.

### 모델별 동시 최대 디스크 (측정·계산치)

`PLAN_ONLY=1 PUSH_TO_HUB=1 bash scripts/revision/run_all.sh` 가 출력한다:

| model | HF캐시 | basis | 셀×2 | 동시최대 |
|---|---:|---:|---:|---:|
| llama32_3b | 21G | 8G | 14G | **43G** |
| llama2_7b | 45G | 16G | 30G | **91G** |
| llama31_8b | 45G | 21G | 30G | **96G** |
| qwen25_7b | 45G | 26G | 30G | **101G** |
| gemma2_9b | 57G | 26G | 38G | **121G** |
| llama2_13b | 78G | 31G | 52G | **161G ← 155G 여유로는 빠듯** |

**llama2_13b 만 단독으로, 그리고 METHODS 를 둘로 쪼개 돌려라:**

```bash
MODELS=llama2_13b METHODS="fullft safeinstr resta safedelta wsr_tune" PUSH_TO_HUB=1 bash scripts/revision/run_all.sh
MODELS=llama2_13b METHODS="lora asft lisa seal safelora salora wsr_lora"  PUSH_TO_HUB=1 bash scripts/revision/run_all.sh
```

---

## 10. 알려진 제약

- **단일 GPU 순차 실행**이다. 216 셀 전체는 수 주 규모다. `MODELS` / `SAFETY_SETS` /
  `METHODS` 로 쪼개서 돌려라.
- **HF 토큰이 필요하다.** `meta-llama/*` 와 `google/gemma-2-9b-it` 는 gated 이고,
  RESTA / AsFT / SafeLoRA 가 base 모델을 직접 로드한다.
- **SaLoRA 는 원저자 권장 설정(alpha=r=16, {q,v})이 아니라 budget-matched 설정**
  (alpha=32, 5모듈)으로 돈다. 논문 표에 각주를 남길 것.
- **SEAL 의 S2 는 full-parameter** 다. LoRA 열과 직접 비교하지 말고 별도 열로 보고할 것.
- SafeDelta 의 `s` 는 fine-tuning 종류를 가로질러 비교 가능한 값이 아니다. 여기서는
  논문 Table 2 와 같이 **full-param 델타**에 `s=0.1` 을 건다.
