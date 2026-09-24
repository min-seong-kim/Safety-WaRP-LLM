# WSR-(R)SN-Tune 작업 기록

논문 Table 3 (`tab:baseline_plus_warp`) 에 **RSN-Tune** 과 **RSN-Tune + WSR-Tune** 행을
추가하기 위한 작업. 2026-09-22 시작.

구현 문서는 `sn_tune/README.md`, 여기는 **무엇을 왜 그렇게 정했고 지금 어디까지 왔는지**만 적는다.

---

## A. 과거 실행 복원 (로그·허브 포렌식, 2026-09-22)

기억에 의존하지 않기 위해 원본 로그(`Safety-Neuron/neuron_detection/logs/`)와 허브
감사 파일(`logs/hf_storage_audit_20260920.json`, 616개 리포)에서 직접 복원했다.

### A-1. 원공간 (R)SN-Tune 의 실제 설정

`kmseong/llama2_7b_chat_gsm8k_ft_freeze_rsn_lr5e-5_new_revised` 가 쓴
`critical_safety_neuron_20260502_022558.txt` 를 **차집합 일치**로 역추적해 입력 쌍을
특정하고, 해당 실행 로그로 교차검증했다.

| 단계 | 코퍼스 | top-k (ffn/attn) | 뉴런 | 뉴런% | 파라미터% |
|---|---|---:|---:|---:|---:|
| safety | circuit_breakers 4994 | **1200 / 200** | 12,998 | 0.956% | 1.135% |
| utility | **wikipedia 1000** | **300 / 50** | 1,826 | 0.134% | 0.185% |
| critical | — | — | 11,329 | 0.833% | 0.967% |

- **foundation 코퍼스는 Wikipedia 다. Alpaca 가 아니다.** 저장소 전체 grep 에 alpaca 가
  없고(유일한 히트는 MBPP 프롬프트 스타일 주석), 로그가 `[Mode] Utility Neuron Detection
  (Wikipedia)` 로 찍혀 있다. `foundation_neuron_detection.py` 도 Wikipedia 를 쓰지만
  알고리즘이 달라(고정 top-k 가 아닌 `--ffn_active_fraction`) 실제로는 쓰이지 않았다.
- **utility 는 300/50 이 집안 표준**이었다 — 7B chat/base, Llama-3.1-8B(+Instruct),
  Qwen2.5-32B 실행이 전부 300/50. safety 만 모델 크기에 따라 키웠다.
- **13B 는 탐색을 실제로 했다**: 1200/200 → 15,828 = 0.743% (버림),
  **1800/300 → 20,406 = 0.958%** (채택, 7B 의 0.956% 와 나란함).
- **검출은 싸다**: 7B/4994 프롬프트 약 4분, wikipedia 1000 문서 약 5분.
- **출발 모델은 plain chat 이다.** sn_tuning 로그 17건 전수가
  `meta-llama/Llama-2-{7b,13b}-chat-hf` / `Llama-3.2-3B-Instruct`. SSFT 모델은 0건.
  SN-Tune 자체가 safety 정렬 단계라 SSFT 를 대체하기 때문이다.
  ⚠️ 이 때문에 Table 3 의 다른 행(SafeInstr / SEAL / WSR-Tune, 전부
  `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` 출발)과 출발점이 다르다 — **각주 필요**.

### A-2. "회전 공간" 세대는 셋이었다

이름이 비슷해 섞이기 쉽다. 전부 2026-05-03 ~ 05-06.

| 세대 | 날짜 | 검출 대상 | SN-Tune 을 어디서 | 대표 리포 |
|---|---|---|---|---|
| **WaRP-SN-Tune** | 05-03 | `basis_coeff` 의 **열** | WaRP 공간 | `kmseong/llama2_7b_chat_WaRP-SN-Tune_lr5e-5` ✅생존 |
| rotation_space | 05-05 | 회전된 *모델*의 행 → 원공간 역매핑 | 원공간 | ❌삭제 |
| basis_rotation | 05-05 | 원본 모델 + 회전 스코어링 훅 | 원공간 | `..._only_rsn_tuned_lr5e-5_basis_rotation` ✅생존 |

살아남은 `WaRP-SN-Tune_lr5e-5` 의 `warp_sn_tune_config.json`:
lr 5e-5 · 3ep · batch 4×4 · warmup 0.1 · max_seq_len 1024 · max_samples 4994 ·
layer_types 5종 · `"note": "Safety neurons are COLUMN indices of basis_coeff"`.
**top_k 와 basis_dir 은 기록되어 있지 않아 검출 파라미터는 복원되지 않았다.**
`_6p` 변형(같은 날 84분 뒤)은 삭제되어 접미사 의미도 알 수 없다.

⚠️ 따라서 **1200/200 이 등장하는 것은 05-05 `_basis_rotation` 로그**이고, 05-03
WaRP-SN-Tune 쪽이 아니다. 두 세대가 섞여 기억된 것으로 보인다.

**과거 WaRP-SN-Tune 은 '열' 버전이다 (2026-09-22 허브에서 재확인).**
살아있는 `kmseong/llama2_7b_chat_WaRP-SN-Tune_lr5e-5` 의 `warp_sn_tune_config.json` 이
`"note": "Safety neurons are COLUMN indices of basis_coeff (WaRP space)"` 라고 명시하고,
그걸 만든 `warp_sn_tune.py` 도 `"index_semantics": "COLUMN indices of basis_coeff = W @ U"`
를 기록한다. 반면 1200/200 을 쓴 05-05 `_basis_rotation` 은 **행** 방식이고 학습도
원공간에서 했으며 리포명이 `*_basis_rotation` 이다.

| 세대 | 날짜 | 버전 | top-k 기록 |
|---|---|---|---|
| `..._WaRP-SN-Tune_lr5e-5` | 05-03 | **열** | 없음 |
| `..._basis_rotation` | 05-05 | **행** | 1200/200 |

### A-3. 이번에 처음 만들어지는 것

허브 616개 리포 전수 확인:
- **WSR-RSN-Tune 은 존재한 적이 없다** (WaRP 계열에 RSN 변형 0건).
- **WaRP-SN 계열의 downstream 모델도 없다.** 05-03 라인은 SN-Tune 에서 멈췄고,
  `finetune_downstream_freeze_warp_sn.py` 로 올린 모델이 하나도 없다
  (= 한 번도 끝까지 안 돌아본 코드).

---

## B. 설계 결정 (2026-09-22)

### B-0. 두 버전의 전체 구조 — 어느 공간에서 무엇을 하는가

가장 헷갈리는 지점이라 먼저 정리한다. **검출·동결·학습이 각각 어느 좌표계에서 일어나는지**가
두 버전의 전부다.

```
                    ┌─ 행(row) 버전 ─────────────┐   ┌─ 열(column) 버전 ────────┐

[검출] 점수 계산     x U Wᵀ  (입력만 회전)            |(xU)_k| · Σ_i|W̃_ik| / √S_k
       고르는 대상   출력 뉴런 j  = W 의 행           basis 방향 k = W̃ 의 열
       후보 풀       out_features (7B ffn_up 11008)  in_features (7B ffn_up 4096)
       인덱스 의미   원공간 인덱스                    WaRP 공간 인덱스

[RSN-Tune] 학습      그 행만                          그 열만
           공간      원공간 (W 직접)                  WaRP 공간 (basis_coeff)

[downstream] 동결    그 행                            그 열
             학습    나머지 전부                      나머지 열 전부
             공간    원공간 (W 직접)                  WaRP 공간 → W = W̃Uᵀ 복원

[저장]               표준 HF 체크포인트                표준 HF 체크포인트 (동일)
```

| | 행 버전 | 열 버전 |
|---|---|---|
| 검출 공간 | 점수만 회전, **대상은 원공간 행** | **WaRP 공간 열** |
| RSN-Tune 공간 | 원공간 | WaRP 공간 |
| 동결 대상 | 행 j (출력 뉴런) | 열 k (입력 basis 방향) |
| downstream 학습 공간 | **원공간** | **WaRP 공간** |
| 구현 | `detect_warp_rotated` → `sn_tune_original` → `finetune_freeze_sn` | `warp_sn_detection` → `warp_sn_tune` → `finetune_downstream_freeze_warp_sn` |

**검출 점수를 정확히 쓰면.** `U` 는 이미 기저이고 `xU` 는 **그 기저에서의 좌표**다
(k번째 성분 = `x · u_k`). "분해한다"가 아니라 "사영한 좌표의 크기를 잰다"가 맞는 표현이다.

| | 행 | 열 |
|---|---|---|
| 점수 | `Σ_t \|x U wⱼ\|` | `Σ_t \|(xU)_k\| · Σ_i \|W̃_ik\| / √S_k` |
| 의미 | `u` 좌표로 바꾼 입력을 **출력뉴런 j** 가 얼마나 크게 받아내는가 | **방향 `u_k`** 가 얼마나 활성화되고 그 방향이 얼마나 쓰이는가 |
| 합치는 축 | `k` 를 모두 더해 `j` 만 남김 | `i`(출력) 를 모두 더해 `k` 만 남김 |

둘 다 같은 `U` 를 쓴다. 다른 것은 **어느 축으로 합치느냐**뿐이다.

⚠️ `√S_k` 정규화는 **열 버전에만 있는 필수 장치**다. `U` 의 열이 특이값 순으로 정렬돼
있어서, 정규화하지 않으면 `u_0` 가 항상 가장 크게 활성화되어 top-k 가 늘 `0,1,2,…` 를
반환한다. 나누고 나면 각 방향의 기대 크기가 평준화되어 **"기대보다 크게 활성화된" 방향**이
선택된다 (`warp_sn_detection.py` 의 주석 참조).

**행 버전이 왜 원공간에서 학습하나.** `W̃ = W U` 는 오른쪽 곱이라 열을 섞고 행은 그대로 둔다:

```
W̃ 의 행 j 동결 → 학습 → 복원 W = W̃ Uᵀ
행 j of W = (행 j of W̃) Uᵀ = (행 j of W·U) Uᵀ = 행 j of W     ← 원래 값 그대로
```

행을 얼리면 WaRP 공간에서 하든 원공간에서 하든 **제약 집합이 같다.** 재파라미터화할 이유가
없어 원공간 도구를 쓴다 (05-05 `basis_rotation` 세대도 그렇게 했다).

열은 다르다. `U` 가 열을 섞으므로 열 k 동결은 원공간 좌표로 표현할 방법이 없다 —
**진짜로 다른 부분공간을 얼린다.** 그래서 "재파라미터화 공간에서 보호한다"는 WSR 주장이
성립하는 것은 **열 버전뿐**이다.

### B-0b. ⚠️ 논문 Table 3 의 기존 행이 어느 버전인지는 **확인되지 않았다**

2026-09-22 조사 결과:
- 논문 Table 3 의 `SN-Tune`(AVG 27.18 / Acc 38.99) 과 `SN-Tune + WSR-Tune`(23.07 / 41.02)
  수치가 저장소 `RESULTS.md` 에 **없다**.
- 이 박스의 HarmBench 결과 트리에 `WaRP-SN` / `basis_rotation` / `rotation_space` 관련
  디렉토리가 **하나도 없다**.
- 따라서 그 수치를 낸 모델을 특정할 수 없다.

간접 증거는 **열 쪽**을 가리킨다 — 이름이 문자 그대로 `WaRP-SN-Tune` 인 유일한 생존 모델
(`kmseong/llama2_7b_chat_WaRP-SN-Tune_lr5e-5`, 05-03)이 열 버전이고, 1200/200 이 나온 05-05
`basis_rotation`(행)은 리포명이 다르다. **정황이지 증거는 아니다.**

**확정 방법**: 그 생존 모델을 이 저장소 조건(sys · GRADING=hard · AdvBench standard)으로
재평가해 23.07 이 재현되는지 본다. 네트워크가 풀리면 `run_all_eval.sh` 한 번이면 된다.

### B-1. 검출은 "기존처럼", 마스크 단위는 **행**

`safety_neuron_detection_v2_basis_rotation.py` 방식을 채택했다. 점수 공식이 원공간 SN 과
**같고 입력만 회전**한다:

```
원공간   score[i] = Σ_{b,t} | (x   @ Wᵀ)_{b,t,i} |
회전     score[i] = Σ_{b,t} | (x U @ Wᵀ)_{b,t,i} |      (M = U @ Wᵀ 로 matmul 1회)
```

점수가 `out_features` 차원이므로 인덱스는 **행(출력 뉴런)** 이고, 원공간 SN 과 같은
대상이라 SN-Tune 과 1:1 비교가 깨끗하다.

**알고 있어야 하는 등가성:** `W̃ = W U` 는 행을 보존하므로

```
W̃ 의 행 j 동결 → 학습 → 복원 W_new = W̃_new Uᵀ
행 j of W_new = (행 j of W̃_old) Uᵀ = (행 j of W·U) Uᵀ = 행 j of W      ← 정확히 동일
```

즉 **행 마스크는 WaRP 공간에서 걸든 원공간에서 걸든 제약 집합이 같다.** 재파라미터화가
바꾸는 것은 (a) 검출 점수, (b) optimizer 기하뿐이다. 그래서 학습은 원공간 도구
(`sn_tune_original.py` / `finetune_freeze_sn.py`)를 그대로 쓴다 — 05-05 `basis_rotation`
세대가 그렇게 했고, 등가성과 일관된다.

⚠️ **논문 서술 주의:** 저장소 본체 WSR-Tune 은 `W̃` 의 **엔트리** 마스크다
(ActSVD ablation arm D). "재파라미터화 공간에서 보호한다"는 주장의 근거가 바로
"마스킹이 원공간과 다르다"인데, **행 마스크는 그 성질을 잃는다.** 각주로 밝히거나
B-2 의 열 버전을 함께 실어야 한다.

### B-2. 열(column) 버전은 **나중에** (사용자 결정 2026-09-22)

> 순서: **행 먼저, 열은 그 다음.** 행 경로 드라이버는
> `sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh`, 열 경로는
> `sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh`. Phase 1 basis 를 공유한다
> (`BASIS_ROOT=outputs/wsr_sn_tune`).


`sn_tune/warp_sn_detection.py` + `finetune_downstream_freeze_warp_sn.py` 경로.
`basis_coeff` 의 **열**(입력 basis 방향)을 점수화하고 동결한다. 원공간과 등가가 아니라
WSR 주장이 살아있고, `actsvd/wsr_actsvd_ablation_spec.md` 의 **arm C**(`U = U_in`,
column mask)와 정확히 일치해 방어하기 쉽다. 대신 점수 기준이 SN-Tune 과 다르다
(`Σ|xU|_k · Σ|basis_coeff|_k / √S_k`).

Phase 1 basis 를 공유하므로 추가 비용은 검출 + 학습뿐이다.

### B-3. 확정된 하이퍼파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| 출발 모델 | `meta-llama/Llama-2-{7b,13b}-chat-hf` | A-1 (plain chat) |
| safety top-k | 7B **1200/200**, 13B **1800/300** | 사용자 결정 — 회전 공간에서도 동일하게 |
| utility top-k | **300/50** (양 모델) | A-1 집안 표준 |
| safety 코퍼스 | `data/circuit_breakers_train.json` 4994 | — |
| utility 코퍼스 | wikipedia 1000 (`sn_tune/corpus/`) | A-1 |
| 학습 | lr 5e-5 · 3ep · batch 4×4 · wd 0.01 · warmup 0.1 · max_len 1024 · bf16 · cosine | 공개 7B/13B RSN 의 `finetune_config.json` |
| downstream | GSM8K 7473 | 동일 |

⚠️ 회전 공간의 1200/200 · 1800/300 은 **과거 실행에서 복원된 값이 아니다** (A-2).
검출 후 파라미터% 를 반드시 기록할 것 — 검출기가 뉴런%와 파라미터%를 둘 다 찍는다.
논문 기준 "≤1%" 는 **파라미터** 쪽이다.

---

## C. 환경

| 용도 | 환경 | transformers |
|---|---|---|
| 회전 검출 (`detect_warp_rotated.py`) | `hb` + **PYTHONPATH 오버레이** | 패치본 |
| 학습 (`sn_tune_original.py`, `finetune_freeze_sn.py`) | `hb` | 정품 4.57.3 |

**`hb_sn` conda 복제는 실패했다** — `conda create --clone` 이 채널 ToS 미동의로 막힌다:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted ...
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

대신 **transformers 패키지만 복사하는 오버레이**를 만들었다 (`setup_patched_transformers.sh`):
99MB, ToS 불필요, 30GB 복제 불필요. 검출은 `PYTHONPATH=.transformers_patched`, 학습은
그대로. 양방향 검증(오버레이=패치 / `hb`=정품) 통과했다. conda 환경이 꼭 필요하면
위 ToS 명령 두 줄을 수락한 뒤 `bash sn_tune/setup_hb_sn.sh`.

### 네트워크

`huggingface.co` 가 **여전히 막혀 있다** — 아펙스 도메인이 물리는 CloudFront
`52.85.128.0/24` 만 블랙홀이고 그 외(hf.co, xet, S3, github, pypi)는 정상.
자세한 진단은 `outputs/revision_seal_lr3e-5/.../PENDING_UPLOAD.md`.

영향과 대응:
- 모델(7B 13G, 13B 25G)·GSM8K 는 캐시에 있었다 → `HF_HUB_OFFLINE=1` 로 진행.
- **토크나이저 파일이 캐시에 없었다** (config/safetensors 만) → `getaddrinfo` 우회로 받음.
- wikipedia 도 없었다 → 같은 우회로 1000문서 덤프 (20초).
- **업로드는 아직 불가.** 관리자 대응 전까지 로컬 보관.

---

## D. 진행 상황 (2026-09-22 17:15 기준)

| 단계 | 7B 행 | 13B 행 | 7B 열 | 13B 열 |
|---|---|---|---|---|
| Phase 1 basis (공유) | ✅ | ✅ | ✅ 재사용 | ✅ 재사용 |
| safety 검출 | ✅ 13,005 | ✅ 20,406 | ✅ 39,680 | ✅ |
| utility 검출 | ✅ 1,866 | ✅ 2,653 | ✅ 16,111 | ✅ |
| critical | ✅ 11,297 | ✅ 17,885 | ✅ 23,569 | ✅ 53,157 |
| (R)SN-Tune | ✅ | ✅ | ✅ | ❌ OOM → 재시도 대기 |
| GSM8K FT | ✅ | ✅ | ❌ cache 오류 → 재시도 대기 | ⬜ |
| 평가 (HarmBench+GSM8K) | ✅ **§D-3** | ✅ **§D-3** | ⬜ | ⬜ |
| 업로드 | ⬜ (네트워크 복구 후) | ⬜ | ⬜ | ⬜ |

**실행 순서 (사용자 결정):** 1200/200 으로 **열 → 행** 순서로 끝까지 수행한 뒤,
600/120 · 400/80 보정 검출을 잰다. 오케스트레이터
(`logs/wsr_rsn/_orchestrate_v2.sh`)가 단계별 `.done` 마커로 이어서 돌린다.

### 실행 중인 작업
- `logs/wsr_rsn/00_orchestrator.log` — 오케스트레이터 (검출→학습 체인, `stages/*.done`)
- `logs/wsr_rsn/08_col_detect_13b.log` — 13B 열 검출 (진행 중)
- 완료: `03_basis.log`(Phase1 ×2), `04/05_row_detect_7b*.log`, `06_row_detect_13b.log`,
  `07_col_detect_7b.log`

### 하드웨어
B200 183GB 단독, 디스크 87T 여유. 두 작업이 GPU 를 공유해 81GB / 99% util.
그래서 검출이 과거 기록(4분)보다 느리다 — 경합 탓이지 이상이 아니다.

### 드라이버
```bash
bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh          # 행 (이번 작업)
MODELS=llama2_7b STOP_AFTER_NEURONS=1 bash ...         # 파라미터% 확인만
bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh           # 열 (나중)
```
검출 단계만 `PYTHONPATH=.transformers_patched` 를 앞세우고, 학습은 정품으로 돈다.
`HF_HUB_OFFLINE=1` 이 드라이버 안에 박혀 있다 (huggingface.co 차단 때문).

### 이번에 고친 것
- `train.py --log_dir` 기본값이 예전 박스 절대경로(`/lustre/gokms0509/...`)라
  `PermissionError: /lustre` 로 죽는다 → 드라이버가 명시적으로 넘기도록 수정.
- 토크나이저 캐시 누락 → 우회로 확보 (7B/13B 둘 다 `chat_template` 확인됨).
- **utility 검출이 wikipedia 를 Hub 에서 직접 받으려다 오프라인에서 죽었다**
  (`ConnectionError: Couldn't reach 'wikimedia/wikipedia' (OfflineModeIsEnabled)`).
  두 검출기에 `--utility_json` 을 추가해 미리 덤프한 JSON 을 읽게 했다 —
  행·열 경로가 **같은 문서**를 보게 되는 이점도 있다.
- 열 드라이버에서 `FREQ_THRESHOLD` 정의가 편집 중 누락 → `set -u` 로 즉사. 복구.

---

## D-2. 검출 결과 (2026-09-22)

모든 값은 Llama-2-chat, circuit_breakers 4994 / wikipedia 1000, freq_threshold 1.0 (정확한 교집합).

### 행(row) 버전 — 회전 점수, 인덱스 = 출력 뉴런

| 모델 | top-k | safety | utility | critical | 잔존율 | critical 파라미터% |
|---|---|---:|---:|---:|---:|---:|
| 7B | 1200/200 · 300/50 | 13,005 | 1,866 | 11,297 | 86.9% | — |
| 13B | 1800/300 · 300/50 | 20,406 | 2,653 | 17,885 | 87.6% | 0.9271% |

safety 뉴런%: 7B **0.9563%**, 13B **0.9581%**.

### 열(column) 버전 — 인덱스 = basis 방향

| 모델 | top-k | safety | utility | critical | 잔존율 | critical 파라미터% |
|---|---|---:|---:|---:|---:|---:|
| 7B | 1200/200 · 300/50 | 39,680 | 16,111 | 23,569 | **59.4%** | **2.4343%** |
| 13B | 1800/300 · 300/50 | 🔄 | 🔄 | 🔄 | — | — |

7B safety 내역: ffn_up 16,378 · ffn_down 15,364 · q 2,641 · k 2,646 · v 2,651
(safety 파라미터 4.0920% = 275.7M / 6.74B).

### 두 버전의 차이가 숫자로 드러난 지점

1. **예산이 4배 다르다.** 같은 1200/200 인데 행은 파라미터 1.14%, 열은 **4.09%**.
   열 후보는 `ffn_up` 기준 4096개(입력 차원)뿐이라 1200 이면 레이어당 29% 이고,
   열 하나가 11008개 파라미터를 차지한다(행은 4096개). **행의 k 를 열에 그대로
   옮기면 안 된다.** → 1200/200 결과를 먼저 보고, 그 다음 600/120 · 400/80 을 잰다
   (사용자 결정 2026-09-22).
2. **utility 겹침이 훨씬 크다.** critical 잔존율이 행 87%, 열 **59%**.
   safety 와 utility 가 둘 다 지배적인 입력 방향에 실리기 때문으로 보인다.
   즉 열 공간에서는 "안전 전용 방향"을 분리하기가 더 어렵다.

### top-k 가 실제로 하는 일 — 왜 같은 1200/200 이 행/열에서 다른 예산이 되나

`--top_number_ffn 1200 --top_number_attn 200` 은 **레이어마다·모듈마다 상위 k개를 고르라**는
지시다. 그 다음 프롬프트 4994개에서 고른 집합의 **교집합**을 취한다 (Eq. 3: `N_safe = ⋂_x N_x`).

k 는 **개수**이지 **비율**이 아니다. 그래서 후보 풀이 다르면 같은 k 가 전혀 다른 예산이 된다.

Llama-2-7B, `up_proj` 의 가중치는 `[out=11008, in=4096]`:

| | 후보 풀 | k/후보 | 교집합 생존 | 생존율 | 단위당 파라미터 |
|---|---:|---:|---:|---:|---:|
| **행**(출력 뉴런) | 11,008 | 10.9% | 105/layer | **8.8%** | 4,096 |
| **열**(입력 basis 방향) | 4,096 | **29.3%** | 512/layer | **42.7%** | **11,008** |

세 가지가 곱해진다:

1. **후보 풀이 2.7배 작다** — 열은 입력 차원(4096)이 후보라 1200 이면 레이어당 29%.
2. **교집합 생존율이 5배 높다** — 행 8.8% vs 열 42.7%. 프롬프트가 바뀌어도 **같은 basis
   방향이 계속 상위에 든다**(안정적). 반면 어떤 출력 뉴런이 가장 세게 켜지는지는
   프롬프트마다 달라서 교집합에서 대부분 걸러진다. *WaRP 좌표계가 프롬프트 공통 구조를
   실제로 뽑아낸다는 방증이기도 하다.*
3. **단위 하나의 파라미터 비용이 2.7배** — `ffn_up` 열 1개 = 11,008개 가중치, 행 1개 = 4,096개.

`ffn_up` 만 보면 행 105×4096 = 43만, 열 512×11008 = 563만으로 **13배**. 다른 모듈이
희석해 전체로는 약 3배가 된다 (행 0.791% vs 열 2.412%).

**결론: 행에서 쓰던 k 를 열에 그대로 옮기면 안 된다.**

### 열 top-k 보정 곡선 (7B safety, 2026-09-22)

| top-k | 개수 | 행기준% | 열기준% |
|---|---:|---:|---:|
| 1200/200 | 39,680 | 2.412% | 4.092% |
| 600/120 | 20,383 | 1.239% | 2.077% |
| **400/80** | **13,537** | **0.823%** | **1.383%** |

거의 선형이다. **400/80 의 safety 가 열기준 1.383%** 이고 critical 은 잔존율 59% 를 적용하면
약 0.8% → 논문의 **≤1% 규격에 들어온다**. 반면 1200/200 의 critical 은 2.43% 로 2.4배 초과다.
→ **열 버전의 제대로 된 운영점은 400/80.** (1200/200 결과는 사용자 요청으로 먼저 확보 중.)

### ✅ 포팅 검증 — 13B 행이 과거 실행을 자릿수까지 재현

| | 뉴런 | 뉴런% |
|---|---:|---:|
| 2026-05-06 (`safety_neuron_20260506_031404.log`, basis rotation 1800/300) | 20,406 | 0.9581% |
| 2026-09-22 (이번) | **20,406** | **0.9581%** |

과거 실행은 **다른 Phase 1 basis**(`phase1_20260506_014300`)를 썼는데도 결과가 같다.
`detect_warp_rotated.py` 포팅이 정확하다는 증거이자, **회전이 행 선택에 영향이 없다**는
§D-2 의 Jaccard 0.998 을 독립적으로 재확인해 준다.

### ⚠️ 핵심 발견 — 회전 검출이 원공간과 거의 같은 뉴런을 고른다 (행 버전)

2026-05 원공간 결과(`safety_neuron_accelerated_20260502_013602.txt`, 같은 모델·같은
1200/200·같은 4994 프롬프트)와 **집합 단위로** 비교:

| module | 원공간 | 회전 | 교집합 | Jaccard |
|---|---:|---:|---:|---:|
| ffn_up | 3,362 | 3,364 | 3,358 | 0.997 |
| ffn_down | 3,362 | 3,364 | 3,358 | 0.997 |
| q | 2,874 | 2,876 | 2,873 | 0.999 |
| k | 2,874 | 2,876 | 2,873 | 0.999 |
| v | 526 | 525 | 524 | 0.994 |
| **TOTAL** | **12,998** | **13,005** | **12,986** | **0.998** |

**이유**: `Σ_t |x U wⱼ|` 는 `‖wⱼ‖` 와 입력 크기에 지배되는데 `U` 가 정규직교라 `‖x‖` 를
보존한다. 그래서 출력뉴런의 **순위**가 거의 안 바뀐다.

**함의**: 행 마스크는 이미 원공간 동결과 수학적 등가였고(§B-1), 검출 집합마저 0.998 이다.
따라서 **행 버전 ≈ (R)SN-Tune** 이고 WaRP basis 의 기여가 사실상 없다. 이 0.998 자체는
**논문에 쓸 수 있는 negative result** 다 — "회전 공간에서 같은 기준으로 고르면 원공간과
같은 것을 고른다, 그래서 마스크 **단위**를 바꿔야 한다"는 WSR 논지의 근거.
열 버전은 그 반대로 예산·잔존율이 모두 크게 달라 실제로 다른 방법임이 확인된다.

### ⚠️ 파라미터% 계산의 한계

`neuron_file.detected_parameter_count` 는 원본 `neuron_percentage_utils.py` 를 그대로
옮긴 것인데, docstring 이 명시하듯 **열(column) 의미를 가정**한다
(`ffn_up` 뉴런 하나 = `intermediate_size` 개 파라미터). 행 의미에서는 `ffn_up` 행 하나가
`hidden_size` 개이므로 행 버전의 파라미터%는 과대평가다.

**뉴런%는 양쪽 모두 정확**하고, 원공간 기록(0.9558%)과 이번 회전(0.9563%)이 맞는 것도
뉴런% 기준이다. 비교는 뉴런% 로 하고, 파라미터%는 열 버전에서만 곧이곧대로 읽을 것.
원본 저장소의 수치와 맞추기 위해 함수는 고치지 않았다.

---

## D-3. 평가 결과 (2026-09-22)

**측정 조건** — RESULTS.md(저장소 루트)의 다른 표와 동일:
HarmBench · AdvBench standard · **sys 모드**(llama-2 `<<SYS>>` 안전 프롬프트 포함) ·
`GRADING=hard`(refusal keyword) · 4공격 · seed 42 · conda env `harmbench`(vLLM 0.16) ·
lm-eval **GSM8K 5-shot flexible-extract**(conda `hb`).
AutoDAN/PAIR test case 는 family base(`llama2_7b`/`llama2_13b`, PAIR 는 `-instruct`) 재사용.

### 행(row) 버전 RSN-Tune

| 모델 | Direct | AutoDAN | PAIR | PAP | **AVG ↓** | **GSM8K ↑** |
|---|---:|---:|---:|---:|---:|---:|
| Llama-2-7B 행 RSN-Tune | 0.0000 | 0.0519 | 0.2885 | 0.2696 | **0.1525** | **0.3980** |
| Llama-2-13B 행 RSN-Tune | 0.0019 | 0.0288 | 0.3385 | 0.4723 | **0.2104** | **0.4845** |

로컬 경로: `outputs/wsr_rsn_row/llama2_{7b,13b}/rsn/gsm8k`
(업로드는 `huggingface.co` 차단 해제 후.)

### 논문과의 대조 — **같은 arm(RSN), 같은 출발점(plain chat)**

논문의 SN-Tune / RSN-Tune 도 **plain chat 에서 출발한다**(사용자 확인 2026-09-22). 따라서
출발점 교란 없이 직접 비교된다. 논문 표(`revisioning_wsr.tex:505-535`)의 해당 행:

| 행 | 7B AVG ↓ | 7B Acc ↑ | 13B AVG ↓ | 13B Acc ↑ |
|---|---:|---:|---:|---:|
| SN-Tune (논문) | 27.18 | 38.99 | 28.97 | 48.22 |
| RSN-Tune (논문) | **20.73** | **40.26** | **28.79** | **49.96** |
| **행 RSN-Tune (이번)** | **15.25** | **39.80** | **21.04** | **48.45** |
| WSR-Tune (논문) | 6.90 | 38.99 | 1.35 | 49.58 |

RSN 끼리 비교하면:

```
7B   AVG 20.73 → 15.25  (-5.48%p, 더 안전)   Acc 40.26 → 39.80  (-0.46%p)
13B  AVG 28.79 → 21.04  (-7.75%p, 더 안전)   Acc 49.96 → 48.45  (-1.51%p)
```

**어떻게 읽어야 하나.**
- 재현성 기준(CLAUDE.md): 동일 설정 단일 실행 간 keyword ASR 이 ±0.05 움직인다.
  7B 의 5.5%p 는 그 경계선, 13B 의 7.8%p 는 경계를 넘는다. 즉 **13B 차이는 실재할 수 있고
  7B 는 노이즈로 볼 여지가 있다.**
- 회전은 원인이 아니다 — 검출 Jaccard 0.998, 뉴런 수 11,297 vs 원공간 11,329(차이 0.3%),
  동결도 수학적 등가(§B-0·§B-1). **이번 행 버전은 RSN-Tune 의 재현**이지 새 방법이 아니다.
- 남은 차이의 후보: run-to-run 분산, critical 집합의 0.3% 차이, 측정 환경.
  논문 수치의 측정 조건(sys/GRADING)이 이 박스와 완전히 같은지는 확인되지 않았다.

**중요한 맥락**: 같은 표에서 **WSR-Tune 은 7B 6.90 / 13B 1.35** 다. RSN-Tune(20.73 / 28.79)과
격차가 크다. 즉 **행(뉴런) 단위 보호와 재파라미터화 공간 보호 사이의 간극이 이 표의 핵심**이고,
그 간극을 만드는 것이 바로 §B-0 에서 말한 **마스크 단위**다. 이번 행 버전이 RSN-Tune 을
재현한 것은 그 해석을 뒷받침한다 — **회전만으로는 WSR-Tune 의 이득이 나오지 않는다.**

⚠️ 앞선 초안에서 이번 결과를 논문의 **SN-Tune** 행과 비교했는데, 우리는 **RSN-Tune** 이므로
잘못된 대조였다. 위 표로 정정한다.

### 소요 시간 (B200 1장)

| 단계 | 7B | 13B |
|---|---|---|
| Phase 1 basis | 14분 | 약 25분 |
| 회전 검출 safety+utility+critical | 약 35분 | 약 40분 |
| RSN-Tune (safety 4994, 3ep) | 6분 20초 | 약 20분 |
| GSM8K FT (7473, 3ep) | 약 25분 | 약 50분 |
| HarmBench 4공격 | \~40분 | \~40분 |
| lm-eval GSM8K 5-shot | 1분 | 1분 |

---

## E. 남은 판단거리

1. **출발 모델이 Table 3 다른 행과 다르다** (plain chat vs SSFT). 각주로 갈지,
   SSFT 출발로 한 세트 더 돌릴지 결정 필요. `START_<model>` 한 줄로 바뀐다.
2. **행 마스크의 원공간 등가성** (B-1). 논문에서 "재파라미터화 공간에서 보호"를
   주장한다면 B-2 의 열 버전이 함께 있어야 방어가 된다.
3. **회전 공간 top-k 의 근거 부재** (B-3 주의). 검출 후 파라미터%를 보고 재조정할지 판단.
4. **`finetune_downstream_freeze_warp_sn.py` 는 한 번도 끝까지 안 돌아본 코드** (A-3).
   열 버전을 돌릴 때 여유를 둘 것.

---

# F. 2026-09-22 (밤) — 출처 확정과 열 버전 재구현

## F-1. 논문 Table 3 의 "SN-Tune + WSR-Tune" 은 **행 버전**이었다 (확정)

§B-0b 의 미확정 사항을 HarmBench 결과 트리로 닫았다. 7B 값 0 / 11.92 / 41.92 / 38.46 (AVG 23.07) 이
`~/HarmBench/results/evaluation_summary_2026-05-06_00-20-42.csv` 의
`llama2_7b-chat-gsm8k_ft_basis_rotation_sn-lr5e-5` 행과 소수점까지 같다. 5월 5~6일 로그 체인:

| 단계 | 스크립트 | 비고 |
|---|---|---|
| 검출 | `Safety-Neuron/.../safety_neuron_detection_v2_basis_rotation.py --use_basis_rotation_score` | 패치 transformers, `x U Wᵀ` 점수, **행** 인덱스, 1200/200, basis `phase1_20260505_164049` → 1.0778% |
| SN-Tune | `sn_tune.py` (원공간) | `..._only_sn_tuned_lr5e-5_basis_rotation` |
| GSM8K | `finetune_gsm8k_freeze_sn.py` (원공간, 행 동결) | `..._gsm8k_ft_freeze_basis_rotation_sn_lr5e-5` |

13B 도 같은 훅(1800/300, `..._gsm8k_ft_freeze_sn_rotation_space_lr5e-5`, AutoDAN 7.12 · PAP 58.38 일치).
**같은 라인의 RSN 모델도 이미 있다**: `kmseong/Llama-2-7b-chat-hf_gsm8k_ft_freeze_basis_rotation_rsn_lr5e-5`,
HarmBench 0.19 / 3.85 / 56.15 / 57.38 → **29.39** (RSN-Tune 20.73 보다 나빠 표에 실리지 않은 것으로 보인다).
§A-3 의 "WaRP-SN 계열 downstream 모델은 없다" 는 틀렸다 — `basis_rotation` 이름으로 셋이 있다.

행 버전 재실행 분산: 같은 방법이 5월 29.39, 9/22 15.25. 방법의 차이가 아니라 run 분산이다(§D-3).
**주의**: 오늘 포팅본의 7B 검출(13,005)은 5월 7B basis_rotation(14,657, Jaccard 0.87)과 다르고 원공간(12,998,
Jaccard 0.998)과 같다. 13B 는 5월과 동일(20,406). 5월 7B basis `phase1_20260505_164049` 의 제조법을 알 수 없어
차이의 원인은 확정 못 한다.

## F-2. 사용자 결정: RSN + WSR 은 **열 버전**으로 (2026-09-22)

행 버전은 (R)SN-Tune 과 제약이 같아(§B-0) "재파라미터화 공간에서 보호" 주장이 서지 않는다.
열 버전 = `basis_coeff` 의 열(입력 basis 방향)을 뉴런으로 정의 — ActSVD ablation arm C 와 같다.
**논문 행과 정의가 다른 새 실험**이므로 표에 실을 때 그 점을 밝혀야 한다.

## F-3. 구현 — 전용 트레이너를 버리고 `train.py --phase 3` 재사용

이전 세대 `finetune_downstream_freeze_warp_sn.py` 는 전 파라미터 학습 + 모듈마다 `basis_coeff@Uᵀ` 물질화로
**10.8 s/it, 178 GB OOM(7B)** 이었다. 새 경로:

```
sn_tune/warp_col_masks.py             열 뉴런 파일 → Phase 3 마스크 디렉토리 (tune / freeze 2종)
sn_tune/scripts/run_wsr_rsn_col_phase3.sh   검출→critical→마스크→[6] RSN-Tune→[7] GSM8K, .done 마커
sn_tune/scripts/eval_wsr_rsn_col.sh   models.yaml 등록 + HarmBench(sys·keyword) + GSM8K 5-shot
sn_tune/summarize_wsr_rsn_col.py      결과표 (outputs/wsr_rsn_col_p3/RESULTS.md)
```

| 단계 | 트레이너 | 마스크 | 데이터 | 학습 범위 |
|---|---|---|---|---|
| [6] RSN-Tune | Phase 3 기본(freeze) 변형 | `masks_tune` (critical 열만 mask=0) | circuit_breakers 4994, 3ep | 5개 projection 의 `basis_coeff` 중 critical 열만 (wd 0 강제) |
| [7] GSM8K | Phase 3 `--non_freeze` | `masks_freeze` (critical 열만 mask=1) | `gsm8k_train_task_7473.json`, 3ep | **전 파라미터** + mask=1 은 detach + `WaRPMaskRestoreCallback` |

[7] 의 플래그는 공개 WSR-Tune 셀과 동일하다 (`scripts/revision/12_wsr_tune.sh`, `run_all_phases_integrated.sh`
둘 다 `--non_freeze`; `outputs/revision/cb/qwen25_7b/gsm8k/wsr_tune/phase3_metadata.json` 의
`mode: non_freeze, trainable_params 7.6B`). lr 5e-5 · wd 0.01 · warmup 0.1 · cosine · max_len 1024 · seed 42 ·
유효배치 16 (7B 4×4, 13B 2×8) · gradient checkpointing. 출발 모델 plain chat.

**걸린 것 둘 (고침):**
- freeze 변형 + `--gradient_checkpointing` 은 step 0 에서 `element 0 of tensors does not require grad` 로 죽는다.
  임베딩이 얼어 있어 reentrant checkpoint 가 그래프를 끊는 것. `models/phase3_extra_learning.py` 에
  `enable_input_require_grads()` 추가 (수치 불변). 공개 모델은 전부 `--non_freeze` 라 이 조합을 쓴 적이 없었다.
- Phase 3 는 `--phase0_model_dir` **문자열**에 chat/instruct 가 있어야 chat template 을 쓴다. [7] 이 받는
  [6] 의 로컬 경로에 그게 없으면 조용히 plain 프롬프트가 된다 → 경로에 `rsn_tune_Llama-2-7b-chat-hf` 로 이름을 박았다.

## F-4. 열 공간에서는 utility 차집합이 다르게 작동한다 — 예산 보정

7B, utility 300/50 고정, 열 기준 파라미터%:

| safety top-k | safety 열 | critical 열 | critical 파라미터% |
|---|---:|---:|---:|
| 400/80 | 13,537 | **198** | 0.016% |
| 600/120 | 20,383 | 4,372 | 0.43% |
| 1200/200 | 39,680 | 23,569 | 2.43% |
| 800/135 | (검출 중) | | (≈1% 목표) |

원공간은 utility 300/50 이 safety 의 14%(1,826/12,998)였는데, 열 공간은 프롬프트 간 순위가 안정적이라
교집합 생존율이 높아 utility 가 16,111 로 **safety(400/80)보다 크다**. 그래서 §D-2 의 "400/80 → critical ≈ 0.8%"
외삽은 틀렸다(실제 0.016%). 예산 조정은 원저자 방식대로 **utility 는 300/50 고정, safety k 만 조정**한다.
주 설정은 이미 critical 이 있는 1200/200(7B, 2.43%) · 1800/300(13B, 3.55%), ≤1% arm 은 보정 후 추가.

## F-5. 부수 발견 — 저장소 루트 RESULTS.md 추가 실험 6 의 전제

추가 실험 6 은 "WSR-Tune 은 q,k,v,up,down 의 basis_coeff 만 학습한다" 는 전제로 AsFT·Lisa 의 학습 범위를
맞췄는데, 공개 WSR-Tune 은 전부 `--non_freeze`(전 파라미터 학습, 마스크 원소만 동결)다. 위 metadata 가 증거.
실험 6 의 `tgtonly` 비교는 WSR-Tune 과 범위가 맞지 않는다 — 오히려 실험 5(전체 학습)가 맞는 비교다. 별건이라
여기 기록만 한다.

## F-6. 부수 발견 — 공개 RSN 기준행의 GSM8K

09-22 재측정: 7B `llama2_7b_chat_gsm8k_ft_freeze_rsn_lr5e-5_new_revised` flexible **0.3336** / strict 0.4026
(논문 40.26 = strict 값), 13B `wvnvwn/llama-2-13b-chat-hf-gsm8k-rsn-tuned-lr5e-5` flexible **0.4594**
(논문 49.96 과 불일치). ASR 은 둘 다 논문과 일치(20.68 / 28.78). Table 3 의 RSN 기준행 downstream 을 어느 값으로
둘지 결정 필요.
