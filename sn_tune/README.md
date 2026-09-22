# `sn_tune/` — SN-Tune · RSN-Tune 과 그 WaRP 공간 버전

논문 Table 3(`tab:baseline_plus_warp`, "Compatibility with existing fine-tuning methods")의
**(R)SN-Tune 행과 (R)SN-Tune + WSR-Tune 행**을 만들기 위한 코드. 네 갈래다:

| arm | 좌표계 | 무엇을 고르나 | 무엇을 학습/동결하나 |
|---|---|---|---|
| **SN-Tune** | 원공간 | `N_safe` | safety 뉴런 **만** 학습 |
| **RSN-Tune** | 원공간 | `N_robust = N_safe \ N_foundation` | critical 뉴런 **만** 학습 |
| **WSR-SN-Tune** | WaRP | `basis_coeff` 의 safety **열** | 그 열 **만** 학습 |
| **WSR-RSN-Tune** | WaRP | critical 열 | 그 열 **만** 학습 |

downstream(GSM8K) 단계에서는 **같은 뉴런/열을 반대로 얼린다** — 그것이 safety 를 보존하면서
태스크를 배우는 메커니즘이다.

> **2026-09-22:** 외부 `Safety-Neuron/neuron_detection` 의존을 없앴다. 검출·학습·패치·환경
> 구성이 전부 이 디렉토리 안에서 끝난다.

---

## 0. 먼저: 환경이 **둘**로 갈린다

이게 이 코드의 가장 큰 함정이다.

| 하는 일 | 환경 | transformers |
|---|---|---|
| 원공간 뉴런 검출 (`detect_original.py`) | **`hb_sn`** | **패치본** |
| 원공간 학습 (`sn_tune_original.py`, `finetune_freeze_sn.py`) | `hb` | 정품 |
| WaRP 공간 전부 (검출 포함) | `hb` | 정품 |

원공간 검출기는 forward 중 모듈에 스태시된 `_last_ffn_up_score` / `_last_q_score` … 를
읽는다. 정품 transformers 에는 그런 게 없다. 그리고 **검출 루프는 프롬프트마다
try/except 로 감싸여 있어서 100% 실패해도 종료 코드가 0 이고**, 모든 섹션이 빈
`{"0": [], "1": [], ...}` 인 파일을 남긴다. 그 빈 파일은 몇 시간 뒤 학습 단계에서
`ValueError: optimizer got an empty parameter list` 로야 드러난다.

반대로 **학습에는 패치가 들어가면 안 된다.** 그래서 환경을 복제해서 검출 쪽만 패치한다.

```bash
bash sn_tune/setup_hb_sn.sh          # hb --clone--> hb_sn, 패치 설치, 양쪽 검증
```

재실행 안전하다(이미 있으면 복제를 건너뛰고 패치/검증만 다시 한다). 원본 정품 파일은
`modeling_*.py.orig` 로 **최초 1회만** 백업한다. `hb` 는 건드리지 않는다.

언제든 따로 확인할 수 있다:

```bash
python -m sn_tune.verify_patch --model_name meta-llama/Llama-2-7b-chat-hf   # hb_sn: 패치 있어야
python -m sn_tune.verify_patch --model_family llama --expect absent          # hb:   정품이어야
```

패치본의 출처·sha256·알려진 버전 드리프트는 **`transformers_patch/PATCH_NOTES.md`** 에 있다.
transformers 를 올리거나 새 모델 계열을 추가하면 그 파일부터 읽어라.

---

## 1. 빠른 실행

```bash
# 원공간 SN + RSN (Llama-2-7B/13B × GSM8K)
bash sn_tune/scripts/run_sn_rsn_gsm8k.sh

# WaRP 공간 WSR-SN + WSR-RSN
bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

# 무엇이 돌지 먼저 보기
DRY_RUN=1 bash sn_tune/scripts/run_sn_rsn_gsm8k.sh

# 뉴런 파일까지만 만들고 멈추기 (top-k 를 맞출 때 쓴다)
STOP_AFTER_NEURONS=1 bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
```

둘 다 단계마다 `.done` 마커를 남기고 있으면 건너뛴다.
**마커를 지워서 재실행하지 마라** — 출력 경로에 하이퍼파라미터가 안 들어 있어서,
값을 바꿔 다시 돌리려면 `OUT_ROOT` 를 바꿔야 한다 (`OUT_ROOT=outputs/sn_tune_topk800`).

---

## 2. 파이프라인

### 원공간 (`run_sn_rsn_gsm8k.sh`)

```
[1] safety 검출   (hb_sn)  circuit_breakers 4994 ─┐
[2] utility 검출  (hb_sn)  wikipedia 1000        ─┤
[3] critical = [1] \ [2]                         ─┘ → N_safe, N_robust
[4] (R)SN-Tune    (hb)     해당 뉴런만 safety 데이터로 학습
[5] GSM8K FT      (hb)     해당 뉴런을 얼리고 나머지 학습
```

### WaRP 공간 (`run_wsr_sn_rsn_gsm8k.sh`)

```
[1] Phase 1 basis U   (train.py --phase 1)        ← BASIS_DIR_<model> 로 재사용 가능
[2] WaRP safety 검출   basis_coeff 의 safety 열
[3] WaRP utility 검출  wikipedia (JSON 으로 덤프해서 먹인다)
[4] critical = [2] \ [3]
[5] WSR-(R)SN-Tune     그 열만 safety 데이터로 학습
[6] GSM8K FT           그 열을 얼리고 나머지 열만 학습 → W = basis_coeff @ Uᵀ 복원
```

WaRP 쪽은 **패치가 필요 없다** — 검출기가 자기 forward hook 으로 점수를 직접 계산한다.

---

## 3. 뉴런 파일 포맷 — 줄 순서가 계약이다

5줄, 각 줄이 `{layer_idx: [indices]}`:

```
line 0: ffn_up   line 1: ffn_down   line 2: q   line 3: k   line 4: v
```

**파일에서 키를 읽지 않는다. 줄 위치로만 배정한다.** 코드상의 키도 `ffn_up, ffn_down, q, k, v`
이지 `attn_q/attn_k/attn_v` 가 아니다 (WaRP 쪽 `--layer_types` 철자와 혼동하지 말 것).
읽는 쪽이 모르는 키를 조용히 건너뛰므로, 키를 잘못 쓰면 에러가 아니라 **0개로 집계**된다.

원공간과 WaRP 공간이 **같은 포맷**을 쓴다. 다만 인덱스의 의미가 다르다 —
원공간은 가중치의 행(출력 뉴런), WaRP 는 `basis_coeff` 의 열(basis 방향).
**두 공간의 파일을 섞어서 critical 을 계산하지 마라.** 숫자만 줄어들고 아무 뜻이 없다.
`critical_neurons.py --space {original,warp}` 가 이를 메타 파일에 남긴다.

검사/요약:

```bash
python -m sn_tune.neuron_file <file>.txt --model_name <hf> --require_nonempty
python -m sn_tune.test_neuron_file        # 포맷·차집합 자체 점검 (GPU 불필요)
```

---

## 4. 동결 메커니즘 — 두 방향이 대칭이 아니다 (의도된 것)

| | `sn_tune_original.py` (학습) | `finetune_freeze_sn.py` (동결) |
|---|---|---|
| 기본값 | 전부 `requires_grad=False` | 전부 `requires_grad=True` |
| hook | safety 위치만 남기는 **keep-mask** | safety 위치 grad 를 **0** |
| restore 콜백 | **없음** | **있음** (`SafetyNeuronRestoreCallback`) |

동결 쪽에 restore 콜백이 필요한 이유: AdamW 의 decoupled weight decay(λθ)는 gradient 와
무관하게 파라미터를 끌어당긴다. grad 를 0 으로 만든 것만으로는 얼어붙지 않는다.

반대로 `sn_tune_original.py` 에는 그 콜백이 없어서, `requires_grad=True` 로 켜진 텐서 안의
non-safety 행이 weight decay 로 조금씩 움직인다. **원본 그대로 두었다** — 이미 공개된
7B/13B 모델과 제조 방식을 맞추기 위해서다.

---

## 5. 하이퍼파라미터

드라이버 기본값은 공개된 7B/13B RSN 모델의 `finetune_config.json` 과 **동일**하다:

```
lr 5e-5 · 3 epoch · batch 4 × grad_accum 4 (유효 16) · max_len 1024
wd 0.01 · warmup 0.1 · cosine · adamw_torch · bf16 · GSM8K 7473 샘플
```

이건 이 저장소의 WSR-Tune Phase 3 동작점과도 같아서, 같은 표에 놓고 비교할 수 있다.

### top-k — 실제로 쓴 값 (로그에서 확인, 2026-09-22)

**safety 와 utility 는 서로 다른 top-k 를 쓴다.** 공개된 모델을 만든 실행의 로그
(`Safety-Neuron/neuron_detection/logs/neuron_detection/`)에서 그대로 확인한 값이다:

| 모델 | 단계 | 코퍼스 | top-k (ffn/attn) | 검출 뉴런 | 뉴런% | 파라미터% |
|---|---|---|---:|---:|---:|---:|
| Llama-2-7B-chat | safety | circuit_breakers 4994 | **1200 / 200** | 12,998 | 0.956% | 1.135% |
| Llama-2-7B-chat | utility | wikipedia 1000 | **300 / 50** | 1,826 | 0.134% | 0.185% |
| Llama-2-7B-chat | critical | — | — | 11,329 | 0.833% | 0.967% |
| Llama-2-13B-chat | safety | circuit_breakers 4994 | 1200 / 200 | 15,828 | 0.743% | — |
| Llama-2-13B-chat | safety | circuit_breakers 4994 | **1800 / 300** | 20,406 | 0.958% | — |

13B 행이 둘인 건 **탐색을 실제로 한 흔적**이다 — 1200/200 은 0.74% 로 낮아서 버리고
1800/300 으로 다시 돌려 0.958%(7B 의 0.956% 와 나란함)를 맞췄다. 드라이버 기본값은
채택된 쪽이다.

**utility 는 300/50 이 집안 표준이었다** — 7B chat/base, Llama-3.1-8B(+Instruct),
Qwen2.5-32B 실행이 전부 300/50 이다. safety 만 모델 크기에 따라 키웠다.

**foundation(utility) 코퍼스는 Wikipedia 1000 문서다. Alpaca 가 아니다.**
`safety_neuron_detection_v2_revised.py --utility_neuron` 이 wikipedia 를 직접 로드했고
(`wikimedia/wikipedia`, `20231101.en`), 저장소 어디에도 alpaca 는 없다.
별도 파일 `foundation_neuron_detection.py` 도 wikipedia 를 쓰지만 알고리즘이 달라
(고정 top-k 가 아니라 `--ffn_active_fraction` 전역 비율) 실제로는 쓰이지 않았다.

### 새 모델의 top-k 맞추기

**검출 1회는 생각보다 싸다 — 7B/4994 프롬프트가 약 4분, wikipedia 1000 문서가 약 5분**
(로그 타임스탬프 기준). 그래서 2~3회 돌려 맞추는 것이 정상 워크플로다:

```bash
STOP_AFTER_NEURONS=1 SAFETY_TOP_FFN_<model>=1200 SAFETY_TOP_ATTN_<model>=200 \
    bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
# 로그의 'Detected safety PARAMETER percentage' 를 보고 top-k 를 비례 조정 → 재실행
```

⚠️ 검출기가 찍는 값이 **둘**이다. 논문의 "≤1%" 는 **파라미터** 기준이고, 뉴런 기준과
다르다(7B safety 는 뉴런 0.956% / 파라미터 1.135%). 파라미터 쪽을 보고 맞춰라.

### 출발 모델 — plain chat 모델이다

sn_tuning 로그 17건 전수: `meta-llama/Llama-2-7b-chat-hf` 12건,
`meta-llama/Llama-2-13b-chat-hf` 3건, `Llama-3.2-3B-Instruct` 2건. **전부 plain 이고
SSFT 모델은 하나도 없다.** 설계상 자연스럽다 — SN-Tune 자체가 safety 정렬 단계라
SSFT 를 대체한다. 드라이버 기본값을 여기에 맞췄다.

⚠️ **이 때문에 Table 3 의 다른 행과 출발점이 다르다.** SafeInstr / SEAL / WSR-Tune 은
`kmseong/llama2_7b-chat-Safety-FT-lr5e-5` 에서 출발한다. 방법론의 차이지 실수는 아니지만
표에 각주로 밝혀야 한다. 출발점을 통일하고 싶으면
`START_llama2_7b=kmseong/llama2_7b-chat-Safety-FT-lr5e-5` 로 덮어라.

---

## 5b. 과거 이력 — 예전에 어떻게 만들었나 (2026-05, 로그·허브에서 복원)

2026-05-03 ~ 05-06 사이에 **세 가지 다른 "회전 공간" 세대**가 있었다. 이름이 비슷해서
헷갈리기 쉬운데 서로 다른 방법이다.

| 세대 | 날짜 | 검출 대상 | SN-Tune 을 어디서 | 대표 리포 |
|---|---|---|---|---|
| **WaRP-SN-Tune** | 05-03 | `basis_coeff` 의 **열** | **WaRP 공간** | `kmseong/llama2_7b_chat_WaRP-SN-Tune_lr5e-5` ✅생존 |
| rotation_space | 05-05 | 회전된 *모델*의 행 → **원공간으로 역매핑** | 원공간 | `..._only_sn_tuned_lr5e-5_rotation_space` ❌삭제 |
| basis_rotation | 05-05 | 원본 모델 + basis-rotated 스코어링 훅 | 원공간 | `..._only_rsn_tuned_lr5e-5_basis_rotation` ✅생존 |

**이 패키지의 WSR-(R)SN-Tune 은 첫 번째(05-03)의 후계다.** 나머지 둘은 "회전 공간에서
점수만 매기고 학습은 원공간에서" 하는 방식이라 성격이 다르다.

살아남은 `llama2_7b_chat_WaRP-SN-Tune_lr5e-5` 의 `warp_sn_tune_config.json` 이 그대로 남아 있다:

```json
{"method": "WaRP-SN-Tune",
 "layer_types": ["ffn_up","ffn_down","attn_q","attn_k","attn_v"],
 "num_layers": 32, "learning_rate": 5e-05, "num_epochs": 3,
 "batch_size": 4, "grad_accum_steps": 4, "warmup_ratio": 0.1,
 "max_seq_len": 1024, "max_samples": 4994, "is_instruct": true,
 "note": "Safety neurons are COLUMN indices of basis_coeff (WaRP space)."}
```

이 값들은 이 저장소 드라이버의 기본값과 **전부 일치한다**. 다만 그 config 에는
`top_k_ffn`/`top_k_attn` 과 `basis_dir` 이 안 들어 있어 **검출 파라미터는 복원되지 않았다.**
`_6p` 변형(같은 날 84분 뒤 업로드)은 삭제되어 접미사 의미도 알 수 없다.

**두 가지가 이번에 처음 만들어진다:**
1. **WSR-RSN-Tune 은 존재한 적이 없다.** 허브 616개 리포 중 WaRP 계열에 RSN 변형이 없다.
   즉 WaRP 공간의 utility/critical 경로는 전부 신규다.
2. **WaRP-SN 계열의 downstream 모델도 없다.** 05-03 라인은 SN-Tune 에서 멈췄고,
   `finetune_downstream_freeze_warp_sn.py` 로 만든 업로드 모델이 하나도 없다.

⚠️ **그래서 WaRP 공간 utility top-k 기본값(500/88)은 근거가 없는 추정치다** — 원공간의
비율(1200:300)을 그대로 옮긴 것뿐이다. 반드시 `STOP_AFTER_NEURONS=1` 로 파라미터% 를
먼저 재고 맞춰라. 참고로 회전 공간은 원공간보다 **작은** k 가 필요했던 전례가 있다
(회전 모델에서 1200/200 → 3.69%, 600/100 → 1.69%, 400/80 → 1.12% 로 3회 탐색).

---

## 6. 함정 모음 (전부 실제로 당한 것들)

- **패치 없이 돌린 검출은 조용히 성공한다.** 빈 파일이 나오고 종료 코드는 0 이다.
  `detect_original.py` 는 이제 시작할 때 패치를, 끝날 때 결과가 비었는지를 검사한다.
  로그의 `Detection complete: success=N, failed=M` 이 권위 있는 지표다.
- **패치본은 CRLF 다** (gemma2·qwen2). `sed` 에서 `$` 앵커가 조용히 안 맞는다.
  설치는 반드시 `cp` 로 — 텍스트 변환을 태우지 마라.
- **패치는 설치된 것보다 오래된 transformers 를 겨냥한다.** 4.57.3 에서 qwen2 의
  `@check_model_inputs` 가 팩토리로 바뀌어 `forward` 가 통째로 가려지는 사고가 있었다.
  `verify_patch.py` 의 [3]번 검사가 이것을 잡는다.
- **저장 경로가 `--output_dir` 그대로가 아니다.** 원본 스크립트들이 `_lr<lr>_<ts>` 나
  `_<ts>` 를 뒤에 붙인다. `--no_timestamp_suffix` 로 끄거나 `--model_dir_file` 로
  최종 경로를 받아라 (드라이버는 후자를 쓴다).
- **`CUDA_VISIBLE_DEVICES` 를 코드에서 하드 지정하지 않는다.** 원본은 파일마다
  다른 값을 박아뒀고(검출 7, sn_tune 0, freeze 5), 일부는 셸 설정을 덮어썼다.
  전부 걷어냈다 — GPU 는 호출자가 정한다.
- **`is_instruct_model` 은 다섯 곳이 같은 규칙을 써야 한다.** `-it` 를 토큰 경계로
  보지 않으면 `gemma-2-9b-it` 가 arm 마다 다른 프롬프트로 학습된다(실제로 터졌다).
  `finetune_freeze_sn.py` 의 사본을 저장소 공통 규칙에 맞춰 두었다.
- **`chat_template.jinja` 는 별도 파일이다.** 업로드가 모델+토크나이저 객체만 밀면
  허브 사본이 `chat_template=None` 으로 로드되어 평가가 학습과 다른 프롬프트를 렌더한다.
  업로드는 폴더째 올리고, **올린 뒤 허브에서** chat_template 을 확인하라.
- **utility 코퍼스는 두 공간이 같은 문서를 봐야 한다.** `build_utility_corpus.py` 의
  seed(112)와 자르는 길이(2000자)는 `detect_original.load_wikipedia_data` 와 맞춰져 있다.
  여기가 어긋나면 두 공간의 critical 이 서로 다른 코퍼스에서 나온 것이 된다.
- **wikipedia 전체를 내려받으면 100GB 를 넘는다.** 기본은 streaming 이다
  (`WIKI_STREAMING=0` 으로 끌 수 있지만 디스크가 있어야 한다).

---

## 7. 파일 목록

```
detect_original.py        원공간 safety/utility 검출      ⚠️ 패치 필요
critical_neurons.py       critical = safety \ utility     (양쪽 공용)
sn_tune_original.py       (R)SN-Tune
finetune_freeze_sn.py     downstream FT + 뉴런 동결
neuron_file.py            5줄 포맷 I/O + 비율 계산        (양쪽 공용)
build_utility_corpus.py   wikipedia → JSON
test_neuron_file.py       자체 점검 (GPU 불필요)
verify_patch.py           패치 검사
setup_hb_sn.sh            hb → hb_sn 복제 + 패치
transformers_patch/       패치본 + PATCH_NOTES.md

warp_sn_detection.py                  WaRP 공간 검출
warp_sn_tune.py                       WaRP 공간 학습
run_warp_sn_pipeline.py               WaRP 검출+학습 파이프라인
finetune_downstream_freeze_warp_sn.py WaRP downstream FT + 열 동결

module.py / detect.py / run.py        별개 계열 (LinearSNWaRP, C = W @ U)

scripts/run_sn_rsn_gsm8k.sh           원공간 드라이버
scripts/run_wsr_sn_rsn_gsm8k.sh       WaRP 공간 드라이버
```
