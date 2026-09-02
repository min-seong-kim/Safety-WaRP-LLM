# 실험 기록 — 2026-08-31 ~ 09-02 (B200 단일 GPU 박스)

이 세션에서 **무엇을 어떻게 돌렸고, 어떤 함정에 걸렸는지**를 남긴다.
숫자 결과는 저장소 루트의 `RESULTS.md`(자동 생성)를 보라.

```bash
python scripts/revision/gen_results_md.py --out RESULTS.md   # 결과표 재생성
```

---

## 1. 이 박스의 환경 (다음 서버에서 재현할 것)

| | |
|---|---|
| GPU | NVIDIA B200 183GB ×1 (sm_100) |
| conda | `~/miniconda3` — 직접 설치 |
| 학습 env | `hb` — `environment_hb.yml` 로 생성. torch 2.10.0+cu128 / transformers 4.57.3 / vllm 0.17.1 |
| 안전평가 env | `harmbench` — vLLM 0.16.0 (HarmBench 전용) |
| 저장소 | `~/Safety-WaRP-LLM`, `~/HarmBench`, `~/lm-evaluation-harness` |

### 환경 구축에서 걸린 것

- **`environment_hb.yml` 의 `apex==0.9.10.dev0` 은 NVIDIA Apex 가 아니다.** PyPI 의 `apex` 는
  Pyramid 웹 인증 툴킷이고, 의존성 `cryptacular` 가 빌드 불가(enscons 가 `xxmodule.c` 를 못 찾음)라
  **pip 섹션 전체가 중단된다**(pip 은 설치 전에 전 패키지 메타데이터를 먼저 해석한다).
  `apex` + `cryptacular` 를 빼고 336/338 로 설치했다. 저장소 어디도 이 둘을 import 하지 않는다.
  같이 딸려온 `pyramid`/`velruse`/`zope-*` 도 미사용.
- `conda init` 은 `~/.bashrc` 만 건드린다. `sbatch`/ssh 같은 **login shell** 에서 conda 를 쓰려면
  `~/.bash_profile` 이 `~/.bashrc` 를 source 해야 한다(이 박스에는 없어서 추가했다).
- `scripts/revision/finish_cb.sh` 는 `$HOME/miniforge3` 를 하드코딩한다. 이 박스는 `miniconda3` 다.

---

## 2. 이번에 돌린 실험

### 2.1 CB 축 잔여 셀 + 라이선스 해제분

gemma-2-9b-it 게이트 라이선스가 승인되어 `asft`/`safelora` 2셀이 풀렸고,
llama2_13b 의 `lisa`/`seal`/`safelora`/`salora`/`wsr_lora` 를 채웠다.

### 2.2 LISA ρ ablation (9셀 × 2)

rebuttal 모델(`...-lisa-cb-r16a32-lr3e-4-ep3-rho0-alt`)의 `finetune_config.json` 을 확인한 결과
**ρ=0.0** 이었는데 revision 초기값은 1.0 이었다. ρ=1.0 은 GSM8K 를 0.39→0.17 로 무너뜨린다.
`common.sh` 의 기본값을 0.0 으로 바꾸고, 비교를 위해 ρ=1.0 도 전부 다시 만들었다.

* 대상 9셀: llama2_7b(gsm8k/medqa/arc/agnews) + llama2_13b, llama32_3b, llama31_8b, qwen25_7b, gemma2_9b
* **결론은 한쪽으로 일반화되지 않는다.** ρ=1.0 은 대체로 안전성↑·유용성↓ 이지만,
  llama32_3b·llama31_8b 에서는 **안전성마저 나빠진다**. 논문에 한쪽만 실으면 체리피킹이 된다.

### 2.3 WSR-LoRA α ablation (6모델)

`wsr_lora.py:83` 이 `scaling = alpha / rank` 다. revision 은 α=32(scaling 2.0),
rebuttal 의 `-rot` 계열은 α=16(scaling 1.0) 이었다. **표에서 두 세대를 비교하고 있었다.**

통제 실험으로 확정했다 — 같은 출발 모델·lr·데이터로 α 만 바꾼 2×2:

| 출발 모델 | α=16 | α=32 |
|---|---|---|
| `wvnvwn/...ssft-cb` | 0.1004 (`-rot`) | 0.2725 (`-new`) |
| `kmseong/...Safety-FT` | 0.1277 | 0.2295 |

α 가 지배 요인(1.8~2.7배), 출발 모델은 부차적. 6모델 확장 결과 **α=16 이 5/6 에서 안전성이
같거나 좋고 downstream 은 4/6 에서 오히려 높다.** baseline 이 위험한 모델일수록 효과가 크다.

> ⚠️ 다른 LoRA arm 은 전부 α=32 다. WSR-LoRA 만 α=16 으로 쓰면 **우리 방법만 업데이트 예산이
> 절반**이 되어 공정성 공격을 받는다(rebuttal 표가 실제로 그 구성이었다). ablation 으로 함께
> 싣고 각주로 밝히는 편이 방어 가능하다.

### 2.4 실패 셀 재학습

`llama2_13b/salora`, `gemma2_9b/seal` 을 재학습했으나 **둘 다 재현**됐다(seed 고정).

* 13B SaLoRA: 학습 로그 정상(loss 0.17 수렴, merge/저장 성공), 허브 산출물 구조도 정상인데
  GSM8K 0.072 / JB 0.451. 3B·8B·Qwen·Gemma 는 같은 설정에서 멀쩡하다 → **13B 고유 실패 모드**.
* Gemma SEAL: GSM8K 0.187 / JB 0.594, Direct ASR 0.45(공격 없이도 절반이 응답).

둘 다 현재 수치로는 논문 표에 싣기 어렵다. 하이퍼파라미터를 바꿔(13B SaLoRA `r_s`, Gemma SEAL
`topp`) 재시도하거나, 실패로 명시하고 각주 처리해야 한다.

---

## 3. 평가 방법

```bash
cd ~/HarmBench
./run_all_eval.sh <repo1> <repo2> ...      # ASR + downstream 한 번에
```

* `HB_VARIANTS` 기본값은 **`sys`**(코드 229행 `${HB_VARIANTS:-sys}`; 주석은 낡아서 nosys 라고 쓰여 있다).
  기존 표가 전부 sys 이므로 그대로 쓰면 된다.
* **sys 와 nosys 를 섞으면 안 된다.** 같은 모델이 1.3~2.1배 차이난다(실측:
  `cbwsrlora-rot` sys 0.0999 / nosys 0.1759).
* task 는 모델명에서 자동 추론된다(`_math_` → `hendrycks_math_safe`, `_gsm8k_` → gsm8k). `DRY_RUN=1` 로 확인 가능.
* `run_all_eval.sh` 는 **스테이지 단위**(HarmBench 전체 → lm-eval 전체)라 모델을 많이 주면
  캐시가 폭발한다(36개 = 500GB+). **3개씩 배치로 끊고 배치마다 캐시를 회수**하라
  (`supervise_eval.sh` 가 그렇게 한다).

---

## 4. 걸렸던 함정 (재발 방지)

### 4.1 `/tmp` 가 noexec 이라 Triton JIT 이 죽는다 ★

```
ImportError: .../__triton_launcher...so: failed to map segment from shared object
```

이 박스의 `/tmp` 는 `tmpfs + noexec`. Triton 은 JIT 으로 `.so` 를 만들어 로드하므로
noexec 파일시스템에서는 **실행 세그먼트를 mmap 할 수 없다**. 캐시 손상이 아니라 위치 문제라
지워도 같은 곳에 다시 만들어져 똑같이 실패한다.

* `harmbench_eval.sh` 는 이미 `$HOME/.triton/cache` 로 우회하고 있었다.
* `lm-evaluation-harness/eval_models.sh` 는 해당 줄이 **주석 처리**돼 있어 lm-eval 만 죽었다 → 주석 해제.
* 재현: `gcc -shared -o /tmp/x.so x.c && python -c "ctypes.CDLL('/tmp/x.so')"` → 동일 오류.

### 4.2 Gemma-2 는 `block_size: 32` 가 필요하다 ★

head_size 256 + FlashInfer block_size 16 조합에 vLLM 버그가 있어 엔진 초기화가 assertion 으로 죽는다.

* HarmBench: `configs/model_configs/models.yaml` 의 gemma 엔트리에 `block_size: 32`.
  자동 등록기(`add_models_to_yaml.py` 의 `build_block()`)가 gemma 를 특별 취급하지 않아
  새로 등록된 셀들에 빠져 있었다 → 패치했다.
* lm-eval: `eval_models.sh` 의 `MODEL_ARGS` 에 `block_size=32`.
  `attention_backend=FLASH_ATTN` 을 줘도 vLLM 0.17 에서는 FlashInfer 가 선택되므로 그 방법으로는 안 된다.

### 4.3 `pgrep -f` 자기 매칭으로 무인 실행이 5시간 멈췄다 ★

감시 스크립트가 `pgrep -f "finish_all_before_eval.sh"` 로 선행 작업 종료를 판단했는데,
**감시용으로 띄운 다른 프로세스의 명령줄에 그 문자열이 들어 있어** pgrep 이 그것을 잡았다.
학습은 04:11 에 끝났는데 평가는 09:24 에야 시작됐다.

→ 프로세스 이름 매칭 대신 **완료 마커 파일**이나 **PID 파일**(`kill -0 $(cat /tmp/orch.pid)`)로 판단하라.

### 4.4 `${VAR:-default}` 는 빈 문자열도 기본값으로 되돌린다

`BASE_BLOCKED_MODELS="" bash run_all.sh` 로 gemma 를 푸는 방법이 문서에 있었지만
`${BASE_BLOCKED_MODELS:-gemma2_9b}` 라 **동작하지 않았다**(PLAN_ONLY 로 0셀이 나왔다).
`:-` → `-` 로 고쳤다.

### 4.5 `out_dir` 이 하이퍼파라미터를 구분하지 않는다

`outputs/revision/<safety>/<model>/<task>/<method>/` 는 ρ 나 α 를 포함하지 않는다(리포명만 구분).
같은 셀의 다른 하이퍼파라미터를 돌리면 **`.done` 마커 때문에 "이미 완료" 로 건너뛴다.**
→ `OUT_ROOT` 를 별도 디렉터리로 격리해서 돌렸다(`outputs/revision_lisa_rho1`, `outputs/revision_wsrlora_a16`).
마커를 지우는 방법은 기존 완료 기록을 잃으므로 쓰지 말 것.

### 4.6 `check_disk` 는 `OUT_ROOT` 가 없으면 여유 0GB 로 읽는다

`df` 가 실패해 빈 문자열 → 0. `disk_ok` 는 `run_cell` 안에서 `mkdir` 보다 **먼저** 호출되므로
새 박스에서는 **모든 셀이 조용히 `(disk)` 로 skip 된다.** 실행 전 `mkdir -p outputs/revision`.

### 4.7 `list_models()` 는 `lastModified` 를 채우지 않는다

전부 `None` 이라 신선도 검증이 조용히 무력화됐다(재학습된 모델의 옛 수치를 그대로 실을 뻔했다).
`expand=["lastModified"]` 를 명시할 것. `gen_results_md.py` 가 이 검증을 하고, 하나도 못 받으면 경고한다.

### 4.8 실행 중인 bash 스크립트를 편집하지 말 것

bash 는 스크립트를 바이트 오프셋으로 이어 읽는다. 실행 중 편집하면 엉뚱한 지점으로 점프한다.
순서를 바꿔야 하면 **편집 대신, 완료를 감지해 종료시키고 새 스크립트를 띄우는** 방식으로 하라.

---

## 5. 이번에 추가한 스크립트

| 파일 | 역할 |
|---|---|
| `gen_results_md.py` | 측정 로그 → `RESULTS.md` 자동 생성(허브 링크 + ASR + downstream). 재학습된 셀의 옛 수치를 `⟲재학습` 으로 걸러낸다 |
| `finish_gemma_13b.sh` | gemma asft/safelora + llama2_13b 5기법 |
| `rerun_lisa_rho0.sh` | LISA ρ=0.0 재학습 |
| `retrain_two_cells.sh` | 실패 셀(13B salora, gemma seal) 재학습 → 재평가 |
| `lisa_rho1_ablation.sh` | LISA ρ=1.0 9셀 (OUT_ROOT 격리) |
| `wsrlora_a16_ablation.sh` | WSR-LoRA α=16 5셀 (OUT_ROOT 격리) |
| `train_then_eval_all.sh` | 학습을 먼저 몰고 평가를 마지막에 일괄 — 모델 재다운로드를 줄인다 |
| `finish_all_before_eval.sh` | 평가 전 잔여 학습 일괄 |
| `~/HarmBench/supervise_eval.sh` | 평가 감시 — 허브 실존 확인 → 3개씩 배치 → 재시도 → 캐시 회수 |
| `~/HarmBench/gemma_ds_fill.sh` | gemma downstream 보충 |

---

## 6. 남은 일

* **Llama-2-7B ASR 5행** — `RESULTS.md` 에 빈칸. 기존 측정치가 있으면 채우면 된다.
* **13B SaLoRA / Gemma SEAL** — 재학습해도 실패 재현. 하이퍼파라미터 변경 재시도 또는 각주 처리.
* **BT 축 47셀** — 손대지 않았다. `SAFETY_SETS=bt bash scripts/revision/run_all.sh`.
  단 `common.sh:116` 의 `SAFEDELTA_DIR=/home/edgeai_lab/SafeDelta` 는 **외부 저장소 런타임 의존**이고
  이 박스에 없다. BT 축에는 safedelta 4셀이 있으므로 먼저 받아와야 한다.
