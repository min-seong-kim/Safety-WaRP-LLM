# Revision 실험 — 생성될 Hugging Face 리포 전체 목록

> **자동 생성 파일이다. 직접 고치지 말 것.**
> `bash scripts/revision/gen_repo_list.sh > scripts/revision/REPO_LIST.md` 로 재생성한다.
> 리포명은 `scripts/revision/common.sh` 의 `hf_repo_id()` / `hf_ssft_repo_id()` 가 유일한
> 생성처다. 하이퍼파라미터를 바꾸면 이름도 바뀌므로 반드시 다시 생성할 것.

| | |
|---|---|
| 네임스페이스 | `kmseong` |
| **새로 만들 학습 셀** | **116개** |
| 논문/rebuttal 에 이미 있어 건너뛰는 셀 | 40개 |
| BT 안전정렬 출발모델 리포 | 0개 |
| **합계 (새로 생성)** | **116개** |

### 이번 범위

| | |
|---|---|
| CB(Circuit Breakers) 축 | 모델 6종 전부 |
| BT(BeaverTails) 축 | **`llama2_7b` 만**, 태스크 `gsm8k medqa arc agnews` 전부 · **12기법 전부 신규** |
| 재사용 (`SKIP_PUBLISHED=1`) | **논문 Table 2/4/10 의 full-param 5종만** — 출발모델·lr·epoch·배치가 이번 설정과 일치함을 확인한 것뿐이다 |
| 재사용하지 않는 것 | rebuttal 의 PEFT 수치. 출발모델 sha256 불일치(`wvnvwn/...ssft-cb` vs `kmseong/...Safety-FT-lr5e-5`), SafeLoRA thr 0.3 미실행(0.15/0.25/0.35 만), AGNEWS 는 epoch1·lr1e-5 동작점 |
| 알려진 잔여 차이 | 논문 MedQA 는 10000 샘플, 이번 신규 셀은 전체 10178 (1.7%). 재사용하는 MedQA 기준행 5개만 해당 |

---

## 명명 규칙

```
kmseong/{model}-{CB|BT}_SSFT-{method}_{task}[_{hparam}]_lr{lr}
```

| 필드 | 값 |
|---|---|
| `model` | `llama2_7b-chat` · `llama2_13b-chat` · `llama3_2_3b-instruct` · `llama3_1_8b-instruct` · `qwen2_5_7b-instruct` · `gemma2_9b-it` |
| `CB`/`BT` | 출발 모델을 안전정렬한 데이터셋 (Circuit Breakers / BeaverTails) |
| `method` | 12종. 기법명의 `_` 는 `-` 로 바꾼다 (`wsr_tune` → `wsr-tune`) — 필드 구분자 `_` 와 충돌하므로 |
| `task` | `gsm8k` `math` `medqa` `arc` `agnews` |
| `hparam` | 기법 고유 값. **없는 기법(Full FT / Vanilla LoRA)은 슬롯을 생략한다** |
| `lr` | 기법에서 자동 도출. full-param 계열 `5e-5`, LoRA 계열 `3e-4` (AG News 만 `7e-5`) |

### 기법별 하이퍼파라미터 태그

| 기법 | 태그 | 값의 근거 |
|---|---|---|
| Full FT | `—` | 하이퍼파라미터 없음 → 슬롯 생략 |
| SafeInstr | `mix0.1` | 논문 §4.1 (downstream 학습셋의 10%) |
| RESTA | `gamma0.3` | 논문 §4.1 |
| SafeDelta | `s0.1` | 논문 §4.1 |
| WSR-Tune | `rho0.1` | 논문 기본 freeze ratio ρ |
| Vanilla LoRA | `—` | 하이퍼파라미터 없음 → 슬롯 생략 |
| AsFT | `lambda1.0` | 사용자 지정 λ |
| LISA | `rho1.0` | 사용자 지정 ρ |
| SEAL | `topp0.8` | 기존 설정 top-p |
| SafeLoRA | `thr0.3` | 사용자 지정 threshold |
| SaLoRA | `rs32rt32` | budget-matched 설정 |
| WSR-LoRA | `rho0.1` | 논문 기본 freeze ratio ρ |

> LoRA 계열 6종은 전부 `r=16 / alpha=32 / dropout=0.05 / targets {q,k,v,up,down}` 로 동일해
> 이름에 넣지 않는다. full-param 계열은 전부 `epochs 3 / effective batch 16 / max_len 1024 /
> seed 42 / bf16`.

---

## 1. 안전정렬 출발 모델 (입력)

이 실험의 모든 셀은 **이미 안전정렬된 모델**에서 출발한다. CB 축 6종과 BT 축 llama2-7b 는
기존 것을 재사용하고, 나머지 BT 5종을 새로 학습해 올린다.

| 모델 키 | base (재사용) | CB 출발모델 (재사용) | BT 출발모델 |
|---|---|---|---|
| `llama2_7b` | `meta-llama/Llama-2-7b-chat-hf` | `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` | `wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv` (재사용) |
| `llama2_13b` | `meta-llama/Llama-2-13b-chat-hf` | `wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5` | — (BT 축 제외) |
| `llama32_3b` | `meta-llama/Llama-3.2-3B-Instruct` | `kmseong/llama3_2_3b-instruct-SSFT-lr5e-5` | — (BT 축 제외) |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | `kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5` | — (BT 축 제외) |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | `wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5` | — (BT 축 제외) |
| `gemma2_9b` | `google/gemma-2-9b-it` | `wvnvwn/gemma-2-9b-it-ssft-lr3e-5` | — (BT 축 제외) |

> `base` 는 SafeLoRA / AsFT / RESTA 가 `V = W_aligned − W_base` 를 만들 때 필요하다
> (gated repo 라 HF 토큰 필요). 새로 만드는 리포는 **BT 출발모델 5종뿐**이다.

---

## 2. 학습 셀 리포

모델별로 나눈다. 각 표의 행은 12개 기법, 열은 안전 데이터 축이다.

### `llama2_7b`  (`llama2_7b-chat`) — 신규 81개

#### CB · GSM8K  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-CB_SSFT-fullft_gsm8k_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `llama2_7b-chat-CB_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `llama2_7b-chat-CB_SSFT-resta_gsm8k_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `llama2_7b-chat-CB_SSFT-safedelta_gsm8k_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `llama2_7b-chat-CB_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4` | **신규** |
| AsFT | `llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4` | **신규** |

#### CB · MedQA  — Figure 4 확장

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-CB_SSFT-fullft_medqa_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `llama2_7b-chat-CB_SSFT-safeinstr_medqa_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `llama2_7b-chat-CB_SSFT-resta_medqa_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `llama2_7b-chat-CB_SSFT-safedelta_medqa_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `llama2_7b-chat-CB_SSFT-wsr-tune_medqa_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4` | **신규** |
| AsFT | `llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.1_lr3e-4` | **신규** |

#### CB · ARC-C  — Figure 4 확장

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-CB_SSFT-fullft_arc_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `llama2_7b-chat-CB_SSFT-safeinstr_arc_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `llama2_7b-chat-CB_SSFT-resta_arc_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `llama2_7b-chat-CB_SSFT-safedelta_arc_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `llama2_7b-chat-CB_SSFT-wsr-tune_arc_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4` | **신규** |
| AsFT | `llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.1_lr3e-4` | **신규** |

#### CB · AG News  — Figure 4 확장

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-CB_SSFT-fullft_agnews_lr5e-5` | **신규** |
| SafeInstr | `llama2_7b-chat-CB_SSFT-safeinstr_agnews_mix0.1_lr5e-5` | **신규** |
| RESTA | `llama2_7b-chat-CB_SSFT-resta_agnews_gamma0.3_lr5e-5` | **신규** |
| SafeDelta | `llama2_7b-chat-CB_SSFT-safedelta_agnews_s0.1_lr5e-5` | **신규** |
| WSR-Tune | `llama2_7b-chat-CB_SSFT-wsr-tune_agnews_rho0.1_lr5e-5` | **신규** |
| Vanilla LoRA | `llama2_7b-chat-CB_SSFT-lora_agnews_lr7e-5` | **신규** |
| AsFT | `llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5` | **신규** |
| LISA | `llama2_7b-chat-CB_SSFT-lisa_agnews_rho1.0_lr7e-5` | **신규** |
| SEAL | `llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-CB_SSFT-safelora_agnews_thr0.3_lr7e-5` | **신규** |
| SaLoRA | `llama2_7b-chat-CB_SSFT-salora_agnews_rs32rt32_lr7e-5` | **신규** |
| WSR-LoRA | `llama2_7b-chat-CB_SSFT-wsr-lora_agnews_rho0.1_lr7e-5` | **신규** |

#### BT · GSM8K  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-BT_SSFT-fullft_gsm8k_lr5e-5` | **신규** |
| SafeInstr | `llama2_7b-chat-BT_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5` | **신규** |
| RESTA | `llama2_7b-chat-BT_SSFT-resta_gsm8k_gamma0.3_lr5e-5` | **신규** |
| SafeDelta | `llama2_7b-chat-BT_SSFT-safedelta_gsm8k_s0.1_lr5e-5` | **신규** |
| WSR-Tune | `llama2_7b-chat-BT_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5` | **신규** |
| Vanilla LoRA | `llama2_7b-chat-BT_SSFT-lora_gsm8k_lr3e-4` | **신규** |
| AsFT | `llama2_7b-chat-BT_SSFT-asft_gsm8k_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_7b-chat-BT_SSFT-lisa_gsm8k_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_7b-chat-BT_SSFT-seal_gsm8k_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-BT_SSFT-safelora_gsm8k_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_7b-chat-BT_SSFT-salora_gsm8k_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_7b-chat-BT_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4` | **신규** |

#### BT · MedQA  — Figure 4 확장

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-BT_SSFT-fullft_medqa_lr5e-5` | **신규** |
| SafeInstr | `llama2_7b-chat-BT_SSFT-safeinstr_medqa_mix0.1_lr5e-5` | **신규** |
| RESTA | `llama2_7b-chat-BT_SSFT-resta_medqa_gamma0.3_lr5e-5` | **신규** |
| SafeDelta | `llama2_7b-chat-BT_SSFT-safedelta_medqa_s0.1_lr5e-5` | **신규** |
| WSR-Tune | `llama2_7b-chat-BT_SSFT-wsr-tune_medqa_rho0.1_lr5e-5` | **신규** |
| Vanilla LoRA | `llama2_7b-chat-BT_SSFT-lora_medqa_lr3e-4` | **신규** |
| AsFT | `llama2_7b-chat-BT_SSFT-asft_medqa_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_7b-chat-BT_SSFT-lisa_medqa_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_7b-chat-BT_SSFT-seal_medqa_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-BT_SSFT-safelora_medqa_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_7b-chat-BT_SSFT-salora_medqa_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_7b-chat-BT_SSFT-wsr-lora_medqa_rho0.1_lr3e-4` | **신규** |

#### BT · ARC-C  — Figure 4 확장

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-BT_SSFT-fullft_arc_lr5e-5` | **신규** |
| SafeInstr | `llama2_7b-chat-BT_SSFT-safeinstr_arc_mix0.1_lr5e-5` | **신규** |
| RESTA | `llama2_7b-chat-BT_SSFT-resta_arc_gamma0.3_lr5e-5` | **신규** |
| SafeDelta | `llama2_7b-chat-BT_SSFT-safedelta_arc_s0.1_lr5e-5` | **신규** |
| WSR-Tune | `llama2_7b-chat-BT_SSFT-wsr-tune_arc_rho0.1_lr5e-5` | **신규** |
| Vanilla LoRA | `llama2_7b-chat-BT_SSFT-lora_arc_lr3e-4` | **신규** |
| AsFT | `llama2_7b-chat-BT_SSFT-asft_arc_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_7b-chat-BT_SSFT-lisa_arc_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_7b-chat-BT_SSFT-seal_arc_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-BT_SSFT-safelora_arc_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_7b-chat-BT_SSFT-salora_arc_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_7b-chat-BT_SSFT-wsr-lora_arc_rho0.1_lr3e-4` | **신규** |

#### BT · AG News  — Figure 4 확장

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_7b-chat-BT_SSFT-fullft_agnews_lr5e-5` | **신규** |
| SafeInstr | `llama2_7b-chat-BT_SSFT-safeinstr_agnews_mix0.1_lr5e-5` | **신규** |
| RESTA | `llama2_7b-chat-BT_SSFT-resta_agnews_gamma0.3_lr5e-5` | **신규** |
| SafeDelta | `llama2_7b-chat-BT_SSFT-safedelta_agnews_s0.1_lr5e-5` | **신규** |
| WSR-Tune | `llama2_7b-chat-BT_SSFT-wsr-tune_agnews_rho0.1_lr5e-5` | **신규** |
| Vanilla LoRA | `llama2_7b-chat-BT_SSFT-lora_agnews_lr7e-5` | **신규** |
| AsFT | `llama2_7b-chat-BT_SSFT-asft_agnews_lambda1.0_lr7e-5` | **신규** |
| LISA | `llama2_7b-chat-BT_SSFT-lisa_agnews_rho1.0_lr7e-5` | **신규** |
| SEAL | `llama2_7b-chat-BT_SSFT-seal_agnews_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_7b-chat-BT_SSFT-safelora_agnews_thr0.3_lr7e-5` | **신규** |
| SaLoRA | `llama2_7b-chat-BT_SSFT-salora_agnews_rs32rt32_lr7e-5` | **신규** |
| WSR-LoRA | `llama2_7b-chat-BT_SSFT-wsr-lora_agnews_rho0.1_lr7e-5` | **신규** |

### `llama2_13b`  (`llama2_13b-chat`) — 신규 7개

#### CB · GSM8K  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama2_13b-chat-CB_SSFT-fullft_gsm8k_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `llama2_13b-chat-CB_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `llama2_13b-chat-CB_SSFT-resta_gsm8k_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `llama2_13b-chat-CB_SSFT-safedelta_gsm8k_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `llama2_13b-chat-CB_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4` | **신규** |
| AsFT | `llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4` | **신규** |

### `llama32_3b`  (`llama3_2_3b-instruct`) — 신규 7개

#### CB · MATH  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama3_2_3b-instruct-CB_SSFT-fullft_math_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `llama3_2_3b-instruct-CB_SSFT-safeinstr_math_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `llama3_2_3b-instruct-CB_SSFT-resta_math_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `llama3_2_3b-instruct-CB_SSFT-safedelta_math_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `llama3_2_3b-instruct-CB_SSFT-wsr-tune_math_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4` | **신규** |
| AsFT | `llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4` | **신규** |

### `llama31_8b`  (`llama3_1_8b-instruct`) — 신규 7개

#### CB · MATH  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `llama3_1_8b-instruct-CB_SSFT-fullft_math_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `llama3_1_8b-instruct-CB_SSFT-safeinstr_math_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `llama3_1_8b-instruct-CB_SSFT-resta_math_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `llama3_1_8b-instruct-CB_SSFT-safedelta_math_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `llama3_1_8b-instruct-CB_SSFT-wsr-tune_math_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4` | **신규** |
| AsFT | `llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4` | **신규** |
| LISA | `llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4` | **신규** |
| SEAL | `llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4` | **신규** |

### `qwen25_7b`  (`qwen2_5_7b-instruct`) — 신규 7개

#### CB · GSM8K  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `qwen2_5_7b-instruct-CB_SSFT-fullft_gsm8k_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `qwen2_5_7b-instruct-CB_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `qwen2_5_7b-instruct-CB_SSFT-resta_gsm8k_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `qwen2_5_7b-instruct-CB_SSFT-safedelta_gsm8k_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `qwen2_5_7b-instruct-CB_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4` | **신규** |
| AsFT | `qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4` | **신규** |
| LISA | `qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4` | **신규** |
| SEAL | `qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4` | **신규** |

### `gemma2_9b`  (`gemma2_9b-it`) — 신규 7개

#### CB · GSM8K  — Table 2 / Table 4

| 기법 | 리포명 (네임스페이스 생략) | 상태 |
|---|---|---|
| Full FT | `gemma2_9b-it-CB_SSFT-fullft_gsm8k_lr5e-5` | 기존 (논문/rebuttal) |
| SafeInstr | `gemma2_9b-it-CB_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5` | 기존 (논문/rebuttal) |
| RESTA | `gemma2_9b-it-CB_SSFT-resta_gsm8k_gamma0.3_lr5e-5` | 기존 (논문/rebuttal) |
| SafeDelta | `gemma2_9b-it-CB_SSFT-safedelta_gsm8k_s0.1_lr5e-5` | 기존 (논문/rebuttal) |
| WSR-Tune | `gemma2_9b-it-CB_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5` | 기존 (논문/rebuttal) |
| Vanilla LoRA | `gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4` | **신규** |
| AsFT | `gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4` | **신규** |
| LISA | `gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4` | **신규** |
| SEAL | `gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5` | **신규** |
| SafeLoRA | `gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4` | **신규** |
| SaLoRA | `gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4` | **신규** |
| WSR-LoRA | `gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4` | **신규** |

> 표에는 네임스페이스를 뺀 리포명만 적었다. 실제 id 는 앞에 `NAMESPACE/` 가 붙는다.

---

## 3. 전체 목록 (평문)

**새로 만들 것만** 나열한다. 스크립트에서 그대로 쓰기 좋은 형태.

```
kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.1_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.1_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-fullft_agnews_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-safeinstr_agnews_mix0.1_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-resta_agnews_gamma0.3_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-safedelta_agnews_s0.1_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-wsr-tune_agnews_rho0.1_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-lora_agnews_lr7e-5
kmseong/llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5
kmseong/llama2_7b-chat-CB_SSFT-lisa_agnews_rho1.0_lr7e-5
kmseong/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-safelora_agnews_thr0.3_lr7e-5
kmseong/llama2_7b-chat-CB_SSFT-salora_agnews_rs32rt32_lr7e-5
kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_agnews_rho0.1_lr7e-5
kmseong/llama2_7b-chat-BT_SSFT-fullft_gsm8k_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-resta_gsm8k_gamma0.3_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safedelta_gsm8k_s0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-lora_gsm8k_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-asft_gsm8k_lambda1.0_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-lisa_gsm8k_rho1.0_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safelora_gsm8k_thr0.3_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-salora_gsm8k_rs32rt32_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-fullft_medqa_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safeinstr_medqa_mix0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-resta_medqa_gamma0.3_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safedelta_medqa_s0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_medqa_rho0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-lora_medqa_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-asft_medqa_lambda1.0_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-lisa_medqa_rho1.0_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-seal_medqa_topp0.8_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safelora_medqa_thr0.3_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-salora_medqa_rs32rt32_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_medqa_rho0.1_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-fullft_arc_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safeinstr_arc_mix0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-resta_arc_gamma0.3_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safedelta_arc_s0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_arc_rho0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-lora_arc_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-asft_arc_lambda1.0_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-lisa_arc_rho1.0_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-seal_arc_topp0.8_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safelora_arc_thr0.3_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-salora_arc_rs32rt32_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_arc_rho0.1_lr3e-4
kmseong/llama2_7b-chat-BT_SSFT-fullft_agnews_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safeinstr_agnews_mix0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-resta_agnews_gamma0.3_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safedelta_agnews_s0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_agnews_rho0.1_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-lora_agnews_lr7e-5
kmseong/llama2_7b-chat-BT_SSFT-asft_agnews_lambda1.0_lr7e-5
kmseong/llama2_7b-chat-BT_SSFT-lisa_agnews_rho1.0_lr7e-5
kmseong/llama2_7b-chat-BT_SSFT-seal_agnews_topp0.8_lr5e-5
kmseong/llama2_7b-chat-BT_SSFT-safelora_agnews_thr0.3_lr7e-5
kmseong/llama2_7b-chat-BT_SSFT-salora_agnews_rs32rt32_lr7e-5
kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_agnews_rho0.1_lr7e-5
kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4
kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4
kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4
kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4
kmseong/llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5
kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4
kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4
kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4
kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4
kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4
kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4
kmseong/llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5
kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4
kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4
kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4
kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4
kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4
kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4
kmseong/qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4
kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4
kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4
kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4
kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4
kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4
kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4
kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4
kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4
```

---

## 4. 검증

이 목록은 아래를 만족한다 (`gen_repo_list.sh` 가 매번 재확인).

- 총 **116개**, 고유 **116개** — 중복 0건
- 리포명 최장 **56자** (Hugging Face 한도: 네임스페이스 제외 96자)
- HF 허용문자 `[A-Za-z0-9._-]` 위반: **0건**

기존 리포와 덮어쓰기 충돌이 없는지는 네트워크가 필요해 별도로 확인한다:

```bash
python - <<'PY'
from huggingface_hub import HfApi
import pathlib
existing = {m.id for m in HfApi().list_models(author="kmseong")}
block = pathlib.Path("scripts/revision/REPO_LIST.md").read_text().split("## 3. 전체 목록 (평문)")[1].split("```")[1]
planned = [l.strip() for l in block.strip().splitlines() if l.strip()]
clash = sorted(set(planned) & existing)
print(f"계획 {len(planned)} · 기존 {len(existing)} · 충돌 {len(clash)}", clash[:5])
PY
```

생성 시각: 2026-08-26 21:43:33 KST
