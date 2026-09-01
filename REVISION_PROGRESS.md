# WSR-Tune revision — 실험 진행 현황

NeurIPS 2026 rebuttal/revision 용 확장 실험(제출 11600)의 **자동 생성** 현황표다.
손으로 적지 않는다 — `scripts/revision/common.sh` 의 레지스트리와 실제 허브/로컬 상태에서 뽑는다.

```
python scripts/revision/gen_progress_md.py --out REVISION_PROGRESS.md
```

## 요약

| 항목 | 셀 수 |
|---|---|
| 새로 만들 셀(전체) | **116** |
| └ CB 축 | 68 |
| └ BT 축 | 48 |
| ✅ 허브 업로드 완료 | **69** |
| 🟡 학습 완료·업로드 대기(로컬) | 0 |
| ⬜ 미실행 | 100 |
| ♻️ 논문 기존 결과 재사용(새로 안 만듦) | 40 |
| 🚫 라이선스 차단(gemma-2-9b-it) | 0 |

## 실험 설계 (모든 셀 공통)

- **출발 모델**: 각 base model 의 *기존 safety-tuned 체크포인트*. 새로 SSFT 하지 않는다.
- **안전 데이터 재사용 규칙**: 안전 데이터가 필요한 기법(SafeInstr/RESTA/SafeDelta/AsFT/LISA/SEAL/SafeLoRA/SaLoRA/WSR-*)은
  **출발 모델이 safety-tune 될 때 쓴 바로 그 데이터셋**(CB 축이면 `circuit_breakers`, BT 축이면 BeaverTails)을 쓴다.
- **프롬프트 동일성**: 한 task 는 task JSON **하나**를 12개 기법이 전부 같이 읽는다.
  `scripts/revision/verify_prompt_parity.py` 가 6개 토크나이즈 경로의 `(input_ids, labels)` 가 바이트 단위로
  같음을 확인한다 — 기법 간 차이가 프롬프트 차이에서 오지 않음을 보장한다.
- **공통**: epochs 3 · effective batch 16 · max_len 1024 · seed 42 · bf16 · cosine · max_grad_norm 1.0

| | full-param | LoRA 계열 |
|---|---|---|
| lr | 5e-5 | 3e-4 (gsm8k/math/medqa/arc) · 7e-5 (agnews) |
| weight decay | 0.01 | 0.0 |
| warmup ratio | 0.1 | 0.03 |
| rank / alpha / dropout | — | 16 / 32 / 0.05 |

기법별 하이퍼파라미터: AsFT λ=1.0 · LISA (rho 1.0, align_step 100, ft_step 900) ·
SafeLoRA threshold 0.3 · SaLoRA (r_s=32, r_t=32) · WSR-Tune/WSR-LoRA ρ(keep_ratio)=0.1 ·
RESTA γ=0.3 · SafeDelta s=0.1.

## 진행 표

범례: ✅ 허브 업로드 · 🟡 로컬 학습 완료(업로드 대기) · ⬜ 미실행 · ♻️ 논문 결과 재사용 · 🚫 라이선스 차단 · `·` 매트릭스 밖

### CB 축 (safety dataset = circuit_breakers)

**Llama-3.2-3B-It**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Llama-2-7B-chat**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| medqa | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| arc | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| agnews | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Qwen2.5-7B-It**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**gemma-2-9b-it**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Llama-3.1-8B-It**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Llama-2-13B-chat**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | ♻️ | ♻️ | ♻️ | ♻️ | ♻️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### BT 축 (safety dataset = BeaverTails)

**Llama-2-7B-chat**

| task | Full FT | SafeInstr | RESTA | SafeDelta | WSR-Tune | Vanilla LoRA | AsFT | LISA | SEAL | SafeLoRA | SaLoRA | WSR-LoRA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| medqa | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| arc | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| agnews | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

## 남은 셀

- cb/llama2_7b/gsm8k/lora → `kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4`
- cb/llama2_7b/gsm8k/asft → `kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`
- cb/llama2_7b/gsm8k/seal → `kmseong/llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`
- cb/llama2_7b/gsm8k/safelora → `kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`
- cb/llama2_7b/gsm8k/salora → `kmseong/llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`
- cb/llama2_7b/medqa/lora → `kmseong/llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4`
- cb/llama2_7b/medqa/asft → `kmseong/llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_lr3e-4`
- cb/llama2_7b/medqa/seal → `kmseong/llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5`
- cb/llama2_7b/medqa/safelora → `kmseong/llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_lr3e-4`
- cb/llama2_7b/medqa/salora → `kmseong/llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_lr3e-4`
- cb/llama2_7b/medqa/wsr_lora → `kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.1_lr3e-4`
- cb/llama2_7b/arc/lora → `kmseong/llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4`
- cb/llama2_7b/arc/asft → `kmseong/llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_lr3e-4`
- cb/llama2_7b/arc/seal → `kmseong/llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5`
- cb/llama2_7b/arc/safelora → `kmseong/llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_lr3e-4`
- cb/llama2_7b/arc/salora → `kmseong/llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_lr3e-4`
- cb/llama2_7b/arc/wsr_lora → `kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.1_lr3e-4`
- cb/llama2_7b/agnews/fullft → `kmseong/llama2_7b-chat-CB_SSFT-fullft_agnews_lr5e-5`
- cb/llama2_7b/agnews/safeinstr → `kmseong/llama2_7b-chat-CB_SSFT-safeinstr_agnews_mix0.1_lr5e-5`
- cb/llama2_7b/agnews/resta → `kmseong/llama2_7b-chat-CB_SSFT-resta_agnews_gamma0.3_lr5e-5`
- cb/llama2_7b/agnews/safedelta → `kmseong/llama2_7b-chat-CB_SSFT-safedelta_agnews_s0.1_lr5e-5`
- cb/llama2_7b/agnews/wsr_tune → `kmseong/llama2_7b-chat-CB_SSFT-wsr-tune_agnews_rho0.1_lr5e-5`
- cb/llama2_7b/agnews/lora → `kmseong/llama2_7b-chat-CB_SSFT-lora_agnews_lr7e-5`
- cb/llama2_7b/agnews/asft → `kmseong/llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5`
- cb/llama2_7b/agnews/seal → `kmseong/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5`
- cb/llama2_7b/agnews/safelora → `kmseong/llama2_7b-chat-CB_SSFT-safelora_agnews_thr0.3_lr7e-5`
- cb/llama2_7b/agnews/salora → `kmseong/llama2_7b-chat-CB_SSFT-salora_agnews_rs32rt32_lr7e-5`
- cb/llama2_7b/agnews/wsr_lora → `kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_agnews_rho0.1_lr7e-5`
- cb/llama2_13b/gsm8k/lora → `kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4`
- cb/llama2_13b/gsm8k/asft → `kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`
- cb/llama32_3b/math/lora → `kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4`
- cb/llama32_3b/math/asft → `kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`
- cb/llama32_3b/math/seal → `kmseong/llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`
- cb/llama32_3b/math/safelora → `kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`
- cb/llama32_3b/math/salora → `kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`
- cb/llama32_3b/math/wsr_lora → `kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`
- cb/llama31_8b/math/lora → `kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4`
- cb/llama31_8b/math/asft → `kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`
- cb/llama31_8b/math/seal → `kmseong/llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`
- cb/llama31_8b/math/safelora → `kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`
- cb/llama31_8b/math/salora → `kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`
- cb/llama31_8b/math/wsr_lora → `kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`
- cb/qwen25_7b/gsm8k/lora → `kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4`
- cb/qwen25_7b/gsm8k/asft → `kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`
- cb/qwen25_7b/gsm8k/seal → `kmseong/qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`
- cb/qwen25_7b/gsm8k/safelora → `kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`
- cb/qwen25_7b/gsm8k/salora → `kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`
- cb/qwen25_7b/gsm8k/wsr_lora → `kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`
- cb/gemma2_9b/gsm8k/lora → `kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4`
- cb/gemma2_9b/gsm8k/seal → `kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`
- cb/gemma2_9b/gsm8k/salora → `kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`
- cb/gemma2_9b/gsm8k/wsr_lora → `kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`
- bt/llama2_7b/gsm8k/fullft → `kmseong/llama2_7b-chat-BT_SSFT-fullft_gsm8k_lr5e-5`
- bt/llama2_7b/gsm8k/safeinstr → `kmseong/llama2_7b-chat-BT_SSFT-safeinstr_gsm8k_mix0.1_lr5e-5`
- bt/llama2_7b/gsm8k/resta → `kmseong/llama2_7b-chat-BT_SSFT-resta_gsm8k_gamma0.3_lr5e-5`
- bt/llama2_7b/gsm8k/safedelta → `kmseong/llama2_7b-chat-BT_SSFT-safedelta_gsm8k_s0.1_lr5e-5`
- bt/llama2_7b/gsm8k/wsr_tune → `kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5`
- bt/llama2_7b/gsm8k/lora → `kmseong/llama2_7b-chat-BT_SSFT-lora_gsm8k_lr3e-4`
- bt/llama2_7b/gsm8k/asft → `kmseong/llama2_7b-chat-BT_SSFT-asft_gsm8k_lambda1.0_lr3e-4`
- bt/llama2_7b/gsm8k/lisa → `kmseong/llama2_7b-chat-BT_SSFT-lisa_gsm8k_rho0.0_lr3e-4`
- bt/llama2_7b/gsm8k/seal → `kmseong/llama2_7b-chat-BT_SSFT-seal_gsm8k_topp0.8_lr5e-5`
- bt/llama2_7b/gsm8k/safelora → `kmseong/llama2_7b-chat-BT_SSFT-safelora_gsm8k_thr0.3_lr3e-4`
- bt/llama2_7b/gsm8k/salora → `kmseong/llama2_7b-chat-BT_SSFT-salora_gsm8k_rs32rt32_lr3e-4`
- bt/llama2_7b/gsm8k/wsr_lora → `kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`
- bt/llama2_7b/medqa/fullft → `kmseong/llama2_7b-chat-BT_SSFT-fullft_medqa_lr5e-5`
- bt/llama2_7b/medqa/safeinstr → `kmseong/llama2_7b-chat-BT_SSFT-safeinstr_medqa_mix0.1_lr5e-5`
- bt/llama2_7b/medqa/resta → `kmseong/llama2_7b-chat-BT_SSFT-resta_medqa_gamma0.3_lr5e-5`
- bt/llama2_7b/medqa/safedelta → `kmseong/llama2_7b-chat-BT_SSFT-safedelta_medqa_s0.1_lr5e-5`
- bt/llama2_7b/medqa/wsr_tune → `kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_medqa_rho0.1_lr5e-5`
- bt/llama2_7b/medqa/lora → `kmseong/llama2_7b-chat-BT_SSFT-lora_medqa_lr3e-4`
- bt/llama2_7b/medqa/asft → `kmseong/llama2_7b-chat-BT_SSFT-asft_medqa_lambda1.0_lr3e-4`
- bt/llama2_7b/medqa/lisa → `kmseong/llama2_7b-chat-BT_SSFT-lisa_medqa_rho0.0_lr3e-4`
- bt/llama2_7b/medqa/seal → `kmseong/llama2_7b-chat-BT_SSFT-seal_medqa_topp0.8_lr5e-5`
- bt/llama2_7b/medqa/safelora → `kmseong/llama2_7b-chat-BT_SSFT-safelora_medqa_thr0.3_lr3e-4`
- bt/llama2_7b/medqa/salora → `kmseong/llama2_7b-chat-BT_SSFT-salora_medqa_rs32rt32_lr3e-4`
- bt/llama2_7b/medqa/wsr_lora → `kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_medqa_rho0.1_lr3e-4`
- bt/llama2_7b/arc/fullft → `kmseong/llama2_7b-chat-BT_SSFT-fullft_arc_lr5e-5`
- bt/llama2_7b/arc/safeinstr → `kmseong/llama2_7b-chat-BT_SSFT-safeinstr_arc_mix0.1_lr5e-5`
- bt/llama2_7b/arc/resta → `kmseong/llama2_7b-chat-BT_SSFT-resta_arc_gamma0.3_lr5e-5`
- bt/llama2_7b/arc/safedelta → `kmseong/llama2_7b-chat-BT_SSFT-safedelta_arc_s0.1_lr5e-5`
- bt/llama2_7b/arc/wsr_tune → `kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_arc_rho0.1_lr5e-5`
- bt/llama2_7b/arc/lora → `kmseong/llama2_7b-chat-BT_SSFT-lora_arc_lr3e-4`
- bt/llama2_7b/arc/asft → `kmseong/llama2_7b-chat-BT_SSFT-asft_arc_lambda1.0_lr3e-4`
- bt/llama2_7b/arc/lisa → `kmseong/llama2_7b-chat-BT_SSFT-lisa_arc_rho0.0_lr3e-4`
- bt/llama2_7b/arc/seal → `kmseong/llama2_7b-chat-BT_SSFT-seal_arc_topp0.8_lr5e-5`
- bt/llama2_7b/arc/safelora → `kmseong/llama2_7b-chat-BT_SSFT-safelora_arc_thr0.3_lr3e-4`
- bt/llama2_7b/arc/salora → `kmseong/llama2_7b-chat-BT_SSFT-salora_arc_rs32rt32_lr3e-4`
- bt/llama2_7b/arc/wsr_lora → `kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_arc_rho0.1_lr3e-4`
- bt/llama2_7b/agnews/fullft → `kmseong/llama2_7b-chat-BT_SSFT-fullft_agnews_lr5e-5`
- bt/llama2_7b/agnews/safeinstr → `kmseong/llama2_7b-chat-BT_SSFT-safeinstr_agnews_mix0.1_lr5e-5`
- bt/llama2_7b/agnews/resta → `kmseong/llama2_7b-chat-BT_SSFT-resta_agnews_gamma0.3_lr5e-5`
- bt/llama2_7b/agnews/safedelta → `kmseong/llama2_7b-chat-BT_SSFT-safedelta_agnews_s0.1_lr5e-5`
- bt/llama2_7b/agnews/wsr_tune → `kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_agnews_rho0.1_lr5e-5`
- bt/llama2_7b/agnews/lora → `kmseong/llama2_7b-chat-BT_SSFT-lora_agnews_lr7e-5`
- bt/llama2_7b/agnews/asft → `kmseong/llama2_7b-chat-BT_SSFT-asft_agnews_lambda1.0_lr7e-5`
- bt/llama2_7b/agnews/lisa → `kmseong/llama2_7b-chat-BT_SSFT-lisa_agnews_rho0.0_lr7e-5`
- bt/llama2_7b/agnews/seal → `kmseong/llama2_7b-chat-BT_SSFT-seal_agnews_topp0.8_lr5e-5`
- bt/llama2_7b/agnews/safelora → `kmseong/llama2_7b-chat-BT_SSFT-safelora_agnews_thr0.3_lr7e-5`
- bt/llama2_7b/agnews/salora → `kmseong/llama2_7b-chat-BT_SSFT-salora_agnews_rs32rt32_lr7e-5`
- bt/llama2_7b/agnews/wsr_lora → `kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_agnews_rho0.1_lr7e-5`

## 지금 막혀 있는 것

**HF 공개 스토리지 쿼터 초과.** 업로드가 `403 Forbidden: You have exceeded your public storage space`
로 거부된다. 학습 자체는 정상이고, 검증에 실패한 셀은 로컬 가중치를 **지우지 않고 보존**한다
(`upload_and_prune.py` 는 4단계 검증을 통과한 셀만 지운다).

풀려면 셋 중 하나가 필요하다 — ① 계정에서 쓰지 않는 리포지토리 삭제, ② HF PRO 구독,
③ HF 에 공개 연구용 스토리지 상향 요청(`website@huggingface.co`).

## 다음 환경으로 넘길 때

이 저장소만 클론하면 이어서 돌아간다. 상태는 전부 파일에 있다:

- 완료 판정은 셀 디렉터리의 `.done` / `.uploaded` 마커다. `run_all.sh` 는 마커가 있는 셀을 건너뛴다.
- 이미 허브에 올라간 셀은 로컬에 가중치가 없어도 `.uploaded` 마커만으로 완료로 인정된다.
- 그러므로 **`outputs/revision/` 의 마커 파일을 지우지 말 것** — 지우면 끝난 셀을 처음부터 다시 돈다.

```bash
conda activate hb
setsid nohup bash scripts/revision/finish_cb.sh > /dev/null 2>&1 &   # CB 축 잔여
SAFETY_SETS=bt bash scripts/revision/run_all.sh                      # BT 축 (학습만 ~35h)
```

BT 축은 Llama-2-7B 한 모델에 4개 task × 12개 기법이고, 출발 모델
`wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv` 는 이미 허브에 있으므로 SSFT 를 다시 하지 않는다.

## 업로드된 모델 (허브)

- [`kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4)
- [`kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4)
- [`kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4)
- [`kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4)
- [`kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4)
- [`kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5)
- [`kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4)
- [`kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4)
- [`kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4)
- [`kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4)
- [`kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4)
- [`kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4)
- [`kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5)
- [`kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4)
- [`kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-fullft_agnews_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-fullft_agnews_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-lisa_agnews_rho0.0_lr7e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_agnews_rho0.0_lr7e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho0.0_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho0.0_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-lora_agnews_lr7e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_agnews_lr7e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-resta_agnews_gamma0.3_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-resta_agnews_gamma0.3_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-safedelta_agnews_s0.1_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safedelta_agnews_s0.1_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-safeinstr_agnews_mix0.1_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safeinstr_agnews_mix0.1_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-safelora_agnews_thr0.3_lr7e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_agnews_thr0.3_lr7e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-salora_agnews_rs32rt32_lr7e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_agnews_rs32rt32_lr7e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_agnews_rho0.1_lr7e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_agnews_rho0.1_lr7e-5)
- [`kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.1_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.1_lr3e-4)
- [`kmseong/llama2_7b-chat-CB_SSFT-wsr-tune_agnews_rho0.1_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-tune_agnews_rho0.1_lr5e-5)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5)
- [`kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5)
- [`kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5)
- [`kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4)

## 재현

```bash
setsid nohup bash scripts/revision/finish_cb.sh > /dev/null 2>&1 &   # CB 축
tail -f logs/revision_unattended/finish_cb_latest.log
```

업로드는 4단계로 검증한다(파일 존재 · 크기 일치 · 허브에서 `AutoConfig` 로드 · 허브 토크나이저의 `chat_template` 존재). 검증을 통과한 셀만 로컬 가중치를 지운다.

