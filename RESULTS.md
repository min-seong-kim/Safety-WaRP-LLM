# Revision 실험 결과 — CB 축 / Llama-2-7B·13B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B

**자동 생성 파일입니다.** 손으로 고치지 마십시오 — 다시 만들려면:

```bash
python scripts/revision/gen_results_md.py --out RESULTS.md
```

## 측정 조건

| 항목 | 값 |
|---|---|
| 安全 평가 | HarmBench · Advbench_behaviors_standard · **sys 모드**(llama-2 `<<SYS>>` 안전 프롬프트 포함) |
| ASR 채점 | `GRADING=hard` (refusal keyword) — 표의 값은 ASR(keyword) |
| 공격 | DirectRequest · AutoDAN · PAIR · PAP (AVG = 4종 평균) |
| downstream | lm-evaluation-harness, 5-shot, `--apply_chat_template`, GSM8K=flexible-extract / MATH=`hendrycks_math_safe` |
| 학습 공통 | epochs 3 · 유효 batch 16 · max_len 1024 · seed 42 · bf16 · cosine · warmup 0.03 · wd 0.0 |
| LoRA 계열 | r=16 · α=32 · dropout 0.05 · targets `q,k,v,up,down` · lr 3e-4 (agnews 7e-5) |
| 출발 모델 | 각 base 의 CB(circuit_breakers) safety-tuned 체크포인트 |

**JB AVG 는 낮을수록 안전**하고, downstream 은 높을수록 좋습니다.

## Llama-2-7B-chat / GSM8K

| 기법 | GSM8K | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla LoRA | 0.3829 | — | — | — | — | — | [`llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_lr3e-4) |
| AsFT | 0.2153 | — | — | — | — | — | [`llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) |
| LISA (ρ=1.0) | 0.1683 | 0.0764 | 0 | 0.0077 | 0.1750 | 0.1227 | [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) |
| SEAL | 0.3813 | — | — | — | — | — | [`llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) |
| SafeLoRA | 0.3723 | — | — | — | — | — | [`llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) |
| WSR-LoRA (α=32) | 0.3685 | — | — | — | — | — | [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) |
| WSR-LoRA (α=16) | 0.3707 | 0.1277 | 0 | 0.0462 | 0.1808 | 0.2838 | [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) |

## Llama-2-13B-chat / GSM8K

| 기법 | GSM8K | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla LoRA | 0.4860 | 0.2361 | 0 | 0.0038 | 0.3385 | 0.6019 | [`llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4) |
| AsFT | 0.3124 | 0.0091 | 0 | 0 | 0.0269 | 0.0096 | [`llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) |
| LISA (ρ=0.0) | 0.4860 | 0.0176 | 0 | 0 | 0.0250 | 0.0454 | [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4) |
| LISA (ρ=1.0) | 0.3738 | 0.0086 | 0 | 0 | 0.0269 | 0.0077 | [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) |
| SEAL | 0.4519 | 0.1028 | 0 | 0.0096 | 0.1519 | 0.2496 | [`llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) |
| SafeLoRA | 0.4625 | 0.0494 | 0 | 0 | 0.1308 | 0.0669 | [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) |
| SaLoRA | 0.0720 | 0.4507 | 0.0346 | 0.1750 | 0.7212 | 0.8719 | [`llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) |
| WSR-LoRA (α=32) | 0.4822 | 0.1262 | 0 | 0.0019 | 0.1481 | 0.3546 | [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) |
| WSR-LoRA (α=16) | 0.4466 | 0.0791 | 0 | 0.0019 | 0.1115 | 0.2031 | [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) |

## Llama-3.2-3B-It / MATH

| 기법 | MATH | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla LoRA | 0.2364 | 0.0781 | 0 | 0 | 0.0558 | 0.2565 | [`llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4) |
| AsFT | 0.0714 | 0.0735 | 0 | 0 | 0.0596 | 0.2342 | [`llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4) |
| LISA (ρ=0.0) | 0.2392 | 0.0400 | 0 | 0 | 0.0404 | 0.1196 | [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4) |
| LISA (ρ=1.0) | 0.1012 | 0.0795 | 0 | 0 | 0.0654 | 0.2527 | [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4) |
| SEAL | 0.2162 | 0.0837 | 0 | 0 | 0.0731 | 0.2615 | [`llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5) |
| SafeLoRA | 0.2260 | 0.0691 | 0 | 0 | 0.0654 | 0.2108 | [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4) |
| SaLoRA | 0.2084 | 0.0964 | 0 | 0 | 0.0923 | 0.2935 | [`llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4) |
| WSR-LoRA (α=32) | 0.2248 | 0.0696 | 0 | 0 | 0.0596 | 0.2188 | [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4) |
| WSR-LoRA (α=16) | 0.2468 | 0.0687 | 0 | 0 | 0.0558 | 0.2188 | [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4) |

## Llama-3.1-8B-It / MATH

| 기법 | MATH | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla LoRA | 0.2456 | 0.0655 | 0 | 0 | 0.0135 | 0.2485 | [`llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4) |
| AsFT | 0.0686 | 0.0390 | 0 | 0 | 0.0038 | 0.1523 | [`llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4) |
| LISA (ρ=0.0) | 0.2254 | 0.0351 | 0 | 0 | 0 | 0.1404 | [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho0.0_lr3e-4) |
| LISA (ρ=1.0) | 0.1222 | 0.0387 | 0 | 0 | 0.0096 | 0.1454 | [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4) |
| SEAL | 0.1318 | 0.0443 | 0 | 0 | 0.0096 | 0.1677 | [`llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5) |
| SafeLoRA | 0.2300 | 0.0416 | 0 | 0 | 0.0077 | 0.1585 | [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4) |
| SaLoRA | 0.1938 | 0.0693 | 0 | 0 | 0.0173 | 0.2600 | [`llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4) |
| WSR-LoRA (α=32) | 0.1972 | 0.0513 | 0 | 0 | 0.0135 | 0.1915 | [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4) |
| WSR-LoRA (α=16) | 0.2240 | 0.0427 | 0 | 0 | 0.0077 | 0.1631 | [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4) |

## Qwen2.5-7B-It / GSM8K

| 기법 | GSM8K | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla LoRA | 0.7149 | 0.0383 | 0.0019 | 0 | 0.0462 | 0.1050 | [`qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4) |
| AsFT | 0.7377 | 0.0356 | 0 | 0 | 0.0231 | 0.1192 | [`qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) |
| LISA (ρ=0.0) | 0.7225 | 0.0489 | 0 | 0 | 0.0327 | 0.1627 | [`qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4) |
| LISA (ρ=1.0) | 0.7278 | 0.0397 | 0 | 0 | 0.0288 | 0.1300 | [`qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) |
| SEAL | 0.7005 | 0.0434 | 0 | 0 | 0.0385 | 0.1350 | [`qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) |
| SafeLoRA | 0.7483 | 0.0225 | 0 | 0 | 0.0173 | 0.0727 | [`qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) |
| SaLoRA | 0.6990 | 0.0480 | 0 | 0 | 0.0385 | 0.1535 | [`qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) |
| WSR-LoRA (α=32) | 0.7278 | 0.0343 | 0 | 0 | 0.0288 | 0.1085 | [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) |
| WSR-LoRA (α=16) | 0.7271 | 0.0319 | 0 | 0 | 0.0269 | 0.1008 | [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) |

## Gemma-2-9B-IT / GSM8K

| 기법 | GSM8K | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla LoRA | 0.6937 | 0.1022 | 0 | 0.0577 | 0.1058 | 0.2454 | [`gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4) |
| AsFT | 0.5997 | 0.0550 | 0 | 0 | 0.0058 | 0.2142 | [`gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) |
| LISA (ρ=0.0) | 0.6740 | 0.0680 | 0 | 0 | 0.0115 | 0.2604 | [`gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho0.0_lr3e-4) |
| LISA (ρ=1.0) | 0.5656 | 0.0344 | 0 | 0 | 0.0019 | 0.1358 | [`gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) |
| SEAL | 0.1865 | 0.5944 | 0.4500 | 0.2558 | 0.8365 | 0.8354 | [`gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) |
| SafeLoRA | 0.6694 | 0.0595 | 0 | 0.0038 | 0.0212 | 0.2131 | [`gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) |
| SaLoRA | 0.4792 | 0.2109 | 0 | 0.1327 | 0.1865 | 0.5242 | [`gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) |
| WSR-LoRA (α=32) | 0.6315 | 0.0735 | 0 | 0.0154 | 0.0327 | 0.2458 | [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) |
| WSR-LoRA (α=16) | 0.6884 | 0.0748 | 0 | 0.0615 | 0.0269 | 0.2108 | [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) |

---

수집된 ASR 레코드 61건 · downstream 레코드 449건.
빈칸(—)은 아직 측정되지 않았거나 실행이 실패한 항목입니다.

`⟲재학습` 표시는 **모델이 마지막 측정 이후 다시 학습되어 기존 수치를 버린** 항목입니다 (허브 `lastModified` 와 측정 로그 시각을 대조해 자동 판정). 재평가 대기 중입니다.

