# Revision 실험 결과 — CB 축 / Llama-2-7B·13B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B

**2026-09-04 스냅샷.** 9/2–9/4 의 로컬 작업이 유실되어, 이 표는 사용자의 정리 스프레드시트를 기준으로 재구성했고 모든 수치를 `~/HarmBench/logs/run_all_*_summary.csv`(ASR) 와 `~/lm-evaluation-harness/logs/eval_*_results.csv`(downstream) 에 대조해 검증했다. Δ 열은 표 안의 값으로부터 다시 계산했다. `scripts/revision/gen_results_md.py` 는 아직 `/home/edgeai_lab/...` 경로를 읽으므로 이 박스에서는 재생성되지 않는다.

## 측정 조건

| 항목 | 값 |
|---|---|
| 安全 평가 | HarmBench · Advbench_behaviors_standard · **sys 모드**(llama-2 `<<SYS>>` 안전 프롬프트 포함) |
| ASR 채점 | `GRADING=hard` (refusal keyword) — 표의 값은 ASR(keyword) |
| 공격 | DirectRequest · AutoDAN · PAIR · PAP (AVG = 4종 평균) |
| downstream | lm-evaluation-harness, 5-shot, `--apply_chat_template`, GSM8K=flexible-extract / MATH=`hendrycks_math_safe` |
| 학습 공통 | epochs 3 · 유효 batch 16 · max_len 1024 · seed 42 · bf16 · cosine |
| full-param 계열 | lr 5e-5 (gemma 1e-5) · wd 0.01 · warmup 0.1 |
| LoRA 계열 | r=16 · **α=32 와 α=16 두 세대** · dropout 0.05 · targets `q,k,v,up,down` · lr 3e-4 |
| 출발 모델 | 각 base 의 CB(circuit_breakers) safety-tuned 체크포인트 |

**읽는 법.** AVG(= JB AVG, 4개 공격 ASR 평균) 는 낮을수록 안전, downstream 은 높을수록 좋다. Δsafe = AVG − 기준, Δdown = downstream − 기준, **Δoverall = Δdown − Δsafe** (클수록 좋음). 기준: full-param 행(SEAL, WSR-Tune)은 **Full FT**, LoRA 행은 **같은 α 의 Vanilla LoRA**. α=16 행은 9/2–9/3 에 새로 학습한 세대다(WSR-LoRA 만 α=16 이던 문제를 없애려 모든 LoRA arm 을 α=16 으로 재생성).

## Llama-2-7B-chat / GSM8K

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat_gsm8k_full_ft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5) | Full FT (SSFT+task) | 0.0000 | 0.0827 | 0.2635 | 0.4850 | 0.2078 | 0.4117 | — | — | — |
| [`llama2_7b_chat_seal_5e-5`](https://huggingface.co/kmseong/llama2_7b_chat_seal_5e-5) | SEAL † | 0.0019 | 0.1077 | 0.4481 | 0.6604 | 0.3045 | 0.3889 | +0.0967 | -0.0228 | -0.1195 |
| [`llama-2-7b-chat-warp-ratio-0.1`](https://huggingface.co/wvnvwn/llama-2-7b-chat-warp-ratio-0.1) | WSR-Tune † | 0.0000 | 0.0000 | 0.0962 | 0.1800 | 0.0691 | 0.3899 | -0.1387 | -0.0218 | +0.1169 |
| [`llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr3e-4) | Vanilla LoRA (α=32) † | 0.0019 | 0.0962 | 0.5962 | 0.7777 | 0.3680 | 0.3920 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0000 | 0.0269 | 0.3731 | 0.6342 | 0.2586 | 0.3616 | — | — | — |
| [`llama-2-7b-chat-lr5e-5-gsm8k-lr5e-5-asft`](https://huggingface.co/wvnvwn/llama-2-7b-chat-lr5e-5-gsm8k-lr5e-5-asft) | AsFT (α=32) † | 0.0000 | 0.0000 | 0.0846 | 0.3046 | 0.0973 | 0.2381 | -0.2707 | -0.1539 | +0.1168 |
| [`llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT (α=16) | 0.0000 | 0.0000 | 0.0846 | 0.1262 | 0.0527 | 0.1971 | -0.2059 | -0.1645 | +0.0414 |
| [`llama2_7b-chat-gsm8k-lisa-cb-r16a32-lr3e-4-ep3-rho0-alt`](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lisa-cb-r16a32-lr3e-4-ep3-rho0-alt) | LISA ρ=0 (α=32) † | 0.0000 | 0.0038 | 0.1327 | 0.3788 | 0.1288 | 0.3867 | -0.2392 | -0.0053 | +0.2339 |
| [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0038 | 0.1635 | 0.2308 | 0.0995 | 0.1440 | -0.1591 | -0.2176 | -0.0585 |
| [`llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr3e-4) | SafeLoRA (α=32) † | 0.0000 | 0.0000 | 0.1308 | 0.2215 | 0.0881 | 0.3283 | -0.2799 | -0.0637 | +0.2162 |
| [`llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA (α=16) | 0.0000 | 0.0000 | 0.1346 | 0.1962 | 0.0827 | 0.3108 | -0.1759 | -0.0508 | +0.1251 |
| [`llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) | SaLoRA (α=32) | 0.0712 | — | 0.8173 | 0.9242 | 0.6042 | — | +0.2362 | — | — |
| [`llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA (α=16) | 0.0000 | 0.1115 | 0.6154 | 0.7912 | 0.3795 | 0.3745 | +0.1209 | +0.0129 | -0.1080 |
| [`llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-cbsalora-new`](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-cbsalora-new) | SaLoRA (α=16) † | 0.0000 | 0.0596 | 0.4615 | 0.7088 | 0.3075 | 0.3874 | +0.0489 | +0.0258 | -0.0231 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) | WSR-LoRA (α=32) | 0.0000 | 0.1192 | 0.2731 | 0.5258 | 0.2295 | 0.3685 | -0.1385 | -0.0235 | +0.1150 |
| [`llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-cbwsrlora-rot`](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-cbwsrlora-rot) | WSR-LoRA (α=16) † | 0.0000 | 0.0000 | 0.1577 | 0.2438 | 0.1004 | 0.3692 | -0.1582 | +0.0076 | +0.1658 |
| [`llama2_7b-chat-CB_SSFT-safegrad_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safegrad_gsm8k_rho1.0_a16_lr3e-4) | SafeGrad ρ=1 (α=16) ‡ | 0.0000 | 0.0000 | 0.0712 | 0.1362 | 0.0518 | 0.3662 | -0.2068 | +0.0046 | +0.2114 |

‡ **SafeGrad** (arXiv:2508.07172) — 2026-09-13 이 박스에서 새로 학습·측정. 구현은 `safegrad/`.
  다른 행과 동일 조건(sys 모드 · GRADING=hard · AdvBench standard · GSM8K 5-shot flexible-extract,
  LoRA r16/α16 · lr 3e-4 · 3 epoch · eff.batch 16 · seed 42)에서 단일 run 으로 측정했다.
  ⚠️ AVG 0.0518 과 AsFT 의 0.0527 차이(0.0009)는 **재현 오차 범위 안**이다
  (`scripts/revision/repro_2026-09/` 기준 동일 설정 재학습 시 keyword ASR 이 ±0.05 움직인다).
  두 기법을 "SafeGrad 가 더 안전하다" 로 읽으면 안 되고, **같은 안전 수준을 downstream 손실
  없이 달성했다**(GSM8K 0.3662 vs AsFT 0.1971)는 점이 차이다.

## Llama-2-13B-chat / GSM8K

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama-2-13b-chat-hf-lr5e-5-gsm8k-lr5e-5`](https://huggingface.co/wvnvwn/llama-2-13b-chat-hf-lr5e-5-gsm8k-lr5e-5) | Full FT (SSFT+task) † | 0.0000 | 0.0038 | 0.1558 | 0.2419 | 0.1004 | 0.4594 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) | SEAL | 0.0000 | 0.0096 | 0.1519 | 0.2496 | 0.1028 | 0.4519 | +0.0024 | -0.0075 | -0.0099 |
| [`llama-2-13b-chat-hf-WaRP-lr5e-5`](https://huggingface.co/wvnvwn/llama-2-13b-chat-hf-WaRP-lr5e-5) | WSR-Tune † | 0.0000 | 0.0000 | 0.0346 | 0.0192 | 0.0135 | 0.4958 | -0.0869 | +0.0364 | +0.1233 |
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_lr3e-4) | Vanilla LoRA (α=32) | 0.0000 | 0.0038 | 0.3385 | 0.6019 | 0.2361 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0000 | 0.0096 | 0.3135 | 0.5404 | 0.2159 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) | AsFT (α=32) | 0.0000 | 0.0000 | 0.0269 | 0.0096 | 0.0091 | 0.3124 | -0.2270 | -0.1736 | +0.0534 |
| [`llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT (α=16) | 0.0000 | 0.0000 | 0.0173 | 0.0173 | 0.0086 | 0.2843 | -0.2073 | -0.2017 | +0.0056 |
| [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) | LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.0269 | 0.0077 | 0.0087 | 0.3738 | -0.2274 | -0.1122 | +0.1152 |
| [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.0462 | 0.0096 | 0.0139 | 0.3624 | -0.2020 | -0.1236 | +0.0784 |
| [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) | SafeLoRA (α=32) | 0.0000 | 0.0000 | 0.1308 | 0.0669 | 0.0494 | 0.4625 | -0.1867 | -0.0235 | +0.1632 |
| [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA (α=16) | 0.0000 | 0.0000 | 0.1058 | 0.0396 | 0.0364 | 0.4230 | -0.1795 | -0.0630 | +0.1165 |
| [`llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) | SaLoRA (α=32) | 0.0346 | 0.1750 | 0.7212 | 0.8719 | 0.4507 | 0.0720 | +0.2146 | -0.4140 | -0.6286 |
| [`llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA (α=16) | 0.0000 | 0.0096 | 0.1788 | 0.5246 | 0.1782 | 0.3844 | -0.0377 | -0.1016 | -0.0639 |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) | WSR-LoRA (α=32) | 0.0000 | 0.0019 | 0.1481 | 0.3546 | 0.1262 | 0.4822 | -0.1099 | -0.0038 | +0.1061 |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | WSR-LoRA (α=16) | 0.0000 | 0.0019 | 0.1115 | 0.2031 | 0.0791 | 0.4466 | -0.1368 | -0.0394 | +0.0974 |

## Llama-3.1-8B-It / MATH

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b_instruct_MATH_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b_instruct_MATH_lr5e-5) | Full FT (SSFT+task) | 0.0000 | 0.0000 | 0.0404 | 0.3308 | 0.0928 | 0.1212 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5) | SEAL | 0.0000 | 0.0000 | 0.0096 | 0.1677 | 0.0443 | 0.1318 | -0.0485 | +0.0106 | +0.0591 |
| [`llama3.1_8b_instruct-MATH-WaRP-lr5e-5`](https://huggingface.co/kmseong/llama3.1_8b_instruct-MATH-WaRP-lr5e-5) | WSR-Tune | 0.0000 | 0.0000 | 0.0115 | 0.2131 | 0.0562 | 0.1370 | -0.0366 | +0.0158 | +0.0524 |
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_lr3e-4) | Vanilla LoRA (α=32) | 0.0000 | 0.0000 | 0.0135 | 0.2485 | 0.0655 | 0.2456 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0154 | 0.2173 | 0.0582 | 0.2530 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4) | AsFT (α=32) | 0.0000 | 0.0000 | 0.0038 | 0.1523 | 0.0390 | 0.0686 | -0.0265 | -0.1770 | -0.1505 |
| [`llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_a16_lr3e-4) | AsFT (α=16) | 0.0000 | 0.0000 | 0.0038 | 0.1308 | 0.0336 | 0.0566 | -0.0246 | -0.1964 | -0.1718 |
| [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4) | LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.0096 | 0.1454 | 0.0388 | 0.1222 | -0.0267 | -0.1234 | -0.0967 |
| [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.0077 | 0.1331 | 0.0352 | 0.1132 | -0.0230 | -0.1398 | -0.1168 |
| [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4) | SafeLoRA (α=32) | 0.0000 | 0.0000 | 0.0077 | 0.1585 | 0.0416 | 0.2300 | -0.0239 | -0.0156 | +0.0083 |
| [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4) | SafeLoRA (α=16) | 0.0000 | 0.0000 | 0.0058 | 0.1519 | 0.0394 | 0.2204 | -0.0188 | -0.0326 | -0.0138 |
| [`llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4) | SaLoRA (α=32) | 0.0000 | 0.0000 | 0.0173 | 0.2600 | 0.0693 | 0.1938 | +0.0038 | -0.0518 | -0.0556 |
| [`llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-salora_math_rs32rt32_a16_lr3e-4) | SaLoRA (α=16) | 0.0000 | 0.0000 | 0.0058 | 0.2565 | 0.0656 | 0.2320 | +0.0074 | -0.0210 | -0.0284 |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4) | WSR-LoRA (α=32) | 0.0000 | 0.0000 | 0.0135 | 0.1915 | 0.0513 | 0.1972 | -0.0142 | -0.0484 | -0.0342 |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4) | WSR-LoRA (α=16) | 0.0000 | 0.0000 | 0.0077 | 0.1631 | 0.0427 | 0.2240 | -0.0155 | -0.0290 | -0.0135 |

## Llama-3.2-3B-It / MATH

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b_instruct_MATH_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b_instruct_MATH_lr5e-5) | Full FT (SSFT+task) | 0.0019 | 0.0000 | 0.0577 | 0.2862 | 0.0865 | 0.2152 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-seal_math_topp0.8_lr5e-5) | SEAL | 0.0000 | 0.0000 | 0.0731 | 0.2615 | 0.0837 | 0.2162 | -0.0028 | +0.0010 | +0.0038 |
| [`llama3_2_3b-instruct-WaRP_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-WaRP_lr5e-5) | WSR-Tune | 0.0000 | 0.0000 | 0.0423 | 0.2135 | 0.0640 | 0.2238 | -0.0225 | +0.0086 | +0.0311 |
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_lr3e-4) | Vanilla LoRA (α=32) | 0.0000 | 0.0000 | 0.0558 | 0.2565 | 0.0781 | 0.2364 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0712 | 0.2581 | 0.0823 | 0.2480 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4) | AsFT (α=32) | 0.0000 | 0.0000 | 0.0596 | 0.2342 | 0.0735 | 0.0714 | -0.0046 | -0.1650 | -0.1604 |
| [`llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_a16_lr3e-4) | AsFT (α=16) | 0.0000 | 0.0000 | 0.0519 | 0.2323 | 0.0711 | 0.0776 | -0.0112 | -0.1704 | -0.1592 |
| [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4) | LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.0654 | 0.2527 | 0.0795 | 0.1012 | +0.0014 | -0.1352 | -0.1366 |
| [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.0596 | 0.2135 | 0.0683 | 0.0882 | -0.0140 | -0.1598 | -0.1458 |
| [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_lr3e-4) | SafeLoRA (α=32) | 0.0000 | 0.0000 | 0.0654 | 0.2108 | 0.0691 | 0.2260 | -0.0090 | -0.0104 | -0.0014 |
| [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4) | SafeLoRA (α=16) | 0.0000 | 0.0000 | 0.0615 | 0.1888 | 0.0626 | 0.2304 | -0.0197 | -0.0176 | +0.0021 |
| [`llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_lr3e-4) | SaLoRA (α=32) | 0.0000 | 0.0000 | 0.0923 | 0.2935 | 0.0965 | 0.2084 | +0.0184 | -0.0280 | -0.0464 |
| [`llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-salora_math_rs32rt32_a16_lr3e-4) | SaLoRA (α=16) | 0.0019 | 0.0000 | 0.0750 | 0.2969 | 0.0935 | 0.2238 | +0.0112 | -0.0242 | -0.0354 |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_lr3e-4) | WSR-LoRA (α=32) | 0.0000 | 0.0000 | 0.0596 | 0.2188 | 0.0696 | 0.2248 | -0.0085 | -0.0116 | -0.0031 |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4) | WSR-LoRA (α=16) | 0.0000 | 0.0000 | 0.0558 | 0.2188 | 0.0687 | 0.2468 | -0.0136 | -0.0012 | +0.0124 |

## Qwen2.5-7B-It / GSM8K

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`qwen-2.5-7B-Instruct-SSFT-gsm8k-lr5e-5`](https://huggingface.co/wvnvwn/qwen-2.5-7B-Instruct-SSFT-gsm8k-lr5e-5) | Full FT (SSFT+task) † | 0.0000 | 0.0000 | 0.0231 | 0.1215 | 0.0362 | 0.6732 | — | — | — |
| [`qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) | SEAL | 0.0000 | 0.0000 | 0.0385 | 0.1350 | 0.0434 | 0.7005 | +0.0072 | +0.0273 | +0.0201 |
| [`qwen-2.5-7B-Instruct-WaRP-lr5e-5`](https://huggingface.co/wvnvwn/qwen-2.5-7B-Instruct-WaRP-lr5e-5) | WSR-Tune † | 0.0000 | 0.0000 | 0.0192 | 0.0965 | 0.0289 | 0.6945 | -0.0073 | +0.0213 | +0.0286 |
| [`qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4) | Vanilla LoRA (α=32) | 0.0019 | 0.0000 | 0.0462 | 0.1050 | 0.0383 | 0.7149 | — | — | — |
| [`qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0385 | 0.0858 | 0.0311 | 0.7119 | — | — | — |
| [`qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) | AsFT (α=32) | 0.0000 | 0.0000 | 0.0231 | 0.1192 | 0.0356 | 0.7377 | -0.0027 | +0.0228 | +0.0255 |
| [`qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT (α=16) | 0.0000 | 0.0000 | 0.0212 | 0.1273 | 0.0371 | 0.7202 | +0.0060 | +0.0083 | +0.0023 |
| [`qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) | LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.0288 | 0.1300 | 0.0397 | 0.7278 | +0.0014 | +0.0129 | +0.0115 |
| [`qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.0308 | 0.1388 | 0.0424 | 0.7134 | +0.0113 | +0.0015 | -0.0098 |
| [`qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) | SafeLoRA (α=32) | 0.0000 | 0.0000 | 0.0173 | 0.0727 | 0.0225 | 0.7483 | -0.0158 | +0.0334 | +0.0492 |
| [`qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA (α=16) | 0.0000 | 0.0000 | 0.0173 | 0.0719 | 0.0223 | 0.7374 | -0.0088 | +0.0255 | +0.0343 |
| [`qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) | SaLoRA (α=32) | 0.0000 | 0.0000 | 0.0385 | 0.1535 | 0.0480 | 0.6990 | +0.0097 | -0.0159 | -0.0256 |
| [`qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA (α=16) | 0.0000 | 0.0000 | 0.0519 | 0.1242 | 0.0440 | 0.7066 | +0.0129 | -0.0053 | -0.0182 |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) | WSR-LoRA (α=32) | 0.0000 | 0.0000 | 0.0288 | 0.1085 | 0.0343 | 0.7278 | -0.0040 | +0.0129 | +0.0169 |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | WSR-LoRA (α=16) | 0.0000 | 0.0000 | 0.0269 | 0.1008 | 0.0319 | 0.7271 | +0.0008 | +0.0152 | +0.0144 |

## Gemma-2-9B-IT / GSM8K

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`gemma-2-9b-it-lr3e-5-gsm8k-lr1e-5`](https://huggingface.co/wvnvwn/gemma-2-9b-it-lr3e-5-gsm8k-lr1e-5) | Full FT (SSFT+task) † | 0.0000 | 0.0038 | 0.0096 | 0.2015 | 0.0537 | 0.6975 | — | — | — |
| [`gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) | SEAL (lr 5e-5, **버그**) ‡ | 0.4500 | 0.2558 | 0.8365 | 0.8354 | 0.5944 | 0.1865 | +0.5407 | -0.5110 | -1.0517 |
| [`gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr1e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr1e-5) | SEAL (lr 1e-5, 재학습) ‡ | 0.0000 | 0.0038 | 0.0077 | 0.1631 | 0.0437 | 0.6831 | -0.0100 | -0.0144 | -0.0044 |
| [`gemma-2-9b-it-lr3e-5-WaRP-lr1e-5`](https://huggingface.co/wvnvwn/gemma-2-9b-it-lr3e-5-WaRP-lr1e-5) | WSR-Tune † | 0.0000 | 0.0000 | 0.0077 | 0.1919 | 0.0499 | 0.7081 | -0.0038 | +0.0106 | +0.0144 |
| [`gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_lr3e-4) | Vanilla LoRA (α=32) | 0.0000 | 0.0577 | 0.1058 | 0.2454 | 0.1022 | 0.6937 | — | — | — |
| [`gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0000 | 0.1423 | 0.0750 | 0.2996 | 0.1292 | 0.7074 | — | — | — |
| [`gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4) | AsFT (α=32) | 0.0000 | 0.0000 | 0.0058 | 0.2142 | 0.0550 | 0.5997 | -0.0472 | -0.0940 | -0.0468 |
| [`gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT (α=16) | 0.0000 | 0.0000 | 0.0077 | 0.1831 | 0.0477 | 0.3662 | -0.0815 | -0.3412 | -0.2597 |
| [`gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4) | LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.0019 | 0.1358 | 0.0344 | 0.5656 | -0.0678 | -0.1281 | -0.0603 |
| [`gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.0019 | 0.1192 | 0.0303 | 0.6528 | -0.0989 | -0.0546 | +0.0443 |
| [`gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_lr3e-4) | SafeLoRA (α=32) | 0.0000 | 0.0038 | 0.0212 | 0.2131 | 0.0595 | 0.6694 | -0.0427 | -0.0243 | +0.0184 |
| [`gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA (α=16) | 0.0019 | 0.0000 | 0.0135 | 0.1823 | 0.0494 | 0.6929 | -0.0798 | -0.0145 | +0.0653 |
| [`gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4) | SaLoRA (α=32) | 0.0000 | 0.1327 | 0.1865 | 0.5242 | 0.2109 | 0.4792 | +0.1087 | -0.2145 | -0.3232 |
| [`gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA (α=16) | 0.0000 | 0.0192 | 0.1038 | 0.3296 | 0.1132 | 0.6823 | -0.0160 | -0.0251 | -0.0091 |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_lr3e-4) | WSR-LoRA (α=32) | 0.0000 | 0.0154 | 0.0327 | 0.2458 | 0.0735 | 0.6315 | -0.0287 | -0.0622 | -0.0335 |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | WSR-LoRA (α=16) | 0.0000 | 0.0615 | 0.0269 | 0.2108 | 0.0748 | 0.6884 | -0.0544 | -0.0190 | +0.0354 |

‡ **gemma2_9b SEAL 은 lr 버그로 망가져 있었다 (2026-09-16 규명·재학습).**
`common.sh` 의 `model_cfg()` 가 gemma 에 대해 `SSFT_LR` 만 3e-5 로 덮어쓰고 `FULL_LR` 은
전역 기본값 **5e-5** 를 그대로 썼다. 이 모델의 다른 full-param arm 은 전부 lr 1e-5 다
(`wvnvwn/gemma-2-9b-it-lr3e-5-gsm8k-lr1e-5` · `...-WaRP-lr1e-5`, 위 측정조건표의
"full-param 계열 | lr 5e-5 (gemma 1e-5)"). 즉 SEAL 만 **다른 arm 의 5배 lr** 로 full-param
3 epoch 을 돌아 정렬과 성능이 함께 무너졌다.

CLAUDE.md 는 이를 "cell-specific failure mode, not a pipeline bug" 로 적고 재학습했지만
**같은 잘못된 lr 을 다시 써서** 같은 결과가 나왔다. 실제로는 레지스트리 버그다.

lr 1e-5 로 재학습한 결과(AVG 0.5944 → **0.0437**, GSM8K 0.1865 → **0.6831**)는 같은 모델의
다른 arm 과 자릿수가 맞고, 오히려 셋 중 **가장 안전하다**(SEAL 0.0437 < WSR-Tune 0.0499 <
Full FT 0.0537). 표에는 재학습본을 쓰고 기존 행은 이력으로 남긴다.

고친 것: `common.sh` 에 **모델별 `FULL_LR`** 도입(gemma2_9b=1e-5, llama2_7b_base=3e-5,
llama31_8b_base=1e-5 — 전부 허브 리포명과 대조해 검증, 나머지 5e-5). 호출자가 `FULL_LR` 을
명시하면 그쪽이 이긴다. 또한 `hf_repo_id()` 가 스스로 `model_cfg` 를 부르게 했다 —
`$( )` 서브셸에서 불리면 모델별 lr 이 반영되지 않아 **조용히 틀린 리포명**(gemma 가 lr5e-5)이
나왔다. `print_plan` / `21_seal.sh` 헤더가 `model_cfg` 전에 전역 lr 을 찍어 이 버그를
가리던 것도 고쳤다.


---

# 추가 실험 (2026-09-02 ~ 09-04 허브 업로드분, 스프레드시트에 없는 셀)

모든 행은 **α=16 세대**이며 기준(Δ)은 같은 모델의 **Vanilla LoRA (α=16)** 이다. 비교를 위해 본표에 있는 행(ρ=0.1 / thr0.3 / step100-900)을 `▸` 로 함께 적었다.

## A. WSR-LoRA ρ(keep_ratio) sweep — α=16

ρ = 동결하는 safety 방향 비율. ρ 가 클수록 더 많이 동결한다(ρ=0.1 이 본표 설정).

**Llama-2-7B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0269 | 0.3731 | 0.6342 | 0.2586 | 0.3616 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4) | WSR-LoRA ρ=0.05 | 0.0000 | 0.0558 | 0.2462 | 0.4873 | 0.1973 | 0.3791 | -0.0613 | +0.0175 | +0.0788 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | ▸ WSR-LoRA ρ=0.1 | 0.0000 | 0.0462 | 0.1808 | 0.2838 | 0.1277 | 0.3707 | -0.1309 | +0.0091 | +0.1400 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4) | WSR-LoRA ρ=0.2 | 0.0000 | 0.0058 | 0.1673 | 0.2404 | 0.1034 | 0.3381 | -0.1552 | -0.0235 | +0.1317 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 | 0.0000 | 0.0019 | 0.1250 | 0.1700 | 0.0742 | 0.3245 | -0.1844 | -0.0371 | +0.1473 |

**Llama-2-13B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0096 | 0.3135 | 0.5404 | 0.2159 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4) | WSR-LoRA ρ=0.05 | 0.0000 | 0.0000 | 0.1269 | 0.2696 | 0.0991 | 0.4723 | -0.1168 | -0.0137 | +0.1031 |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | ▸ WSR-LoRA ρ=0.1 | 0.0000 | 0.0019 | 0.1115 | 0.2031 | 0.0791 | 0.4466 | -0.1368 | -0.0394 | +0.0974 |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4) | WSR-LoRA ρ=0.2 | 0.0000 | 0.0000 | 0.0865 | 0.0800 | 0.0416 | 0.4549 | -0.1743 | -0.0311 | +0.1432 |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 | 0.0000 | 0.0000 | 0.0788 | 0.0412 | 0.0300 | 0.4481 | -0.1859 | -0.0379 | +0.1480 |

**Llama-3.1-8B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0154 | 0.2173 | 0.0582 | 0.2530 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.05_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.05_a16_lr3e-4) | WSR-LoRA ρ=0.05 | 0.0000 | 0.0000 | 0.0058 | 0.1738 | 0.0449 | 0.2202 | -0.0133 | -0.0328 | -0.0195 |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4) | ▸ WSR-LoRA ρ=0.1 | 0.0000 | 0.0000 | 0.0077 | 0.1631 | 0.0427 | 0.2240 | -0.0155 | -0.0290 | -0.0135 |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.2_a16_lr3e-4) | WSR-LoRA ρ=0.2 | 0.0000 | 0.0000 | 0.0115 | 0.1673 | 0.0447 | 0.2394 | -0.0135 | -0.0136 | -0.0001 |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 | 0.0000 | 0.0000 | 0.0058 | 0.1615 | 0.0418 | 0.2400 | -0.0164 | -0.0130 | +0.0034 |

**Llama-3.2-3B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0712 | 0.2581 | 0.0823 | 0.2480 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.05_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.05_a16_lr3e-4) | WSR-LoRA ρ=0.05 | 0.0000 | 0.0000 | 0.0712 | 0.2296 | 0.0752 | 0.2468 | -0.0071 | -0.0012 | +0.0059 |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.1_a16_lr3e-4) | ▸ WSR-LoRA ρ=0.1 | 0.0000 | 0.0000 | 0.0558 | 0.2188 | 0.0687 | 0.2468 | -0.0136 | -0.0012 | +0.0124 |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.2_a16_lr3e-4) | WSR-LoRA ρ=0.2 | 0.0000 | 0.0000 | 0.0577 | 0.2258 | 0.0709 | 0.2584 | -0.0114 | +0.0104 | +0.0218 |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 | 0.0000 | 0.0000 | 0.0481 | 0.2223 | 0.0676 | 0.2568 | -0.0147 | +0.0088 | +0.0235 |

**Qwen2.5-7B-It / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0385 | 0.0858 | 0.0311 | 0.7119 | — | — | — |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4) | WSR-LoRA ρ=0.05 | 0.0000 | 0.0000 | 0.0250 | 0.1031 | 0.0320 | 0.7074 | +0.0009 | -0.0045 | -0.0054 |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | ▸ WSR-LoRA ρ=0.1 | 0.0000 | 0.0000 | 0.0269 | 0.1008 | 0.0319 | 0.7271 | +0.0008 | +0.0152 | +0.0144 |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4) | WSR-LoRA ρ=0.2 | 0.0000 | 0.0000 | 0.0192 | 0.0985 | 0.0294 | 0.7036 | -0.0017 | -0.0083 | -0.0066 |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 | 0.0000 | 0.0000 | 0.0212 | 0.0988 | 0.0300 | 0.7354 | -0.0011 | +0.0235 | +0.0246 |

**Gemma-2-9B-IT / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.1423 | 0.0750 | 0.2996 | 0.1292 | 0.7074 | — | — | — |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.05_a16_lr3e-4) | WSR-LoRA ρ=0.05 | 0.0000 | 0.0712 | 0.0250 | 0.2296 | 0.0814 | 0.6793 | -0.0478 | -0.0281 | +0.0197 |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | ▸ WSR-LoRA ρ=0.1 | 0.0000 | 0.0615 | 0.0269 | 0.2108 | 0.0748 | 0.6884 | -0.0544 | -0.0190 | +0.0354 |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.2_a16_lr3e-4) | WSR-LoRA ρ=0.2 | 0.0000 | 0.0192 | 0.0192 | 0.2058 | 0.0611 | 0.6907 | -0.0681 | -0.0167 | +0.0514 |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 | 0.0000 | 0.0077 | 0.0135 | 0.1869 | 0.0520 | 0.7081 | -0.0772 | +0.0007 | +0.0779 |

## B. SafeLoRA threshold 0.3 vs 0.35 — α=16

threshold 가 높을수록 projection 되는 layer 가 많아 안전 쪽으로 강하게 보정한다.

**Llama-2-7B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0269 | 0.3731 | 0.6342 | 0.2586 | 0.3616 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | ▸ SafeLoRA thr=0.3 | 0.0000 | 0.0000 | 0.1346 | 0.1962 | 0.0827 | 0.3108 | -0.1759 | -0.0508 | +0.1251 |
| [`llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.35_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.35_a16_lr3e-4) | SafeLoRA thr=0.35 | 0.0000 | 0.0000 | 0.1269 | 0.1812 | 0.0770 | 0.2949 | -0.1816 | -0.0667 | +0.1149 |

**Llama-2-13B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0096 | 0.3135 | 0.5404 | 0.2159 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | ▸ SafeLoRA thr=0.3 | 0.0000 | 0.0000 | 0.1058 | 0.0396 | 0.0364 | 0.4230 | -0.1795 | -0.0630 | +0.1165 |
| [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.35_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.35_a16_lr3e-4) | SafeLoRA thr=0.35 | 0.0000 | 0.0000 | 0.0808 | 0.0288 | 0.0274 | 0.3958 | -0.1885 | -0.0902 | +0.0983 |

**Llama-3.1-8B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0154 | 0.2173 | 0.0582 | 0.2530 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4) | ▸ SafeLoRA thr=0.3 | 0.0000 | 0.0000 | 0.0058 | 0.1519 | 0.0394 | 0.2204 | -0.0188 | -0.0326 | -0.0138 |
| [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4) | SafeLoRA thr=0.35 | 0.0000 | 0.0000 | 0.0058 | 0.1673 | 0.0433 | 0.2126 | -0.0149 | -0.0404 | -0.0255 |

**Llama-3.2-3B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0712 | 0.2581 | 0.0823 | 0.2480 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.3_a16_lr3e-4) | ▸ SafeLoRA thr=0.3 | 0.0000 | 0.0000 | 0.0615 | 0.1888 | 0.0626 | 0.2304 | -0.0197 | -0.0176 | +0.0021 |
| [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4) | SafeLoRA thr=0.35 | 0.0000 | 0.0000 | 0.0577 | 0.1858 | 0.0609 | 0.2214 | -0.0214 | -0.0266 | -0.0052 |

## C. LISA alignment/finetune step 비율 — ρ=1.0, α=16

본표는 `align_step 100 / ft_step 900`, 여기서는 `500 / 500` (safety 데이터 비중 5배).

**Llama-2-7B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0269 | 0.3731 | 0.6342 | 0.2586 | 0.3616 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | ▸ LISA ρ=1 step 100/900 | 0.0000 | 0.0038 | 0.1635 | 0.2308 | 0.0995 | 0.1440 | -0.1591 | -0.2176 | -0.0585 |
| [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_step500-500_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_step500-500_a16_lr3e-4) | LISA ρ=1 step 500/500 | 0.0000 | 0.0000 | 0.1000 | 0.1138 | 0.0534 | 0.1691 | -0.2052 | -0.1925 | +0.0127 |

**Llama-2-13B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0096 | 0.3135 | 0.5404 | 0.2159 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | ▸ LISA ρ=1 step 100/900 | 0.0000 | 0.0000 | 0.0462 | 0.0096 | 0.0139 | 0.3624 | -0.2020 | -0.1236 | +0.0784 |
| [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_step500-500_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_step500-500_a16_lr3e-4) | LISA ρ=1 step 500/500 | 0.0000 | 0.0019 | 0.1692 | 0.0546 | 0.0564 | 0.3298 | -0.1595 | -0.1562 | +0.0033 |

**Llama-3.1-8B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0154 | 0.2173 | 0.0582 | 0.2530 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4) | ▸ LISA ρ=1 step 100/900 | 0.0000 | 0.0000 | 0.0077 | 0.1331 | 0.0352 | 0.1132 | -0.0230 | -0.1398 | -0.1168 |
| [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_step500-500_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_step500-500_a16_lr3e-4) | LISA ρ=1 step 500/500 | 0.0000 | 0.0000 | 0.0077 | 0.1731 | 0.0452 | 0.0808 | -0.0130 | -0.1722 | -0.1592 |

**Llama-3.2-3B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0712 | 0.2581 | 0.0823 | 0.2480 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_a16_lr3e-4) | ▸ LISA ρ=1 step 100/900 | 0.0000 | 0.0000 | 0.0596 | 0.2135 | 0.0683 | 0.0882 | -0.0140 | -0.1598 | -0.1458 |
| [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_step500-500_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_step500-500_a16_lr3e-4) | LISA ρ=1 step 500/500 | 0.0000 | 0.0000 | 0.0500 | 0.2012 | 0.0628 | 0.0848 | -0.0195 | -0.1632 | -0.1437 |

## D. Llama-2-7B ARC / MedQA — LISA α=16

α=16 Vanilla LoRA 가 ARC/MedQA 에는 없어 Δ 는 비워 두었다. 참고용 α=32 행(8/28 측정)을 `▸` 로 함께 적었다.

**Llama-2-7B-chat / ARC-c**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | ARC-c | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_arc_lr3e-4) | ▸ Vanilla LoRA (α=32) | 0.0000 | 0.0000 | 0.1692 | 0.3308 | 0.1250 | 0.6203 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_lr3e-4) | ▸ LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.0962 | 0.2612 | 0.0893 | 0.3447 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.0942 | 0.2723 | 0.0916 | 0.3097 | — | — | — |

**Llama-2-7B-chat / MedQA**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MedQA | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_medqa_lr3e-4) | ▸ Vanilla LoRA (α=32) | 0.0000 | — | 0.2308 | 0.3900 | 0.2069 | 0.4588 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_lr3e-4) | ▸ LISA ρ=1 (α=32) | 0.0000 | 0.0000 | 0.1000 | 0.1408 | 0.0602 | 0.3496 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_a16_lr3e-4) | LISA ρ=1 (α=16) | 0.0000 | 0.0000 | 0.1192 | 0.1565 | 0.0689 | 0.3181 | — | — | — |

---

† = rebuttal 시기(7–8월)에 만든 모델(`wvnvwn/*`, `-matched-`, `-alt`, 초기 SEAL)을 그대로 쓴 행. Llama-2-7B 의 α=32 LoRA 계열은 대부분 이 세대다 (출발 모델 `wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb`, revision 세대와 safetensors 가 동일한지 미확인). Llama-2-7B LISA(α=32) 는 ρ=0 모델이고 나머지 모델의 LISA 는 ρ=1.0 이다. Llama-2-13B SaLoRA(α=32) 와 Gemma-2-9B SEAL 은 재학습해도 같은 붕괴가 재현되는 셀이다. **Llama-2-7B SaLoRA(α=32)** 는 스프레드시트에서 α=16 모델의 수치가 잘못 들어가 있던 행이다 — 여기서는 08-28 측정치(AutoDAN 누락, AVG 는 3개 공격 평균)로 바꾸고 GSM8K 는 미측정(—)으로 두었으며, α=16 kmseong 모델은 별도 행으로 분리했다.


# BeaverTails (BV) — Llama-2-7B-Chat / GSM8K (rebuttal 상세 복원)

[tem_csv.txt](tem_csv.txt)의 BeaverTails 실험을 [Reviewer 3 답변 기록](WSR_Tune_NeurIPS2026_Rebuttal_Full_Record.md)의 **Constructed BeaverTails Dataset** 및 후속 **BeaverTails 정정 표**와 대조했다. 아래 수치는 기존 RESULTS.md와 같은 **0–1 비율**이며, rebuttal의 백분율은 100으로 나누어 비교했다. `PAP`는 원본의 공격명이다(요청의 `paa`에 해당). ASR은 원본의 keyword 채점 결과이고 task는 GSM8K이다.

주소는 원본에 namespace까지 명시된 경우에만 Hugging Face 링크로 옮겼으며, 원격 저장소의 현재 존재 여부는 확인하지 않았다. **없음**은 이 두 기록에서 확인되지 않는다는 뜻이다. 원본의 AVG를 그대로 보존하며, 반올림된 공격별 수치의 평균과 마지막 자리에서 차이가 날 수 있다. 기존 CB/α=16 revision 설정을 이 BV rebuttal 실험에 적용하지 않는다.

## 주요 방법 비교

| 모델 주소 | 메소드 | Direct | AutoDAN | PAIR | PAP | ASR AVG | Task acc (GSM8K) |
|---|---|---:|---:|---:|---:|---:|---:|
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvvanilla](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvvanilla) | Full FT (원본 L371) | 0.0019 | 0.0673 | 0.3404 | 0.3569 | 0.1916 | 0.4185 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvsafeinstr0_1](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvsafeinstr0_1) | SafeInstr (원본 L372) | 0.0000 | 0.0288 | 0.1942 | 0.1338 | 0.0892 | 0.4033 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvresta0_3](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvresta0_3) | Resta (원본 L373) | 0.0019 | 0.0462 | 0.3654 | 0.3288 | 0.1856 | 0.3935 |
| 없음 | SEAL (BV 기록 없음) | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvsafedelta0_1](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvsafedelta0_1) | SafeDelta (0.1) (원본 L374) | 0.0000 | 0.0000 | 0.1019 | 0.2008 | 0.0757 | 0.2449 |
| [kmseong/llama2_7b-chat-gsm8k-sn-tuned-lr5e-5-bt](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-sn-tuned-lr5e-5-bt) | SN-Tune (원본 L376) | 0.0019 | 0.1327 | 0.2962 | 0.4308 | 0.2154 | 0.4109 |
| [kmseong/llama2_7b-chat-gsm8k-rsn-tuned-lr5e-5-bt](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-rsn-tuned-lr5e-5-bt) | RSN-Tune (원본 L377) | 0.0019 | 0.1365 | 0.3500 | 0.4973 | 0.2464 | 0.4124 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp4994](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp4994) | WSR-Tune (Full) (원본 L384) | 0.0000 | 0.0019 | 0.1327 | 0.1527 | 0.0718 | 0.4109 |
| [kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr3e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr3e-4-beavertails) | Vanilla LoRA (lr=3e-4) (원본 L382) | 0.0019 | 0.0558 | 0.2308 | 0.4088 | 0.1743 | 0.3995 |
| [kmseong/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr3e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr3e-4-beavertails) | Lisa (lr=3e-4; rebuttal 표 미수록) (원본 L64) | 0.0000 | 0.0173 | 0.1846 | 0.1569 | 0.0897 | 0.2138 |
| [kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr3e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr3e-4-beavertails) | SafeLoRA (정정값; lr=3e-4) (원본 L378) | 0.0019 | 0.0269 | 0.1981 | 0.3031 | 0.1325 | 0.3882 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvsalora-new](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvsalora-new) | SaLoRA (최종 rebuttal; bvsalora-new) (원본 L379) | 0.0019 | 0.0981 | 0.3442 | 0.4008 | 0.2113 | 0.3904 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvwsr-4994](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvwsr-4994) | WSR-LoRA (4994; lr=3e-4) (원본 L383) | 0.0019 | 0.0346 | 0.2192 | 0.2615 | 0.1293 | 0.4155 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvasft](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvasft) | AsFT (ASR 기록 충돌; 아래 두 원본 행 참조) | 미확정 | 미확정 | 미확정 | 미확정 | 미확정 | 0.2638 |

대조 및 누락 사항:

- **SafeLoRA:** 초기 BV 표의 8.81/32.83은 오류로 정정되었다. 최종 13.25/38.82와 원본 L378이 일치하므로 **0.1325/0.3882**를 채택했다.
- **SaLoRA:** 최종 rebuttal 21.13/39.04는 `bvsalora-new`(L379, L485)와 일치한다. 이전 `matched-r16-a32` 실험(L56–58)은 다른 모델이므로 섞지 않았다.
- **SN-Tune·RSN-Tune:** raw ASR/task 값은 rebuttal과 일치한다. 초기 rebuttal의 잘못된 downstream delta는 복사하지 않았다.
- **Lisa:** 최종 BEAVER 요약의 L380은 평가 칸이 비어 있지만, 별도 BV 실험 L64에 주소와 6개 수치가 모두 있다. 주표에는 lr=3e-4 행을 기재했다. rho·교대 step 설정은 원본에 없어 확인할 수 없다.
- **AsFT:** 동일 주소에 L95와 L381의 공격별 ASR/AVG가 서로 다르며 task acc만 같다. Reviewer 3의 BV 표에는 AsFT가 없어 어느 ASR이 최종값인지 결정하지 않았다. 아래에 두 행을 모두 보존했다.
- **SEAL:** BV 모델 주소와 평가 결과 모두 없음. CB SEAL 결과로 대신 채우지 않았다.

## AsFT의 상충하는 원본 기록

| 모델 주소 | 메소드 | Direct | AutoDAN | PAIR | PAP | ASR AVG | Task acc (GSM8K) |
|---|---|---:|---:|---:|---:|---:|---:|
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvasft](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvasft) | AsFT (lr=3e-4; 초기 BV 블록) (원본 L95) | 0.0000 | 0.0038 | 0.1731 | 0.1646 | 0.0854 | 0.2638 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvasft](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr3e-4-bvasft) | AsFT (lr=3e-4; 최종 BEAVER 요약) (원본 L381) | 0.0038 | 0.0192 | 0.4000 | 0.6135 | 0.2591 | 0.2638 |

## Constructed BeaverTails Dataset의 데이터 ablation

아래 50%/10%는 **importance 데이터 비율**, Basis 50%/25%는 **basis 구축 데이터 비율**이다. 동결 비율 sweep과 구분한다. 원본의 `bvwarp2500`/`bvwarp500`은 rebuttal에서 각각 약 50%/10%로 표시한 실험이다.

| 모델 주소 | 메소드 | Direct | AutoDAN | PAIR | PAP | ASR AVG | Task acc (GSM8K) |
|---|---|---:|---:|---:|---:|---:|---:|
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp2500](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp2500) | WSR-Tune (importance BV 50%; 2500) (원본 L9) | 0.0000 | 0.0019 | 0.1404 | 0.1727 | 0.0788 | 0.4086 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp500](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp500) | WSR-Tune (importance BV 10%; 500) (원본 L11) | 0.0000 | 0.0000 | 0.1442 | 0.1758 | 0.0800 | 0.4086 |
| 없음 (namespace 미기재; `llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvwarp-cbimportance`) | WSR-Tune-CB (BV basis / CB importance) (원본 L112) | 0.0000 | 0.0019 | 0.1192 | 0.1796 | 0.0752 | 0.4086 |
| 없음 (namespace 미기재; `llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvbasis0_50`) | WSR-Tune (Basis BV 50%) (원본 L127) | 0.0000 | 0.0000 | 0.1500 | 0.2654 | 0.1038 | 0.4018 |
| 없음 (namespace 미기재; `llama2-7b-chat-lr5e-5-gsm8k-lr5e-5-bvbasis0_25`) | WSR-Tune (Basis BV 25%) (원본 L126) | 0.0000 | 0.0000 | 0.1423 | 0.2269 | 0.0923 | 0.3821 |

교차 importance 및 Basis 50%/25% 모델은 원본에 모델 이름만 있고 소유자 namespace가 없어 주소를 **없음**으로 표시했다. `bvbasis1_00`(L128)은 주표의 `bvwarp4994`와 수치가 다르므로 Full 행으로 대체하지 않았다.

---

# 추가 실험 2 (2026-09-04 ~ 09-07, hb_repro 환경 재학습분) — 6개 모델 · α=16

원본 박스 라이브러리(torch 2.10 / transformers 4.57.3 / peft 0.18.1)로 만든 conda env `hb_repro` 에서 이 박스(GPU 0, 공유)에 새로 학습·업로드한 24셀. 학습 설정은 A/B 와 동일(r=16, α=16, lr 3e-4, 3 epoch, eff.batch 16, seed 42; WSR-LoRA 는 `wsr_lora.py --reparam`, SafeLoRA 는 `--method safe_lora`). 측정 조건도 A/B 와 같다(HarmBench sys · keyword, family 기본 util; lm-eval GSM8K flexible / `hendrycks_math_safe`). 단, **Llama-2-7B 의 HarmBench 는 util 0.85** 로 측정했다(A/B 의 7B 행은 0.95; 공유 GPU 에서 0.95 는 기동 불가). 참조는 A/B 와 같은 모델의 Vanilla LoRA α=16. 같은 셀을 seed 고정으로 재학습해도 Δ 방향 코사인이 ≈0.54 이고 ASR 이 ±0.05 안에서 흔들리므로(2026-09-04 재현 실험, `logs/revision_repro/REPORT_safelora_3b_thr0.35_a16.md`), 이 크기의 차이는 유의하게 읽지 않는다.

## E. WSR-LoRA freeze ratio ρ = 0.4 / 0.5 — α=16 (A 의 ρ sweep 확장)

**Llama-2-7B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0269 | 0.3731 | 0.6342 | 0.2586 | 0.3616 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4) | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.1096 | 0.1500 | 0.0649 | 0.3161 | -0.1937 | -0.0455 | +0.1482 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4) | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.1058 | 0.1554 | 0.0653 | 0.2896 | -0.1933 | -0.0720 | +0.1213 |

**Llama-2-13B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0096 | 0.3135 | 0.5404 | 0.2159 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4) | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0769 | 0.0281 | 0.0262 | 0.4435 | -0.1897 | -0.0425 | +0.1472 |
| [`llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4) | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0654 | 0.0231 | 0.0221 | 0.4420 | -0.1938 | -0.0440 | +0.1498 |

**Llama-3.1-8B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0154 | 0.2173 | 0.0582 | 0.2530 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.4_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.4_a16_lr3e-4) | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0038 | 0.1738 | 0.0444 | 0.2236 | -0.0138 | -0.0294 | -0.0156 |
| [`llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.5_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.5_a16_lr3e-4) | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0058 | 0.1662 | 0.0430 | 0.2332 | -0.0152 | -0.0198 | -0.0046 |

**Llama-3.2-3B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0712 | 0.2581 | 0.0823 | 0.2480 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.4_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.4_a16_lr3e-4) | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0500 | 0.2150 | 0.0663 | 0.2538 | -0.0160 | +0.0058 | +0.0218 |
| [`llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.5_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.5_a16_lr3e-4) | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0481 | 0.2008 | 0.0622 | 0.2568 | -0.0201 | +0.0088 | +0.0289 |

**Qwen2.5-7B-It / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0385 | 0.0858 | 0.0311 | 0.7119 | — | — | — |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4) | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0192 | 0.1035 | 0.0307 | 0.7331 | -0.0004 | +0.0212 | +0.0216 |
| [`qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4) | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0192 | 0.0954 | 0.0286 | 0.7278 | -0.0025 | +0.0159 | +0.0184 |

**Gemma-2-9B-IT / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.1423 | 0.0750 | 0.2996 | 0.1292 | 0.7074 | — | — | — |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4) | WSR-LoRA ρ=0.4 | 0.0000 | 0.0077 | 0.0115 | 0.1892 | 0.0521 | 0.6983 | -0.0771 | -0.0091 | +0.0680 |
| [`gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4) | WSR-LoRA ρ=0.5 | 0.0000 | 0.0058 | 0.0096 | 0.1812 | 0.0491 | 0.7180 | -0.0801 | +0.0106 | +0.0907 |

## F. SafeLoRA threshold 0.2 / 0.25 — α=16 (B 의 threshold 확장)

**Llama-2-7B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0269 | 0.3731 | 0.6342 | 0.2586 | 0.3616 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4) | SafeLoRA thr=0.2 | 0.0000 | 0.0212 | 0.2519 | 0.5204 | 0.1984 | 0.3700 | -0.0602 | +0.0084 | +0.0686 |
| [`llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4) | SafeLoRA thr=0.25 | 0.0000 | 0.0019 | 0.1673 | 0.2465 | 0.1039 | 0.3518 | -0.1547 | -0.0098 | +0.1449 |

**Llama-2-13B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0096 | 0.3135 | 0.5404 | 0.2159 | 0.4860 | — | — | — |
| [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4) | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.1538 | 0.2327 | 0.0966 | 0.4875 | -0.1193 | +0.0015 | +0.1208 |
| [`llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4) | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.1538 | 0.1158 | 0.0674 | 0.4655 | -0.1485 | -0.0205 | +0.1280 |

**Llama-3.1-8B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0154 | 0.2173 | 0.0582 | 0.2530 | — | — | — |
| [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.2_a16_lr3e-4) | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.0115 | 0.1885 | 0.0500 | 0.2160 | -0.0082 | -0.0370 | -0.0288 |
| [`llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.25_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.25_a16_lr3e-4) | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0096 | 0.1823 | 0.0480 | 0.2106 | -0.0102 | -0.0424 | -0.0322 |

**Llama-3.2-3B-It / MATH**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lora_math_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0712 | 0.2581 | 0.0823 | 0.2480 | — | — | — |
| [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.2_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.2_a16_lr3e-4) | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.0596 | 0.2423 | 0.0755 | 0.2490 | -0.0068 | +0.0010 | +0.0078 |
| [`llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.25_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.25_a16_lr3e-4) | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0635 | 0.2050 | 0.0671 | 0.2318 | -0.0152 | -0.0162 | -0.0010 |

**Qwen2.5-7B-It / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.0385 | 0.0858 | 0.0311 | 0.7119 | — | — | — |
| [`qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4) | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.0192 | 0.0673 | 0.0216 | 0.7415 | -0.0095 | +0.0296 | +0.0391 |
| [`qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4) | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0173 | 0.0692 | 0.0216 | 0.7559 | -0.0095 | +0.0440 | +0.0535 |

**Gemma-2-9B-IT / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.1423 | 0.0750 | 0.2996 | 0.1292 | 0.7074 | — | — | — |
| [`gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4) | SafeLoRA thr=0.2 | 0.0000 | 0.0038 | 0.0173 | 0.2085 | 0.0574 | 0.6952 | -0.0718 | -0.0122 | +0.0596 |
| [`gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4) | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0154 | 0.2208 | 0.0590 | 0.7096 | -0.0702 | +0.0022 | +0.0724 |

요약: ρ 를 0.5 까지 올리면 Llama-3.2-3B · Gemma-2-9B · Llama-2-13B 에서는 AVG 가 계속 내려가고 task 가 유지되어 Δoverall 이 A 의 최고치(ρ=0.3)를 소폭 넘고, Llama-2-7B · Llama-3.1-8B 에서는 task 가 떨어져 ρ=0.3 근처가 최적이다. SafeLoRA 는 thr=0.2 가 Vanilla LoRA 에 가깝고 thr=0.25 가 B 의 thr=0.3 수준이다. 로그: HarmBench `logs/run_all_2026-09-0{4,5,6,7}_*_summary.csv`, 원표 `logs/revision_sweep/RESULTS_sweep.md`.


# 추가 실험 3 (2026-09-07 ~ 09-08) — MedQA · ARC-C · BeaverTails-GSM8K · 전 기법 α=16

`scripts/revision/run_qa_a16.sh` + `run_bt_then_eval.sh` 로 21셀을 한 번에 학습·평가했다.
**LoRA scaling 을 1.0 으로 통일**한 것이 이 절의 전제다 — r=16, α=16 이므로 여섯 LoRA 기법의
업데이트 예산이 모두 같다. 위 A~F 절 및 본문 표의 α=32 행과는 **직접 비교할 수 없다**.

공통 설정: 출발 모델 CB축 `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` · BT축
`wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv`, epochs 3 · effective batch 16 · max_len 1024 · seed 42 ·
bf16 · cosine, LoRA lr 3e-4 (r16/α16/dropout .05, targets q,k,v,up,down), SEAL 은 full-param lr 5e-5.
안전 데이터는 출발 모델이 안전정렬된 데이터셋을 그대로 쓴다(CB=`circuit_breakers`, BT=`beavertails_cb_train`).
ASR 은 HarmBench keyword 채점(`sys` 조건), Δ 는 각 표의 **Vanilla LoRA(α=16)** 기준이다.

† SEAL 의 S2 는 full-parameter SFT 라 LoRA 예산과 다르다. 같은 표에 두되 예산이 동등한 비교가 아니다.


**G. Llama-2-7B-chat / MedQA (CB 축) — 10,178 샘플 전량**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MedQA | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_medqa_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_medqa_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0019 | 0.2346 | 0.4081 | 0.1612 | 0.4595 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_a16_lr3e-4) | AsFT λ=1.0 | 0.0000 | 0.0000 | 0.0519 | 0.1023 | 0.0386 | 0.3284 | -0.1226 | -0.1311 | -0.0085 |
| [`llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_a16_lr3e-4) | Lisa ρ=1.0 | 0.0000 | 0.0000 | 0.1212 | 0.1573 | 0.0696 | 0.3181 | -0.0916 | -0.1414 | -0.0498 |
| [`llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_medqa_thr0.3_a16_lr3e-4) | SafeLoRA thr=0.3 | 0.0000 | 0.0019 | 0.2038 | 0.3877 | 0.1484 | 0.4595 | -0.0128 | +0.0000 | +0.0128 |
| [`llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_medqa_rs32rt32_a16_lr3e-4) | SaLoRA r_s=32,r_t=32 | 0.0000 | 0.0115 | 0.2923 | 0.6954 | 0.2498 | 0.4211 | +0.0886 | -0.0384 | -0.1270 |
| [`llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_medqa_topp0.8_lr5e-5) | SEAL top-p 0.8 † | 0.0038 | 0.1327 | 0.4558 | 0.7531 | 0.3364 | 0.3574 | +0.1752 | -0.1021 | -0.2773 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_medqa_rho0.3_a16_lr3e-4) | **WSR-LoRA ρ=0.3** | 0.0000 | 0.0000 | 0.1365 | 0.2642 | 0.1002 | 0.4517 | -0.0610 | -0.0078 | +0.0532 |

**H. Llama-2-7B-chat / ARC-C (CB 축)**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | ARC-C | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-lora_arc_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lora_arc_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0000 | 0.0000 | 0.2000 | 0.3623 | 0.1406 | 0.6109 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_a16_lr3e-4) | AsFT λ=1.0 | 0.0000 | 0.0000 | 0.0596 | 0.1808 | 0.0601 | 0.3874 | -0.0805 | -0.2235 | -0.1430 |
| [`llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_a16_lr3e-4) | Lisa ρ=1.0 | 0.0000 | 0.0000 | 0.0942 | 0.2723 | 0.0916 | 0.3157 | -0.0490 | -0.2952 | -0.2462 |
| [`llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-safelora_arc_thr0.3_a16_lr3e-4) | SafeLoRA thr=0.3 | 0.0000 | 0.0000 | 0.1846 | 0.3312 | 0.1290 | 0.6195 | -0.0116 | +0.0086 | +0.0202 |
| [`llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-salora_arc_rs32rt32_a16_lr3e-4) | SaLoRA r_s=32,r_t=32 | 0.0000 | 0.0038 | 0.2135 | 0.3685 | 0.1464 | 0.6297 | +0.0058 | +0.0188 | +0.0130 |
| [`llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-seal_arc_topp0.8_lr5e-5) | SEAL top-p 0.8 † | 0.0000 | 0.0058 | 0.2923 | 0.5354 | 0.2084 | 0.6067 | +0.0678 | -0.0042 | -0.0720 |
| [`llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_arc_rho0.3_a16_lr3e-4) | **WSR-LoRA ρ=0.3** | 0.0000 | 0.0000 | 0.1885 | 0.2969 | 0.1213 | 0.6157 | -0.0193 | +0.0048 | +0.0241 |

**I. Llama-2-7B-chat / GSM8K (BT 축) — 위 「BeaverTails (BV)」 절 326~331행의 재수행분**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-BT_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-lora_gsm8k_a16_lr3e-4) | ▸ Vanilla LoRA (α=16) | 0.0019 | 0.0096 | 0.2192 | 0.3238 | 0.1386 | 0.3662 | — | — | — |
| [`llama2_7b-chat-BT_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT λ=1.0 | 0.0038 | 0.2692 | 0.2635 | 0.2585 | 0.1987 | 0.2115 | +0.0601 | -0.1547 | -0.2148 |
| [`llama2_7b-chat-BT_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | Lisa ρ=1.0 | 0.0000 | 0.0077 | 0.1615 | 0.1519 | 0.0803 | 0.1873 | -0.0583 | -0.1789 | -0.1206 |
| [`llama2_7b-chat-BT_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA thr=0.3 | 0.0019 | 0.0038 | 0.2019 | 0.2731 | 0.1202 | 0.3753 | -0.0184 | +0.0091 | +0.0275 |
| [`llama2_7b-chat-BT_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA r_s=32,r_t=32 | 0.0019 | 0.1827 | 0.3615 | 0.4412 | 0.2468 | 0.3616 | +0.1082 | -0.0046 | -0.1128 |
| [`llama2_7b-chat-BT_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-seal_gsm8k_topp0.8_lr5e-5) | SEAL top-p 0.8 † | 0.0019 | 0.0558 | 0.2750 | 0.2608 | 0.1484 | 0.3904 | +0.0098 | +0.0242 | +0.0144 |
| [`llama2_7b-chat-BT_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-chat-BT_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | **WSR-LoRA ρ=0.3** | 0.0019 | 0.0058 | 0.1865 | 0.2265 | 0.1052 | 0.3692 | -0.0334 | +0.0030 | +0.0364 |

**관측**

- **WSR-LoRA(ρ=0.3) 가 세 설정 모두에서 Pareto 위에 있다.** MedQA 는 Vanilla LoRA 대비
  utility 를 0.0078 만 잃고 AVG 를 0.0610 낮췄고(Δoverall +0.0532), BT/GSM8K 에서는
  utility 가 오히려 높으면서(0.3692 vs 0.3662) AVG 가 가장 낮은 축이다(0.1052).
- **AsFT · Lisa 는 더 안전하지만 utility 를 크게 깎아서 얻은 것이다.** MedQA 에서 AVG 는
  0.0386 / 0.0696 로 가장 낮지만 정확도가 0.3284 / 0.3181 로 Vanilla LoRA(0.4595)보다
  0.13~0.14 낮다. ARC-C 도 같은 패턴이다(0.3874 / 0.3157 vs 0.6109).
- **⚠️ BT/GSM8K 의 AsFT 는 AutoDAN 이 0.2692 로 튄다.** 같은 기법의 CB 축 두 셀은 AutoDAN
  이 모두 0.0000 인데 BT 축만 이렇고, utility 도 0.2115 로 낮다. 위 「AsFT 의 상충하는 원본
  기록」 절에 남아 있는 불안정성과 방향이 같으므로, 이 행을 결론에 쓰기 전에 재학습으로
  재현 여부를 확인할 것.
- **⚠️ BT/GSM8K 의 Lisa(ρ=1.0) 는 utility 가 0.1873 으로 붕괴했다.** CLAUDE.md 에 기록된
  "ρ=1.0 은 GSM8K 0.39→0.17 로 붕괴" 가 그대로 재현됐다. 이 절은 사용자 지정으로 ρ=1.0
  만 돌렸으므로, 보고할 때는 ρ=0.0 쪽도 만들어 양쪽을 함께 제시할 것.
- SEAL 은 CB/MedQA 에서 AVG 0.3364 로 이 절에서 가장 나쁘다. 기존 CB/GSM8K SEAL(0.3045)
  과 같은 경향이다.

**알려진 차이**: MedQA 는 전체 10,178 샘플을 썼다. 논문의 MedQA 행은 10,000 샘플이므로
+1.7% 차이가 있다(`scripts/revision/common.sh` 에 기록된 기존 차이).

로그: HarmBench `logs/run_all_2026-09-08_01-30-49_summary.csv` ·
요약 `HarmBench/results/evaluation_summary_2026-09-08_01-31-04.csv` ·
lm-eval `logs/eval_20260908_040823_gpu0_results.csv`.
학습 로그 `logs/qa_a16_20260907_173959.log` · `logs/bt_then_eval_20260907_200827.log`.

# 추가 실험 4 (2026-09-08) — 논문 Table 3 확장: SEAL + WSR-Tune

논문 Table 3(기존 안전기법 위에 WSR-Tune 을 얹었을 때의 변화)에 **SEAL** 열을 추가한다.
7B 는 이미 쌍이 있었고(`llama2_7b_chat_seal_5e-5` / `..._seal_warp_5e-5`), 13B 는 SEAL 만
있어 이번에 WaRP 팔을 만들었다: `seal/scripts/run_warp_seal_13b.sh`.

**두 팔이 같은 데이터를 본다.** SEAL 은 selector 가 고른 부분집합으로 학습하므로, WaRP 팔이
selector 를 다시 돌리면 (학습은 bitwise 재현되지 않아) 다른 부분집합이 뽑혀 **데이터 차이가
교란요인**이 된다. 그래서 selector 를 재학습하지 않고 기존 13B SEAL 저장소의
`sft_config.json > select_meta` 에 저장돼 있던 인덱스 **5,978/7,473 (top-p 0.8)** 을 그대로
복원해 썼다. 두 팔의 차이는 **WaRP 재매개변수화 하나뿐**이다.

설정(기존 13B SEAL 의 `sft_config.json` 에서 읽어 맞춤): 출발 모델
`wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5` · epochs 3 · lr 5e-5 · wd 0.01 · warmup 0.1 · cosine ·
batch 1×16 · max_len 1024 · seed 42 · bf16 · gradient_checkpointing.
WaRP 는 7B WSR-SEAL 과 동일하게 circuit_breakers 전량(4994) · ρ=0.1 · `--perlayer` ·
layer_type `attn_q,attn_k,attn_v,ffn_up,ffn_down` · target_layers `all`
(적용 모듈 200개, 동결 계수 비율 10.07%, 학습 8.81B / 전체 13.02B).

**Llama-2-13B-chat / GSM8K**

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K |
|---|---|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) | SEAL | 0.0000 | 0.0096 | 0.1519 | 0.2496 | 0.1028 | 0.4519 |
| [`llama2_13b-chat-CB_SSFT-seal-warp_gsm8k_topp0.8_rho0.1_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-seal-warp_gsm8k_topp0.8_rho0.1_lr5e-5) | **SEAL + WSR-Tune** | 0.0000 | 0.0000 | 0.0269 | 0.0131 | **0.0100** | **0.4867** |
| | 변화 | +0.0000 | -0.0096 | -0.1250 | -0.2365 | **-0.0928** | **+0.0348** |

**Llama-2-7B-chat / GSM8K** (기존 쌍, 참고)

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K |
|---|---|---:|---:|---:|---:|---:|---:|
| [`llama2_7b_chat_seal_5e-5`](https://huggingface.co/kmseong/llama2_7b_chat_seal_5e-5) | SEAL | 0.0019 | 0.1077 | 0.4481 | 0.6604 | 0.3045 | 0.3889 |
| [`llama2_7b_chat_seal_warp_5e-5`](https://huggingface.co/kmseong/llama2_7b_chat_seal_warp_5e-5) | SEAL + WSR-Tune | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 0.3829 |

**관측**

- 13B 에서 WSR-Tune 을 얹으면 **안전과 downstream 이 동시에 좋아진다** — AVG 0.1028→0.0100
  (-0.0928), GSM8K 0.4519→0.4867 (+0.0348).
  이는 논문 Table 3 에서 SN-Tune·SafeInstr 에 대해 관찰된 패턴(둘 다 개선)과 같은 방향이다.
- 특히 AutoDAN 이 0.0096→0.0000, PAP 가 0.2496→0.0131 로 크게 떨어졌다.
- **⚠️ 7B WSR-SEAL 의 ASR 은 이 박스에 기록이 없다.** lm-eval 의 GSM8K(0.3829)만
  `lm-evaluation-harness/logs/eval_20260715_122811_results.csv` 에 남아 있고, HarmBench
  요약 CSV 에는 이 모델 행이 없다. 7B 쌍을 표에 실으려면 ASR 을 재측정해야 한다
  (모델은 허브에 있으므로 `run_all_eval.sh kmseong/llama2_7b_chat_seal_warp_5e-5` 한 번이면 된다).

로그: HarmBench `logs/run_all_2026-09-08_14-06-45_summary.csv` ·
학습 `seal/logs/warp_seal_13b_20260908_102907.log` ·
스크립트 `seal/scripts/run_warp_seal_13b.sh`.

## G. 원공간 동결 비율 sweep — 논문 Table 1 의 chat 라인 재현 (safety only)

**2026-09-13 측정.** 재파라미터화 없이(U=V=I) **원래 weight 공간**에서 safety importance 를
재고 상위 ρ 를 얼린 뒤 gsm8k 로 full-param FT 했다. WSR-Tune 과의 차이는 **마스크가 어느
좌표계에서 매겨지는가** 하나뿐이다. 논문 Table 1 은 llama2-7b **base** 모델 실험이고, 이 표는
같은 실험을 **chat 라인**(safety-tuned Llama-2-7B-chat)에서 다시 돌린 것이다.

학습: `scripts/run_origspace_freeze_sweep.sh` — 출발 `kmseong/llama2_7b-chat-Safety-FT-lr5e-5`,
Phase 2 `--original_space_mask` + circuit_breakers 4994(레이어별 quantile),
Phase 3 gsm8k full-param FT, lr 5e-5 · 3ep · eff.batch 16 · wd 0.01 · warmup 0.1 · seed 42 · bf16.
측정: 표 상단의 공통 조건과 동일(sys 모드 · GRADING=hard · AdvBench standard · 4종 공격).
**downstream 은 측정하지 않았다**(`HB_ONLY=1`) — 이 표만으로 Δoverall 을 논하면 안 된다.

| 동결 ρ | 모델 | Direct | AutoDAN | PAIR | PAP | AVG ↓ |
|---|---|---:|---:|---:|---:|---:|
| 0.00 | [`llama2_7b-chat_gsm8k_full_ft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5) † | 0.0000 | 0.0827 | 0.2635 | 0.4850 | 0.2078 |
| 0.05 | [`...origspace-freeze-p05-gsm8k-lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-origspace-freeze-p05-gsm8k-lr5e-5) | 0.0000 | 0.0096 | 0.1481 | 0.2942 | 0.1130 |
| 0.20 | [`...origspace-freeze-p20-gsm8k-lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-origspace-freeze-p20-gsm8k-lr5e-5) | 0.0000 | 0.0019 | 0.1135 | 0.1927 | 0.0770 |
| 0.30 | [`...origspace-freeze-p30-gsm8k-lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-origspace-freeze-p30-gsm8k-lr5e-5) | 0.0000 | 0.0019 | 0.1058 | 0.1700 | 0.0694 |
| 0.40 | [`...origspace-freeze-p40-gsm8k-lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-origspace-freeze-p40-gsm8k-lr5e-5) | 0.0000 | 0.0038 | 0.1019 | 0.1569 | **0.0657** |
| 0.50 | [`...origspace-freeze-p50-gsm8k-lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-origspace-freeze-p50-gsm8k-lr5e-5) | 0.0000 | 0.0038 | 0.1038 | 0.1573 | 0.0662 |
| *0.10* | *[`llama-2-7b-chat-warp-ratio-0.1`](https://huggingface.co/wvnvwn/llama-2-7b-chat-warp-ratio-0.1)* † — **WSR-Tune(재파라미터화)** | 0.0000 | 0.0000 | 0.0962 | 0.1800 | *0.0691* |

† 기존 측정치(같은 출발 모델·같은 동작점·같은 평가 조건). 나머지 5행이 이번에 새로 학습·측정한 것.

**읽는 법 — 세 가지만 말할 수 있다.**

1. **곡선은 ρ≈0.3 에서 포화된다.** 0.2078 → 0.1130 → 0.0770 → 0.0694 → 0.0657 → 0.0662.
   ρ=0.5 는 ρ=0.4 보다 나아지지 않는다. 더 얼려도 더 안전해지지 않는다.
2. **ρ≥0.3 의 원공간 동결은 WSR-Tune 10% 와 같은 안전 수준이다.** 0.0657~0.0694 vs 0.0691 의
   차이는 전부 이 저장소의 재현 오차(`scripts/revision/repro_2026-09/`, 동일 설정 재학습 시
   keyword ASR ±0.05) 안이므로 **우열로 읽으면 안 된다**.
3. **논문 Table 1 의 구도는 chat 라인에서 재현되지 않는다.** base 모델에서는 원공간 50%
   동결(12.85)조차 WSR-Tune 10%(11.32)에 못 미쳤지만, 여기서는 ρ=0.3 에서 이미 동률이다.
   ⚠️ 따라서 **"원공간 마스킹은 비율을 아무리 올려도 WSR-Tune 에 도달하지 못한다"** 는 서술을
   chat 모델 결과에 그대로 쓰면 이 표와 충돌한다. 살아남는 주장은 **예산 효율**이다 —
   같은 안전 수준을 3배 적은 예산(10% vs 30%)으로 달성한다. 다만 그 주장을 완성하려면
   downstream(GSM8K)이 필요하다: ρ=0.3~0.5 는 학습 가능 파라미터가 그만큼 줄어 성능 손실이
   클 것으로 예상되나 **이번에 측정하지 않았다**.

**동결이 실제로 걸렸는지 검증했다.** 원공간 이진 마스크면 mask=1 위치는 출발 모델과
bit-identical 이어야 하지만, bf16 은 가수가 8비트라 **학습된 파라미터도 55% 가 반올림으로
값이 그대로**다. 마스크 없는 파라미터의 일치율을 바닥값 `r` 로 두고 `f = (관측-r)/(1-r)` 로
보정하면: p05 4.46% · p20 19.34% · p30 29.23% · p40 39.14% · p50 48.98% — 요청값과 일치
(오차 −0.5~−1.0%p, ρ 에 비례하는 일관된 과소추정).

로그: HarmBench 요약 `HarmBench/results/evaluation_summary_2026-09-13_21-06-01.csv` ·
학습 `logs/origspace_freeze/sweep_20260913_192410.log`.

## Llama-3.1-8B **Base** / GSM8K  (논문 Table 7 확장 · 전 행 재측정)

2026-09-16 이 박스(edgeai-1, B200)에서 **safety 와 downstream 을 한 배치에서** 측정했다. 출발 모델 [`Llama-3.1-8B-base-SSFT_lr5e-5`](https://huggingface.co/kmseong/Llama-3.1-8B-base-SSFT_lr5e-5) (CB 로 안전정렬된 **base**). full-param 계열은 lr 1e-5, LoRA 계열은 r=16/α=16/lr 3e-4, 공통 3 epoch · effective batch 16 · max_len 1024 · seed 42 · bf16.

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`Llama-3.1-8B-base-SSFT_lr5e-5`](https://huggingface.co/kmseong/Llama-3.1-8B-base-SSFT_lr5e-5) | SSFT 출발모델 (downstream FT 전) | 0.0000 | 0.0000 | 0.0058 | 0.1435 | 0.0373 | 0.0834 | — | — | — |
| [`Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5`](https://huggingface.co/kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5) | Full Params FT | 0.0000 | 0.0019 | 0.0904 | 0.2600 | 0.0881 | 0.5732 | — | — | — |
| [`llama3.1-8b-base-gsm8k-safeinstr-ratio_10p-lr1e-5`](https://huggingface.co/kmseong/llama3.1-8b-base-gsm8k-safeinstr-ratio_10p-lr1e-5) | SafeInstr (10%) | 0.0000 | 0.0000 | 0.0288 | 0.2335 | 0.0656 | 0.5694 | -0.0225 | -0.0038 | +0.0187 |
| [`llama3_1_8b-base-CB_SSFT-resta_gsm8k_gamma0.3_lr1e-5`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-resta_gsm8k_gamma0.3_lr1e-5) | RESTA γ=0.3 † | 0.0000 | 0.0000 | 0.0769 | 0.2500 | 0.0817 | 0.4845 | -0.0063 | -0.0887 | -0.0824 |
| [`llama3.1-8b-base-gsm8k-safedelta-scale0.1-lr1e-5`](https://huggingface.co/kmseong/llama3.1-8b-base-gsm8k-safedelta-scale0.1-lr1e-5) | SafeDelta s=0.1 | 0.0000 | 0.0000 | 0.0077 | 0.1519 | 0.0399 | 0.4132 | -0.0482 | -0.1600 | -0.1118 |
| [`llama3.1-8B_base_gsm8k_ft_freeze_sn_lr1e-5`](https://huggingface.co/kmseong/llama3.1-8B_base_gsm8k_ft_freeze_sn_lr1e-5) | SN-Tune | 0.0904 | 0.2096 | 0.4827 | 0.6912 | 0.3685 | 0.6300 | +0.2804 | +0.0568 | -0.2236 |
| [`llama3.1-8B_base_gsm8k_ft_freeze_rsn_lr1e-5`](https://huggingface.co/kmseong/llama3.1-8B_base_gsm8k_ft_freeze_rsn_lr1e-5) | RSN-Tune | 0.2327 | 0.1212 | 0.4038 | 0.3838 | 0.2854 | 0.6353 | +0.1973 | +0.0621 | -0.1352 |
| [`llama3.1-8b-base-warp-gsm8k-lr1e-5`](https://huggingface.co/kmseong/llama3.1-8b-base-warp-gsm8k-lr1e-5) | WSR-Tune | 0.0000 | 0.0000 | 0.0135 | 0.1650 | 0.0446 | 0.5512 | -0.0435 | -0.0220 | +0.0215 |
| [`llama3_1_8b-base-CB_SSFT-seal_gsm8k_topp0.8_lr1e-5`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-seal_gsm8k_topp0.8_lr1e-5) | SEAL (top-p 0.8) | 0.0000 | 0.0019 | 0.0808 | 0.2500 | 0.0832 | 0.5595 | -0.0049 | -0.0137 | -0.0088 |
| [`llama3_1_8b-base-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-lora_gsm8k_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0115 | 0.8173 | 0.3885 | 0.6812 | 0.4746 | 0.5724 | — | — | — |
| [`llama3_1_8b-base-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT λ=1.0 (α=16) | 0.0000 | 0.0000 | 0.0096 | 0.1562 | 0.0414 | 0.2714 | -0.4332 | -0.3010 | +0.1322 |
| [`llama3_1_8b-base-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | LISA ρ=1.0 (α=16) | 0.0000 | 0.0000 | 0.0019 | 0.1873 | 0.0473 | 0.3025 | -0.4273 | -0.2699 | +0.1574 |
| [`llama3_1_8b-base-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA thr=0.3 (α=16) | 0.0000 | 0.0038 | 0.0731 | 0.3358 | 0.1032 | 0.4837 | -0.3714 | -0.0887 | +0.2827 |
| [`llama3_1_8b-base-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA r_s=r_t=32 (α=16) ‡ | 0.0154 | 0.9327 | 0.2827 | 0.7069 | 0.4844 | 0.5641 | +0.0098 | -0.0083 | -0.0181 |
| [`llama3_1_8b-base-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama3_1_8b-base-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 (α=16) | 0.0000 | 0.0596 | 0.0808 | 0.2750 | 0.1038 | 0.5929 | -0.3708 | +0.0205 | +0.3913 |

**Δ 기준행**: full-param 9행은 **Full Params FT**, LoRA 6행은 **Vanilla LoRA(α=16)**. SSFT 출발모델은 downstream FT 전이라 기준에서 제외했다(GSM8K 0.0834 는 gsm8k 학습을 안 한 값이다).

**측정 조건**은 이 문서의 다른 표와 같다 — HarmBench AdvBench standard · sys 모드 · `GRADING=hard`(keyword) · lm-eval 5-shot **flexible-extract**. base 모델이라 프롬프트는 `Question: {q}\nAnswer:` 이고, 학습 / HarmBench(`llama-2-base`≡`llama-3-base`) / lm-eval(`gsm8k.yaml doc_to_text`) 세 곳이 글자 단위로 같음을 확인했다. AutoDAN/PAIR test case 는 `llama3_1_8b-base` 것을 재사용했다.

### 논문 Table 7 과의 관계

**이 표는 논문 Table 7 의 GSM8K 열과 직접 비교하면 안 된다.** 그 표는 safety 와 downstream 이 **서로 다른 모델**에서 나왔다 — safety 는 lr 1e-5 리포, GSM8K 는 lr 5e-5 리포다(2026-09-16 규명). 위 표는 전부 lr 1e-5 모델 하나에서 둘 다 잰 것이라 **한 모델 = 한 행**이 성립한다.

| 행 | 논문 AVG | 위 표 AVG | 논문 GSM8K (lr5e-5 모델) | 위 표 GSM8K (lr1e-5 모델) |
|---|---:|---:|---:|---:|
| Full Params FT | 8.81 | 8.81 | 42.38 | 57.32 |
| SafeInstr | 6.56 | 6.56 | 32.83 | 56.94 |
| SafeDelta | 3.99 | 3.99 | 10.54 | 41.32 |
| SN-Tune | 36.85 | 36.85 | 48.90 | 63.00 |
| RSN-Tune | 28.54 | 28.54 | 46.02 | 63.53 |
| WSR-Tune | 4.46 | 4.46 | 44.66 | 55.12 |

**ASR 은 소수점까지 그대로 재현된다.** Full Params FT 는 이번에 기존 결과 재사용이 아니라 **새 키로 독립 측정**했는데도 0.0000 / 0.0019 / 0.0904 / 0.2600 으로 논문값(0.00/0.19/9.04/26.00)과 완전히 같았다. GSM8K 만 모델이 달라 큰 차이가 난다.

† **RESTA 는 새로 만든 것이다.** 논문 Resta 행을 만든 리포 `kmseong/llama3.1-8b-base-lr5e-5-gsm8k-resta-gamma0.3` 와 `kmseong/llama3.1-8b-base-gsm8k-resta-gamma0.3-lr1e-5` 가 **둘 다 허브에서 404** 이고(2026-09-16 확인), kmseong 전체에 llama3.1-8b **base** 용 resta 리포가 하나도 없다. 그래서 논문 §4.1 레시피대로 다시 병합했다: `W_resta = W_ft + 0.3·(W_align − W_base)`, W_ft=`Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5`, W_align=`Llama-3.1-8B-base-SSFT_lr5e-5`, W_base=`meta-llama/Llama-3.1-8B` (`scripts/resta_add_safety.py`, 가중치 합 1.000000 = mergekit `linear` normalize 와 동치). **원본과 bit-identical 하다고 보장할 수 없다** — 없어진 리포가 어떤 W_ft 로 만들어졌는지 확인할 방법이 없다. 논문 AVG 8.37 / GSM8K 38.82 와 비교하면 이 재생성본은 AVG 0.0817 / GSM8K 0.4845 다.

‡ SaLoRA 의 AutoDAN 은 keyword 오탐이 섞여 있다(원 0.9327 → 보정 하한 0.7423). 아래 'AutoDAN keyword ASR 의 오탐' 각주 참조.

## Llama-2-7B **Base** / GSM8K  (논문 Table 7 확장 · rebuttal PEFT 9셀)

2026-09-15 학습(다른 세션) · 2026-09-16 이 박스(edgeai-1, B200)에서 평가. 출발 모델 [`llama2_7b-base-CB_SSFT-lr3e-5`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-lr3e-5). LISA 와 WSR-LoRA 는 ρ 두 값이 모두 허브에 있어 둘 다 실었다.

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-base-CB_SSFT-lora_gsm8k_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-lora_gsm8k_a16_lr3e-4) | Vanilla LoRA (α=16) | 0.0692 | 0.7442 | 0.7885 | 0.7292 | 0.5828 | 0.3791 | — | — | — |
| [`llama2_7b-base-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4) | AsFT λ=1.0 (α=16) | 0.0000 | 0.0000 | 0.0731 | 0.3323 | 0.1013 | 0.1713 | -0.4814 | -0.2078 | +0.2736 |
| [`llama2_7b-base-CB_SSFT-lisa_gsm8k_rho0.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-lisa_gsm8k_rho0.0_a16_lr3e-4) | LISA ρ=0.0 (α=16) | 0.0000 | 0.0000 | 0.1038 | 0.3192 | 0.1058 | 0.3920 | -0.4770 | +0.0129 | +0.4899 |
| [`llama2_7b-base-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4) | LISA ρ=1.0 (α=16) | 0.0000 | 0.0000 | 0.1096 | 0.2958 | 0.1013 | 0.1706 | -0.4814 | -0.2085 | +0.2729 |
| [`llama2_7b-base-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4) | SafeLoRA thr=0.3 (α=16) | 0.0077 | 0.0596 | 0.5404 | 0.6615 | 0.3173 | 0.3465 | -0.2655 | -0.0326 | +0.2329 |
| [`llama2_7b-base-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4) | SaLoRA r_s=r_t=32 (α=16) † | 0.0635 | 0.9404 | 0.7173 | 0.7531 | 0.6186 | 0.3654 | +0.0358 | -0.0137 | -0.0495 |
| [`llama2_7b-base-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-wsr-lora_gsm8k_rho0.1_a16_lr3e-4) | WSR-LoRA ρ=0.1 (α=16) | 0.0038 | 0.7404 | 0.5192 | 0.5665 | 0.4575 | 0.3783 | -0.1253 | -0.0008 | +0.1245 |
| [`llama2_7b-base-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4) | WSR-LoRA ρ=0.3 (α=16) | 0.0019 | 0.6135 | 0.3442 | 0.5077 | 0.3668 | 0.3723 | -0.2160 | -0.0068 | +0.2092 |
| [`llama2_7b-base-CB_SSFT-seal_gsm8k_topp0.8_lr3e-5`](https://huggingface.co/kmseong/llama2_7b-base-CB_SSFT-seal_gsm8k_topp0.8_lr3e-5) | SEAL top-p 0.8 (full-param, lr 3e-5) | 0.0000 | 0.0135 | 0.2788 | 0.4308 | 0.1808 | 0.3829 | — | — | — |

† SaLoRA 의 AutoDAN 0.9404 는 **오탐이 지배적이다** — 489건 중 348건이 `[PROMPT]` 자리표시자를 그대로 출력한 것이라 보정 하한은 0.2404 다. 위 '각주: AutoDAN keyword ASR 의 오탐' 절 참조.

**측정 조건**은 이 문서의 다른 표와 동일하다 — HarmBench AdvBench standard · sys 모드 · `GRADING=hard`(keyword) · lm-eval 5-shot **flexible-extract**. base 모델이라 프롬프트는 `Question: {q}\nAnswer:` 이고 HarmBench 는 `llama-2-base` 템플릿(= `llama-3-base` 와 동일 문자열)을 쓴다. AutoDAN/PAIR test case 는 `llama2_7b-base` 것을 재사용했다.

**Δ 기준행**은 Vanilla LoRA(α=16). SEAL 은 full-param 이라 기준이 Full FT 여야 하는데 이 라인의 Full FT 행이 없어 Δ 를 비웠다.

**논문 Table 7 의 Llama-2-7B Base 7행은 이 표에 붙이지 않았다.** llama3_1_8b-base 와 달리 그 행들의 HarmBench 결과 파일이 이 박스의 결과 트리에 **없다** — AutoDAN 10개 · PAIR 10개 디렉토리의 모든 조합을 훑어도 논문값(FullFT 18.56 / SafeInstr 16.06 / Resta 12.56 / SafeDelta 11.80 / SN 23.38 / RSN 32.37 / WSR-Tune 11.32)과 일치하는 키가 하나도 없다. 따라서 '기존 결과로 논문값 재현 → 같은 잣대 확인' 을 할 수 없다. 이어 붙이려면 그 7개 모델을 이 박스에서 새로 재야 한다(7 × 4공격 ≈ 3~4시간).

---

# 추가 실험 5 (2026-09-19) — AsFT · Lisa 를 **full-parameter** 로 (16셀)

> **완료.** ASR 16/16 · downstream 16/16. HarmBench rc=0 / lm-eval rc=0, 실패 0건.
> ARC-C 의 Δ 기준행(`kmseong/llama2_7b-chat-arc_ssft_lr5e-5`)만 측정 진행 중이다.
> 마지막 갱신: 2026-09-19

`revisioning_wsr.tex` 의 Table 2(498-505) / Table 4(817-824) 에서 AsFT·Lisa 행이 `0.00`
플레이스홀더로 비어 있다. 두 기법은 **full-parameter 블록**에 놓여 있는데 기존 측정치는 전부
LoRA(α=16/32) 라 그 자리에 넣을 수 없어, 같은 동작점의 full-param 판을 새로 학습했다.

**학습 조건** — 3 epoch · effective batch 16 · max_len 1024 · seed 42 · bf16 · cosine ·
wd 0.01 · warmup 0.1. lr 은 모델별: **gemma2_9b 만 1e-5**, 나머지 5e-5.
출발 모델은 각 모델의 CB safety-tuned 체크포인트. 기법 하이퍼파라미터는
`revisioning_wsr.tex:1089-1090` 과 동일 — **AsFT λ=1.0**, **Lisa ρ=1.0 / align 100 / ft 900**.

**측정 조건**은 이 문서의 다른 표와 같다 — HarmBench · AdvBench standard · **sys 모드** ·
`GRADING=hard`(keyword) · Direct/AutoDAN/PAIR/PAP · lm-eval 5-shot
(GSM8K=flexible-extract / MATH=`hendrycks_math_safe` exact_match / MedQA=acc / ARC=exact_match).
ASR 은 `results/evaluation_summary_2026-09-19_11-46-21.csv` 와 소수점 4자리까지 대조했다.
`Δoverall = Δdown − Δsafe` (클수록 좋음).

⚠️ **재현 오차를 감안해 읽을 것.** 이 저장소는 같은 설정 재학습 시 keyword ASR 이 **±0.05**
움직인다(`scripts/revision/repro_2026-09/`). |Δsafe| 가 0.05 미만인 행은 단일 run 의 차이로
방향을 단정하면 안 된다 — 아래에서 Qwen·Gemma 가 여기 해당한다.

표 재생성: `python scripts/revision/build_exp5_section.py`

## 한눈에 보기 (Δoverall)

| 모델 / 태스크 | AsFT | Lisa |
|---|---:|---:|
| Llama-2-7B / GSM8K | **+0.1238** | -0.0284 |
| Llama-2-13B / GSM8K | **+0.1153** | -0.0103 |
| Llama-3.1-8B / MATH | **+0.1295** | +0.1207 |
| Llama-3.2-3B / MATH | **+0.0358** | -0.0546 |
| Qwen2.5-7B / GSM8K | +0.0560 | +0.0401 |
| Gemma-2-9B / GSM8K | +0.0347 | +0.0046 |

**AsFT 가 6개 모델 전부에서 Lisa 보다 Δoverall 이 높다.** 두 기법의 안전성 개선폭(Δsafe)은
비슷한데 downstream 에서 갈린다 — Lisa 는 안전 데이터로 교대 학습하느라 downstream 스텝이
실질적으로 줄어드는 것으로 보인다(7B GSM8K −0.173, 3B MATH −0.059).

**full-param AsFT 는 LoRA 판과 성질이 다르다.** LoRA AsFT 는 GSM8K 를 0.4117 → 0.1971 로
절반 넘게 잃었지만(이 문서 'Llama-2-7B-chat / GSM8K' 표), full-param 은 0.3882(−0.024) 에
그치고 13B·Qwen·Gemma 에서는 오히려 기준보다 **높다**. 안전성을 downstream 으로 사지 않았다.

### Llama-2-7B-Chat / GSM8K
Δ 기준행: `kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5` — AVG 0.2078 / downstream 0.4117

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0846 | 0.1573 | **0.0605** | 0.3882 | -0.1473 | -0.0235 | +0.1238 |
| [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.1000 | 0.1535 | **0.0634** | 0.2388 | -0.1444 | -0.1729 | -0.0284 |

### Llama-2-13B-Chat / GSM8K
Δ 기준행: `wvnvwn/llama-2-13b-chat-hf-lr5e-5-gsm8k-lr5e-5` — AVG 0.1004 / downstream 0.4594

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0654 | 0.0146 | **0.0200** | 0.4943 | -0.0804 | +0.0349 | +0.1153 |
| [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0365 | 0.0154 | **0.0130** | 0.3616 | -0.0874 | -0.0978 | -0.0103 |

### Llama-3.1-8B-Instruct / MATH
Δ 기준행: `kmseong/llama3_1_8b_instruct_MATH_lr5e-5` — AVG 0.0928 / downstream 0.1212

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0019 | 0.1615 | **0.0409** | 0.1988 | -0.0519 | +0.0776 | +0.1295 |
| [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0019 | 0.0785 | **0.0201** | 0.1692 | -0.0727 | +0.0480 | +0.1207 |

### Llama-3.2-3B-Instruct / MATH
Δ 기준행: `kmseong/llama3_2_3b_instruct_MATH_lr5e-5` — AVG 0.0865 / downstream 0.2152

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0385 | 0.1931 | **0.0579** | 0.2224 | -0.0286 | +0.0072 | +0.0358 |
| [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0577 | 0.2723 | **0.0825** | 0.1566 | -0.0040 | -0.0586 | -0.0546 |

### Qwen2.5-7B-Instruct / GSM8K
Δ 기준행: `wvnvwn/qwen-2.5-7B-Instruct-SSFT-gsm8k-lr5e-5` — AVG 0.0362 / downstream 0.6732

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0115 | 0.0673 | **0.0197** | 0.7127 | -0.0165 | +0.0395 | +0.0559 |
| [`qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0173 | 0.1523 | **0.0424** | 0.7195 | +0.0062 | +0.0463 | +0.0401 |

### Gemma-2-9B-IT / GSM8K
Δ 기준행: `wvnvwn/gemma-2-9b-it-lr3e-5-gsm8k-lr1e-5` — AVG 0.0537 / downstream 0.6975

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr1e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr1e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0077 | 0.2138 | **0.0554** | 0.7339 | +0.0017 | +0.0364 | +0.0347 |
| [`gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr1e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr1e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0077 | 0.2646 | **0.0681** | 0.7165 | +0.0144 | +0.0190 | +0.0046 |

### Llama-2-7B-Chat / MedQA
Δ 기준행: **없음** — medqa full-param 은 lr 3e-5 / 1e-5 만 존재하고 이 라인(lr 5e-5)과 맞는 것이 없다 (사용자 결정 2026-09-19: raw 만 싣는다)

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.0808 | 0.1862 | **0.0667** | 0.4643 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0808 | 0.1688 | **0.0624** | 0.3888 | — | — | — |

### Llama-2-7B-Chat / ARC-C
Δ 기준행: `kmseong/llama2_7b-chat-arc_ssft_lr5e-5` (동작점 일치 확인, **측정 예정**)

| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_fullft_lr5e-5) | AsFT λ=1.0 (full) | 0.0000 | 0.0000 | 0.1173 | 0.2204 | **0.0844** | 0.6391 | — | — | — |
| [`llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_fullft_lr5e-5) | Lisa ρ=1.0 (full) | 0.0000 | 0.0000 | 0.0577 | 0.1088 | **0.0416** | 0.5734 | — | — | — |


## 표를 읽기 전에 — 기준행 검증 (2026-09-19)

AsFT/Lisa 의 downstream 이 기준 Full FT 보다 높은 셀이 6개 중 5개라 기준행을 전수 검증했다.

### ① Llama-3.1-8B / MATH — 기준값은 맞다, 차이는 진짜다

`AsFT 0.1988 / Lisa 0.1692` 가 `기준 0.1212` 보다 높아 의심스러워 네 가지를 확인했다.

| 확인 | 결과 | 판정 |
|---|---|---|
| 기준행을 **이 박스에서 재측정** | MATH **0.1238** (기록 0.1212) · ASR 0.0977 (기록 0.0928) | 기록값 정확. 평가 방식 안 바뀜 |
| lm-eval 설정 대조 | task·n-shot·dataset·`doc_to_text`·filter·gen_kwargs **전부 동일** | 평가 차이 아님 |
| 학습 프롬프트 포맷 | 양쪽 다 `data/math_task_format.py` 공유 → 생성물이 모두 `Final Answer: $...$` | 데이터 포맷 차이 아님 |
| **AsFT·Lisa 재학습** (같은 설정, 경로/리포만 분리) | AsFT 0.1988→**0.1984**, Lisa 0.1692→**0.1668** | 재현 오차 **0.0004~0.0024** |

**결론: +0.075 는 재현 오차의 30배가 넘는 실제 차이다.** 단일 run 노이즈가 아니다.

참고로 **출발 모델**(downstream FT 전)의 MATH 는 **0.0096** 이다. 수학을 못 푸는 게 아니라
CB 안전정렬 탓에 수학 문제를 **거부한다**(`'I cannot provide a solution that involves a
ceiling function.'`). 즉 이 열은 수학 실력보다 "형식을 익혔는가" 를 재고 있고, full FT 는
0.0096 → 0.1238 로 13배 올렸다 — **full FT 가 MATH 를 망가뜨린 것이 아니다.**

생성물을 보면 차이는 순수하게 계산 정확도다. 같은 문제에서 기준 모델은
`6/5·30 = 18`(정답 36) 로 **분수 곱셈을 틀리고**, 우리 모델은 소수로 바꿔 맞힌다.
다만 **왜** 벌점이 계산 능력을 보존하는지는 이 실험이 설명하지 못한다 —
사실만 보고하고 메커니즘 주장은 하지 말 것.

### ② Llama-2-7B / ARC-C — 기준행을 **쓸 수 없다**

`kmseong/llama2_7b-chat-arc_ssft_lr5e-5` 는 동작점이 정확히 일치하지만(base=Safety-FT-lr5e-5,
lr 5e-5, 4×4, wd 0.01, 3ep, cosine) **모델이 망가져 있다**:

| | ASR AVG | ARC 정확도 |
|---|---:|---:|
| 기준행 | **0.6004** | **0.0794** ← 4지선다 무작위(0.25)보다 낮다 |
| 우리 AsFT | 0.0844 | 0.6391 |
| 우리 Lisa | 0.0416 | 0.5734 |

답 대신 프롬프트 지시문을 되뱉는다: `' [the_answer_letter] where the [the_answer_letter]
is one of A, B, C or D'`. ARC 학습 데이터가 1119개뿐인데 lr 5e-5·3epoch 를 돌린 탓으로 보인다.
**우리 모델 둘은 정상**이다(`C`,`B`,`D` 단일 문자 출력 확인).
→ **ARC 의 Δ 는 MedQA 와 같이 비운다.**

### ③ 평가 운영상의 함정 — GPU 잔여 프로세스

`run_all_eval.sh` 가 `rc=0` 으로 정상 종료를 보고한 뒤에도 **자식 프로세스가 GPU 를 붙들고
남는다**(실측: 10개 프로세스 / 59GB). 그 상태에서 다음 평가를 걸면 vLLM 이
`max_model_len=131072` 의 KV cache 를 못 잡아 조용히 죽는다
(`ValueError: 16.0 GiB KV cache is needed ... available 12.14 GiB`).
**평가와 평가 사이에 `nvidia-smi` 로 점유를 확인할 것.**

---

## AsFT: LoRA 판과 Full FT 판은 무엇이 다른가

> AsFT(arXiv:2506.08473)는 **원 논문도 참조 구현도 LoRA 전용**이다. 개정본 Table 2/4 는
> AsFT 를 full-parameter 블록에 싣기 때문에 우리가 일반화해서 새로 구현했다.
> **논문에 실을 때 "우리가 일반화한 full-param 판" 임을 반드시 밝혀야 한다.**

### 아이디어 자체는 한 줄이다

안전정렬이 가중치를 움직인 방향을 `V` 라 하고, 그 방향을 **벗어나는** 업데이트 성분만 때린다.

```
V_l = W_l^aligned − W_l^base          ← "안전정렬이 움직인 방향"
Ĉ_l = V_l V_lᵀ / ‖V_l‖_F              ← 그 방향이 만드는 부분공간

벌점 =  λ · Σ_l ‖ (I − Ĉ_l) · ΔW_l ‖²_F
                  └─ V 에 직교하는(=safety basin 을 벗어나는) 성분만 남긴다
```

두 판의 차이는 **`ΔW` 를 무엇으로 보느냐, 벌점을 어떻게 거느냐** 둘뿐이다.
`V` 와 `Ĉ` 의 정의는 완전히 같다. (`Ĉ` 를 `‖V‖²` 가 아닌 `‖V‖_F` 로 나누므로 엄밀한
사영행렬이 아니다 — SafeLoRA 에서 이어져 온 정의이고 참조 구현대로 두었다.)

### 나란히 보기

| | **LoRA 판** (기존) | **Full FT 판** (이번에 구현) |
|---|---|---|
| ΔW | `B A` (어댑터 곱) | `W − W₀` (W₀ = 출발 모델 가중치) |
| 스케일 `s=α/r` | **곱하지 않음** (참조 구현 그대로. λ=1.0 이 이 정의 위에서 정해진 값) | 해당 없음 |
| 초기 상태 | `B=0` → 벌점 0 | `W=W₀` → ΔW=0 → 벌점 0 (같은 초기조건) |
| 벌점을 거는 법 | **loss 에 더하고 autograd** | **그래디언트를 직접 계산해 `param.grad` 에 누적** |
| 학습 대상 | 어댑터만 | 전 파라미터 |
| 구현 | `models/asft_baseline.AsFTRegularizer` | `models/asft_baseline.AsFTFullParamRegularizer` |
| 러너 | `finetune_gsm8k_lora.py --method asft` | `asft/finetune_asft_full.py` |

### 왜 LoRA 의 방법을 그대로 못 쓰는가

`Ĉ` 는 out×out 이다. `up_proj` 면 11008² 라 fp32 로 32층이면 **15GB 가 넘는다.**
그래서 LoRA 판은 `Ĉ` 를 만들지 않고 `V` 만 들고 다음 항등식을 쓴다:

```
X = (I−Ĉ)B = B − V(VᵀB)/‖V‖_F        (out × r)
‖X A‖²_F = trace( (XᵀX)(AAᵀ) )        ← r=16 이라 r×r 두 개의 곱으로 끝
```

**이 트릭은 `A` 가 저랭크일 때만 성립한다.** full-param 에는 `A` 가 없다.
같은 식을 ΔW=W−W₀ 에 그대로 적용하면 모듈마다 `G=VᵀΔW`(n×n)와 `X=(I−Ĉ)ΔW`(m×n)가
**backward 까지 그래프에 남는다**:

| | 모듈 1개 (fp32) | 32층 합계 |
|---|---:|---:|
| `G` (down_proj, 11008²) | 484 MB | — |
| `X` (down_proj, 4096×11008) | 180 MB | — |
| **합** | **664 MB** | **≈ 21 GB** |

7B 조차 감당이 안 된다. 그래서 다른 방법이 필요했다.

### Full FT 판: 해석적 그래디언트

`Ĉ` 가 대칭이므로 벌점의 그래디언트를 손으로 쓸 수 있다:

```
Y = ΔW − V(VᵀΔW)/‖V‖_F                       ← (I−Ĉ)ΔW
∇ = 2λ (I−Ĉ)² ΔW = 2λ ( Y − V(VᵀY)/‖V‖_F )
```

이걸 `no_grad` 로 계산해 `param.grad` 에 더한다. 모듈 하나를 끝내면 임시버퍼가 바로
해제되므로 피크가 m×n 몇 장으로 끝난다 — **21GB → 수백 MB.**

### 구현에서 반드시 지켜야 할 세 가지

| # | 규칙 | 어기면 |
|---|---|---|
| 1 | `param.grad` 에 **누적**(`add_`), 대입 금지 | SFT 그래디언트가 통째로 날아간다 (SafeGrad 에서 기록된 함정과 동일) |
| 2 | 누적 주기의 **마지막 micro-batch 에서 한 번만** | 벌점은 데이터와 무관 → 매번 더하면 grad_accum 배 과다계상 |
| 3 | `ΔW = W − W₀` 의 **뺄셈은 fp32** | `W ≈ W₀` 라 bf16 으로 빼면 자리수 소실로 ΔW 가 뭉개진다 |

`on_pre_optimizer_step` 콜백을 쓰면 안 된다 — transformers 4.57 기준 **gradient clipping
이후**에 불린다. 참조 구현은 벌점을 loss 에 얹으므로 SFT 항과 **같이** clip 되어야 한다.
`training_step` 은 clip 이전이라 이 조건을 만족한다.

뺄셈 뒤의 행렬곱은 **TF32** 로 돌린다(벌점 구간에서만, 전역 플래그는 즉시 원복).
fp32 로 하면 7B 기준 step 당 2.6초가 붙어 4.24 s/it 였는데 **1.03 s/it** 로 줄었고,
autograd 대조 오차는 3e-06 에 그친다(임계 1e-3). V·W 가 애초에 bf16 저장이라
fp32 의 여분 정밀도는 의미가 없다.

### 검증

| 검증 | 결과 |
|---|---|
| 해석적 그래디언트 vs autograd (실제 7B 학습 첫 step) | `rel_err = 3.0e-06` |
| 같은 검증 (13B) | `rel_err = 2.6e-06` |
| 스모크 4경로 (GPU / CPU offload / fp32 저장 / λ=0) | 전부 통과 |

`--asft_check_equiv` 로 매 학습에서 자동 대조한다. 스모크: `bash asft/scripts/_smoke_asft_full.sh`

### λ=1.0 은 full-param 에서도 작동했다

8셀의 최종 벌점 `Σ‖(I−Ĉ)ΔW‖²_F`:

| 셀 | 최종 벌점 | 셀 | 최종 벌점 |
|---|---:|---|---:|
| llama2_13b / gsm8k | 3.24e-03 | qwen25_7b / gsm8k | 1.12e-02 |
| llama2_7b / medqa | 3.39e-03 | llama31_8b / math | 2.18e-02 |
| llama2_7b / gsm8k | 4.46e-03 | gemma2_9b / gsm8k | 2.56e-02 |
| llama32_3b / math | 9.76e-03 | llama2_7b / arc | 3.05e-02 |

벌점은 학습 초반 SFT loss 와 맞먹는 크기(13B 기준 peak **0.584** vs loss 0.441)까지
올라갔다가 그 그래디언트가 되밀어 3e-3~3e-2 로 내려앉는다.
**최종값이 작다는 것은 제약이 없었다는 뜻이 아니라 충족됐다는 뜻이다** —
모델·태스크가 달라도 같은 범위로 수렴한 것이 근거다.
(⚠️ 끝점만 보고 정규화의 세기를 판단하면 안 된다. 실제로 7B 최종값 4.46e-03 하나만 보고
"λ 가 약하다" 고 오판했다가 13B 의 중간 궤적을 보고 정정했다.)

---

### ⚠️ Lisa 트레이너는 loss 를 grad_accum 으로 나누지 않는다

`LisaTrainer.training_step` 은 `Trainer.training_step` 을 통째로 오버라이드하고
`accelerator.backward(loss)` 를 **나누지 않은 loss** 에 건다. 실측 결과 micro-batch 하나가
벌점을 통째로(= grad_accum 배) 더한다(방향 cos = 1.000000). **이 저장소의 기존 LISA 결과가
전부 이 동작으로 만들어졌으므로** 바꾸지 않고 그대로 재현했다.
자세한 내용: `scripts/revision/SESSION_2026-09-19.md`.


---

# 실험 2 (원공간 동결 스윕 · base 라인) — ⏸ **중단, 재개 대기**

논문 Table 1 의 "FT (X% frozen)" 행을 **base 라인**으로 만드는 실험.
재파라미터화 없이(U=V=I) 원래 weight 공간에서 safety importance 상위 ρ 를 얼리고
GSM8K 로 full-param FT 한다.

**계획 (10셀)**

| 모델 | 출발(SSFT) | ρ=0 기준행 | full lr | 동결 비율 |
|---|---|---|---|---|
| Llama-2-7B base | `kmseong/llama2_7b-base-CB_SSFT-lr3e-5` | `kmseong/llama2_7b-base-gsm8k_ssft_lr3e-5` | 3e-5 | 10/20/30/40/50% |
| Llama-3.1-8B base | `kmseong/Llama-3.1-8B-base-SSFT_lr5e-5` | `kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5` | 1e-5 | 10/20/30/40/50% |

**진행 상태 (2026-09-19 중단 시점)**

| 셀 | 상태 |
|---|---|
| llama2_7b_base / p10 | ✅ 학습·업로드·검증 완료 (`kmseong/llama2_7b-base-origspace-freeze-p10-gsm8k-lr3e-5`) |
| llama2_7b_base / p20 | ⏸ 학습 중 중단 (`.done` 없음 → 재개 시 처음부터) |
| 나머지 8셀 | 대기 |

**평가는 아직 하지 않았다** — 수치 없음.

**중단 사유**: AsFT/Lisa 의 학습 범위가 WSR-Tune 과 다르다는 것이 확인돼(아래 절), 그 재학습을
먼저 하기로 했다. 재개: `bash scripts/revision/31_origspace_base.sh` (`.done` 마커로 이어간다).
셀당 약 15분(Phase2 1.5분 + Phase3 12분 + 업로드 2분), 10셀 ≈ 2시간 30분.

⚠️ 평가 시 **ρ=0 기준행과 원본 base·SSFT 모델도 이 박스에서 같이 재측정**할 것.
실험 1 에서 MATH·ARC 기준행이 구 박스 측정값이라 겪은 문제와 같은 상황이다.

---

# 추가 실험 6 (2026-09-19) — **학습 범위를 WSR-Tune 과 맞춘** AsFT · Lisa (8셀)

## 왜 다시 했는가 — 이전 비교는 불공정했다

WSR-Tune 은 `basis_coeff` 만 학습한다(`models/phase3_extra_learning.py:2053-2056` 에서
나머지를 전부 `requires_grad=False`). `basis_coeff` 는 변환된 5개 projection
(q,k,v,up,down)에만 있으므로 **실질적으로 그 5개만 학습**한다.

반면 추가 실험 5 의 AsFT·Lisa 는 `fine_tuning_type: Full Parameter` 로 **전 파라미터**를
학습했다. AsFT 는 한술 더 떠서 벌점조차 `target_modules`(5개)에만 걸리므로,
**o_proj / gate_proj / embed_tokens / lm_head 를 아무 제약 없이 자유롭게 바꿨다.**

| | 학습 범위 | 제약 범위 |
|---|---|---|
| WSR-Tune | q,k,v,up,down | q,k,v,up,down |
| AsFT (실험 5) | **전체** | q,k,v,up,down |
| Lisa (실험 5) | **전체** | 전체 |
| AsFT·Lisa (실험 6) | q,k,v,up,down | 〃 |

두 러너에 `--train_only_targets` 를 추가해 맞췄다. 작은 모델로 검증한 결과 AsFT·Lisa 가
**정확히 같은 파라미터 집합**을 남긴다(학습 57,344 / 동결 4,120,896). 실제 모델에서는
Llama-2-7B 66.73% / 13B 67.67% / 3.1-8B 56.83% / 3.2-3B 57.57% 가 학습된다
(Llama-3 계열이 낮은 것은 vocab 128k 라 embedding·lm_head 비중이 크기 때문).

## 결과 — 같은 학습 범위에서 WSR-Tune 이 4셀 중 3셀 1위

### Llama-2-7B-Chat / GSM8K
기준 **Full Params FT**: ASR 0.2078 / downstream 0.4117

| 기법 | 모델 | 학습 범위 | JB AVG ↓ | down ↑ | Δsafe | Δdown | **Δoverall** |
|---|---|---|---:|---:|---:|---:|---:|
| **WSR-Tune** | [`llama-2-7b-chat-warp-ratio-0.1`](https://huggingface.co/wvnvwn/llama-2-7b-chat-warp-ratio-0.1) | q,k,v,up,down | 0.0691 | 0.3899 | -0.1387 | -0.0218 | **+0.1169** ★ |
| **AsFT** | [`llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0624 | 0.2252 | -0.1454 | -0.1865 | **-0.0411** |
| **Lisa** | [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0548 | 0.2411 | -0.1530 | -0.1706 | **-0.0176** |
| *AsFT (참고)* | [`llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5) | *전체* | *0.0605* | *0.3882* | *-0.1473* | *-0.0235* | *+0.1238* |
| *Lisa (참고)* | [`llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5) | *전체* | *0.0634* | *0.2388* | *-0.1444* | *-0.1729* | *-0.0285* |

### Llama-2-13B-Chat / GSM8K
기준 **Full Params FT**: ASR 0.1004 / downstream 0.4594

| 기법 | 모델 | 학습 범위 | JB AVG ↓ | down ↑ | Δsafe | Δdown | **Δoverall** |
|---|---|---|---:|---:|---:|---:|---:|
| **WSR-Tune** | [`llama-2-13b-chat-hf-WaRP-lr5e-5`](https://huggingface.co/wvnvwn/llama-2-13b-chat-hf-WaRP-lr5e-5) | q,k,v,up,down | 0.0135 | 0.4958 | -0.0869 | +0.0364 | **+0.1233** ★ |
| **AsFT** | [`llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0120 | 0.3161 | -0.0884 | -0.1433 | **-0.0549** |
| **Lisa** | [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0195 | 0.3745 | -0.0809 | -0.0849 | **-0.0040** |
| *AsFT (참고)* | [`llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5) | *전체* | *0.0200* | *0.4943* | *-0.0804* | *+0.0349* | *+0.1153* |
| *Lisa (참고)* | [`llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5) | *전체* | *0.0130* | *0.3616* | *-0.0874* | *-0.0978* | *-0.0104* |

### Llama-3.1-8B-Instruct / MATH
기준 **Full Params FT**: ASR 0.0928 / downstream 0.1238

| 기법 | 모델 | 학습 범위 | JB AVG ↓ | down ↑ | Δsafe | Δdown | **Δoverall** |
|---|---|---|---:|---:|---:|---:|---:|
| **WSR-Tune** | [`llama3.1_8b_instruct-MATH-WaRP-lr5e-5`](https://huggingface.co/kmseong/llama3.1_8b_instruct-MATH-WaRP-lr5e-5) | q,k,v,up,down | 0.0562 | 0.1370 | -0.0366 | +0.0132 | **+0.0498** |
| **AsFT** | [`llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0373 | 0.1476 | -0.0555 | +0.0238 | **+0.0793** |
| **Lisa** | [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0309 | 0.1950 | -0.0619 | +0.0712 | **+0.1331** ★ |
| *AsFT (참고)* | [`llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5) | *전체* | *0.0409* | *0.1988* | *-0.0519* | *+0.0750* | *+0.1269* |
| *Lisa (참고)* | [`llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5) | *전체* | *0.0201* | *0.1692* | *-0.0727* | *+0.0454* | *+0.1181* |

### Llama-3.2-3B-Instruct / MATH
기준 **Full Params FT**: ASR 0.0865 / downstream 0.2152

| 기법 | 모델 | 학습 범위 | JB AVG ↓ | down ↑ | Δsafe | Δdown | **Δoverall** |
|---|---|---|---:|---:|---:|---:|---:|
| **WSR-Tune** | [`llama3_2_3b-instruct-WaRP_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-WaRP_lr5e-5) | q,k,v,up,down | 0.0640 | 0.2238 | -0.0225 | +0.0086 | **+0.0311** ★ |
| **AsFT** | [`llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0766 | 0.1074 | -0.0099 | -0.1078 | **-0.0979** |
| **Lisa** | [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5_tgtonly`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5_tgtonly) | q,k,v,up,down | 0.0775 | 0.1448 | -0.0090 | -0.0704 | **-0.0614** |
| *AsFT (참고)* | [`llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5) | *전체* | *0.0579* | *0.2224* | *-0.0286* | *+0.0072* | *+0.0358* |
| *Lisa (참고)* | [`llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5`](https://huggingface.co/kmseong/llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5) | *전체* | *0.0825* | *0.1566* | *-0.0040* | *-0.0586* | *-0.0546* |

## 읽는 법

**안전성은 세 기법이 대체로 구별되지 않는다.** Δsafe 차이가 대부분 재현 오차(±0.05) 안쪽이다.
갈리는 것은 **downstream 보존력**이다 — 같은 파라미터 예산에서 WSR-Tune 이 훨씬 잘 지킨다:

| 셀 | WSR-Tune | AsFT | Lisa |
|---|---:|---:|---:|
| 7B GSM8K | **0.3899** | 0.2252 | 0.2411 |
| 13B GSM8K | **0.4958** | 0.3161 | 0.3745 |
| 3B MATH | **0.2238** | 0.1074 | 0.1448 |
| 8B MATH | 0.1370 | 0.1476 | **0.1950** |

**예외: Llama-3.1-8B / MATH 한 셀에서 Lisa 가 1위다**(+0.1331 vs WSR-Tune +0.0498).
원인 미상이다. "기준 downstream 이 낮은 셀이라" 는 가설을 세웠으나, 기준이 더 높은
3.2-3B/MATH(0.2152 > 0.1238)에서 WSR-Tune 이 이겼으므로 **반증됐다.**
⚠️ 이 셀을 빼고 보고하면 cherry-picking 이다. 4셀 전부 싣고 예외를 각주로 남길 것.

**전체학습과 비교하면 AsFT 의 이전 우위가 어디서 왔는지 드러난다.** 7B GSM8K 에서
AsFT 는 전체학습 +0.1238 → 타깃한정 **-0.0411** 로 0.165 폭락한다. 추가 파라미터
(o_proj·gate_proj·embed·lm_head)를 제약 없이 학습한 덕이었다. 13B 에서도 같다
(+0.1153 → -0.0549). **논문에는 두 설정을 모두 싣는 것이 안전하다** — "전 파라미터를
주면 AsFT 도 대등하지만, 같은 예산에서는 크게 뒤진다" 가 더 강한 주장이고,
baseline 을 약화시켰다는 의심도 차단한다.

## ⚠️ 이 표의 한계

**WSR-Tune 수치는 기존 측정값 재사용이고, AsFT·Lisa 는 2026-09-19 신규 측정이다.**
같은 측정 조건(sys 모드 · GRADING=hard · lm-eval 5-shot)이지만 측정 시점이 다르다.
같은 배치의 Llama-3.1-8B Full FT 기준행을 오늘 재측정했을 때 MATH 0.1212 → 0.1238
(차이 0.0026)로 재현됐으므로 신뢰할 만하지만, **완전히 같은 잣대로 만들려면 WSR-Tune
4개 모델도 오늘 파이프라인으로 재측정해야 한다**(약 40분).

리포: `kmseong/{model}-CB_SSFT-{asft_*_lambda1.0|lisa_*_rho1.0}_fullft_lr5e-5_tgtonly`

마지막 갱신: 2026-09-19
