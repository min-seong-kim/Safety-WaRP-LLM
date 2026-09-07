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
| [`gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5`](https://huggingface.co/kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5) | SEAL | 0.4500 | 0.2558 | 0.8365 | 0.8354 | 0.5944 | 0.1865 | +0.5407 | -0.5110 | -1.0517 |
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

## 추가 BV LoRA learning-rate 기록 (주표와 별개)

최종 rebuttal에서 선택한 행과 혼동하지 않도록 나머지 명시적인 baseline LR 실험을 별도로 보존한다. `matched-r16-a32` 모델의 rank/alpha는 주소상 r=16/α=32이다. WSR-LoRA의 rank/alpha는 해당 주소에 없어 추정하지 않았다.

| 모델 주소 | 메소드 | Direct | AutoDAN | PAIR | PAP | ASR AVG | Task acc (GSM8K) |
|---|---|---:|---:|---:|---:|---:|---:|
| [kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr1e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr1e-4-beavertails) | Vanilla LoRA (lr=1e-4) (원본 L53) | 0.0019 | 0.0019 | 0.1865 | 0.2031 | 0.0983 | 0.3313 |
| [kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr2e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr2e-4-beavertails) | Vanilla LoRA (lr=2e-4) (원본 L54) | 0.0019 | 0.0173 | 0.2212 | 0.3192 | 0.1399 | 0.3745 |
| [kmseong/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr1e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr1e-4-beavertails) | SaLoRA (이전 matched; lr=1e-4) (원본 L56) | 0.0019 | 0.0231 | 0.1712 | 0.1915 | 0.0969 | 0.3245 |
| [kmseong/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr2e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr2e-4-beavertails) | SaLoRA (이전 matched; lr=2e-4) (원본 L57) | 0.0019 | 0.1923 | 0.5000 | 0.6665 | 0.3402 | 0.3897 |
| [kmseong/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr3e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr3e-4-beavertails) | SaLoRA (이전 matched; lr=3e-4) (원본 L58) | 0.0423 | 0.3827 | 0.7288 | 0.8196 | 0.4933 | 0.3351 |
| [kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr1e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr1e-4-beavertails) | SafeLoRA (lr=1e-4) (원본 L59) | 0.0019 | 0.0019 | 0.1769 | 0.1919 | 0.0931 | 0.3563 |
| [kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr2e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr2e-4-beavertails) | SafeLoRA (lr=2e-4) (원본 L60) | 0.0019 | 0.0115 | 0.2115 | 0.2992 | 0.1310 | 0.3654 |
| [kmseong/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr1e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr1e-4-beavertails) | Lisa (lr=1e-4) (원본 L62) | 0.1788 | 0.0885 | 0.6365 | 0.4142 | 0.3295 | 0.1948 |
| [kmseong/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr2e-4-beavertails](https://huggingface.co/kmseong/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr2e-4-beavertails) | Lisa (lr=2e-4) (원본 L63) | 0.0000 | 0.0000 | 0.1058 | 0.1127 | 0.0546 | 0.2009 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr1e-4-bvasft](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr1e-4-bvasft) | AsFT (lr=1e-4) (원본 L93) | 0.0000 | 0.0019 | 0.1538 | 0.1696 | 0.0813 | 0.2547 |
| [wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr2e-4-bvasft](https://huggingface.co/wvnvwn/llama2-7b-chat-lr5e-5-gsm8k-lr2e-4-bvasft) | AsFT (lr=2e-4) (원본 L94) | 0.0000 | 0.0038 | 0.1904 | 0.1708 | 0.0912 | 0.2585 |
| 없음 (namespace 미기재; `llama2-7b-chat-lr5e-5-gsm8k-lr1e-4-bvwsr-4994`) | WSR-LoRA (4994; lr=1e-4) (원본 L76) | 0.0000 | 0.0019 | 0.1615 | 0.1904 | 0.0885 | 0.3177 |
| 없음 (namespace 미기재; `llama2-7b-chat-lr5e-5-gsm8k-lr2e-4-bvwsr-4994`) | WSR-LoRA (4994; lr=2e-4) (원본 L77) | 0.0019 | 0.0212 | 0.1635 | 0.2396 | 0.1066 | 0.3851 |


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
