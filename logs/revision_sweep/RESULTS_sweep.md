# Sweep 결과 (2026-09-04~, hb_repro env: torch 2.10 / transformers 4.57.3 / peft 0.18.1, GPU 0)
WSR-LoRA ρ=0.4/0.5 (α16) · SafeLoRA thr=0.2/0.25 (α16). HarmBench sys ASR(keyword), 참조 = 같은 모델의 Vanilla LoRA α=16.
Δsafe = AVG − ref AVG, Δdown = task − ref task, Δoverall = Δdown − Δsafe.

## Llama-3.2-3B-it / MATH  (ref Vanilla LoRA α16: AVG 0.0823, MATH 0.2480)
| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.4_a16_lr3e-4` | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0500 | 0.2150 | 0.0663 | 0.2538 | -0.0160 | +0.0058 | +0.0218 |
| `llama3_2_3b-instruct-CB_SSFT-wsr-lora_math_rho0.5_a16_lr3e-4` | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0481 | 0.2008 | 0.0622 | 0.2568 | -0.0201 | +0.0088 | +0.0289 |
| `llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.2_a16_lr3e-4` | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.0596 | 0.2423 | 0.0755 | 0.2490 | -0.0068 | +0.0010 | +0.0078 |
| `llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.25_a16_lr3e-4` | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0635 | 0.2050 | 0.0671 | 0.2318 | -0.0152 | -0.0162 | -0.0010 |
(기존 행 참고: ρ0.05/0.1/0.2/0.3 AVG 0.0752/0.0687/0.0709/0.0676, MATH 0.2468/0.2468/0.2584/0.2568 · thr0.3/0.35 AVG 0.0626/0.0609, MATH 0.2304/0.2214)

## Llama-2-7B-chat / GSM8K  (ref Vanilla LoRA α16: AVG 0.2586, GSM8K 0.3616) — HarmBench util 0.85 (기존 7B 행은 0.95)
| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4` | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.1096 | 0.1500 | 0.0649 | 0.3161 | -0.1937 | -0.0455 | +0.1482 |
| `llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4` | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.1058 | 0.1554 | 0.0653 | 0.2896 | -0.1933 | -0.0720 | +0.1213 |
| `llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4` | SafeLoRA thr=0.2 | 0.0000 | 0.0212 | 0.2519 | 0.5204 | 0.1984 | 0.3700 | -0.0602 | +0.0084 | +0.0686 |
| `llama2_7b-chat-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4` | SafeLoRA thr=0.25 | 0.0000 | 0.0019 | 0.1673 | 0.2465 | 0.1039 | 0.3518 | -0.1547 | -0.0098 | +0.1449 |
(ρ0.4 Direct 는 09-07 gap-fill 에서 측정(0.0000). 기존 행: ρ0.1/0.2/0.3 AVG 0.1277/0.1034/0.0742, GSM8K 0.3707/0.3381/0.3245 · thr0.3/0.35 AVG 0.0827/0.0770, GSM8K 0.3108/0.2949)

## Llama-3.1-8B-it / MATH  (ref Vanilla LoRA α16: AVG 0.0582, MATH 0.2530)
| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | MATH | Δsafe | Δdown | Δoverall |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.4_a16_lr3e-4` | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0038 | 0.1738 | 0.0444 | 0.2236 | -0.0138 | -0.0294 | -0.0156 |
| `llama3_1_8b-instruct-CB_SSFT-wsr-lora_math_rho0.5_a16_lr3e-4` | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0058 | 0.1662 | 0.0430 | 0.2332 | -0.0152 | -0.0198 | -0.0046 |
| `llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.2_a16_lr3e-4` | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.0115 | 0.1885 | 0.0500 | 0.2160 | -0.0082 | -0.0370 | -0.0288 |
| `llama3_1_8b-instruct-CB_SSFT-safelora_math_thr0.25_a16_lr3e-4` | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0096 | 0.1823 | 0.0480 | 0.2106 | -0.0102 | -0.0424 | -0.0322 |
(기존 행: ρ0.05/0.1/0.2/0.3 AVG 0.0449/0.0427/0.0447/0.0418, MATH 0.2202/0.2240/0.2394/0.2400 · thr0.3/0.35 AVG 0.0394/0.0433, MATH 0.2204/0.2126)

## Qwen2.5-7B-it / GSM8K  (ref Vanilla LoRA α16: AVG 0.0311, GSM8K 0.7119)
| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4` | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0192 | 0.1035 | 0.0307 | 0.7331 | -0.0004 | +0.0212 | +0.0216 |
| `qwen2_5_7b-instruct-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4` | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0192 | 0.0954 | 0.0286 | 0.7278 | -0.0025 | +0.0159 | +0.0184 |
| `qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4` | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.0192 | 0.0673 | 0.0216 | 0.7415 | -0.0095 | +0.0296 | +0.0391 |
| `qwen2_5_7b-instruct-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4` | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0173 | 0.0692 | 0.0216 | 0.7559 | -0.0095 | +0.0440 | +0.0535 |
(기존 행: ρ0.05/0.1/0.2/0.3 AVG 0.0320/0.0319/0.0294/0.0300, GSM8K 0.7074/0.7271/0.7036/0.7354 · thr0.3 AVG 0.0223, GSM8K 0.7374)

## Gemma-2-9B-it / GSM8K  (ref Vanilla LoRA α16: AVG 0.1292, GSM8K 0.7074)
| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4` | WSR-LoRA ρ=0.4 | 0.0000 | 0.0077 | 0.0115 | 0.1892 | 0.0521 | 0.6983 | -0.0771 | -0.0091 | +0.0680 |
| `gemma2_9b-it-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4` | WSR-LoRA ρ=0.5 | 0.0000 | 0.0058 | 0.0096 | 0.1812 | 0.0491 | 0.7180 | -0.0801 | +0.0106 | +0.0907 |
| `gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4` | SafeLoRA thr=0.2 | 0.0000 | 0.0038 | 0.0173 | 0.2085 | 0.0574 | 0.6952 | -0.0718 | -0.0122 | +0.0596 |
| `gemma2_9b-it-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4` | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.0154 | 0.2208 | 0.0590 | 0.7096 | -0.0702 | +0.0022 | +0.0724 |
(기존 행: ρ0.05/0.1/0.2/0.3 AVG 0.0814/0.0748/0.0611/0.0520, GSM8K 0.6793/0.6884/0.6907/0.7081 · thr0.3 AVG 0.0494, GSM8K 0.6929)

## Llama-2-13B-chat / GSM8K  (ref Vanilla LoRA α16: AVG 0.2159, GSM8K 0.4860)
| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.4_a16_lr3e-4` | WSR-LoRA ρ=0.4 | 0.0000 | 0.0000 | 0.0769 | 0.0281 | 0.0262 | 0.4435 | -0.1897 | -0.0425 | +0.1472 |
| `llama2_13b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.5_a16_lr3e-4` | WSR-LoRA ρ=0.5 | 0.0000 | 0.0000 | 0.0654 | 0.0231 | 0.0221 | 0.4420 | -0.1938 | -0.0440 | +0.1498 |
| `llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.2_a16_lr3e-4` | SafeLoRA thr=0.2 | 0.0000 | 0.0000 | 0.1538 | 0.2327 | 0.0966 | 0.4875 | -0.1193 | +0.0015 | +0.1208 |
| `llama2_13b-chat-CB_SSFT-safelora_gsm8k_thr0.25_a16_lr3e-4` | SafeLoRA thr=0.25 | 0.0000 | 0.0000 | 0.1538 | 0.1158 | 0.0674 | 0.4655 | -0.1485 | -0.0205 | +0.1280 |
(기존 행: ρ0.05/0.1/0.2/0.3 AVG 0.0991/0.0791/0.0416/0.0300, GSM8K 0.4723/0.4466/0.4549/0.4481 · thr0.3/0.35 AVG 0.0364/0.0274, GSM8K 0.4230/0.3958)
측정 조건: hb_repro env 학습, HarmBench util = family 기본값(Qwen/gemma 0.6, 13B 0.7), lm-eval GSM8K cap 0.85 (원본과 동일). 2026-09-06 14:21 배치.


---
전체 완료 2026-09-07 03:13. 24셀 학습·업로드, 96 HarmBench 조합, 24 lm-eval 모두 완료(실패 0). 원본 sweep 대비 편차: llama2_7b HarmBench util 0.85(원본 0.95). WSR-LoRA importance 배치 등 하이퍼파라미터 변경 없음.
