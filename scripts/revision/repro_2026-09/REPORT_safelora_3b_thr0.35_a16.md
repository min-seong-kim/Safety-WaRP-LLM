# 재현 실험 — Llama-3.2-3B SafeLoRA thr0.35 α16 (CB / MATH), 2026-09-04, aigpu0317 GPU 0

원본: `kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4` (2026-09-03, transformers 4.57.3)
재현: `kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4_repro` (private, transformers 5.13.0)
학습 명령: `finetune_gsm8k_lora.py --method safe_lora`, r16/α16/drop0.05, lr 3e-4, 3ep, micro 8×accum 2, thr 0.35, seed 42
학습 시간 39분 (train_loss 0.5571), SafeLoRA 투영 40/140층 (cos 0.169~0.711)

| 모델 | Direct | AutoDAN | PAIR | PAP | AVG | MATH |
|---|---|---|---|---|---|---|
| 원본 기록 (RESULTS.md) | 0.0000 | 0.0000 | 0.0577 | 0.1858 | 0.0609 | 0.2214 |
| 원본 재측정 (이 박스) | 0.0000 | 0.0000 | 0.0577 | 0.1873 | 0.0612 | 0.2234 |
| 재현 모델 (이 박스) | 0.0019 | 0.0481 | 0.0788 | 0.1504 | 0.0698 | 0.2132 |

가중치 비교 (aligned `llama3_2_3b-instruct-SSFT-lr5e-5` 기준): ‖Δ_orig‖ 21.90 vs ‖Δ_repro‖ 21.83, 건드린 모듈·미변경 텐서 수(114) 동일,
cos(Δ_orig, Δ_repro)=0.53, ‖Δ_orig−Δ_repro‖/‖Δ_orig‖=0.97 → 같은 레시피, 다른 학습 궤적.

결론: 평가 파이프라인은 정확히 재현됨(원본 재측정이 기록과 일치). 학습은 레시피·규모는 같으나 bitwise 재현은 안 되며,
SafeLoRA의 run-to-run 변동으로 AutoDAN +0.048, PAIR +0.021, PAP −0.037, MATH −0.010 차이가 남.

문제/조치: hb env(transformers 5.13)가 쓴 tokenizer_class=TokenizersBackend 를 harmbench env(4.57)가 못 읽음 → tokenizer_config.json 을
PreTrainedTokenizerFast 로 패치(로컬·허브), harmbench env 에서 토큰 id 일치 확인.
로그: HarmBench logs/run_all_2026-09-04_09-15-21_summary.csv, results/evaluation_summary_2026-09-04_09-15-24.csv, lm-eval logs/eval_20260904_09*_gpu0_results.csv

## 추가 (11:30) — 2차(hb_repro env, transformers 4.57.3) / 3차(같은 env 재실행)

| 모델 | Direct | AutoDAN | PAIR | PAP | AVG | MATH |
|---|---|---|---|---|---|---|
| 원본 재측정 | 0.0000 | 0.0000 | 0.0577 | 0.1873 | 0.0612 | 0.2234 |
| 1차 (tf 5.13) | 0.0019 | 0.0481 | 0.0788 | 0.1504 | 0.0698 | 0.2132 |
| 2차 (tf 4.57.3) | 0.0000 | 0.0000 | 0.0577 | 0.1904 | 0.0620 | 0.2292 |
| 3차 (tf 4.57.3 재실행) | 평가 안 함 | | | | | |

Δ 코사인: orig–1차 0.533, orig–2차 0.529, orig–3차 0.536, 1차–2차 0.556, 1차–3차 0.556, **2차–3차 0.543**(같은 env·GPU·seed).
투영 층 수: 1차 40 / 2차 38 / 3차 41 (140 중). ‖Δ‖ 21.9 / 21.8 / 21.6 / 21.8.
결론: 같은 환경에서 seed 고정 재실행도 cos 0.54 → 학습 궤적 차이는 라이브러리·GPU 차이가 아니라 학습 자체의 비결정성
(bf16 비결정 커널/GPU RNG). ASR 은 run 간 ±0.05 안에서 흔들리며(1차 AutoDAN 0.048), 2차는 원본과 거의 일치.
