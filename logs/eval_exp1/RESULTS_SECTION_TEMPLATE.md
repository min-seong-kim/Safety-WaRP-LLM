# 추가 실험 5 (2026-09-19) — AsFT · Lisa 를 **full-parameter** 로 (16셀)

`revisioning_wsr.tex` 의 Table 2(498-505) / Table 4(817-824) 에서 AsFT·Lisa 행이 `0.00`
플레이스홀더로 비어 있다. 두 기법은 **full-parameter 블록**에 놓여 있는데 기존 측정치는 전부
LoRA(α=16/32) 라 그 자리에 넣을 수 없어, 같은 동작점의 full-param 판을 새로 학습했다.

**학습 조건** — 3 epoch · effective batch 16 · max_len 1024 · seed 42 · bf16 · cosine ·
wd 0.01 · warmup 0.1. lr 은 모델별(common.sh): **gemma2_9b 만 1e-5**, 나머지 5e-5.
출발 모델은 각 모델의 CB(circuit_breakers) safety-tuned 체크포인트.
기법 하이퍼파라미터는 `revisioning_wsr.tex:1089-1090` 과 동일 — **AsFT λ=1.0**,
**Lisa ρ=1.0 / alignment_step 100 / finetune_step 900**.

**측정 조건**은 이 문서의 다른 표와 같다 — HarmBench · AdvBench standard · **sys 모드** ·
`GRADING=hard`(keyword) · Direct/AutoDAN/PAIR/PAP · lm-eval 5-shot
(GSM8K=flexible-extract / MATH=`hendrycks_math_safe` / MedQA=`medqa_4options` / ARC=`arc_challenge`).

**Δ 기준행**은 각 (모델, 태스크) 의 **Full Params FT** 다.
⚠️ **MedQA 만 예외로 Δ 를 비웠다** — llama2-7b-chat 의 medqa full-param 기준행이 lr 3e-5 / 1e-5
로만 존재하고 이 라인의 동작점(lr 5e-5)과 맞는 것이 없다. raw 값만 싣는다(사용자 결정 2026-09-19).

## ⚠️ AsFT full-parameter 는 원 논문에 없는 일반화다

AsFT(arXiv:2506.08473)와 참조 구현은 **LoRA 전용**이다(ΔW = B A 가정). full-param 행을 만들려면
같은 벌점을 ΔW = W − W₀ 로 일반화해야 한다:

    L = L_SFT + λ · Σ_l ‖ (I − Ĉ_l) ΔW_l ‖²_F ,  Ĉ_l = V_l V_lᵀ / ‖V_l‖_F ,  V_l = W_l^aligned − W_l^base

구현은 `models/asft_baseline.AsFTFullParamRegularizer` + `asft/finetune_asft_full.py` 다.
논문에 실을 때 **"우리가 일반화한 full-param 판"** 임을 반드시 밝혀야 한다.

**λ=1.0 은 full-param 에서도 제대로 작동한다.** 8셀의 최종 벌점 Σ‖(I−Ĉ)ΔW‖²_F:

| 셀 | 최종 벌점 | 셀 | 최종 벌점 |
|---|---:|---|---:|
| llama2_13b / gsm8k | 3.24e-03 | qwen25_7b / gsm8k | 1.12e-02 |
| llama2_7b / medqa | 3.39e-03 | llama31_8b / math | 2.18e-02 |
| llama2_7b / gsm8k | 4.46e-03 | gemma2_9b / gsm8k | 2.56e-02 |
| llama32_3b / math | 9.76e-03 | llama2_7b / arc | 3.05e-02 |

벌점은 학습 초반 SFT loss 와 맞먹는 크기(13B 기준 peak 0.584 vs loss 0.441)까지 올라갔다가
그 그래디언트가 되밀어 3e-3~3e-2 로 내려앉는다. **최종값이 작다는 것은 제약이 없었다는 뜻이
아니라 충족됐다는 뜻이다.** 모델·태스크가 달라도 같은 범위로 수렴한 것이 근거다.

## ⚠️ Lisa 트레이너는 loss 를 grad_accum 으로 나누지 않는다

`LisaTrainer.training_step` 은 `Trainer.training_step` 을 통째로 오버라이드하고
`accelerator.backward(loss)` 를 **나누지 않은 loss** 에 건다. 실측 결과 micro-batch 하나가
벌점을 통째로(= ga 배) 더한다(방향 cos = 1.000000). **이 저장소의 기존 LISA 결과가 전부
이 동작으로 만들어졌으므로** 바꾸지 않고 그대로 재현했다. 자세한 내용은
`scripts/revision/SESSION_2026-09-19.md`.

<!-- 표는 아래에 gen_exp_results_rows.py 출력으로 채운다 -->
