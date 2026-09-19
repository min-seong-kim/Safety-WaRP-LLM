#!/usr/bin/env bash
# Llama-3.1-8B-Instruct 의 **출발 모델**(downstream FT 전, CB safety-tuned) MATH 점수.
# 기준 Full FT(0.1238) 보다 높은지가 핵심 — 높다면 full FT 는 MATH 에서 순손실이고
# AsFT(0.1988)/Lisa(0.1692) 는 "더 잘 배운" 것이 아니라 "덜 망가뜨린" 것이 된다.
# 이름에 task 토큰이 없어 자동 추론이 gsm8k 로 떨어지므로 LM_TASKS 로 못박는다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/eval_exp1/startmath.pid
LM_ONLY=1 LM_TASKS="hendrycks_math_safe" \
REPOS_ONLY="kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5" \
  bash scripts/revision/32_eval_cells.sh
echo "STARTMATH_EXIT=$?"
