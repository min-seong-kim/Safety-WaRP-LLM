#!/usr/bin/env bash
# 출발 모델(= downstream FT 전, safety-tuned)의 downstream 성능을 잰다.
#
# 왜: 기준 Full FT 보다 AsFT/Lisa 의 downstream 이 높은 셀이 6개 중 5개다.
#     "제약을 건 기법이 더 잘 배운다" 는 이상하므로, 기준선이 학습의 상한이 아니라
#     **모델을 망가뜨리는 동작점**일 가능성을 확인한다. 출발 모델이 이미 full FT 결과보다
#     높다면 full FT 는 그 태스크에서 순손실이고, 제약은 정규화로 작동한 것이 된다.
#
# 이름에 task 토큰이 없어 자동 추론이 gsm8k 로 떨어지므로 LM_TASKS 로 못박는다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/eval_exp1/startmodels.pid

echo "########## MATH 출발 모델 2개 ##########"
LM_ONLY=1 LM_TASKS="hendrycks_math_safe" \
REPOS_ONLY="kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5 kmseong/llama3_2_3b-instruct-SSFT-lr5e-5" \
  bash scripts/revision/32_eval_cells.sh

echo "########## GSM8K 출발 모델 2개 (대조군) ##########"
LM_ONLY=1 LM_TASKS="gsm8k" \
REPOS_ONLY="kmseong/llama2_7b-chat-Safety-FT-lr5e-5 wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5" \
  bash scripts/revision/32_eval_cells.sh

echo "STARTMODELS_EXIT=$?"
