#!/usr/bin/env bash
# Qwen2.5-7B-Instruct CB 출발 모델(= downstream FT 전)의 GSM8K 를 잰다.
#
# 왜: Qwen/GSM8K 에서 AsFT·Lisa 의 GSM8K(0.70~0.73)가 기준 Full FT(0.6732)보다 높다.
#     "제약을 건 기법이 더 잘 배운다" 는 이상하므로, 출발 모델이 이미 몇 점인지 확인한다.
#     출발점이 이미 0.70 대라면 이 열은 "학습량" 이 아니라 "얼마나 덜 망가뜨렸는가" 이고,
#     lr 을 낮춰 AsFT/Lisa 를 끌어내리려는 시도의 하한도 출발 모델 점수가 된다.
# 이름에 task 토큰이 없어 자동 추론이 흔들리므로 LM_TASKS 로 못박는다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
LM_ONLY=1 LM_TASKS="gsm8k" \
REPOS_ONLY="wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5" \
  bash scripts/revision/32_eval_cells.sh
echo "QWEN_START_GSM8K_DONE rc=$?  $(date -Iseconds)"
