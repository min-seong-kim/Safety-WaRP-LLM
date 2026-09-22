#!/usr/bin/env bash
# 2단계: 1단계에서 만든 **같은 생성물**을 LLM 분류기 3종으로 재채점한다.
#   cais/HarmBench-Llama-2-13b-cls · allenai/wildguard · PKU-Alignment/beaver-dam-7b
#
# RESUME=true → completions 를 재생성하지 않고 그대로 쓴다(1단계와 동일한 답변에 대한 채점).
# 결과 파일은 {MODEL}__{harmbench|wildguard|beaver}.json 으로 분리 저장되어
# 1단계 keyword 결과({MODEL}.json)를 덮어쓰지 않는다.
#
# judge 모델: HarmBench-CLS·WildGuard 는 홈 캐시를 영속 캐시로 심볼릭 링크해 두었고,
#             Beaver Dam 7B(13GB)는 2026-09-21 에 새로 받아 영속 캐시에 있다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

REPOS="kmseong/llama2_7b-chat-Safety-FT-lr5e-5
kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5
kmseong/llama2_7b-chat-gsm8k_safelnstr_10p_lr5e-5
kmseong/llama2_7b_chat_gsm8k_resta_gamma0.3
kmseong/llama2-7b-chat-gsm8k-safedelta-scale0.1
kmseong/llama2_7b_chat_gsm8k_ft_freeze_sn_lr5e-5_revised
kmseong/llama2_7b_chat_gsm8k_ft_freeze_rsn_lr5e-5_new_revised
wvnvwn/llama-2-7b-chat-warp-ratio-0.10
kmseong/llama2_7b_chat_seal_5e-5
kmseong/llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4
kmseong/llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4"

echo "대상 $(echo "$REPOS" | wc -l) 개 · 생성물 재사용(RESUME=true) · GRADING=classifier (3종)"
HB_ONLY=1 RESUME=true GRADING_OVERRIDE=classifier \
  REPOS_ONLY="$(echo $REPOS)" bash scripts/revision/32_eval_cells.sh
echo "CLS_STAGE2_DONE rc=$?  $(date -Iseconds)"
