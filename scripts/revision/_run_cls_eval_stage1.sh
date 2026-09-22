#!/usr/bin/env bash
# 1단계: llama2-7b-chat baseline 11개의 test case + completions 를 **새로 생성**하고
#        refusal keyword 로 채점해 기존 논문 수치가 재현되는지 확인한다.
#
# 왜: 논문은 전부 refusal keyword 로 평가했다. 리뷰어가 "왜 LLM classifier 로 안 했나" 를
#     물을 수 있으므로, 같은 생성물에 대해 keyword / HarmBench-CLS / WildGuard / Beaver Dam
#     네 채점기를 비교하려 한다. 그 전에 keyword 가 기존 값을 재현하는지부터 본다.
#
# RESUME=false → 기존 completions 가 있어도 처음부터 다시 생성한다(사용자 지시).
# 2단계(분류기 채점)는 _run_cls_eval_stage2.sh 가 RESUME=true 로 같은 생성물을 재사용한다.
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

echo "대상 $(echo "$REPOS" | wc -l) 개 · 새 생성(RESUME=false) · GRADING=hard"
HB_ONLY=1 RESUME=false GRADING_OVERRIDE=hard \
  REPOS_ONLY="$(echo $REPOS)" bash scripts/revision/32_eval_cells.sh
echo "CLS_STAGE1_DONE rc=$?  $(date -Iseconds)"
