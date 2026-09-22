#!/usr/bin/env bash
# llama2-13b-chat baseline 11개의 completions 를 **새로 생성**하고 refusal keyword 로
# 채점해, 논문 Table(tab:main_all_models) 의 13B 열이 재현되는지 확인한다.
#
# 7B 1단계(_run_cls_eval_stage1.sh)와 동일한 조건:
#   · sys 모드(llama-2 <<SYS>> 안전 프롬프트) · GRADING=hard · seed 42
#   · test case 는 재사용(공격별 EFFECTIVE_EXPERIMENT_NAME 단위로 공유), completions 만 재생성
#
# 기준값 출처: minseong/HarmBench/tables.csv (사용자 제공, 논문 값과 일치)
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

REPOS="wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5
wvnvwn/llama-2-13b-chat-hf-lr5e-5-gsm8k-lr5e-5
wvnvwn/llama-2-13b-chat-hf-lr5e-5-safeinstr-0.1
wvnvwn/llama-2-13b-chat-hf-lr5e-5-resta-0.1
wvnvwn/llama-2-13b-chat-hf-lr5e-5-safedelta-scale0.1
wvnvwn/llama-2-13b-chat-hf-gsm8k-sn-tuned-lr5e-5
wvnvwn/llama-2-13b-chat-hf-gsm8k-rsn-tuned-lr5e-5
wvnvwn/llama-2-13b-chat-hf-WaRP-lr5e-5
kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4"

echo "대상 $(echo "$REPOS" | wc -l) 개 · 새 생성(RESUME=false) · GRADING=hard · sys"
HB_ONLY=1 RESUME=false GRADING_OVERRIDE=hard \
  REPOS_ONLY="$(echo $REPOS)" bash scripts/revision/32_eval_cells.sh
echo "CLS_EVAL_13B_DONE rc=$?  $(date -Iseconds)"
