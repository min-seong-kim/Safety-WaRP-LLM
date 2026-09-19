#!/usr/bin/env bash
# Llama-3.1-8B-Instruct / MATH 의 AsFT·Lisa 재학습 (재현성 확인용).
# 기존 셀과 완전히 같은 설정이지만 출력 경로와 리포명(_rerun)을 분리해 덮어쓰지 않는다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_rerun/orch.pid
COMBOS="llama31_8b:math" \
EXP1_OUT_ROOT="$PWD/outputs/asft_lisa_rerun" \
EXP1_LOG_DIR="$PWD/logs/asft_lisa_rerun" \
EXP1_REPO_SUFFIX="_rerun" \
  bash scripts/revision/30_asft_lisa_fullft.sh
echo "RERUN_EXIT=$?"
