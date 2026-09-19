#!/usr/bin/env bash
# AsFT / Lisa 를 **q,k,v,up,down 만 학습**하도록 재학습 (WSR-Tune 과 동일 범위).
# 기존 셀(전 파라미터 학습)과 구분하려고 출력 경로·리포명(_tgtonly)을 분리한다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_tgtonly/orch.pid
COMBOS="llama2_7b:gsm8k llama31_8b:math" \
TRAIN_ONLY_TARGETS=1 \
EXP1_OUT_ROOT="$PWD/outputs/asft_lisa_tgtonly" \
EXP1_LOG_DIR="$PWD/logs/asft_lisa_tgtonly" \
EXP1_REPO_SUFFIX="_tgtonly" \
  bash scripts/revision/30_asft_lisa_fullft.sh
echo "TGTONLY_EXIT=$?"
