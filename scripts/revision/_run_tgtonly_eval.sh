#!/usr/bin/env bash
# 타깃한정(_tgtonly) 재학습본 평가. EVAL_ROOT 의 .uploaded 셀만 자동으로 잡는다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_tgtonly/eval.pid
EVAL_ROOT="$PWD/outputs/asft_lisa_tgtonly" bash scripts/revision/32_eval_cells.sh
echo "TGTONLY_EVAL_EXIT=$?"
