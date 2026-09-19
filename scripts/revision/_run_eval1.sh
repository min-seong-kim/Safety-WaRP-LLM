#!/usr/bin/env bash
# 실험 1 평가 런처. setsid 로 독립 세션(부모 셸과 프로세스 그룹 분리).
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/eval_exp1/eval.pid
bash scripts/revision/32_eval_cells.sh
echo "EVAL1_EXIT=$?"
