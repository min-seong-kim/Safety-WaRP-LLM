#!/usr/bin/env bash
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/eval_exp1/gapfill.pid
REPOS_ONLY="kmseong/llama2_7b-chat-arc_ssft_lr5e-5" bash scripts/revision/32_eval_cells.sh
echo "GAPFILL_EXIT=$?"
