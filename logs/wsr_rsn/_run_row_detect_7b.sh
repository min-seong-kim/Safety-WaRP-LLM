#!/usr/bin/env bash
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
MODELS=llama2_7b ARMS=rsn STOP_AFTER_NEURONS=1 \
bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh
echo "=== EXIT rc=$? ==="
