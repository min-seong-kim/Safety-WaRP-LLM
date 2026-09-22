#!/usr/bin/env bash
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
MODELS=llama2_13b ARMS=rsn STOP_AFTER_NEURONS=1 \
BASIS_DIR_llama2_13b="outputs/wsr_sn_tune/llama2_13b/phase1/phase1_20260922_160144/basis" \
bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh
echo "=== EXIT rc=$? ==="
