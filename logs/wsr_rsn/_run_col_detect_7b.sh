#!/usr/bin/env bash
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODELS=llama2_7b ARMS=rsn STOP_AFTER_NEURONS=1 \
BASIS_DIR_llama2_7b="outputs/wsr_sn_tune/llama2_7b/phase1/phase1_20260922_154717/basis" \
OUT_ROOT=outputs/wsr_rsn_col \
UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh
echo "=== EXIT rc=$? ==="
