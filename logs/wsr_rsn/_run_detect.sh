#!/usr/bin/env bash
# WSR-RSN-Tune: Phase1 basis + WaRP safety/utility 검출 + critical (학습 전까지)
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
export HF_HUB_OFFLINE=1          # huggingface.co 가 막혀 있다. 필요한 건 전부 캐시에 있다.
export TOKENIZERS_PARALLELISM=false
MODELS="llama2_7b llama2_13b" \
ARMS="rsn" \
STOP_AFTER_NEURONS=1 \
UTILITY_JSON="sn_tune/corpus/wikipedia_utility_1000.json" \
bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh
echo "=== EXIT rc=$? ==="
