#!/usr/bin/env bash
# HarmBench-CLS 판정 프롬프트 변형으로 재채점. VARIANT=lenient|comply, JOBS=<목록>
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh harmbench
cd ~/HarmBench
python evaluate_completions_batched.py \
  --cls_path cais/HarmBench-Llama-2-13b-cls \
  --behaviors_path ./data/behavior_datasets/Advbench_behaviors_standard.csv \
  --prompt_variant "${VARIANT:?}" --jobs "${JOBS:?}"
echo "RC=$?"; echo "HB_VARIANT_DONE ${VARIANT} $(date -Iseconds)"
