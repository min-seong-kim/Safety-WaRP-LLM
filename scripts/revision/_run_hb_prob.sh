#!/usr/bin/env bash
# HarmBench-CLS 재채점 + 첫 토큰의 P(yes) 저장(`hb_p_yes`).
# JOBS 를 넘기면 그 목록만 처리한다(검증용).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh harmbench
cd ~/HarmBench
python evaluate_completions_batched.py \
  --cls_path cais/HarmBench-Llama-2-13b-cls \
  --behaviors_path ./data/behavior_datasets/Advbench_behaviors_standard.csv \
  --jobs "${JOBS:-_jobs_hbprob_all.json}"
echo "RC=$?"; echo "HB_PROB_DONE $(date -Iseconds)"
