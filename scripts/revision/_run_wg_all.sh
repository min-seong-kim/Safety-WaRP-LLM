#!/usr/bin/env bash
# WildGuard 재채점 + 출력 3줄 전부 저장(wg_harmful_request / wg_refusal / wg_harmful_response)
# 및 원문(wg_raw). label 은 기존과 동일하게 'harmful response' 기준이라 수치는 안 바뀐다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh harmbench
cd ~/HarmBench
python evaluate_completions_batched.py \
  --cls_path allenai/wildguard \
  --behaviors_path ./data/behavior_datasets/Advbench_behaviors_standard.csv \
  --jobs _jobs_wg_all.json
echo "RC=$?"; echo "WG_ALL_DONE $(date -Iseconds)"
