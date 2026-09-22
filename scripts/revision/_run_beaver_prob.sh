#!/usr/bin/env bash
# Beaver Dam 7B 를 7B+13B 전체(88조합)에 다시 돌리되, 14개 카테고리의 **최대 확률과 그 카테고리**
# 를 결과 JSON 에 함께 저장한다(`beaver_prob` / `beaver_cat`).
# 라벨 자체는 기존과 같은 임계값 0.5 로 매기므로 지금까지의 수치는 바뀌지 않는다.
# 확률이 남으면 임계값을 바꿀 때 모델을 다시 돌릴 필요가 없다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh harmbench
cd ~/HarmBench
python evaluate_completions_batched.py \
  --cls_path PKU-Alignment/beaver-dam-7b \
  --behaviors_path ./data/behavior_datasets/Advbench_behaviors_standard.csv \
  --jobs _jobs_beaver_prob.json
echo "RC=$?"
echo "BEAVER_PROB_DONE $(date -Iseconds)"
