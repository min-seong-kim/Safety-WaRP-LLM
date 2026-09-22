#!/usr/bin/env bash
# 13B 11개 모델 × 4공격의 **이번에 새로 생성한** completions 를 LLM 분류기 3종으로 채점한다.
# 대상 모델 가중치는 전혀 로드하지 않는다 — completions JSON 만 읽는다.
# 배치판이라 분류기는 각 1회만 로드된다(7B 동일 규모 실측 22분 41초).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh harmbench
cd ~/HarmBench
BEH=./data/behavior_datasets/Advbench_behaviors_standard.csv
for pair in "cais/HarmBench-Llama-2-13b-cls:harmbench" "allenai/wildguard:wildguard" "PKU-Alignment/beaver-dam-7b:beaver"; do
  CLS="${pair%%:*}"; TAG="${pair##*:}"
  echo "══════════ $CLS ($TAG) ══════════ $(date -Iseconds)"
  python evaluate_completions_batched.py --cls_path "$CLS" --behaviors_path "$BEH" --jobs "_jobs13b_${TAG}.json"
  echo "RC_${TAG}=$?"
done
echo "CLS_13B_BATCHED_DONE $(date -Iseconds)"
