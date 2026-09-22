#!/usr/bin/env bash
# 2단계(배치판): 분류기 3종을 각각 **1회만 로드**해 44조합을 한 번에 채점한다.
#
# 기존 경로(harmbench_eval.sh)의 병목 두 가지를 제거했다:
#   · (조합 × 분류기) 마다 프로세스 재기동 → 13B 를 44번 재로드   (132회 → 3회)
#   · behavior 마다 cls.classify()/generate() 호출 (배치 크기 1)   → 파일당 1배치
# 채점 로직은 eval_utils 의 것을 그대로 재사용하며, 기존 결과와 behavior 별 라벨까지
# 일치함을 `--verify_against` 로 확인했다(8/8, 불일치 0건).
#
# 생성물(completions)은 1단계에서 만든 것을 그대로 쓴다 — 재생성하지 않는다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh harmbench
cd ~/HarmBench

BEH=./data/behavior_datasets/Advbench_behaviors_standard.csv
for pair in "cais/HarmBench-Llama-2-13b-cls:harmbench" "allenai/wildguard:wildguard" "PKU-Alignment/beaver-dam-7b:beaver"; do
  CLS="${pair%%:*}"; TAG="${pair##*:}"
  echo "══════════ $CLS ($TAG) ══════════ $(date -Iseconds)"
  python evaluate_completions_batched.py --cls_path "$CLS" --behaviors_path "$BEH" --jobs "_jobs_${TAG}.json"
  echo "RC_${TAG}=$?"
done
echo "CLS_STAGE2_BATCHED_DONE $(date -Iseconds)"
