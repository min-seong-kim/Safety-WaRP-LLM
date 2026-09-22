#!/usr/bin/env bash
# 새로 학습한 SEAL(lr 3e-5) 평가: 안전성(refusal keyword) + AGNews 정확도.
# 로컬 디렉토리를 평가한다(허브 끊김 + 업로드 안 함).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache
M="$PWD/outputs/eval_local/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr3e-5"

echo "════════ 1) HarmBench (refusal keyword) ════════ $(date -Iseconds)"
HB_ONLY=1 RESUME=false GRADING_OVERRIDE=hard VALIDATE_REPOS=0 PREFETCH=0 \
  REPOS_ONLY="$M" bash scripts/revision/32_eval_cells.sh
echo "SAFETY_RC=$?"

echo "════════ 2) AGNews 정확도 ════════ $(date -Iseconds)"
source scripts/env.sh hb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF="expandable_segments:True"
python -u agnews_eval/evaluate_agnews_sst2.py "$M" \
  --task agnews \
  --agnews-data "$PWD/data/agnews_test_1k_seed42.json" \
  --output-root "$PWD/evaluation_results/agnews_sst2" \
  --batch-size 64 --max-length 1024 --max-new-tokens 32
echo "ACC_RC=$?"
echo "SEAL_LR3E5_EVAL_DONE $(date -Iseconds)"
