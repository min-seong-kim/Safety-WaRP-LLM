#!/usr/bin/env bash
# agnews 정확도만 다시 (안전성은 이미 끝남). 인자는 --tasks 가 아니라 --task 다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source scripts/env.sh hb
export HF_HOME=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache
export CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF="expandable_segments:True"
for r in kmseong/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5 \
         kmseong/llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5 \
         kmseong/llama2_7b-chat-CB_SSFT-lisa_agnews_rho1.0_lr7e-5; do
  echo "---- $r"
  python -u agnews_eval/evaluate_agnews_sst2.py "$r" \
    --task agnews \
    --agnews-data "$PWD/data/agnews_test_1k_seed42.json" \
    --output-root "$PWD/evaluation_results/agnews_sst2" \
    --batch-size 64 --max-length 1024 --max-new-tokens 32
  echo "   rc=$?"
done
echo "AGNEWS_ACC_DONE $(date -Iseconds)"
