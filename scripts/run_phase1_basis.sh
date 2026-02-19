#!/bin/bash

# Phase 1: Basis Construction
# ✅ Φ @ Φ^T 방식으로 SVD

echo "========================================="
echo "Phase 1: Basis Construction"
echo "========================================="

# Phase 0에서 학습된 모델 경로
PHASE0_MODEL="./checkpoints/phase0_20260213_230047"

if [ ! -d "$PHASE0_MODEL" ]; then
    echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
    echo "먼저 scripts/run_phase0_base_training.sh를 실행하세요."
    exit 1
fi

python train.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset circuit_breakers \
    --circuit_breakers_samples_phase1 4994 \
    --batch_size 2 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "========================================="
echo "Phase 1 완료!"
echo "다음 단계: scripts/run_phase2_importance.sh 실행"
echo "========================================="
