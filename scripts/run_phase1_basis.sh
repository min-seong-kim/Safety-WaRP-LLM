#!/bin/bash

# Phase 1: Basis Construction
# ✅ Φ @ Φ^T 방식으로 SVD

echo "========================================="
echo "Phase 1: Basis Construction"
echo "========================================="

# Phase 0에서 학습된 모델 경로
PHASE0_MODEL="./checkpoints/phase0_20260118_173314/final_model"

if [ ! -d "$PHASE0_MODEL" ]; then
    echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
    echo "먼저 scripts/run_phase0_base_training.sh를 실행하세요."
    exit 1
fi

python train_fixed.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset harmful_prompts \
    --harmful_prompts_path ./data/harmful_prompts_200.txt \
    --batch_size 2 \
    --layer_type ffn_down,ffn_up,attn_q,attn_k,attn_v \
    --target_layers "26-27" \
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
