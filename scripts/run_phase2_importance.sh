#!/bin/bash

# Phase 2: Importance Scoring
# ✅ model.eval() 모드, optimizer.step 제거

echo "========================================="
echo "Phase 2: Importance Scoring (Fixed)"
echo "========================================="

# Phase 0, 1 결과 경로
PHASE0_MODEL="./checkpoints/phase0_20260118_173314/final_model"
BASIS_DIR="./checkpoints/phase1_20260118_182158/basis"

if [ ! -d "$PHASE0_MODEL" ]; then
    echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
    exit 1
fi

if [ ! -d "$BASIS_DIR" ]; then
    echo "ERROR: Basis를 찾을 수 없습니다: $BASIS_DIR"
    echo "먼저 scripts/run_phase1_basis.sh를 실행하세요."
    exit 1
fi

python train_fixed.py \
    --phase 2 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    --circuit_breakers_samples 1000 \
    --keep_ratio 0.1 \
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
echo "Phase 2 완료!"
echo "다음 단계: scripts/run_phase3_learning.sh 실행"
echo "========================================="
