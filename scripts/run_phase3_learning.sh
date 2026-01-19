#!/bin/bash

# Phase 3: Incremental Learning
# ✅ WaRP 모듈 사용, 마스크 자동 적용

echo "========================================="
echo "Phase 3: Incremental Learning (Fixed)"
echo "========================================="

# 이전 Phase 결과 경로
PHASE0_MODEL="./checkpoints/phase0_20260118_173314/final_model"
BASIS_DIR="./checkpoints/phase1_20260118_182158/basis"
MASKS_DIR="./checkpoints/phase2_20260119_105831/checkpoints/masks"

if [ ! -d "$PHASE0_MODEL" ]; then
    echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
    exit 1
fi

if [ ! -d "$BASIS_DIR" ]; then
    echo "ERROR: Basis를 찾을 수 없습니다: $BASIS_DIR"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "ERROR: Masks를 찾을 수 없습니다: $MASKS_DIR"
    echo "먼저 scripts/run_phase2_importance.sh를 실행하세요."
    exit 1
fi

python train_fixed.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --masks_dir "$MASKS_DIR" \
    --gsm8k_samples 1000 \
    --epochs 3 \
    --utility_lr 1e-5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --layer_type ffn_down,ffn_up,attn_q,attn_k,attn_v \
    --target_layers "26-27" \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "========================================="
echo "Phase 3 완료!"
echo "최종 모델: ./checkpoints/phase3_learning/final_model"
echo "========================================="
