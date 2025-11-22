#!/bin/bash

# Safety-WaRP-LLM: All Phases Training Pipeline
# Phase 1 (Basis) -> Phase 2 (Importance) -> Phase 3 (Learning)

set -e  # Exit on error

echo "========================================================================"
echo "Safety-WaRP-LLM: Complete Training Pipeline"
echo "========================================================================"

# 설정
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
SAFETY_SAMPLES=500
UTILITY_SAMPLES=0
BATCH_SIZE=4
DTYPE="bfloat16"
DEVICE="cuda"
EPOCHS=3
LEARNING_RATE=1e-5
KEEP_RATIO=0.1
TARGET_LAYERS=31

# 기본 디렉토리
BASE_OUTPUT_DIR="./checkpoints"
mkdir -p $BASE_OUTPUT_DIR

# ========================================================================
# Phase 1: Basis Construction
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 1: Basis Construction"
echo "========================================================================"

python train.py \
    --phase 1 \
    --model_name $MODEL_NAME \
    --safety_samples $SAFETY_SAMPLES \
    --batch_size $BATCH_SIZE \
    --target_layers $TARGET_LAYERS \
    --dtype $DTYPE \
    --device $DEVICE \
    --output_dir $BASE_OUTPUT_DIR \
    --debug 2>&1 | tee phase1.log

# Phase 1 출력 경로 추출 (최신 phase1 디렉토리 찾기)
PHASE1_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase1_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE1_BASIS_DIR="$PHASE1_OUTPUT_DIR/checkpoints/basis"

if [ ! -d "$PHASE1_BASIS_DIR" ]; then
    echo "❌ Phase 1 basis directory not found: $PHASE1_BASIS_DIR"
    exit 1
fi
echo "✅ Phase 1 completed. Basis saved to: $PHASE1_BASIS_DIR"

# ========================================================================
# Phase 2: Importance Scoring
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 2: Importance Scoring"
echo "========================================================================"

python train.py \
    --phase 2 \
    --basis_dir $PHASE1_BASIS_DIR \
    --safety_samples $SAFETY_SAMPLES \
    --batch_size $BATCH_SIZE \
    --keep_ratio $KEEP_RATIO \
    --output_dir $BASE_OUTPUT_DIR \
    --debug 2>&1 | tee phase2.log

# Phase 2 출력 경로 추출 (최신 phase2 디렉토리 찾기)
PHASE2_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase2_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE2_MASKS_DIR="$PHASE2_OUTPUT_DIR/checkpoints/masks"

if [ ! -d "$PHASE2_MASKS_DIR" ]; then
    echo "❌ Phase 2 masks directory not found: $PHASE2_MASKS_DIR"
    exit 1
fi
echo "✅ Phase 2 completed. Masks saved to: $PHASE2_MASKS_DIR"

# ========================================================================
# Phase 3: Incremental Learning
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 3: Incremental Learning"
echo "========================================================================"

python train.py \
    --phase 3 \
    --basis_dir $PHASE1_BASIS_DIR \
    --masks_dir $PHASE2_MASKS_DIR \
    --utility_samples $UTILITY_SAMPLES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --output_dir $BASE_OUTPUT_DIR \
    --debug 2>&1 | tee phase3.log

# Phase 3 출력 경로 추출 (최신 phase3 디렉토리 찾기)
PHASE3_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase3_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

echo "✅ Phase 3 completed. Model checkpoints saved to: $PHASE3_OUTPUT_DIR"

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "========================================================================"
echo "Pipeline Completed Successfully!"
echo "========================================================================"
echo ""
echo "📁 Output Directories:"
echo "  Phase 1 (Basis):     $PHASE1_OUTPUT_DIR"
echo "  Phase 2 (Importance): $PHASE2_OUTPUT_DIR"
echo "  Phase 3 (Learning):   $PHASE3_OUTPUT_DIR"
echo ""
echo "📊 Key Results:"
echo "  Basis:   $PHASE1_BASIS_DIR"
echo "  Masks:   $PHASE2_MASKS_DIR"
echo "  Model:   $PHASE3_OUTPUT_DIR/checkpoints"
echo ""
echo "========================================================================"
