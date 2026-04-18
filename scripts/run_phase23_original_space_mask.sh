#!/bin/bash

# Phase 2 + 3: Original-Space Importance Mask Pipeline
#
# 목표:
# - basis/rotation/WaRP 없이 original weight space에서 importance mask 생성
# - mask=1(중요 weight)은 freeze, 나머지만 downstream finetuning

set -e
set -o pipefail
export CUDA_VISIBLE_DEVICES=0

echo "========================================================================"
echo "Phase 2 + 3: Original-Space Importance Mask Pipeline"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

PHASE0_MODEL="kmseong/llama3.1_8b_base-Safety-FT-lr3e-5"  # Phase 0 모델 (HF 모델 ID 또는 로컬 경로)

# Phase 2 (importance)
PHASE2_DATASET="circuit_breakers"   # circuit_breakers | wikipedia
PHASE2_SAMPLES=4994
PHASE2_WIKIPEDIA_SAMPLES=4994
KEEP_RATIO=0.1

# Phase 3 (downstream)
PHASE3_DATASET="gsm8k"              # safety | gsm8k | metamath | math
GSM8K_SAMPLES=0
METAMATH_SAMPLES=0
MATH_SAMPLES=0
MATH_SUBJECTS="all"
MATH_LEVELS="all"
CIRCUIT_BREAKERS_SAMPLES_PHASE3=4994

# Common training
EPOCHS=3
LR_LIST=("3e-5")
BATCH_SIZE=4
GRAD_ACCUM=4
WARMUP_RATIO=0.1
LR_SCHEDULER="cosine"
BASE_WEIGHT_DECAY=0.01
MAX_LENGTH=1024

LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_LAYERS="all"
OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
DEVICE="cuda"
DTYPE="bfloat16"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Phase 0 Model: $PHASE0_MODEL"
echo "  Phase 2 Dataset: $PHASE2_DATASET"
echo "  Keep Ratio: $KEEP_RATIO"
echo "  Phase 3 Dataset: $PHASE3_DATASET"
echo "  LR Sweep: ${LR_LIST[*]}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Grad Accum: $GRAD_ACCUM"
echo "  Epochs: $EPOCHS"
echo ""

# ========================================================================
# Phase 2: Importance Scoring in Original Space
# ========================================================================
echo "========================================================================"
echo "PHASE 2: Importance Scoring (Original Space)"
echo "========================================================================"

if [ "$PHASE2_DATASET" = "circuit_breakers" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 circuit_breakers --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase2 $PHASE2_SAMPLES"
elif [ "$PHASE2_DATASET" = "wikipedia" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 wikipedia --wikipedia_samples_phase2 $PHASE2_WIKIPEDIA_SAMPLES"
else
    echo "ERROR: Unknown PHASE2_DATASET: $PHASE2_DATASET"
    exit 1
fi

python train.py \
    --phase 2 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --original_space_mask \
    $PHASE2_DATASET_ARG \
    --keep_ratio $KEEP_RATIO \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed $SEED \
    2>&1 | tee "$LOG_DIR/phase2_original_space_${TIMESTAMP}.log"

PHASE2_OUTPUT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "phase2_original_space_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE2_MASKS_DIR="$PHASE2_OUTPUT_DIR/checkpoints/masks"

if [ ! -d "$PHASE2_MASKS_DIR" ]; then
    echo "ERROR: Phase 2 masks directory not found: $PHASE2_MASKS_DIR"
    exit 1
fi

echo ""
echo "✓ Phase 2 completed"
echo "  Masks saved to: $PHASE2_MASKS_DIR"
echo ""

# ========================================================================
# Phase 3: Downstream Finetuning with Original-Space Mask
# ========================================================================
if [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $GSM8K_SAMPLES"
elif [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset safety --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase3 $CIRCUIT_BREAKERS_SAMPLES_PHASE3"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset metamath --metamath_samples $METAMATH_SAMPLES"
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset math --math_samples $MATH_SAMPLES --math_subjects $MATH_SUBJECTS --math_levels $MATH_LEVELS"
else
    echo "ERROR: Unknown PHASE3_DATASET: $PHASE3_DATASET"
    exit 1
fi

PHASE3_OUTPUT_DIRS=()
for LEARNING_RATE in "${LR_LIST[@]}"; do
    LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')

    echo "========================================================================"
    echo "PHASE 3: Original-Space Downstream Finetuning (LR=$LEARNING_RATE)"
    echo "========================================================================"

    python train.py \
        --phase 3 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --masks_dir "$PHASE2_MASKS_DIR" \
        --original_space_mask \
        $PHASE3_DATASET_ARG \
        --epochs $EPOCHS \
        --utility_lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --warmup_ratio $WARMUP_RATIO \
        --lr_scheduler_type $LR_SCHEDULER \
        --base_weight_decay $BASE_WEIGHT_DECAY \
        --max_length $MAX_LENGTH \
        --layer_type "$LAYER_TYPE" \
        --target_layers $TARGET_LAYERS \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed $SEED \
        2>&1 | tee "$LOG_DIR/phase3_original_space_lr${LR_SAFE}_${TIMESTAMP}.log"

    PHASE3_OUTPUT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "phase3_original_space_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -d "$PHASE3_OUTPUT_DIR" ]; then
        PHASE3_OUTPUT_DIRS+=("$PHASE3_OUTPUT_DIR")
        echo "✓ Phase 3 completed (LR=$LEARNING_RATE): $PHASE3_OUTPUT_DIR"
    else
        echo "WARNING: Phase 3 output directory not found for LR=$LEARNING_RATE"
    fi

    echo ""
done

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "Original-Space Mask Pipeline Completed"
echo "========================================================================"
echo "Phase 2 masks: $PHASE2_MASKS_DIR"
echo ""
echo "Phase 3 model outputs:"
for dir in "${PHASE3_OUTPUT_DIRS[@]}"; do
    echo "  - $dir/final_model"
done
echo ""
echo "Log files:"
echo "  - $LOG_DIR/phase2_original_space_${TIMESTAMP}.log"
for LEARNING_RATE in "${LR_LIST[@]}"; do
    LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')
    echo "  - $LOG_DIR/phase3_original_space_lr${LR_SAFE}_${TIMESTAMP}.log"
done
echo "========================================================================"
