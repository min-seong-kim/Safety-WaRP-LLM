#!/bin/bash

# Phase 2 + 3: No-Rotation Experiment
# - Phase 1 basis 없이 Phase 2/3만 수행
# - WaRP identity basis(U=I) 사용

set -e

echo "========================================="
echo "Phase 2 + 3: No-Rotation Experiment"
echo "========================================="

# ==================================================
# Configuration
# ==================================================
PHASE0_MODEL="kmseong/Llama-3.2-3B-SSFT"

# Phase 2
PHASE2_DATASET="circuit_breakers"   # Options: circuit_breakers, wikipedia
PHASE2_SAMPLES=4994                  # circuit_breakers 사용 시 적용
PHASE2_WIKIPEDIA_SAMPLES=4994        # wikipedia 사용 시 적용
KEEP_RATIO=0.1
BATCH_SIZE=2
MAX_LENGTH=512

# Phase 3
PHASE3_DATASET="gsm8k"             # Options: safety, gsm8k, metamath, math
GSM8K_SAMPLES=0
METAMATH_SAMPLES=0
MATH_SAMPLES=0
MATH_SUBJECTS="all"
MATH_LEVELS="all"
CIRCUIT_BREAKERS_SAMPLES_PHASE3=4994
EPOCHS=3
UTILITY_LR=1e-5
GRAD_ACCUM=4

# Common
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_LAYERS="all"
OUTPUT_DIR="/lustre/gokms0509/Safety-WaRP-LLM/checkpoints"
LOG_DIR="/home/gokms0509/Safety-WaRP-LLM/logs"
DEVICE="cuda"
DTYPE="bfloat16"
SEED=42

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ==================================================
# Phase 2: Importance Scoring (No Rotation)
# ==================================================
echo ""
echo "========================================="
echo "Phase 2: Importance Scoring (No Rotation)"
echo "========================================="

if [ "$PHASE2_DATASET" = "circuit_breakers" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 circuit_breakers --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples $PHASE2_SAMPLES"
elif [ "$PHASE2_DATASET" = "wikipedia" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 wikipedia --wikipedia_samples_phase2 $PHASE2_WIKIPEDIA_SAMPLES"
else
    echo "ERROR: Unknown PHASE2_DATASET: $PHASE2_DATASET"
    echo "Choose from: circuit_breakers, wikipedia"
    exit 1
fi

python train.py \
    --phase 2 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --no_rotation \
    --perlayer \
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
    2>&1 | tee "$LOG_DIR/phase2_no_rotation.log"

PHASE2_OUTPUT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "phase2_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE2_MASKS_DIR="$PHASE2_OUTPUT_DIR/checkpoints/masks"

if [ ! -d "$PHASE2_MASKS_DIR" ]; then
    echo "ERROR: Phase 2 masks directory not found: $PHASE2_MASKS_DIR"
    exit 1
fi

echo "✓ Phase 2 completed"
echo "  Masks: $PHASE2_MASKS_DIR"

# ==================================================
# Phase 3: Incremental Learning (No Rotation)
# ==================================================
echo ""
echo "========================================="
echo "Phase 3: Incremental Learning (No Rotation)"
echo "========================================="

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
    echo "Choose from: safety, gsm8k, metamath, math"
    exit 1
fi

python train.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --masks_dir "$PHASE2_MASKS_DIR" \
    --no_rotation \
    $PHASE3_DATASET_ARG \
    --epochs $EPOCHS \
    --utility_lr $UTILITY_LR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed $SEED \
    2>&1 | tee "$LOG_DIR/phase3_no_rotation.log"

echo ""
echo "========================================="
echo "No-Rotation Experiment Completed"
echo "========================================="
echo "Phase 2 masks: $PHASE2_MASKS_DIR"
echo "Phase 3 outputs: $OUTPUT_DIR/phase3_*"
echo "Logs: $LOG_DIR/phase2_no_rotation.log, $LOG_DIR/phase3_no_rotation.log"
echo "========================================="
