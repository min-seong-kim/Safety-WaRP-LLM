#!/bin/bash

# Safety-WaRP-LLM: Complete Training Pipeline (Integrated)
# Phase 1 (Basis) -> Phase 2 (Importance) -> Phase 3 (Learning)
# 
# run_phase1_basis.sh, run_phase2_importance.sh, run_phase3_learning.sh를 통합
# 한 번에 모든 phase를 순차적으로 실행
source /home/yonsei_jong/miniconda3/etc/profile.d/conda.sh
conda activate hb
set -e  # Exit on error
set -o pipefail  # Ensure failures are not hidden by tee pipelines
export CUDA_VISIBLE_DEVICES=1

echo "========================================================================"
echo "Safety-WaRP-LLM: Complete Training Pipeline (Integrated)"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

# Phase 0 모델
PHASE0_MODEL="./checkpoints/phase0_20260417_114858"  


# Phase 1: Basis Construction
# ==============================
# Dataset 선택 (Safety 또는 Utility)
# Options: circuit_breakers, wikipedia
PHASE1_DATASET="circuit_breakers"
PHASE1_SAMPLES=4994


# Phase 2: Importance Scoring
# ==============================
# Dataset 선택 (동일하게 사용)
PHASE2_DATASET="circuit_breakers"
PHASE2_SAMPLES=4994
KEEP_RATIO=0.1

# Two-Mask 설정 (비활성화하려면 TWO_MASK="" 로 설정)
# preserve_mask AND NOT adapt_mask → adapt에 중요한 파라미터는 Phase 3에서 학습 가능
TWO_MASK=""           # "" = 비활성화 (기본), "true" = 활성화
ADAPT_DATASET="safety" # adapt 데이터셋: gsm8k, math, metamath, wikipedia, safety
ADAPT_SAMPLES=4994       # 0=전체

# Phase 3: Incremental Learning
# ==============================
# Dataset 선택 (Utility 또는 Safety)
PHASE3_DATASET="gsm8k" # Options: safety, gsm8k, metamath, math

# Phase3=MATH 설정
MATH_SUBJECTS="all"  # 예: Algebra,Geometry
MATH_LEVELS="all"    # 예: 1,2,3,4,5

if [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_SAMPLES=4994
elif [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_SAMPLES=0  # 0 = all samples
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_SAMPLES=10000  # 0 = all samples
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_SAMPLES=0  # 0 = all samples
fi

# 공통 설정
BATCH_SIZE=4
DTYPE="bfloat16"
DEVICE="cuda"
EPOCHS=3
# LR_LIST=("1e-5" "3e-5" "5e-5")
LR_LIST=("3e-5")  
TARGET_LAYERS="all"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
# attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_down,ffn_up
BASE_OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BASE_OUTPUT_DIR
mkdir -p $LOG_DIR

echo "Configuration:"
echo "  Phase 0 Model: $PHASE0_MODEL"
echo "  Phase 1 Dataset: $PHASE1_DATASET (samples=$PHASE1_SAMPLES)"
echo "  Phase 2 Dataset: $PHASE2_DATASET (samples=$PHASE2_SAMPLES)"
echo "  Phase 3 Dataset: $PHASE3_DATASET (samples=$PHASE3_SAMPLES)"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Output Dir: $BASE_OUTPUT_DIR"
echo ""

# Phase 3 Dataset validation
if [[ ! "$PHASE3_DATASET" =~ ^(safety|gsm8k|metamath|math)$ ]]; then
    echo "❌ ERROR: Unknown Phase 3 dataset: $PHASE3_DATASET"
    echo "Choose from: safety, gsm8k, metamath, math"
    exit 1
fi

# ========================================================================
# Phase 1: Basis Construction
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 1: Basis Construction"
echo "========================================================================"
echo ""

if [ "$PHASE1_DATASET" = "circuit_breakers" ]; then
    PHASE1_DATASET_ARG="--circuit_breakers_samples_phase1 $PHASE1_SAMPLES"
elif [ "$PHASE1_DATASET" = "wikipedia" ]; then
    PHASE1_DATASET_ARG="--wikipedia_samples_phase1 $PHASE1_SAMPLES"
else
    echo "❌ ERROR: Unknown Phase 1 dataset: $PHASE1_DATASET"
    echo "Choose from: circuit_breakers, wikipedia"
    exit 1
fi

python train.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset "$PHASE1_DATASET" \
    $PHASE1_DATASET_ARG \
    --batch_size $BATCH_SIZE \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir $BASE_OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed 42 \
    2>&1 | tee $LOG_DIR/phase1_${TIMESTAMP}.log

# Phase 1 출력 경로 추출 (최신 phase1 디렉토리 찾기)
PHASE1_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase1_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE1_BASIS_DIR="$PHASE1_OUTPUT_DIR/basis"

if [ ! -d "$PHASE1_BASIS_DIR" ]; then
    echo "❌ ERROR: Phase 1 basis directory not found: $PHASE1_BASIS_DIR"
    exit 1
fi

echo ""
echo "✅ Phase 1 completed successfully"
echo "   Basis saved to: $PHASE1_BASIS_DIR"
echo ""

# ========================================================================
# Phase 2: Importance Scoring
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 2: Importance Scoring"
echo "========================================================================"
echo ""

if [ "$PHASE2_DATASET" = "circuit_breakers" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 $PHASE2_SAMPLES"
elif [ "$PHASE2_DATASET" = "wikipedia" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 wikipedia --wikipedia_samples_phase2 $PHASE2_SAMPLES"
else
    echo "❌ ERROR: Unknown Phase 2 dataset: $PHASE2_DATASET"
    echo "Choose from: circuit_breakers, wikipedia"
    exit 1
fi

# Two-Mask 인자 구성
if [ -n "$TWO_MASK" ]; then
    TWO_MASK_ARG="--two_mask --adapt_dataset_phase2 $ADAPT_DATASET --adapt_samples_phase2 $ADAPT_SAMPLES"
    echo "[Two-Mask] Enabled: adapt_dataset=$ADAPT_DATASET, adapt_samples=$ADAPT_SAMPLES"
else
    TWO_MASK_ARG=""
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --phase 2 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$PHASE1_BASIS_DIR" \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    $PHASE2_DATASET_ARG \
    --keep_ratio $KEEP_RATIO \
    --batch_size $BATCH_SIZE \
    --max_length 1024 \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir $BASE_OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed 42 \
    --perlayer \
    $TWO_MASK_ARG \
    2>&1 | tee $LOG_DIR/phase2_${TIMESTAMP}.log

# Phase 2 출력 경로 추출 (최신 phase2 디렉토리 찾기)
PHASE2_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase2_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE2_MASKS_DIR="$PHASE2_OUTPUT_DIR/checkpoints/masks"

if [ ! -d "$PHASE2_MASKS_DIR" ]; then
    echo "❌ ERROR: Phase 2 masks directory not found: $PHASE2_MASKS_DIR"
    exit 1
fi

echo ""
echo "✅ Phase 2 completed successfully"
echo "   Masks saved to: $PHASE2_MASKS_DIR"
echo ""

# ========================================================================
# Phase 3: Incremental Learning (LR sweep)
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 3: Incremental Learning (LR sweep: ${LR_LIST[*]})"
echo "========================================================================"
echo ""

if [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset safety --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase3 $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset metamath --metamath_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset math --math_samples $PHASE3_SAMPLES --math_subjects $MATH_SUBJECTS --math_levels $MATH_LEVELS"
else
    echo "❌ ERROR: Unknown Phase 3 dataset: $PHASE3_DATASET"
    echo "Choose from: safety, gsm8k, metamath, math"
    exit 1
fi

PHASE3_OUTPUT_DIRS=()

for LEARNING_RATE in "${LR_LIST[@]}"; do
    LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Phase 3: LR = $LEARNING_RATE"
    echo "──────────────────────────────────────────────────────────────────────"

    python train.py \
        --phase 3 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --basis_dir "$PHASE1_BASIS_DIR" \
        --masks_dir "$PHASE2_MASKS_DIR" \
        $PHASE3_DATASET_ARG \
        --epochs $EPOCHS \
        --utility_lr $LEARNING_RATE \
        --batch_size 1 \
        --gradient_accumulation_steps 16 \
        --layer_type "$LAYER_TYPE" \
        --target_layers $TARGET_LAYERS \
        --output_dir $BASE_OUTPUT_DIR \
        --log_dir $LOG_DIR \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed 42 \
        --non_freeze \
        2>&1 | tee $LOG_DIR/phase3_lr${LR_SAFE}_${TIMESTAMP}.log

    PHASE3_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase3_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

    if [ ! -d "$PHASE3_OUTPUT_DIR" ]; then
        echo "⚠️  WARNING: Phase 3 (LR=$LEARNING_RATE) output directory not found"
    else
        echo "✅ Phase 3 (LR=$LEARNING_RATE) completed: $PHASE3_OUTPUT_DIR"
        PHASE3_OUTPUT_DIRS+=("$PHASE3_OUTPUT_DIR")
    fi
done

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "========================================================================"
echo "🎉 Complete Pipeline Finished Successfully!"
echo "========================================================================"
echo ""
echo "📁 Output Directories:"
echo "  Phase 1 Output:  $PHASE1_OUTPUT_DIR"
echo "  Phase 2 Output:  $PHASE2_OUTPUT_DIR"
echo ""
echo "📊 Key Artifacts:"
echo "  ✅ Basis:  $PHASE1_BASIS_DIR"
echo "  ✅ Masks:  $PHASE2_MASKS_DIR"
echo ""
echo "  Phase 3 Models (per LR):"
for dir in "${PHASE3_OUTPUT_DIRS[@]}"; do
    echo "    ✅ $dir/final_model"
done
echo ""
echo "📝 Log Files:"
echo "  Phase 1: $LOG_DIR/phase1_${TIMESTAMP}.log"
echo "  Phase 2: $LOG_DIR/phase2_${TIMESTAMP}.log"
for LEARNING_RATE in "${LR_LIST[@]}"; do
    LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')
    echo "  Phase 3 (LR=$LEARNING_RATE): $LOG_DIR/phase3_lr${LR_SAFE}_${TIMESTAMP}.log"
done
echo ""
echo "⚙️  Training Configuration Summary:"
echo "  - Phase 1 Dataset: $PHASE1_DATASET ($PHASE1_SAMPLES samples)"
echo "  - Phase 2 Dataset: $PHASE2_DATASET ($PHASE2_SAMPLES samples), Two-Mask: ${TWO_MASK:-disabled}"
echo "  - Phase 3 Dataset: $PHASE3_DATASET ($PHASE3_SAMPLES samples)"
echo "  - Keep Ratio:      $KEEP_RATIO"
echo "  - Learning Rates:  ${LR_LIST[*]}"
echo "  - Epochs:          $EPOCHS"
echo "  - Batch Size:      $BATCH_SIZE"
echo "  - Layer Types:     $LAYER_TYPE"
echo ""

if [ "$PHASE3_DATASET" = "safety" ]; then
    echo "🔐 Safety Training Mode:"
    echo "  - Using HuggingFace Trainer (Non-Freeze)"
    echo "  - All params trainable, WaRP masking via forward"
    echo "  - Automatic gradient blocking (mask=1)"
elif [ "$PHASE3_DATASET" = "gsm8k" ]; then
    echo "📚 Utility Training Mode (GSM8K):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Math reasoning (GSM8K) learning"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    echo "📚 Utility Training Mode (MetaMath):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Advanced math reasoning (MetaMath) learning"
elif [ "$PHASE3_DATASET" = "math" ]; then
    echo "📚 Utility Training Mode (Hendrycks MATH):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Subject filter: $MATH_SUBJECTS, Level filter: $MATH_LEVELS"
fi

echo ""
echo "========================================================================"
