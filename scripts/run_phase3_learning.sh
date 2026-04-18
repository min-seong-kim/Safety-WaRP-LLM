#!/bin/bash

# Phase 3: Incremental Learning
set -e
set -o pipefail

echo "========================================="
echo "Phase 3: Incremental Learning (Fixed)"
echo "========================================="

# 이전 Phase 결과 경로 (로컬 디렉토리 또는 Hugging Face 모델 ID)
# PHASE0_MODEL="./checkpoints/phase0_20260213_230047"  # 로컬 디렉토리 예시
PHASE0_MODEL="kmseong/llama2_7b-chat-Safety-FT-lr3e-5"
BASIS_DIR="/NHNHOME/0226010080_A/kms/phase1_20260417_130943/basis"
MASKS_DIR="./checkpoints/phase2_20260417_135720/checkpoints/masks"

# PHASE0_MODEL="kmseong/llama2_7b-Safety-FT-lr3e-5"
# BASIS_DIR="/NHNHOME/0226010080_A/kms/phase1_20260417_130853/basis"
# MASKS_DIR="./checkpoints/phase2_20260417_135654/checkpoints/masks"

# ========================================
# Dataset 선택 (CONFIGURE THIS)
# ========================================
# 옵션 1: GSM8K (Utility Learning) - SFTTrainer 방식
DATASET="gsm8k"
GSM8K_SAMPLES=0

# 옵션 2: Safety (Safety Learning) - phase0_SSFT 커스텀 루프 방식
# DATASET="safety"
# CIRCUIT_BREAKERS_SAMPLES=4994

# 옵션 3: MetaMath (Utility Learning) - SFTTrainer 방식
# DATASET="metamath"
# METAMATH_SAMPLES=10000  # 0 = all samples

# 옵션 4: Hendrycks MATH (Utility Learning) - SFTTrainer 방식
# DATASET="math"
# MATH_SAMPLES=0           # 0 = all samples
# MATH_SUBJECTS="all"     # 예: Algebra,Geometry
# MATH_LEVELS="all"       # 예: 1,2,3,4,5
# 
# ========================================

# 공통 학습 설정 (run_all 스타일)
# LR_LIST=("1e-6" "5e-6" "1e-7" "5e-7")  
LR_LIST=("3e-5") 
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
TARGET_LAYERS="all"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# non_freeze 모드를 끄고 싶으면 빈 문자열로 변경
NON_FREEZE_FLAG="--non_freeze"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR"

# PHASE0_MODEL이 로컬 경로처럼 보일 때만 디렉토리 체크
if [[ "$PHASE0_MODEL" == ./* || "$PHASE0_MODEL" == /* ]]; then
    if [ ! -d "$PHASE0_MODEL" ]; then
        echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
        exit 1
    fi
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

echo ""
echo "Dataset 설정:"
if [ "$DATASET" = "gsm8k" ]; then
    echo "  - Type: Utility Learning (GSM8K)"
    echo "  - Samples: $GSM8K_SAMPLES (0=all)"
    DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $GSM8K_SAMPLES"
elif [ "$DATASET" = "safety" ]; then
    echo "  - Type: Safety Learning (Circuit Breakers)"
    echo "  - Samples: $CIRCUIT_BREAKERS_SAMPLES"
    DATASET_ARG="--phase3_dataset safety --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase3 $CIRCUIT_BREAKERS_SAMPLES"
elif [ "$DATASET" = "metamath" ]; then
    echo "  - Type: Utility Learning (MetaMath)"
    echo "  - Samples: $METAMATH_SAMPLES (0=all)"
    DATASET_ARG="--phase3_dataset metamath --metamath_samples $METAMATH_SAMPLES"
elif [ "$DATASET" = "math" ]; then
    echo "  - Type: Utility Learning (Hendrycks MATH)"
    echo "  - Samples: $MATH_SAMPLES (0=all)"
    echo "  - Subjects: $MATH_SUBJECTS"
    echo "  - Levels: $MATH_LEVELS"
    DATASET_ARG="--phase3_dataset math --math_samples $MATH_SAMPLES --math_subjects $MATH_SUBJECTS --math_levels $MATH_LEVELS"
else
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Choose from: gsm8k, safety, metamath, math"
    exit 1
fi
echo ""

echo "LR sweep: ${LR_LIST[*]}"
echo "Epochs: $EPOCHS, Batch: $BATCH_SIZE, GradAccum: $GRAD_ACCUM"
echo ""

PHASE3_OUTPUT_DIRS=()

for LEARNING_RATE in "${LR_LIST[@]}"; do
    LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')
    LOG_FILE="$LOG_DIR/phase3_${DATASET}_lr${LR_SAFE}_${TIMESTAMP}.log"

    echo "-----------------------------------------"
    echo "Phase 3: LR = $LEARNING_RATE"
    echo "-----------------------------------------"

    python train.py \
        --phase 3 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --basis_dir "$BASIS_DIR" \
        --masks_dir "$MASKS_DIR" \
        $DATASET_ARG \
        --epochs $EPOCHS \
        --utility_lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --layer_type "$LAYER_TYPE" \
        --target_layers $TARGET_LAYERS \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --device cuda \
        --dtype bfloat16 \
        --seed 42 \
        --gradient_checkpointing \
        --non_freeze \
        2>&1 | tee "$LOG_FILE"

    PHASE3_OUTPUT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "phase3_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ ! -d "$PHASE3_OUTPUT_DIR" ]; then
        echo "WARNING: Phase 3 (LR=$LEARNING_RATE) output directory not found"
    else
        echo "OK: Phase 3 (LR=$LEARNING_RATE) completed: $PHASE3_OUTPUT_DIR"
        PHASE3_OUTPUT_DIRS+=("$PHASE3_OUTPUT_DIR")
    fi
    echo ""
done

echo ""
echo "========================================="
echo "Phase 3 완료! (Dataset: $DATASET, LR sweep: ${LR_LIST[*]})"
echo "최종 모델들:"
for dir in "${PHASE3_OUTPUT_DIRS[@]}"; do
    echo "  - $dir/final_model"
done
echo "로그: $LOG_DIR/phase3_${DATASET}_lr*_${TIMESTAMP}.log"
echo ""
echo "========================================="
