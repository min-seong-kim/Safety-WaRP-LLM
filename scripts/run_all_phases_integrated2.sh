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
export CUDA_VISIBLE_DEVICES=5

echo "========================================================================"
echo "Safety-WaRP-LLM: Complete Training Pipeline (Integrated)"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

# Phase 0 모델
PHASE0_MODEL="kmseong/llama3_2_3b-instruct-SSFT-lr5e-5"  


# Phase 1: Basis Construction
# ==============================
# Dataset 선택 (Safety 또는 Utility)
# Options: circuit_breakers, wikipedia
PHASE1_DATASET="circuit_breakers"
PHASE1_SAMPLES=4994
# 기존 basis가 있으면 Phase 1 스킵 (빈 문자열이면 Phase 1 수행)
PHASE1_BASIS_DIR_OVERRIDE="/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong/Safety-WaRP-LLM/checkpoints/phase1_20260505_232643/basis"


# Phase 2: Importance Scoring
# ==============================
# Dataset 선택 (동일하게 사용)
PHASE2_DATASET="circuit_breakers"
PHASE2_SAMPLES=4994
KEEP_RATIO_LIST=("0.1")  

# Two-Mask 설정 (비활성화하려면 TWO_MASK="" 로 설정)
# preserve_mask AND NOT adapt_mask → adapt에 중요한 파라미터는 Phase 3에서 학습 가능
# TWO_MASK=""           # "" = 비활성화 (기본), "true" = 활성화
# ADAPT_DATASET="math" # adapt 데이터셋: gsm8k, math, metamath, wikipedia, safety
# ADAPT_SAMPLES=4994       # 0=전체

# Phase 3: Incremental Learning
# ==============================
# Dataset 선택 (Utility 또는 Safety)
PHASE3_DATASET="math" # Options: safety, gsm8k, metamath, math, agnews, medqa

# SafeInstr: safety data mixing (0.0 = 비활성화, 0.1 = 학습 데이터의 10%)
SAFEINSTR_RATIO=0.1
CIRCUIT_BREAKERS_PATH="./data/circuit_breakers_train.json"

# Phase3=MATH 설정
MATH_SUBJECTS="all"  # 예: Algebra,Geometry
MATH_LEVELS="all"    # 예: 1,2,3,4,5

# Phase3=AGNEWS 설정
AGNEWS_DATASET_PATH="/home/yonsei_jong/Safety-WaRP-LLM/data/agnews_train_8000.jsonl"   # --agnews_dataset_path 필수 (agnews 선택 시)
AGNEWS_SAMPLES=8000      # 0=전체

# Phase3=MEDQA 설정
MEDQA_DATASET_PATH="/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong/Safety-WaRP-LLM/data/medqa_train_10178.jsonl"   # --medqa_dataset_path 필수 (medqa 선택 시)
MEDQA_SAMPLES=0      # 0=전체

if [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_SAMPLES=4994
elif [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_SAMPLES=0  # 0 = all samples
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_SAMPLES=10000  # 0 = all samples
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_SAMPLES=0  # 0 = all samples
elif [ "$PHASE3_DATASET" = "agnews" ]; then
    PHASE3_SAMPLES=$AGNEWS_SAMPLES
elif [ "$PHASE3_DATASET" = "medqa" ]; then
    PHASE3_SAMPLES=$MEDQA_SAMPLES
fi

# 공통 설정
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
DTYPE="bfloat16"
DEVICE="cuda"
EPOCHS=3
# LR_LIST=("1e-5" "3e-5" "5e-5")
LR_LIST=("5e-5")  
TARGET_LAYERS="all"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
BASE_OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BASE_OUTPUT_DIR
mkdir -p $LOG_DIR

echo "Configuration:"
echo "  Phase 0 Model: $PHASE0_MODEL"
echo "  Phase 1 Dataset: $PHASE1_DATASET (samples=$PHASE1_SAMPLES)"
echo "  Phase 2 Dataset: $PHASE2_DATASET (samples=$PHASE2_SAMPLES)"
echo "  Phase 3 Dataset: $PHASE3_DATASET (samples=$PHASE3_SAMPLES)"  echo "  SafeInstr Ratio: $SAFEINSTR_RATIO"echo "  Keep Ratios: ${KEEP_RATIO_LIST[*]}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Output Dir: $BASE_OUTPUT_DIR"
echo ""

# Phase 3 Dataset validation
if [[ ! "$PHASE3_DATASET" =~ ^(safety|gsm8k|metamath|math|agnews|medqa)$ ]]; then
    echo "❌ ERROR: Unknown Phase 3 dataset: $PHASE3_DATASET"
    echo "Choose from: safety, gsm8k, metamath, math, agnews, medqa"
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

if [ -n "$PHASE1_BASIS_DIR_OVERRIDE" ]; then
    PHASE1_BASIS_DIR="$PHASE1_BASIS_DIR_OVERRIDE"
    if [ ! -d "$PHASE1_BASIS_DIR" ]; then
        echo "❌ ERROR: PHASE1_BASIS_DIR_OVERRIDE not found: $PHASE1_BASIS_DIR"
        exit 1
    fi
    echo "✅ Phase 1 skipped (existing basis provided)"
    echo "   Using basis: $PHASE1_BASIS_DIR"
    echo ""
else
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
fi

# ========================================================================
# Phase 2 & Phase 3: Keep Ratio Sweep
# Phase 1 basis 공유, keep_ratio마다 Phase 2(mask 생성) → Phase 3(학습) 수행
# ========================================================================

if [ "$PHASE2_DATASET" = "circuit_breakers" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 $PHASE2_SAMPLES"
elif [ "$PHASE2_DATASET" = "wikipedia" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 wikipedia --wikipedia_samples_phase2 $PHASE2_SAMPLES"
else
    echo "ERROR: Unknown Phase 2 dataset: $PHASE2_DATASET"
    exit 1
fi

if [ -n "$TWO_MASK" ]; then
    TWO_MASK_ARG="--two_mask --adapt_dataset_phase2 $ADAPT_DATASET --adapt_samples_phase2 $ADAPT_SAMPLES"
    echo "[Two-Mask] Enabled: adapt_dataset=$ADAPT_DATASET, adapt_samples=$ADAPT_SAMPLES"
else
    TWO_MASK_ARG=""
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset safety --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase3 $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset metamath --metamath_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset math --math_samples $PHASE3_SAMPLES --math_subjects $MATH_SUBJECTS --math_levels $MATH_LEVELS"
elif [ "$PHASE3_DATASET" = "agnews" ]; then
    if [ -z "$AGNEWS_DATASET_PATH" ]; then
        echo "ERROR: AGNEWS_DATASET_PATH must be set when PHASE3_DATASET=agnews"
        exit 1
    fi
    PHASE3_DATASET_ARG="--phase3_dataset agnews --agnews_dataset_path $AGNEWS_DATASET_PATH --agnews_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "medqa" ]; then
    if [ -z "$MEDQA_DATASET_PATH" ]; then
        echo "ERROR: MEDQA_DATASET_PATH must be set when PHASE3_DATASET=medqa"
        exit 1
    fi
    PHASE3_DATASET_ARG="--phase3_dataset medqa --medqa_dataset_path $MEDQA_DATASET_PATH --medqa_samples $PHASE3_SAMPLES"
else
    echo "ERROR: Unknown Phase 3 dataset: $PHASE3_DATASET"
    exit 1
fi

PHASE3_OUTPUT_DIRS=()

for KEEP_RATIO in "${KEEP_RATIO_LIST[@]}"; do
    KR_SAFE=$(echo "$KEEP_RATIO" | sed 's/[^a-zA-Z0-9_-]/_/g')

    # ------------------------------------------------------------------
    # Phase 2: Importance Scoring (keep_ratio=$KEEP_RATIO)
    # ------------------------------------------------------------------
    echo ""
    echo "========================================================================"
    echo "PHASE 2: Importance Scoring  (keep_ratio=$KEEP_RATIO)"
    echo "========================================================================"
    echo ""

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
        2>&1 | tee $LOG_DIR/phase2_kr${KR_SAFE}_${TIMESTAMP}.log

    PHASE2_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase2_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    PHASE2_MASKS_DIR="$PHASE2_OUTPUT_DIR/checkpoints/masks"

    if [ ! -d "$PHASE2_MASKS_DIR" ]; then
        echo "ERROR: Phase 2 (kr=$KEEP_RATIO) masks not found: $PHASE2_MASKS_DIR"
        exit 1
    fi

    echo ""
    echo "Phase 2 (kr=$KEEP_RATIO) completed: $PHASE2_MASKS_DIR"
    echo ""

    # ------------------------------------------------------------------
    # Phase 3: Incremental Learning (keep_ratio=$KEEP_RATIO, LR sweep)
    # ------------------------------------------------------------------
    echo "========================================================================"
    echo "PHASE 3: Incremental Learning  (keep_ratio=$KEEP_RATIO, LR: ${LR_LIST[*]})"
    echo "========================================================================"
    echo ""

    for LEARNING_RATE in "${LR_LIST[@]}"; do
        LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')

        echo "──────────────────────────────────────────────────────────────────────"
        echo "  Phase 3: keep_ratio=$KEEP_RATIO  LR=$LEARNING_RATE"
        echo "──────────────────────────────────────────────────────────────────────"

        # SafeInstr 인자 구성
        if (( $(echo "$SAFEINSTR_RATIO > 0" | bc -l) )); then
            SAFEINSTR_ARG="--safety_mix_ratio $SAFEINSTR_RATIO --circuit_breakers_path $CIRCUIT_BREAKERS_PATH"
        else
            SAFEINSTR_ARG=""
        fi

        python train.py \
            --phase 3 \
            --phase0_model_dir "$PHASE0_MODEL" \
            --basis_dir "$PHASE1_BASIS_DIR" \
            --masks_dir "$PHASE2_MASKS_DIR" \
            $PHASE3_DATASET_ARG \
            --epochs $EPOCHS \
            --utility_lr $LEARNING_RATE \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --layer_type "$LAYER_TYPE" \
            --target_layers $TARGET_LAYERS \
            --output_dir $BASE_OUTPUT_DIR \
            --log_dir $LOG_DIR \
            --device $DEVICE \
            --dtype $DTYPE \
            --seed 42 \
            --non_freeze \
            $SAFEINSTR_ARG \
            2>&1 | tee $LOG_DIR/phase3_kr${KR_SAFE}_lr${LR_SAFE}_${TIMESTAMP}.log

        PHASE3_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase3_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

        if [ ! -d "$PHASE3_OUTPUT_DIR" ]; then
            echo "WARNING: Phase 3 (kr=$KEEP_RATIO, LR=$LEARNING_RATE) output not found"
        else
            echo "Phase 3 (kr=$KEEP_RATIO, LR=$LEARNING_RATE) completed: $PHASE3_OUTPUT_DIR"
            PHASE3_OUTPUT_DIRS+=("kr${KEEP_RATIO}_lr${LEARNING_RATE}:$PHASE3_OUTPUT_DIR")
        fi
        echo ""
    done

done

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "========================================================================"
echo "Complete Pipeline Finished!"
echo "========================================================================"
echo ""
echo "Phase 1 Basis:  $PHASE1_BASIS_DIR"
echo ""
echo "Phase 3 Models (keep_ratio x LR):"
for entry in "${PHASE3_OUTPUT_DIRS[@]}"; do
    label="${entry%%:*}"
    dir="${entry#*:}"
    echo "  [$label]  $dir/final_model"
done
echo ""
echo "Logs:"
echo "  Phase 1: $LOG_DIR/phase1_${TIMESTAMP}.log"
for KEEP_RATIO in "${KEEP_RATIO_LIST[@]}"; do
    KR_SAFE=$(echo "$KEEP_RATIO" | sed 's/[^a-zA-Z0-9_-]/_/g')
    echo "  Phase 2 (kr=$KEEP_RATIO): $LOG_DIR/phase2_kr${KR_SAFE}_${TIMESTAMP}.log"
    for LEARNING_RATE in "${LR_LIST[@]}"; do
        LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')
        echo "  Phase 3 (kr=$KEEP_RATIO, LR=$LEARNING_RATE): $LOG_DIR/phase3_kr${KR_SAFE}_lr${LR_SAFE}_${TIMESTAMP}.log"
    done
done
echo ""
echo "Configuration:"
echo "  - Phase 1/2 Dataset: $PHASE1_DATASET"
echo "  - Phase 3 Dataset:   $PHASE3_DATASET"
echo "  - Keep Ratios:       ${KEEP_RATIO_LIST[*]}"
echo "  - Learning Rates:    ${LR_LIST[*]}"
echo "  - Epochs:            $EPOCHS"
echo "  - Layer Types:       $LAYER_TYPE"
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
elif [ "$PHASE3_DATASET" = "agnews" ]; then
    echo "📰 Utility Training Mode (AG News):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - News classification (AG News) learning"
    echo "  - Dataset path: $AGNEWS_DATASET_PATH, Samples: $AGNEWS_SAMPLES"
elif [ "$PHASE3_DATASET" = "medqa" ]; then
    echo "🏥 Utility Training Mode (MedQA USMLE):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Medical QA MCQ (MedQA) learning"
    echo "  - Dataset path: $MEDQA_DATASET_PATH, Samples: $MEDQA_SAMPLES"
fi

echo ""
echo "========================================================================"
