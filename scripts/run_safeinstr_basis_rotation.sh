#!/bin/bash
# ============================================================================
# safeInstr + WaRP Basis Rotation 실험 스크립트
#
# 파이프라인:
#   1) (선택) WaRP Phase 1: safety-tuned 모델에서 SVD basis 구성
#   2) WaRP Phase 2: safety dataset으로 importance mask 계산 (keep_ratio=0.1)
#   3) WaRP Phase 3 (non-freeze): GSM8K downstream FT
#      + safety dataset (circuit_breakers) X% 혼합 (safeInstr)
#      + WaRP basis rotation 적용 (rotated space에서 학습)
#      + Phase 2 mask 적용 (상위 10% safety-important 파라미터 freeze)
#
# 컨셉:
#   - 기존 safeInstr: full param FT + safety data mixing
#   - 본 스크립트:    WaRP basis-rotated space FT + safety data mixing
#     → safety basis 방향을 학습 공간으로 활용하면서 안전성 데이터를 명시적으로 혼합
#
# 비교 실험 (3-way):
#   A) Baseline GSM8K FT           (safety 없음)
#   B) safeInstr                   (GSM8K + X% safety, freeze 없음)
#   C) WaRP basis-rotated safeInstr (본 스크립트: rotated space + safety mixing)
# ============================================================================

source /home/yonsei_jong/miniconda3/etc/profile.d/conda.sh
conda activate hb
set -e
set -o pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "========================================================================"
echo "safeInstr + WaRP Basis Rotation Experiment"
echo "========================================================================"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Phase 0 모델 (safety-tuned → Phase 1 basis 구성 원점이자 Phase 3 시작 모델)
PHASE0_MODEL="kmseong/llama2_7b-Safety-FT-lr3e-5"

# WaRP Phase 1 basis 경로
# 기존 basis가 있으면 SKIP_PHASE1=true + EXISTING_BASIS_DIR 지정
SKIP_PHASE1=true
EXISTING_BASIS_DIR="/home/yonsei_jong/Safety-WaRP-LLM/checkpoints/phase1_20260426_232842/basis"

# Phase 1 설정 (SKIP_PHASE1=false일 때 사용)
PHASE1_DATASET="circuit_breakers"
PHASE1_SAMPLES=4994
PHASE1_LAYER_TYPE="attn_q,attn_k,attn_v,ffn_up,ffn_down"
PHASE1_BATCH_SIZE=4

# WaRP Phase 2 설정 (importance mask)
SKIP_PHASE2=false
EXISTING_MASKS_DIR=""
PHASE2_DATASET="circuit_breakers"    # safety dataset으로 importance scoring
PHASE2_KEEP_RATIO=0.1                # 상위 10% safety-important 파라미터 freeze
PHASE2_SAMPLES=4994
PHASE2_BATCH_SIZE=4

# safeInstr 설정
SAFEINSTR_RATIO=0.05         # 0.0 = safety data 혼합 없음 (GSM8K only), 0.05 = 5% 혼합
CIRCUIT_BREAKERS_PATH="./data/circuit_breakers_train.json"

# Phase 3 downstream task 설정
PHASE3_DATASET="gsm8k"
GSM8K_SAMPLES=0             # 0 = 전체 (7,473 samples)

# 공통 학습 설정
EPOCHS=3
LEARNING_RATE="3e-5"
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
DTYPE="bfloat16"
DEVICE="cuda"
TARGET_LAYERS="all"
# ⚠ LAYER_TYPE은 반드시 사용하는 basis가 포함한 layer type과 일치해야 함
# phase1_20260426_232842/basis 에는 attn_q,attn_k,attn_v,ffn_down,ffn_up 만 존재
# attn_o,ffn_gate를 추가하려면 Phase 1을 해당 layer type으로 재실행 후 basis 교체 필요
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"

# 출력 경로
BASE_OUTPUT_DIR="$(realpath -m "$REPO_DIR/checkpoints")"
LOG_DIR="$(realpath -m "$REPO_DIR/logs")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Phase 0 Model     : $PHASE0_MODEL"
echo "  Skip Phase 1      : $SKIP_PHASE1"
echo "  Skip Phase 2      : $SKIP_PHASE2"
echo "  Phase 2 Dataset   : $PHASE2_DATASET  (keep_ratio=$PHASE2_KEEP_RATIO)"
echo "  safeInstr Ratio   : $SAFEINSTR_RATIO ($(echo "$SAFEINSTR_RATIO * 100" | bc -l | xargs printf '%.0f')%)"
echo "  Circuit Breakers  : $CIRCUIT_BREAKERS_PATH"
echo "  Phase 3 Dataset   : $PHASE3_DATASET (samples=$GSM8K_SAMPLES)"
echo "  Epochs            : $EPOCHS"
echo "  Learning Rate     : $LEARNING_RATE"
echo "  Batch Size        : $BATCH_SIZE  (grad_accum=$GRAD_ACCUM_STEPS)"
echo "  Layer Type        : $LAYER_TYPE"
echo "  Output Dir        : $BASE_OUTPUT_DIR"
echo ""

# ============================================================================
# Step 1: WaRP Phase 1 — Safety Basis 구성
# ============================================================================
echo "========================================================================"
echo "STEP 1: WaRP Phase 1 — Safety Basis Construction"
echo "========================================================================"

if [ "$SKIP_PHASE1" = "true" ]; then
    if [ -z "$EXISTING_BASIS_DIR" ] || [ ! -d "$EXISTING_BASIS_DIR" ]; then
        echo "❌ ERROR: SKIP_PHASE1=true이지만 EXISTING_BASIS_DIR이 없습니다: $EXISTING_BASIS_DIR"
        exit 1
    fi
    BASIS_DIR="$(realpath "$EXISTING_BASIS_DIR")"
    echo "⏭  Phase 1 건너뜀 (기존 basis 사용)"
    echo "   Basis Dir: $BASIS_DIR"
else
    echo "▶  Phase 1 실행 중..."

    if [ "$PHASE1_DATASET" = "circuit_breakers" ]; then
        PHASE1_DATASET_ARG="--circuit_breakers_samples_phase1 $PHASE1_SAMPLES"
    elif [ "$PHASE1_DATASET" = "wikipedia" ]; then
        PHASE1_DATASET_ARG="--wikipedia_samples_phase1 $PHASE1_SAMPLES"
    else
        echo "❌ ERROR: Unknown Phase 1 dataset: $PHASE1_DATASET"
        exit 1
    fi

    python train.py \
        --phase 1 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --safety_dataset "$PHASE1_DATASET" \
        $PHASE1_DATASET_ARG \
        --batch_size $PHASE1_BATCH_SIZE \
        --layer_type "$PHASE1_LAYER_TYPE" \
        --target_layers "$TARGET_LAYERS" \
        --output_dir "$BASE_OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed 42 \
        --no_wandb \
        2>&1 | tee "$LOG_DIR/safeinstr_phase1_${TIMESTAMP}.log"

    # 최신 phase1 디렉토리에서 basis 경로 추출
    PHASE1_OUTPUT_DIR=$(find "$BASE_OUTPUT_DIR" -maxdepth 1 -name "phase1_*" -type d \
        -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    BASIS_DIR="$(realpath "$PHASE1_OUTPUT_DIR/basis")"

    if [ ! -d "$BASIS_DIR" ]; then
        echo "❌ ERROR: Phase 1 basis 디렉토리가 생성되지 않았습니다: $BASIS_DIR"
        exit 1
    fi

    echo "✅ Phase 1 완료"
    echo "   Basis Dir: $BASIS_DIR"
fi

echo ""

# ============================================================================
# Step 2: WaRP Phase 2 — Importance Mask 계산
# ============================================================================
echo "========================================================================"
echo "STEP 2: WaRP Phase 2 — Importance Scoring (dataset=$PHASE2_DATASET, keep_ratio=$PHASE2_KEEP_RATIO)"
echo "========================================================================"

if [ "$SKIP_PHASE2" = "true" ]; then
    if [ -z "$EXISTING_MASKS_DIR" ] || [ ! -d "$EXISTING_MASKS_DIR" ]; then
        echo "❌ ERROR: SKIP_PHASE2=true이지만 EXISTING_MASKS_DIR이 없습니다: $EXISTING_MASKS_DIR"
        exit 1
    fi
    MASKS_DIR="$(realpath "$EXISTING_MASKS_DIR")"
    echo "⏭  Phase 2 건너뜀 (기존 masks 사용)"
    echo "   Masks Dir: $MASKS_DIR"
else
    echo "▶  Phase 2 실행 중 (importance scoring)..."

    if [ "$PHASE2_DATASET" = "circuit_breakers" ]; then
        PHASE2_DATASET_ARG="--circuit_breakers_samples_phase2 $PHASE2_SAMPLES"
    elif [ "$PHASE2_DATASET" = "wikipedia" ]; then
        PHASE2_DATASET_ARG="--wikipedia_samples_phase2 $PHASE2_SAMPLES"
    else
        echo "❌ ERROR: Unknown Phase 2 dataset: $PHASE2_DATASET"
        exit 1
    fi

    PHASE2_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    python train.py \
        --phase 2 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --basis_dir "$BASIS_DIR" \
        --dataset_phase2 "$PHASE2_DATASET" \
        $PHASE2_DATASET_ARG \
        --circuit_breakers_path "$CIRCUIT_BREAKERS_PATH" \
        --keep_ratio $PHASE2_KEEP_RATIO \
        --batch_size $PHASE2_BATCH_SIZE \
        --max_length 1024 \
        --layer_type "$LAYER_TYPE" \
        --target_layers "$TARGET_LAYERS" \
        --output_dir "$BASE_OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --perlayer \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed 42 \
        --no_wandb \
        2>&1 | tee "$LOG_DIR/safeinstr_phase2_${PHASE2_TIMESTAMP}.log"

    # 최신 phase2 디렉토리에서 masks 경로 추출
    # save_masks() 저장 구조: {output_dir}/phase2_{ts}/checkpoints/masks
    PHASE2_OUTPUT_DIR=$(find "$BASE_OUTPUT_DIR" -maxdepth 1 -name "phase2_*" -type d \
        -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    MASKS_DIR="$(realpath "$PHASE2_OUTPUT_DIR/checkpoints/masks")"

    if [ ! -d "$MASKS_DIR" ]; then
        echo "❌ ERROR: Phase 2 masks 디렉토리가 생성되지 않았습니다: $MASKS_DIR"
        exit 1
    fi

    echo "✅ Phase 2 완료"
    echo "   Masks Dir: $MASKS_DIR"
fi

echo ""

# ============================================================================
# Step 3: WaRP Phase 3 (non-freeze + safeInstr)
#   - Phase 2 건너뜀 (--no_masks → 전체 파라미터 학습 가능)
#   - Phase 0 모델 → WaRP 모듈로 변환 후 rotated space에서 FT
#   - GSM8K + SAFEINSTR_RATIO * safety data 혼합
# ============================================================================
echo "========================================================================"
echo "STEP 3: WaRP Phase 3 — safeInstr (non-freeze, rotated space)"
echo "========================================================================"
echo ""
echo "  ✦ --masks_dir       : Phase 2 mask 적용 (keep_ratio=$PHASE2_KEEP_RATIO, $PHASE2_DATASET)"
echo "  ✦ --non_freeze      : non-freeze 모드 (WaRP 외 레이어도 학습)"
echo "  ✦ --safety_mix_ratio: downstream의 ${SAFEINSTR_RATIO}비율 safety data 혼합"
echo ""

PHASE3_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PHASE3_OUTPUT_DIR="$(realpath -m "$BASE_OUTPUT_DIR/safeinstr_warp_phase2mask_${PHASE3_TIMESTAMP}")"
mkdir -p "$PHASE3_OUTPUT_DIR"

# phase3_dataset 인수 조립
if [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $GSM8K_SAMPLES"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset metamath --metamath_samples 0"
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset math --math_samples 0"
else
    echo "❌ ERROR: 지원하지 않는 Phase 3 dataset: $PHASE3_DATASET"
    exit 1
fi

python train.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --masks_dir "$MASKS_DIR" \
    --non_freeze \
    $PHASE3_DATASET_ARG \
    --safety_mix_ratio $SAFEINSTR_RATIO \
    --circuit_breakers_path "$CIRCUIT_BREAKERS_PATH" \
    --epochs $EPOCHS \
    --utility_lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --layer_type "$LAYER_TYPE" \
    --target_layers "$TARGET_LAYERS" \
    --output_dir "$PHASE3_OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed 42 \
    --no_wandb \
    2>&1 | tee "$LOG_DIR/safeinstr_phase3_${PHASE3_TIMESTAMP}.log"

echo ""
echo "========================================================================"
echo "✅ safeInstr + WaRP Basis Rotation (Phase 2 mask) 실험 완료"
echo "========================================================================"
echo ""
echo "  Phase 3 모델 저장 위치: $PHASE3_OUTPUT_DIR"
echo "  로그 파일: $LOG_DIR/safeinstr_phase3_${PHASE3_TIMESTAMP}.log"
echo ""
echo "다음 단계 (평가):"
echo "  1. GSM8K 평가   : python -m eval.gsm8k --model $PHASE3_OUTPUT_DIR"
echo "  2. HarmBench 평가: (HarmBench 저장소에서) python generate_completions.py ..."
echo "========================================================================"
