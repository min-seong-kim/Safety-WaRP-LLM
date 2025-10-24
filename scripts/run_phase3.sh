#!/bin/bash

# Safety-WaRP-LLM: Phase 3 - Incremental Learning
# Phase 1과 Phase 2 결과를 사용하여 GSM8K로 미세조정

set -e

echo "=================================================="
echo "Safety-WaRP-LLM: Phase 3 - Incremental Learning"
echo "=================================================="

# 설정
PHASE=3
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
UTILITY_SAMPLES=1000    # GSM8K train split 샘플 수
EPOCHS=3                # 훈련 에포크
BATCH_SIZE=2            # 미세조정은 배치 크기를 작게
DEVICE="cuda:0"
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
SEED=42
DEBUG=false

# Phase 1 basis + Phase 2 masks 디렉토리 (자동 감지 또는 수동 지정)
BASIS_DIR="${1:-}"
MASKS_DIR="${2:-}"

# 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
OUTPUT_DIR="$PROJECT_ROOT/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs"

# Phase 1/2 결과가 없으면 가장 최근 디렉토리 찾기
if [ -z "$BASIS_DIR" ] || [ ! -d "$BASIS_DIR" ]; then
    echo "[*] Finding most recent Phase 1 basis directory..."
    PHASE1_DIR=$(ls -td "$OUTPUT_DIR"/phase1_* 2>/dev/null | head -1)
    if [ -z "$PHASE1_DIR" ]; then
        echo "ERROR: No Phase 1 basis directory found!"
        echo "Please run Phase 1 first: bash scripts/run_phase1.sh"
        exit 1
    fi
    BASIS_DIR="$PHASE1_DIR/checkpoints/basis"
fi

if [ -z "$MASKS_DIR" ] || [ ! -d "$MASKS_DIR" ]; then
    echo "[*] Finding most recent Phase 2 masks directory..."
    PHASE2_DIR=$(ls -td "$OUTPUT_DIR"/phase2_* 2>/dev/null | head -1)
    if [ -z "$PHASE2_DIR" ]; then
        echo "ERROR: No Phase 2 masks directory found!"
        echo "Please run Phase 2 first: bash scripts/run_phase2.sh"
        exit 1
    fi
    MASKS_DIR="$PHASE2_DIR/checkpoints/masks"
fi

# 경로 확인
if [ ! -d "$BASIS_DIR" ]; then
    echo "ERROR: Basis directory not found: $BASIS_DIR"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "ERROR: Masks directory not found: $MASKS_DIR"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Phase: $PHASE"
echo "  Model: $MODEL_NAME"
echo "  Basis Directory: $BASIS_DIR"
echo "  Masks Directory: $MASKS_DIR"
echo "  Utility Samples: $UTILITY_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo "  Seed: $SEED"
echo "  Debug: $DEBUG"
echo ""

# Python 환경 확인
echo "[*] Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "[*] Running Phase 3..."
echo "=================================================="

# Phase 3 실행
cd "$PROJECT_ROOT"
python train.py \
    --phase "$PHASE" \
    --model_name "$MODEL_NAME" \
    --basis_dir "$BASIS_DIR" \
    --masks_dir "$MASKS_DIR" \
    --utility_samples "$UTILITY_SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --target_layers "all" \
    --layer_type "ffn_down" \
    --device "$DEVICE" \
    --dtype "bfloat16" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --seed "$SEED" \
    $([ "$DEBUG" = true ] && echo "--debug")

echo ""
echo "=================================================="
echo "✓ Phase 3 completed!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: $OUTPUT_DIR"
echo "  - Logs: $LOG_DIR"
echo ""
echo "Next step: Evaluate model using safety_evaluator.py"
echo ""
