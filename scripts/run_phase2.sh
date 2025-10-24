#!/bin/bash

# Safety-WaRP-LLM: Phase 2 - Importance Scoring
# Phase 1에서 저장된 basis를 사용하여 중요한 가중치 방향을 식별

set -e

echo "=================================================="
echo "Safety-WaRP-LLM: Phase 2 - Importance Scoring"
echo "=================================================="

# 설정
PHASE=2
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
SAFETY_SAMPLES=50  # 테스트용
BATCH_SIZE=4
DEVICE="cuda:0"
KEEP_RATIO=0.1
SEED=42
DEBUG=false

# Phase 1 basis 디렉토리 (자동 감지 또는 수동 지정)
PHASE1_DIR="${1:-}"

# 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
OUTPUT_DIR="$PROJECT_ROOT/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs"

# Phase 1 기본 디렉토리가 없으면 가장 최근 디렉토리 찾기
if [ -z "$PHASE1_DIR" ] || [ ! -d "$PHASE1_DIR" ]; then
    echo "[*] Finding most recent Phase 1 basis directory..."
    PHASE1_DIR=$(ls -td "$OUTPUT_DIR"/phase1_* 2>/dev/null | head -1)
    if [ -z "$PHASE1_DIR" ]; then
        echo "ERROR: No Phase 1 basis directory found!"
        echo "Please specify basis directory: bash scripts/run_phase2.sh /path/to/basis"
        exit 1
    fi
fi

# Phase 1 basis 경로 확인
BASIS_PATH="$PHASE1_DIR/checkpoints/basis"
if [ ! -d "$BASIS_PATH" ]; then
    echo "ERROR: Basis directory not found: $BASIS_PATH"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Phase: $PHASE"
echo "  Model: $MODEL_NAME"
echo "  Basis Directory: $BASIS_PATH"
echo "  Safety Samples: $SAFETY_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Keep Ratio: $KEEP_RATIO"
echo "  Device: $DEVICE"
echo "  Seed: $SEED"
echo "  Debug: $DEBUG"
echo ""

# Python 환경 확인
echo "[*] Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "[*] Running Phase 2..."
echo "=================================================="

# Phase 2 실행
cd "$PROJECT_ROOT"
python train.py \
    --phase "$PHASE" \
    --model_name "$MODEL_NAME" \
    --basis_dir "$BASIS_PATH" \
    --safety_samples "$SAFETY_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --target_layers "all" \
    --layer_type "ffn_down" \
    --keep_ratio "$KEEP_RATIO" \
    --device "$DEVICE" \
    --dtype "bfloat16" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --seed "$SEED" \
    $([ "$DEBUG" = true ] && echo "--debug")

echo ""
echo "=================================================="
echo "✓ Phase 2 completed!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: $OUTPUT_DIR"
echo "  - Logs: $LOG_DIR"
echo ""
