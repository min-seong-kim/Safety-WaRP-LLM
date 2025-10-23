#!/bin/bash

# Safety-WaRP-LLM: Phase 1 - Basis Construction
# 안전 데이터로부터 FFN down_proj 활성화 기반 basis 계산

set -e  # 에러 발생시 스크립트 중단

echo "=================================================="
echo "Safety-WaRP-LLM: Phase 1 - Basis Construction"
echo "=================================================="

# 설정
PHASE=1
MODEL_NAME="meta-llama/Llama-3-8B"
SAFETY_SAMPLES=50  # 테스트용 작은 샘플 수
BATCH_SIZE=4
DEVICE="cuda:0"
SEED=42
DEBUG=false

# 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
OUTPUT_DIR="$PROJECT_ROOT/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs"

# 선택적 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAFETY_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Phase: $PHASE"
echo "  Model: $MODEL_NAME"
echo "  Safety Samples: $SAFETY_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Seed: $SEED"
echo "  Debug: $DEBUG"
echo ""

# Python 환경 확인
echo "[*] Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "[*] Running Phase 1..."
echo "=================================================="

# Phase 1 실행
python "$PROJECT_ROOT/train.py" \
    --phase "$PHASE" \
    --model_name "$MODEL_NAME" \
    --safety_samples "$SAFETY_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
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
echo "✓ Phase 1 completed!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: $OUTPUT_DIR"
echo "  - Logs: $LOG_DIR"
echo ""
