#!/bin/bash

# Safety-WaRP-LLM: Phase 3 + HuggingFace 자동 업로드
# 이 스크립트는 Phase 3를 완료하고 자동으로 HuggingFace Hub에 업로드합니다.

set -e

echo "=================================================="
echo "Safety-WaRP-LLM: Phase 3 + HuggingFace Upload"
echo "=================================================="

# 설정
PHASE=3
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
UTILITY_SAMPLES=${1:-100}  # 기본값 100
BATCH_SIZE=2
EPOCHS=3
LEARNING_RATE=1e-5
DEVICE="cuda:0"
SEED=42

# HuggingFace 설정
HF_USERNAME=${HF_USERNAME:-kmseong}
HF_MODEL_NAME=${HF_MODEL_NAME:-WaRP-Safety-Llama3_8B_Instruct}
HF_MODEL_ID="$HF_USERNAME/$HF_MODEL_NAME"
HF_PRIVATE=${HF_PRIVATE:-false}

# 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
OUTPUT_DIR="$PROJECT_ROOT/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs"

# Phase 1 basis 디렉토리 자동 감지
PHASE1_DIR=$(ls -td "$OUTPUT_DIR"/phase1_* 2>/dev/null | head -1)
if [ -z "$PHASE1_DIR" ]; then
    echo "ERROR: No Phase 1 basis directory found!"
    echo "Please run Phase 1 first: bash scripts/run_phase1.sh"
    exit 1
fi
BASIS_PATH="$PHASE1_DIR/checkpoints/basis"

# Phase 2 masks 디렉토리 자동 감지
PHASE2_DIR=$(ls -td "$OUTPUT_DIR"/phase2_* 2>/dev/null | head -1)
if [ -z "$PHASE2_DIR" ]; then
    echo "ERROR: No Phase 2 masks directory found!"
    echo "Please run Phase 2 first: bash scripts/run_phase2.sh"
    exit 1
fi
MASKS_PATH="$PHASE2_DIR/checkpoints/masks"

# HuggingFace 토큰 확인
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "WARNING: HUGGINGFACE_TOKEN environment variable not set"
    echo "Please set: export HUGGINGFACE_TOKEN='your_token'"
    echo ""
    echo "Continuing without automatic upload..."
    PUSH_TO_HUB=false
else
    PUSH_TO_HUB=true
    echo "✓ HuggingFace token found"
fi

echo ""
echo "Configuration:"
echo "  Phase: $PHASE"
echo "  Model: $MODEL_NAME"
echo "  Basis: $BASIS_PATH"
echo "  Masks: $MASKS_PATH"
echo "  Utility Samples: $UTILITY_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo "  Seed: $SEED"
if [ "$PUSH_TO_HUB" = true ]; then
    echo "  HuggingFace:"
    echo "    - Model ID: $HF_MODEL_ID"
    echo "    - Private: $HF_PRIVATE"
fi
echo ""

# Python 환경 확인
echo "[*] Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "[*] Running Phase 3 with HuggingFace upload..."
echo "=================================================="

# Phase 3 실행
cd "$PROJECT_ROOT"

if [ "$PUSH_TO_HUB" = true ]; then
    python train.py \
        --phase "$PHASE" \
        --model_name "$MODEL_NAME" \
        --basis_dir "$BASIS_PATH" \
        --masks_dir "$MASKS_PATH" \
        --utility_samples "$UTILITY_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --target_layers "all" \
        --layer_type "ffn_down" \
        --device "$DEVICE" \
        --dtype "bfloat16" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --seed "$SEED" \
        --push_to_hub \
        --hub_model_id "$HF_MODEL_ID" \
        --hf_token "$HUGGINGFACE_TOKEN" \
        $([ "$HF_PRIVATE" = true ] && echo "--hub_private")
else
    python train.py \
        --phase "$PHASE" \
        --model_name "$MODEL_NAME" \
        --basis_dir "$BASIS_PATH" \
        --masks_dir "$MASKS_PATH" \
        --utility_samples "$UTILITY_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --target_layers "all" \
        --layer_type "ffn_down" \
        --device "$DEVICE" \
        --dtype "bfloat16" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --seed "$SEED"
fi

echo ""
echo "=================================================="
if [ "$PUSH_TO_HUB" = true ]; then
    echo "✓ Phase 3 completed with HuggingFace upload!"
    echo "Model available at: https://huggingface.co/$HF_MODEL_ID"
else
    echo "✓ Phase 3 completed!"
    echo "To upload to HuggingFace, set HUGGINGFACE_TOKEN and run:"
    echo "  bash scripts/run_phase3_with_upload.sh"
fi
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: $OUTPUT_DIR"
echo "  - Logs: $LOG_DIR"
echo ""
