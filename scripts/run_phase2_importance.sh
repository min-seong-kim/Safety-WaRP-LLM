#!/bin/bash

# Phase 2: Importance Scoring
# ✅ model.eval() 모드, optimizer.step 제거
#
# 두 가지 데이터셋으로 importance score 계산 가능:
# 1. Safety Basis: circuit_breakers 데이터셋 사용
# 2. Utility Basis: wikipedia 데이터셋 사용

echo "========================================="
echo "Phase 2: Importance Scoring"
echo "========================================="

# Phase 0, 1 결과 경로
# PHASE0_MODEL="./checkpoints/phase0_20260213_230047"  # 로컬 디렉토리 예시
PHASE0_MODEL="kmseong/llama3.2_3b_new_SSFT_lr3e-5"
BASIS_DIR="./checkpoints/phase1_20260407_154217/basis"

# ========================================
# Dataset 선택 (수정 필요)
# ========================================
# 옵션 1: Safety Basis (circuit_breakers 데이터셋)
DATASET="circuit_breakers"
SAFETY_SAMPLES=4994
#
# 옵션 2: Utility Basis (Wikipedia 데이터셋)
# DATASET="wikipedia"
# Utility_SAMPLES=1000
# ========================================

echo ""
echo "Dataset 설정:"
if [ "$DATASET" = "circuit_breakers" ]; then
    echo "  - Type: Safety Basis (circuit_breakers)"
    echo "  - Samples: 4994 (fixed)"
    DATASET_ARG="--dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 $SAFETY_SAMPLES"
elif [ "$DATASET" = "wikipedia" ]; then
    echo "  - Type: Utility Basis (Wikipedia)"
    echo "  - Samples: $Utility_SAMPLES"
    DATASET_ARG="--dataset_phase2 wikipedia --wikipedia_samples_phase2 $Utility_SAMPLES"
else
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Choose from: circuit_breakers, wikipedia"
    exit 1
fi
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 각 WaRP layer 별로 keep ratio 적용할 거면 --perlayer 옵션 추가
# --layer_type attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_down,ffn_up
python train.py \
    --phase 2 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    $DATASET_ARG \
    --keep_ratio 0.1 \
    --batch_size 4 \
    --max_length 1024 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42 \
    --perlayer

echo ""
echo "========================================="
echo "Phase 2 완료! (Dataset: $DATASET)"
echo "다음 단계: scripts/run_phase3_learning.sh 실행"
echo "========================================="
