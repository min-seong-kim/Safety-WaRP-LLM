#!/bin/bash

# Phase 1: Basis Construction
# ✅ Φ @ Φ^T 방식으로 SVD
# 
# 두 가지 basis 구성 가능:
# 1. Safety Basis: circuit_breakers 데이터셋 사용
# 2. Utility Basis: wikipedia 데이터셋 사용

echo "========================================="
echo "Phase 1: Basis Construction"
echo "========================================="

# Phase 0 모델 경로 (로컬 디렉토리 또는 Hugging Face 모델 ID)
# PHASE0_MODEL="./checkpoints/phase0_20260213_230047"  # 로컬 디렉토리 예시
# PHASE0_MODEL="kmseong/Llama-3.2-3B-SSFT"
PHASE0_MODEL="meta-llama/Llama-3.2-3B"
# ========================================
# Dataset 선택 (수정 필요)
# ========================================
# 옵션 1: Safety Basis (circuit_breakers 데이터셋)
# DATASET="circuit_breakers"
# SAMPLES=4994
#
# 옵션 2: Utility Basis (Wikipedia 데이터셋)
DATASET="wikipedia"
SAMPLES=1000
# ========================================

# 로컬 경로일 때만 디렉토리 존재 체크
if [[ "$PHASE0_MODEL" == ./* || "$PHASE0_MODEL" == /* ]]; then
    if [ ! -d "$PHASE0_MODEL" ]; then
        echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
        echo "먼저 scripts/run_phase0_base_training.sh를 실행하세요."
        exit 1
    fi
fi

echo ""
echo "Dataset 설정:"
if [ "$DATASET" = "circuit_breakers" ]; then
    echo "  - Type: Safety Basis (circuit_breakers)"
    echo "  - Samples: $SAMPLES"
    DATASET_ARG="--circuit_breakers_samples_phase1 $SAMPLES"
elif [ "$DATASET" = "wikipedia" ]; then
    echo "  - Type: Utility Basis (Wikipedia)"
    echo "  - Samples: $SAMPLES"
    DATASET_ARG="--wikipedia_samples_phase1 $SAMPLES"
else
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Choose from: circuit_breakers, wikipedia"
    exit 1
fi
echo ""

python train.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset "$DATASET" \
    $DATASET_ARG \
    --batch_size 4 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "========================================="
echo "Phase 1 완료! (Dataset: $DATASET)"
echo "다음 단계: scripts/run_phase2_importance.sh 실행"
echo "========================================="
