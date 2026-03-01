#!/bin/bash

# Phase 3: Incremental Learning
# ✅ WaRP 모듈 사용, 마스크 자동 적용

echo "========================================="
echo "Phase 3: Incremental Learning (Fixed)"
echo "========================================="

# 이전 Phase 결과 경로 (로컬 디렉토리 또는 Hugging Face 모델 ID)
# PHASE0_MODEL="./checkpoints/phase0_20260213_230047"  # 로컬 디렉토리 예시
PHASE0_MODEL="meta-llama/Llama-3.2-3B"
BASIS_DIR="./checkpoints/phase1_20260225_162836/basis"
MASKS_DIR="./checkpoints/phase2_20260225_180414/checkpoints/masks"

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

python train.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --masks_dir "$MASKS_DIR" \
    --gsm8k_samples 0 \
    --epochs 3 \
    --utility_lr 1e-5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42 

echo ""
echo "========================================="
echo "Phase 3 완료!"
echo "최종 모델: ./checkpoints/phase3_learning/final_model"
echo "========================================="
