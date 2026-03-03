#!/bin/bash

# Phase 2: Importance Scoring
# ✅ model.eval() 모드, optimizer.step 제거

echo "========================================="
echo "Phase 2: Importance Scoring (Fixed)"
echo "========================================="

# Phase 0, 1 결과 경로
# PHASE0_MODEL="./checkpoints/phase0_20260213_230047"  # 로컬 디렉토리 예시
PHASE0_MODEL="kmseong/safety-warp-llama-3.2-3b-phase0_20260213_230047"
BASIS_DIR="./checkpoints/phase1_20260301_165137/basis"

# 로컬 경로일 때만 디렉토리 존재 체크
if [[ "$PHASE0_MODEL" == ./* || "$PHASE0_MODEL" == /* ]]; then
    if [ ! -d "$PHASE0_MODEL" ]; then
        echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
        exit 1
    fi
fi

if [ ! -d "$BASIS_DIR" ]; then
    echo "ERROR: Basis를 찾을 수 없습니다: $BASIS_DIR"
    echo "먼저 scripts/run_phase1_basis.sh를 실행하세요."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 각 WaRP layer 별로 keep ratio 적용할 거면 --perlayer 옵션 추가
python train.py \
    --phase 2 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    --circuit_breakers_samples 4994 \
    --keep_ratio 0.03 \
    --batch_size 2 \
    --max_length 512 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "========================================="
echo "Phase 2 완료!"
echo "다음 단계: scripts/run_phase3_learning.sh 실행"
echo "========================================="
