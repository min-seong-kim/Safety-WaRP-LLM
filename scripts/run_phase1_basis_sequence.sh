#!/bin/bash
# Phase 1: Basis Construction — SEQUENCE-WISE variant
#   기본(token-wise)과 달리 각 시퀀스를 하나의 벡터로 pooling 하여 공분산을 구성.
#   Gram = Σ_시퀀스 φ̄ φ̄ᵀ   (φ̄ = --seq_pool 로 mean/last/sum 선택)
#
#   token-wise 와 산출물 포맷(basis/layer_NN_svd.pt)이 동일하므로
#   이후 Phase 2/3 는 --basis_dir 로 이 basis 를 그대로 사용하면 된다.
#
# 사용법:  bash scripts/run_phase1_basis_sequence.sh
#   환경변수로 오버라이드 가능: SEQ_POOL, DATASET, SAMPLES, PHASE0_MODEL, CUDA_VISIBLE_DEVICES
set -e
cd "$(dirname "$0")/.."   # repo root

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-/venv/hb/bin/python}

# ---- 설정 ----
PHASE0_MODEL=${PHASE0_MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}
SEQ_POOL=${SEQ_POOL:-mean}                 # mean | last | sum
DATASET=${DATASET:-circuit_breakers}       # circuit_breakers | wikipedia
SAMPLES=${SAMPLES:-4994}
LAYER_TYPE=${LAYER_TYPE:-attn_q,attn_k,attn_v,ffn_down,ffn_up}
TARGET_LAYERS=${TARGET_LAYERS:-all}
BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints}
LOG_DIR=${LOG_DIR:-./logs}

echo "========================================="
echo "Phase 1: Basis Construction (SEQUENCE-WISE)"
echo "  model     : $PHASE0_MODEL"
echo "  granularity: sequence  (pool=$SEQ_POOL)"
echo "  dataset   : $DATASET (samples=$SAMPLES)"
echo "  layer_type: $LAYER_TYPE"
echo "========================================="

if [ "$DATASET" = "circuit_breakers" ]; then
    DATASET_ARG="--circuit_breakers_samples_phase1 $SAMPLES"
elif [ "$DATASET" = "wikipedia" ]; then
    DATASET_ARG="--wikipedia_samples_phase1 $SAMPLES"
else
    echo "ERROR: Unknown dataset: $DATASET (choose circuit_breakers | wikipedia)"
    exit 1
fi

"${PYTHON_BIN}" train.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset "$DATASET" \
    $DATASET_ARG \
    --basis_granularity sequence \
    --seq_pool "$SEQ_POOL" \
    --batch_size $BATCH_SIZE \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --device cuda \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "========================================="
echo "Phase 1 (sequence-wise, pool=$SEQ_POOL) 완료!"
echo "생성된 basis 로 Phase 2 실행 시 --basis_dir <checkpoints/phase1_*/basis> 지정"
echo "========================================="
