#!/bin/bash

# Safety-WaRP-LLM: SN-Tune + WaRP Rotation 실험
#
# Phase 0 model: kmseong/llama2_7b_only_sn_tuned_lr3e-5
#   (Safety Neuron Tuned 모델 - original weight space에서 safety neuron을 식별/학습)
#
# 실험 목적:
#   SN-Tuned 모델에 WaRP rotation을 적용하고,
#   Phase 2 gradient importance 대신 미리 탐지된 safety neuron을
#   WaRP basis_coeff 공간에서 직접 freeze하여 학습.
#
# 파이프라인:
#   Phase 1 : safety dataset → SVD basis (rotation) 구성
#   Phase 2* : generate_sn_masks_for_warp.py
#              (gradient importance X → safety neuron file로 mask 생성)
#              - ffn_up / ffn_gate  : mask[j, :] = True  (row j = output neuron j)
#              - ffn_down           : mask[:, j] = True  (col j = input  neuron j, approx)
#              - attn_q/k/v         : mask[j, :] = True  (row j = output neuron j)
#   Phase 3 : WaRP freeze 모드로 GSM8K fine-tuning
#              (safety-critical basis_coeff frozen, 나머지 학습)
#
# 비교 기준: finetune_gsm8k_freeze_sn.py (SN neuron direct freeze, no rotation)

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
echo "Safety-WaRP-LLM: SN-Tune + WaRP Rotation Experiment"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

# SN-Tuned base model
PHASE0_MODEL="kmseong/llama2_7b_chat_only_sn_tuned_lr3e-5"

# Safety neuron file (pre-detected, used instead of Phase 2 gradient importance)
SAFETY_NEURONS_FILE="/home/yonsei_jong/Safety-Neuron/neuron_detection/output_neurons/llama_2_7b_chat_safety_neuron_accelerated_20260416_160653.txt"

# Phase 1: safety basis construction
PHASE1_DATASET="circuit_breakers"
PHASE1_SAMPLES=4994

# Phase 3: downstream fine-tuning
PHASE3_DATASET="gsm8k"
PHASE3_SAMPLES=0       # 0 = all ~7473 samples

# Training hyper-parameters
BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=3
LR="7e-5"
DTYPE="bfloat16"
DEVICE="cuda"
TARGET_LAYERS="all"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_up,ffn_down"

BASE_OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LR_SAFE=$(echo "$LR" | sed 's/[^a-zA-Z0-9_-]/_/g')

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Phase 0 Model (SN-Tuned): $PHASE0_MODEL"
echo "  Safety Neurons File:      $SAFETY_NEURONS_FILE"
echo "  Phase 1 Dataset:          $PHASE1_DATASET ($PHASE1_SAMPLES samples)"
echo "  Phase 3 Dataset:          $PHASE3_DATASET (samples=$PHASE3_SAMPLES)"
echo "  LR: $LR  |  Batch: $BATCH_SIZE  |  GradAccum: $GRAD_ACCUM  |  Epochs: $EPOCHS"
echo "  Layer Types: $LAYER_TYPE"
echo ""
echo "Pipeline:"
echo "  1. Phase 1   : SVD basis construction from safety data"
echo "  2. Phase 2*  : generate_sn_masks_for_warp.py"
echo "                 (safety neuron file → WaRP basis_coeff masks)"
echo "  3. Phase 3   : WaRP freeze-mode training on GSM8K"
echo ""

# ========================================================================
# Phase 1: Basis Construction
# safety dataset으로 SVD basis(rotation) 구성
# ========================================================================
echo "========================================================================"
echo "PHASE 1: Basis Construction (SVD rotation from safety data)"
echo "========================================================================"
echo ""

python train.py \
    --phase 1 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --safety_dataset "$PHASE1_DATASET" \
    --circuit_breakers_samples_phase1 $PHASE1_SAMPLES \
    --batch_size $BATCH_SIZE \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir $BASE_OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/sn_warp_phase1_${TIMESTAMP}.log"

PHASE1_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase1_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
PHASE1_BASIS_DIR="$PHASE1_OUTPUT_DIR/basis"

if [ ! -d "$PHASE1_BASIS_DIR" ]; then
    echo "ERROR: Phase 1 basis directory not found: $PHASE1_BASIS_DIR"
    exit 1
fi

echo ""
echo "Phase 1 completed: $PHASE1_BASIS_DIR"
echo ""

# ========================================================================
# Phase 2* (replacement): generate masks from safety neuron file
#
# gradient importance scoring 대신, 미리 탐지된 safety neuron 인덱스를
# WaRP basis_coeff 공간의 mask로 변환.
#
# ffn_up   / attn_q/k/v : mask[j, :] = True  (row j = output neuron j)
# ffn_down              : mask[:, j] = True  (col j = input  neuron j, approx)
# ========================================================================
echo "========================================================================"
echo "PHASE 2*: Generate Masks from Safety Neuron File"
echo "  (replaces Phase 2 gradient-based importance scoring)"
echo "========================================================================"
echo ""

MASKS_DIR=$(python scripts/generate_sn_masks_for_warp.py \
    --safety_neurons_file "$SAFETY_NEURONS_FILE" \
    --model_dir "$PHASE0_MODEL" \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir $BASE_OUTPUT_DIR \
    2>&1 | tee "$LOG_DIR/sn_warp_phase2_sn_${TIMESTAMP}.log" | tail -1)

if [ ! -d "$MASKS_DIR" ]; then
    echo "ERROR: Mask directory not found: $MASKS_DIR"
    echo "Check log: $LOG_DIR/sn_warp_phase2_sn_${TIMESTAMP}.log"
    exit 1
fi

echo ""
echo "Phase 2* completed: $MASKS_DIR"
echo ""

# ========================================================================
# Phase 3: Downstream Training (WaRP Freeze Mode)
#
# WaRP rotation (Phase 1 basis) 적용 후,
# safety neuron에 해당하는 basis_coeff를 FROZEN,
# 나머지 basis_coeff는 GSM8K로 학습.
# (--non_freeze 미사용 → freeze mode)
# ========================================================================
echo "========================================================================"
echo "PHASE 3: Downstream Training (WaRP Freeze + Safety Neuron Mask)"
echo "  basis_coeff[safety neurons] FROZEN"
echo "  basis_coeff[other]          TRAINABLE on GSM8K"
echo "========================================================================"
echo ""

python train.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$PHASE1_BASIS_DIR" \
    --masks_dir "$MASKS_DIR" \
    --phase3_dataset $PHASE3_DATASET \
    --gsm8k_samples $PHASE3_SAMPLES \
    --epochs $EPOCHS \
    --utility_lr $LR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir $BASE_OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed 42 \
    --gradient_checkpointing \
    --non_freeze \
    2>&1 | tee "$LOG_DIR/sn_warp_phase3_lr${LR_SAFE}_${TIMESTAMP}.log"

PHASE3_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase3_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
FINAL_MODEL="$PHASE3_OUTPUT_DIR/final_model"

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "========================================================================"
echo "Experiment Completed!"
echo "========================================================================"
echo ""
echo "  Phase 0 (SN-Tuned):  $PHASE0_MODEL"
echo "  Phase 1 Basis:       $PHASE1_BASIS_DIR"
echo "  Phase 2* Masks:      $MASKS_DIR"
echo "  Final Model:         $FINAL_MODEL"
echo ""
echo "  Logs:"
echo "    Phase 1   : $LOG_DIR/sn_warp_phase1_${TIMESTAMP}.log"
echo "    Phase 2*  : $LOG_DIR/sn_warp_phase2_sn_${TIMESTAMP}.log"
echo "    Phase 3   : $LOG_DIR/sn_warp_phase3_lr${LR_SAFE}_${TIMESTAMP}.log"
echo ""
echo "Evaluation:"
echo "  # HarmBench safety"
echo "  cd /home/yonsei_jong/HarmBench"
echo "  python generate_completions.py --model $FINAL_MODEL ..."
echo ""
echo "  # GSM8K accuracy"
echo "  cd $REPO_DIR"
echo "  lm_eval --model hf --model_args pretrained=$FINAL_MODEL \\"
echo "    --tasks gsm8k --num_fewshot 5"
echo "========================================================================"
