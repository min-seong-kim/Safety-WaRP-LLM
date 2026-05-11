#!/usr/bin/env bash
# =============================================================================
# WaRP-SN Downstream Fine-tuning with Safety Neuron Column Freezing
#
# Pipeline:
#   1. Load WaRP-SN-Tuned model (standard HF checkpoint)
#   2. Load Phase-1 safety basis U matrices
#   3. Reparameterize: basis_coeff = W @ U  (WaRP space)
#   4. Freeze safety neuron columns of basis_coeff via gradient hooks
#   5. Fine-tune on downstream task (GSM8K / MBPP)
#   6. Save: W = basis_coeff @ U^T → standard HF checkpoint
#
# NOTE: Uses 'hb' conda env (standard transformers, no patched modeling_llama).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
ROOT_DIR="/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong"
WARP_DIR="${ROOT_DIR}/Safety-WaRP-LLM"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-hb}"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "[ERROR] conda.sh not found: ${CONDA_SH}" >&2; exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] python not found in env '${CONDA_ENV}'" >&2; exit 1
fi
echo "[INFO] Python: ${PYTHON_BIN}"
"$PYTHON_BIN" --version

# ---------------------------------------------------------------------------
# 1. Configuration — edit these variables
# ---------------------------------------------------------------------------

GPU=6

# WaRP-SN-Tuned model (output of run_warp_sn.sh)
# Option A — edit directly:
MODEL_PATH="${ROOT_DIR}/minseong_results/warp_sn_llama-2-7b-chat-hf/warp_sn_tuned_lr5e-5_20260504_013418"
# Option B — pass via environment variable:
#   export MODEL_PATH=/path/to/warp_sn_tuned_model
# MODEL_PATH="${MODEL_PATH:-}"

# Phase-1 safety basis directory (same one used in run_warp_sn.sh)
# Option A — edit directly:
BASIS_DIR="${WARP_DIR}/checkpoints/phase1_20260503_221547/basis"
# Option B — pass via environment variable:
#   export BASIS_DIR=/path/to/phase1/basis
# BASIS_DIR="${BASIS_DIR:-}"

# WaRP-space safety neuron file (produced by run_warp_sn.sh detection step)
# Option A — edit directly:
NEURON_FILE="${ROOT_DIR}/minseong_results/warp_sn_llama-2-7b-chat-hf/warp_safety_neurons_20260504_013418.txt"
# Option B — pass via environment variable:
#   export NEURON_FILE=/path/to/warp_safety_neurons_TIMESTAMP.txt
# NEURON_FILE="${NEURON_FILE:-}"

# Downstream task
TASK="gsm8k"           # gsm8k | mbpp
NUM_TRAIN_SAMPLES=7473  # gsm8k full train = 7473; mbpp sanitized train = 374

# Layer types (must match the basis and neuron file)
LAYER_TYPES="ffn_up,ffn_down,attn_q,attn_k,attn_v"

# Training hyper-parameters
LR="5e-5"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=1024

OUTPUT_DIR="${ROOT_DIR}/minseong_results/warp_sn_llama-2-7b-chat-hf/downstream_freeze_${TASK}"
mkdir -p "$OUTPUT_DIR"

# Optional: upload to HuggingFace after training
UPLOAD_NAME="${UPLOAD_NAME:-}"
# UPLOAD_NAME="kmseong/llama2_7b_chat_WaRP-SN_GSM8K_freeze_lr${LR}"
HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# 2. Validate inputs
# ---------------------------------------------------------------------------

check_required() {
    local var_name="$1"
    local var_val="$2"
    if [[ -z "$var_val" ]]; then
        echo "[ERROR] ${var_name} is not set." >&2
        echo "        Set it in the script or via: export ${var_name}=..." >&2
        exit 1
    fi
}

check_required "MODEL_PATH" "$MODEL_PATH"
check_required "BASIS_DIR"  "$BASIS_DIR"
check_required "NEURON_FILE" "$NEURON_FILE"

if [[ ! -d "$MODEL_PATH" && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}" >&2; exit 1
fi
if [[ ! -d "$BASIS_DIR" ]]; then
    echo "[ERROR] BASIS_DIR not found: ${BASIS_DIR}" >&2; exit 1
fi
if [[ ! -f "$NEURON_FILE" ]]; then
    echo "[ERROR] NEURON_FILE not found: ${NEURON_FILE}" >&2; exit 1
fi

echo "=================================================================="
echo "WaRP-SN Downstream FT — Safety Column Freezing"
echo "=================================================================="
echo "  model        : ${MODEL_PATH}"
echo "  basis_dir    : ${BASIS_DIR}"
echo "  neuron_file  : ${NEURON_FILE}"
echo "  task         : ${TASK}"
echo "  samples      : ${NUM_TRAIN_SAMPLES}"
echo "  lr           : ${LR}"
echo "  epochs       : ${NUM_EPOCHS}"
echo "  batch×accum  : ${BATCH_SIZE}×${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "  gpu          : ${GPU}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

cd "$WARP_DIR"

CMD=(
    "$PYTHON_BIN" "finetune_downstream_freeze_warp_sn.py"
    --model_name_or_path  "$MODEL_PATH"
    --basis_dir           "$BASIS_DIR"
    --neuron_file         "$NEURON_FILE"
    --layer_types         "$LAYER_TYPES"
    --task                "$TASK"
    --num_train_samples   "$NUM_TRAIN_SAMPLES"
    --output_dir          "$OUTPUT_DIR"
    --learning_rate       "$LR"
    --num_epochs          "$NUM_EPOCHS"
    --batch_size          "$BATCH_SIZE"
    --grad_accum          "$GRAD_ACCUM"
    --max_length          "$MAX_LENGTH"
    --gpu                 "$GPU"
    --dtype               "bfloat16"
    --seed                42
)

if [[ -n "$UPLOAD_NAME" ]]; then
    CMD+=(--upload_name "$UPLOAD_NAME")
    if [[ -n "$HF_TOKEN" ]]; then
        CMD+=(--hf_token "$HF_TOKEN")
        export HF_TOKEN
    fi
fi

export CUDA_VISIBLE_DEVICES="$GPU"

echo "[INFO] Running: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=================================================================="
echo "WaRP-SN Downstream FT finished"
echo "  Output: ${OUTPUT_DIR}"
echo "=================================================================="
