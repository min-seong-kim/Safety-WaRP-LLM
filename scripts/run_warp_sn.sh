#!/usr/bin/env bash
# =============================================================================
# WaRP-SN Pipeline: Rotation → Detection → SN-Tune
#
# This script intentionally keeps the argument interface compatible with
# run_warp_sn_pipeline.py. The corresponding Python modules add internal
# diagnostics, gradient checks, and safer masking behavior.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
ROOT_DIR="${ROOT_DIR:-/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/minseong}"
WARP_DIR="${WARP_DIR:-${ROOT_DIR}/Safety-WaRP-LLM}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-hb}"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "[ERROR] conda.sh not found: ${CONDA_SH}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] python not found in env '${CONDA_ENV}': ${PYTHON_BIN}" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch
import transformers
print("[INFO] Python:", sys.executable)
print("[INFO] torch:", torch.__version__)
print("[INFO] transformers:", transformers.__version__)
print("[INFO] cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] cuda_device_count:", torch.cuda.device_count())
PY

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

GPU="${GPU:-6}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
TAG="${TAG:-llama-2-7b-chat-hf}"

# Phase-1 safety basis directory produced by train.py --phase 1
BASIS_DIR="${BASIS_DIR:-${WARP_DIR}/checkpoints/phase1_20260503_221547/basis}"

SAFETY_DATASET="${SAFETY_DATASET:-${WARP_DIR}/data/circuit_breakers_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/minseong_results/warp_sn_${TAG}}"
mkdir -p "$OUTPUT_DIR"

# Must match available basis subdirectories.
LAYER_TYPES="${LAYER_TYPES:-ffn_up,ffn_down,attn_q,attn_k,attn_v}"

# Detection hyperparameters.
NUM_PROMPTS="${NUM_PROMPTS:-4994}"
TOP_K_FFN="${TOP_K_FFN:-1500}"
TOP_K_ATTN="${TOP_K_ATTN:-250}"
FREQ_THRESHOLD="${FREQ_THRESHOLD:-1.0}"  # 1.0 = exact intersection (original)

# SN-Tune hyperparameters.
SN_LR="${SN_LR:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-4994}"

# Optional: reuse an existing WaRP-SN neuron file.
EXISTING_NEURON_FILE="${EXISTING_NEURON_FILE:-}"

# Optional upload.
UPLOAD_NAME="${UPLOAD_NAME:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Logging.
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/warp_sn_pipeline_${TIMESTAMP}.log}"

# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

if [[ ! -d "$WARP_DIR" ]]; then
    echo "[ERROR] WARP_DIR not found: ${WARP_DIR}" >&2
    exit 1
fi
if [[ -z "$BASIS_DIR" || ! -d "$BASIS_DIR" ]]; then
    echo "[ERROR] BASIS_DIR not found: ${BASIS_DIR}" >&2
    exit 1
fi
if [[ ! -f "$SAFETY_DATASET" ]]; then
    echo "[ERROR] SAFETY_DATASET not found: ${SAFETY_DATASET}" >&2
    exit 1
fi
if [[ -n "$EXISTING_NEURON_FILE" && ! -f "$EXISTING_NEURON_FILE" ]]; then
    echo "[ERROR] EXISTING_NEURON_FILE not found: ${EXISTING_NEURON_FILE}" >&2
    exit 1
fi
if [[ ! -f "${WARP_DIR}/run_warp_sn_pipeline.py" ]]; then
    echo "[ERROR] run_warp_sn_pipeline.py not found in ${WARP_DIR}" >&2
    exit 1
fi

echo "=================================================================="
echo "WaRP-SN Pipeline"
echo "=================================================================="
echo "  model       : ${BASE_MODEL_NAME}"
echo "  basis_dir   : ${BASIS_DIR}"
echo "  dataset     : ${SAFETY_DATASET}"
echo "  output_dir  : ${OUTPUT_DIR}"
echo "  log_file    : ${LOG_FILE}"
echo "  layer_types : ${LAYER_TYPES}"
echo "  gpu         : ${GPU}"
echo "  SN_LR       : ${SN_LR}"
echo "  epochs      : ${NUM_EPOCHS}"
echo "  batch/accum : ${BATCH_SIZE}/${GRAD_ACCUM}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

cd "$WARP_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

CMD=(
    "$PYTHON_BIN" "run_warp_sn_pipeline.py"
    --model_name "$BASE_MODEL_NAME"
    --basis_dir "$BASIS_DIR"
    --dataset_file "$SAFETY_DATASET"
    --output_dir "$OUTPUT_DIR"
    --layer_types "$LAYER_TYPES"
    --num_prompts "$NUM_PROMPTS"
    --top_k_ffn "$TOP_K_FFN"
    --top_k_attn "$TOP_K_ATTN"
    --freq_threshold "$FREQ_THRESHOLD"
    --learning_rate "$SN_LR"
    --num_epochs "$NUM_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM"
    --max_seq_len "$MAX_SEQ_LEN"
    --max_samples "$MAX_SAMPLES"
    --gpu "$GPU"
    --dtype "bfloat16"
    --seed 112
)

if [[ -n "$EXISTING_NEURON_FILE" ]]; then
    echo "[INFO] Reusing existing neuron file: ${EXISTING_NEURON_FILE}"
    CMD+=(--existing_neuron_file "$EXISTING_NEURON_FILE")
fi

if [[ -n "$UPLOAD_NAME" ]]; then
    CMD+=(--upload_name "$UPLOAD_NAME")
    if [[ -n "$HF_TOKEN" ]]; then
        CMD+=(--hf_token "$HF_TOKEN")
        export HF_TOKEN
    fi
fi

echo "[INFO] Running command:"
printf '  %q' "${CMD[@]}"
echo
echo

set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

echo
if [[ "$STATUS" -ne 0 ]]; then
    echo "=================================================================="
    echo "[ERROR] WaRP-SN Pipeline failed with status ${STATUS}"
    echo "  Log: ${LOG_FILE}"
    echo "=================================================================="
    exit "$STATUS"
fi

echo "=================================================================="
echo "WaRP-SN Pipeline finished successfully"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log   : ${LOG_FILE}"
echo "=================================================================="
