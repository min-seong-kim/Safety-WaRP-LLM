#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 MODEL_PATH_OR_HF_ID [additional evaluator arguments]" >&2
  echo "Example: GPU=0 $0 wvnvwn/llama2-7b-chat-lr5e-5-sst2-lr2e-4-cbwsr-rot" >&2
  exit 2
fi

MODEL="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVALUATOR="$SCRIPT_DIR/evaluate_agnews_sst2.py"
DEFAULT_HB_PYTHON="/NHNHOME/26msit001_A/BASE/edge_ai_lab/jongbokwon/.miniconda3/envs/hb/bin/python"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$DEFAULT_HB_PYTHON" ]]; then
  PY="$DEFAULT_HB_PYTHON"
else
  PY="$(command -v python3)"
fi

GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/evaluation_results/agnews_sst2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "$PY" -u "$EVALUATOR" "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "$@"
