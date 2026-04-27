#!/bin/bash

# Dual Importance Analysis
#
# 목적:
#   WaRP basis-rotated space(basis_coeff = W @ U)에서
#   preserve(안전) 데이터셋과 adapt(downstream) 데이터셋으로
#   각각 importance score를 계산하고, 두 mask의 겹침을 분석한다.
#
# 핵심 질문:
#   "Phase 3에서 freeze된 파라미터가 downstream task에서도 중요한가?"
#   - A_blocked% 높음 → freeze가 downstream 성능을 방해함
#   - A_blocked% 낮음 → freeze해도 downstream task은 다른 방향을 자유롭게 사용 가능
#
# 지표:
#   Jaccard             = |P∩A| / |P∪A|        (전체 겹침 비율)
#   A_blocked_ratio     = |P∩A| / |A|           (adapt 중요 파라미터 중 preserve에 막히는 비율) ← 핵심
#   Preserve→adapt      = |P∩A| / |P|           (freeze 중 adapt에도 중요한 비율)
#   Pearson_r           (두 importance score의 상관계수)
#
# 출력:
#   analysis/dual_importance_<preserve>_vs_<adapt>_kr<ratio>_<timestamp>/
#     metadata.json
#     importances_preserve.pt
#     importances_adapt.pt
#     overlap_stats.json / .csv
#     masks_preserve / masks_adapt
#     figures/
#       overlap_by_layer.png
#       scatter_global.png
#       heatmap_jaccard.png
#       heatmap_adapt_blocked.png

set -e
set -o pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================================================"
echo "Dual Importance Analysis (WaRP Basis-Rotated Space)"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

PHASE0_MODEL="meta-llama/Llama-3.2-3B"
BASIS_DIR="./checkpoints/phase1_20260410_113227/basis"   # Phase 1 결과 경로

# ── Preserve 데이터셋 (안전 방향) ──────────────────────────────────────
PRESERVE_DATASET="circuit_breakers"
PRESERVE_SAMPLES=4994          # 0 = 전체
CIRCUIT_BREAKERS_PATH="./data/circuit_breakers_train.json"

# ── Adapt 데이터셋 (downstream) ────────────────────────────────────────
ADAPT_DATASET="gsm8k"           # gsm8k | math | metamath | circuit_breakers | wikipedia
ADAPT_SAMPLES=0             # 0 = 전체
MATH_SUBJECTS="all"
MATH_LEVELS="all"

# ── keep_ratio 설정 ────────────────────────────────────────────────────
# 단일 값 → KEEP_RATIO 사용
# 여러 값 비교 → KEEP_RATIO_LIST 사용 (KEEP_RATIO 무시)
KEEP_RATIO=0.1
# KEEP_RATIO_LIST="0.05,0.1,0.2,0.3"   # 여러 ratio 비교 시 주석 해제

# ── 레이어 설정 ────────────────────────────────────────────────────────
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_LAYERS="all"

# ── 계산 설정 ──────────────────────────────────────────────────────────
BATCH_SIZE=4
MAX_LENGTH=1024
DTYPE="bfloat16"
DEVICE="cuda"

# ── 저장 ───────────────────────────────────────────────────────────────
OUTPUT_DIR="./analysis"
LOG_DIR="./logs"

# ── 사전 계산된 importance 재사용 (아래 두 변수 설정 시 model load & gradient 계산 생략) ──
# LOAD_PRESERVE_IMP=""   # 예: "./analysis/dual.../importances_preserve.pt"
# LOAD_ADAPT_IMP=""      # 예: "./analysis/dual.../importances_adapt.pt"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  PHASE0_MODEL      : $PHASE0_MODEL"
echo "  BASIS_DIR         : $BASIS_DIR"
echo "  PRESERVE_DATASET  : $PRESERVE_DATASET  (samples=$PRESERVE_SAMPLES)"
echo "  ADAPT_DATASET     : $ADAPT_DATASET  (samples=$ADAPT_SAMPLES)"
echo "  KEEP_RATIO        : ${KEEP_RATIO_LIST:-$KEEP_RATIO}"
echo "  LAYER_TYPE        : $LAYER_TYPE"
echo "  TARGET_LAYERS     : $TARGET_LAYERS"
echo ""

# ========================================================================
# Build argument string
# ========================================================================

ARGS=(
    --phase0_model_dir "$PHASE0_MODEL"
    --basis_dir "$BASIS_DIR"
    --preserve_dataset "$PRESERVE_DATASET"
    --preserve_samples "$PRESERVE_SAMPLES"
    --circuit_breakers_path "$CIRCUIT_BREAKERS_PATH"
    --adapt_dataset "$ADAPT_DATASET"
    --adapt_samples "$ADAPT_SAMPLES"
    --keep_ratio "$KEEP_RATIO"
    --layer_type "$LAYER_TYPE"
    --target_layers "$TARGET_LAYERS"
    --batch_size "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --dtype "$DTYPE"
    --device "$DEVICE"
    --output_dir "$OUTPUT_DIR"
    --log_dir "$LOG_DIR"
)

# keep_ratio_list (여러 비율 비교)
if [ -n "${KEEP_RATIO_LIST:-}" ]; then
    ARGS+=(--keep_ratio_list "$KEEP_RATIO_LIST")
fi

# MATH 관련 옵션
if [ "$ADAPT_DATASET" = "math" ] || [ "$PRESERVE_DATASET" = "math" ]; then
    ARGS+=(--math_subjects "$MATH_SUBJECTS" --math_levels "$MATH_LEVELS")
fi

# 사전 계산된 importance 재사용
if [ -n "${LOAD_PRESERVE_IMP:-}" ] && [ -n "${LOAD_ADAPT_IMP:-}" ]; then
    echo "  [SKIP] Pre-computed importances detected — model gradient pass will be skipped"
    ARGS+=(--load_preserve_importances "$LOAD_PRESERVE_IMP")
    ARGS+=(--load_adapt_importances    "$LOAD_ADAPT_IMP")
fi

# ========================================================================
# Run
# ========================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "Running analysis..."
echo "========================================================================"

python analyze_dual_importance.py "${ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/dual_importance_${ADAPT_DATASET}_${TIMESTAMP}.log"

echo ""
echo "========================================================================"
echo "Analysis completed."
echo "  Results: $OUTPUT_DIR"
echo "  Log    : $LOG_DIR/dual_importance_${ADAPT_DATASET}_${TIMESTAMP}.log"
echo "========================================================================"
