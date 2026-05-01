#!/bin/bash
# ============================================================================
# WaRP-Rotated Safe LoRA 실험 스크립트
#
# 파이프라인:
#   1) (선택) WaRP Phase 1: safety data로 SVD basis 구성
#   2) (선택) WaRP Phase 2: importance mask 계산 (keep_ratio=0.1)
#      safety-critical 레이어를 LoRA target_modules에서 제외
#   3) LoRA + GSM8K fine-tuning — WaRP rotated space에서 훈련
#      (lora_A input dim = k, x → x @ U_k 로 safety subspace로 제한)
#   4) WaRP-Rotated Safe LoRA projection 적용
#      → C_rot = (V U_k)(V U_k)^T / ‖V U_k‖_F
#         U_k = WaRP basis top-k columns (safety-relevant input subspace)
#         V   = W_aligned − W_base       (alignment delta)
#
# 비교 실험 (4-way):
#   A) Baseline LoRA           (projection 없음)
#   B) Safe LoRA               (원본, C = V V^T / ‖V‖)
#   C) WaRP-Rotated Safe LoRA  (본 스크립트)
#
# 결과 평가:
#   - HarmBench: safety 보존 여부
#   - GSM8K: utility 유지 여부
# ============================================================================

source /home/yonsei_jong/miniconda3/etc/profile.d/conda.sh
conda activate hb
set -e
set -o pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SAFELORA_DIR="/home/yonsei_jong/SafeLoRA"
cd "$REPO_DIR"

echo "========================================================================"
echo "WaRP-Rotated Safe LoRA Experiment"
echo "========================================================================"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# 모델 설정
BASE_MODEL="meta-llama/Llama-2-7b-hf"
ALIGNED_MODEL="kmseong/llama2_7b-Safety-FT-lr3e-5"

# WaRP Phase 1 basis 경로 설정
# 이미 생성된 basis가 있으면 해당 경로를 지정하고 SKIP_PHASE1=true
SKIP_PHASE1=false
EXISTING_BASIS_DIR="/home/yonsei_jong/Safety-WaRP-LLM/checkpoints/phase1_20260426_161130/basis"

# Step 2 건너뛰기 설정 (Step 3만 실행 시)
SKIP_STEP2=false
EXISTING_LORA_DIR="/home/yonsei_jong/Safety-WaRP-LLM/checkpoints/lora_gsm8k_20260426_163346"

# Phase 2 importance mask 설정
SKIP_PHASE2=false
EXISTING_MASKS_DIR=""
PHASE2_DATASET="circuit_breakers"    # safety dataset으로 importance scoring
PHASE2_KEEP_RATIO=0.1                # 상위 10% safety-critical 파라미터 freeze
PHASE2_SAMPLES=4994
PHASE2_BATCH_SIZE=4

# LoRA-A element-wise importance freeze 설정
# safety data로 lora_A 파라미터의 |grad| 누적 → top-k% element freeze (WaRP Phase 2 원리)
LORA_IMPORTANCE_KEEP_RATIO=0.1       # 0.0=미사용, 0.1=top 10%% freeze
LORA_IMPORTANCE_SAMPLES=4994          # importance scoring 샘플 수
SAFETY_DATA_PATH="$(realpath -m "$REPO_DIR/data/circuit_breakers_train.json")"

# Phase 1 설정 (SKIP_PHASE1=false일 때)
PHASE1_DATASET="circuit_breakers"
PHASE1_SAMPLES=4994
PHASE1_LAYER_TYPE="attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down"
PHASE1_BATCH_SIZE=4

# LoRA fine-tuning 설정 (Step 2)
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_EPOCHS=3
LORA_LR="2e-4"
LORA_BATCH_SIZE=4
LORA_GRAD_ACCUM=4
NUM_TRAIN_SAMPLES=7473    # GSM8K 전체

# WaRP-Rotated Safe LoRA 설정 (Step 3)
TOP_K_RATIO=0.5           # top-k = in_dim * TOP_K_RATIO
SELECT_TYPE="number"   # "threshold" or "number"
THRESHOLD=0.5             # threshold 모드: cosine < THRESHOLD인 레이어 투영
NUM_PROJ_LAYERS=20        # number 모드: cosine 낮은 순 N개 투영
USE_APPROX=true           # approximate projector 사용 여부

# 출력 경로
BASE_OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  BASE_MODEL:               $BASE_MODEL"
echo "  ALIGNED_MODEL:            $ALIGNED_MODEL"
echo "  SKIP_PHASE1:              $SKIP_PHASE1"
echo "  TOP_K_RATIO:              $TOP_K_RATIO"
echo "  LORA_IMPORTANCE_KEEP_RATIO: $LORA_IMPORTANCE_KEEP_RATIO"
echo "  SELECT_TYPE:              $SELECT_TYPE"
echo "  THRESHOLD:                $THRESHOLD"
echo "  LORA LR:                  $LORA_LR  EPOCHS: $LORA_EPOCHS"
echo ""

# ============================================================================
# Step 1: WaRP Phase 1 Basis Construction
# ============================================================================
if [ "$SKIP_PHASE1" = "true" ] && [ -n "$EXISTING_BASIS_DIR" ] && [ -d "$EXISTING_BASIS_DIR" ]; then
    BASIS_DIR="$EXISTING_BASIS_DIR"
    echo "Skipping Phase 1 (using existing basis: $BASIS_DIR)"
else
    echo "========================================================================"
    echo "STEP 1: WaRP Phase 1 — Safety Basis Construction"
    echo "  (SVD of safety data activations → U_k = safety-relevant input subspace)"
    echo "========================================================================"
    echo ""

    python train.py \
        --phase 1 \
        --phase0_model_dir "$ALIGNED_MODEL" \
        --safety_dataset "$PHASE1_DATASET" \
        --circuit_breakers_samples_phase1 $PHASE1_SAMPLES \
        --batch_size $PHASE1_BATCH_SIZE \
        --layer_type "$PHASE1_LAYER_TYPE" \
        --target_layers all \
        --output_dir "$BASE_OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --device cuda \
        --dtype bfloat16 \
        --seed 42 \
        2>&1 | tee "$LOG_DIR/warp_safelora_phase1_${TIMESTAMP}.log"

    PHASE1_OUTPUT_DIR=$(find "$BASE_OUTPUT_DIR" -maxdepth 1 -name "phase1_*" \
        -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    BASIS_DIR="$(realpath "$PHASE1_OUTPUT_DIR/basis")"

    if [ ! -d "$BASIS_DIR" ]; then
        echo "ERROR: Phase 1 basis not found: $BASIS_DIR"
        exit 1
    fi

    echo ""
    echo "Phase 1 completed: $BASIS_DIR"
    echo ""
fi

# ============================================================================
# Step 2: LoRA + GSM8K Fine-tuning
# ============================================================================
if [ "$SKIP_STEP2" = "true" ] && [ -n "$EXISTING_LORA_DIR" ] && [ -d "$EXISTING_LORA_DIR" ]; then
    LORA_OUTPUT_DIR="$EXISTING_LORA_DIR"
    echo "Skipping Step 2 (using existing LoRA: $LORA_OUTPUT_DIR)"
else
echo "========================================================================"
echo "STEP 2: LoRA + GSM8K Fine-tuning (WaRP Rotated Space)"
echo "  lora_A 훈련 input dim = k = d_in * ${TOP_K_RATIO}  (safety subspace로 제한)"
echo "  x → x @ U_k  (WaRP basis top-k 방향으로 투영 후 학습)"
echo "  lora_A importance freeze: top ${LORA_IMPORTANCE_KEEP_RATIO} (안전 데이터 |grad| 기준)"
echo "========================================================================"
echo ""

cd "$SAFELORA_DIR"

LORA_OUTPUT_DIR="$(realpath -m "$REPO_DIR/checkpoints/lora_gsm8k_${TIMESTAMP}")"
mkdir -p "$LORA_OUTPUT_DIR"

LR_SAFE=$(echo "$LORA_LR" | sed 's/[^a-zA-Z0-9_-]/_/g')

python safe_lora_gsm8k_training.py \
    --base-model "$BASE_MODEL" \
    --aligned-model "$ALIGNED_MODEL" \
    --num-train-samples $NUM_TRAIN_SAMPLES \
    --epochs $LORA_EPOCHS \
    --lr $LORA_LR \
    --batch-size $LORA_BATCH_SIZE \
    --grad-accum $LORA_GRAD_ACCUM \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --output-lora-path "$LORA_OUTPUT_DIR" \
    --basis-dir "$BASIS_DIR" \
    --top-k-ratio $TOP_K_RATIO \
    --warp-rotated \
    --lora-importance-keep-ratio $LORA_IMPORTANCE_KEEP_RATIO \
    --lora-importance-samples $LORA_IMPORTANCE_SAMPLES \
    --safety-data-path "$SAFETY_DATA_PATH" \
    --skip-step3 \
    2>&1 | tee "$REPO_DIR/$LOG_DIR/warp_safelora_lora_lr${LR_SAFE}_${TIMESTAMP}.log"

if [ ! -d "$LORA_OUTPUT_DIR" ]; then
    echo "ERROR: LoRA output not found: $LORA_OUTPUT_DIR"
    exit 1
fi

echo ""
echo "Step 2 completed: $LORA_OUTPUT_DIR"
echo ""

cd "$REPO_DIR"
fi  # end SKIP_STEP2

# ============================================================================
# Step 3: WaRP-Rotated Safe LoRA Projection
# ============================================================================
echo "========================================================================"
echo "STEP 3: WaRP-Rotated Safe LoRA Projection"
echo "  V_rot = V @ U_k  (alignment delta × safety basis top-k)"
echo "  C_rot = V_rot V_rot^T / ‖V_rot‖_F"
echo "  B_proj = C_rot @ B  (project lora_B onto rotated alignment subspace)"
echo "========================================================================"
echo ""

USE_APPROX_FLAG=""
if [ "$USE_APPROX" = "true" ]; then
    USE_APPROX_FLAG="--use-approx"
fi

WARP_SAFELORA_OUTPUT="$BASE_OUTPUT_DIR/warp_safelora_${TIMESTAMP}"
mkdir -p "$WARP_SAFELORA_OUTPUT"

python scripts/apply_warp_safe_lora.py \
    --base-model "$BASE_MODEL" \
    --aligned-model "$ALIGNED_MODEL" \
    --lora-adapter-path "$LORA_OUTPUT_DIR" \
    --basis-dir "$BASIS_DIR" \
    --output-path "$WARP_SAFELORA_OUTPUT" \
    --top-k-ratio $TOP_K_RATIO \
    --select-type $SELECT_TYPE \
    --threshold $THRESHOLD \
    --num-proj-layers $NUM_PROJ_LAYERS \
    $USE_APPROX_FLAG \
    --log-path "$LOG_DIR/warp_safelora_projection_${TIMESTAMP}.log" \
    2>&1 | tee "$LOG_DIR/warp_safelora_step3_${TIMESTAMP}.log"

if [ ! -d "$WARP_SAFELORA_OUTPUT" ]; then
    echo "ERROR: WaRP Safe LoRA output not found: $WARP_SAFELORA_OUTPUT"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "WaRP-Rotated Safe LoRA Experiment Completed!"
echo "========================================================================"
echo ""
echo "  WaRP Basis:         $BASIS_DIR"
if [ -n "$MASKS_DIR" ]; then
echo "  Phase 2 Masks:      $MASKS_DIR"
fi
echo "  LoRA adapter:       $LORA_OUTPUT_DIR"
echo "  Final model:        $WARP_SAFELORA_OUTPUT/merged_model"
echo ""
echo "  Logs:"
echo "    Phase 1:      $LOG_DIR/warp_safelora_phase1_${TIMESTAMP}.log"
echo "    LoRA FT:      $LOG_DIR/warp_safelora_lora_lr${LR_SAFE}_${TIMESTAMP}.log"
echo "    Projection:   $LOG_DIR/warp_safelora_projection_${TIMESTAMP}.log"
echo ""
echo "Evaluation:"
echo "  # Safety (HarmBench)"
echo "  cd /home/yonsei_jong/HarmBench"
echo "  python generate_completions.py --model $WARP_SAFELORA_OUTPUT/merged_model ..."
echo ""
echo "  # Utility (GSM8K)"
echo "  lm_eval --model hf \\"
echo "    --model_args pretrained=$WARP_SAFELORA_OUTPUT/merged_model \\"
echo "    --tasks gsm8k --num_fewshot 5"
echo "========================================================================"
