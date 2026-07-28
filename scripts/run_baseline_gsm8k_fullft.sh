#!/bin/bash

# ========================================================================
# Baseline: 일반 full-parameter downstream FT (GSM8K) — 시간/VRAM 측정용
#
# WaRP(Phase 3) 대조군. run_all_phases_integrated.sh 의 Phase 3 와
# 모델 / 데이터 / 하이퍼파라미터 / 모델 로드 조건을 모두 동일하게 맞춘다.
#
#   동일:  epochs=3, lr=5e-5, bs=2, grad_accum=8(effective 16), max_length=1024,
#          weight_decay=0.01, warmup_ratio=0.1, cosine, max_grad_norm=1.0,
#          adamw_torch, bf16, seed=42, gsm8k 7473 samples, chat template,
#          attn_implementation=eager, use_cache=False, gradient_checkpointing=off
#   다름:  WaRP 재파라미터화 / 마스킹 없음 (= 순수 full FT),
#          constrained SFT 없음 (Phase 3 의 reference-logp 사전계산 단계가 없음)
# ========================================================================

CONDA_ENV_NAME="${CONDA_ENV_NAME:-hb}"
_CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$_CONDA_BASE" ] && [ -f "$_CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
fi
set -e
set -o pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"

# Phase 3 와 동일한 하이퍼파라미터
EPOCHS=3
LEARNING_RATE=5e-5
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
MAX_LENGTH=1024
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
LR_SCHEDULER="cosine"
MAX_GRAD_NORM=1.0
NUM_TRAIN_SAMPLES=7473
SEED=42
LOGGING_STEPS=10

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs"
PROFILE_DIR="$LOG_DIR/profile_baseline_${TIMESTAMP}"
OUTPUT_DIR="./checkpoints/baseline_fullft_gsm8k_lr${LEARNING_RATE}_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$PROFILE_DIR"

# W&B (Phase 3 와 같은 프로젝트, 대조군임을 알 수 있게 group/tag 지정)
USE_WANDB=1
WANDB_PROJECT="Safety-WaRP-LLM"
export WANDB_RUN_GROUP="baseline_fullft_${TIMESTAMP}"
export WANDB_TAGS="baseline,full_ft,gsm8k"

if [ "$USE_WANDB" = "1" ]; then
    if ! python -c "import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
        echo "❌ W&B 인증이 되어 있지 않습니다. 'wandb login' 후 다시 실행하세요."
        echo "   (W&B 없이 돌리려면 USE_WANDB=0)"
        exit 1
    fi
    REPORT_TO="wandb"
else
    REPORT_TO="none"
fi

echo "========================================================================"
echo "Baseline Full-Parameter FT (GSM8K)  —  WaRP Phase 3 대조군"
echo "========================================================================"
echo "  Model:            $MODEL"
echo "  Epochs / LR:      $EPOCHS / $LEARNING_RATE"
echo "  Batch / Accum:    $BATCH_SIZE / $GRAD_ACCUM_STEPS (effective $((BATCH_SIZE*GRAD_ACCUM_STEPS)))"
echo "  Samples:          $NUM_TRAIN_SAMPLES"
echo "  Output:           $OUTPUT_DIR"
echo "  Profile JSON:     $PROFILE_DIR/baseline_fullft.json"
echo "  W&B:              project=$WANDB_PROJECT group=$WANDB_RUN_GROUP report_to=$REPORT_TO"
echo ""

BASELINE_START=$SECONDS

python gsm8k_eval/finetune_gsm8k_full_params.py \
    --model_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --num_eval_samples 0 \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM_STEPS \
    --max_length $MAX_LENGTH \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER \
    --max_grad_norm $MAX_GRAD_NORM \
    --seed $SEED \
    --logging_steps $LOGGING_STEPS \
    --safety_mix_ratio 0.0 \
    --attn_implementation eager \
    --profile_json "$PROFILE_DIR/baseline_fullft.json" \
    --report_to "$REPORT_TO" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee $LOG_DIR/baseline_fullft_gsm8k_${TIMESTAMP}.log

ELAPSED=$((SECONDS - BASELINE_START))
echo ""
printf '⏱  Baseline wall-clock: %02d:%02d:%02d  (%ds)\n' \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) $ELAPSED

if [ -f "$PROFILE_DIR/baseline_fullft.json" ]; then
    python - "$PROFILE_DIR/baseline_fullft.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"🖥  peak VRAM: device={d['peak_device_gb']:.2f}GB  "
      f"torch_reserved={d['peak_torch_reserved_gb']:.2f}GB  "
      f"torch_alloc={d['peak_torch_alloc_gb']:.2f}GB")
print()
print(f"{'stage':<24}{'time':>10}{'device':>10}{'torch_resv':>13}")
print('-' * 57)
for s in d['stages']:
    print(f"{s['stage']:<24}{s['duration']:>10}{s['device_peak_gb']:>9.2f}G"
          f"{s['torch_reserved_peak_gb']:>12.2f}G")
PYEOF
fi

echo ""
echo "Output:  $OUTPUT_DIR"
echo "Log:     $LOG_DIR/baseline_fullft_gsm8k_${TIMESTAMP}.log"
echo "Profile: $PROFILE_DIR/baseline_fullft.json"
