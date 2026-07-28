#!/usr/bin/env bash
#
# LISA (Bi-State Optimization) on GSM8K — matched-sweep LoRA 예산 + epoch 10 단일 지점.
#
# matched 스윕(scripts/run_matched_lora_lr_sweep.sh)과 같은 LoRA/옵티마이저 설정을 쓰되
# epoch 만 3 → 10 으로 늘리고 lr 은 3e-4 한 점만 돌린다. 따라서
# outputs/matched_lora_lr_sweep/lisa/lr_3e-4 (epoch 3) 과 epoch 축으로 직접 짝비교가 된다.
#
#   optimizer      : AdamW  (finetune_gsm8k_lisa.py 가 optim="adamw_torch" 로 고정)
#   lr             : 3e-4
#   batch          : 16 (grad_accum 1 → 유효 배치 16)
#   LoRA           : r=16, alpha=32, dropout=0.05, targets q,k,v,up,down
#   warmup / wd    : 0.03 / 0.0,  scheduler cosine,  seed 42
#   epochs         : 10        → 7473/16 = 468 step/epoch × 10 ≈ 4,680 step
#   BSO            : rho=1.0, alignment_step=100, finetune_step=900, guide_data_num=4994
#   safety data    : data/circuit_breakers_train.json
#   downstream     : GSM8K train 7,473
#
# 사용법:
#   bash scripts/run_lisa_cb_ep10.sh
#
# 오버라이드:
#   PY=/path/to/python LR=3e-4 EPOCHS=10 \
#   PUSH_TO_HUB=1 HF_NAMESPACE=kmseong \
#   bash scripts/run_lisa_cb_ep10.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-/venv/hb/bin/python}"
MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/lisa_cb_epoch_sweep}"

LR="${LR:-3e-4}"
EPOCHS="${EPOCHS:-10}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-42}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES=(q_proj k_proj v_proj up_proj down_proj)

# BSO — 이 저장소의 다른 LISA 실행과 동일하게 유지 (교차 비교 가능하도록).
RHO="${RHO:-1.0}"
ALIGNMENT_STEP="${ALIGNMENT_STEP:-100}"
FINETUNE_STEP="${FINETUNE_STEP:-900}"
GUIDE_DATA_NUM="${GUIDE_DATA_NUM:-4994}"

NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-7473}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-}"
HF_REPO_ID="${HF_REPO_ID:-}"

# ⚠️ CUDA_VISIBLE_DEVICES 를 여기서 설정하지 않는다 (스케줄러/단일 GPU 가 정한다).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$SAFETY_DATA" ]]; then
    echo "Safety dataset not found: $SAFETY_DATA" >&2
    exit 1
fi

if [[ "$PUSH_TO_HUB" == "1" && -z "$HF_NAMESPACE" && -z "$HF_REPO_ID" ]]; then
    echo "PUSH_TO_HUB=1 이면 HF_NAMESPACE 또는 HF_REPO_ID 가 필요합니다" >&2
    exit 1
fi

# rho 를 경로/레포 이름에 넣는다. 넣지 않으면 epoch 만 같고 rho 가 다른 run 이
# 같은 디렉터리를 가리켜 skip 가드에 걸려 조용히 건너뛰어진다.
OUT_DIR="$OUTPUT_ROOT/lr_${LR}_ep${EPOCHS}_rho${RHO}"

# 완료된 run 은 건너뛴다 (죽어도 재실행하면 이어서 간다).
if [[ -f "$OUT_DIR/finetune_config.json" ]]; then
    echo "[LISA lr=$LR ep=$EPOCHS] already complete; skipping ($OUT_DIR)"
    exit 0
fi
mkdir -p "$OUT_DIR"

PUSH_ARGS=()
if [[ "$PUSH_TO_HUB" == "1" ]]; then
    repo="${HF_REPO_ID:-${HF_NAMESPACE}/llama2_7b-chat-gsm8k-lisa-cb-r${LORA_R}a${LORA_ALPHA}-lr${LR}-ep${EPOCHS}-rho${RHO}}"
    PUSH_ARGS=(--upload_name "$repo")
    echo "  will push to      : $repo"
fi

echo "LISA on GSM8K — matched LoRA budget, epoch $EPOCHS"
echo "  model             : $MODEL"
echo "  safety data       : $SAFETY_DATA (guide_data_num=$GUIDE_DATA_NUM)"
echo "  downstream        : GSM8K train $NUM_TRAIN_SAMPLES"
echo "  optimizer         : AdamW (adamw_torch, 스크립트 고정)"
echo "  lr / epochs       : $LR / $EPOCHS"
echo "  effective batch   : $((MICRO_BATCH * GRAD_ACCUM)) ($MICRO_BATCH x $GRAD_ACCUM)"
echo "  LoRA              : r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  targets           : ${TARGET_MODULES[*]}"
echo "  warmup / wd       : $WARMUP_RATIO / $WEIGHT_DECAY  (scheduler cosine, seed $SEED)"
echo "  BSO               : rho=$RHO align=$ALIGNMENT_STEP finetune=$FINETUNE_STEP"
echo "  output            : $OUT_DIR"

"$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
    --model_path "$MODEL" \
    --output_dir "$OUT_DIR" \
    --dataset_name openai/gsm8k \
    --dataset_subset main \
    --train_split train \
    --num_train_samples "$NUM_TRAIN_SAMPLES" \
    --num_eval_samples 0 \
    --safety_data_path "$SAFETY_DATA" \
    --guide_data_num "$GUIDE_DATA_NUM" \
    --rho "$RHO" \
    --alignment_step "$ALIGNMENT_STEP" \
    --finetune_step "$FINETUNE_STEP" \
    --lora \
    --lora_target_modules "${TARGET_MODULES[@]}" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --max_length "$MAX_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --lr_scheduler_type cosine \
    --seed "$SEED" \
    --bf16 \
    --gradient_checkpointing \
    --report_to none \
    "${PUSH_ARGS[@]}" 2>&1 | tee "$OUT_DIR/run.log"

echo "done: $OUT_DIR"
