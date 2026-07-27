#!/usr/bin/env bash
#
# Matched GSM8K LR sweep for:
#   1) standard LoRA, 2) SafeLoRA, 3) SaLoRA, 4) LISA.
#
# Common settings:
#   LR={1e-4,2e-4,3e-4}, r=16, alpha=32, dropout=0.05,
#   batch/GPU=16, no gradient accumulation,
#   warmup=0.03, weight decay=0, cosine scheduler,
#   targets={q,k,v,up,down}, GSM8K train split, seed=42.
#
# Method-specific safety behavior is intentionally retained:
#   SafeLoRA: post-hoc projection using base/aligned model delta.
#   SaLoRA:   all 4,994 circuit-breakers examples are used for calibration.
#             The current runner uses the same sample count for utility
#             calibration, so 4,994 GSM8K examples are calibrated as well.
#   LISA:     all 4,994 safety examples are used during bi-state training.
#
# Usage:
#   bash scripts/run_matched_lora_lr_sweep.sh
#
# Useful overrides:
#   PY=/path/to/python OUTPUT_ROOT=/scratch/exp \
#   METHODS="lora safelora" PUSH_TO_HUB=1 HF_NAMESPACE=myname \
#   bash scripts/run_matched_lora_lr_sweep.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
SAFELORA_BASE_MODEL="${SAFELORA_BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
# SafeLoRA 의 alignment delta V = W_aligned - W_base 에서 W_aligned 로 쓸 모델.
# 기본값은 파인튜닝 대상 모델($MODEL) — 공식 구현의 가정(적응 대상 == aligned 모델).
# 다른 안전 데이터로 SSFT 한 모델을 지정하면 SafeLoRA 도 안전 데이터 축에 의존하게 된다.
SAFELORA_ALIGNED_MODEL="${SAFELORA_ALIGNED_MODEL:-$MODEL}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/beavertails_cb_train.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/matched_lora_lr_sweep}"
METHODS="${METHODS:-lora safelora salora lisa}"

# 안전 데이터셋을 바꿔 돌릴 때 실행을 구분하는 태그 (HF repo 이름 접미사).
# 비워두면(기본값) 기존 circuit_breakers 실행과 경로/이름이 완전히 동일하게 유지된다.
RUN_TAG="${RUN_TAG:-}"
NAME_SUFFIX="${RUN_TAG:+-$RUN_TAG}"

LRS=(1e-4 2e-4 3e-4)
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
MICRO_BATCH=16
GRAD_ACCUM=1
EFFECTIVE_BATCH=$((MICRO_BATCH * GRAD_ACCUM))
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
SEED="${SEED:-42}"
SAFETY_SAMPLES=4994

TARGET_MODULES_CSV="q_proj,k_proj,v_proj,up_proj,down_proj"
TARGET_MODULES_LIST=(q_proj k_proj v_proj up_proj down_proj)
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"

# Method-specific settings (not part of the matched optimizer/LoRA budget).
SAFELORA_THRESHOLD="${SAFELORA_THRESHOLD:-0.35}"
SALORA_RANK_SAFE="${SALORA_RANK_SAFE:-32}"
SALORA_RANK_UTIL="${SALORA_RANK_UTIL:-32}"
SALORA_CALIB_BATCH="${SALORA_CALIB_BATCH:-2}"
SALORA_NITER="${SALORA_NITER:-20}"
LISA_RHO="${LISA_RHO:-1.0}"
LISA_ALIGNMENT_STEP="${LISA_ALIGNMENT_STEP:-100}"
LISA_FINETUNE_STEP="${LISA_FINETUNE_STEP:-900}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$SAFETY_DATA" ]]; then
    echo "Safety dataset not found: $SAFETY_DATA" >&2
    exit 1
fi

if [[ "$PUSH_TO_HUB" == "1" && -z "$HF_NAMESPACE" ]]; then
    echo "HF_NAMESPACE is required when PUSH_TO_HUB=1" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

has_method() {
    [[ " $METHODS " == *" $1 "* ]]
}

push_args_lora_family() {
    local repo_id=$1
    PUSH_ARGS=()
    if [[ "$PUSH_TO_HUB" == "1" ]]; then
        PUSH_ARGS=(--push_to_hub --hf_repo_id "$repo_id")
    fi
}

push_args_lisa() {
    local repo_id=$1
    PUSH_ARGS=()
    if [[ "$PUSH_TO_HUB" == "1" ]]; then
        PUSH_ARGS=(--upload_name "$repo_id")
    fi
}

echo "Matched LoRA-family GSM8K LR sweep"
echo "  model             : $MODEL"
echo "  methods           : $METHODS"
echo "  learning rates    : ${LRS[*]}"
echo "  effective batch   : $EFFECTIVE_BATCH ($MICRO_BATCH x $GRAD_ACCUM)"
echo "  LoRA              : r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  targets           : $TARGET_MODULES_CSV"
echo "  optimizer settings: warmup=$WARMUP_RATIO weight_decay=$WEIGHT_DECAY scheduler=cosine"
echo "  safety data       : $SAFETY_DATA ($SAFETY_SAMPLES examples)"
echo "  safelora base     : $SAFELORA_BASE_MODEL"
echo "  safelora aligned  : $SAFELORA_ALIGNED_MODEL"
echo "  run tag           : ${RUN_TAG:-<none>}"
echo "  output            : $OUTPUT_ROOT"

for lr in "${LRS[@]}"; do
    if has_method lora; then
        out_dir="$OUTPUT_ROOT/lora/lr_$lr"
        if [[ -f "$out_dir/summary.json" ]]; then
            echo "[LoRA lr=$lr] already complete; skipping"
        else
            mkdir -p "$out_dir"
            push_args_lora_family "${HF_NAMESPACE}/llama2_7b-chat-gsm8k-lora-matched-r16-a32-lr${lr}${NAME_SUFFIX}"
            "$PY" finetune_gsm8k_lora.py \
                --method lora \
                --model_name "$MODEL" \
                --output_dir "$out_dir" \
                --target_modules "$TARGET_MODULES_CSV" \
                --layer_type "$LAYER_TYPES" \
                --target_layers all \
                --lora_r "$LORA_R" \
                --lora_alpha "$LORA_ALPHA" \
                --lora_dropout "$LORA_DROPOUT" \
                --learning_rate "$lr" \
                --epochs "$EPOCHS" \
                --batch_size "$MICRO_BATCH" \
                --gradient_accumulation_steps "$GRAD_ACCUM" \
                --max_length "$MAX_LENGTH" \
                --warmup_ratio "$WARMUP_RATIO" \
                --weight_decay "$WEIGHT_DECAY" \
                --seed "$SEED" \
                --dtype bfloat16 \
                --gradient_checkpointing \
                --save_merged_model \
                "${PUSH_ARGS[@]}" 2>&1 | tee "$out_dir/run.log"
        fi
    fi

    if has_method safelora; then
        out_dir="$OUTPUT_ROOT/safelora/lr_$lr"
        if [[ -f "$out_dir/summary.json" ]]; then
            echo "[SafeLoRA lr=$lr] already complete; skipping"
        else
            mkdir -p "$out_dir"
            push_args_lora_family "${HF_NAMESPACE}/llama2_7b-chat-gsm8k-safelora-matched-r16-a32-lr${lr}${NAME_SUFFIX}"
            "$PY" finetune_gsm8k_lora.py \
                --method safe_lora \
                --model_name "$MODEL" \
                --output_dir "$out_dir" \
                --target_modules "$TARGET_MODULES_CSV" \
                --layer_type "$LAYER_TYPES" \
                --target_layers all \
                --safelora_base_model "$SAFELORA_BASE_MODEL" \
                --safelora_aligned_model "$SAFELORA_ALIGNED_MODEL" \
                --safelora_select_type threshold \
                --safelora_threshold "$SAFELORA_THRESHOLD" \
                --safelora_load_dtype float32 \
                --lora_r "$LORA_R" \
                --lora_alpha "$LORA_ALPHA" \
                --lora_dropout "$LORA_DROPOUT" \
                --learning_rate "$lr" \
                --epochs "$EPOCHS" \
                --batch_size "$MICRO_BATCH" \
                --gradient_accumulation_steps "$GRAD_ACCUM" \
                --max_length "$MAX_LENGTH" \
                --warmup_ratio "$WARMUP_RATIO" \
                --weight_decay "$WEIGHT_DECAY" \
                --seed "$SEED" \
                --dtype bfloat16 \
                --gradient_checkpointing \
                --save_merged_model \
                "${PUSH_ARGS[@]}" 2>&1 | tee "$out_dir/run.log"
        fi
    fi

    if has_method salora; then
        out_dir="$OUTPUT_ROOT/salora/lr_$lr"
        if [[ -f "$out_dir/summary.json" ]]; then
            echo "[SaLoRA lr=$lr] already complete; skipping"
        else
            mkdir -p "$out_dir"
            push_args_lora_family "${HF_NAMESPACE}/llama2_7b-chat-gsm8k-salora-matched-r16-a32-lr${lr}${NAME_SUFFIX}"
            "$PY" finetune_gsm8k_salora.py \
                --model_name "$MODEL" \
                --output_dir "$out_dir" \
                --safety_data_path "$SAFETY_DATA" \
                --salora_rank_safe "$SALORA_RANK_SAFE" \
                --salora_rank_util "$SALORA_RANK_UTIL" \
                --salora_calib_samples "$SAFETY_SAMPLES" \
                --salora_calib_batch_size "$SALORA_CALIB_BATCH" \
                --salora_niter "$SALORA_NITER" \
                --target_modules "$TARGET_MODULES_CSV" \
                --layer_type "$LAYER_TYPES" \
                --target_layers all \
                --lora_r "$LORA_R" \
                --lora_alpha "$LORA_ALPHA" \
                --lora_dropout "$LORA_DROPOUT" \
                --learning_rate "$lr" \
                --epochs "$EPOCHS" \
                --batch_size "$MICRO_BATCH" \
                --gradient_accumulation_steps "$GRAD_ACCUM" \
                --max_length "$MAX_LENGTH" \
                --warmup_ratio "$WARMUP_RATIO" \
                --weight_decay "$WEIGHT_DECAY" \
                --seed "$SEED" \
                --dtype bfloat16 \
                --gradient_checkpointing \
                "${PUSH_ARGS[@]}" 2>&1 | tee "$out_dir/run.log"
        fi
    fi

    if has_method lisa; then
        out_dir="$OUTPUT_ROOT/lisa/lr_$lr"
        if [[ -f "$out_dir/finetune_config.json" ]]; then
            echo "[LISA lr=$lr] already complete; skipping"
        else
            mkdir -p "$out_dir"
            push_args_lisa "${HF_NAMESPACE}/llama2_7b-chat-gsm8k-lisa-matched-r16-a32-lr${lr}${NAME_SUFFIX}"
            "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
                --model_path "$MODEL" \
                --output_dir "$out_dir" \
                --dataset_name openai/gsm8k \
                --dataset_subset main \
                --train_split train \
                --num_train_samples 7473 \
                --num_eval_samples 0 \
                --safety_data_path "$SAFETY_DATA" \
                --guide_data_num "$SAFETY_SAMPLES" \
                --rho "$LISA_RHO" \
                --alignment_step "$LISA_ALIGNMENT_STEP" \
                --finetune_step "$LISA_FINETUNE_STEP" \
                --lora \
                --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
                --lora_r "$LORA_R" \
                --lora_alpha "$LORA_ALPHA" \
                --lora_dropout "$LORA_DROPOUT" \
                --learning_rate "$lr" \
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
                "${PUSH_ARGS[@]}" 2>&1 | tee "$out_dir/run.log"
        fi
    fi
done

echo "All requested runs completed: $OUTPUT_ROOT"
