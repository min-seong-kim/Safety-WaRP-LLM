#!/usr/bin/env bash
#
# SaLoRA 원인 4종을 한꺼번에 되돌린 "fixed" 설정 1점.
#
# 관측: matched SaLoRA (lr3e-4) 가 무보호 기준선보다 ASR 이 나쁘다.
#   SSFT+DT (무보호)          ASR 0.2064  GSM8K 0.4117
#   SaLoRA matched (bv)       ASR 0.4933  GSM8K 0.3351
#
# 실측 상대 드리프트 ||ΔW||_F/||W||_F (같은 lr/rank/데이터):
#   plain LoRA        0.0488
#   SaLoRA matched    0.0899   ← LoRA 보다 1.84배 더 움직임 (= scaling 2 와 일치)
#   SaLoRA faithful   0.0275   ← matched 대비 3.31배 작음
#
# 되돌린 것:
#   (1) scaling  α/r = 2 → 1      : --lora_alpha 16 (== --lora_r). 원본 SaLoRA 가정.
#   (2) target   q,k,v,up,down → q,v : 논문 설정. 움직일 표면적 2.5배 축소.
#   (3) rank_safe 32 → 128         : C = I − V_sV_sᵀ 가 지키는 출력 차원을
#                                    4096 중 32(0.8%) → 128(3.1%) 로 확대.
#   (4) _top_eigvecs oversampling  : models/salora.py 에서 randomized SVD 를
#                                    q → q+10 sketch 후 절단하도록 수정(기본값).
#                                    합성 검증에서 부분공간 오차 0.617 → 0.000.
#
# (1)(2) 는 이미 salora-faithful-qv-a16-r16-lr3e-4 로 검증 가능하므로,
# 이 run 은 거기에 (3)(4) 를 더한 것이다 → faithful 과의 차이가 (3)+(4) 효과.
#
# base/safety 는 circuit_breakers 계열로 고정한다. faithful 과 matched(cb) 가
# 같은 base 라 3점 사다리(matched → faithful → fixed)가 성립하기 때문이다.
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-/venv/hb/bin/python}"
MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"

LR="${LR:-3e-4}"
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"          # (1) scaling = 1
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
RANK_SAFE="${RANK_SAFE:-128}"           # (3) 보호 부분공간 확대
RANK_UTIL="${RANK_UTIL:-32}"
CALIB_SAMPLES="${CALIB_SAMPLES:-4994}"
CALIB_BATCH="${CALIB_BATCH:-2}"
NITER="${NITER:-20}"

TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"   # (2) 논문 설정
LAYER_TYPES="${LAYER_TYPES:-attn_q,attn_v}"

MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"

OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/salora_fixed/lr_${LR}_ep${EPOCHS}}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
HF_REPO_ID="${HF_REPO_ID:-${HF_NAMESPACE}/llama2_7b-chat-gsm8k-salora-fixed-qv-r${LORA_R}a${LORA_ALPHA}-rs${RANK_SAFE}-svdfix-lr${LR}-ep${EPOCHS}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

[[ -f "$SAFETY_DATA" ]] || { echo "Safety dataset not found: $SAFETY_DATA" >&2; exit 1; }

if [[ -f "$OUT_DIR/summary.json" ]]; then
    echo "already complete; skipping ($OUT_DIR)"; exit 0
fi
mkdir -p "$OUT_DIR"

PUSH_ARGS=()
[[ "$PUSH_TO_HUB" == "1" ]] && PUSH_ARGS=(--push_to_hub --hf_repo_id "$HF_REPO_ID")

echo "SaLoRA fixed (원인 4종 반영)"
echo "  base            : $MODEL"
echo "  safety data     : $SAFETY_DATA (calib=$CALIB_SAMPLES)"
echo "  (1) scaling     : alpha/r = $LORA_ALPHA/$LORA_R = $(awk "BEGIN{print $LORA_ALPHA/$LORA_R}")"
echo "  (2) targets     : $TARGET_MODULES"
echo "  (3) rank_safe   : $RANK_SAFE (util=$RANK_UTIL)"
echo "  (4) svd fix     : _top_eigvecs oversample=10 (models/salora.py)"
echo "  lr / epochs     : $LR / $EPOCHS   batch $((MICRO_BATCH*GRAD_ACCUM))"
echo "  output          : $OUT_DIR"
[[ "$PUSH_TO_HUB" == "1" ]] && echo "  push            : $HF_REPO_ID"

"$PY" finetune_gsm8k_salora.py \
    --model_name "$MODEL" \
    --output_dir "$OUT_DIR" \
    --safety_data_path "$SAFETY_DATA" \
    --salora_rank_safe "$RANK_SAFE" \
    --salora_rank_util "$RANK_UTIL" \
    --salora_calib_samples "$CALIB_SAMPLES" \
    --salora_calib_batch_size "$CALIB_BATCH" \
    --salora_niter "$NITER" \
    --target_modules "$TARGET_MODULES" \
    --layer_type "$LAYER_TYPES" \
    --target_layers all \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$MICRO_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_length "$MAX_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --dtype bfloat16 \
    --gradient_checkpointing \
    "${PUSH_ARGS[@]}" 2>&1 | tee "$OUT_DIR/run.log"

echo "done: $OUT_DIR"
