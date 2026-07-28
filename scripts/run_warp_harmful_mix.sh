#!/bin/bash

# ========================================================================
# 실험: harmful fine-tuning 공격에 대한 내성 측정 (2 arm)
#
#   질문: 안전 좌표를 동결하면, downstream 데이터에 유해 응답이 섞여 들어와도
#         안전성이 유지되는가? 그리고 그 효과는 "회전(rotation)" 때문인가,
#         아니면 단순히 "중요 파라미터 동결" 때문인가?
#
#   ARM=warp      : WaRP. basis 로 회전한 좌표계에서 중요도 측정 → 동결
#                   (--basis_dir + --masks_dir, --non_freeze)
#   ARM=original  : 대조군. 회전 없음. 원본 weight 공간에서 |dL/dW| 로 중요도를
#                   측정해 그대로 entry-wise 동결 (--original_space_mask)
#
#   두 arm 은 시작 모델 / 학습 데이터 / 하이퍼파라미터 / keep_ratio 가 전부 같고,
#   **좌표계만** 다르다. 그래서 차이가 나면 그건 회전의 기여다.
#
#   시작 모델 : kmseong/llama2_7b-chat-Safety-FT-lr5e-5
#   학습 데이터: gsm8k train 전체(7473)
#               + data/beavertails_harmful_747.json 의 prompt + response(유해)
#                 747개 (= gsm8k 의 10%)  → 총 8220
#               (부분집합은 scripts/build_harmful_mix_subset.py 로 고정 저장)
#   keep_ratio: 0.1  (양쪽 동일 — 동결 파라미터 수가 같아 공정 비교)
#
#   ⚠️ 산출 모델은 유해 응답을 직접 학습하므로 안전성이 저하될 수 있는
#      "공격 모델"이다. 평가/비교 목적으로만 사용할 것.
#
#   사용:
#     bash scripts/run_warp_harmful_mix.sh              # ARM=warp (기본)
#     ARM=original bash scripts/run_warp_harmful_mix.sh # 회전 없는 대조군
#     CONTROL_USE_REFUSAL=1 bash scripts/run_warp_harmful_mix.sh   # 거부 응답 대조
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

# ========================================================================
# Configuration
# ========================================================================
ARM="${ARM:-warp}"                 # warp | original
if [[ ! "$ARM" =~ ^(warp|original)$ ]]; then
    echo "❌ ARM 은 warp 또는 original 이어야 합니다 (현재: $ARM)"; exit 1
fi

PHASE0_MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
KEEP_RATIO=0.1

# ARM=warp 가 재사용할 Phase 1 / Phase 2 산출물
BASIS_DIR="./checkpoints/phase1_20260727_155833/basis"
MASKS_DIR="./checkpoints/phase2_20260727_162036/checkpoints/masks"

# ARM=original 의 Phase 2 (원본 공간 중요도). 비어 있으면 새로 계산한다.
#   회전 좌표계의 마스크와는 다른 물건이므로 warp arm 의 마스크를 재사용할 수 없다.
ORIG_MASKS_DIR_OVERRIDE=""
PHASE2_DATASET="circuit_breakers"
PHASE2_SAMPLES=4994

# 혼합할 harmful 데이터 (양쪽 arm 공통)
MIX_DATA_PATH="./data/beavertails_harmful_747.json"
MIX_RATIO=0.1                    # gsm8k 7473 x 0.1 = 747
MIX_PROMPT_FIELD="prompt"
MIX_RESPONSE_FIELD="response"    # response=유해 응답 / refusal=거부 응답

# 대조군 스위치: 1 이면 **같은 747개 프롬프트**에 거부 응답을 붙여 학습한다.
CONTROL_USE_REFUSAL="${CONTROL_USE_REFUSAL:-0}"
if [ "$CONTROL_USE_REFUSAL" = "1" ]; then
    MIX_RESPONSE_FIELD="refusal"
fi

# 학습 하이퍼파라미터 (직전 kr0.1/0.4/0.5 실행과 동일 — 양쪽 arm 공통)
EPOCHS=3
LEARNING_RATE=5e-5
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
MAX_LENGTH=1024
DTYPE="bfloat16"
DEVICE="auto"
SEED=42
TARGET_LAYERS="all"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"

BASE_OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="$LOG_DIR/profile_harmfulmix_${ARM}_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR" "$PROFILE_DIR"

# W&B
USE_WANDB=1
WANDB_PROJECT="Safety-WaRP-LLM"
export WANDB_RUN_GROUP="harmfulmix_${ARM}_${TIMESTAMP}"
export WANDB_TAGS="harmful_mix,${ARM},gsm8k,kr${KEEP_RATIO},no_csft"

# HF 업로드
UPLOAD_TO_HF=1
HF_PRIVATE=0
if [ "$ARM" = "warp" ]; then ARM_TAG="WaRP"; else ARM_TAG="OrigSpace"; fi
if [ "$MIX_RESPONSE_FIELD" = "response" ]; then MIX_TAG="harmfulmix10p"; else MIX_TAG="refusalmix10p"; fi
HF_REPO="kmseong/llama2_7b_chat-${ARM_TAG}-kr${KEEP_RATIO}-gsm8k_${MIX_TAG}_lr${LEARNING_RATE}"

# ========================================================================
# Preflight
# ========================================================================
echo "========================================================================"
echo "harmful-mix Phase 3   [ARM=$ARM]"
echo "========================================================================"
echo "  Start model  : $PHASE0_MODEL"
if [ "$ARM" = "warp" ]; then
    echo "  좌표계       : WaRP 회전 공간 (basis 사용)"
    echo "  Basis        : $BASIS_DIR"
    echo "  Masks        : $MASKS_DIR  (keep_ratio=$KEEP_RATIO)"
else
    echo "  좌표계       : 원본 weight 공간 (회전 없음, |dL/dW| 중요도)"
    echo "  Masks        : $([ -n "$ORIG_MASKS_DIR_OVERRIDE" ] && echo "$ORIG_MASKS_DIR_OVERRIDE" || echo '이번 실행에서 Phase 2 새로 계산')"
fi
echo "  Downstream   : gsm8k (전체 7473)"
echo "  Mixed data   : $MIX_DATA_PATH  (ratio=$MIX_RATIO, response='$MIX_RESPONSE_FIELD')"
if [ "$MIX_RESPONSE_FIELD" = "response" ]; then
    echo "                 ⚠️ 유해 응답 학습 (harmful FT 공격 설정)"
fi
echo "  Epochs / LR  : $EPOCHS / $LEARNING_RATE"
echo "  HF repo      : $HF_REPO  ($([ "$HF_PRIVATE" = "1" ] && echo private || echo public))"
echo ""

[ -f "$MIX_DATA_PATH" ] || { echo "❌ 혼합 데이터 없음: $MIX_DATA_PATH"; exit 1; }
if [ "$ARM" = "warp" ]; then
    for p in "$BASIS_DIR" "$MASKS_DIR"; do
        [ -d "$p" ] || { echo "❌ 경로 없음: $p"; exit 1; }
    done
fi

python - "$MIX_DATA_PATH" "$MIX_PROMPT_FIELD" "$MIX_RESPONSE_FIELD" <<'PYEOF'
import json, sys
path, pf, rf = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(path, encoding='utf-8'))
ok = sum(1 for d in data if d.get(pf) and d.get(rf))
print(f"  혼합 데이터 확인: {len(data)}개 중 '{pf}'+'{rf}' 유효 {ok}개")
if ok == 0:
    print(f"  ❌ 필드명 불일치. 사용 가능한 키: {list(data[0].keys())}")
    sys.exit(1)
PYEOF

if [ "$USE_WANDB" = "1" ]; then
    python -c "import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null \
        || { echo "❌ W&B 미인증. 'wandb login' 후 재실행 (또는 USE_WANDB=0)"; exit 1; }
    WANDB_ARG="--wandb_project $WANDB_PROJECT"
else
    WANDB_ARG="--no_wandb"
fi
if [ "$UPLOAD_TO_HF" = "1" ]; then
    hf auth whoami >/dev/null 2>&1 || { echo "❌ HF 미로그인. 'hf auth login' 후 재실행 (또는 UPLOAD_TO_HF=0)"; exit 1; }
fi
echo "✅ preflight 통과"
echo ""

fmt_hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

show_profile() {
    [ -f "$1" ] || return 0
    python - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"🖥  peak VRAM: device={d['peak_device_gb']:.2f}GB  torch_reserved={d['peak_torch_reserved_gb']:.2f}GB")
for s in d['stages']:
    print(f"    {s['stage']:<24}{s['duration']:>10}   peak {s['device_peak_gb']:.2f}GB")
PYEOF
}

# ========================================================================
# ARM=original 전용: Phase 2 (원본 공간 중요도) — 회전 없이 |dL/dW| 로 마스크 생성
# ========================================================================
if [ "$ARM" = "original" ]; then
    if [ -n "$ORIG_MASKS_DIR_OVERRIDE" ]; then
        MASKS_DIR="$ORIG_MASKS_DIR_OVERRIDE"
        [ -d "$MASKS_DIR" ] || { echo "❌ ORIG_MASKS_DIR_OVERRIDE 경로 없음: $MASKS_DIR"; exit 1; }
        echo "✅ Phase 2 (original space) 스킵 — 기존 마스크 사용: $MASKS_DIR"
    else
        echo "========================================================================"
        echo "PHASE 2 (original space): |dL/dW| importance, keep_ratio=$KEEP_RATIO"
        echo "========================================================================"
        P2_PROFILE_JSON="$PROFILE_DIR/phase2_original_space.json"
        P2_START=$SECONDS

        python train.py \
            --phase 2 \
            --phase0_model_dir "$PHASE0_MODEL" \
            --original_space_mask \
            --dataset_phase2 $PHASE2_DATASET \
            --circuit_breakers_samples_phase2 $PHASE2_SAMPLES \
            --circuit_breakers_path ./data/circuit_breakers_train.json \
            --keep_ratio $KEEP_RATIO \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --layer_type "$LAYER_TYPE" \
            --target_layers $TARGET_LAYERS \
            --output_dir $BASE_OUTPUT_DIR \
            --log_dir $LOG_DIR \
            --device $DEVICE \
            --dtype $DTYPE \
            --seed $SEED \
            --perlayer \
            --profile_json "$P2_PROFILE_JSON" \
            $WANDB_ARG --wandb_run_name "p2_origspace_kr${KEEP_RATIO}_${TIMESTAMP}" \
            2>&1 | tee $LOG_DIR/harmfulmix_${ARM}_phase2_${TIMESTAMP}.log

        echo ""
        echo "⏱  Phase 2 wall-clock: $(fmt_hms $((SECONDS - P2_START)))"
        show_profile "$P2_PROFILE_JSON"

        P2_OUT=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase2_original_space_*" -type d -printf '%T@ %p\n' \
            | sort -rn | head -1 | cut -d' ' -f2-)
        MASKS_DIR="$P2_OUT/checkpoints/masks"
        [ -d "$MASKS_DIR" ] || { echo "❌ Phase 2 마스크를 찾지 못했습니다: $MASKS_DIR"; exit 1; }
        echo ""
        echo "✅ Phase 2 (original space) 완료: $MASKS_DIR"
        echo ""
    fi
fi

# ========================================================================
# Phase 3
# ========================================================================
echo "========================================================================"
echo "PHASE 3  [ARM=$ARM]"
echo "========================================================================"

if [ "$ARM" = "warp" ]; then
    ARM_ARGS="--basis_dir $BASIS_DIR --non_freeze"
else
    ARM_ARGS="--original_space_mask"      # basis_dir 불필요 (회전 없음)
fi

PHASE3_PROFILE_JSON="$PROFILE_DIR/phase3_harmfulmix.json"
START=$SECONDS

python train.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    $ARM_ARGS \
    --masks_dir "$MASKS_DIR" \
    --phase3_dataset gsm8k \
    --gsm8k_samples 0 \
    --safety_mix_ratio $MIX_RATIO \
    --circuit_breakers_path "$MIX_DATA_PATH" \
    --mix_prompt_field "$MIX_PROMPT_FIELD" \
    --mix_response_field "$MIX_RESPONSE_FIELD" \
    --epochs $EPOCHS \
    --utility_lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_length $MAX_LENGTH \
    --layer_type "$LAYER_TYPE" \
    --target_layers $TARGET_LAYERS \
    --output_dir $BASE_OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --device $DEVICE \
    --dtype $DTYPE \
    --seed $SEED \
    --profile_json "$PHASE3_PROFILE_JSON" \
    $WANDB_ARG --wandb_run_name "p3_harmfulmix_${ARM}_kr${KEEP_RATIO}_lr${LEARNING_RATE}_${TIMESTAMP}" \
    2>&1 | tee $LOG_DIR/harmfulmix_${ARM}_phase3_${TIMESTAMP}.log

ELAPSED=$((SECONDS - START))
echo ""
echo "⏱  Phase 3 wall-clock: $(fmt_hms $ELAPSED)  (${ELAPSED}s)"
show_profile "$PHASE3_PROFILE_JSON"

# arm 별 정확한 디렉토리명으로 찾는다. phase3_* 로 뭉뚱그리면 이전 실행의
# 다른 arm 디렉토리를 집어올 수 있다.
if [ "$ARM" = "warp" ]; then
    P3_PATTERN="phase3_non_freeze_*"
else
    P3_PATTERN="phase3_original_space_*"
fi
PHASE3_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "$P3_PATTERN" -type d -printf '%T@ %p\n' \
    | sort -rn | head -1 | cut -d' ' -f2-)
echo ""
echo "Phase 3 output: $PHASE3_OUTPUT_DIR"
[ -d "$PHASE3_OUTPUT_DIR/final_model" ] || { echo "❌ final_model 을 찾지 못했습니다."; exit 1; }

# ========================================================================
# HF 업로드 + 검증 + 로컬 정리
# ========================================================================
if [ "$UPLOAD_TO_HF" = "1" ]; then
    [ "$HF_PRIVATE" = "1" ] && HF_PRIVATE_ARG="--private" || HF_PRIVATE_ARG=""
    [ "$MIX_RESPONSE_FIELD" = "response" ] && HF_WARN_ARG="--research_warning" || HF_WARN_ARG=""
    echo ""
    echo "── HF 업로드: $HF_REPO ──"
    python scripts/upload_and_cleanup_phase3.py \
        --model_dir "$PHASE3_OUTPUT_DIR/final_model" \
        --repo_name "$HF_REPO" \
        --base_model "$PHASE0_MODEL" \
        --keep_ratio "$KEEP_RATIO" \
        --learning_rate "$LEARNING_RATE" \
        --dataset "gsm8k + beavertails-${MIX_TAG} [${ARM_TAG}]" \
        --metadata_json "$PHASE3_OUTPUT_DIR/metadata.json" \
        $HF_PRIVATE_ARG $HF_WARN_ARG \
        --delete_after_verify \
        2>&1 | tee -a $LOG_DIR/harmfulmix_${ARM}_upload_${TIMESTAMP}.log
    echo ""
    echo "🔗 https://huggingface.co/$HF_REPO"
fi

echo ""
echo "========================================================================"
echo "완료  [ARM=$ARM]"
echo "  Masks   : $MASKS_DIR"
echo "  Log     : $LOG_DIR/harmfulmix_${ARM}_phase3_${TIMESTAMP}.log"
echo "  Profile : $PROFILE_DIR/"
echo "  Meta    : $PHASE3_OUTPUT_DIR/metadata.json"
echo "========================================================================"
