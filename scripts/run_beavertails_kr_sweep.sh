#!/bin/bash

# ========================================================================
# 실험: beavertails 기반 WaRP, freeze ratio 스윕
#
#   Phase 1 : beavertails (harmful prompt + llama3_output = refusal) 로 basis 생성
#   Phase 2 : 같은 beavertails 로 중요도 측정 → keep_ratio 별 mask
#   Phase 3 : gsm8k 전체(7473) 로 downstream FT, 순수 WaRP (--non_freeze, csft 없음)
#
#   keep_ratio: 0.01 0.05 0.15 0.20 0.30 0.40 0.50
#
#   ⚠️ circuit_breakers 와 beavertails 는 스키마도 개수(4994)도 같지만 내용이 다르다
#      (prompt 교집합 7개). basis/mask 를 섞어 쓰지 않도록 주의.
#
#   디스크 전략: Phase 3 가 끝나면 모델은 HF 업로드 후 삭제, 마스크도 삭제한다.
#      마스크 7세트를 동시에 두면 126GB 라 담을 수 없다.
#      (Phase 2 는 keep_ratio 마다 중요도를 재계산한다 — 원래는 1회면 충분하지만
#       7세트를 한 번에 저장할 디스크가 없어 순차 방식을 택했다. 약 42분 손해.)
#
#   사용:
#     bash scripts/run_beavertails_kr_sweep.sh
#     KEEP_RATIO_LIST="0.01 0.05" bash scripts/run_beavertails_kr_sweep.sh   # 일부만
#     PHASE1_BASIS_DIR_OVERRIDE=./checkpoints/phase1_XXX/basis bash ...       # basis 재사용
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
PHASE0_MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"

# 안전 데이터: beavertails (prompt + llama3_output)
#   --safety_dataset circuit_breakers 는 "이 스키마의 JSON 로더"를 뜻하며,
#   실제 파일은 --circuit_breakers_path 로 지정한다.
SAFETY_DATA_PATH="./data/beavertails_cb_train.json"
SAFETY_SAMPLES=4994

KEEP_RATIO_LIST="${KEEP_RATIO_LIST:-0.01 0.05 0.15 0.20 0.30 0.40 0.50}"

# Phase 1 재사용 (비우면 새로 생성)
PHASE1_BASIS_DIR_OVERRIDE="${PHASE1_BASIS_DIR_OVERRIDE:-}"

# Phase 3: downstream
PHASE3_DATASET="gsm8k"
PHASE3_SAMPLES=0          # 0 = 전체 7473

# 학습 하이퍼파라미터 (기존 실험과 동일)
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
PROFILE_DIR="$LOG_DIR/profile_btsweep_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR" "$PROFILE_DIR"

# 디스크: Phase 3 후 마스크 삭제 (0 이면 보존 — 7세트면 126GB 필요)
DELETE_MASKS_AFTER_PHASE3=1
MIN_FREE_GB=40            # 각 keep_ratio 시작 전 최소 여유 공간

# W&B
USE_WANDB=1
WANDB_PROJECT="Safety-WaRP-LLM"
export WANDB_RUN_GROUP="btsweep_${TIMESTAMP}"
export WANDB_TAGS="beavertails,warp,gsm8k,kr_sweep,no_csft"

# HF
UPLOAD_TO_HF=1
HF_PRIVATE=0
HF_REPO_PREFIX="kmseong/llama2_7b_chat-WaRP-beavertails"
HF_UPLOADED=()

# ========================================================================
# Helpers
# ========================================================================
fmt_hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }
free_gb()  { df -BG --output=avail /home/edgeai_lab | tail -1 | tr -dc '0-9'; }

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
# Preflight
# ========================================================================
echo "========================================================================"
echo "beavertails WaRP keep_ratio sweep"
echo "========================================================================"
echo "  Start model : $PHASE0_MODEL"
echo "  Safety data : $SAFETY_DATA_PATH  (prompt + llama3_output = refusal)"
echo "  Keep ratios : $KEEP_RATIO_LIST"
echo "  Downstream  : $PHASE3_DATASET (전체)"
echo "  Phase 3     : 순수 WaRP (--non_freeze, csft 없음)"
echo "  Free disk   : $(free_gb) GB"
echo ""

[ -f "$SAFETY_DATA_PATH" ] || { echo "❌ 안전 데이터 없음: $SAFETY_DATA_PATH"; exit 1; }
python - "$SAFETY_DATA_PATH" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
ok = sum(1 for x in d if x.get('prompt') and x.get('llama3_output'))
print(f"  안전 데이터 확인: {len(d)}개 중 prompt+llama3_output 유효 {ok}개")
if ok == 0:
    sys.exit(1)
PYEOF

if [ "$USE_WANDB" = "1" ]; then
    python -c "import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null \
        || { echo "❌ W&B 미인증"; exit 1; }
    WANDB_ARG="--wandb_project $WANDB_PROJECT"
else
    WANDB_ARG="--no_wandb"
fi
[ "$UPLOAD_TO_HF" = "1" ] && { hf auth whoami >/dev/null 2>&1 || { echo "❌ HF 미로그인"; exit 1; }; }
[ "$HF_PRIVATE" = "1" ] && HF_PRIVATE_ARG="--private" || HF_PRIVATE_ARG=""
echo "✅ preflight 통과"
echo ""

SWEEP_START=$SECONDS

# ========================================================================
# Phase 1: basis (beavertails)
# ========================================================================
if [ -n "$PHASE1_BASIS_DIR_OVERRIDE" ]; then
    BASIS_DIR="$PHASE1_BASIS_DIR_OVERRIDE"
    [ -d "$BASIS_DIR" ] || { echo "❌ basis 경로 없음: $BASIS_DIR"; exit 1; }
    echo "✅ Phase 1 스킵 — 기존 basis 사용: $BASIS_DIR"
else
    echo "========================================================================"
    echo "PHASE 1: Basis (beavertails)"
    echo "========================================================================"
    P1_JSON="$PROFILE_DIR/phase1.json"
    P1_START=$SECONDS
    python train.py \
        --phase 1 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --safety_dataset circuit_breakers \
        --circuit_breakers_path "$SAFETY_DATA_PATH" \
        --circuit_breakers_samples_phase1 $SAFETY_SAMPLES \
        --batch_size $BATCH_SIZE \
        --layer_type "$LAYER_TYPE" \
        --target_layers $TARGET_LAYERS \
        --output_dir $BASE_OUTPUT_DIR \
        --log_dir $LOG_DIR \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed $SEED \
        --profile_json "$P1_JSON" \
        $WANDB_ARG --wandb_run_name "p1_beavertails_${TIMESTAMP}" \
        2>&1 | tee $LOG_DIR/btsweep_phase1_${TIMESTAMP}.log

    echo ""; echo "⏱  Phase 1: $(fmt_hms $((SECONDS - P1_START)))"; show_profile "$P1_JSON"

    P1_OUT=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase1_*" -type d -printf '%T@ %p\n' \
        | sort -rn | head -1 | cut -d' ' -f2-)
    BASIS_DIR="$P1_OUT/basis"
    [ -d "$BASIS_DIR" ] || { echo "❌ basis 생성 실패: $BASIS_DIR"; exit 1; }
    echo ""; echo "✅ Phase 1 완료: $BASIS_DIR"
fi
echo ""

# ========================================================================
# keep_ratio 스윕: Phase 2 → Phase 3 → 업로드 → 정리
# ========================================================================
for KR in $KEEP_RATIO_LIST; do
    KR_SAFE=$(echo "$KR" | tr -c 'a-zA-Z0-9_-' '_')
    echo "########################################################################"
    echo "# keep_ratio = $KR    (여유 디스크 $(free_gb) GB)"
    echo "########################################################################"

    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
        echo "❌ 디스크 여유 $(free_gb)GB < ${MIN_FREE_GB}GB — 중단합니다."
        echo "   완료된 keep_ratio: ${HF_UPLOADED[*]:-없음}"
        exit 1
    fi

    # ---------------- Phase 2 ----------------
    P2_JSON="$PROFILE_DIR/phase2_kr${KR_SAFE}.json"
    P2_START=$SECONDS
    python train.py \
        --phase 2 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --basis_dir "$BASIS_DIR" \
        --dataset_phase2 circuit_breakers \
        --circuit_breakers_path "$SAFETY_DATA_PATH" \
        --circuit_breakers_samples_phase2 $SAFETY_SAMPLES \
        --keep_ratio $KR \
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
        --profile_json "$P2_JSON" \
        $WANDB_ARG --wandb_run_name "p2_bt_kr${KR_SAFE}_${TIMESTAMP}" \
        2>&1 | tee $LOG_DIR/btsweep_phase2_kr${KR_SAFE}_${TIMESTAMP}.log

    echo ""; echo "⏱  Phase 2 (kr=$KR): $(fmt_hms $((SECONDS - P2_START)))"; show_profile "$P2_JSON"

    P2_OUT=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase2_*" -type d -printf '%T@ %p\n' \
        | sort -rn | head -1 | cut -d' ' -f2-)
    MASKS_DIR="$P2_OUT/checkpoints/masks"
    [ -d "$MASKS_DIR" ] || { echo "❌ 마스크 생성 실패: $MASKS_DIR"; exit 1; }

    # ---------------- Phase 3 ----------------
    P3_JSON="$PROFILE_DIR/phase3_kr${KR_SAFE}.json"
    P3_START=$SECONDS
    python train.py \
        --phase 3 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --basis_dir "$BASIS_DIR" \
        --masks_dir "$MASKS_DIR" \
        --phase3_dataset $PHASE3_DATASET \
        --gsm8k_samples $PHASE3_SAMPLES \
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
        --non_freeze \
        --profile_json "$P3_JSON" \
        $WANDB_ARG --wandb_run_name "p3_bt_kr${KR_SAFE}_lr${LEARNING_RATE}_${TIMESTAMP}" \
        2>&1 | tee $LOG_DIR/btsweep_phase3_kr${KR_SAFE}_${TIMESTAMP}.log

    echo ""; echo "⏱  Phase 3 (kr=$KR): $(fmt_hms $((SECONDS - P3_START)))"; show_profile "$P3_JSON"

    P3_OUT=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase3_non_freeze_*" -type d -printf '%T@ %p\n' \
        | sort -rn | head -1 | cut -d' ' -f2-)
    [ -d "$P3_OUT/final_model" ] || { echo "❌ final_model 없음: $P3_OUT"; exit 1; }

    # ---------------- 업로드 + 정리 ----------------
    if [ "$UPLOAD_TO_HF" = "1" ]; then
        HF_REPO="${HF_REPO_PREFIX}-kr${KR}_lr${LEARNING_RATE}"
        echo ""; echo "── HF 업로드: $HF_REPO ──"
        python scripts/upload_and_cleanup_phase3.py \
            --model_dir "$P3_OUT/final_model" \
            --repo_name "$HF_REPO" \
            --base_model "$PHASE0_MODEL" \
            --keep_ratio "$KR" \
            --learning_rate "$LEARNING_RATE" \
            --dataset "$PHASE3_DATASET (basis/mask: beavertails)" \
            --metadata_json "$P3_OUT/metadata.json" \
            $HF_PRIVATE_ARG \
            --delete_after_verify \
            2>&1 | tee -a $LOG_DIR/btsweep_upload_${TIMESTAMP}.log
        HF_UPLOADED+=("kr${KR}:$HF_REPO")
        echo "🔗 https://huggingface.co/$HF_REPO"
    fi

    if [ "$DELETE_MASKS_AFTER_PHASE3" = "1" ]; then
        echo "🧹 마스크 삭제: $P2_OUT ($(du -sh $P2_OUT | cut -f1))"
        rm -rf "$P2_OUT"
    fi

    echo ""
    echo "✅ keep_ratio=$KR 완료  (여유 디스크 $(free_gb) GB, 누적 $(fmt_hms $((SECONDS - SWEEP_START))))"
    echo ""
done

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "스윕 완료  총 $(fmt_hms $((SECONDS - SWEEP_START)))"
echo "========================================================================"
echo "Basis: $BASIS_DIR"
echo ""
echo "업로드된 모델:"
for e in "${HF_UPLOADED[@]}"; do
    echo "  [${e%%:*}]  https://huggingface.co/${e#*:}"
done
echo ""
echo "Profile JSON: $PROFILE_DIR/"
echo "여유 디스크: $(free_gb) GB"
