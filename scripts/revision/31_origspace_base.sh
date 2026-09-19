#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  실험 2 — 논문 Table 1 을 **base 모델 라인**으로 (2026-09-19, 사용자 지시)
#
#  Table 1 은 "원래 weight 공간에서 상위 ρ 를 얼리고 downstream FT 하면 안전성이
#  얼마나 보존되는가" 를 보는 표다. 개정본은 chat 라인으로 다시 만들었고
#  (scripts/run_origspace_freeze_sweep.sh, kmseong/llama2_7b-chat-origspace-freeze-*),
#  이 스크립트는 같은 것을 **base 라인**으로 만든다.
#
#  모델별로 필요한 행
#    (a) 원본 base 모델                     — 학습 없음, 평가만
#    (b) CB 로 안전정렬된 base (SSFT)       — 이미 허브에 있다, 평가만
#    (c) gsm8k FT, 동결 0%                  — 이미 허브에 있다, 평가만 (= plain full FT)
#    (d) gsm8k FT, 동결 10/20/30/40/50%     — ★ 이 스크립트가 새로 학습한다 (모델당 5셀)
#
#  동작점은 각 라인의 full-param lr 을 따른다 (common.sh):
#    llama2_7b_base  → lr 3e-5      llama31_8b_base → lr 1e-5
#  공통: 3ep · eff.batch 16 · max_len 1024 · seed 42 · bf16 · cosine · wd 0.01 · warmup 0.1
#
#  ⚠️ base 모델은 리포/디렉토리 이름에 chat/instruct/-it 토큰이 들어가면 안 된다.
#     러너들이 모델 참조 문자열로 chat template 사용 여부를 정하기 때문이다.
#
#  사용:
#    bash scripts/revision/31_origspace_base.sh            # 전체 (재개 가능)
#    DRY_RUN=1 bash scripts/revision/31_origspace_base.sh
#    BASE_MODELS=llama2_7b_base bash scripts/revision/31_origspace_base.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
source scripts/revision/common.sh

BASE_MODELS="${BASE_MODELS:-llama2_7b_base llama31_8b_base}"
RATIOS="${RATIOS:-0.1 0.2 0.3 0.4 0.5}"
TASK="${TASK:-gsm8k}"
EXP2_ROOT="${EXP2_ROOT:-$REPO_DIR/outputs/origspace_base}"
EXP2_LOG="${EXP2_LOG:-$REPO_DIR/logs/origspace_base}"
mkdir -p "$EXP2_ROOT" "$EXP2_LOG"

echo "════════════════════════════════════════════════════════════════"
echo "  실험 2: 원공간 동결 스윕 — base 라인"
echo "  모델   : $BASE_MODELS"
echo "  동결비율: $RATIOS   (ρ=0 은 기존 리포 재사용)"
echo "════════════════════════════════════════════════════════════════"

for mkey in $BASE_MODELS; do
  model_cfg "$mkey" || { echo "[!] 알 수 없는 모델: $mkey"; continue; }
  MTAG="$(hf_model_tag "$mkey")"       # llama2_7b-base / llama3_1_8b-base
  echo ""
  echo "▶▶ $mkey  (출발=$ALIGNED_CB, lr=$FULL_LR, 태그=$MTAG)"

  MODEL_TAG="$MTAG" \
  PHASE0_MODEL="$ALIGNED_CB" \
  KEEP_RATIOS="$RATIOS" \
  PHASE3_DATASET="$TASK" \
  LR="$FULL_LR" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$MB_FULL" \
  GRAD_ACCUM="$(( EFFECTIVE_BATCH / MB_FULL ))" \
  WARMUP_RATIO="$FULL_WARMUP_RATIO" \
  LR_SCHEDULER="$FULL_SCHEDULER" \
  WEIGHT_DECAY="$FULL_WEIGHT_DECAY" \
  MAX_LENGTH="$MAX_LENGTH" \
  SEED="$SEED" \
  LAYER_TYPE="$LAYER_TYPES" \
  TARGET_LAYERS="$TARGET_LAYERS" \
  PHASE2_SAMPLES="$SAFETY_SAMPLES" \
  OUT_ROOT="$EXP2_ROOT/$mkey" \
  CKPT_ROOT="$REPO_DIR/checkpoints/origspace_base_$mkey" \
  LOG_DIR="$EXP2_LOG/$mkey" \
  HF_NAMESPACE="$HF_NAMESPACE" \
  PUSH_TO_HUB="${EXP2_PUSH_TO_HUB:-1}" \
  DRY_RUN="${DRY_RUN:-0}" \
    bash scripts/run_origspace_freeze_sweep.sh
  echo "▶▶ $mkey 완료 (rc=$?)"
done

echo ""
echo "ALL_BASE_CELLS_FINISHED"
