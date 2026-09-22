#!/usr/bin/env bash
# ablation: 원공간 동결 10% 를 **복원 콜백 없이** (gradient hook 만) 학습한다.
#   (2026-09-21 사용자 지시 — 논문 Table 1(base) 의 옛 수치가 이 방식일 가능성 검증)
#
# ORIGSPACE_NO_RESTORE=1 이면 OriginalSpaceMaskRestoreCallback 을 달지 않는다.
# gradient 는 0 이 되지만 AdamW 의 weight decay / 모멘텀이 mask=1 위치를 계속 움직이므로
# **진짜 동결이 아니다**. 안전 weight 가 조금씩 깎여 ASR 이 올라갈 것으로 예상된다.
#
# ⚠️ 기존 p10(복원 콜백 있음, ASR 0.1578) 을 덮어쓰지 않도록 MODEL_TAG 와 OUT_ROOT 을 분리한다.
#    MODEL_TAG 는 리포명에만 쓰이고, 'base' 토큰이 유지되므로 chat-template 판정에 영향이 없다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

# 앞선 p05 작업(평가 포함)이 끝날 때까지 대기 — GPU 충돌 방지
PIDFILE=logs/base_p05.pid
if [ -f "$PIDFILE" ]; then
  TPID="$(cat "$PIDFILE")"
  if kill -0 "$TPID" 2>/dev/null; then
    echo "[대기] p05 작업(PID=$TPID) 종료 대기 $(date -Iseconds)"
    while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
  fi
fi

source scripts/revision/common.sh >/dev/null 2>&1
model_cfg llama2_7b_base >/dev/null

export ORIGSPACE_NO_RESTORE=1
OUT="$PWD/outputs/origspace_base_norestore/llama2_7b_base"
mkdir -p "$OUT" "$PWD/logs/origspace_base_norestore"

MODEL_TAG="llama2_7b-base-norestore" \
PHASE0_MODEL="$ALIGNED_CB" \
KEEP_RATIOS="0.1" \
PHASE3_DATASET=gsm8k \
LR="$FULL_LR" EPOCHS="$EPOCHS" BATCH_SIZE="$MB_FULL" \
GRAD_ACCUM="$(( EFFECTIVE_BATCH / MB_FULL ))" \
WARMUP_RATIO="$FULL_WARMUP_RATIO" LR_SCHEDULER="$FULL_SCHEDULER" \
WEIGHT_DECAY="$FULL_WEIGHT_DECAY" MAX_LENGTH="$MAX_LENGTH" SEED="$SEED" \
LAYER_TYPE="$LAYER_TYPES" TARGET_LAYERS="$TARGET_LAYERS" \
PHASE2_SAMPLES="$SAFETY_SAMPLES" \
OUT_ROOT="$OUT" \
CKPT_ROOT="$PWD/checkpoints/origspace_base_norestore" \
LOG_DIR="$PWD/logs/origspace_base_norestore" \
HF_NAMESPACE="$HF_NAMESPACE" PUSH_TO_HUB=1 \
  bash scripts/run_origspace_freeze_sweep.sh
echo "NORESTORE_TRAIN_RC=$?"

CELL="$OUT/p10"; NAME="llama2_7b-base-norestore-origspace-freeze-p10-gsm8k-lr${FULL_LR}"
if [ -f "$CELL/.done" ]; then
  MD="$(cat "$CELL/MODEL_DIR" 2>/dev/null)"
  if [ -n "$MD" ] && [ -d "$MD" ]; then
    mkdir -p outputs/eval_local; ln -sfn "$MD" "outputs/eval_local/$NAME"
    PREFETCH=0 REPOS_ONLY="$PWD/outputs/eval_local/$NAME" bash scripts/revision/32_eval_cells.sh
    echo "NORESTORE_EVAL_RC=$?"
  fi
else
  echo "❌ 학습 미완료"
fi
echo "NORESTORE_DONE $(date -Iseconds)"
