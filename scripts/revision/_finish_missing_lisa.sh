#!/usr/bin/env bash
# 3×3 sweep 에서 마감 가드에 걸려 빠진 셀 하나를 보충한다.
#   Lisa r=16 alpha=16 lr=5e-4  (AsFT 가 마감을 넘겨 끝나는 바람에 시작되지 못했다)
# 그 뒤 실험 2 갭필까지 이어서 돈다 (원래 체인은 3x3 드라이버 종료를 기다리고 있다).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

# 3x3 드라이버(평가 포함)가 끝날 때까지 대기 — GPU 충돌 방지
PIDFILE=logs/qwen_3x3.pid
if [ -f "$PIDFILE" ]; then
  TPID="$(cat "$PIDFILE")"
  if kill -0 "$TPID" 2>/dev/null; then
    echo "[보충] 3x3 드라이버(PID=$TPID) 종료 대기 $(date -Iseconds)"
    while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
  fi
fi

OUT="$PWD/outputs/revision_qwen_3x3/r16a16_lr5e-4"
echo "[보충] Lisa r16a16 lr5e-4 학습 시작 $(date -Iseconds)"
env SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS="lisa" \
    LORA_R=16 LORA_ALPHA=16 LORA_LR_DEFAULT=5e-4 LISA_RHO=1.0 SKIP_PUBLISHED=0 \
    HF_REPO_SUFFIX="" \
    OUT_ROOT="$OUT" LOG_ROOT="$PWD/logs/revision_qwen_3x3/r16a16_lr5e-4" \
    PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0 \
    bash scripts/revision/20_lora_family.sh
echo "MISSING_LISA_RC=$?"

cell="$OUT/cb/qwen25_7b/gsm8k/lisa"
if [ -f "$cell/.done" ]; then
  md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  name="qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr5e-4"
  if [ -n "$md" ] && [ -d "$md" ]; then
    mkdir -p outputs/eval_local; ln -sfn "$md" "outputs/eval_local/$name"
    echo "[보충] 평가 $name"
    PREFETCH=0 REPOS_ONLY="$PWD/outputs/eval_local/$name" bash scripts/revision/32_eval_cells.sh
    echo "MISSING_LISA_EVAL_RC=$?"
  fi
fi
echo "MISSING_LISA_DONE $(date -Iseconds)"
