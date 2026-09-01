#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  순서 변경(2026-09-01 사용자 지시): 학습을 먼저 몰아서 끝내고, 평가는 마지막에 일괄.
#
#   ① LISA rho=1.0 학습 9셀   — 이미 실행 중인 lisa_rho1_ablation.sh 가 담당.
#      9셀 업로드가 끝나면 그 스크립트를 **평가 진입 전에** 중단시킨다
#      (실행 중인 bash 파일은 편집이 위험하므로 편집 대신 종료로 처리).
#   ② WSR-LoRA α=16 학습 5셀  — basis 를 모델마다 새로 만든다.
#   ③ 전체 14셀 일괄 평가      — ASR + downstream.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; HB=/home/edgeai_lab/HarmBench
cd "$REPO"; export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"
TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/train_then_eval_${TS}.log"; ln -sfn "$LOG" "$ULOG/train_then_eval_latest.log"
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
freeg(){ df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

LISA_ROOT="$REPO/outputs/revision_lisa_rho1"
A16_ROOT="$REPO/outputs/revision_wsrlora_a16"

# ── ① LISA rho=1.0 학습 9셀 완료 대기 → 평가 진입 전 중단 ──────────────────
log "════ ① LISA rho=1.0 학습 완료 대기 (9셀) ════"
while :; do
  n=$(find "$LISA_ROOT" -name '.uploaded' 2>/dev/null | wc -l)
  pid=$(ps -eo pid,cmd | grep "revision/lisa_rho1_ablation.sh" | grep -v grep | awk '{print $1}' | head -1)
  if (( n >= 9 )); then
    log "9/9 업로드 완료 — 평가 진입 전에 중단시킨다"
    [ -n "$pid" ] && { pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; sleep 5; }
    # 혹시 평가가 이미 떴으면 정리
    for p in $(ps -eo pid,cmd | grep -E "run_all_eval.sh|harmbench_eval.sh" | grep -v grep | awk '{print $1}'); do kill "$p" 2>/dev/null; done
    break
  fi
  if [ -z "$pid" ]; then log "lisa 스크립트가 사라짐 (완료 $n/9) — 다음 단계로"; break; fi
  log "  학습 진행 $n/9 — 3분 후 재확인"
  sleep 180
done

# ── ② WSR-LoRA α=16 학습 5셀 ───────────────────────────────────────────────
log "════ ② WSR-LoRA α=16 학습 (5셀) ════"
export WSR_LORA_ALPHA=16
export OUT_ROOT="$A16_ROOT"
export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb BASE_BLOCKED_MODELS=""
export DEADLINE_HOURS="${DEADLINE_HOURS:-96}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
mkdir -p "$OUT_ROOT"
for mk in llama32_3b llama31_8b qwen25_7b gemma2_9b llama2_13b; do
  log "── α16 · $mk  (여유 $(freeg)G)"
  MODELS="$mk" METHODS=wsr_lora bash "$HERE/run_all.sh"
done

# ── ③ 전체 14셀 일괄 평가 ──────────────────────────────────────────────────
log "════ ③ 일괄 평가 (LISA rho1.0 9셀 + WSR-LoRA α16 5셀) ════"
REPOS=$(bash -c '
source '"$HERE"'/common.sh >/dev/null 2>&1
export LISA_RHO=1.0
for c in "llama2_7b gsm8k" "llama2_7b medqa" "llama2_7b arc" "llama2_7b agnews" \
         "llama2_13b gsm8k" "llama32_3b math" "llama31_8b math" "qwen25_7b gsm8k" "gemma2_9b gsm8k"; do
  set -- $c; hf_repo_id cb "$1" "$2" lisa; done
export WSR_LORA_ALPHA=16
for c in "llama2_13b gsm8k" "llama32_3b math" "llama31_8b math" "qwen25_7b gsm8k" "gemma2_9b gsm8k"; do
  set -- $c; hf_repo_id cb "$1" "$2" wsr_lora; done')
cd "$HB"
arr=($REPOS); log "평가 대상 ${#arr[@]}개"
i=0
while (( i < ${#arr[@]} )); do
  chunk=("${arr[@]:i:3}")
  log "── 배치: ${chunk[*]}  (여유 $(freeg)G)"
  for try in 1 2; do
    RESUME=true VALIDATE_REPOS=1 ./run_all_eval.sh "${chunk[@]}" && break
    log "   시도 $try 실패 — 재시도"
  done
  for r in "${chunk[@]}"; do
    d="$HOME/.cache/huggingface/hub/models--${r//\//--}"
    [ -d "$d" ] && rm -rf "$d" && log "   캐시 회수: $(basename "$d")"
  done
  i=$((i+3))
done
log "════ 전체 완료 ════"
