#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  LISA rho=1.0 재학습 + 평가 (2026-09-01 사용자 지시).
#  rho=0.0 셀 9개와 짝을 이루는 rho 민감도 ablation.
#
#  ⚠️ out_dir 은 rho 를 구분하지 않는다(리포명만 구분). 기존 rho0.0 의 .done 마커를
#     건드리지 않도록 OUT_ROOT 를 별도 디렉터리로 격리한다.
#  ⚠️ 이전 rho1.0 리포 9개는 2026-08-31 에 삭제되어 재사용 불가 → 전부 새로 학습.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; HB=/home/edgeai_lab/HarmBench
cd "$REPO"; export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"
TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/lisa_rho1_${TS}.log"; ln -sfn "$LOG" "$ULOG/lisa_rho1_latest.log"
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "선행 작업(평가·보충·재학습) 종료 대기"
while pgrep -f "supervise_eval\.sh|fill_missing_lmeval\.sh|retrain_two_cells\.sh" > /dev/null; do sleep 120; done
log "선행 종료 확인 — LISA rho=1.0 시작"

export LISA_RHO=1.0
export OUT_ROOT="$REPO/outputs/revision_lisa_rho1"
export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb BASE_BLOCKED_MODELS=""
export DEADLINE_HOURS="${DEADLINE_HOURS:-72}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
mkdir -p "$OUT_ROOT"

log "════ 학습 (9셀) ════"
for mk in llama2_7b llama2_13b llama32_3b llama31_8b qwen25_7b gemma2_9b; do
  log "── LISA ρ=1.0 · $mk"
  MODELS="$mk" METHODS=lisa bash "$HERE/run_all.sh"
done

log "════ 평가 ════"
REPOS=$(bash -c 'export LISA_RHO=1.0; source '"$HERE"'/common.sh >/dev/null 2>&1
for c in "llama2_7b gsm8k" "llama2_7b medqa" "llama2_7b arc" "llama2_7b agnews" \
         "llama2_13b gsm8k" "llama32_3b math" "llama31_8b math" "qwen25_7b gsm8k" "gemma2_9b gsm8k"; do
  set -- $c; hf_repo_id cb "$1" "$2" lisa; done')
cd "$HB"
i=0; arr=($REPOS)
while (( i < ${#arr[@]} )); do
  chunk=("${arr[@]:i:3}")
  log "── 평가 배치: ${chunk[*]}"
  RESUME=true VALIDATE_REPOS=1 ./run_all_eval.sh "${chunk[@]}" || log "   배치 실패(계속)"
  for r in "${chunk[@]}"; do
    d="$HOME/.cache/huggingface/hub/models--${r//\//--}"
    [ -d "$d" ] && rm -rf "$d" && log "   캐시 회수: $(basename "$d")"
  done
  i=$((i+3))
done
log "════ 완료 ════"
