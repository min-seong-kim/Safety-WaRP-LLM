#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  WSR-LoRA α=16 (scaling 1.0) 확장 ablation — 2026-09-01 사용자 지시.
#  llama2_7b/gsm8k 는 이미 완료(JB 0.1277 vs α32 0.2295) → 나머지 5개 모델로 확장해
#  6개 모델 전체의 "업데이트 예산 대 안전성" 곡선을 만든다.
#
#  ⚠️ out_dir 이 alpha 를 구분하지 않는다(리포명만 _a16 으로 갈림) → OUT_ROOT 격리.
#  ⚠️ 각 모델의 Phase 1 basis 가 필요하다(사용 후 자동 회수돼 전부 재생성).
#     stage 02 가 셀 실행 전에 자동으로 만든다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; HB=/home/edgeai_lab/HarmBench
cd "$REPO"; export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"
TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/wsrlora_a16_${TS}.log"; ln -sfn "$LOG" "$ULOG/wsrlora_a16_latest.log"
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "선행 작업(LISA rho1.0 등) 종료 대기"
while pgrep -f "lisa_rho1_ablation\.sh|supervise_eval\.sh|fill_missing_lmeval\.sh|retrain_two_cells\.sh" > /dev/null; do sleep 120; done
log "선행 종료 확인 — WSR-LoRA α=16 시작"

export WSR_LORA_ALPHA=16
export OUT_ROOT="$REPO/outputs/revision_wsrlora_a16"
export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb BASE_BLOCKED_MODELS=""
export DEADLINE_HOURS="${DEADLINE_HOURS:-96}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
mkdir -p "$OUT_ROOT"

log "════ 학습 (5셀 · 각 모델 basis 선행 생성) ════"
for mk in llama32_3b llama31_8b qwen25_7b gemma2_9b llama2_13b; do
  log "── WSR-LoRA α=16 · $mk   (여유 $(df -BG --output=avail / | tail -1 | tr -dc '0-9')G)"
  MODELS="$mk" METHODS=wsr_lora bash "$HERE/run_all.sh"
done

log "════ 평가 ════"
REPOS=$(bash -c 'export WSR_LORA_ALPHA=16; source '"$HERE"'/common.sh >/dev/null 2>&1
for c in "llama2_13b gsm8k" "llama32_3b math" "llama31_8b math" "qwen25_7b gsm8k" "gemma2_9b gsm8k"; do
  set -- $c; hf_repo_id cb "$1" "$2" wsr_lora; done')
cd "$HB"
arr=($REPOS); i=0
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
