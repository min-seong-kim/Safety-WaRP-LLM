#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  2026-08-31 사용자 결정 반영 — 일괄 평가 전에 남은 학습을 모두 끝낸다.
#
#   ① llama2_13b/gsm8k : seal · wsr_lora        (중단분)
#   ② LISA rho=0.0 재학습 8셀                    (rho1.0 리포는 삭제됨)
#   ③ WSR-LoRA α16 ablation : llama2_7b/gsm8k   (α32 유지 + ablation 방침)
#
#  ③ 은 WSR_LORA_ALPHA=16 이라 리포명이 ..._rho0.1_a16_lr3e-4 로 분리되어
#  기존 α32 셀을 덮어쓰지 않는다.
#
#    setsid nohup bash scripts/revision/finish_all_before_eval.sh > /dev/null 2>&1 &
#    tail -f logs/revision_unattended/finish_all_latest.log
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"

TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO_DIR/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/finish_all_${TS}.log"; ln -sfn "$LOG" "$ULOG/finish_all_latest.log"
exec > >(tee -a "$LOG") 2>&1
mkdir -p "$REPO_DIR/outputs/revision"

export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb
export BASE_BLOCKED_MODELS=""
export DEADLINE_HOURS="${DEADLINE_HOURS:-96}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

echo "════ 시작 ts=$TS pid=$$  python=$(command -v python) ════"

echo ""; echo "════ ① llama2_13b : seal · wsr_lora ════"
MODELS=llama2_13b METHODS="seal wsr_lora" bash "$HERE/run_all.sh"

echo ""; echo "════ ② LISA rho=0.0 재학습 ════"
for mk in llama2_7b llama32_3b llama31_8b qwen25_7b gemma2_9b; do
  echo "── LISA · $mk"
  MODELS="$mk" METHODS=lisa bash "$HERE/run_all.sh"
done

echo ""; echo "════ ③ WSR-LoRA α16 ablation (llama2_7b/gsm8k) ════"
WSR_LORA_ALPHA=16 MODELS=llama2_7b METHODS=wsr_lora TASKS=gsm8k bash "$HERE/run_all.sh"

echo ""; echo "════ 진행 현황 재생성 ════"
python "$HERE/gen_progress_md.py" --out "$REPO_DIR/REVISION_PROGRESS.md" || true
echo "════ 완료 $(date +%Y%m%d_%H%M%S) ════"
