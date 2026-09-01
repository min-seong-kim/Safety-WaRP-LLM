#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  LISA 전 셀 rho=0.0 재수행 (2026-08-31 사용자 결정).
#
#  rebuttal 모델 kmseong/llama2_7b-chat-gsm8k-lisa-...-rho0-alt 가 rho 0.0 이었고
#  (finetune_config.json 로 확인), revision 초기값 1.0 과 어긋나 있었다.
#  common.sh 의 LISA_RHO 를 0.0 으로 바꿨으므로 리포명이 rho0.0 으로 바뀐다.
#  기존 rho1.0 리포 9개는 이미 삭제했다.
#
#  순서: 사용자 지정대로 llama2_7b/gsm8k 부터. llama2_13b 는 선행 드라이버
#  (finish_gemma_13b.sh) 가 이미 처리하므로 여기서는 .done 으로 자동 skip 된다.
#
#    setsid nohup bash scripts/revision/rerun_lisa_rho0.sh > /dev/null 2>&1 &
#    tail -f logs/revision_unattended/rerun_lisa_latest.log
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"

TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO_DIR/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/rerun_lisa_${TS}.log"; ln -sfn "$LOG" "$ULOG/rerun_lisa_latest.log"
exec > >(tee -a "$LOG") 2>&1

mkdir -p "$REPO_DIR/outputs/revision"

# 선행 드라이버가 아직 돌고 있으면 끝날 때까지 기다린다 (단일 GPU).
while pgrep -f "finish_gemma_13b.sh" > /dev/null; do
  echo "[wait] finish_gemma_13b.sh 실행 중 — 60s 후 재확인  $(date '+%H:%M:%S')"
  sleep 60
done
echo "[wait] 선행 드라이버 종료 확인 — LISA 재수행 시작"

export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb METHODS=lisa
export BASE_BLOCKED_MODELS=""          # gemma 라이선스 승인됨
export DEADLINE_HOURS="${DEADLINE_HOURS:-96}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

echo "════════════════════════════════════════════════════════════════"
echo " LISA rho=0.0 재수행  ts=$TS  pid=$$"
echo "   LISA_RHO : $(bash -c 'source scripts/revision/common.sh >/dev/null 2>&1; echo $LISA_RHO')"
echo "════════════════════════════════════════════════════════════════"

# 사용자 지정 순서: llama2_7b 먼저, 그 다음 나머지.
for mk in llama2_7b llama32_3b llama31_8b qwen25_7b gemma2_9b llama2_13b; do
  echo ""; echo "════ LISA · $mk ════"
  MODELS="$mk" bash "$HERE/run_all.sh"
done

echo ""; echo "════ 진행 현황 재생성 ════"
python "$HERE/gen_progress_md.py" --out "$REPO_DIR/REVISION_PROGRESS.md"
echo "════ 완료  $(date +%Y%m%d_%H%M%S) ════"
