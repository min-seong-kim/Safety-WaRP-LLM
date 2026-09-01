#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  2026-08-31 재개분 — 이전 환경에서 못 끝낸 CB 축 7 셀을 학습→업로드한다.
#
#    ① gemma2_9b  : asft · safelora   (게이트 라이선스 승인되어 해제됨)
#    ② llama2_13b : lisa · seal · safelora · salora · wsr_lora
#                   (wsr_lora 때문에 stage 02 에서 Phase1 basis 를 먼저 만든다)
#
#    setsid nohup bash scripts/revision/finish_gemma_13b.sh > /dev/null 2>&1 &
#    tail -f logs/revision_unattended/finish_gemma_13b_latest.log
#
#  run_all.sh 를 MODELS/METHODS 로 좁혀 두 번 호출한다. 범위를 좁히므로
#  이미 허브에 올라간 62 셀은 대상에 들어오지 않는다(.done 마커가 없어도 안전).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"

TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO_DIR/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/finish_gemma_13b_${TS}.log"; ln -sfn "$LOG" "$ULOG/finish_gemma_13b_latest.log"
exec > >(tee -a "$LOG") 2>&1

# df 가 실패하면 여유 0GB 로 읽혀 모든 셀이 (disk) 로 skip 된다.
mkdir -p "$REPO_DIR/outputs/revision"

export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb
export DEADLINE_HOURS="${DEADLINE_HOURS:-72}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

echo "════════════════════════════════════════════════════════════════"
echo " 재개 실행  ts=$TS  pid=$$"
echo "   python : $(command -v python)"
echo "   마감   : $(date -d "@$REVISION_DEADLINE_EPOCH" '+%m-%d %H:%M') (${DEADLINE_HOURS}h)"
echo "════════════════════════════════════════════════════════════════"

echo ""; echo "════ ① gemma2_9b : asft · safelora ════"
BASE_BLOCKED_MODELS="" MODELS=gemma2_9b METHODS="asft safelora" \
  bash "$HERE/run_all.sh"

echo ""; echo "════ ② llama2_13b : lisa · seal · safelora · salora · wsr_lora ════"
MODELS=llama2_13b METHODS="lisa seal safelora salora wsr_lora" \
  bash "$HERE/run_all.sh"

echo ""; echo "════ ③ 진행 현황 재생성 ════"
python "$HERE/gen_progress_md.py" --out "$REPO_DIR/REVISION_PROGRESS.md"

echo ""; echo "════ 완료  ts=$TS → $(date +%Y%m%d_%H%M%S) ════"
