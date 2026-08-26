#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  본 실행이 끝나기를 기다렸다가, 빠진 셀만 채우고 push 한다.
#
#  왜 필요한가
#  ─────────
#  실행 도중 구현을 교체하거나(예: SaLoRA → salora/salora_lora.py) 셀 하나가 실패하면
#  그 셀에는 `.done` 이 없다. run_all.sh 는 `.done` 기반이라 다시 돌리면 **빠진 것만**
#  채운다. 이 스크립트는 본 러너와 GPU 를 다투지 않도록 **끝날 때까지 기다렸다가** 그걸 한다.
#
#  띄우는 법:
#    setsid nohup bash scripts/revision/gapfill_after.sh <본러너PID> > /dev/null 2>&1 &
#
#  환경변수: WAIT_PID(인자 대신), DEADLINE_HOURS(기본 6 — 본 실행이 끝난 뒤부터 셈)
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
cd "$REPO_DIR"

WAIT_PID="${1:-${WAIT_PID:-}}"
DEADLINE_HOURS="${DEADLINE_HOURS:-6}"
TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO_DIR/logs/revision_unattended"
mkdir -p "$ULOG"
LOG="$ULOG/gapfill_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "════ gap-fill 대기 시작  ts=$TS ════"
echo "  기다리는 PID : ${WAIT_PID:-<없음 — 즉시 진행>}"
echo "  로그         : $LOG"

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "  본 러너(PID $WAIT_PID) 종료 확인 — $(date '+%Y-%m-%d %H:%M:%S')"
fi
# GPU 가 완전히 비워질 때까지 잠깐 더
sleep 60

echo ""
echo "════ 빠진 셀 확인 ════"
export PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
export CONTINUE_ON_ERROR=1
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
PLAN_ONLY=1 bash "$HERE/run_all.sh" 2>&1 | grep -E "학습 셀:|디스크:"

echo ""
echo "════ gap-fill 실행 (완료된 셀은 자동으로 건너뛴다) ════"
# CB 먼저, 그다음 BT — 본 실행과 같은 우선순위.
for pass in ${PASS_ORDER:-cb bt}; do
  (( $(date +%s) >= REVISION_DEADLINE_EPOCH )) && { echo "마감 초과 — '$pass' 생략"; break; }
  echo "── PASS $pass"
  SAFETY_SETS="$pass" bash "$HERE/run_all.sh" || true
done

echo ""
echo "════ 결과 ════"
find "${OUT_ROOT:-$REPO_DIR/outputs/revision}" -name .uploaded 2>/dev/null \
  | sort | while read -r f; do echo "  $(cat "$f")"; done | tee "$ULOG/uploaded_gapfill_${TS}.txt"
n=$(wc -l < "$ULOG/uploaded_gapfill_${TS}.txt" 2>/dev/null || echo 0)
echo "  → 누적 ${n}개 업로드"

git add -A logs/revision_unattended scripts/revision 2>/dev/null
if ! git diff --cached --quiet; then
  git commit -q -m "revision: gap-fill ${TS} — ${n} cells uploaded (cumulative)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && echo "  커밋: $(git log --oneline -1)"
fi
timeout 300 git push origin HEAD 2>&1 | tail -2 || echo "  [WARN] push 실패 — 커밋은 로컬에 남아 있다"
echo "════ 끝  $(date '+%Y-%m-%d %H:%M:%S') ════"
