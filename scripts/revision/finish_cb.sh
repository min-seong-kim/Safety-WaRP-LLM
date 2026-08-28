#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  BT 축 직전까지 — CB 축을 끝까지 밀고, 허브에 올리고, 요약 md 를 쓰고 push 한다.
#
#    setsid nohup bash scripts/revision/finish_cb.sh > /dev/null 2>&1 &
#    tail -f logs/revision_unattended/finish_cb_latest.log
#
#  단계:  ① 밀린 로컬 셀 업로드(디스크 회수)  ② CB 잔여 셀 학습→업로드
#         ③ REVISION_PROGRESS.md 생성        ④ git commit + push
#  BT 축은 건드리지 않는다(SAFETY_SETS=cb 고정).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/miniforge3/envs/hb/bin:$PATH"

TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO_DIR/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/finish_cb_${TS}.log"; ln -sfn "$LOG" "$ULOG/finish_cb_latest.log"
exec > >(tee -a "$LOG") 2>&1

DEADLINE_HOURS="${DEADLINE_HOURS:-30}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1
# 싼 모델부터 — 박스가 언제 죽을지 모르니 완성 셀 수를 최대화한다.
MODEL_ORDER="${MODEL_ORDER:-llama32_3b qwen25_7b llama2_13b}"

echo "════════════════════════════════════════════════════════════════"
echo " CB 마무리 실행  ts=$TS  pid=$$"
echo "   마감    : $(date -d "@$REVISION_DEADLINE_EPOCH" '+%m-%d %H:%M') (${DEADLINE_HOURS}h)"
echo "   모델순서: $MODEL_ORDER   (BT 축은 실행하지 않는다)"
echo "════════════════════════════════════════════════════════════════"

# ── ① 밀린 업로드 먼저: 디스크를 되찾아야 13B 셀이 돈다 ──
echo ""; echo "════ ① 밀린 로컬 셀 업로드 ════"
source "$HERE/common.sh"
for d in "$OUT_ROOT"/*/*/*/*/; do
  [ -f "$d/.done" ] && [ ! -f "$d/.uploaded" ] || continue
  c="${d%/}"; me=$(basename "$c"); tk=$(basename "$(dirname "$c")")
  mk=$(basename "$(dirname "$(dirname "$c")")"); sf=$(basename "$(dirname "$(dirname "$(dirname "$c")")")")
  repo="$(hf_repo_id "$sf" "$mk" "$tk" "$me")"
  echo "── $sf/$mk/$tk/$me → $repo"
  python "$HERE/upload_and_prune.py" --cell_dir "$c" --repo_id "$repo" --prune 2>&1 \
    | grep -E "검증 통과|검증 실패|GiB 회수|\[ERROR\]" | sed 's/^/     /'
done
echo "   디스크 여유: $(df -BG --output=avail / | tail -1 | tr -d ' ')"

# ── ② CB 잔여 셀 ──
echo ""; echo "════ ② CB 잔여 셀 학습 ════"
SAFETY_SETS=cb MODELS="$MODEL_ORDER" bash "$HERE/run_all.sh"; RC=$?
echo "   run_all rc=$RC"

# ── ③ 요약 md ──
echo ""; echo "════ ③ REVISION_PROGRESS.md ════"
python "$HERE/gen_progress_md.py" --out "$REPO_DIR/REVISION_PROGRESS.md" || echo "   [WARN] md 생성 실패"

# ── ④ git ──
echo ""; echo "════ ④ git ════"
# 존재하지 않는 경로를 섞으면 git add 가 통째로 실패해 커밋이 비어 버린다.
git add -A REVISION_PROGRESS.md logs/revision_unattended scripts/revision 2>/dev/null
n=$(find "$OUT_ROOT" -name .uploaded 2>/dev/null | wc -l)
if git diff --cached --quiet; then echo "   커밋할 변경 없음"; else
  git commit -q -m "revision: CB axis finish ${TS} — ${n} cells on the hub

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && echo "   $(git log --oneline -1)"
fi
timeout 300 git push origin HEAD 2>&1 | tail -2 || echo "   [WARN] push 실패 — 커밋은 로컬에 남아 있다"
echo ""; echo "════ 끝  $(date '+%Y-%m-%d %H:%M:%S')  ════"
