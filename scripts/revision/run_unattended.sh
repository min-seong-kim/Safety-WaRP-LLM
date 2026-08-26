#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  무인 실행 래퍼 — 세션이 끊겨도 계속 돌고, 끝나면 git commit/push 까지 한다.
#
#  띄우는 법 (셸이 닫혀도 살아남는다):
#    cd /home/edgeai_lab/Safety-WaRP-LLM
#    conda activate hb
#    setsid nohup bash scripts/revision/run_unattended.sh > /dev/null 2>&1 &
#    tail -f logs/revision_unattended/latest.log      # 진행 확인
#
#  ⚠️ 이 스크립트는 사람이 지켜보지 않는다는 전제로 만들었다:
#   · DEADLINE_HOURS 를 넘기면 **새 셀을 시작하지 않고** 정리 후 종료한다.
#     (실행 중이던 셀은 끝까지 간다. 박스가 강제 종료되기 전에 빠져나오기 위한 장치.)
#   · 한 셀이 실패해도 다음으로 넘어간다(CONTINUE_ON_ERROR=1).
#   · 학습이 끝난 셀은 즉시 HF 에 올라가므로, 박스가 죽어도 **결과는 허브에 남는다.**
#   · 마지막에 git commit/push 를 시도한다. 자격증명이 없으면 커밋만 하고 경고를 남긴다.
#
#  주요 환경변수
#    DEADLINE_HOURS   새 셀을 시작하지 않을 시각까지의 시간 (기본 40)
#    MODELS           기본은 싼 모델부터 (3B → 7B → 8B → qwen → gemma → 13B)
#    PUSH_TO_HUB      기본 1
#    GIT_PUSH         기본 1 (0 이면 커밋만)
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
cd "$REPO_DIR"

TS=$(date +%Y%m%d_%H%M%S)
ULOG_DIR="$REPO_DIR/logs/revision_unattended"
mkdir -p "$ULOG_DIR"
LOG="$ULOG_DIR/run_${TS}.log"
ln -sfn "$LOG" "$ULOG_DIR/latest.log"
exec > >(tee -a "$LOG") 2>&1

DEADLINE_HOURS="${DEADLINE_HOURS:-40}"
GIT_PUSH="${GIT_PUSH:-1}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
export CONTINUE_ON_ERROR=1
# ── 우선순위 ────────────────────────────────────────────────────────────────
#  전체 116셀은 학습만 ~85h, 업로드까지 하면 100~114h (4~5일) 걸린다.
#  마감 전에 잘릴 수밖에 없으므로 **완성된 그룹이 최대한 많이 남도록** 순서를 잡는다.
#    PASS 1 = CB 축 (논문 Table 2/4 확장 — 리뷰어 "베이스라인 부족" 에 직접 답하는 부분)
#    PASS 2 = BT 축 (안전 데이터 출처 robustness — rebuttal 에 이미 GSM8K 한 셀은 있다)
#  각 pass 안에서는 싼 모델부터. 13B 는 비싸고 디스크도 빠듯해 맨 뒤.
#  추정치(B200, rebuttal 실측 39분/7b-gsm8k-fullft 기준):
#    cb: 3b/math 2.8h · 7b/arc 0.6h · 7b/gsm8k 4.3h · qwen 4.3h · 7b/agnews 5.5h
#        · gemma 5.8h · 8b/math 7.2h · 13b 8.2h · 7b/medqa 12.1h      = 51h
#    bt: arc 1.1h · agnews 5.5h · gsm8k 7.3h · medqa 20.7h            = 35h
MODEL_ORDER="${MODEL_ORDER:-llama32_3b llama2_7b qwen25_7b gemma2_9b llama31_8b llama2_13b}"
PASS_ORDER="${PASS_ORDER:-cb bt}"

DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
export REVISION_DEADLINE_EPOCH="$DEADLINE_EPOCH"

echo "════════════════════════════════════════════════════════════════"
echo " 무인 실행 시작  ts=$TS"
echo "   PID          : $$"
echo "   로그         : $LOG"
echo "   마감         : $(date -d "@$DEADLINE_EPOCH" '+%Y-%m-%d %H:%M:%S') (${DEADLINE_HOURS}h 뒤 · 이후 새 셀 시작 안 함)"
echo "   순서         : [$PASS_ORDER] × [$MODEL_ORDER]"
echo "   HF 업로드    : $PUSH_TO_HUB    git push: $GIT_PUSH"
echo "════════════════════════════════════════════════════════════════"

# ── 사전 점검: 여기서 걸리면 며칠을 날린다 ──
fail=0
command -v python >/dev/null || { echo "[FATAL] python 없음 — conda activate hb 했는가?"; fail=1; }
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { echo "[FATAL] CUDA 사용 불가"; fail=1; }
python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1 \
  || { echo "[FATAL] HF 토큰 없음 — 업로드가 전부 실패한다. hf auth login 먼저."; fail=1; }
if [[ "$GIT_PUSH" == "1" ]]; then
  git ls-remote origin HEAD >/dev/null 2>&1 || echo "[WARN] git remote 읽기 실패"
fi
(( fail )) && { echo "사전 점검 실패 — 종료"; exit 1; }
echo "사전 점검 통과"

# ── 실험: 안전축 pass 를 나눠 돈다 (CB 를 먼저 끝내기 위해) ──
START=$(date +%s)
RC=0
for pass in $PASS_ORDER; do
  if deadline_passed 2>/dev/null || (( $(date +%s) >= DEADLINE_EPOCH )); then
    echo ""; echo "════ 마감 초과 — '$pass' pass 를 시작하지 않는다 ════"; break
  fi
  echo ""
  echo "████████████████████████████████████████████████████████████████"
  echo "  PASS: $pass 축     ($(date '+%m-%d %H:%M:%S'))"
  echo "████████████████████████████████████████████████████████████████"
  SAFETY_SETS="$pass" MODELS="$MODEL_ORDER" bash "$HERE/run_all.sh" || RC=$?
done
ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "════ 실험 종료 (rc=$RC, ${ELAPSED}s = $(( ELAPSED / 3600 ))h$(( ELAPSED % 3600 / 60 ))m) ════"

# ── 결과 요약 ──
echo ""
echo "════ 완료된 셀 ════"
find "${OUT_ROOT:-$REPO_DIR/outputs/revision}" -name .uploaded 2>/dev/null \
  | sort | while read -r f; do echo "  $(cat "$f")"; done | tee "$ULOG_DIR/uploaded_${TS}.txt"
n_up=$(wc -l < "$ULOG_DIR/uploaded_${TS}.txt" 2>/dev/null || echo 0)
echo "  → ${n_up}개 업로드 완료"

# ── git commit / push ──
echo ""
echo "════ git ════"
git add -A logs/revision_unattended scripts/revision 2>/dev/null
# 산출물(가중치)은 .gitignore 대상이라 들어가지 않는다. 로그와 업로드 목록만 남긴다.
if git diff --cached --quiet; then
  echo "  커밋할 변경 없음"
else
  git commit -q -m "revision: unattended run ${TS} — ${n_up} cells uploaded

$(head -40 "$ULOG_DIR/uploaded_${TS}.txt" 2>/dev/null)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    && echo "  커밋 완료: $(git log --oneline -1)"
fi

if [[ "$GIT_PUSH" == "1" ]]; then
  if timeout 300 git push origin HEAD 2>&1 | tail -3; then
    echo "  push 완료"
  else
    echo "  [WARN] push 실패 — 커밋은 로컬에 남아 있다."
    echo "         자격증명 설정 후  git push origin main  을 수동으로 실행할 것."
  fi
fi

echo ""
echo "════ 끝  $(date '+%Y-%m-%d %H:%M:%S') ════"
exit "$RC"
