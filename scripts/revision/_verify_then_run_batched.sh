#!/usr/bin/env bash
# 검증(wildguard·beaver)이 끝나기를 기다렸다가, 불일치가 0건이면 전체 재실행한다.
# 불일치가 있으면 재실행하지 않고 멈춘다 — 잘못된 채점 로직으로 44×3 을 돌리면 안 된다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
LOG=$HOME/HarmBench/_verify_rest.log

echo "[chain] 검증 완료 대기 $(date -Iseconds)"
until grep -q "VERIFY_ALL_DONE" "$LOG" 2>/dev/null || grep -qE "^Traceback" "$LOG" 2>/dev/null; do sleep 20; done

if grep -qE "^Traceback" "$LOG" 2>/dev/null; then
  echo "❌ 검증 중 예외 발생 — 재실행하지 않는다"; grep -A5 "^Traceback" "$LOG" | head -8; exit 1
fi

# "대조 불일치 N건" 을 모두 모아 N 이 전부 0 인지 본다 (harmbench 것은 별도 로그)
BAD=$( { grep -ho "대조 불일치 [0-9]*건" "$LOG" $HOME/HarmBench/_verify_harmbench.log 2>/dev/null; } \
       | grep -o "[0-9]*" | awk '{s+=$1} END {print s+0}' )
CNT=$( { grep -c "대조 불일치" "$LOG" $HOME/HarmBench/_verify_harmbench.log 2>/dev/null | cut -d: -f2; } | awk '{s+=$1} END {print s+0}' )
echo "[chain] 검증 요약: 분류기 ${CNT}종 대조, 불일치 합계 ${BAD}건"
grep -h "대조 불일치" "$LOG" $HOME/HarmBench/_verify_harmbench.log 2>/dev/null

if [ "$BAD" != "0" ] || [ "$CNT" -lt 3 ]; then
  echo "❌ 불일치가 있거나 3종 검증이 다 끝나지 않았다 — 재실행 중단"; exit 1
fi

echo "[chain] ✅ 3종 모두 일치 — 전체 재실행 시작 $(date -Iseconds)"
bash scripts/revision/_run_cls_eval_stage2_batched.sh
echo "CHAIN_DONE rc=$?  $(date -Iseconds)"
