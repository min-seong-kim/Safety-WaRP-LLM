#!/usr/bin/env bash
# 배치 분류기 재실행(PID 인자)이 끝나면 GPU 가 비므로 13B 검증을 이어서 시작한다.
# 프로세스명 매칭(pgrep -f)은 이 박스에서 감시 스크립트 자신을 잡아 5시간을 날린 전력이
# 있으므로, PID 에 대한 kill -0 만 쓴다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
WAIT_PID="${1:?선행 PID 필요}"
echo "[chain] PID $WAIT_PID 대기 시작 $(date -Iseconds)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[chain] 선행 종료 확인 $(date -Iseconds)"
# vLLM 이 GPU 메모리를 완전히 반납할 때까지 여유를 둔다
sleep 60
echo "[chain] 13B 검증 시작 $(date -Iseconds)"
bash scripts/revision/_run_cls_eval_13b.sh
echo "[chain] CHAIN_13B_DONE rc=$? $(date -Iseconds)"
