#!/usr/bin/env bash
# nosys 확인 작업(PID 인자)이 끝나면 13B 분류기 채점을 시작한다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
W="${1:?선행 PID 필요}"
echo "[chain] PID $W 대기 $(date -Iseconds)"
while kill -0 "$W" 2>/dev/null; do sleep 30; done
echo "[chain] 선행 종료 $(date -Iseconds)"; sleep 45
bash scripts/revision/_run_cls_13b_batched.sh
echo "[chain] CHAIN_CLS13B_DONE rc=$? $(date -Iseconds)"
