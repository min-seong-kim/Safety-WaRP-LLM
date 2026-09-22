#!/usr/bin/env bash
# 다음 '사건'(검출 완료 / 스테이지 완료 / 실패)까지 대기 후 요약. 최대 대기 초는 $1.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
MAX=${1:-2400}
snap() { ls "$L/stages" 2>/dev/null | sort | tr '\n' ' '; echo -n "|"; ls sn_tune/output_neurons_warp_row/*/ sn_tune/output_neurons_warp/*/ 2>/dev/null | sort | tr '\n' ' '; }
BASE="$(snap)"
for ((i=0; i<MAX/15; i++)); do
  [[ "$(snap)" != "$BASE" ]] && break
  kill -0 "$(cat $L/orch.pid)" 2>/dev/null || break
  sleep 15
done
echo "######## $(date +%H:%M:%S) 상태 ########"
echo "-- 오케스트레이터: $(kill -0 "$(cat $L/orch.pid)" 2>/dev/null && echo 실행중 || echo 종료)"
echo "-- 스테이지 마커:"; ls -la "$L/stages" 2>/dev/null | tail -n +4
echo "-- 최근 오케스트레이터 로그:"; tail -6 "$L/00_orchestrator.log"
for f in 06_row_detect_13b 07_col_detect_7b 08_col_detect_13b 10_col_train_7b 11_col_train_13b; do
  [[ -f "$L/$f.log" ]] || continue
  echo "-- $f:"
  grep -E 'Total (safety|utility) neurons|PARAMETER percentage|neuron percentage|critical 로 남은|\[RUN \]|Traceback|Error|OutOfMemory|EXIT rc' "$L/$f.log" | tail -4
  p=$(grep -oE '[0-9]+/(4994|1000|[0-9]+)' "$L/$f.log" | tail -1); [[ -n "$p" ]] && echo "   진행: $p"
done
echo "-- GPU:"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
