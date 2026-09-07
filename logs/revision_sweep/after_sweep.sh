#!/usr/bin/env bash
# sweep 종료 후 자동 gap-fill (네트워크 오류 등으로 빠진 평가 조합 재실행)
cd ~/Safety-WaRP-LLM
SP=$(awk '{print $2}' logs/revision_sweep/sweep.pid)
while kill -0 "$SP" 2>/dev/null; do sleep 120; done
echo "[after_sweep] sweep 종료 감지 $(date) → gap-fill 시작"
bash logs/revision_sweep/gapfill_eval.sh
echo "[after_sweep] gap-fill rc=$? $(date)"; echo done > logs/revision_sweep/gapfill.rc
