#!/usr/bin/env bash
# 평가 종료까지 대기 후 요약 출력
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
p=$(cat "$L/eval.pid")
while kill -0 "$p" 2>/dev/null; do sleep 60; done
echo "######## 평가 종료 $(date -Iseconds) ########"
grep -E '\[SUMMARY\]|Average ASR|EVAL COMPLETE|rc=' "$L/20_eval_row.log" | tail -20
echo "---- lm-eval ----"
grep -E 'exact_match|gsm8k|\|' "$L/20_eval_row.log" | tail -20
