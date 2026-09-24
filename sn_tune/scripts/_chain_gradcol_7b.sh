#!/usr/bin/env bash
# 2026-09-22 밤 체인 v3 (사용자 결정 "2번: 기울기 열 선택"):
#   1. Phase 2 arm C (p2.pid) 종료 대기 → 마스크 디렉토리 확인
#   2. sn arm 드라이버(705473) 종료 대기 (GPU 확보)
#   3. gradcol 7B 학습 [6]→[7]  (백그라운드) + 동시에 sn 셀 평가 (GPU util 0.4)
#   4. gradcol 7B 평가 → 요약
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh >/dev/null 2>&1 || true
L=logs/wsr_rsn_col; OUT=outputs/wsr_rsn_col_p3
PY="$(conda env list | awk '$1=="hb"{print $NF}')/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }

log "1. Phase 2 arm C 대기 (pid $(cat $L/p2.pid))"
wait_pid "$(cat $L/p2.pid)"
MASKS_C="$(find outputs/wsr_rsn_col_p3/_gradcol/llama2_7b/safety_rho0.1 -maxdepth 3 -type d -path '*phase2_armC_*/checkpoints/masks' | sort | tail -1)"
[[ -n "$MASKS_C" && -f "$MASKS_C/metadata.json" ]] || { log "⚠️ arm C 마스크가 없다 — $L/p2_armC_7b_safety_rho0.1.log 확인"; exit 1; }
log "   masks_C = $MASKS_C"; cat "$MASKS_C/budget_report.json" 2>/dev/null | head -12

log "2. sn arm 드라이버(705473) 종료 대기"
wait_pid 705473
SN_CELL="$OUT/llama2_7b/k1200_200_u300_50_sn"
log "   sn 셀 MODEL_DIR: $(cat $SN_CELL/MODEL_DIR 2>/dev/null || echo 없음)"

log "3. gradcol 7B 학습 시작 (백그라운드) + sn 셀 평가 (동시)"
CELL="$OUT/llama2_7b/gradcol_rho0.1_sn"
MASKS_C="$MASKS_C" CELL="$CELL" MODEL=llama2_7b nohup bash sn_tune/scripts/run_wsr_gradcol.sh > "$L/train_gradcol_7b_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo $! > "$L/gradcol.pid"
if [[ -f "$SN_CELL/MODEL_DIR" ]]; then
    sleep 240   # gradcol 이 모델을 올린 뒤 남는 메모리로 평가
    CELLS="$SN_CELL" GPU_UTIL_7B=0.40 bash sn_tune/scripts/eval_wsr_rsn_col.sh > "$L/eval_sn_$(date +%Y%m%d_%H%M%S).log" 2>&1
    log "   sn 평가 rc=$?"
fi
wait_pid "$(cat $L/gradcol.pid)"
log "   gradcol MODEL_DIR: $(cat $CELL/MODEL_DIR 2>/dev/null || echo 없음)"

log "4. gradcol 7B 평가"
[[ -f "$CELL/MODEL_DIR" ]] && { CELLS="$CELL" bash sn_tune/scripts/eval_wsr_rsn_col.sh > "$L/eval_gradcol_$(date +%Y%m%d_%H%M%S).log" 2>&1; log "   평가 rc=$?"; }
"$PY" sn_tune/summarize_wsr_rsn_col.py --out_root "$OUT" 2>/dev/null | tail -40
date -Iseconds > "$L/chain_gradcol_7b.done"
log "CHAIN GRADCOL 7B COMPLETE"
