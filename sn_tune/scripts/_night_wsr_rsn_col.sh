#!/usr/bin/env bash
# 야간 무인 체인 (2026-09-22): 학습 오케스트레이터 종료 대기 → 미완료 셀이 있으면 13B 를 1×16 으로
# 한 번 재시작 → 평가(HarmBench harmbench conda · sys · keyword, GSM8K hb conda) → 결과표.
# 업로드 없음 (huggingface.co 차단). pid 파일만 본다 (pgrep 금지).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh >/dev/null 2>&1 || true
L=logs/wsr_rsn_col
log() { echo "[$(date '+%F %T')] $*"; }

wait_pid() { local p; p=$(cat "$1" 2>/dev/null) || return 0; while kill -0 "$p" 2>/dev/null; do sleep 60; done; }

all_done() {
    [[ -f outputs/wsr_rsn_col_p3/llama2_7b/k1200_200_u300_50/MODEL_DIR && \
       -f outputs/wsr_rsn_col_p3/llama2_13b/k1800_300_u300_50/MODEL_DIR ]]
}

log "1. 학습 오케스트레이터($(cat $L/orch.pid)) 종료 대기"
wait_pid "$L/orch.pid"
log "   종료. 7B MODEL_DIR: $(cat outputs/wsr_rsn_col_p3/llama2_7b/k1200_200_u300_50/MODEL_DIR 2>/dev/null || echo 없음)"
log "   13B MODEL_DIR: $(cat outputs/wsr_rsn_col_p3/llama2_13b/k1800_300_u300_50/MODEL_DIR 2>/dev/null || echo 없음)"

for attempt in 1 2; do
    all_done && break
    if (( attempt == 1 )); then
        log "2. 미완료 셀 있음 → 13B micro-batch 1×16 으로 재시작 (마커로 이어감)"
        MB="1"
    else
        log "2b. 아직 미완료 → 13B 1×16 + gradient checkpointing 그대로, 마지막 재시도"
        MB="1"
    fi
    for f in outputs/wsr_rsn_col_p3/*/*/rsn_tune*/phase3_* outputs/wsr_rsn_col_p3/*/*/gsm8k/phase3_*; do
        # final_model 이 없는 부분 실행 디렉토리는 치운다 (newest_subdir 오판 방지)
        [[ -d "$f" && ! -f "$f/final_model/config.json" ]] && { log "   partial 제거: $f"; rm -rf "$f"; }
    done
    TS=$(date +%Y%m%d_%H%M%S)
    MB_llama2_13b="$MB" nohup bash sn_tune/scripts/run_wsr_rsn_col_phase3.sh > "$L/orch_${TS}.log" 2>&1 &
    echo $! > "$L/orch.pid"
    log "   재시작 pid $(cat $L/orch.pid) → $L/orch_${TS}.log"
    wait_pid "$L/orch.pid"
done
all_done || log "   ⚠️ 여전히 미완료 셀이 있다. 있는 셀만 평가한다."

log "3. 평가 시작"
bash sn_tune/scripts/eval_wsr_rsn_col.sh > "$L/eval_$(date +%Y%m%d_%H%M%S).log" 2>&1
log "   평가 rc=$?"
cat outputs/wsr_rsn_col_p3/RESULTS.md 2>/dev/null
date -Iseconds > "$L/night.done"
log "NIGHT CHAIN COMPLETE"
