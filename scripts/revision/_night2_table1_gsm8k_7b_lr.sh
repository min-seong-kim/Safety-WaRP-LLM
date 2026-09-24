#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  2026-09-23 밤 (2) — Llama-2-7B base × GSM8K 의 lr 을 올려 본다: 7e-5 (GPU0) · 1e-4 (GPU1)
#
#  _night_table1_gsm8k.sh (A/B, lr 5e-5) 가 끝난 뒤 시작한다(그 pid 가 사라질 때까지 대기).
#  셀·절차는 A 와 같다: 동결 0/10/30/50% + WSR-Tune 10%, 한 GPU 순차, 실패 셀 1회 재시도,
#  HarmBench ASR + gsm8k 5-shot, 평가는 flock 직렬화. WSR basis/mask 재사용(과제·lr 무관).
#  드라이버는 33b_table1_lr.sh (33_ 의 사본 + LR 옵션). 결과: outputs/table1_gsm8k_7b_base_lr<LR>/
#  업로드 없음.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
PY=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3/envs/hb/bin/python
LOCK="$REPO_DIR/logs/table1_gsm8k_eval.lock"
MAIN_LOG="$REPO_DIR/logs/table1_gsm8k_night2.log"
echo $$ > "$REPO_DIR/logs/table1_gsm8k_night2.pid"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

PREV_PID="$(cat "$REPO_DIR/logs/table1_gsm8k_night.pid" 2>/dev/null)"
# 2026-09-24 01:55 수정: GPU0 은 A 가 끝나 비었으므로 lr 7e-5 는 즉시 시작, lr 1e-4(GPU1) 만 이전 작업 종료를 기다린다.
wait_prev() {
  log "이전 작업(pid $PREV_PID) 종료 대기 — GPU1 용"
  while [ -n "$PREV_PID" ] && kill -0 "$PREV_PID" 2>/dev/null; do sleep 60; done
  log "이전 작업 종료 확인"
}

W="$REPO_DIR/outputs/table1_math_7b_base/wsr_p10"
pipeline() {  # lr gpu
  local L="$1" G="$2"
  local R="$REPO_DIR/outputs/table1_gsm8k_7b_base_lr$L" D="$REPO_DIR/logs/table1_gsm8k_7b_base_lr$L"
  mkdir -p "$D"
  local common=(MODEL=llama2_7b_base TASK=gsm8k LR="$L" GPU_A="$G" QUEUE_A="p00 wsr_p10 p10 p30 p50" QUEUE_B=""
                WSR_BASIS_DIR="$(cat "$W/BASIS_DIR")" WSR_MASKS_DIR="$(cat "$W/MASKS_DIR")")
  for attempt in 1 2; do
    log "[lr$L] 학습 시도 $attempt (GPU$G)"
    env "${common[@]}" STAGE=train bash scripts/revision/33b_table1_lr.sh > "$D/train_attempt$attempt.log" 2>&1
    local ndone; ndone=$(ls "$R"/{p00,p10,p30,p50,wsr_p10}/.done 2>/dev/null | wc -l)
    log "[lr$L] 학습 시도 $attempt 종료 — 완료 셀 $ndone/5 $(grep -h -E '❌' "$D"/queue_gpu*.log 2>/dev/null | tail -2 | tr '\n' ' ')"
    [ "$ndone" -ge 5 ] && break
  done
  log "[lr$L] 평가 대기(lock)"
  (
    flock 9
    log "[lr$L] 평가 시작 (HarmBench + gsm8k, GPU$G)"
    env "${common[@]}" STAGE=eval EVAL_REFS="" HB_GPU="$G" LM_GPU="$G" PARALLEL=0 \
      bash scripts/revision/33b_table1_lr.sh > "$D/eval_night.log" 2>&1
    log "[lr$L] 평가 종료 rc=$? (No module: $(grep -c 'No module named' "$D/eval_night.log"))"
  ) 9>"$LOCK"
}

pipeline 7e-5 0 & PA=$!
( wait_prev; pipeline 1e-4 1 ) & PB=$!
wait $PA $PB

log "════ 요약 ════"
for L in 7e-5 1e-4; do
  pre=llama2_7b-base
  keys=""; for c in p00 p10 p30 p50; do keys="$keys $pre-origspace-freeze-$c-gsm8k-lr$L"; done; keys="$keys $pre-wsr-tune-kr0.1-gsm8k-lr$L"
  "$PY" scripts/revision/_asr_from_results.py $keys | tee -a "$MAIN_LOG"
  "$PY" - $keys <<'PYEOF' | tee -a "$MAIN_LOG"
import glob, json, os, sys
root = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness/eval_results"
for k in sys.argv[1:]:
    fs = sorted([f for f in glob.glob(f"{root}/**/results_*.json", recursive=True)
                 if os.path.dirname(f).rstrip("/").endswith(k)], key=os.path.getmtime)
    v = None
    for f in reversed(fs):
        r = json.load(open(f))["results"]
        if "gsm8k" in r:
            v = r["gsm8k"].get("exact_match,flexible-extract"); break
    print(f"{k}\tgsm8k(flexible)={v}")
PYEOF
done
log "NIGHT2_DONE"
