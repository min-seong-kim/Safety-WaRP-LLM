#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  2026-09-23 밤 — 논문 Table 1 후보 A/B: base × GSM8K × lr 5e-5 (무인 실행)
#
#    A: Llama-2-7B base   (출발 kmseong/llama2_7b-base-CB_SSFT-lr3e-5)     GPU0
#    B: Llama-3.1-8B base (출발 kmseong/Llama-3.1-8B-base-SSFT_lr5e-5)     GPU1
#
#  셀: 동결 0/10/30/50% (원공간) + WSR-Tune 10%. 한 GPU 에서 순차(p00 → wsr → p10 → p30 → p50).
#  WSR basis / ρ=0.1 mask 는 오늘 MATH 실험에서 **같은 출발 모델**로 만든 것을 재사용(과제·lr 무관).
#  학습 실패 셀은 한 번 더 시도한다(.done 있는 셀은 건너뛴다).
#  평가: HarmBench 4공격 keyword ASR + lm-eval gsm8k 5-shot (flexible-extract), 같은 GPU 에서 순차.
#        두 모델의 평가는 flock 으로 직렬화 — models.yaml 동시 쓰기 방지.
#  업로드 없음.
#
#  사용: bash scripts/revision/_night_table1_gsm8k.sh      (pid: logs/table1_gsm8k_night.pid)
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
PY=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3/envs/hb/bin/python
LOCK="$REPO_DIR/logs/table1_gsm8k_eval.lock"
MAIN_LOG="$REPO_DIR/logs/table1_gsm8k_night.log"
echo $$ > "$REPO_DIR/logs/table1_gsm8k_night.pid"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

pipeline() {  # model gpu tag
  local M="$1" G="$2" T="$3"
  local W="$REPO_DIR/outputs/table1_math_${T}_base/wsr_p10"
  local R="$REPO_DIR/outputs/table1_gsm8k_${T}_base" D="$REPO_DIR/logs/table1_gsm8k_${T}_base"
  mkdir -p "$D"
  local common=(MODEL="$M" TASK=gsm8k GPU_A="$G" QUEUE_A="p00 wsr_p10 p10 p30 p50" QUEUE_B=""
                WSR_BASIS_DIR="$(cat "$W/BASIS_DIR")" WSR_MASKS_DIR="$(cat "$W/MASKS_DIR")")
  for attempt in 1 2; do
    log "[$T] 학습 시도 $attempt (GPU$G)"
    env "${common[@]}" STAGE=train bash scripts/revision/33_table1_math_8b_base.sh > "$D/train_attempt$attempt.log" 2>&1
    local ndone; ndone=$(ls "$R"/{p00,p10,p30,p50,wsr_p10}/.done 2>/dev/null | wc -l)
    log "[$T] 학습 시도 $attempt 종료 — 완료 셀 $ndone/5 $(grep -h -E '❌' "$D"/queue_gpu*.log 2>/dev/null | tail -2 | tr '\n' ' ')"
    [ "$ndone" -ge 5 ] && break
  done
  log "[$T] 평가 대기(lock)"
  (
    flock 9
    log "[$T] 평가 시작 (HarmBench + gsm8k, GPU$G)"
    env "${common[@]}" STAGE=eval EVAL_REFS="" HB_GPU="$G" LM_GPU="$G" PARALLEL=0 \
      bash scripts/revision/33_table1_math_8b_base.sh > "$D/eval_night.log" 2>&1
    log "[$T] 평가 종료 rc=$? (No module: $(grep -c 'No module named' "$D/eval_night.log"))"
  ) 9>"$LOCK"
}

log "════ 시작: A(7B, GPU0) · B(8B, GPU1) — base × GSM8K × lr 5e-5 ════"
pipeline llama2_7b_base 0 7b & PA=$!
sleep 30
pipeline llama31_8b_base 1 8b & PB=$!
wait $PA $PB

# ── 요약 ─────────────────────────────────────────────────────────────────────
log "════ 요약 ════"
for pre in llama2_7b-base llama3_1_8b-base; do
  keys=""; for c in p00 p10 p30 p50; do keys="$keys $pre-origspace-freeze-$c-gsm8k-lr5e-5"; done; keys="$keys $pre-wsr-tune-kr0.1-gsm8k-lr5e-5"
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
log "NIGHT_DONE"
