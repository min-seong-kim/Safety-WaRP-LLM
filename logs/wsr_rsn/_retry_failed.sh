#!/usr/bin/env bash
# 실패한 열 학습 두 단계 재시도. 현재 오케스트레이터(행 학습)가 끝나면 시작한다.
#   col_train_7b  : [5] 는 이미 성공(.tune.done) → [6] GSM8K FT 만 다시. cache_dir 수정 반영됨.
#   col_train_13b : [5] 에서 OOM → batch 1 × grad_accum 16 (유효 16 유지) + expandable_segments
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
B7=$(cat outputs/wsr_sn_tune/llama2_7b/BASIS_DIR)
B13=$(cat outputs/wsr_sn_tune/llama2_13b/BASIS_DIR)

# 진행 중인 오케스트레이터 종료 대기 (pgrep 금지 — pid 파일만 본다)
if [[ -f "$L/orch.pid" ]]; then
  p=$(cat "$L/orch.pid")
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
  echo "[wait] 오케스트레이터 pid $p 종료"
fi

stage() {
  local name="$1"; shift; local log="$1"; shift; [[ "$1" == "--" ]] && shift
  [[ -f "$L/stages/${name}.done" ]] && { echo "[SKIP] $name"; return 0; }
  echo "=== [$(date +%H:%M:%S)] RETRY $name -> $log"
  "$@" >> "$log" 2>&1; local rc=$?
  if [[ $rc -eq 0 ]]; then
    date -Iseconds > "$L/stages/${name}.done"; rm -f "$L/stages/${name}.failed"
    echo "=== [$(date +%H:%M:%S)] OK $name"
  else
    echo "$rc" > "$L/stages/${name}.failed"; echo "=== [$(date +%H:%M:%S)] FAIL $name rc=$rc"
  fi
  return $rc
}

echo "########## 재시도 1: 7B 열 GSM8K FT (cache_dir 수정) ##########"
stage col_train_7b "$L/10_col_train_7b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      MODELS=llama2_7b ARMS=rsn BASIS_DIR_llama2_7b="$B7" \
      OUT_ROOT=outputs/wsr_rsn_col UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

echo "########## 재시도 2: 13B 열 (batch 1 × accum 16) ##########"
stage col_train_13b "$L/11_col_train_13b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      MODELS=llama2_13b ARMS=rsn BASIS_DIR_llama2_13b="$B13" \
      OUT_ROOT=outputs/wsr_rsn_col UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      BATCH_SIZE=1 GRAD_ACCUM=16 \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

echo "########## RETRY COMPLETE $(date -Iseconds) ##########"
ls -la "$L/stages/"
