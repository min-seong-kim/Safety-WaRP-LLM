#!/usr/bin/env bash
# 실패 단계 재시도 v2 — 사용자 지시 반영: batch 4×4 유지, 로컬 GSM8K JSON 사용.
#   col_train_7b  : [5] 성공(.tune.done) → [6] 만 재실행 (cache_dir 수정 + 로컬 JSON)
#   col_train_13b : [5] OOM 이었음 → weight 버퍼 CPU 오프로드(16.4GiB 회수)로 4×4 유지
#   row_train_13b : 오케스트레이터가 실패로 남겼으면 재시도
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
B7=$(cat outputs/wsr_sn_tune/llama2_7b/BASIS_DIR)
B13=$(cat outputs/wsr_sn_tune/llama2_13b/BASIS_DIR)

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

echo "########## 1. 7B 열 GSM8K FT ##########"
stage col_train_7b "$L/10_col_train_7b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      MODELS=llama2_7b ARMS=rsn BASIS_DIR_llama2_7b="$B7" \
      OUT_ROOT=outputs/wsr_rsn_col UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

echo "########## 2. 13B 열 (batch 4×4 + weight 오프로드) ##########"
stage col_train_13b "$L/11_col_train_13b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      MODELS=llama2_13b ARMS=rsn BASIS_DIR_llama2_13b="$B13" \
      OUT_ROOT=outputs/wsr_rsn_col UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

echo "########## 3. 13B 행 (실패했다면) ##########"
if [[ -f "$L/stages/row_train_13b.failed" ]]; then
  stage row_train_13b "$L/13_row_train_13b.log" -- \
    env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        MODELS=llama2_13b ARMS=rsn BASIS_DIR_llama2_13b="$B13" \
        bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh
else
  echo "[SKIP] row_train_13b 실패 기록 없음"
fi

echo "########## RETRY v2 COMPLETE $(date -Iseconds) ##########"
ls -la "$L/stages/"
