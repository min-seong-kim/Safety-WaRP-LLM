#!/usr/bin/env bash
# WSR-(R)SN-Tune 오케스트레이터: 검출 → 학습 순차 실행. 단계마다 .done 마커로 재개.
# 주의: 이 스크립트가 도는 동안 자신이나 호출하는 드라이버를 **편집하지 말 것**
#       (bash 는 파일을 바이트 오프셋으로 읽어서 중간으로 점프한다).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
mkdir -p "$L/stages"
B7=$(cat outputs/wsr_sn_tune/llama2_7b/BASIS_DIR)
B13=$(cat outputs/wsr_sn_tune/llama2_13b/BASIS_DIR)

stage() {  # stage <이름> <로그> -- <명령...>
  local name="$1"; shift; local log="$1"; shift; [[ "$1" == "--" ]] && shift
  if [[ -f "$L/stages/${name}.done" ]]; then echo "[SKIP] $name"; return 0; fi
  echo "=== [$(date +%H:%M:%S)] START $name -> $log"
  "$@" >> "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    date -Iseconds > "$L/stages/${name}.done"; echo "=== [$(date +%H:%M:%S)] OK $name"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL $name rc=$rc  (로그: $log)"
    echo "$rc" > "$L/stages/${name}.failed"
  fi
  return $rc
}

wait_pid() {  # pid 파일이 가리키는 프로세스 종료 대기 (pgrep 금지 — CLAUDE.md 함정)
  local f="$1"; [[ -f "$f" ]] || return 0
  local p; p=$(cat "$f")
  while kill -0 "$p" 2>/dev/null; do sleep 20; done
  echo "[wait] pid $p ($f) 종료"
}

echo "########## 진행 중인 검출 대기 ##########"
wait_pid "$L/col7b.pid"
wait_pid "$L/row13b.pid"

echo "########## 1. 13B 열 검출 ##########"
stage col_detect_13b "$L/08_col_detect_13b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      MODELS=llama2_13b ARMS=rsn STOP_AFTER_NEURONS=1 \
      BASIS_DIR_llama2_13b="$B13" OUT_ROOT=outputs/wsr_rsn_col \
      UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

echo "########## 2. 열 학습 (7B → 13B) ##########"
stage col_train_7b "$L/10_col_train_7b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      MODELS=llama2_7b ARMS=rsn \
      BASIS_DIR_llama2_7b="$B7" OUT_ROOT=outputs/wsr_rsn_col \
      UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

stage col_train_13b "$L/11_col_train_13b.log" -- \
  env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      MODELS=llama2_13b ARMS=rsn \
      BASIS_DIR_llama2_13b="$B13" OUT_ROOT=outputs/wsr_rsn_col \
      UTILITY_JSON=sn_tune/corpus/wikipedia_utility_1000.json \
      bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh

echo "########## 3. 행 학습 (7B → 13B) ##########"
stage row_train_7b "$L/12_row_train_7b.log" -- \
  env MODELS=llama2_7b ARMS=rsn BASIS_DIR_llama2_7b="$B7" \
      bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh

stage row_train_13b "$L/13_row_train_13b.log" -- \
  env MODELS=llama2_13b ARMS=rsn BASIS_DIR_llama2_13b="$B13" \
      bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh

echo "########## ORCHESTRATION COMPLETE $(date -Iseconds) ##########"
ls -la "$L/stages/"
