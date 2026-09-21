#!/usr/bin/env bash
# 학습(_run_qwen_a32_v2.sh)이 끝나면 두 셀을 HarmBench + lm-eval 로 평가한다.
#
# 평가는 **로컬 가중치**로 한다. 2026-09-20 현재 HF 공개 저장공간이 403(할당량 초과)
# 이라 업로드가 실패할 수 있고, 성공하더라도 이 박스는 다운로드가 ≈9MB/s 라
# 허브에서 다시 받는 것이 낭비다. 디렉토리 이름을 리포 이름과 똑같이 맞추면
# HarmBench/lm-eval 의 key 유도와 task/chat-template 추론이 허브 평가와 동일하다
# (scripts/revision/_stage_local_eval.sh 주석 참조). PREFETCH=0 이 필수다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

PIDFILE=logs/qwen_a32_v2_train.pid
# ⚠️ pgrep -f <스크립트명> 은 감시자 자신까지 잡는다(CLAUDE.md). PID 파일로만 판정한다.
if [ -f "$PIDFILE" ]; then
  TPID="$(cat "$PIDFILE")"
  echo "[chain] 학습 PID=$TPID 종료 대기"
  while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
  echo "[chain] 학습 종료 감지 $(date -Iseconds)"
fi

CELLROOT=outputs/revision_qwen_a32_v2/cb/qwen25_7b/gsm8k
mkdir -p outputs/eval_local
PATHS=()
declare -A NAMES=(
  [asft]=kmseong/qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4_v2
  [lisa]=kmseong/qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4_v2
)
for m in asft lisa; do
  cell="$CELLROOT/$m"
  [ -f "$cell/.done" ] || { echo "[chain] $m: .done 없음 — 건너뜀"; continue; }
  md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  [ -n "$md" ] && [ -d "$md" ] || { echo "[chain] $m: MODEL_DIR 없음"; continue; }
  ls "$md"/*.safetensors >/dev/null 2>&1 || { echo "[chain] $m: safetensors 없음 ($md)"; continue; }
  name="${NAMES[$m]##*/}"
  ln -sfn "$md" "outputs/eval_local/$name"
  PATHS+=("$PWD/outputs/eval_local/$name")
done

if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "[chain] 평가할 셀이 없다 — 종료"; exit 1
fi
echo "[chain] 평가 대상:"; printf '   %s\n' "${PATHS[@]##*/}"

PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
echo "QWEN_A32_V2_EVAL_DONE rc=$?  $(date -Iseconds)"
