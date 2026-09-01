#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  명백히 실패한 두 셀을 재학습하고 다시 평가한다 (2026-09-01 사용자 지시).
#    · cb/llama2_13b/gsm8k/salora  — GSM8K 0.0675 / JB 0.4506 (유용성·안전성 동시 붕괴)
#    · cb/gemma2_9b/gsm8k/seal     — GSM8K 0.2820 / JB 0.6894 (36셀 중 최악)
#  두 셀 모두 3B·8B·Qwen 동일 설정에서는 정상이라 셀 단독 문제로 판단.
#  gemma seal 은 이전 박스 산출물이라 학습 로그가 없다 → 이 박스의 수정된 코드로 새로 만든다.
#
#  선행 작업(평가 감시 + 누락 보충)이 끝난 뒤 시작한다. 단일 GPU 라 겹치면 안 된다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; HB=/home/edgeai_lab/HarmBench
cd "$REPO"; export PATH="$HOME/miniconda3/envs/hb/bin:$PATH"
TS=$(date +%Y%m%d_%H%M%S)
ULOG="$REPO/logs/revision_unattended"; mkdir -p "$ULOG"
LOG="$ULOG/retrain_two_${TS}.log"; ln -sfn "$LOG" "$ULOG/retrain_two_latest.log"
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "선행 작업(평가/보충) 종료 대기"
while pgrep -f "supervise_eval\.sh|fill_missing_lmeval\.sh" > /dev/null; do sleep 120; done
log "선행 종료 확인 — 재학습 시작"

export PUSH_TO_HUB=1 CONTINUE_ON_ERROR=1 SAFETY_SETS=cb
export BASE_BLOCKED_MODELS=""
export DEADLINE_HOURS="${DEADLINE_HOURS:-48}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
mkdir -p "$REPO/outputs/revision"

# 마커 제거 (이미 지웠지만 재실행 안전용)
rm -f outputs/revision/cb/llama2_13b/gsm8k/salora/.done outputs/revision/cb/llama2_13b/gsm8k/salora/.uploaded
rm -f outputs/revision/cb/gemma2_9b/gsm8k/seal/.done  outputs/revision/cb/gemma2_9b/gsm8k/seal/.uploaded

log "════ ① 재학습: llama2_13b / salora ════"
MODELS=llama2_13b METHODS=salora bash "$HERE/run_all.sh"
log "════ ② 재학습: gemma2_9b / seal ════"
MODELS=gemma2_9b METHODS=seal bash "$HERE/run_all.sh"

log "════ ③ 재평가 (ASR + downstream) ════"
cd "$HB"
for m in kmseong/llama2_13b-chat-CB_SSFT-salora_gsm8k_rs32rt32_lr3e-4 \
         kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5; do
  log "── $m"
  RESUME=false VALIDATE_REPOS=0 ./run_all_eval.sh "$m" || log "   실패: $m"
done
log "════ 완료 ════"
