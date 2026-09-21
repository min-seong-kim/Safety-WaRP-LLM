#!/usr/bin/env bash
# 3×3 sweep 이 끝나면 실험 2(논문 Table 1 의 base 라인)의 빠진 행만 채운다.
#
# 실험 2 구성: (a) 원본 base · (b) CB safety-tuned(SSFT) · (c) SSFT+gsm8k FT 동결 0% ·
#              (d) 동결 10/20/30/40/50%.
# (d) 10셀과 (b) 7B 는 HarmBench 4/4 + lm-eval 이 이미 끝났다. 아래 5개만 **HarmBench 미측정**
# 이고 lm-eval 은 전부 있다 → HB_ONLY=1 로 ASR 만 채운다 (모델당 약 6분).
#
# ⚠️ 전부 base 모델이라 chat template 이 붙으면 안 된다. 허브 리포 id 로 평가하므로
#    야간 기록의 "경로의 'it' 오탐" 경로를 타지 않는다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

PIDFILE=logs/qwen_3x3.pid
if [ -f "$PIDFILE" ]; then
  TPID="$(cat "$PIDFILE")"
  if kill -0 "$TPID" 2>/dev/null; then
    echo "[chain] 3x3 sweep(PID=$TPID) 종료 대기 $(date -Iseconds)"
    while kill -0 "$TPID" 2>/dev/null; do sleep 120; done
  fi
fi
echo "[chain] 실험2 갭필 시작 $(date -Iseconds)"

REPOS="meta-llama/Llama-2-7b-hf \
meta-llama/Llama-3.1-8B \
kmseong/Llama-3.1-8B-base-SSFT_lr5e-5 \
kmseong/llama2_7b-base-gsm8k_ssft_lr3e-5 \
kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5"

HB_ONLY=1 REPOS_ONLY="$REPOS" bash scripts/revision/32_eval_cells.sh
echo "EXP2_GAPFILL_DONE rc=$?  $(date -Iseconds)"
