#!/usr/bin/env bash
# 실험 2 base 라인의 빠진 구간: llama2_7b_base 원공간 동결 5% (2026-09-21, 사용자 지정)
#
# 기존 sweep 은 10~50% 만 있다. chat 라인에는 p05 가 있으므로 base 도 채운다.
# 동작점은 base 라인 기본값 그대로: lr 3e-5 · 3ep · 유효배치 16 · wd 0.01 · warmup 0.1 · seed 42.
# 출발 모델 kmseong/llama2_7b-base-CB_SSFT-lr3e-5 (레지스트리 ALIGNED_CB).
# 원공간 동결이라 Phase 1 basis 는 필요 없다 (Phase2 가 load_basis 를 no-op 으로 오버라이드).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

BASE_MODELS=llama2_7b_base RATIOS=0.05 TASK=gsm8k \
  bash scripts/revision/31_origspace_base.sh
echo "BASE_P05_TRAIN_RC=$?"

CELL=outputs/origspace_base/llama2_7b_base/p05
NAME=llama2_7b-base-origspace-freeze-p05-gsm8k-lr3e-5
if [ -f "$CELL/.done" ]; then
  MD="$(cat "$CELL/MODEL_DIR" 2>/dev/null)"
  if [ -n "$MD" ] && [ -d "$MD" ]; then
    mkdir -p outputs/eval_local; ln -sfn "$MD" "outputs/eval_local/$NAME"
    echo "=== 평가 $NAME ==="
    PREFETCH=0 REPOS_ONLY="$PWD/outputs/eval_local/$NAME" bash scripts/revision/32_eval_cells.sh
    echo "BASE_P05_EVAL_RC=$?"
  else
    echo "❌ MODEL_DIR 없음"
  fi
else
  echo "❌ 학습 미완료 — 평가 생략"
fi
echo "BASE_P05_DONE $(date -Iseconds)"
