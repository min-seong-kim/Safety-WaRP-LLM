#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  gemma2_9b SEAL 재학습 — full-param lr 을 **1e-5** 로 (버그 수정 후)
#
#  왜 다시 도는가 (2026-09-16 규명)
#  --------------------------------
#  기존 모델 kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5 는
#    Direct ASR 0.4500 · AVG 0.5944 · GSM8K 0.1865
#  로 같은 모델의 다른 arm(Direct 0.0000 · AVG 0.03~0.21 · GSM8K 0.48~0.71)과 자릿수가 다르다.
#  원인은 셀 고유 실패가 아니라 **레지스트리 버그**였다: common.sh 의 model_cfg() 가
#  gemma 에 대해 SSFT_LR 만 3e-5 로 덮어쓰고 FULL_LR 은 전역 기본값 5e-5 를 그대로 썼다.
#  gemma 의 다른 full-param 행은 전부 lr 1e-5 다
#  (wvnvwn/gemma-2-9b-it-lr3e-5-gsm8k-lr1e-5 · wvnvwn/gemma-2-9b-it-lr3e-5-WaRP-lr1e-5,
#   RESULTS.md 측정조건표 "full-param 계열 | lr 5e-5 (gemma 1e-5)").
#  → 다른 arm 의 5배 lr 로 3 epoch full-param SFT 를 돌린 것이라 정렬·성능이 함께 무너졌다.
#  CLAUDE.md 가 "재학습해도 같은 결과" 라고 적은 것도 같은 잘못된 lr 을 다시 썼기 때문이다.
#
#  common.sh 에 GEMMA2_9B_FULL_LR(기본 1e-5) 를 넣어 고쳤으므로 이 스크립트는 그냥 stage 21 을
#  부른다. 결과 리포는 이름이 달라 기존 것과 충돌하지 않는다:
#      kmseong/gemma2_9b-it-CB_SSFT-seal_gsm8k_topp0.8_lr1e-5
#
#  ⚠️ 프롬프트 쪽은 이미 정상이다 — verify_prompt_parity.py 로 6개 경로 전부 일치 확인
#     (is_instruct_model 이 `-it` 를 토큰 경계로 잡아 chat template 을 쓴다).
#
#  사용: bash scripts/revision/retrain_gemma_seal.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_DIR"
CONDA_ROOT="${CONDA_ROOT:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3}"
export PY="${PY:-$CONDA_ROOT/envs/hb/bin/python}"
export HF_HOME="${HF_HOME:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache}"
export PATH="$(dirname "$PY"):$PATH"

export SAFETY_SETS=cb MODELS=gemma2_9b TASKS=gsm8k METHODS=seal
export PUSH_TO_HUB="${PUSH_TO_HUB:-1}" PRUNE_AFTER_UPLOAD="${PRUNE_AFTER_UPLOAD:-0}"
export PRUNE_BASIS=0 PRUNE_HF_CACHE=0 CONTINUE_ON_ERROR=1
export OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/revision_gemma_seal_lr1e-5}"   # 기존 셀과 격리
export SEAL_CKPT_ROOT="${SEAL_CKPT_ROOT:-$REPO_DIR/seal/ckpt/revision_gemma_lr1e-5}"
mkdir -p "$OUT_ROOT" "$SEAL_CKPT_ROOT" logs/revision

bash scripts/revision/21_seal.sh
