#!/usr/bin/env bash
# origspace freeze 스윕의 **safety 전용** 평가 (HarmBench only, lm-eval 생략).
#
#   기존 결과표와 같은 조건으로 돌린다:
#     sys 모드(llama-2 <<SYS>> 안전 프롬프트 포함) · GRADING=hard(refusal keyword)
#     Advbench_behaviors_standard · DirectRequest/AutoDAN/PAIR/PAP · seed 42
#   ρ=0 기준행은 이미 측정돼 있다:
#     kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5  → AVG 0.2078 (RESULTS.md)
#
#   bash scripts/eval_origspace_freeze_safety.sh                 # 새 배치 5개
#   REPOS_EXTRA="kmseong/..." bash scripts/eval_origspace_freeze_safety.sh
set -uo pipefail
BASE=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong
export CONDA_SH="${CONDA_SH:-$BASE/miniconda3/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export TMPDIR="${TMPDIR:-$BASE/tmp}"      # /tmp 이 noexec 이라 triton 이 죽는다
export HB_ONLY=1                           # ← safety 만 본다 (lm-eval 생략)

RATIOS="${RATIOS:-p05 p20 p30 p40 p50}"
LR="${LR:-5e-5}"
REPOS=()
for t in $RATIOS; do
  REPOS+=("kmseong/llama2_7b-chat-origspace-freeze-${t}-gsm8k-lr${LR}")
done
# 추가로 볼 모델 (예: 기존 배치 p10, Before-FT 모델)
for r in ${REPOS_EXTRA:-}; do REPOS+=("$r"); done

echo "평가 대상 (${#REPOS[@]}개):"; printf '  - %s\n' "${REPOS[@]}"
cd "$BASE/HarmBench"
bash ./run_all_eval.sh "${REPOS[@]}"
echo "ORIGSPACE_EVAL_EXIT=$?"
