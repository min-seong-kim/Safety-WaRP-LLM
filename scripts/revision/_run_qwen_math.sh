#!/usr/bin/env bash
# Qwen2.5-7B-Instruct (CB safety-tuned) / MATH — 세 기법 (2026-09-20, 사용자 지정)
#
#   WSR-Tune  full-param ρ=0.1        (basis/mask 는 GSM8K 때 만든 것을 재사용 — 태스크 무관)
#   AsFT      LoRA r16/α32 λ=1.0
#   Lisa      LoRA r16/α32 ρ=1.0 · align 100 / ft 900
#
# 기준행(Full FT / Vanilla LoRA)은 만들지 않는다 — 사용자 지정, 절대값만 보고한다.
# 그래서 Δsafe/Δdown/Δoverall 은 계산하지 않는다.
#
# micro-batch 4 (유효 배치는 16 그대로). GSM8K WSR-Tune 셀과 같은 값이라 두 태스크가 짝이 맞는다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

export SAFETY_SETS=cb MODELS=qwen25_7b TASKS=math
export SKIP_PUBLISHED=0
export QWEN25_7B_MB_WARP=4
export LISA_RHO=1.0 LORA_ALPHA=32
export PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0
export LOG_ROOT="$PWD/logs/revision_qwen_math"

echo "═══ [1/4] Stage 02 확인 (basis/mask 재사용 — 건너뛰어야 정상) ═══ $(date -Iseconds)"
METHODS=wsr_tune bash scripts/revision/02_warp_basis_mask.sh; echo "STAGE02_RC=$?"

echo "═══ [2/4] LoRA α=32 — AsFT · Lisa ═══ $(date -Iseconds)"
METHODS="asft lisa" bash scripts/revision/20_lora_family.sh; echo "STAGE20_RC=$?"

echo "═══ [3/4] WSR-Tune (full-param) ═══ $(date -Iseconds)"
METHODS=wsr_tune bash scripts/revision/12_wsr_tune.sh; echo "STAGE12_RC=$?"

echo "═══ [4/4] 평가 ═══ $(date -Iseconds)"
mkdir -p outputs/eval_local
PATHS=()
add() {  # <method> <repo 이름>
  local cell="outputs/revision/cb/qwen25_7b/math/$1" name="$2"
  [ -f "$cell/.done" ] || { echo "  [건너뜀] $1 — .done 없음"; return; }
  local md; md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  [ -n "$md" ] && [ -d "$md" ] || { echo "  [건너뜀] $1 — MODEL_DIR 없음"; return; }
  ln -sfn "$md" "outputs/eval_local/$name"; PATHS+=("$PWD/outputs/eval_local/$name")
}
add wsr_tune qwen2_5_7b-instruct-CB_SSFT-wsr-tune_math_rho0.1_lr5e-5
add asft     qwen2_5_7b-instruct-CB_SSFT-asft_math_lambda1.0_lr3e-4
add lisa     qwen2_5_7b-instruct-CB_SSFT-lisa_math_rho1.0_lr3e-4
[ "${#PATHS[@]}" -eq 0 ] && { echo "❌ 평가할 셀이 없다"; exit 1; }
printf '  평가 대상: %s\n' "${PATHS[@]##*/}"
PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
echo "QWEN_MATH_DONE rc=$?  $(date -Iseconds)"
