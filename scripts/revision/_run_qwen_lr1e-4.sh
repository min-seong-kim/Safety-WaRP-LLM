#!/usr/bin/env bash
# Qwen2.5-7B-Instruct (CB safety-tuned) / GSM8K — AsFT · Lisa 를 **lr 1e-4** 로 (2026-09-20)
#
# lr 3e-4 세대(오늘 학습한 _v2)와 비교하기 위한 learning-rate ablation.
#   AsFT λ=1.0 / Lisa ρ=1.0 · align 100 · ft 900, 둘 다 LoRA r16/α32.
#   lr 외의 모든 설정은 동일: 3ep · 유효배치 16(4×4) · max_len 1024 · seed 42 · bf16 ·
#   cosine · wd 0 · warmup 0.03 · targets q,k,v,up,down.
#
# LORA_LR_DEFAULT 하나가 학습 명령과 리포명(_lr1e-4)을 **함께** 바꾼다 (hf_lr_tag → lora_lr).
# OUT_ROOT 은 따로 둔다 — out_dir 은 하이퍼파라미터를 이름에 담지 않아서, 같은 셀을 다른 lr 로
# 돌리면 기존 .done 에 걸려 건너뛴다(CLAUDE.md 에 기록된 함정).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

export SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS="asft lisa"
export LORA_LR_DEFAULT=1e-4
export LORA_ALPHA=32 LISA_RHO=1.0
export SKIP_PUBLISHED=0
export OUT_ROOT="$PWD/outputs/revision_qwen_lr1e-4"
export LOG_ROOT="$PWD/logs/revision_qwen_lr1e-4"
export PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0

echo "═══ [1/2] 학습 (AsFT · Lisa, lr 1e-4) ═══ $(date -Iseconds)"
bash scripts/revision/20_lora_family.sh; echo "STAGE20_RC=$?"

echo "═══ [2/2] 평가 ═══ $(date -Iseconds)"
mkdir -p outputs/eval_local
PATHS=()
add() {  # <method> <repo 이름>
  local cell="$OUT_ROOT/cb/qwen25_7b/gsm8k/$1" name="$2"
  [ -f "$cell/.done" ] || { echo "  [건너뜀] $1 — .done 없음"; return; }
  local md; md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  [ -n "$md" ] && [ -d "$md" ] || { echo "  [건너뜀] $1 — MODEL_DIR 없음"; return; }
  ln -sfn "$md" "outputs/eval_local/$name"; PATHS+=("$PWD/outputs/eval_local/$name")
}
add asft qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr1e-4
add lisa qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr1e-4
[ "${#PATHS[@]}" -eq 0 ] && { echo "❌ 평가할 셀이 없다"; exit 1; }
printf '  평가 대상: %s\n' "${PATHS[@]##*/}"
PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
echo "QWEN_LR1E4_DONE rc=$?  $(date -Iseconds)"
