#!/usr/bin/env bash
# Qwen2.5-7B-It / GSM8K — AsFT·Lisa 의 (rank,alpha) × lr 3×3 sweep (2026-09-21, 사용자 지정)
#
#   (r,alpha) : r8/a16(scaling 2) · r8/a8(scaling 1) · r16/a16(scaling 1)
#   lr        : 1e-4 · 3e-4 · 5e-4
#   9 설정 × 2 기법(AsFT λ=1.0, Lisa ρ=1.0) = 18 셀.
#   나머지는 전부 고정: dropout 0.05 · targets q,k,v,up,down · 3ep · 유효배치 16(4×4) ·
#   max_len 1024 · seed 42 · bf16 · cosine · wd 0 · warmup 0.03.
#
# ⚠️ 리포명에 rank 가 안 들어간다(lora_alpha_tag 는 alpha 만). r=8 은 HF_REPO_SUFFIX=_r8 로
#    구분한다. r16/a16/lr3e-4 만 기존 α=16 세대와 이름이 겹쳐 _v2 를 붙인다(기존본 보존).
# ⚠️ OUT_ROOT 을 설정마다 분리한다 — out_dir 은 하이퍼파라미터를 이름에 담지 않는다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

DEADLINE_H="${DEADLINE_H:-6.5}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + $(python3 -c "print(int($DEADLINE_H*3600))") ))
echo "새 셀 시작 마감: $(date -d @$REVISION_DEADLINE_EPOCH '+%m-%d %H:%M')  (이후엔 평가로 넘어감)"

run_group() {           # $1 = rank, $2 = alpha
  local R="$1" A="$2" PATHS=()
  for LR in 1e-4 3e-4 5e-4; do
    local SUF=""
    [ "$R" != "16" ] && SUF="_r${R}"
    [ "$R" = "16" ] && [ "$A" = "16" ] && [ "$LR" = "3e-4" ] && SUF="_v2"
    local TAG="r${R}a${A}_lr${LR}"
    local OUT="$PWD/outputs/revision_qwen_3x3/$TAG"
    echo ""
    echo "═══ r=$R alpha=$A (scaling $(python3 -c "print($A/$R)")) lr=$LR ═══ $(date -Iseconds)"
    mkdir -p "$OUT" "$PWD/logs/revision_qwen_3x3/$TAG"
    env SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS="asft lisa" \
        LORA_R="$R" LORA_ALPHA="$A" LORA_LR_DEFAULT="$LR" LISA_RHO=1.0 SKIP_PUBLISHED=0 \
        HF_REPO_SUFFIX="$SUF" \
        OUT_ROOT="$OUT" LOG_ROOT="$PWD/logs/revision_qwen_3x3/$TAG" \
        PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0 \
        REVISION_DEADLINE_EPOCH="$REVISION_DEADLINE_EPOCH" \
        bash scripts/revision/20_lora_family.sh
    echo "${TAG}_RC=$?"
    for m in asft lisa; do
      local cell="$OUT/cb/qwen25_7b/gsm8k/$m"
      [ -f "$cell/.done" ] || continue
      local md; md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
      [ -n "$md" ] && [ -d "$md" ] || continue
      local repo; repo="$(head -1 "$cell/.uploaded" 2>/dev/null)"
      [ -n "$repo" ] || { echo "  [평가제외] $TAG/$m — 업로드 기록 없음"; continue; }
      mkdir -p outputs/eval_local; ln -sfn "$md" "outputs/eval_local/${repo##*/}"
      PATHS+=("$PWD/outputs/eval_local/${repo##*/}")
    done
  done
  if [ "${#PATHS[@]}" -gt 0 ]; then
    echo ""; echo "═══ 평가 (r=$R a=$A, ${#PATHS[@]}개) ═══ $(date -Iseconds)"
    printf '  %s\n' "${PATHS[@]##*/}"
    PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
    echo "EVAL_RC_r${R}a${A}=$?"
  fi
}

run_group 8 16
run_group 8 8
run_group 16 16
echo "QWEN_3X3_DONE rc=$?  $(date -Iseconds)"
