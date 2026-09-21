#!/usr/bin/env bash
# Qwen2.5-7B-It / GSM8K — AsFT·Lisa 의 LoRA **용량**(rank·alpha) sweep (2026-09-21, 사용자 지정)
#
# 목적: downstream(GSM8K)을 끌어내리는 설정을 찾는다. lr 은 축에서 뺐다 —
#   이미 5점(3e-5~5e-3)을 돌렸고 AsFT 의 GSM8K 가 0.7202~0.7346 으로 평평했다(167배 구간).
#   용량(rank·alpha)이 진짜 레버다. 용량 → 0 이면 출발 모델 점수(GSM8K 0.6437)로 수렴한다.
#
#   축 A  scaling 축 : r=16 고정, alpha 2·4·8   (scaling 0.125 / 0.25 / 0.5)
#   축 B  rank 축    : scaling=2 고정(alpha=2r), r 2·4·8
#   기존 앵커: r16a16(scaling 1) · r16a32(scaling 2) 는 이미 허브에 있다.
#
# ⚠️ 리포명에 rank 가 들어가지 않는다(lora_alpha_tag 는 alpha 만 붙인다).
#    r≠16 이면 HF_REPO_SUFFIX=_r<R> 로 구분하지 않으면 기존 _a16 모델을 덮어쓴다.
# ⚠️ OUT_ROOT 도 설정마다 분리한다 — out_dir 은 하이퍼파라미터를 이름에 담지 않아
#    같은 경로면 기존 .done 에 걸려 건너뛴다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

# 새 셀 시작 마감 (기본 6시간). 이후엔 진행 중인 셀만 끝내고 평가로 넘어간다.
DEADLINE_H="${DEADLINE_H:-6}"
export REVISION_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_H*3600 ))
echo "새 셀 시작 마감: $(date -d @$REVISION_DEADLINE_EPOCH '+%m-%d %H:%M')"

# "<r> <alpha>" 목록. 축 A 먼저(같은 rank 라 비교가 깔끔하다), 그다음 축 B.
CONFIGS="${CONFIGS:-16:8 16:4 16:2 8:16 4:8 2:4}"

run_axis() {   # $@ = 설정 목록
  local PATHS=()
  for cfg in "$@"; do
    R="${cfg%%:*}"; A="${cfg##*:}"
    TAG="r${R}a${A}"
    SUF=""; [ "$R" != "16" ] && SUF="_r${R}"
    OUT="$PWD/outputs/revision_qwen_cap/$TAG"
    echo ""
    echo "═══════════ r=$R alpha=$A (scaling $(python3 -c "print($A/$R)")) ═══════════ $(date -Iseconds)"
    mkdir -p "$OUT" "$PWD/logs/revision_qwen_cap/$TAG"
    env SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS="asft lisa" \
        LORA_R="$R" LORA_ALPHA="$A" LISA_RHO=1.0 SKIP_PUBLISHED=0 \
        HF_REPO_SUFFIX="$SUF" \
        OUT_ROOT="$OUT" LOG_ROOT="$PWD/logs/revision_qwen_cap/$TAG" \
        PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0 \
        REVISION_DEADLINE_EPOCH="$REVISION_DEADLINE_EPOCH" \
        bash scripts/revision/20_lora_family.sh
    echo "${TAG}_RC=$?"
    for m in asft lisa; do
      cell="$OUT/cb/qwen25_7b/gsm8k/$m"
      [ -f "$cell/.done" ] || continue
      md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"; [ -n "$md" ] && [ -d "$md" ] || continue
      # 리포명은 손으로 만들지 않고 업로드가 남긴 기록에서 읽는다
      repo="$(head -1 "$cell/.uploaded" 2>/dev/null)"
      [ -n "$repo" ] || repo="qwen-${TAG}-${m}"   # 업로드 실패 시 임시 이름
      name="${repo##*/}"
      mkdir -p outputs/eval_local; ln -sfn "$md" "outputs/eval_local/$name"
      PATHS+=("$PWD/outputs/eval_local/$name")
    done
  done
  if [ "${#PATHS[@]}" -gt 0 ]; then
    echo ""; echo "═══════════ 평가 ${#PATHS[@]}개 ═══════════ $(date -Iseconds)"
    printf '  %s\n' "${PATHS[@]##*/}"
    PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
    echo "AXIS_EVAL_RC=$?"
  fi
}

# 축을 나눠 평가한다 — 중간에 시간이 모자라도 앞쪽 결과는 남는다.
run_axis 16:8 16:4 16:2
run_axis 8:16 4:8 2:4

echo "QWEN_CAP_SWEEP_DONE rc=$?  $(date -Iseconds)"
