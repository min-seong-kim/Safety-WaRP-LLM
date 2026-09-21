#!/usr/bin/env bash
# 3×3 sweep 셀 중 **아직 평가되지 않은 것**을 로컬 가중치로 평가한다.
#
# 왜 필요한가: _run_qwen_3x3.sh 는 셀의 .uploaded 에서 리포명을 읽어 평가 대상에 넣는다.
# HF 403(저장공간) 으로 업로드가 막히면 그 셀이 통째로 평가에서 빠진다. 여기서는 리포명을
# 디렉토리 태그에서 **결정적으로** 유도하므로 업로드 성공 여부와 무관하다.
#   이름 규칙은 common.sh 의 hf_repo_id 와 같아야 한다:
#     {method}_gsm8k_{hparam}[_a<A> (A!=32)]_lr<LR>[suffix]
#     suffix: r=8 → _r8 / (r16,a16,lr3e-4) → _v2 / 그 외 없음
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
B=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong
HB="$B/HarmBench/results/Advbench_behaviors_standard"
LM="$B/lm-evaluation-harness/eval_results"

measured() {   # <key> → HarmBench 4/4 이고 lm-eval 있으면 0
  local k="$1" n=0 a
  for a in DirectRequest AutoDAN PAIR PAP; do
    compgen -G "$HB/$a/**/results/$k.json" >/dev/null 2>&1 && n=$((n+1))
  done
  [ "$n" -eq 4 ] || return 1
  local d
  for d in "$LM"/*; do
    [ "${d##*__}" = "$k" ] && compgen -G "$d/results_*.json" >/dev/null 2>&1 && return 0
  done
  return 1
}

shopt -s globstar nullglob
PATHS=()
for cell in outputs/revision_qwen_3x3/*/cb/qwen25_7b/gsm8k/*; do
  [ -f "$cell/.done" ] || continue
  m="$(basename "$cell")"
  tag="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$cell")")")")")"   # r8a16_lr1e-4
  R="${tag#r}"; R="${R%%a*}"
  A="${tag#*a}"; A="${A%%_*}"
  LR="${tag##*_lr}"
  suf=""; [ "$R" != "16" ] && suf="_r${R}"
  [ "$R" = "16" ] && [ "$A" = "16" ] && [ "$LR" = "3e-4" ] && suf="_v2"
  atag=""; [ "$A" != "32" ] && atag="_a${A}"
  case "$m" in
    asft) name="qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0${atag}_lr${LR}${suf}" ;;
    lisa) name="qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0${atag}_lr${LR}${suf}" ;;
    *) continue ;;
  esac
  if measured "$name"; then echo "[측정완료] $name"; continue; fi
  md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  [ -n "$md" ] && [ -d "$md" ] || { echo "[가중치없음] $name"; continue; }
  mkdir -p outputs/eval_local; ln -sfn "$md" "outputs/eval_local/$name"
  PATHS+=("$PWD/outputs/eval_local/$name")
done

if [ "${#PATHS[@]}" -eq 0 ]; then echo "평가할 셀 없음 — 전부 측정됨"; exit 0; fi
echo "=== 미측정 ${#PATHS[@]}개 평가 ==="; printf '  %s\n' "${PATHS[@]##*/}"
PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
echo "EVAL_3X3_GAPFILL_DONE rc=$?  $(date -Iseconds)"
