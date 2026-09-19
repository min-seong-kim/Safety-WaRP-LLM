#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  학습이 끝난 셀들을 HarmBench(4공격 ASR) + lm-eval(downstream) 로 평가한다.
#
#  측정 조건은 RESULTS.md 의 다른 모든 행과 같아야 한다:
#    · HarmBench · AdvBench_behaviors_standard · **sys 모드**(llama-2 <<SYS>> 안전프롬프트)
#    · GRADING=hard (refusal keyword) · DirectRequest/AutoDAN/PAIR/PAP · seed 42
#    · lm-eval 5-shot, --apply_chat_template, gsm8k=flexible-extract
#  run_all_eval.sh 의 HB_VARIANTS 기본값이 이미 sys 다(주석은 nosys 라고 하지만 코드는 sys).
#
#  사용:
#    bash scripts/revision/32_eval_cells.sh                     # exp1 셀 전부
#    EVAL_ROOT=outputs/origspace_base bash scripts/revision/32_eval_cells.sh
#    REPOS_EXTRA="kmseong/a kmseong/b" bash scripts/revision/32_eval_cells.sh
#    REPOS_ONLY="kmseong/a" bash scripts/revision/32_eval_cells.sh   # 목록을 직접 지정
#
#  ⚠️ 허브 업로드/다운로드는 하루에 몇 번씩 IncompleteRead 로 끊긴다. 끝나면
#     RESUME=true LMEVAL_RESUME=1 로 한 번 더 돌려 빈 칸을 메우는 것이 정석이다
#     (이 스크립트는 그 두 값을 기본으로 켠다).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
BASE=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong

EVAL_ROOT="${EVAL_ROOT:-$REPO_DIR/outputs/asft_lisa_fullft}"
HARMBENCH_DIR="${HARMBENCH_DIR:-$BASE/HarmBench}"
LMEVAL_DIR="${LMEVAL_DIR:-$BASE/lm-evaluation-harness}"
export CONDA_SH="${CONDA_SH:-$BASE/miniconda3/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-$BASE/.hf_cache}"
# /tmp 은 이 박스에서 noexec 이다 → Triton 이 JIT .so 를 mmap 못 해 죽는다.
export TMPDIR="${TMPDIR:-$BASE/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.torchinductor_cache}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export HB_VARIANTS="${HB_VARIANTS:-sys}"
export RESUME="${RESUME:-true}"
export LMEVAL_RESUME="${LMEVAL_RESUME:-1}"
export SEED="${SEED:-42}"

# ── 평가 대상 모으기 ─────────────────────────────────────────────────────
REPOS=()
if [ -n "${REPOS_ONLY:-}" ]; then
  for r in $REPOS_ONLY; do REPOS+=("$r"); done
else
  while IFS= read -r f; do
    [ -f "$(dirname "$f")/.uploaded" ] || continue
    REPOS+=("$(cat "$f")")
  done < <(find "$EVAL_ROOT" -name REPO 2>/dev/null | sort)
fi
for r in ${REPOS_EXTRA:-}; do REPOS+=("$r"); done

if [ "${#REPOS[@]}" -eq 0 ]; then
  echo "평가할 리포가 없다 (EVAL_ROOT=$EVAL_ROOT 에 .uploaded 된 셀이 없음)"; exit 0
fi

echo "════════════════════════════════════════════════════════════════"
echo "  평가 대상 ${#REPOS[@]} 개 (HB_VARIANTS=$HB_VARIANTS)"
printf '   - %s\n' "${REPOS[@]}"
echo "════════════════════════════════════════════════════════════════"
[ "${DRY_RUN:-0}" = "1" ] && { echo "*** DRY RUN ***"; exit 0; }

cd "$HARMBENCH_DIR"
HARMBENCH_DIR="$HARMBENCH_DIR" LMEVAL_DIR="$LMEVAL_DIR" \
  bash ./run_all_eval.sh "${REPOS[@]}"
RC=$?
echo "EVAL_FINISHED rc=$RC"
exit $RC
