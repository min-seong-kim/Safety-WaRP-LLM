#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  업로드가 막힌(=.uploaded 없는) 실험 2 셀을 "로컬 가중치"로 평가한다.
#
#  배경 (2026-09-20): HF 공개 저장공간 할당량 초과(403)로 업로드가 실패했다.
#  학습 결과는 로컬에 멀쩡하고, HarmBench/lm-eval 은 로컬 디렉토리를 그대로
#  받으므로 허브를 거치지 않고 같은 조건으로 평가할 수 있다.
#
#  동일성 검증 (2026-09-20):
#    add_models_to_yaml.probe_repo(로컬) == probe_repo(허브) == (True, False, 4096)
#    yaml key 도 양쪽 모두 `llama2_7b-base-origspace-freeze-pNN-gsm8k-lr3e-5`.
#
#  주의:
#    · PREFETCH=0 필수. prefetch 가 `hf download <로컬경로>` 를 5회 재시도하며 시간을 태운다.
#    · lm-eval 결과 디렉토리 이름은 `/`→`__` 치환이라 로컬 경로일 때 길어진다.
#      모델 이름은 항상 마지막 `__` 뒤쪽이므로 추출 시 rsplit("__",1)[-1] 로 정규화할 것.
#    · HarmBench 쪽 결과 경로는 yaml key 기반이라 허브 평가와 완전히 같은 이름이 나온다.
#
#  나중에 할당량이 풀려 업로드에 성공하면, 같은 key 가 이미 있으므로 재평가는
#  RESUME 에 걸려 건너뛴다(= 중복 측정 없음).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="${STAGE_ROOT:-$PWD/outputs/origspace_base}"

# 1) 업로드 안 된 완료 셀만 스테이징
bash scripts/revision/_stage_local_eval.sh >/dev/null
PATHS=()
for cell in "$ROOT"/*/*; do
  [ -d "$cell" ] || continue
  [ -f "$cell/.done" ] || continue
  [ -f "$cell/.uploaded" ] && continue          # 허브에 있는 것은 허브로 평가
  md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"; [ -n "$md" ] && [ -d "$md" ] || continue
  ls "$md"/*.safetensors >/dev/null 2>&1 || continue
  mkey="$(basename "$(dirname "$cell")")"; tag="$(basename "$cell")"
  # 이름 규칙은 run_origspace_freeze_sweep.sh 의 repo_id() 와 같아야 한다
  case "$mkey" in
    llama2_7b_base)  name="llama2_7b-base-origspace-freeze-${tag}-gsm8k-lr3e-5" ;;
    llama31_8b_base) name="llama3_1_8b-base-origspace-freeze-${tag}-gsm8k-lr1e-5" ;;
    *) echo "  [!] 이름 규칙 모름: $mkey"; continue ;;
  esac
  ln -sfn "$md" "$PWD/outputs/eval_local/$name"
  PATHS+=("$PWD/outputs/eval_local/$name")
done

if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "로컬 평가할 셀이 없다 (전부 업로드됐거나 학습 미완료)"; exit 0
fi

echo "════════════════════════════════════════════════════════════════"
echo "  로컬 평가 ${#PATHS[@]} 개 (업로드 실패분)"
printf '   - %s\n' "${PATHS[@]##*/}"
echo "════════════════════════════════════════════════════════════════"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
echo "LOCAL_EVAL_EXIT=$?"
