#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  로컬 가중치를 "허브 리포와 똑같은 이름"의 심볼릭 링크로 모아 둔다.
#
#  왜 필요한가 (2026-09-20):
#    HF 공개 저장공간 할당량 초과(403)로 업로드가 막혔는데, 학습된 가중치는
#    로컬에 멀쩡히 있다. HarmBench/lm-eval 은 둘 다 로컬 디렉토리를 받을 수
#    있고, key 유도와 task 자동추론이 전부 "이름 문자열" 기반이므로 디렉토리
#    이름만 리포 이름과 같게 맞추면 허브 평가와 완전히 동일한 결과 key 가 나온다.
#    (HarmBench/add_models_to_yaml.py 에 로컬 디렉토리 분기를 추가해 두었다.)
#
#  사용:
#    bash scripts/revision/_stage_local_eval.sh                  # 링크만 생성
#    bash scripts/revision/_stage_local_eval.sh --print          # 경로 목록만 출력
#
#  ⚠️ 평가할 때 PREFETCH=0 을 반드시 줘야 한다. prefetch 단계가 `hf download <경로>`
#     를 시도해 셀당 5회 재시도(50초)를 헛되이 태운다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
ROOT="${STAGE_ROOT:-$REPO_DIR/outputs/origspace_base}"
DEST="${STAGE_DEST:-$REPO_DIR/outputs/eval_local}"
PRINT_ONLY=0; [ "${1:-}" = "--print" ] && PRINT_ONLY=1

source scripts/revision/common.sh >/dev/null 2>&1 || true
mkdir -p "$DEST"

paths=()
for cell in "$ROOT"/*/*; do
  [ -d "$cell" ] || continue
  [ -f "$cell/.done" ] || continue
  md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  [ -n "$md" ] && [ -d "$md" ] || continue
  ls "$md"/*.safetensors >/dev/null 2>&1 || continue

  # 리포 이름: UPLOAD.json 이 있으면 그것을 쓰고(실제로 쓴 이름), 없으면 규칙으로 만든다.
  repo=""
  if [ -f "$cell/UPLOAD.json" ]; then
    repo="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['repo_id'])" "$cell/UPLOAD.json" 2>/dev/null)"
  fi
  if [ -z "$repo" ]; then
    mkey="$(basename "$(dirname "$cell")")"; tag="$(basename "$cell")"
    model_cfg "$mkey" >/dev/null 2>&1 || continue
    repo="kmseong/$(hf_model_tag "$mkey")-origspace-freeze-${tag}-${TASK:-gsm8k}-lr${FULL_LR}"
  fi
  name="${repo##*/}"
  [ "$PRINT_ONLY" = "1" ] || ln -sfn "$md" "$DEST/$name"
  paths+=("$DEST/$name")
done

if [ "$PRINT_ONLY" = "1" ]; then
  printf '%s\n' "${paths[@]:-}"
else
  echo "스테이징 ${#paths[@]} 개 → $DEST"
  printf '   %s\n' "${paths[@]:-}"
fi
