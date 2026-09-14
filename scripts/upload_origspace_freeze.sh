#!/usr/bin/env bash
# origspace freeze 스윕 산출물 업로드 패스.
#
#   run_origspace_freeze_sweep.sh 의 업로드 단계는 --cell_dir 에 final_model 경로를 넘겨
#   실패한다(upload_and_prune.py 는 .done 과 MODEL_DIR 이 들어 있는 **셀 디렉토리**를 받는다).
#   학습은 정상이므로 이 스크립트로 업로드만 따로 돌린다. 재실행 안전(.uploaded 로 건너뜀).
#
#   bash scripts/upload_origspace_freeze.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-$PWD/outputs/origspace_freeze}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
PHASE3_DATASET="${PHASE3_DATASET:-gsm8k}"
LR="${LR:-5e-5}"

FAILED=()
for CELL in "$OUT_ROOT"/p*; do
  [ -d "$CELL" ] || continue
  TAG="$(basename "$CELL")"
  REPO="${HF_NAMESPACE}/llama2_7b-chat-origspace-freeze-${TAG}-${PHASE3_DATASET}-lr${LR}"

  if [ ! -f "$CELL/.done" ]; then
    echo "[skip] $TAG — 학습 미완료"; continue
  fi
  if [ -f "$CELL/.uploaded" ]; then
    echo "[skip] $TAG — 이미 업로드: $(cat "$CELL/.uploaded")"; continue
  fi

  echo ""
  echo "▶ $TAG → $REPO"
  if "$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$REPO"; then
    echo "$REPO" > "$CELL/.uploaded"
    echo "  ✅ $TAG"
  else
    echo "  ❌ $TAG 업로드 실패"
    FAILED+=("$TAG")
  fi
done

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "⚠️ 실패 ${#FAILED[@]}건:"; printf '   ✗ %s\n' "${FAILED[@]}"; exit 1
fi
echo "✅ 업로드 패스 완료"
