#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  batch3 체인이 끝난 뒤 실험 2 의 잔여분을 처리한다.
#    1) 업로드 재시도 — 그 사이 HF 저장공간이 확보됐다면 여기서 성공한다
#    2) 그래도 실패한 셀은 로컬 가중치로 평가 (허브 없이 동일 조건)
#
#  두 단계 모두 재실행 안전: .uploaded 마커로 건너뛰고, 평가는 RESUME 으로 건너뛴다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh hb >/dev/null 2>&1 || true
PY="${PY:-python}"
ROOT="$PWD/outputs/origspace_base"

echo "════════ (1) 업로드 재시도 ════════"
ok=0; fail=0
for cell in "$ROOT"/*/*; do
  [ -d "$cell" ] || continue
  [ -f "$cell/.done" ] || continue
  [ -f "$cell/.uploaded" ] && continue
  mkey="$(basename "$(dirname "$cell")")"; tag="$(basename "$cell")"
  case "$mkey" in
    llama2_7b_base)  repo="kmseong/llama2_7b-base-origspace-freeze-${tag}-gsm8k-lr3e-5" ;;
    llama31_8b_base) repo="kmseong/llama3_1_8b-base-origspace-freeze-${tag}-gsm8k-lr1e-5" ;;
    *) echo "  [!] 이름 규칙 모름: $mkey/$tag"; continue ;;
  esac
  echo "  ▶ $repo"
  if "$PY" scripts/revision/upload_and_prune.py --cell_dir "$cell" --repo_id "$repo"; then
    echo "$repo" > "$cell/.uploaded"; ok=$((ok+1)); echo "    ✅ 성공"
  else
    fail=$((fail+1)); echo "    ❌ 실패 (저장공간 할당량으로 추정) → 로컬 평가로 넘긴다"
  fi
done
echo "  업로드 재시도: 성공 $ok · 실패 $fail"

echo ""
echo "════════ (2) 실패분 로컬 평가 ════════"
bash scripts/revision/_eval_local_origspace.sh
echo "POST_CHAIN_DONE $(date -Is)"
