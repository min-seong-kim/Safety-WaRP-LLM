#!/usr/bin/env bash
# 업로드가 끝난 셀의 **가중치만** 지운다 (2026-09-20, 디스크 한계).
#
# upload_and_prune.py --verify_only --prune 을 쓴다:
#   · 허브 사본을 먼저 재검증(파일/크기/AutoConfig/chat_template)하고
#   · 통과한 셀만 가중치를 지운다. config/tokenizer/.done/.uploaded 는 남는다.
#     (.done/.uploaded 는 "이미 끝난 셀"이라는 유일한 기록이다 — 지우면 재학습된다.)
#
# 제외: 지금 학습 중인 Qwen WSR-Tune 의 산출물.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
PY="${PY:-python}"
EXCLUDE_RE="${EXCLUDE_RE:-outputs/revision/cb/qwen25_7b/gsm8k/wsr_tune}"

OK=0; SKIP=0; FAIL=0
for cell in $(find outputs -name .done 2>/dev/null | xargs -n1 dirname | sort); do
  if [[ "$cell" =~ $EXCLUDE_RE ]]; then echo "[제외] $cell (학습 중)"; SKIP=$((SKIP+1)); continue; fi
  repo=""
  [ -f "$cell/.uploaded" ] && repo="$(head -1 "$cell/.uploaded")"
  if [ -z "$repo" ] && [ -f "$cell/UPLOAD.json" ]; then
    repo="$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['repo_id'])" "$cell/UPLOAD.json" 2>/dev/null)"
  fi
  if [ -z "$repo" ]; then echo "[건너뜀] $cell — 리포명 없음"; SKIP=$((SKIP+1)); continue; fi
  md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
  if [ -n "$md" ] && ! ls "$md"/*.safetensors "$md"/*.bin >/dev/null 2>&1; then
    echo "[건너뜀] $cell — 이미 가중치 없음"; SKIP=$((SKIP+1)); continue
  fi
  echo ""
  echo "▶ $cell  →  $repo"
  if $PY scripts/revision/upload_and_prune.py --cell_dir "$cell" --repo_id "$repo" --verify_only --prune 2>&1 | sed 's/^/   /'; then
    OK=$((OK+1))
  else
    echo "   ❌ 검증 실패 — 가중치 보존"; FAIL=$((FAIL+1))
  fi
done
echo ""
echo "════ prune 결과: 성공 $OK / 건너뜀 $SKIP / 실패 $FAIL ════"
echo "PRUNE_CELLS_DONE $(date -Iseconds)"
