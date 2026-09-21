#!/usr/bin/env bash
# 403(HF 공개 저장공간 초과)으로 막혔던 업로드를 몰아서 재시도한다 (2026-09-20, 사용자가 공간 정리).
#
#   1) Qwen α=32 재학습분 (_v2)  — asft, 그리고 학습이 끝나면 lisa
#   2) 실험 2 원공간 동결 base 라인 7셀
#
# 재실행 안전: .uploaded 가 있으면 건너뛴다. --prune 은 주지 않는다 —
# 평가를 로컬 가중치로 하고 있으므로 로컬을 지우면 안 된다.
# 리포명은 손으로 적지 않고 common.sh 의 레지스트리에서 도출한다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
source scripts/revision/common.sh >/dev/null 2>&1
PY="${PY:-python}"

# ⚠️ hf_repo_id 는 현재 셸의 LISA_RHO / HF_REPO_SUFFIX 를 읽는다. 학습을 돌린
#    _run_qwen_a32_v2.sh 와 **같은 값**을 명시하지 않으면 리포명이 rho0.0 으로
#    나와 엉뚱한 이름에 올라간다 (common.sh 의 LISA_RHO 기본값은 0.0 이다).
export HF_REPO_SUFFIX=_v2
export LISA_RHO=1.0

OK=(); FAILED=()

up() {  # <cell_dir> <repo_id>
  local cell="$1" repo="$2" tag="${1#outputs/}"
  if [ ! -f "$cell/.done" ];     then echo "[skip] $tag — 학습 미완료"; return 0; fi
  if [ -f "$cell/.uploaded" ];   then echo "[skip] $tag — 이미 업로드"; return 0; fi
  echo ""
  echo "▶ $tag → $repo"
  if "$PY" scripts/revision/upload_and_prune.py --cell_dir "$cell" --repo_id "$repo"; then
    echo "$repo" > "$cell/.uploaded"; echo "  ✅ $tag"; OK+=("$repo")
  else
    echo "  ❌ $tag"; FAILED+=("$tag")
  fi
}

# ── 1) Qwen α=32 _v2 : AsFT (Lisa 는 아래에서 학습 종료 후) ────────────────
up outputs/revision_qwen_a32_v2/cb/qwen25_7b/gsm8k/asft \
   "$(hf_repo_id cb qwen25_7b gsm8k asft)"

# ── 2) 실험 2 원공간 동결 base 라인 ───────────────────────────────────────
#    repo_id 규칙은 scripts/run_origspace_freeze_sweep.sh:74 와 같아야 한다:
#      {ns}/{MODEL_TAG}-origspace-freeze-{pNN}-{dataset}-lr{FULL_LR}
for mkey in llama2_7b_base llama31_8b_base; do
  model_cfg "$mkey" >/dev/null || { echo "[!] 알 수 없는 모델: $mkey"; continue; }
  mtag="$(hf_model_tag "$mkey")"
  for cell in outputs/origspace_base/"$mkey"/p*; do
    [ -d "$cell" ] || continue
    up "$cell" "${HF_NAMESPACE}/${mtag}-origspace-freeze-$(basename "$cell")-gsm8k-lr${FULL_LR}"
  done
done

# ── 3) Qwen Lisa : 학습 종료를 기다렸다가 (post_cell 이 이미 올렸으면 건너뜀) ──
PIDFILE=logs/qwen_a32_v2_train.pid
if [ -f "$PIDFILE" ]; then
  TPID="$(cat "$PIDFILE")"
  if kill -0 "$TPID" 2>/dev/null; then
    echo ""; echo "[upload] Lisa 학습(PID=$TPID) 종료 대기"
    while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
  fi
fi
up outputs/revision_qwen_a32_v2/cb/qwen25_7b/gsm8k/lisa \
   "$(hf_repo_id cb qwen25_7b gsm8k lisa)"

echo ""
echo "════════════ 업로드 결과 ════════════"
echo "성공 ${#OK[@]}건"; [ "${#OK[@]}" -gt 0 ] && printf '   ✅ %s\n' "${OK[@]}"
echo "실패 ${#FAILED[@]}건"; [ "${#FAILED[@]}" -gt 0 ] && printf '   ❌ %s\n' "${FAILED[@]}"
echo "UPLOAD_PENDING_DONE failed=${#FAILED[@]}  $(date -Iseconds)"
