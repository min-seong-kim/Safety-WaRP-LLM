#!/usr/bin/env bash
# 학습2 종료 대기 → tokenizer_class 확인 → HF 업로드(_repro457, private) → run_all_eval.sh (GPU 0 순차)
set -uo pipefail
REPO=$HOME/Safety-WaRP-LLM; HB=$HOME/HarmBench
cd "$REPO"
export PY=$HOME/.conda/envs/hb/bin/python
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
CELL=$REPO/outputs/revision_repro_hb457/cb/llama32_3b/math/safelora
HFREPO=kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4_repro457
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
TP=$(awk '{print $2}' logs/revision_repro/train2.pid)
log "학습2 종료 대기 (pid $TP)"
while kill -0 "$TP" 2>/dev/null; do sleep 60; done
log "학습2 종료. rc=$(cat logs/revision_repro/train2.rc 2>/dev/null || echo '?')"
[[ -f "$CELL/.done" ]] || { log "❌ $CELL/.done 없음 — 학습 실패. 중단."; exit 1; }
MD=$(cat "$CELL/MODEL_DIR")
log "tokenizer_class: $(grep -o '"tokenizer_class": "[^"]*"' "$MD/tokenizer_config.json")  (4.57 로 저장했으면 PreTrainedTokenizerFast 여야 함)"
if grep -q TokenizersBackend "$MD/tokenizer_config.json"; then
  log "TokenizersBackend 발견 → 패치"; sed -i 's/"tokenizer_class": "TokenizersBackend"/"tokenizer_class": "PreTrainedTokenizerFast"/' "$MD/tokenizer_config.json"
fi
log "════ 1. HF 업로드 → $HFREPO (private)"
"$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$HFREPO" --private || { log "❌ 업로드/검증 실패"; exit 1; }
$HOME/.conda/envs/harmbench/bin/python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$HFREPO'); print('harmbench env tokenizer OK:', type(t).__name__)" || { log "❌ harmbench env 토크나이저 로드 실패"; exit 1; }
log "════ 2. 평가 (GPU 0 순차: HarmBench sys 4공격 → lm-eval hendrycks_math_safe)"
cd "$HB"
HB_GPU=0 LM_GPU=0 PARALLEL=0 RESUME=true VALIDATE_REPOS=1 HB_VARIANTS=sys SEED=42 ./run_all_eval.sh "$HFREPO"
rc=$?
log "════ 완료 rc=$rc"
echo "$rc" > "$REPO/logs/revision_repro/eval2.rc"
