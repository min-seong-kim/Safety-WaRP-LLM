#!/usr/bin/env bash
# 학습 종료 대기 → HF 업로드(_repro, private) → run_all_eval.sh 로 [재현 모델, 원본 허브 모델] 둘 다 GPU 0 순차 평가
set -uo pipefail
REPO=$HOME/Safety-WaRP-LLM; HB=$HOME/HarmBench
cd "$REPO"
export PY=$HOME/.conda/envs/hb/bin/python
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
CELL=$REPO/outputs/revision_repro/cb/llama32_3b/math/safelora
ORIG=kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4
HFREPO=${ORIG}_repro
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
TP=$(awk '{print $2}' logs/revision_repro/train.pid)
log "학습 종료 대기 (pid $TP)"
while kill -0 "$TP" 2>/dev/null; do sleep 60; done
log "학습 프로세스 종료. rc=$(cat logs/revision_repro/train.rc 2>/dev/null || echo '?')"
[[ -f "$CELL/.done" ]] || { log "❌ $CELL/.done 없음 — 학습 실패. 중단."; exit 1; }
log "════ 1. HF 업로드 → $HFREPO (private, 로컬 유지)"
"$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$HFREPO" --private || { log "❌ 업로드/검증 실패"; exit 1; }
log "════ 2. 평가 (GPU 0 순차: HarmBench sys 4공격 → lm-eval hendrycks_math_safe) — 재현 + 원본"
cd "$HB"
HB_GPU=0 LM_GPU=0 PARALLEL=0 RESUME=true VALIDATE_REPOS=1 HB_VARIANTS=sys SEED=42 ./run_all_eval.sh "$HFREPO" "$ORIG"
rc=$?
log "════ 완료 rc=$rc"
echo "$rc" > "$REPO/logs/revision_repro/eval.rc"
