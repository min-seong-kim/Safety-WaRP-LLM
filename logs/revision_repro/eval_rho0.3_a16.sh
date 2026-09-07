#!/usr/bin/env bash
# 재현 실험 후속: 학습 종료 대기 → HF 업로드(_repro 접미사, private) → models.yaml 선등록(util 0.8)
#                → run_all_eval.sh (HarmBench 4공격 + lm-eval gsm8k) 를 GPU 0 에서 순차 실행
set -uo pipefail
REPO=$HOME/Safety-WaRP-LLM; HB=$HOME/HarmBench
cd "$REPO"
export PY=$HOME/.conda/envs/hb/bin/python
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
CELL=$REPO/outputs/revision_repro/cb/llama2_7b/gsm8k/wsr_lora
HFREPO=kmseong/llama2_7b-chat-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4_repro
KEY=${HFREPO#*/}
HB_UTIL=${HB_UTIL:-0.8}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "학습 종료 대기 (pid $(cat logs/revision_repro/train.pid | awk '{print $2}'))"
TP=$(awk '{print $2}' logs/revision_repro/train.pid)
while kill -0 "$TP" 2>/dev/null; do sleep 60; done
log "학습 프로세스 종료. rc=$(cat logs/revision_repro/train.rc 2>/dev/null || echo '?')"
[[ -f "$CELL/.done" ]] || { log "❌ $CELL/.done 없음 — 학습 실패. 중단."; exit 1; }
[[ -f "$CELL/wsrlora_run_config.json" ]] && log "run_config: $(python3 -c "import json;d=json.load(open('$CELL/wsrlora_run_config.json'));print({k:d[k] for k in ['alpha','freeze_ratio','learning_rate','epochs','effective_batch_size','scaling']})")"

log "════ 1. HF 업로드 → $HFREPO (private, 로컬 유지)"
"$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$HFREPO" --private || { log "❌ 업로드/검증 실패"; exit 1; }

log "════ 2. models.yaml 선등록 (gpu_memory_utilization=$HB_UTIL, 원본 0.95 는 공유 GPU 라 불가)"
cd "$HB"
$HOME/.conda/envs/harmbench/bin/python - "$KEY" "$HFREPO" "$HB_UTIL" <<'PYEOF'
import sys, yaml, shutil, datetime
import add_models_to_yaml as A
key, repo, util = sys.argv[1], sys.argv[2], float(sys.argv[3])
y = A.YAML_PATH
d = yaml.safe_load(open(y)) or {}
if key in d:
    print(f"이미 등록됨: {key} util={d[key]['model'].get('gpu_memory_utilization')}"); sys.exit(0)
fam = A.detect_family(key); df = A.FAMILY_DEFAULTS[fam]
has_w, use_fast, cfg_max = A.probe_repo(repo, "main")
assert has_w, "가중치 없음"
block = A.build_block(key, repo, use_fast, df["dtype"], cfg_max or df["max_model_len"], df["chat_template"], util, df["num_gpus"])
shutil.copy2(y, f"{y}.bak_{datetime.datetime.now():%Y%m%d_%H%M%S}")
open(y, "a").write("\n" + block + "\n")
print(block)
PYEOF

log "════ 3. 평가 (GPU 0 순차: HarmBench sys 4공격 → lm-eval gsm8k, util cap $HB_UTIL)"
HB_GPU=0 LM_GPU=0 PARALLEL=0 GPU_UTIL_CAP="$HB_UTIL" RESUME=true VALIDATE_REPOS=1 \
  HB_VARIANTS=sys SEED=42 ./run_all_eval.sh "$HFREPO"
rc=$?
log "════ 완료 rc=$rc"
echo "$rc" > "$REPO/logs/revision_repro/eval.rc"
