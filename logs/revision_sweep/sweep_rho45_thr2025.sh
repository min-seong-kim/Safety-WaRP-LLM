#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Sweep (2026-09-04, hb_repro env = 원본 박스 라이브러리: torch 2.10 / transformers 4.57.3 / peft 0.18.1)
#    6 모델 × { WSR-LoRA ρ=0.4, 0.5 (α16) ; SafeLoRA thr=0.2, 0.25 (α16) } = 24 셀, primary task 만.
#    모델 단위로: 학습(→허브 업로드·로컬 삭제) 4셀 → 그 4개 리포 평가(HarmBench sys 4공격 + lm-eval) → 다음 모델.
#    GPU 0 단독(공유 GPU). OUT_ROOT 는 하이퍼파라미터별로 분리(out_dir 이 hparam 을 안 담기 때문).
#    basis(checkpoints/revision/warp/cb/<model>) 는 ρ 0.4/0.5 가 공유하므로 PRUNE_BASIS=0.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO=$HOME/Safety-WaRP-LLM; HB=$HOME/HarmBench; cd "$REPO"
export CUDA_VISIBLE_DEVICES=0
export PY=$HOME/.conda/envs/hb_repro/bin/python
export PATH=$HOME/.conda/envs/hb_repro/bin:$PATH
export TRITON_CACHE_DIR=$HOME/.triton/cache_hb_repro TORCHINDUCTOR_CACHE_DIR=$HOME/.torchinductor_cache_hb_repro
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export SAFETY_SETS=cb PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=1 PRUNE_BASIS=0 PRUNE_HF_CACHE=0 CONTINUE_ON_ERROR=1
export BASE_BLOCKED_MODELS=""   # gemma-2 base 접근 확인됨(2026-09-04) → SafeLoRA 허용
SW=$REPO/outputs/revision_sweep
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "선행 작업(3차 재현 학습, 2차 평가) 종료 대기"
for f in train3 eval2; do
  pid=$(awk '{print $2}' "$REPO/logs/revision_repro/$f.pid" 2>/dev/null || true)
  while [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
log "선행 종료 확인 — sweep 시작"

declare -A TASK=([llama32_3b]=math [llama2_7b]=gsm8k [llama31_8b]=math [qwen25_7b]=gsm8k [gemma2_9b]=gsm8k [llama2_13b]=gsm8k)
CFGS=("wsr_lora rho0.4 KEEP_RATIO=0.4 WSR_LORA_ALPHA=16"
      "wsr_lora rho0.5 KEEP_RATIO=0.5 WSR_LORA_ALPHA=16"
      "safelora thr0.2 SAFELORA_THRESHOLD=0.2 LORA_ALPHA=16"
      "safelora thr0.25 SAFELORA_THRESHOLD=0.25 LORA_ALPHA=16")
MODELS_ORDER="${MODELS_ORDER:-llama32_3b llama2_7b llama31_8b qwen25_7b gemma2_9b llama2_13b}"

for mk in $MODELS_ORDER; do
  task=${TASK[$mk]}
  echo; echo "████████████████████████  $mk / $task   ($(date '+%m-%d %H:%M:%S'))  ████████████████████████"
  REPOS=()
  for cfg in "${CFGS[@]}"; do
    set -- $cfg; m=$1; tag=$2; shift 2
    stages="20"; [ "$m" = wsr_lora ] && stages="02 20"
    out="$SW/${m}_${tag}_a16"; mkdir -p "$out"
    log "── 학습  $mk / $m $tag   (OUT_ROOT=$out, 여유 $(df -BG --output=avail "$out" | tail -1 | tr -dc '0-9')G)"
    env "$@" MODELS="$mk" TASKS="$task" METHODS="$m" STAGES="$stages" OUT_ROOT="$out" bash scripts/revision/run_all.sh
    r=$(env "$@" bash -c "source scripts/revision/common.sh >/dev/null 2>&1; model_cfg $mk; hf_repo_id cb $mk $task $m")
    if [ -f "$out/cb/$mk/$task/$m/.uploaded" ]; then REPOS+=("$r"); log "   ✅ 업로드됨: $r"
    else log "   ⚠️  업로드 마커 없음(학습 실패/미완): $r — 평가에서 제외"; fi
  done

  if [ ${#REPOS[@]} -eq 0 ]; then log "   $mk: 평가할 리포 없음"; continue; fi
  log "── 평가  $mk: ${REPOS[*]}"
  cd "$HB"
  if [ "$mk" = llama2_7b ]; then
    # llama2_7b family 기본 util 0.95 는 공유 GPU 에서 기동 불가(2026-09-04 실측) → 0.85 로 선등록
    for r in "${REPOS[@]}"; do
      $HOME/.conda/envs/harmbench/bin/python - "${r#*/}" "$r" 0.85 <<'PYEOF'
import sys, yaml, shutil, datetime
import add_models_to_yaml as A
key, repo, util = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = yaml.safe_load(open(A.YAML_PATH)) or {}
if key in d: print(f"이미 등록됨: {key}"); sys.exit(0)
fam = A.detect_family(key); df = A.FAMILY_DEFAULTS[fam]
has_w, use_fast, cfg_max = A.probe_repo(repo, "main")
if not has_w: print("가중치 없음:", repo); sys.exit(0)
block = A.build_block(key, repo, use_fast, df["dtype"], cfg_max or df["max_model_len"], df["chat_template"], util, df["num_gpus"])
shutil.copy2(A.YAML_PATH, f"{A.YAML_PATH}.bak_{datetime.datetime.now():%Y%m%d_%H%M%S}")
open(A.YAML_PATH, "a").write("\n" + block + "\n"); print("등록(util", util, "):", key)
PYEOF
    done
  fi
  HB_GPU=0 LM_GPU=0 PARALLEL=0 RESUME=true VALIDATE_REPOS=1 HB_VARIANTS=sys SEED=42 ./run_all_eval.sh "${REPOS[@]}" \
    || log "   ⚠️  $mk 평가 배치에 실패한 조합이 있다(계속)"
  cd "$REPO"
done
log "════ sweep 전체 완료 ════"
echo done > "$REPO/logs/revision_sweep/sweep.rc"
