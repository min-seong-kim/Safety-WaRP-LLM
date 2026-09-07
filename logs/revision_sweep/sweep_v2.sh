#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Sweep v2 (2026-09-05): v1 과 같은 24셀이지만 GPU 를 고정하지 않고 "여유 메모리가 가장 큰 GPU" 를 스테이지마다 고른다.
#    - 학습: 여유 ≥ TRAIN_NEED GB 인 GPU 를 기다렸다가 사용
#    - HarmBench: models.yaml util 그대로(가족 기본값) → util×97.9+3 GB 여유 GPU 를 기다림
#    - lm-eval: 같은 GPU, GPU_UTIL_CAP 은 그 시점 여유 메모리에 맞춰 0.85/0.7/0.6/0.5 중 최대 (원본은 gsm8k 0.85, math 0.4)
#    끝난 셀(.done/.uploaded)과 완료된 평가 조합(RESUME=true, LMEVAL_RESUME=1)은 건너뛴다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO=$HOME/Safety-WaRP-LLM; HB=$HOME/HarmBench; cd "$REPO"
export PY=$HOME/.conda/envs/hb_repro/bin/python
export PATH=$HOME/.conda/envs/hb_repro/bin:$PATH
export TRITON_CACHE_DIR=$HOME/.triton/cache_hb_repro TORCHINDUCTOR_CACHE_DIR=$HOME/.torchinductor_cache_hb_repro
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export SAFETY_SETS=cb PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=1 PRUNE_BASIS=0 PRUNE_HF_CACHE=0 CONTINUE_ON_ERROR=1
export BASE_BLOCKED_MODELS=""
SW=$REPO/outputs/revision_sweep
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 여유 메모리(GiB)가 need 이상인 GPU 중 가장 여유 큰 것을 고른다. 없으면 60초마다 재확인(30분마다 로그).
wait_gpu() {  # <need_gib>  → stdout: gpu index
  local need=$1 n=0 best bestfree line idx free
  while true; do
    best=""; bestfree=0
    while IFS=, read -r idx free; do
      idx=${idx// /}; free=$(( ${free// /} / 1024 ))
      if [ "$free" -gt "$bestfree" ]; then best=$idx; bestfree=$free; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    if [ -n "$best" ] && [ "$bestfree" -ge "$need" ]; then echo "$best"; return 0; fi
    n=$((n+1)); [ $((n % 30)) -eq 1 ] && log "   ⏳ GPU 대기: 필요 ${need}GiB, 최대 여유 GPU${best:-?} ${bestfree}GiB" >&2
    sleep 60
  done
}
gpu_free_gib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | head -1 | awk '{printf "%d", $1/1024}'; }

declare -A TASK=([llama32_3b]=math [llama2_7b]=gsm8k [llama31_8b]=math [qwen25_7b]=gsm8k [gemma2_9b]=gsm8k [llama2_13b]=gsm8k)
declare -A HBUTIL=([llama32_3b]=0.5 [llama2_7b]=0.85 [llama31_8b]=0.6 [qwen25_7b]=0.6 [gemma2_9b]=0.6 [llama2_13b]=0.7)
declare -A TRAIN_NEED=([llama32_3b]=30 [llama2_7b]=40 [llama31_8b]=45 [qwen25_7b]=45 [gemma2_9b]=50 [llama2_13b]=60)
CFGS=("wsr_lora rho0.4 KEEP_RATIO=0.4 WSR_LORA_ALPHA=16"
      "wsr_lora rho0.5 KEEP_RATIO=0.5 WSR_LORA_ALPHA=16"
      "safelora thr0.2 SAFELORA_THRESHOLD=0.2 LORA_ALPHA=16"
      "safelora thr0.25 SAFELORA_THRESHOLD=0.25 LORA_ALPHA=16")
MODELS_ORDER="${MODELS_ORDER:-llama32_3b llama2_7b llama31_8b qwen25_7b gemma2_9b llama2_13b}"

run_eval() {  # <model> <repo...>
  local mk=$1; shift; local repos=("$@")
  local hbneed; hbneed=$(awk -v u="${HBUTIL[$mk]}" 'BEGIN{printf "%d", u*97.9+4}')
  local g; g=$(wait_gpu "$hbneed")
  local free; free=$(gpu_free_gib "$g")
  local cap=0.5; for c in 0.85 0.7 0.6; do if [ "$free" -ge "$(awk -v u=$c 'BEGIN{printf "%d", u*97.9+2}')" ]; then cap=$c; break; fi; done
  log "── 평가  $mk on GPU$g (free ${free}GiB, HB util ${HBUTIL[$mk]}, LM cap $cap): ${repos[*]}"
  ( cd "$HB" && HB_GPU=$g LM_GPU=$g PARALLEL=0 GPU_UTIL_CAP=$cap RESUME=true LMEVAL_RESUME=1 VALIDATE_REPOS=1 HB_VARIANTS=sys SEED=42 \
      ./run_all_eval.sh "${repos[@]}" ) || log "   ⚠️  $mk 평가 배치에 실패한 조합이 있다(계속)"
}

log "sweep v2 시작 (GPU 자동 선택)"
for mk in $MODELS_ORDER; do
  task=${TASK[$mk]}
  echo; echo "████████████████████████  $mk / $task   ($(date '+%m-%d %H:%M:%S'))  ████████████████████████"
  REPOS=(); ALL_DONE=1
  for cfg in "${CFGS[@]}"; do
    set -- $cfg; m=$1; tag=$2; shift 2
    stages="20"; [ "$m" = wsr_lora ] && stages="02 20"
    out="$SW/${m}_${tag}_a16"; mkdir -p "$out"
    r=$(env "$@" bash -c "source scripts/revision/common.sh >/dev/null 2>&1; model_cfg $mk; hf_repo_id cb $mk $task $m")
    if [ -f "$out/cb/$mk/$task/$m/.uploaded" ]; then REPOS+=("$r"); log "   [skip] 이미 업로드됨: $r"; continue; fi
    ALL_DONE=0
    g=$(wait_gpu "${TRAIN_NEED[$mk]}")
    log "── 학습  $mk / $m $tag on GPU$g (free $(gpu_free_gib $g)GiB, 디스크 여유 $(df -BG --output=avail "$out" | tail -1 | tr -dc '0-9')G)"
    CUDA_VISIBLE_DEVICES=$g env "$@" MODELS="$mk" TASKS="$task" METHODS="$m" STAGES="$stages" OUT_ROOT="$out" bash scripts/revision/run_all.sh
    if [ -f "$out/cb/$mk/$task/$m/.uploaded" ]; then REPOS+=("$r"); log "   ✅ 업로드됨: $r"
    else log "   ⚠️  업로드 마커 없음(학습 실패/미완): $r — 평가에서 제외"; fi
  done
  [ ${#REPOS[@]} -gt 0 ] && run_eval "$mk" "${REPOS[@]}"
done

log "════ 학습/평가 1차 완료 → gap-fill (실패 조합 재실행) ════"
ALL=()
for cfg in "${CFGS[@]}"; do set -- $cfg; m=$1; tag=$2; shift 2
  for d in $SW/${m}_${tag}_a16/cb/*/*/$m; do [ -f "$d/.uploaded" ] || continue
    mk=$(basename "$(dirname "$(dirname "$d")")"); t=$(basename "$(dirname "$d")")
    ALL+=("$(env "$@" bash -c "source scripts/revision/common.sh >/dev/null 2>&1; model_cfg $mk; hf_repo_id cb $mk $t $m")"); done; done
g=$(wait_gpu 88); log "── gap-fill on GPU$g: ${#ALL[@]}개 리포 (RESUME)"
( cd "$HB" && HB_GPU=$g LM_GPU=$g PARALLEL=0 GPU_UTIL_CAP=0.85 RESUME=true LMEVAL_RESUME=1 VALIDATE_REPOS=1 HB_VARIANTS=sys SEED=42 PREFETCH_MODE=blocking ./run_all_eval.sh "${ALL[@]}" ) || log "   ⚠️  gap-fill 에도 실패 조합 있음"
log "════ sweep v2 전체 완료 ════"; echo done > "$REPO/logs/revision_sweep/sweep.rc"
