#!/usr/bin/env bash
# sweep 에서 네트워크 등으로 실패한 평가 조합을 RESUME=true 로 메운다. 업로드 마커(.uploaded)가 있는 sweep 리포 전부를 넣는다.
set -uo pipefail
REPO=$HOME/Safety-WaRP-LLM; HB=$HOME/HarmBench; cd "$REPO"
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
REPOS=()
for cfg in "wsr_lora rho0.4 KEEP_RATIO=0.4 WSR_LORA_ALPHA=16" "wsr_lora rho0.5 KEEP_RATIO=0.5 WSR_LORA_ALPHA=16" "safelora thr0.2 SAFELORA_THRESHOLD=0.2 LORA_ALPHA=16" "safelora thr0.25 SAFELORA_THRESHOLD=0.25 LORA_ALPHA=16"; do
  set -- $cfg; m=$1; tag=$2; shift 2
  for d in outputs/revision_sweep/${m}_${tag}_a16/cb/*/*/$m; do
    [ -f "$d/.uploaded" ] || continue
    mk=$(basename "$(dirname "$(dirname "$d")")"); task=$(basename "$(dirname "$d")")
    REPOS+=("$(env "$@" bash -c "source scripts/revision/common.sh >/dev/null 2>&1; model_cfg $mk; hf_repo_id cb $mk $task $m")")
  done
done
echo "gap-fill 대상 ${#REPOS[@]}개:"; printf '  %s\n' "${REPOS[@]}"
cd "$HB"
HB_GPU=0 LM_GPU=0 PARALLEL=0 GPU_UTIL_CAP=0.85 RESUME=true LMEVAL_RESUME=1 VALIDATE_REPOS=1 HB_VARIANTS=sys SEED=42 PREFETCH_MODE=blocking ./run_all_eval.sh "${REPOS[@]}"
