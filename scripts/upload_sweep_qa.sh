#!/usr/bin/env bash
#
# 하이퍼파라미터 스윕 결과물(LISA rho / SafeLoRA threshold)을 HF Hub 에 업로드.
#
#   outputs/lisa_rho_sweep_qa/{task}_lr{LR}_rho{RHO}/
#       → {NS}/llama2_7b-chat-{task}-lisa-r16-a32-lr{LR}-cb-rho{RHO}
#   outputs/safelora_thr_sweep_qa/{task}_lr{LR}_thr{THR}/merged_model/
#       → {NS}/llama2_7b-chat-{task}-safelora-r16-a32-lr{LR}-cb-thr{THR}
#
# repo 이름은 디렉토리 이름에서 유도한다 — 스윕 스크립트의 --upload_name / --hf_repo_id
# 규칙과 손으로 맞추면 어긋나기 쉽다.
#
# 모델 위치가 method 마다 다르다:
#   LISA     → <out_dir>/               (merge_and_unload 후 output_dir 직하)
#   SafeLoRA → <out_dir>/merged_model/  (사후 투영 후 dense fold)
#
# 사용:
#   bash scripts/upload_sweep_qa.sh
#   ONLY=lisa bash scripts/upload_sweep_qa.sh      # 한쪽만
set -uo pipefail          # -e 없음: 하나 실패해도 나머지를 계속 올린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
NS="${HF_NS:-kmseong}"
ONLY="${ONLY:-both}"      # lisa | safelora | both

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
exec > >(tee -a "logs/upload_sweep_qa_${TS}.log") 2>&1

upload() {   # $1=로컬 소스 디렉토리  $2=repo_id
    local src=$1 repo=$2
    if [ ! -f "$src/config.json" ]; then
        echo "  SKIP — 모델 없음: $src"; return 1
    fi
    echo "──────────────────────────────────────────────"
    echo "  $src"
    echo "  → $NS 네임스페이스: $repo"
    "$PY" - "$src" "$repo" <<'EOF'
import sys
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    folder_path=src, repo_id=repo, repo_type="model",
    allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "tokenizer*", "*.txt"],
    ignore_patterns=["trainer/*", "checkpoint-*/*", "*.log", "cache/*"],
)
info = api.model_info(repo)
print(f"  OK: https://huggingface.co/{repo}  (files: {len(info.siblings)})")
EOF
    return $?
}

ok=0; fail=0; failed=()

# ═══════════ LISA rho 스윕 ═══════════
if [ "$ONLY" = "both" ] || [ "$ONLY" = "lisa" ]; then
  echo "═══════════ LISA rho 스윕 ═══════════"
  for d in outputs/lisa_rho_sweep_qa/*/; do
    [ -f "$d/finetune_config.json" ] || { echo "미완료 건너뜀: $d"; continue; }
    name=$(basename "${d%/}")                       # arc_lr3e-4_rho0
    task=${name%%_*}                                # arc
    rest=${name#*_}                                 # lr3e-4_rho0
    lr=${rest%%_*}; lr=${lr#lr}                     # 3e-4
    rho=${rest##*_rho}                              # 0
    upload "${d%/}" "$NS/llama2_7b-chat-${task}-lisa-r16-a32-lr${lr}-cb-rho${rho}" \
      && ok=$((ok+1)) || { fail=$((fail+1)); failed+=("$name"); }
  done
fi

# ═══════════ SafeLoRA threshold 스윕 ═══════════
if [ "$ONLY" = "both" ] || [ "$ONLY" = "safelora" ]; then
  echo "═══════════ SafeLoRA threshold 스윕 ═══════════"
  for d in outputs/safelora_thr_sweep_qa/*/; do
    [ -f "$d/summary.json" ] || { echo "미완료 건너뜀: $d"; continue; }
    name=$(basename "${d%/}")                       # arc_lr3e-4_thr0.30
    task=${name%%_*}
    rest=${name#*_}                                 # lr3e-4_thr0.30
    lr=${rest%%_*}; lr=${lr#lr}
    thr=${rest##*_thr}
    upload "${d%/}/merged_model" "$NS/llama2_7b-chat-${task}-safelora-r16-a32-lr${lr}-cb-thr${thr}" \
      && ok=$((ok+1)) || { fail=$((fail+1)); failed+=("$name"); }
  done
fi

echo ""
echo "════════════════════ 결과 ════════════════════"
echo "  성공 $ok / 실패 $fail"
[ ${#failed[@]} -gt 0 ] && { echo "  실패: ${failed[*]}"; exit 1; }
echo "  전부 업로드 완료."
