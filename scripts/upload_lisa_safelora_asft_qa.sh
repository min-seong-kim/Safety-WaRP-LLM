#!/usr/bin/env bash
#
# run_lisa_safelora_asft_qa.sh 결과물을 HF Hub 에 업로드한다.
# scripts/upload_lisa_safelora_cls.sh 와 동일한 규약 — task/method 만 확장.
#
# repo 이름 규약:
#   ${HF_NS}/llama2_7b-chat-{task}-{method}-r16-a32-lr{LR}-cb
#     task   = arc | medqa   (cls 쪽은 sst2 | agnews)
#     method = lisa | safelora | asft
#     -cb    = safety data / alignment delta 가 circuit_breakers 계열임을 표시
#
# 모델 위치가 method 마다 다르다:
#   LISA             → <out_dir>/                (merge_and_unload 후 output_dir 직하)
#   SafeLoRA / AsFT  → <out_dir>/merged_model/   (merge_and_unload 결과)
#
# 사용:
#   bash scripts/upload_lisa_safelora_asft_qa.sh
#   TASKS=arc METHODS=asft bash scripts/upload_lisa_safelora_asft_qa.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
HF_NS="${HF_NS:-kmseong}"
LR="${LR:-3e-4}"
TASKS="${TASKS:-arc medqa}"
METHODS="${METHODS:-lisa safelora asft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/lisa_safelora_asft_qa}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
exec > >(tee -a "logs/upload_lisa_safelora_asft_qa_${TS}.log") 2>&1

echo "업로드 대상: ${METHODS} × ${TASKS} @ lr${LR} → ${HF_NS}/*"

fail=0
for task in $TASKS; do
  for method in $METHODS; do
    out_dir="$OUTPUT_ROOT/${method}/${task}_lr${LR}"
    if [[ "$method" == "lisa" ]]; then
      src="$out_dir"
    else
      src="$out_dir/merged_model"
    fi
    repo="${HF_NS}/llama2_7b-chat-${task}-${method}-r16-a32-lr${LR}-cb"

    if [[ ! -f "$src/config.json" ]]; then
      echo "[$method/$task] SKIP — 모델 없음: $src"; fail=1; continue
    fi
    # 학습 산출물 디렉토리를 통째로 올리면 run.log / trainer 체크포인트까지 딸려간다.
    # 가중치+토크나이저+설정만 골라 올린다.
    echo "──────────────────────────────────────────────"
    echo "[$method/$task] $src → $repo"
    "$PY" - "$src" "$repo" <<'EOF'
import sys
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    folder_path=src, repo_id=repo, repo_type="model",
    allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "tokenizer*", "*.txt", "*.jinja"],
    ignore_patterns=["trainer/*", "checkpoint-*/*", "*.log", "cache/*"],
)
info = api.model_info(repo)
print(f"OK: https://huggingface.co/{repo}  (files: {len(info.siblings)})")
EOF
  done
done

echo ""
[[ "$fail" == "0" ]] && echo "전부 업로드 완료." || { echo "일부 실패 — 위 SKIP 확인"; exit 1; }
