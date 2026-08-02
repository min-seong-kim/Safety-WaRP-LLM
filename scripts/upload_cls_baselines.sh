#!/usr/bin/env bash
#
# run_cls_baselines.sh 산출물(태스크당 4개 × 2태스크 = 8개)을 HF Hub 에 업로드한다.
#
# repo 이름은 **디렉토리 이름에서 유도**한다. 손으로 적다가 모델과 이름이 어긋나는 사고를
# 막기 위함이다(과거에 실제로 있었던 실수).
#
#   {task}_fullft_lr1e-5_ep1                  -> llama2_7b-chat-{task}-fullft-lr1e-5-ep1
#   {task}_safeinstr0.1_lr1e-5_ep1            -> llama2_7b-chat-{task}-safeinstr0.1-lr1e-5-ep1-cb
#   {task}_fullft_lr1e-5_ep1-SafeDelta-s0.4   -> llama2_7b-chat-{task}-safedelta-lr1e-5-ep1-cb-s0.4
#   {task}_resta_gamma0.5_lr1e-5_ep1          -> llama2_7b-chat-{task}-resta-lr1e-5-ep1-gamma0.5
#
# ⚠️ transformers 5.x/4.x 는 chat template 을 별도 `chat_template.jinja` 로 저장한다.
#    이 파일이 빠지면 허브에서 받은 모델이 chat_template=None 이 되어 평가 포맷이 어긋난다
#    (WaRP 업로드에서 실제로 발생). 업로드 전/후로 존재를 검증한다.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/cls_baselines}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
TASKS="${TASKS:-agnews sst2}"
LR="${LR:-1e-5}"; EPOCHS="${EPOCHS:-1}"
SAFEINSTR_RATIO="${SAFEINSTR_RATIO:-0.1}"
SAFEDELTA_SCALE="${SAFEDELTA_SCALE:-0.4}"
RESTA_GAMMA="${RESTA_GAMMA:-0.5}"
DRY_RUN="${DRY_RUN:-0}"
MODES="${MODES:-baseline safeinstr safedelta resta}"   # 올릴 종류 선택

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
exec > >(tee -a "logs/upload_cls_baselines_${TS}.log") 2>&1

declare -a SRC=() REPO=()
for task in $TASKS; do
  for mode in $MODES; do
    case "$mode" in
      baseline)
        SRC+=("$OUT_ROOT/${task}_fullft_lr${LR}_ep${EPOCHS}")
        REPO+=("${HF_NAMESPACE}/llama2_7b-chat-${task}-fullft-lr${LR}-ep${EPOCHS}") ;;
      safeinstr)
        SRC+=("$OUT_ROOT/${task}_safeinstr${SAFEINSTR_RATIO}_lr${LR}_ep${EPOCHS}")
        REPO+=("${HF_NAMESPACE}/llama2_7b-chat-${task}-safeinstr${SAFEINSTR_RATIO}-lr${LR}-ep${EPOCHS}-cb") ;;
      safedelta)
        SRC+=("$OUT_ROOT/${task}_fullft_lr${LR}_ep${EPOCHS}-SafeDelta-s${SAFEDELTA_SCALE}")
        REPO+=("${HF_NAMESPACE}/llama2_7b-chat-${task}-safedelta-lr${LR}-ep${EPOCHS}-cb-s${SAFEDELTA_SCALE}") ;;
      resta)
        SRC+=("$OUT_ROOT/${task}_resta_gamma${RESTA_GAMMA}_lr${LR}_ep${EPOCHS}")
        REPO+=("${HF_NAMESPACE}/llama2_7b-chat-${task}-resta-lr${LR}-ep${EPOCHS}-gamma${RESTA_GAMMA}") ;;
      *) echo "알 수 없는 mode: $mode (baseline|safeinstr|safedelta|resta)" >&2; exit 1 ;;
    esac
  done
done

echo "════════ upload plan (${#SRC[@]}) ════════"
for i in "${!SRC[@]}"; do
  status="✗ 없음"
  [[ -f "${SRC[$i]}/config.json" ]] && status="✓"
  printf '  %s %-70s -> %s\n' "$status" "$(basename "${SRC[$i]}")" "${REPO[$i]}"
done
echo ""
[[ "$DRY_RUN" == "1" ]] && { echo "DRY_RUN=1 — 업로드하지 않음"; exit 0; }

failed=()
for i in "${!SRC[@]}"; do
  src="${SRC[$i]}"; repo="${REPO[$i]}"
  if [[ ! -f "$src/config.json" ]]; then
    echo "[skip] $src 없음"; failed+=("$(basename "$src")/missing"); continue
  fi
  echo "──────── uploading $(basename "$src") -> $repo ────────"
  "$PY" - "$src" "$repo" <<'PYEOF'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi

src, repo = Path(sys.argv[1]), sys.argv[2]

# 업로드 전 chat_template 존재 검증
has_jinja = (src / "chat_template.jinja").exists()
tc = src / "tokenizer_config.json"
has_inline = tc.exists() and "chat_template" in json.loads(tc.read_text())
if not (has_jinja or has_inline):
    raise SystemExit(f"ERROR: {src} 에 chat_template 이 없습니다. 업로드 중단.")

api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    folder_path=str(src), repo_id=repo, repo_type="model",
    ignore_patterns=["*.log", "wandb/*", ".wandb/*", "checkpoint-*/*", "runs/*"],
    commit_message="upload full-param baseline / defense model",
)

# 업로드 후 실제로 허브에 반영됐는지 재검증
files = set(api.list_repo_files(repo))
n_shards = sum(1 for f in files if f.endswith(".safetensors"))
tpl_ok = "chat_template.jinja" in files
if not tpl_ok and "tokenizer_config.json" in files:
    from huggingface_hub import hf_hub_download
    tpl_ok = "chat_template" in json.loads(
        Path(hf_hub_download(repo, "tokenizer_config.json", force_download=True)).read_text())
print(f"  -> {repo}  shards={n_shards}  chat_template={tpl_ok}")
if n_shards == 0 or not tpl_ok:
    raise SystemExit(f"ERROR: 업로드 검증 실패 (shards={n_shards}, chat_template={tpl_ok})")
PYEOF
  if [[ $? -ne 0 ]]; then echo "[FAIL] $repo"; failed+=("$repo"); else echo "[OK] $repo"; fi
done

echo ""
echo "════════ summary ════════"
if [[ ${#failed[@]} -gt 0 ]]; then echo "실패: ${failed[*]}"; exit 1; fi
echo "전부 업로드 완료 (${#SRC[@]}개)."
