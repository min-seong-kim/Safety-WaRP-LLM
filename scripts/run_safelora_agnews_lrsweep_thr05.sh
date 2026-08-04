#!/usr/bin/env bash
#
# AG News SafeLoRA lr 스윕 (threshold 0.5 고정, epoch 3) — 학습 → 평가 → 업로드.
#
#   lr: 1e-6 3e-6 5e-6 7e-6 9e-6 1e-5 3e-5 5e-5 7e-5 9e-5   (10 점)
#   threshold 0.5 = 이 레포의 agnews run 에서 160개 LoRA 레이어 중 약 124개가 투영되는 지점.
#
# ⚠️ GPU 0 만 사용한다 (요청). 따라서 lr 을 순차 실행한다.
#
# 하이퍼파라미터는 기존 러너(scripts/run_safelora_thr_sweep_qa.sh)를 그대로 호출해서 쓴다 —
# 이미 업로드된 thr0.5 모델들(lr 3e-5 / 5e-5)과 매칭을 깨뜨리지 않기 위함이다:
#   r16/α32/dropout0.05, target q,k,v,up,down, batch 16×1, max_len 1024,
#   warmup 0.03, wd 0, seed 42, bf16, 출발점 kmseong/llama2_7b-chat-Safety-FT-lr5e-5.
#
# 이미 허브에 있는 lr 점(3e-5, 5e-5)은 **재학습하지 않는다.** 같은 러너·같은 출발점으로 만들어진
# 것이 로그로 확인되므로, 재학습하면 같은 모델을 덮어쓸 뿐이다. 대신 허브에서 받아 평가만 한다.
#
# 디스크: 모델 1개 ≈ 13.5GB × 8개 신규 = 108GB. 업로드 검증 통과분은 로컬에서 지운다.
#
# 사용:
#   bash scripts/run_safelora_agnews_lrsweep_thr05.sh
#   LRS="1e-6 3e-6" bash scripts/run_safelora_agnews_lrsweep_thr05.sh
#   WAIT_FOR="run_safelora_agnews_thr_ep3.sh" bash scripts/run_safelora_agnews_lrsweep_thr05.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-/home/edgeai_lab/miniconda3/envs/hb/bin/python}"
export PY
export PATH="$(dirname "$PY"):$PATH"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
SAFELORA_BASE_MODEL="${SAFELORA_BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
TASK="${TASK:-agnews}"
THR="${THR:-0.5}"
EPOCHS="${EPOCHS:-3}"
LRS="${LRS:-1e-6 3e-6 5e-6 7e-6 9e-6 1e-5 3e-5 5e-5 7e-5 9e-5}"
GPU="${GPU:-0}"                       # ⚠️ 요청대로 GPU 0 만 사용
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
DO_UPLOAD="${DO_UPLOAD:-1}"
DO_EVAL="${DO_EVAL:-1}"
DELETE_AFTER="${DELETE_AFTER:-1}"     # 업로드 검증 성공 시 로컬 삭제 (디스크)
MIN_FREE_GB="${MIN_FREE_GB:-40}"
WAIT_FOR="${WAIT_FOR:-}"              # 이 패턴의 프로세스가 끝날 때까지 대기 후 시작

OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/safelora_${TASK}_lrsweep_ep${EPOCHS}_thr${THR}}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_DIR/evaluation_results/safelora_${TASK}_lrsweep_ep${EPOCHS}_thr${THR}}"
# 앞선 threshold 스윕에서 이미 평가한 결과가 있으면 재활용한다 (13.5GB 재다운로드 회피).
PRIOR_EVAL_DIR="${PRIOR_EVAL_DIR:-$REPO_DIR/evaluation_results/safelora_${TASK}_thr_ep${EPOCHS}}"
AGNEWS_TEST="${AGNEWS_TEST:-$REPO_DIR/data/agnews_test_1k_seed42.json}"
SST2_TEST="${SST2_TEST:-$REPO_DIR/data/sst2_validation_full.json}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUT_ROOT" "$EVAL_ROOT/raw"
exec > >(tee -a "logs/safelora_${TASK}_lrsweep_thr${THR}_${TS}.log") 2>&1

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
hr()      { echo "────────────────────────────────────────────────────────────────"; }

repo_of()   { echo "${HF_NAMESPACE}/llama2_7b-chat-${TASK}-safelora-r16-a32-lr${1}-ep${EPOCHS}-cb-thr${THR}"; }
merged_of() { echo "$OUT_ROOT/${TASK}_lr${1}_thr${THR}/merged_model"; }
tag_of()    { echo "${TASK}_safelora_lr${1}_ep${EPOCHS}_thr${THR}"; }

# 앞선 run 이 GPU 를 쓰고 있으면 끝날 때까지 기다린다 (같은 GPU 를 동시에 쓰지 않기 위함).
if [[ -n "$WAIT_FOR" ]]; then
  while pgrep -f "$WAIT_FOR" > /dev/null 2>&1; do sleep 30; done
  echo "선행 작업($WAIT_FOR) 종료 확인 — 시작 $(date +%H:%M:%S)"
fi

echo "════════════════════════════════════════════════════════════════"
echo " SafeLoRA ${TASK} lr sweep (thr=${THR} 고정)  ts=${TS}"
echo "   start model : $MODEL"
echo "   safelora base(C 계산용): $SAFELORA_BASE_MODEL"
echo "   lrs         : $LRS"
echo "   epochs      : $EPOCHS   GPU: $GPU (단일)"
echo "   output      : $OUT_ROOT      free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

failed=()

# 허브에 이미 완성된 모델이 있는지 확인 (shards 3개 + chat_template).
repo_ready() {   # $1=repo
  "$PY" - "$1" <<'PYEOF' > /dev/null 2>&1
import sys
from huggingface_hub import HfApi
files = set(HfApi().list_repo_files(sys.argv[1]))
n = sum(1 for f in files if f.endswith(".safetensors"))
ok = n >= 1 and ("chat_template.jinja" in files or "tokenizer_config.json" in files)
sys.exit(0 if ok else 1)
PYEOF
}

eval_model() {   # $1=model(경로 또는 HF id) $2=tag
  local model=$1 tag=$2 slug dst
  dst="$EVAL_ROOT/${tag}.json"
  [[ -f "$dst" ]] && { echo "  [eval] $tag 이미 완료 — skip"; return 0; }
  # 이전 스윕에서 같은 모델을 이미 평가했으면 그 결과를 가져온다.
  if [[ -f "$PRIOR_EVAL_DIR/${tag}.json" ]]; then
    cp "$PRIOR_EVAL_DIR/${tag}.json" "$dst"
    echo "  [eval] $tag 이전 평가 결과 재사용 ($PRIOR_EVAL_DIR)"
    return 0
  fi
  echo "  [eval] $tag (GPU $GPU) $(date +%H:%M:%S)"
  GPU="$GPU" PYTHON_BIN="$PY" OUTPUT_ROOT="$EVAL_ROOT/raw" \
    bash agnews_eval/run_agnews_sst2_eval.sh "$model" \
        --task "$TASK" \
        --agnews-data "$AGNEWS_TEST" --sst2-data "$SST2_TEST" \
        > "$EVAL_ROOT/raw/${tag}.log" 2>&1 \
    || { echo "  [eval] $tag FAILED — $EVAL_ROOT/raw/${tag}.log"; return 1; }
  slug="$("$PY" -c "import re,sys;s=re.sub(r'[^A-Za-z0-9._-]+','__',sys.argv[1].strip());print(s.strip('._-') or 'model')" "$model")"
  [[ -f "$EVAL_ROOT/raw/$slug/summary.json" ]] \
    || { echo "  [eval] $tag summary.json 없음"; return 1; }
  cp "$EVAL_ROOT/raw/$slug/summary.json" "$dst"
  "$PY" -c "
import json,sys
s=json.load(open(sys.argv[1]))
for t,v in s.items():
    print(f\"  [eval] {sys.argv[2]}  {t}: acc={v['accuracy']:.4f} macro_f1={v['macro_f1']:.4f} invalid={v['invalid_count']}\")
" "$dst" "$tag"
}

# ⚠️ chat_template.jinja 가 빠지면 허브 모델이 chat_template=None 이 되어 평가 프롬프트가
#    학습과 달라진다(이 레포에서 실제로 났던 사고). 업로드 전/후로 검증한다.
upload_model() {   # $1=local_dir $2=repo
  local src=$1 repo=$2
  echo "  [upload] -> $repo $(date +%H:%M:%S)"
  "$PY" - "$src" "$repo" <<'PYEOF'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi

src, repo = Path(sys.argv[1]), sys.argv[2]
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
    commit_message="SafeLoRA thr0.5 lr sweep — merged model",
)

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
}

# ═══════════════════ lr 루프 (GPU 0 단일, 순차) ═══════════════════
for lr in $LRS; do
  echo ""
  echo "████████████████ lr=$lr (thr=$THR, ep=$EPOCHS) ████████████████  $(date +%H:%M:%S)  free=$(free_gb)GB"
  merged="$(merged_of "$lr")"
  repo="$(repo_of "$lr")"
  tag="$(tag_of "$lr")"

  # ── 이미 허브에 있는 lr 점은 재학습/재업로드하지 않고 평가만 한다.
  if [[ ! -f "$merged/config.json" ]] && repo_ready "$repo"; then
    echo "[lr $lr] 허브에 이미 존재 — 학습/업로드 skip, 평가만 수행 ($repo)"
    [[ "$DO_EVAL" == "1" ]] && { eval_model "$repo" "$tag" || failed+=("eval/lr$lr"); }
    continue
  fi

  # ── 1) 학습 + 사후 투영
  if [[ -f "$merged/config.json" ]]; then
    echo "[lr $lr] 학습 이미 완료 — skip"
  else
    avail=$(free_gb)
    if [[ "$avail" -lt "$MIN_FREE_GB" ]]; then
      echo "[lr $lr] SKIP — 디스크 부족 (${avail}GB < ${MIN_FREE_GB}GB)"; failed+=("disk/lr$lr"); continue
    fi
    hr; echo "[lr $lr] SafeLoRA 학습 $(date +%H:%M:%S)"; hr
    CUDA_VISIBLE_DEVICES="$GPU" MODEL="$MODEL" SAFELORA_BASE_MODEL="$SAFELORA_BASE_MODEL" \
    TASKS="$TASK" THRS="$THR" LR="$lr" EPOCHS="$EPOCHS" \
    OUTPUT_ROOT="$OUT_ROOT" PUSH_TO_HUB=0 PY="$PY" \
      bash scripts/run_safelora_thr_sweep_qa.sh \
      || { echo "[lr $lr] 학습 FAILED"; failed+=("train/lr$lr"); continue; }
  fi
  [[ -f "$merged/config.json" ]] || { echo "[lr $lr] merged_model 없음"; failed+=("merged/lr$lr"); continue; }

  # 투영 레이어 수는 학습 산출물 summary.json 에만 있는데, 아래에서 로컬 디렉토리를 지우므로
  # 최종 표가 그 값을 잃는다. 삭제 전에 평가 결과 옆으로 복사해 둔다.
  cp "$(dirname "$merged")/summary.json" "$EVAL_ROOT/${tag}.safelora.json" 2>/dev/null || true

  # ── 2) 평가 (업로드/삭제 전에 로컬 모델로)
  eval_ok=1
  if [[ "$DO_EVAL" == "1" ]]; then
    eval_model "$merged" "$tag" || { failed+=("eval/lr$lr"); eval_ok=0; }
  fi

  # ── 3) 업로드 + 로컬 정리 (평가 실패 시에는 지우지 않는다 — 재평가에 재다운로드가 필요해진다)
  if [[ "$DO_UPLOAD" == "1" ]]; then
    if upload_model "$merged" "$repo"; then
      if [[ "$DELETE_AFTER" == "1" && "$eval_ok" == "1" ]]; then
        echo "  rm -rf $(dirname "$merged")"; rm -rf "$(dirname "$merged")"
      elif [[ "$DELETE_AFTER" == "1" ]]; then
        echo "  평가 실패로 로컬 보존"
      fi
    else
      echo "[lr $lr] 업로드 FAILED — 로컬 보존"; failed+=("upload/lr$lr")
    fi
  fi
done

# ═══════════════════ summary ═══════════════════
echo ""
echo "════════════ SafeLoRA lr sweep (${TASK}, thr ${THR}, ep ${EPOCHS}) ════════════"
"$PY" - "$EVAL_ROOT" "$OUT_ROOT" "$TASK" "$THR" <<'PYEOF'
import json, re, sys
from pathlib import Path

eval_root, out_root, task, thr = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]

def lr_key(s):
    return float(s)          # "3e-5" 같은 문자열은 그대로 float 로 읽힌다

rows = []
for p in sorted(eval_root.glob("*.json")):
    m = re.search(r"_lr(?P<lr>[0-9.]+e-?[0-9]+)_ep", p.stem)
    if not m:
        continue
    lr = m["lr"]
    s = json.load(open(p))
    proj = "-"
    # 업로드 후 로컬을 지우므로, 삭제 전에 복사해 둔 사본을 먼저 본다.
    sj = eval_root / f"{p.stem}.safelora.json"
    if not sj.is_file():
        sj = out_root / f"{task}_lr{lr}_thr{thr}" / "summary.json"
    if sj.is_file():
        sl = json.load(open(sj)).get("safelora") or {}
        if sl.get("num_layers_projected") is not None:
            proj = f"{sl['num_layers_projected']}/{sl['num_lora_layers']}"
    for t, v in s.items():
        rows.append((lr_key(lr), lr, proj, v["accuracy"], v["macro_f1"],
                     v.get("invalid_count", 0), v.get("samples", "")))

if not rows:
    print("  (평가 결과 없음)")
else:
    rows.sort(key=lambda r: r[0])
    print(f"  {'lr':<8} {'투영 레이어':<12} {'acc':>8} {'macro_f1':>9} {'invalid':>8} {'n':>6}")
    print("  " + "-" * 56)
    for _, lr, proj, acc, f1, inv, n in rows:
        print(f"  {lr:<8} {proj:<12} {acc:>8.4f} {f1:>9.4f} {inv:>8} {str(n):>6}")
print(f"\n  raw: {eval_root}")
PYEOF

echo "  free: $(free_gb)GB"
if [[ ${#failed[@]} -gt 0 ]]; then echo "실패: ${failed[*]}"; exit 1; fi
echo "완료."
