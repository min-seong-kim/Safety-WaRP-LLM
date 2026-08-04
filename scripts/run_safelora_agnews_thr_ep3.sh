#!/usr/bin/env bash
#
# AG News SafeLoRA threshold 스윕 (lr 3e-5, epoch 3) — thr 0.4 / 0.3 학습 → 평가 → 업로드.
#
#   기준점: kmseong/llama2_7b-chat-agnews-safelora-r16-a32-lr3e-5-ep3-cb-thr0.5 (이미 허브에 있음)
#   이번 추가: 같은 조건에서 threshold 만 0.4 / 0.3 으로 낮춘 두 모델.
#
# SafeLoRA 의 threshold 는 "cos((C·B)A, BA) ≤ threshold 인 레이어만 B ← C·B 로 투영"의 문턱이다.
# 낮출수록 투영되는 레이어가 줄어든다 = 안전 개입 축소 = safety↓ downstream↑ 방향.
# 이 레포의 agnews run 에서 관측된 cos 분포는 대략 0.18~0.70 이고, thr 0.5 에서 124/160 투영이었다.
#
# ⚠️ 학습 자체는 표준 LoRA 이고 threshold 는 **사후** 투영이라 thr 마다 학습이 동일하다.
#    러너가 투영 전 adapter 를 저장하지 않아 thr 마다 재학습한다(agnews 기준 ~6분이라 감수).
#
# 하이퍼파라미터는 기존 러너(scripts/run_safelora_thr_sweep_qa.sh)를 그대로 호출해서 쓴다 —
# thr0.5 모델과 매칭(r16/α32/dropout0.05, batch 16×1, warmup 0.03, wd 0, seed 42, max_len 1024)을
# 깨뜨리지 않기 위함이다.
#
# 사용:
#   bash scripts/run_safelora_agnews_thr_ep3.sh
#   THRS=0.4 bash scripts/run_safelora_agnews_thr_ep3.sh
#   EVAL_REF=1 bash scripts/run_safelora_agnews_thr_ep3.sh   # 기존 thr0.5 도 허브에서 받아 평가
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
THRS="${THRS:-0.4 0.3}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-3}"
GPU="${GPU:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
DO_UPLOAD="${DO_UPLOAD:-1}"
DO_EVAL="${DO_EVAL:-1}"
EVAL_REF="${EVAL_REF:-1}"            # 기존 thr0.5 모델도 평가해 3점 비교표를 만든다
REF_THR="${REF_THR:-0.5}"

OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/safelora_cls_lr${LR}_ep${EPOCHS}_thr}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_DIR/evaluation_results/safelora_${TASK}_thr_ep${EPOCHS}}"
AGNEWS_TEST="${AGNEWS_TEST:-$REPO_DIR/data/agnews_test_1k_seed42.json}"
SST2_TEST="${SST2_TEST:-$REPO_DIR/data/sst2_validation_full.json}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUT_ROOT" "$EVAL_ROOT/raw"
exec > >(tee -a "logs/safelora_${TASK}_thr_ep${EPOCHS}_${TS}.log") 2>&1

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
hr()      { echo "────────────────────────────────────────────────────────────────"; }

repo_of()  { echo "${HF_NAMESPACE}/llama2_7b-chat-${TASK}-safelora-r16-a32-lr${LR}-ep${EPOCHS}-cb-thr${1}"; }
merged_of(){ echo "$OUT_ROOT/${TASK}_lr${LR}_thr${1}/merged_model"; }

echo "════════════════════════════════════════════════════════════════"
echo " SafeLoRA ${TASK} threshold sweep  ts=${TS}"
echo "   start model : $MODEL"
echo "   safelora base(C 계산용): $SAFELORA_BASE_MODEL"
echo "   thresholds  : $THRS      lr: $LR   epochs: $EPOCHS   GPU: $GPU"
echo "   기준점      : $(repo_of "$REF_THR")  (이미 업로드됨)"
echo "   output      : $OUT_ROOT      free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

failed=()

# ── 평가 헬퍼 (agnews_eval 하네스를 그대로 사용) ──────────────────────────
# evaluate_agnews_sst2.py 의 기본 데이터 경로는 이 박스에 없는 레이아웃이라 명시적으로 넘긴다.
# --task 명시: 경로에 태스크명이 없으면 auto 추론이 두 태스크를 다 돌려버린다.
eval_model() {   # $1=model(경로 또는 HF id) $2=tag
  local model=$1 tag=$2 slug dst
  dst="$EVAL_ROOT/${tag}.json"
  [[ -f "$dst" ]] && { echo "  [eval] $tag 이미 완료 — skip"; return 0; }
  echo "  [eval] $tag (GPU $GPU) $(date +%H:%M:%S)"
  GPU="$GPU" PYTHON_BIN="$PY" OUTPUT_ROOT="$EVAL_ROOT/raw" \
    bash agnews_eval/run_agnews_sst2_eval.sh "$model" \
        --task "$TASK" \
        --agnews-data "$AGNEWS_TEST" --sst2-data "$SST2_TEST" \
        > "$EVAL_ROOT/raw/${tag}.log" 2>&1 \
    || { echo "  [eval] $tag FAILED — $EVAL_ROOT/raw/${tag}.log"; return 1; }
  slug="$("$PY" -c "import re,sys;s=re.sub(r'[^A-Za-z0-9._-]+','__',sys.argv[1].strip());print(s.strip('._-') or 'model')" "$model")"
  [[ -f "$EVAL_ROOT/raw/$slug/summary.json" ]] \
    || { echo "  [eval] $tag summary.json 없음 ($EVAL_ROOT/raw/$slug)"; return 1; }
  cp "$EVAL_ROOT/raw/$slug/summary.json" "$dst"
  "$PY" -c "
import json,sys
s=json.load(open(sys.argv[1]))
for t,v in s.items():
    print(f\"  [eval] {sys.argv[2]}  {t}: acc={v['accuracy']:.4f} macro_f1={v['macro_f1']:.4f} invalid={v['invalid_count']}\")
" "$dst" "$tag"
}

# ── 업로드 헬퍼 ─────────────────────────────────────────────────────────
# ⚠️ chat_template 은 별도 chat_template.jinja 로 저장된다. 빠지면 허브 모델이
#    chat_template=None 이 되어 평가 프롬프트가 학습과 달라진다(이 레포에서 실제로 났던 사고).
#    업로드 전/후로 존재를 검증한다.
upload_model() {   # $1=local_dir $2=repo
  local src=$1 repo=$2
  echo "  [upload] $(basename "$(dirname "$src")") -> $repo $(date +%H:%M:%S)"
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
    commit_message="SafeLoRA (threshold sweep) merged model",
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

# ═══════════════════ threshold 루프 ═══════════════════
for thr in $THRS; do
  echo ""
  echo "████████████████ thr=$thr ████████████████  $(date +%H:%M:%S)  free=$(free_gb)GB"
  merged="$(merged_of "$thr")"
  repo="$(repo_of "$thr")"

  # ── 1) 학습 + 사후 투영 (기존 러너 재사용: 하이퍼파라미터가 갈라지지 않게)
  if [[ -f "$merged/config.json" ]]; then
    echo "[thr $thr] 학습 이미 완료 — skip"
  else
    hr; echo "[thr $thr] SafeLoRA 학습 $(date +%H:%M:%S)"; hr
    CUDA_VISIBLE_DEVICES="$GPU" MODEL="$MODEL" SAFELORA_BASE_MODEL="$SAFELORA_BASE_MODEL" \
    TASKS="$TASK" THRS="$thr" LR="$LR" EPOCHS="$EPOCHS" \
    OUTPUT_ROOT="$OUT_ROOT" PUSH_TO_HUB=0 PY="$PY" \
      bash scripts/run_safelora_thr_sweep_qa.sh \
      || { echo "[thr $thr] 학습 FAILED"; failed+=("train/thr$thr"); continue; }
  fi
  [[ -f "$merged/config.json" ]] || { echo "[thr $thr] merged_model 없음: $merged"; failed+=("merged/thr$thr"); continue; }

  # ── 2) 평가 (업로드 전에 로컬 모델로)
  if [[ "$DO_EVAL" == "1" ]]; then
    eval_model "$merged" "${TASK}_safelora_lr${LR}_ep${EPOCHS}_thr${thr}" || failed+=("eval/thr$thr")
  fi

  # ── 3) 업로드
  if [[ "$DO_UPLOAD" == "1" ]]; then
    upload_model "$merged" "$repo" || { echo "[thr $thr] 업로드 FAILED"; failed+=("upload/thr$thr"); }
  fi
done

# ═══════════════════ 기준점(thr 0.5) 평가 — 비교표를 완성하기 위해 ═══════════════════
if [[ "$DO_EVAL" == "1" && "$EVAL_REF" == "1" ]]; then
  echo ""
  hr; echo "기준점 thr=$REF_THR 평가 (허브에서 로드) $(date +%H:%M:%S)"; hr
  eval_model "$(repo_of "$REF_THR")" "${TASK}_safelora_lr${LR}_ep${EPOCHS}_thr${REF_THR}" \
    || failed+=("eval/ref-thr$REF_THR")
fi

# ═══════════════════ summary ═══════════════════
echo ""
echo "════════════════ SafeLoRA threshold 비교 (${TASK}, lr ${LR}, ep ${EPOCHS}) ════════════════"
"$PY" - "$EVAL_ROOT" "$OUT_ROOT" "$TASK" "$LR" <<'PYEOF'
import json, re, sys
from pathlib import Path

eval_root, out_root, task, lr = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]

rows = []
for p in sorted(eval_root.glob("*.json")):
    m = re.search(r"_thr(?P<thr>[0-9.]+)$", p.stem)
    if not m:
        continue
    thr = m["thr"]
    s = json.load(open(p))
    # 투영된 레이어 수는 학습 산출물의 summary.json 에 있다 (기준점은 로컬에 없을 수 있다)
    proj = "-"
    sj = out_root / f"{task}_lr{lr}_thr{thr}" / "summary.json"
    if sj.is_file():
        sl = json.load(open(sj)).get("safelora") or {}
        if sl.get("num_layers_projected") is not None:
            proj = f"{sl['num_layers_projected']}/{sl['num_lora_layers']}"
    for t, v in s.items():
        rows.append((float(thr), thr, proj, v["accuracy"], v["macro_f1"], v.get("invalid_count", 0),
                     v.get("samples", "")))

if not rows:
    print("  (평가 결과 없음)")
else:
    rows.sort(key=lambda r: -r[0])
    print(f"  {'threshold':<10} {'투영 레이어':<12} {'acc':>8} {'macro_f1':>9} {'invalid':>8} {'n':>6}")
    print("  " + "-" * 58)
    for _, thr, proj, acc, f1, inv, n in rows:
        print(f"  {thr:<10} {proj:<12} {acc:>8.4f} {f1:>9.4f} {inv:>8} {str(n):>6}")
print(f"\n  raw: {eval_root}")
PYEOF

echo "  free: $(free_gb)GB"
if [[ ${#failed[@]} -gt 0 ]]; then echo "실패: ${failed[*]}"; exit 1; fi
echo "완료."
