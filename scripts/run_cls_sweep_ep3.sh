#!/usr/bin/env bash
#
# SST-2 / AG News × {full FT, SafeDelta(s=0.4), RESTA(γ=0.5), WSR-Tune(kr=0.1)} × lr {3e-6, 5e-6}, epoch 3.
#
#   출발점 W_align = kmseong/llama2_7b-chat-Safety-FT-lr5e-5   (4개 arm 전부 동일)
#   공통    = full-parameter, effective batch 16 (2×8), max_len 1024, seed 42, bf16,
#             cosine + warmup 0.1, weight_decay 0        ← WaRP Phase 3 와 동일한 최적화 설정
#
# 기존 러너를 조합만 한다 (로직 중복 금지):
#   scripts/run_cls_baselines.sh   STAGE=A(MODES=baseline) → full FT,  STAGE=B → SafeDelta + RESTA
#   scripts/run_warp_cls.sh        Phase 1/2(1회 공유) + Phase 3
#   scripts/upload_cls_baselines.sh / scripts/upload_and_cleanup_phase3.py → 업로드
#
# ⚠️ 디스크: 모델 1개 ≈ 13.5GB, lr 하나당 8개 ≈ 108GB. 그래서 **lr 단위로 업로드 후 로컬 삭제**한다.
#    업로드 검증(shard 수 + chat_template 존재)이 통과한 것만 지운다.
#
# 완료 마커로 재개 가능하다. 중간에 죽어도 그대로 재실행하면 이어서 간다.
#
# 사용:
#   bash scripts/run_cls_sweep_ep3.sh
#   LR_LIST=3e-6 bash scripts/run_cls_sweep_ep3.sh      # lr 하나만
#   DELETE_AFTER=0 bash scripts/run_cls_sweep_ep3.sh    # 업로드만 하고 로컬 보존 (디스크 주의)
#   STOP_AFTER_MASKS=1 bash scripts/run_cls_sweep_ep3.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# hb 환경의 python (base 에는 torch/transformers 가 없다)
PY="${PY:-/home/edgeai_lab/miniconda3/envs/hb/bin/python}"
export PY
export PATH="$(dirname "$PY"):$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"     # RESTA 의 model3
TASKS="${TASKS:-agnews sst2}"
LR_LIST="${LR_LIST:-3e-6 5e-6}"
EPOCHS="${EPOCHS:-3}"
KEEP_RATIO="${KEEP_RATIO:-0.1}"
SAFEDELTA_SCALE="${SAFEDELTA_SCALE:-0.4}"
RESTA_GAMMA="${RESTA_GAMMA:-0.5}"
LAYER_TYPE="${LAYER_TYPE:-attn_q,attn_k,attn_v,ffn_up,ffn_down}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

DO_UPLOAD="${DO_UPLOAD:-1}"
DO_EVAL="${DO_EVAL:-1}"
DELETE_AFTER="${DELETE_AFTER:-1}"
STOP_AFTER_MASKS="${STOP_AFTER_MASKS:-0}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/cls_baselines}"
STATE_DIR="$REPO_DIR/outputs/cls_sweep_ep3"
STATE_ENV="$STATE_DIR/basis_mask.env"

# ⚠️ 평가는 **업로드/삭제 직전, 모델이 아직 로컬에 있을 때** 돌린다.
#    맨 마지막에 허브 id 로 평가하면 16개 × 13.5GB 를 다시 내려받아야 한다.
EVAL_ROOT="${EVAL_ROOT:-$REPO_DIR/evaluation_results/cls_sweep_ep3}"
# evaluate_agnews_sst2.py 의 기본 데이터 경로(`<repo>/../dataset/classification/`)는 다른 머신
# 레이아웃 기준이라 이 박스에 없다. 실제 파일 위치를 명시적으로 넘긴다.
AGNEWS_TEST="${AGNEWS_TEST:-$REPO_DIR/data/agnews_test_1k_seed42.json}"
SST2_TEST="${SST2_TEST:-$REPO_DIR/data/sst2_validation_full.json}"
EVAL_BATCH="${EVAL_BATCH:-128}"

mkdir -p logs "$STATE_DIR/markers"
TS=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "logs/cls_sweep_ep3_${TS}.log") 2>&1

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
mark()    { touch "$STATE_DIR/markers/$1"; }
marked()  { [[ -f "$STATE_DIR/markers/$1" ]]; }
hr()      { echo "────────────────────────────────────────────────────────────────"; }

failed=()

for f in "$AGNEWS_TEST" "$SST2_TEST"; do
  [[ -f "$f" ]] || { echo "평가 데이터 없음: $f" >&2; exit 1; }
done

echo "════════════════════════════════════════════════════════════════"
echo " CLS sweep (ep${EPOCHS})  ts=${TS}"
echo "   W_align : $MODEL"
echo "   tasks   : $TASKS      lr: $LR_LIST      epochs: $EPOCHS"
echo "   arms    : fullft | SafeDelta s=$SAFEDELTA_SCALE | RESTA γ=$RESTA_GAMMA | WSR-Tune kr=$KEEP_RATIO"
echo "   python  : $PY"
echo "   eval    : $DO_EVAL (agnews=$(basename "$AGNEWS_TEST"), sst2=$(basename "$SST2_TEST"))"
echo "   upload  : $DO_UPLOAD   delete-after-verify: $DELETE_AFTER   free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

# ═══════════════════ Phase 1/2 (1회, 모든 lr·태스크 공유) ═══════════════════
# Phase 1/2 는 downstream 과 무관하다: 안전 모델 + 안전 데이터 + layer_type 에만 의존.
if [[ -f "$STATE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_ENV"
fi
if [[ -z "${PHASE1_BASIS_DIR:-}" || ! -d "${PHASE1_BASIS_DIR:-/nonexistent}" \
   || -z "${PHASE2_MASK_DIR:-}"  || ! -d "${PHASE2_MASK_DIR:-/nonexistent}" ]]; then
  hr; echo "[phase1/2] basis + mask 생성 $(date +%H:%M:%S)"; hr
  MODEL="$MODEL" LAYER_TYPE="$LAYER_TYPE" TARGET_LAYERS="$TARGET_LAYERS" \
  KEEP_RATIO="$KEEP_RATIO" PY="$PY" STOP_AFTER_MASKS=1 \
    bash scripts/run_warp_cls.sh || { echo "Phase 1/2 FAILED"; exit 1; }

  PHASE1_BASIS_DIR="$(ls -dt "$REPO_DIR"/checkpoints/phase1_*/basis 2>/dev/null | head -1)"
  PHASE2_MASK_DIR="$(ls -dt "$REPO_DIR"/checkpoints/phase2_*/checkpoints/masks 2>/dev/null | head -1)"
  [[ -d "$PHASE1_BASIS_DIR" && -d "$PHASE2_MASK_DIR" ]] \
    || { echo "basis/mask 경로를 찾지 못했습니다"; exit 1; }
  { echo "PHASE1_BASIS_DIR=$PHASE1_BASIS_DIR"; echo "PHASE2_MASK_DIR=$PHASE2_MASK_DIR"; } > "$STATE_ENV"
fi
echo "  basis : $PHASE1_BASIS_DIR"
echo "  masks : $PHASE2_MASK_DIR"
[[ "$STOP_AFTER_MASKS" == "1" ]] && { echo "STOP_AFTER_MASKS=1 — 종료"; exit 0; }

# ═══════════════════ 평가 헬퍼 ═══════════════════
# agnews_eval/run_agnews_sst2_eval.sh 를 그대로 쓴다 (프롬프트/파싱 규칙이 갈라지지 않게).
# --task 를 명시하는 이유: WaRP 산출물 경로(checkpoints/phase3_<ts>/final_model)에는 태스크명이
# 없어서 auto 추론이 두 태스크를 다 돌려버린다. 학습한 태스크만 평가한다.
eval_model() {   # $1=model_dir $2=task $3=gpu $4=tag
  local model=$1 task=$2 gpu=$3 tag=$4 slug dst
  dst="$EVAL_ROOT/${tag}.json"
  [[ -f "$dst" ]] && { echo "  [eval] $tag 이미 완료 — skip"; return 0; }
  [[ -n "$model" ]] || { echo "  [eval] $tag 모델 경로를 찾지 못했습니다"; return 1; }
  [[ -d "$model" || "$model" != /* ]] || { echo "  [eval] $tag 모델 없음: $model"; return 1; }
  mkdir -p "$EVAL_ROOT/raw"
  echo "  [eval] $tag (GPU $gpu) $(date +%H:%M:%S)"
  GPU="$gpu" PYTHON_BIN="$PY" OUTPUT_ROOT="$EVAL_ROOT/raw" BATCH_SIZE="$EVAL_BATCH" \
    bash agnews_eval/run_agnews_sst2_eval.sh "$model" \
        --task "$task" \
        --agnews-data "$AGNEWS_TEST" --sst2-data "$SST2_TEST" \
        > "$EVAL_ROOT/raw/${tag}.log" 2>&1 \
    || { echo "  [eval] $tag FAILED — $EVAL_ROOT/raw/${tag}.log"; return 1; }
  # evaluator 는 결과를 model 문자열 slug 디렉토리에 쓴다. 같은 규칙으로 slug 를 재계산해 찾는다.
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
  return 0
}

# 태스크별로 GPU 를 갈라 두 갈래 병렬 실행 (agnews→GPU_A, sst2→GPU_B).
eval_arms_parallel() {   # $1=lr  $2...=arm 이름들
  local lr=$1; shift
  local arms=("$@") pids=() rc=0 task gpu
  for task in $TASKS; do
    gpu=0; [[ "$task" == "sst2" ]] && gpu=1
    (
      local a m ok=0
      for a in "${arms[@]}"; do
        case "$a" in
          fullft)    m="$OUT_ROOT/${task}_fullft_lr${lr}_ep${EPOCHS}" ;;
          safedelta) m="$OUT_ROOT/${task}_fullft_lr${lr}_ep${EPOCHS}-SafeDelta-s${SAFEDELTA_SCALE}" ;;
          resta)     m="$OUT_ROOT/${task}_resta_gamma${RESTA_GAMMA}_lr${lr}_ep${EPOCHS}" ;;
          warp)      m="$(warp_model_path "$task" "$lr")" ;;
          *) echo "  [eval] 알 수 없는 arm: $a"; ok=1; continue ;;
        esac
        eval_model "$m" "$task" "$gpu" "${task}_${a}_lr${lr}_ep${EPOCHS}" || ok=1
      done
      exit $ok
    ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

# ═══════════════════ 업로드 헬퍼 ═══════════════════
# full FT / SafeDelta / RESTA : upload_cls_baselines.sh 가 디렉토리명에서 repo 명을 유도하고
#                               업로드 전/후로 chat_template 존재를 검증한다.
upload_baseline_arms() {   # $1=lr
  local lr=$1
  MODES="baseline safedelta resta" TASKS="$TASKS" LR="$lr" EPOCHS="$EPOCHS" \
  SAFEDELTA_SCALE="$SAFEDELTA_SCALE" RESTA_GAMMA="$RESTA_GAMMA" \
  OUT_ROOT="$OUT_ROOT" HF_NAMESPACE="$HF_NAMESPACE" PY="$PY" \
    bash scripts/upload_cls_baselines.sh
}

cleanup_baseline_arms() {  # $1=lr — 업로드 검증 통과 후에만 호출된다
  local lr=$1 task d
  for task in $TASKS; do
    for d in "$OUT_ROOT/${task}_fullft_lr${lr}_ep${EPOCHS}" \
             "$OUT_ROOT/${task}_fullft_lr${lr}_ep${EPOCHS}-SafeDelta-s${SAFEDELTA_SCALE}" \
             "$OUT_ROOT/${task}_resta_gamma${RESTA_GAMMA}_lr${lr}_ep${EPOCHS}"; do
      [[ -d "$d" ]] && { echo "  rm -rf $d"; rm -rf "$d"; }
    done
  done
}

# WSR-Tune : Phase 3 산출물은 checkpoints/phase3_<ts>/final_model 이고 태스크명이 경로에 없다.
#            → 태스크별 run.log 의 "Final model saved to:" 로 정확히 매핑한다.
warp_model_path() {   # $1=task $2=lr  → stdout: final_model 경로
  local task=$1 lr=$2
  local log="$REPO_DIR/outputs/warp_cls/${task}_kr${KEEP_RATIO}_lr${lr}_ep${EPOCHS}/run.log"
  [[ -f "$log" ]] || return 1
  grep -oP '(?<=Final model saved to: ).*' "$log" | tail -1
}

upload_warp_arm() {   # $1=task $2=lr $3=can_delete(0|1)
  local task=$1 lr=$2 can_delete=${3:-1}
  local model_dir phase3_dir repo
  model_dir="$(warp_model_path "$task" "$lr")" || { echo "  [warp/$task] run.log 없음"; return 1; }
  [[ -n "$model_dir" && -d "$model_dir" ]] || { echo "  [warp/$task] final_model 없음: '$model_dir'"; return 1; }
  phase3_dir="$(dirname "$model_dir")"
  repo="${HF_NAMESPACE}/llama2_7b-chat-${task}-warp-kr${KEEP_RATIO}-lr${lr}-ep${EPOCHS}-cb"

  # ⚠️ 과거 WaRP 업로드에서 chat_template.jinja 가 누락돼 허브 모델이 chat_template=None 이 된 적이 있다.
  #    업로드 전에 로컬에 존재하는지 강제 확인한다 (없으면 여기서 멈추는 게 낫다).
  "$PY" - "$model_dir" <<'PYEOF' || return 1
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
ok = (d / "chat_template.jinja").exists()
tc = d / "tokenizer_config.json"
if not ok and tc.exists():
    ok = "chat_template" in json.loads(tc.read_text())
if not ok:
    raise SystemExit(f"ERROR: {d} 에 chat_template 이 없습니다 (평가 프롬프트가 어긋납니다).")
PYEOF

  local extra=()
  [[ "$DELETE_AFTER" == "1" && "$can_delete" == "1" ]] && extra=(--delete_after_verify)
  "$PY" scripts/upload_and_cleanup_phase3.py \
      --model_dir "$model_dir" --repo_name "$repo" \
      --base_model "$MODEL" --dataset "$task" \
      --keep_ratio "$KEEP_RATIO" --learning_rate "$lr" \
      --metadata_json "$phase3_dir/metadata.json" \
      "${extra[@]}" || return 1

  if [[ "$DELETE_AFTER" == "1" && "$can_delete" == "1" ]]; then
    echo "  rm -rf $phase3_dir"; rm -rf "$phase3_dir"
  fi
  return 0
}

# ═══════════════════ lr 루프 ═══════════════════
for LR in $LR_LIST; do
  echo ""
  echo "████████████████ lr=$LR  (ep${EPOCHS}) ████████████████  $(date +%H:%M:%S)  free=$(free_gb)GB"

  # ── 1) full-param SFT (baseline) : 태스크별 GPU 병렬
  if marked "train_baseline_lr${LR}"; then
    echo "[lr $LR] full FT 이미 완료 — skip"
  else
    hr; echo "[lr $LR] full-param SFT $(date +%H:%M:%S)"; hr
    MODEL="$MODEL" TASKS="$TASKS" LR="$LR" EPOCHS="$EPOCHS" STAGE=A MODES=baseline \
    OUT_ROOT="$OUT_ROOT" PY="$PY" \
      bash scripts/run_cls_baselines.sh \
      && mark "train_baseline_lr${LR}" \
      || { echo "[lr $LR] full FT FAILED"; failed+=("fullft/lr$LR"); continue; }
  fi

  # ── 2) SafeDelta + RESTA (baseline 산출물에 사후 적용)
  if marked "posthoc_lr${LR}"; then
    echo "[lr $LR] SafeDelta/RESTA 이미 완료 — skip"
  else
    hr; echo "[lr $LR] SafeDelta(s=$SAFEDELTA_SCALE) + RESTA(γ=$RESTA_GAMMA) $(date +%H:%M:%S)"; hr
    MODEL="$MODEL" BASE_MODEL="$BASE_MODEL" TASKS="$TASKS" LR="$LR" EPOCHS="$EPOCHS" STAGE=B \
    SAFEDELTA_SCALE="$SAFEDELTA_SCALE" RESTA_GAMMA="$RESTA_GAMMA" \
    OUT_ROOT="$OUT_ROOT" PY="$PY" \
      bash scripts/run_cls_baselines.sh \
      && mark "posthoc_lr${LR}" \
      || { echo "[lr $LR] SafeDelta/RESTA FAILED"; failed+=("posthoc/lr$LR"); }
  fi

  # ── 3a) 평가 (삭제 전에! 로컬 모델로 돌린다)
  # 평가가 실패하면 삭제하지 않는다 — 지우고 나면 재평가에 13.5GB×N 재다운로드가 필요해진다.
  eval_base_ok=1
  if [[ "$DO_EVAL" == "1" ]]; then
    hr; echo "[lr $LR] 평가: fullft / safedelta / resta $(date +%H:%M:%S)"; hr
    eval_arms_parallel "$LR" fullft safedelta resta \
      || { failed+=("eval-baseline/lr$LR"); eval_base_ok=0; }
  fi

  # ── 3b) 업로드 + 로컬 정리 (full FT / SafeDelta / RESTA)
  if [[ "$DO_UPLOAD" == "1" ]]; then
    if marked "upload_baseline_lr${LR}"; then
      echo "[lr $LR] baseline arm 업로드 이미 완료 — skip"
    else
      hr; echo "[lr $LR] 업로드: fullft / safedelta / resta $(date +%H:%M:%S)"; hr
      if upload_baseline_arms "$LR"; then
        mark "upload_baseline_lr${LR}"
        if [[ "$DELETE_AFTER" == "1" && "$eval_base_ok" == "1" ]]; then
          cleanup_baseline_arms "$LR"
        elif [[ "$DELETE_AFTER" == "1" ]]; then
          echo "  [lr $LR] 평가 실패로 로컬 보존 (재평가 후 수동 삭제)"
        fi
      else
        echo "[lr $LR] baseline arm 업로드 FAILED — 로컬 보존"; failed+=("upload-baseline/lr$LR")
      fi
    fi
  fi

  # ── 4) WSR-Tune Phase 3 (basis/mask 재사용, 태스크별 GPU 병렬)
  if marked "train_warp_lr${LR}"; then
    echo "[lr $LR] WSR-Tune 이미 완료 — skip"
  else
    hr; echo "[lr $LR] WSR-Tune Phase 3 (kr=$KEEP_RATIO) $(date +%H:%M:%S)"; hr
    MODEL="$MODEL" TASKS="$TASKS" LR="$LR" EPOCHS="$EPOCHS" KEEP_RATIO="$KEEP_RATIO" \
    LAYER_TYPE="$LAYER_TYPE" TARGET_LAYERS="$TARGET_LAYERS" \
    PHASE1_BASIS_DIR="$PHASE1_BASIS_DIR" PHASE2_MASK_DIR="$PHASE2_MASK_DIR" PY="$PY" \
      bash scripts/run_warp_cls.sh \
      && mark "train_warp_lr${LR}" \
      || { echo "[lr $LR] WSR-Tune FAILED"; failed+=("warp/lr$LR"); }
  fi

  # ── 5a) WSR-Tune 평가 (삭제 전에!)
  eval_warp_ok=1
  if [[ "$DO_EVAL" == "1" ]]; then
    hr; echo "[lr $LR] 평가: warp $(date +%H:%M:%S)"; hr
    eval_arms_parallel "$LR" warp || { failed+=("eval-warp/lr$LR"); eval_warp_ok=0; }
  fi

  # ── 5b) WSR-Tune 업로드 + 로컬 정리
  if [[ "$DO_UPLOAD" == "1" ]]; then
    for task in $TASKS; do
      if marked "upload_warp_${task}_lr${LR}"; then
        echo "[lr $LR] warp/$task 업로드 이미 완료 — skip"; continue
      fi
      hr; echo "[lr $LR] 업로드: warp/$task $(date +%H:%M:%S)"; hr
      if upload_warp_arm "$task" "$LR" "$eval_warp_ok"; then
        mark "upload_warp_${task}_lr${LR}"
      else
        echo "[lr $LR] warp/$task 업로드 FAILED — 로컬 보존"; failed+=("upload-warp-$task/lr$LR")
      fi
    done
  fi

  echo "████ lr=$LR 종료 $(date +%H:%M:%S)  free=$(free_gb)GB"
done

# ═══════════════════ summary ═══════════════════
echo ""
echo "════════════════════ summary ════════════════════"
for LR in $LR_LIST; do
  for task in $TASKS; do
    printf '  lr=%-6s %-7s  fullft=%s safedelta=%s resta=%s warp=%s\n' "$LR" "$task" \
      "$(marked "upload_baseline_lr${LR}" && echo up || echo -)" \
      "$(marked "upload_baseline_lr${LR}" && echo up || echo -)" \
      "$(marked "upload_baseline_lr${LR}" && echo up || echo -)" \
      "$(marked "upload_warp_${task}_lr${LR}" && echo up || echo -)"
  done
done
echo "  free: $(free_gb)GB"

if [[ "$DO_EVAL" == "1" ]]; then
  echo ""
  echo "════════════════ downstream 성능 (학습한 태스크) ════════════════"
  "$PY" - "$EVAL_ROOT" <<'PYEOF'
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob("*.json")):
    m = re.match(r"(?P<task>agnews|sst2)_(?P<arm>\w+?)_lr(?P<lr>[0-9.e+-]+)_ep(?P<ep>\d+)$", p.stem)
    if not m:
        continue
    s = json.load(open(p))
    for task, v in s.items():
        rows.append((m["lr"], m["task"], m["arm"], task, v["accuracy"], v["macro_f1"],
                     v.get("invalid_count", 0), v.get("samples", "")))

if not rows:
    print("  (평가 결과 없음)")
else:
    order = {"fullft": 0, "safedelta": 1, "resta": 2, "warp": 3}
    rows.sort(key=lambda r: (r[0], r[1], order.get(r[2], 9)))
    print(f"  {'lr':<7} {'task':<7} {'arm':<10} {'eval':<7} {'acc':>8} {'macro_f1':>9} {'invalid':>8} {'n':>6}")
    print("  " + "-" * 66)
    for lr, task, arm, ev, acc, f1, inv, n in rows:
        print(f"  {lr:<7} {task:<7} {arm:<10} {ev:<7} {acc:>8.4f} {f1:>9.4f} {inv:>8} {str(n):>6}")
print(f"\n  raw: {root}")
PYEOF
fi

if [[ ${#failed[@]} -gt 0 ]]; then echo "실패: ${failed[*]}"; exit 1; fi
echo "완료."
