#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  논문 Table 1 — Llama-3.1-8B **base** × **MATH** (2026-09-23, 사용자 지시)
#
#  동기: GSM8K·lr 1e-5 (실험 2)에서는 downstream FT 의 안전성 붕괴가 작아서
#  (ASR 0.037→0.088) 원공간 동결과 WSR-Tune 의 차이가 잘 드러나지 않았다.
#  교수님 요청 = "기존 masking 과 WSR-Tune 의 safety 차이가 더 크게 나는 셋업".
#  → 과제를 MATH 로, lr 을 5e-5 로 올려 붕괴를 키운다.
#
#  셀 (모두 출발 = kmseong/Llama-3.1-8B-base-SSFT_lr5e-5, MATH 7500, lr 5e-5)
#    p00     : 동결 0% (= plain full FT)     ← 원공간 경로 + all-zero 마스크
#    p10/30/50 : 원공간 동결 10/30/50%        ← 실험 2 의 Phase 2 마스크 재사용
#    wsr_p10 : WSR-Tune ρ=0.1                 ← Phase 1/2 새로 계산 (basis 는 prune 됨)
#
#  ⚠️ p00 을 finetune_task_full_params.py 로 돌리면 안 된다. base 모델에서 그 러너는
#     `question + "\n"` 프롬프트를 쓰고, train.py Phase 3 는 `Question: {q}\nAnswer:`
#     를 쓴다 → 동결 0% 행만 프롬프트가 달라진다. 그래서 모든 셀을 train.py Phase 3
#     (같은 _load_local_task + _tokenize_question_answer_example) 로 돌린다.
#
#  공통: 3ep · eff.batch 16 · max_len 1024 · seed 42 · bf16 · cosine · wd 0.01 ·
#        warmup 0.1 · max_grad_norm 1.0 · q,k,v,up,down 전 레이어
#
#  GPU 두 장을 두 큐로 나눠 쓴다.
#    GPU0 : wsr_p10 (P1→P2→P3) → p50
#    GPU1 : p00 → p10 → p30
#  업로드 없음(사용자 결정). 셀마다 .done / MODEL_DIR 를 남기므로 재실행하면 이어간다.
#
#  사용:
#    bash scripts/revision/33_table1_math_8b_base.sh            # 학습 + 평가
#    STAGE=train bash ...   /   STAGE=eval bash ...
#    HB_ONLY=1 EVAL_REFS="" STAGE=eval bash ...   # ASR 만, 새 셀만
#    DRY_RUN=1 bash ...
#    MODEL=llama2_7b_base bash ...                 # Llama-2-7B base 라인 (출발 llama2_7b-base-CB_SSFT-lr3e-5)
#  ⚠️ 실행 중인 이 스크립트를 편집하지 말 것.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
MINSEONG="$(dirname "$REPO_DIR")"

PY="${PY:-$MINSEONG/miniconda3/envs/hb/bin/python}"
export HF_HOME="${HF_HOME:-$MINSEONG/.hf_cache}"
export TMPDIR="${TMPDIR:-$MINSEONG/.tmp}"            # /tmp 는 noexec
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
mkdir -p "$TMPDIR"

STAGE="${STAGE:-all}"          # all | train | eval
DRY_RUN="${DRY_RUN:-0}"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"

# ── 모델 선택 (MODEL=llama31_8b_base 기본 | llama2_7b_base) ─────────────────────
#  원공간 마스크는 H 절(실험 2)에서 **같은 출발 모델**·같은 CB 4994 로 만든 것을 재사용한다.
#  마스크는 출발 모델 + safety 데이터에만 의존하므로 downstream 과제와 무관하다.
MODEL="${MODEL:-llama31_8b_base}"
TASK="${TASK:-math}"          # math | gsm8k  (2026-09-23 밤: GSM8K 추가)
declare -A ORIG_MASK
case "$MODEL" in
  llama31_8b_base)
    START="kmseong/Llama-3.1-8B-base-SSFT_lr5e-5"
    BASE_REF="meta-llama/Llama-3.1-8B"; NAME_PREFIX="llama3_1_8b-base"; TITLE="Llama-3.1-8B base"
    MB_FULL=2;  ACC_FULL=8      # 실험 2 의 8B-base 원공간 셀과 같은 분할
    MB_WARP=1;  ACC_WARP=16     # common.sh LLAMA31_8B_BASE_MB_WARP
    DEF_ROOT="$REPO_DIR/outputs/table1_${TASK}_8b_base"; DEF_LOGD="$REPO_DIR/logs/table1_${TASK}_8b_base"
    OS_P2="$REPO_DIR/checkpoints/origspace_base_llama31_8b_base"
    ORIG_MASK=(
      [p10]="$OS_P2/phase2_original_space_20260920_050350/checkpoints/masks"
      [p20]="$OS_P2/phase2_original_space_20260920_052655/checkpoints/masks"
      [p30]="$OS_P2/phase2_original_space_20260920_054716/checkpoints/masks"
      [p40]="$OS_P2/phase2_original_space_20260920_060924/checkpoints/masks"
      [p50]="$OS_P2/phase2_original_space_20260920_063147/checkpoints/masks" )
    DEF_QUEUE_A="wsr_p10 p50"; DEF_QUEUE_B="p00 p10 p30" ;;
  llama2_7b_base)
    START="kmseong/llama2_7b-base-CB_SSFT-lr3e-5"
    BASE_REF="meta-llama/Llama-2-7b-hf"; NAME_PREFIX="llama2_7b-base"; TITLE="Llama-2-7B base"
    MB_FULL=4;  ACC_FULL=4      # 실험 2 의 7B-base 원공간 셀과 같은 분할 (common.sh LLAMA2_7B_BASE_MB_FULL)
    MB_WARP=1;  ACC_WARP=16     # common.sh LLAMA2_7B_BASE_MB_WARP
    DEF_ROOT="$REPO_DIR/outputs/table1_${TASK}_7b_base"; DEF_LOGD="$REPO_DIR/logs/table1_${TASK}_7b_base"
    OS_P2="$REPO_DIR/checkpoints/origspace_base_llama2_7b_base"
    ORIG_MASK=(
      [p10]="$OS_P2/phase2_original_space_20260919_170522/checkpoints/masks"
      [p20]="$OS_P2/phase2_original_space_20260919_172320/checkpoints/masks"
      [p30]="$OS_P2/phase2_original_space_20260920_041205/checkpoints/masks"
      [p40]="$OS_P2/phase2_original_space_20260920_043109/checkpoints/masks"
      [p50]="$OS_P2/phase2_original_space_20260920_044650/checkpoints/masks" )
    DEF_QUEUE_A="wsr_p10"; DEF_QUEUE_B="p00 p10 p30 p50" ;;
  *) echo "알 수 없는 MODEL=$MODEL"; exit 1 ;;
esac
QUEUE_A="${QUEUE_A-$DEF_QUEUE_A}"; QUEUE_B="${QUEUE_B-$DEF_QUEUE_B}"   # "-": 빈 값 = 그 큐를 비운다

case "$TASK" in
  math)  TASK_JSON="$REPO_DIR/data/math_train_task_7500.json";  LM_TASK=hendrycks_math_safe ;;
  gsm8k) TASK_JSON="$REPO_DIR/data/gsm8k_train_task_7473.json"; LM_TASK=gsm8k ;;   # lm-eval gsm8k 5-shot = "Question: {q}\nAnswer:" (학습 프롬프트와 동일, H 절 확인)
  *) echo "알 수 없는 TASK=$TASK"; exit 1 ;;
esac
SAFE_JSON="$REPO_DIR/data/circuit_breakers_train.json"
LR=5e-5; EPOCHS=3; MAX_LENGTH=1024; SEED=42; DTYPE=bfloat16
WD=0.01; WARMUP=0.1; SCHED=cosine; MAX_GRAD_NORM=1.0
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"; TARGET_LAYERS=all
SAFETY_SAMPLES=4994
MB_P12=2

ROOT="${ROOT:-$DEF_ROOT}"
LOGD="${LOGD:-$DEF_LOGD}"
mkdir -p "$ROOT" "$LOGD"

declare -A ORIG_RATIO=([p00]=0.0 [p10]=0.1 [p20]=0.2 [p30]=0.3 [p40]=0.4 [p50]=0.5)

# 평가용 이름 — chat/instruct/it 토큰 금지(base plain 프롬프트 판정이 이름 문자열 기반)
eval_name() {
  case "$1" in
    wsr_p10) echo "${NAME_PREFIX}-wsr-tune-kr0.1-${TASK}-lr${LR}" ;;
    *)       echo "${NAME_PREFIX}-origspace-freeze-$1-${TASK}-lr${LR}" ;;
  esac
}

log() { echo "[$(date '+%F %T')] $*"; }
newest() { find "$1" -maxdepth 1 -name "$2*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

# ── 마스크 무결성: 메타데이터의 keep_ratio 가 기대값과 같아야 한다 ─────────────
check_mask() {  # dir expected_ratio
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys, os
d, exp = sys.argv[1], float(sys.argv[2])
m = json.load(open(os.path.join(d, "metadata.json")))
assert m.get("space") == "original_weight_space", m.get("space")
assert abs(float(m["keep_ratio"]) - exp) < 1e-9, (m["keep_ratio"], exp)
print(f"  mask ok: keep_ratio={m['keep_ratio']} space={m['space']}")
PYEOF
}

# ── 동결 0% 용 all-zero 마스크 (p10 마스크의 모양·키를 그대로 복제) ────────────
make_zero_mask() {
  local dst="$ROOT/masks_p00"
  [ -f "$dst/.ok" ] && { echo "$dst"; return 0; }
  "$PY" - "${ORIG_MASK[p10]}" "$dst" >&2 <<'PYEOF'
import json, os, sys, glob, numpy as np, torch
src, dst = sys.argv[1], sys.argv[2]
n = 0
for f in sorted(glob.glob(os.path.join(src, "*", "*.pt"))):
    obj = torch.load(f, weights_only=False)
    out = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else torch.zeros_like(v)) for k, v in obj.items()}
    rel = os.path.relpath(f, src)
    os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
    torch.save(out, os.path.join(dst, rel)); n += 1
meta = json.load(open(os.path.join(src, "metadata.json")))
meta["keep_ratio"] = 0.0
meta["note"] = "all-zero mask derived from keep_ratio=0.1 mask shape (Table 1 MATH, freeze 0%)"
json.dump(meta, open(os.path.join(dst, "metadata.json"), "w"), indent=2)
assert n == 160, f"expected 160 mask files (32 layers x 5 types), got {n}"
print(f"  zero mask: {n} files → {dst}")
PYEOF
  [ $? -eq 0 ] && touch "$dst/.ok" && echo "$dst"
}

phase3_common_args() {
  echo --phase3_dataset "$TASK" --phase3_task_data_path "$TASK_JSON" --phase3_task_samples 0 \
       --epochs "$EPOCHS" --utility_lr "$LR" --base_weight_decay "$WD" \
       --warmup_ratio "$WARMUP" --lr_scheduler_type "$SCHED" --max_grad_norm "$MAX_GRAD_NORM" \
       --max_length "$MAX_LENGTH" --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
       --device cuda --dtype "$DTYPE" --seed "$SEED" --no_wandb
}

# ── 원공간 셀 (p00/p10/p30/p50) ───────────────────────────────────────────────
run_orig() {  # tag gpu
  local tag="$1" gpu="$2" cell="$ROOT/$1"
  mkdir -p "$cell"
  [ -f "$cell/.done" ] && { log "[skip] $tag 완료: $(cat "$cell/MODEL_DIR")"; return 0; }
  local masks
  if [ "$tag" = p00 ]; then masks="$(make_zero_mask)" || return 1
  else masks="${ORIG_MASK[$tag]}"; fi
  check_mask "$masks" "${ORIG_RATIO[$tag]}" || { log "❌ $tag 마스크 검증 실패"; return 1; }
  log "▶ $tag  (GPU$gpu, 원공간 동결 ${ORIG_RATIO[$tag]}, masks=$masks)"
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry] train.py --phase 3 --original_space_mask --masks_dir $masks $(phase3_common_args)"; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --phase 3 \
      --phase0_model_dir "$START" --original_space_mask --masks_dir "$masks" \
      $(phase3_common_args) \
      --batch_size "$MB_FULL" --gradient_accumulation_steps "$ACC_FULL" \
      --output_dir "$cell" --log_dir "$LOGD" \
      --profile_json "$cell/phase3_profile.json" > "$cell/run.log" 2>&1
  local rc=$?
  local p3; p3="$(newest "$cell" phase3_original_space_)"
  if [ $rc -eq 0 ] && [ -f "$p3/final_model/config.json" ]; then
    echo "$p3/final_model" > "$cell/MODEL_DIR"; date -Iseconds > "$cell/.done"
    log "✔ $tag → $p3/final_model"
  else
    log "❌ $tag 실패 rc=$rc — $cell/run.log"; return 1
  fi
}

# ── WSR-Tune 셀 (P1 basis → P2 mask ρ=0.1 --perlayer → P3 non_freeze) ────────
run_wsr() {  # gpu
  local gpu="$1" cell="$ROOT/wsr_p10"
  mkdir -p "$cell"
  [ -f "$cell/.done" ] && { log "[skip] wsr_p10 완료: $(cat "$cell/MODEL_DIR")"; return 0; }
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry] wsr: phase1 → phase2(--perlayer 0.1) → phase3 --non_freeze on GPU$gpu"; return 0; fi

  # 재사용: 같은 출발 모델·같은 CB 로 만든 basis / ρ=0.1 mask 는 과제·lr 과 무관하다.
  if [ -n "${WSR_BASIS_DIR:-}" ] && [ ! -s "$cell/BASIS_DIR" ]; then echo "$WSR_BASIS_DIR" > "$cell/BASIS_DIR"; fi
  if [ -n "${WSR_MASKS_DIR:-}" ] && [ ! -s "$cell/MASKS_DIR" ]; then echo "$WSR_MASKS_DIR" > "$cell/MASKS_DIR"; fi
  local basis=""
  [ -s "$cell/BASIS_DIR" ] && [ -d "$(cat "$cell/BASIS_DIR")" ] && basis="$(cat "$cell/BASIS_DIR")"
  if [ -z "$basis" ]; then
    log "▶ wsr_p10 Phase 1 (basis, GPU$gpu)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --phase 1 \
        --phase0_model_dir "$START" --safety_dataset circuit_breakers \
        --circuit_breakers_path "$SAFE_JSON" --circuit_breakers_samples_phase1 "$SAFETY_SAMPLES" \
        --basis_save_dtype bfloat16 --basis_omit_ut --batch_size "$MB_P12" --max_length "$MAX_LENGTH" \
        --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
        --output_dir "$cell" --log_dir "$LOGD" --device cuda --dtype "$DTYPE" --seed "$SEED" --no_wandb \
        --profile_json "$cell/phase1_profile.json" > "$cell/phase1.log" 2>&1 \
      || { log "❌ wsr Phase 1 실패 — $cell/phase1.log"; return 1; }
    basis="$(newest "$cell" phase1)/basis"
    [ -d "$basis" ] || { log "❌ basis 없음: $basis"; return 1; }
    echo "$basis" > "$cell/BASIS_DIR"
  fi

  local masks=""
  [ -s "$cell/MASKS_DIR" ] && [ -d "$(cat "$cell/MASKS_DIR")" ] && masks="$(cat "$cell/MASKS_DIR")"
  if [ -z "$masks" ]; then
    log "▶ wsr_p10 Phase 2 (mask ρ=0.1 --perlayer, GPU$gpu)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --phase 2 \
        --phase0_model_dir "$START" --basis_dir "$basis" \
        --dataset_phase2 circuit_breakers --circuit_breakers_path "$SAFE_JSON" \
        --circuit_breakers_samples_phase2 "$SAFETY_SAMPLES" \
        --keep_ratio 0.1 --perlayer --batch_size "$MB_P12" --max_length "$MAX_LENGTH" \
        --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
        --output_dir "$cell" --log_dir "$LOGD" --device cuda --dtype "$DTYPE" --seed "$SEED" --no_wandb \
        --profile_json "$cell/phase2_profile.json" > "$cell/phase2.log" 2>&1 \
      || { log "❌ wsr Phase 2 실패 — $cell/phase2.log"; return 1; }
    masks="$(find "$(newest "$cell" phase2)" -type d -name masks | head -1)"
    [ -d "$masks" ] || { log "❌ mask 없음"; return 1; }
    echo "$masks" > "$cell/MASKS_DIR"
  fi

  log "▶ wsr_p10 Phase 3 (non_freeze, GPU$gpu, basis=$basis masks=$masks)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --phase 3 \
      --phase0_model_dir "$START" --basis_dir "$basis" --masks_dir "$masks" \
      $(phase3_common_args) \
      --batch_size "$MB_WARP" --gradient_accumulation_steps "$ACC_WARP" \
      --non_freeze --gradient_checkpointing \
      --output_dir "$cell" --log_dir "$LOGD" \
      --profile_json "$cell/phase3_profile.json" > "$cell/run.log" 2>&1
  local rc=$?
  local p3; p3="$(newest "$cell" phase3_non_freeze_)"
  if [ $rc -eq 0 ] && [ -f "$p3/final_model/config.json" ]; then
    echo "$p3/final_model" > "$cell/MODEL_DIR"; date -Iseconds > "$cell/.done"
    log "✔ wsr_p10 → $p3/final_model"
  else
    log "❌ wsr_p10 Phase 3 실패 rc=$rc — $cell/run.log"; return 1
  fi
}

run_cell() {  # tag gpu
  case "$1" in
    wsr_p10) run_wsr "$2" ;;
    *)       run_orig "$1" "$2" ;;
  esac
}

# ── 단일 셀 (STAGE=cell CELL=p50 CELL_GPU=1) — 놀고 있는 GPU 로 셀을 옮길 때 ─────
if [ "$STAGE" = cell ]; then
  run_cell "${CELL:?CELL 필요}" "${CELL_GPU:?}"
  exit $?
fi

# ── 학습 ─────────────────────────────────────────────────────────────────────
if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
  [ -f "$TASK_JSON" ] || { log "❌ $TASK_JSON 없음"; exit 1; }
  log "════ Table 1 · $TITLE · $TASK · lr $LR · start=$START ════"
  log "  GPU$GPU_A 큐: $QUEUE_A   |   GPU$GPU_B 큐: $QUEUE_B"
  ( for c in $QUEUE_A; do run_cell "$c" "$GPU_A"; done ) > "$LOGD/queue_gpu${GPU_A}.log" 2>&1 &
  QA=$!
  ( for c in $QUEUE_B; do run_cell "$c" "$GPU_B"; done ) > "$LOGD/queue_gpu${GPU_B}.log" 2>&1 &
  QB=$!
  log "큐 GPU$GPU_A pid=$QA  (로그 $LOGD/queue_gpu${GPU_A}.log)"
  log "큐 GPU$GPU_B pid=$QB  (로그 $LOGD/queue_gpu${GPU_B}.log)"
  wait $QA; wait $QB
  cat "$LOGD/queue_gpu${GPU_A}.log" "$LOGD/queue_gpu${GPU_B}.log"
  log "TRAIN_FINISHED"
fi

# ── 평가 (HarmBench 4공격 + lm-eval hendrycks_math_safe 5-shot) ───────────────
if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  mkdir -p "$REPO_DIR/outputs/eval_local"
  PATHS=()
  for tag in p00 p10 p20 p30 p40 p50 wsr_p10; do
    cell="$ROOT/$tag"
    [ -f "$cell/.done" ] || { log "[eval] $tag 미완료 — 제외"; continue; }
    md="$(cat "$cell/MODEL_DIR")"
    link="$REPO_DIR/outputs/eval_local/$(eval_name "$tag")"
    ln -sfn "$md" "$link"; PATHS+=("$link")
  done
  # 출발 모델(SSFT)·원본 base 의 MATH 점수도 같은 잣대로 잰다 (HarmBench 는 RESUME 로 기존 결과 재사용)
  # EVAL_REFS="" 로 끄면 새 셀만 잰다 (두 모델의 ASR 은 RESULTS.md H 절에 이미 있다).
  for r in ${EVAL_REFS-$START $BASE_REF}; do PATHS+=("$r"); done
  log "[eval] 대상 ${#PATHS[@]}개"; printf '   - %s\n' "${PATHS[@]}"
  [ "$DRY_RUN" = 1 ] && exit 0
  # 셸이 다른 conda 설치(jeesuppark/svd_safety)로 활성화돼 있으면 그 CONDA_* 변수 때문에
  # minseong conda 의 `conda activate harmbench` 가 rc=0 을 내면서 PATH 를 안 바꾼다
  # → HarmBench 가 /usr/bin/python 으로 돌아 "No module named transformers" (2026-09-23 실측).
  unset CONDA_EXE CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER
  PREFETCH=0 PREFETCH_MODE=off LM_TASKS="$LM_TASK" REPOS_ONLY="${PATHS[*]}" \
    bash scripts/revision/32_eval_cells.sh 2>&1 | tee "$LOGD/eval_$(date +%Y%m%d_%H%M%S).log"
  log "EVAL_FINISHED"
fi
