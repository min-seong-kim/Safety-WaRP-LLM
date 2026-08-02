#!/usr/bin/env bash
#
# SST-2 / AG News 용 baseline + 방어기법 4종 (총 8개 모델).
#
#   출발점 W_align = kmseong/llama2_7b-chat-Safety-FT-lr5e-5
#   공통 설정      = full-parameter SFT, lr 1e-5, epoch 1, effective batch 16
#                    (WaRP Phase 3 kr0.1-lr1e-5-ep1 과 동일한 최적화/토큰화 설정)
#
#   Stage A (학습, GPU 2장 병렬)
#     1) baseline  : downstream FT 만
#     2) SafeInstr : downstream + circuit_breakers 10% 혼합
#   Stage B (사후, baseline 산출물 재사용)
#     3) SafeDelta : W_align + M⊙(W_ft − W_align) + C     (s = 0.4)
#     4) RESTA     : W_ft + γ·(W_align − Llama-2-7b-chat-hf)  (γ = 0.5)
#
# ⚠️ SafeDelta / RESTA 는 **baseline FT 모델**(SafeInstr 아님)에 적용한다.
#    둘 다 "방어 없이 튜닝된 모델을 사후에 고친다"는 설정의 기법이기 때문이다.
#
# 완료된 단계는 건너뛰므로 중간에 죽어도 재실행하면 이어서 간다.
# 사용:
#   bash scripts/run_cls_baselines.sh
#   TASKS=sst2 STAGE=A bash scripts/run_cls_baselines.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"   # RESTA 의 model3
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"
SAFEDELTA_DIR="${SAFEDELTA_DIR:-/home/edgeai_lab/SafeDelta}"
SAFEDELTA_SAFE_DATA="${SAFEDELTA_SAFE_DATA:-$SAFEDELTA_DIR/llama2/safedelta/data/circuit_breakers_train.json}"

TASKS="${TASKS:-agnews sst2}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"          # effective 16
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-42}"
SAFEINSTR_RATIO="${SAFEINSTR_RATIO:-0.1}"
SAFEDELTA_SCALE="${SAFEDELTA_SCALE:-0.4}"
RESTA_GAMMA="${RESTA_GAMMA:-0.5}"
NSAMPLES="${NSAMPLES:-512}"            # SafeDelta stage2
SEQ_LEN="${SEQ_LEN:-512}"

GPU_A="${GPU_A:-0}"                    # agnews
GPU_B="${GPU_B:-1}"                    # sst2
STAGE="${STAGE:-AB}"                   # A=학습만, B=사후만, AB=전부
MODES="${MODES:-baseline safeinstr}"   # Stage A 에서 돌릴 학습 종류
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/cls_baselines}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUT_ROOT"
exec > >(tee -a "logs/cls_baselines_${TS}.log") 2>&1

task_path() {
    case "$1" in
        sst2)   echo "$REPO_DIR/data/sst2_train_8k_seed42.json" ;;
        agnews) echo "$REPO_DIR/data/agnews_train_8k_seed42.json" ;;
        arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
        medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
        *)      echo "" ;;
    esac
}
task_gpu() { [[ "$1" == "sst2" ]] && echo "$GPU_B" || echo "$GPU_A"; }
free_gb()  { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

ft_dir()       { echo "$OUT_ROOT/${1}_fullft_lr${LR}_ep${EPOCHS}"; }
safeinstr_dir(){ echo "$OUT_ROOT/${1}_safeinstr${SAFEINSTR_RATIO}_lr${LR}_ep${EPOCHS}"; }
safedelta_dir(){ echo "$(ft_dir "$1")-SafeDelta-s${SAFEDELTA_SCALE}"; }
resta_dir()    { echo "$OUT_ROOT/${1}_resta_gamma${RESTA_GAMMA}_lr${LR}_ep${EPOCHS}"; }

for t in $TASKS; do
    p="$(task_path "$t")"
    [[ -n "$p" && -f "$p" ]] || { echo "task data 없음: $t -> $p" >&2; exit 1; }
done
[[ -f "$SAFETY_DATA" ]] || { echo "safety data 없음: $SAFETY_DATA" >&2; exit 1; }

echo "════════════════════════════════════════════════════════════════"
echo " CLS baselines  ts=${TS}"
echo "   W_align   : $MODEL"
echo "   base(RESTA model3): $BASE_MODEL"
echo "   tasks     : $TASKS      stage: $STAGE      modes: $MODES"
echo "   full-param SFT: lr=$LR ep=$EPOCHS batch=${BATCH_SIZE}x${GRAD_ACCUM} (eff $((BATCH_SIZE*GRAD_ACCUM)))"
echo "   SafeInstr : ratio=$SAFEINSTR_RATIO   SafeDelta: s=$SAFEDELTA_SCALE   RESTA: γ=$RESTA_GAMMA"
echo "   GPU       : agnews=$GPU_A  sst2=$GPU_B       free: $(free_gb)GB"
echo "   output    : $OUT_ROOT"
echo "════════════════════════════════════════════════════════════════"

failed=()

# ═══════════ Stage A: 학습 (태스크별 GPU 병렬, 각 GPU 안에서는 baseline → SafeInstr 순차) ═══════════
train_one() {   # $1=task $2=mode(baseline|safeinstr) $3=gpu
  local task=$1 mode=$2 gpu=$3 out ratio
  if [[ "$mode" == "safeinstr" ]]; then out="$(safeinstr_dir "$task")"; ratio="$SAFEINSTR_RATIO"
  else                                  out="$(ft_dir "$task")";        ratio=0; fi

  if [[ -f "$out/summary.json" ]]; then echo "[$task/$mode] 이미 완료 — skip"; return 0; fi
  local avail; avail=$(free_gb)
  if [[ "$avail" -lt "$MIN_FREE_GB" ]]; then
    echo "[$task/$mode] SKIP — 디스크 부족 (${avail}GB)"; return 1; fi

  echo "[$task/$mode] start (GPU $gpu, free ${avail}GB) $(date +%H:%M:%S)"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" finetune_task_full_params.py \
      --model_path "$MODEL" \
      --task_data_path "$(task_path "$task")" \
      --task_name "$task" \
      --output_dir "$out" \
      --learning_rate "$LR" --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
      --max_length "$MAX_LENGTH" --seed "$SEED" \
      --safety_data_path "$SAFETY_DATA" --safety_mix_ratio "$ratio" \
      --report_to none \
      > "$out/run.log" 2>&1 \
    && { echo "[$task/$mode] done $(date +%H:%M:%S)"; return 0; } \
    || { echo "[$task/$mode] FAILED — $out/run.log 확인"; return 1; }
}

train_task_chain() {   # 한 GPU 에서 MODES 를 순차 실행 (기본: baseline → SafeInstr)
  local task=$1 gpu=$2 rc=0 mode
  for mode in $MODES; do
    train_one "$task" "$mode" "$gpu" || rc=1
  done
  return $rc
}

if [[ "$STAGE" == *A* ]]; then
  echo ""
  echo "──────────── Stage A: full-param SFT (baseline + SafeInstr) ────────────"
  pids=(); names=()
  for task in $TASKS; do
    train_task_chain "$task" "$(task_gpu "$task")" &
    pids+=("$!"); names+=("$task")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || failed+=("${names[$i]}/stageA"); done
fi

# ═══════════ Stage B1: SafeDelta (사후) ═══════════
run_safedelta() {   # $1=task $2=gpu
  local task=$1 gpu=$2
  local sft="$(ft_dir "$task")" out="$(safedelta_dir "$task")"
  if [[ -f "$out/config.json" ]]; then echo "[$task/safedelta] 이미 완료 — skip"; return 0; fi
  [[ -f "$sft/config.json" ]] || { echo "[$task/safedelta] baseline FT 없음: $sft"; return 1; }
  echo "[$task/safedelta] start (GPU $gpu, s=$SAFEDELTA_SCALE) $(date +%H:%M:%S)"
  # ⚠️ run_safedelta.py 는 `from configs import ...` 때문에 llama2/ 안에서 실행해야 한다.
  ( cd "$SAFEDELTA_DIR/llama2" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_safedelta.py \
        --model_name_align "$MODEL" \
        --model_name_ft "$sft" \
        --scale "$SAFEDELTA_SCALE" \
        --nsamples "$NSAMPLES" --seq_len "$SEQ_LEN" \
        --safe_data_path "$SAFEDELTA_SAFE_DATA" ) > "$sft/safedelta_s${SAFEDELTA_SCALE}.log" 2>&1 \
    && { echo "[$task/safedelta] done $(date +%H:%M:%S) → $out"; return 0; } \
    || { echo "[$task/safedelta] FAILED — $sft/safedelta_s${SAFEDELTA_SCALE}.log 확인"; return 1; }
}

# ═══════════ Stage B2: RESTA (사후, weight-space merge) ═══════════
run_resta() {   # $1=task
  local task=$1
  local sft="$(ft_dir "$task")" out="$(resta_dir "$task")"
  if [[ -f "$out/resta_merge.json" ]]; then echo "[$task/resta] 이미 완료 — skip"; return 0; fi
  [[ -f "$sft/config.json" ]] || { echo "[$task/resta] baseline FT 없음: $sft"; return 1; }
  echo "[$task/resta] start (γ=$RESTA_GAMMA) $(date +%H:%M:%S)"
  "$PY" scripts/resta_add_safety.py \
      --model1 "$sft"        --weight1 1.0 \
      --model2 "$MODEL"      --weight2 "$RESTA_GAMMA" \
      --model3 "$BASE_MODEL" --weight3 "-$RESTA_GAMMA" \
      --output_path "$out" --dtype bfloat16 > "$out.log" 2>&1 \
    && { echo "[$task/resta] done $(date +%H:%M:%S) → $out"; return 0; } \
    || { echo "[$task/resta] FAILED — $out.log 확인"; return 1; }
}

if [[ "$STAGE" == *B* ]]; then
  echo ""
  echo "──────────── Stage B: SafeDelta(s=$SAFEDELTA_SCALE) + RESTA(γ=$RESTA_GAMMA) ────────────"
  pids=(); names=()
  for task in $TASKS; do
    ( run_safedelta "$task" "$(task_gpu "$task")" && run_resta "$task" ) &
    pids+=("$!"); names+=("$task")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || failed+=("${names[$i]}/stageB"); done
fi

echo ""
echo "════════════════════ summary ════════════════════"
for task in $TASKS; do
  for d in "$(ft_dir "$task")" "$(safeinstr_dir "$task")" "$(safedelta_dir "$task")" "$(resta_dir "$task")"; do
    if [[ -f "$d/config.json" ]]; then
      printf '  ✓ %s\n' "$d"
    else
      printf '  ✗ %s (없음)\n' "$d"
    fi
  done
done
echo "  free: $(free_gb)GB"
if [[ ${#failed[@]} -gt 0 ]]; then echo "실패: ${failed[*]}"; exit 1; fi
echo "완료. 업로드는 scripts/upload_cls_baselines.sh 로 수행할 것."
