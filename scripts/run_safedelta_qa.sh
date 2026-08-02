#!/usr/bin/env bash
#
# SafeDelta (ICML'25, /home/edgeai_lab/SafeDelta) baseline — MedQA / SST-2.
#
# SafeDelta 는 SafeLoRA 처럼 **사후 보정**이라 2단계로 돈다:
#   Stage 1 : 방어 없이 평범하게 fine-tune → W_sft   (여기서는 LoRA r16/α32 merge, 다른 baseline 과 동일 예산)
#   Stage 2 : W_sd = W_orig + M⊙ΔW + C               (llama2/run_safedelta.py)
#
# --scale (내부 s) 의 의미 — README 한글 주석과 코드가 반대이니 주의:
#   safedelta_runner.py:141  loss_constraint = tmp.mean() * s * scale
#   safedelta_runner.py:145  sorted_mask = cumulative_weights <= loss_constraint
#   safedelta_runner.py:159  q[mask_sub] = w_sft[mask_sub]
#   → s 는 **허용 안전 손실 예산**. 크면 fine-tuned 델타를 더 많이 살린다
#     = utility↑ safety↓ (README 의 "Delta 클수록 safety↑" 는 틀림).
#   기본 0.1, 여기 기본값 0.4 = 4배 느슨한 방어.
#
# ⚠️ SafeDelta 원본은 run_safedelta.py 에 CUDA_VISIBLE_DEVICES="2,3" 이 import 시점
#    하드코딩돼 있었다(이 박스는 GPU 2장뿐이라 그대로면 실패). setdefault 로 고쳐 두었으니
#    셸에서 지정한 값이 존중된다. 되돌리지 말 것.
# ⚠️ run_safedelta.py 는 `from configs import ...` 때문에 llama2/ 안에서 실행해야 한다.
#
# 사용:
#   bash scripts/run_safedelta_qa.sh
#   TASKS=sst2 SCALE=0.025 bash scripts/run_safedelta_qa.sh
#   STOP_AFTER_STAGE1=1 bash scripts/run_safedelta_qa.sh
#
# 완료된 단계는 건너뛰므로 중간에 죽어도 재실행하면 이어서 간다.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
SAFEDELTA_DIR="${SAFEDELTA_DIR:-/home/edgeai_lab/SafeDelta}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# W_orig = 정렬(안전) 모델. 다른 baseline 과 동일한 출발점.
MODEL="${MODEL:-wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb}"
SAFE_DATA="${SAFE_DATA:-$SAFEDELTA_DIR/llama2/safedelta/data/circuit_breakers_train.json}"

TASKS="${TASKS:-medqa sst2}"
SCALE="${SCALE:-0.4}"
LR="${LR:-3e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/safedelta_qa}"

# ── Stage 1 LoRA (다른 baseline 과 동일 matched 예산) ──
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
MICRO_BATCH="${MICRO_BATCH:-16}"; GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-3}"; MAX_LENGTH="${MAX_LENGTH:-1024}"
WARMUP_RATIO=0.03; WEIGHT_DECAY=0.0; SEED="${SEED:-42}"
TARGET_MODULES_CSV="q_proj,k_proj,v_proj,up_proj,down_proj"
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"

# ── Stage 2 SafeDelta ──
NSAMPLES="${NSAMPLES:-512}"
SEQ_LEN="${SEQ_LEN:-512}"

MIN_FREE_GB="${MIN_FREE_GB:-30}"
STOP_AFTER_STAGE1="${STOP_AFTER_STAGE1:-0}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

TS=$(date +%Y%m%d_%H%M%S)
TAG=$(echo "$TASKS" | tr ' ' '-')
mkdir -p logs "$OUTPUT_ROOT"
exec > >(tee -a "logs/safedelta_${TAG}_${TS}.log") 2>&1

task_path() {
    case "$1" in
        medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
        sst2)   echo "$REPO_DIR/data/sst2_train_8k_seed42.json" ;;
        agnews) echo "$REPO_DIR/data/agnews_train_8k_seed42.json" ;;
        arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
        *)      echo "" ;;
    esac
}
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

[[ -f "$SAFE_DATA" ]] || { echo "safe data 없음: $SAFE_DATA" >&2; exit 1; }
[[ -f "$SAFEDELTA_DIR/llama2/run_safedelta.py" ]] || { echo "SafeDelta 없음: $SAFEDELTA_DIR" >&2; exit 1; }
for t in $TASKS; do
    p="$(task_path "$t")"
    [[ -n "$p" && -f "$p" ]] || { echo "task data 없음: $t -> $p" >&2; exit 1; }
done

echo "════════════════════════════════════════════════════════════════"
echo " SafeDelta  ts=${TS}"
echo "   W_orig(align) : $MODEL      (GPU $CUDA_VISIBLE_DEVICES)"
echo "   safe data     : $SAFE_DATA"
echo "   tasks         : $TASKS      scale(s): $SCALE      lr: $LR"
echo "   Stage1 LoRA   : r=$LORA_R alpha=$LORA_ALPHA batch=${MICRO_BATCH}x${GRAD_ACCUM} ep=$EPOCHS"
echo "   Stage2        : nsamples=$NSAMPLES seq_len=$SEQ_LEN"
echo "   s 의미        : 허용 안전손실 예산 (클수록 utility↑ safety↓). 기본 0.1"
echo "   output        : $OUTPUT_ROOT      free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

failed=()
for task in $TASKS; do
  TASK_DATA="$(task_path "$task")"
  ft_dir="$OUTPUT_ROOT/${task}_lr${LR}_sft"          # Stage 1 산출물
  sft_model="$ft_dir/merged_model"

  # ═══════════ Stage 1: 방어 없는 LoRA SFT (W_sft) ═══════════
  if [[ -f "$ft_dir/summary.json" ]]; then
    echo "[${task}] Stage1 이미 완료 — skip"
  else
    avail=$(free_gb)
    if [[ "$avail" -lt "$MIN_FREE_GB" ]]; then
      echo "[${task}] Stage1 SKIP — 디스크 부족 (${avail}GB)"; failed+=("${task}/stage1(disk)"); continue
    fi
    echo "──────────────────────────────────────────────────────────────"
    echo "[${task}] Stage1 (plain LoRA SFT) start  ($(date +%H:%M:%S), free ${avail}GB)"
    mkdir -p "$ft_dir"
    if ! "$PY" finetune_gsm8k_lora.py \
        --method lora \
        --model_name "$MODEL" \
        --output_dir "$ft_dir" \
        --task_data_path "$TASK_DATA" \
        --target_modules "$TARGET_MODULES_CSV" \
        --layer_type "$LAYER_TYPES" --target_layers all \
        --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
        --learning_rate "$LR" --epochs "$EPOCHS" \
        --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_length "$MAX_LENGTH" \
        --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
        --seed "$SEED" --dtype bfloat16 --save_merged_model \
        2>&1 | tee "$ft_dir/run.log"; then
      echo "[${task}] Stage1 FAILED"; failed+=("${task}/stage1"); continue
    fi
    echo "[${task}] Stage1 done  ($(date +%H:%M:%S))"
  fi

  [[ "$STOP_AFTER_STAGE1" == "1" ]] && { echo "[${task}] STOP_AFTER_STAGE1=1 — Stage2 건너뜀"; continue; }
  [[ -f "$sft_model/config.json" ]] || { echo "[${task}] Stage1 산출물 없음: $sft_model"; failed+=("${task}/stage1-missing"); continue; }

  # ═══════════ Stage 2: SafeDelta 사후 보정 ═══════════
  # run_safedelta.py 는 결과를 {model_name_ft}-SafeDelta-s{s} 에 저장한다.
  out_model="${sft_model}-SafeDelta-s${SCALE}"
  if [[ -f "$out_model/config.json" ]]; then
    echo "[${task}] Stage2 이미 완료 — skip ($out_model)"; continue
  fi
  echo "──────────────────────────────────────────────────────────────"
  echo "[${task}] Stage2 (SafeDelta s=$SCALE) start  ($(date +%H:%M:%S), free $(free_gb)GB)"
  push_args=()
  [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--upload_name \
    "${HF_NAMESPACE}/llama2_7b-chat-${task}-safedelta-r16-a32-lr${LR}-cb-s${SCALE}")

  # ⚠️ from configs import ... 때문에 llama2/ 안에서 실행해야 한다.
  if ( cd "$SAFEDELTA_DIR/llama2" && "$PY" run_safedelta.py \
        --model_name_align "$MODEL" \
        --model_name_ft "$sft_model" \
        --scale "$SCALE" \
        --nsamples "$NSAMPLES" --seq_len "$SEQ_LEN" \
        --safe_data_path "$SAFE_DATA" \
        "${push_args[@]}" ) 2>&1 | tee "$ft_dir/safedelta_s${SCALE}.log"; then
    echo "[${task}] Stage2 done  ($(date +%H:%M:%S)) → $out_model"
  else
    echo "[${task}] Stage2 FAILED"; failed+=("${task}/stage2")
  fi
done

echo ""
echo "════════════════════ summary ════════════════════"
find "$OUTPUT_ROOT" -maxdepth 2 -name config.json -path "*SafeDelta*" | sort | sed 's|/config.json||;s|^|  |'
if [[ ${#failed[@]} -gt 0 ]]; then echo "실패: ${failed[*]}"; exit 1; fi
echo "완료. 평가는 medqa_eval / sst2 하네스 + ASR 로 별도 수행할 것."
