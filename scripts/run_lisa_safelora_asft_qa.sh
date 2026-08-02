#!/usr/bin/env bash
#
# LISA / SafeLoRA / AsFT baselines on the QA tasks (ARC-Challenge, MedQA).
#
# scripts/run_lisa_safelora_cls.sh 의 "matched" 동작점을 그대로 가져오되 downstream 만
# SST-2/AG News → ARC-C / MedQA 로 교체하고 AsFT 를 추가한 버전이다.
#   matched : r=16, alpha=32, dropout=0.05, targets={q,k,v,up,down},
#             micro-batch 16 / grad_accum 1, cosine, warmup 0.03, wd 0, seed 42, 3 epochs
#
# 데이터: scripts/prepare_qa_task_data.py 산출물 (프롬프트 포맷은 arc_eval / medqa_eval
#         하네스에서 그대로 가져오므로 기존 평가 스크립트와 일치한다)
#   arc   → data/arc_challenge_train_task_1119.json    (ARC-Challenge train 전체)
#   medqa → data/medqa_train_task_10178.json           (MedQA train 전체)
#
# 세 method 의 안전 메커니즘은 각 논문 그대로:
#   LISA     : circuit_breakers 안전 응답으로 bi-state alternation + proximal(rho)
#   SafeLoRA : 학습은 표준 LoRA, 학습 후 lora_B ← C·B 사후 투영
#              (C = VVᵀ/‖V‖, V = W_aligned − W_base)
#   AsFT     : 학습 중 매 step  L += λ·Σ‖(I−C)BA‖²_F   (C 는 SafeLoRA 와 동일 행렬)
#              참조 구현 /home/edgeai_lab/AsFT, λ=1 (AsFT_reg1_p_0.1.sh 기본값)
#
# 사용:
#   bash scripts/run_lisa_safelora_asft_qa.sh
# GPU 2장을 태스크별로 나눠 쓰는 법 (이 박스 권장):
#   CUDA_VISIBLE_DEVICES=0 TASKS=arc   bash scripts/run_lisa_safelora_asft_qa.sh &
#   CUDA_VISIBLE_DEVICES=1 TASKS=medqa bash scripts/run_lisa_safelora_asft_qa.sh &
# 오버라이드 예:
#   METHODS="asft" LRS="3e-4" bash scripts/run_lisa_safelora_asft_qa.sh
#
# 완료된 run 은 건너뛰므로 중간에 죽어도 재실행하면 이어서 간다.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ═══════════════════ config ═══════════════════
PY="${PY:-python}"
# 이 박스에는 SLURM 이 없다 → GPU 를 여기서 직접 고른다.
# (SLURM 이 있는 환경으로 옮기면 이 줄을 반드시 지울 것 — CLAUDE.md 참고)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# 출발점 = safety FT 모델 (circuit_breakers 로 SSFT 한 llama2-7b-chat)
MODEL="${MODEL:-wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb}"
# SafeLoRA / AsFT 의 alignment delta V = W_aligned − W_base
SAFELORA_BASE_MODEL="${SAFELORA_BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
SAFELORA_ALIGNED_MODEL="${SAFELORA_ALIGNED_MODEL:-$MODEL}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"

TASKS="${TASKS:-arc medqa}"
METHODS="${METHODS:-lisa safelora asft}"
LRS="${LRS:-3e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/lisa_safelora_asft_qa}"

# ── matched 동작점 (method 간 동일) ──
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
SEED="${SEED:-42}"
TARGET_MODULES_CSV="q_proj,k_proj,v_proj,up_proj,down_proj"
TARGET_MODULES_LIST=(q_proj k_proj v_proj up_proj down_proj)
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"

# ── method 고유 설정 (matched 예산 밖) ──
SAFETY_SAMPLES="${SAFETY_SAMPLES:-4994}"
SAFELORA_THRESHOLD="${SAFELORA_THRESHOLD:-0.35}"
LISA_RHO="${LISA_RHO:-1.0}"
LISA_ALIGNMENT_STEP="${LISA_ALIGNMENT_STEP:-100}"
LISA_FINETUNE_STEP="${LISA_FINETUNE_STEP:-900}"
ASFT_LAMBDA_REG="${ASFT_LAMBDA_REG:-1.0}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-}"

TS=$(date +%Y%m%d_%H%M%S)
TAG=$(echo "$TASKS" | tr ' ' '-')
mkdir -p logs "$OUTPUT_ROOT"
exec > >(tee -a "logs/lisa_safelora_asft_qa_${TAG}_${TS}.log") 2>&1

task_path() {
    case "$1" in
        arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
        medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
        sst2)   echo "$REPO_DIR/data/sst2_train_8k_seed42.json" ;;
        agnews) echo "$REPO_DIR/data/agnews_train_8k_seed42.json" ;;
        *)      echo "" ;;
    esac
}

has_method() { [[ " $METHODS " == *" $1 "* ]]; }

[[ -f "$SAFETY_DATA" ]] || { echo "safety data 없음: $SAFETY_DATA" >&2; exit 1; }
for t in $TASKS; do
    p="$(task_path "$t")"
    [[ -n "$p" ]] || { echo "알 수 없는 task: $t (arc|medqa|sst2|agnews)" >&2; exit 1; }
    [[ -f "$p" ]]  || { echo "task data 없음: $p — scripts/prepare_qa_task_data.py 먼저 실행" >&2; exit 1; }
done
if [[ "$PUSH_TO_HUB" == "1" && -z "$HF_NAMESPACE" ]]; then
    echo "PUSH_TO_HUB=1 이면 HF_NAMESPACE 가 필요합니다" >&2; exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo " LISA / SafeLoRA / AsFT on ${TASKS}   ts=${TS}"
echo "   python          : $PY  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "   base model      : $MODEL"
echo "   safelora/asft base   : $SAFELORA_BASE_MODEL"
echo "   safelora/asft aligned: $SAFELORA_ALIGNED_MODEL"
echo "   safety data     : $SAFETY_DATA ($SAFETY_SAMPLES examples, LISA only)"
echo "   methods         : $METHODS      lrs: $LRS"
echo "   matched         : r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT "\
"batch=${MICRO_BATCH}x${GRAD_ACCUM} epochs=$EPOCHS targets=$TARGET_MODULES_CSV"
echo "   lisa            : rho=$LISA_RHO align=$LISA_ALIGNMENT_STEP finetune=$LISA_FINETUNE_STEP"
echo "   asft            : lambda_reg=$ASFT_LAMBDA_REG"
echo "   output          : $OUTPUT_ROOT"
echo "════════════════════════════════════════════════════════════════"

for task in $TASKS; do
  TASK_DATA="$(task_path "$task")"
  for lr in $LRS; do

    # ═══════════ LISA ═══════════
    if has_method lisa; then
      out_dir="$OUTPUT_ROOT/lisa/${task}_lr${lr}"
      if [[ -f "$out_dir/finetune_config.json" ]]; then
        echo "[LISA ${task} lr=$lr] 이미 완료 — skip"
      else
        echo "──────────────────────────────────────────────────────────────"
        echo "[LISA ${task} lr=$lr] start  ($(date +%H:%M:%S))"
        mkdir -p "$out_dir"
        push_args=()
        [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--upload_name \
          "${HF_NAMESPACE}/llama2_7b-chat-${task}-lisa-r16-a32-lr${lr}-cb")
        "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
            --model_path "$MODEL" \
            --output_dir "$out_dir" \
            --task_data_path "$TASK_DATA" \
            --num_eval_samples 0 \
            --safety_data_path "$SAFETY_DATA" \
            --guide_data_num "$SAFETY_SAMPLES" \
            --rho "$LISA_RHO" \
            --alignment_step "$LISA_ALIGNMENT_STEP" \
            --finetune_step "$LISA_FINETUNE_STEP" \
            --lora \
            --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
            --lr_scheduler_type cosine \
            --seed "$SEED" --bf16 --report_to none \
            "${push_args[@]}" 2>&1 | tee "$out_dir/run.log"
        echo "[LISA ${task} lr=$lr] done  ($(date +%H:%M:%S))"
      fi
    fi

    # ═══════════ SafeLoRA ═══════════
    if has_method safelora; then
      out_dir="$OUTPUT_ROOT/safelora/${task}_lr${lr}"
      if [[ -f "$out_dir/summary.json" ]]; then
        echo "[SafeLoRA ${task} lr=$lr] 이미 완료 — skip"
      else
        echo "──────────────────────────────────────────────────────────────"
        echo "[SafeLoRA ${task} lr=$lr] start  ($(date +%H:%M:%S))"
        mkdir -p "$out_dir"
        push_args=()
        [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--push_to_hub --hf_repo_id \
          "${HF_NAMESPACE}/llama2_7b-chat-${task}-safelora-r16-a32-lr${lr}-cb")
        "$PY" finetune_gsm8k_lora.py \
            --method safe_lora \
            --model_name "$MODEL" \
            --output_dir "$out_dir" \
            --task_data_path "$TASK_DATA" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers all \
            --safelora_base_model "$SAFELORA_BASE_MODEL" \
            --safelora_aligned_model "$SAFELORA_ALIGNED_MODEL" \
            --safelora_select_type threshold \
            --safelora_threshold "$SAFELORA_THRESHOLD" \
            --safelora_load_dtype float32 \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
            --seed "$SEED" --dtype bfloat16 --save_merged_model \
            "${push_args[@]}" 2>&1 | tee "$out_dir/run.log"
        echo "[SafeLoRA ${task} lr=$lr] done  ($(date +%H:%M:%S))"
      fi
    fi

    # ═══════════ AsFT ═══════════
    if has_method asft; then
      out_dir="$OUTPUT_ROOT/asft/${task}_lr${lr}"
      if [[ -f "$out_dir/summary.json" ]]; then
        echo "[AsFT ${task} lr=$lr] 이미 완료 — skip"
      else
        echo "──────────────────────────────────────────────────────────────"
        echo "[AsFT ${task} lr=$lr] start  ($(date +%H:%M:%S))"
        mkdir -p "$out_dir"
        push_args=()
        [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--push_to_hub --hf_repo_id \
          "${HF_NAMESPACE}/llama2_7b-chat-${task}-asft-r16-a32-lr${lr}-cb")
        "$PY" finetune_gsm8k_lora.py \
            --method asft \
            --model_name "$MODEL" \
            --output_dir "$out_dir" \
            --task_data_path "$TASK_DATA" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers all \
            --asft_base_model "$SAFELORA_BASE_MODEL" \
            --asft_aligned_model "$SAFELORA_ALIGNED_MODEL" \
            --asft_lambda_reg "$ASFT_LAMBDA_REG" \
            --asft_store_dtype float32 \
            --asft_check_equiv \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
            --seed "$SEED" --dtype bfloat16 --save_merged_model \
            "${push_args[@]}" 2>&1 | tee "$out_dir/run.log"
        echo "[AsFT ${task} lr=$lr] done  ($(date +%H:%M:%S))"
      fi
    fi

  done
done

echo ""
echo "════════════════════ summary ════════════════════"
find "$OUTPUT_ROOT" \( -name summary.json -o -name finetune_config.json \) | sort | sed 's/^/  /'
echo "완료. 평가는 별도 하네스(arc/medqa eval + HarmBench ASR)에서 수행할 것."
