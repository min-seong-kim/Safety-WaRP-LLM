#!/usr/bin/env bash
#
# AsFT lambda 스윕 (ARC-Challenge / MedQA).
# LISA rho 스윕(run_lisa_rho_sweep_qa.sh) / SafeLoRA threshold 스윕의 AsFT 대응판.
#
# 기준선: scripts/run_lisa_safelora_asft_qa.sh 의 AsFT run (lambda_reg=1.0)
#         → kmseong/llama2_7b-chat-{arc,medqa}-asft-r16-a32-lr3e-4-cb
#
# AsFT 에서 LISA 의 rho 에 해당하는 손잡이가 lambda 다:
#     L = L_SFT + λ·Σ_l ‖(I−Ĉ_l)·B_l A_l‖²_F      (Ĉ = VVᵀ/‖V‖_F, V = W_aligned − W_base)
#   둘 다 매 step loss 에 더해지는 2차 벌점이지만 방향성이 다르다:
#     rho    — 파라미터 공간 전 방향을 등방적으로 잡아당김 (downstream 학습도 같이 막힘)
#     lambda — ΔW 중 **안전 부분공간과 직교하는 성분만** 벌함 (부분공간 안쪽은 자유)
#
# ⚠️ 스케일 주의: 기준선 λ=1.0 에서 측정된 정규화 항은 ARC 0.0176 / MedQA 0.0057 로
#    CE loss 대비 매우 작다. 즉 λ=1.0 은 이미 거의 물리지 않는 지점이다.
#    (참조 구현이 Ĉ 를 ‖V‖² 가 아닌 ‖V‖_F 로 나누고, ΔW=BA 에 s=α/r 스케일링을 빼기 때문)
#    → λ 를 낮추면 λ=0(=순수 LoRA)에 수렴하고, 곡선이 생기는 구간은 오히려 고측(10~1000)이다.
#    각 run 의 summary.json 에 남는 final_reg_loss 가 rho 의 drift 사다리에 대응하는
#    진단 지표이니 스윕 후 반드시 같이 볼 것.
#
# 사용:
#   bash scripts/run_asft_lambda_sweep_qa.sh
#   LAMBDAS="10 100" TASKS=medqa bash scripts/run_asft_lambda_sweep_qa.sh
#   PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/run_asft_lambda_sweep_qa.sh
#
# 완료된 run 은 건너뛰므로 중간에 죽어도 재실행하면 이어서 간다.
set -uo pipefail          # -e 는 쓰지 않는다: 한 run 이 실패해도 나머지를 계속 돌린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# 기준선과 동일한 출발점/데이터
MODEL="${MODEL:-wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb}"
ASFT_BASE_MODEL="${ASFT_BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
ASFT_ALIGNED_MODEL="${ASFT_ALIGNED_MODEL:-$MODEL}"

TASKS="${TASKS:-arc medqa}"
LAMBDAS="${LAMBDAS:-0.5}"
LR="${LR:-3e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/asft_lambda_sweep_qa}"

# ── matched 동작점 (기준선과 동일, 건드리지 않는다) ──
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
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"

MIN_FREE_GB="${MIN_FREE_GB:-20}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

TS=$(date +%Y%m%d_%H%M%S)
TAG=$(echo "$TASKS" | tr ' ' '-')
mkdir -p logs "$OUTPUT_ROOT"
exec > >(tee -a "logs/asft_lambda_sweep_qa_${TAG}_${TS}.log") 2>&1

task_path() {
    case "$1" in
        arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
        medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
        *)      echo "" ;;
    esac
}
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

for t in $TASKS; do
    p="$(task_path "$t")"
    [[ -n "$p" ]] || { echo "알 수 없는 task: $t (arc|medqa)" >&2; exit 1; }
    [[ -f "$p" ]]  || { echo "task data 없음: $p" >&2; exit 1; }
done

echo "════════════════════════════════════════════════════════════════"
echo " AsFT lambda sweep   ts=${TS}"
echo "   base model : $MODEL   (GPU $CUDA_VISIBLE_DEVICES)"
echo "   asft base  : $ASFT_BASE_MODEL"
echo "   tasks      : $TASKS      lambdas: $LAMBDAS      lr: $LR"
echo "   고정        : r=$LORA_R alpha=$LORA_ALPHA batch=${MICRO_BATCH}x${GRAD_ACCUM} ep=$EPOCHS"
echo "   기준선(λ=1.0): $HF_NAMESPACE/llama2_7b-chat-{task}-asft-r16-a32-lr${LR}-cb"
echo "   output     : $OUTPUT_ROOT      free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

failed=()
for task in $TASKS; do
  TASK_DATA="$(task_path "$task")"
  for lam in $LAMBDAS; do
    out_dir="$OUTPUT_ROOT/${task}_lr${LR}_lambda${lam}"
    if [[ -f "$out_dir/summary.json" ]]; then
      echo "[${task} λ=$lam] 이미 완료 — skip"; continue
    fi
    avail=$(free_gb)
    if [[ "$avail" -lt "$MIN_FREE_GB" ]]; then
      echo "[${task} λ=$lam] SKIP — 디스크 부족 (${avail}GB < ${MIN_FREE_GB}GB)"
      failed+=("${task}/λ${lam}(disk)"); continue
    fi
    echo "──────────────────────────────────────────────────────────────"
    echo "[${task} λ=$lam] start  ($(date +%H:%M:%S), free ${avail}GB)"
    mkdir -p "$out_dir"
    push_args=()
    [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--push_to_hub --hf_repo_id \
      "${HF_NAMESPACE}/llama2_7b-chat-${task}-asft-r16-a32-lr${LR}-cb-lambda${lam}")

    if "$PY" finetune_gsm8k_lora.py \
        --method asft \
        --model_name "$MODEL" \
        --output_dir "$out_dir" \
        --task_data_path "$TASK_DATA" \
        --target_modules "$TARGET_MODULES_CSV" \
        --layer_type "$LAYER_TYPES" --target_layers all \
        --asft_base_model "$ASFT_BASE_MODEL" \
        --asft_aligned_model "$ASFT_ALIGNED_MODEL" \
        --asft_lambda_reg "$lam" \
        --asft_store_dtype float32 \
        --asft_check_equiv \
        --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
        --learning_rate "$LR" --epochs "$EPOCHS" \
        --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_length "$MAX_LENGTH" \
        --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
        --seed "$SEED" --dtype bfloat16 --save_merged_model \
        "${push_args[@]}" 2>&1 | tee "$out_dir/run.log"; then
      echo "[${task} λ=$lam] done  ($(date +%H:%M:%S))"
    else
      echo "[${task} λ=$lam] FAILED"; failed+=("${task}/λ${lam}")
    fi
  done
done

echo ""
echo "════════════════════ summary (정규화 항 실측) ════════════════════"
for f in $(find "$OUTPUT_ROOT" -name summary.json | sort); do
    "$PY" -c "
import json
d=json.load(open('$f')); a=d.get('asft') or {}
print(f\"  λ={a.get('lambda_reg'):<8} final_reg_loss={a.get('final_reg_loss')}  \"
      f\"layers={a.get('num_layers')}  {'$f'.rsplit('/',2)[1]}\")"
done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "실패/스킵: ${failed[*]}"; exit 1
fi
echo "완료. 평가는 arc_eval / medqa_eval 하네스에서 별도 수행할 것."
