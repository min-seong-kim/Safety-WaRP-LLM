#!/usr/bin/env bash
#
# LISA rho 스윕 (ARC-Challenge / MedQA) — safety↓ downstream↑ 방향 탐색.
#
# 기준선: scripts/run_lisa_safelora_asft_qa.sh 의 LISA run (rho=1.0)
#         → kmseong/llama2_7b-chat-{arc,medqa}-lisa-r16-a32-lr3e-4-cb
#
# 이 스크립트는 그 run 에서 **rho 하나만** 바꾼다. alignment_step/finetune_step,
# LoRA 예산, 옵티마이저 설정, 시작 모델, safety 데이터는 전부 동일하게 유지 —
# 하나라도 같이 흔들면 "rho 축의 효과"라는 대조가 깨진다.
#
# 왜 0.1 이 아니라 {0, 0.01, 0.001} 인가 (run_lisa_align_sweep.sh:5-12 측정치):
#     무보호 SSFT+DT  GSM8K 0.4117 / ASR 0.2064
#     rho=0 (alt만)   GSM8K 0.3867 / ASR 0.1288
#     rho=0.1         GSM8K 0.19   ← 이미 효용 붕괴
#   proximal 은 rho=0.1 만으로도 무제약 drift 의 99.94% 를 제거한다. 즉 rho 는 연속
#   다이얼이 아니라 계단 함수에 가깝고, 변화는 0~0.1 구간에 몰려 있다. 측정된 drift
#   (첫 전환 시 82~332)로 penalty=rho/2·‖Δ‖² 를 어림하면 CE loss(~0.5~0.9)와 맞먹는
#   지점이 rho≈0.01 이라 그 부근을 훑는다.
#
# ⚠️ ARC 주의: 1119 samples / batch16 / 3ep = 210 step 인데 alignment_step=100 이므로
#    학습의 47.6% 가 safety 데이터이고 그게 앞쪽(warmup+고LR 구간)에 몰려 있다.
#    ARC 의 downstream 약세는 rho 보다 이 스케줄이 주된 원인일 수 있다 — rho 를 아무리
#    낮춰도 이 구조는 그대로다. 그 축을 건드리려면 ALIGNMENT_STEP 을 함께 낮출 것
#    (예: ALIGNMENT_STEP=10 FINETUNE_STEP=90 → 의도했던 10% 비율 복원).
#
# 사용:
#   bash scripts/run_lisa_rho_sweep_qa.sh
#   TASKS=medqa RHOS="0 0.01" bash scripts/run_lisa_rho_sweep_qa.sh
#   PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/run_lisa_rho_sweep_qa.sh
# GPU 2장을 태스크별로 나눠 쓰는 법:
#   CUDA_VISIBLE_DEVICES=0 TASKS=arc   bash scripts/run_lisa_rho_sweep_qa.sh &
#   CUDA_VISIBLE_DEVICES=1 TASKS=medqa bash scripts/run_lisa_rho_sweep_qa.sh &
#
# 완료된 run 은 건너뛰므로 중간에 죽어도 재실행하면 이어서 간다.
set -uo pipefail          # -e 는 쓰지 않는다: 한 run 이 실패해도 나머지를 계속 돌린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ═══════════════════ config ═══════════════════
PY="${PY:-python}"
# 이 박스에는 SLURM 이 없다 → GPU 를 여기서 직접 고른다.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# 기준선과 동일한 출발점/데이터
MODEL="${MODEL:-wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"

TASKS="${TASKS:-arc medqa}"
RHOS="${RHOS:-0 0.01 0.001}"
LR="${LR:-3e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/lisa_rho_sweep_qa}"

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
TARGET_MODULES_LIST=(q_proj k_proj v_proj up_proj down_proj)

SAFETY_SAMPLES="${SAFETY_SAMPLES:-4994}"
ALIGNMENT_STEP="${ALIGNMENT_STEP:-100}"
FINETUNE_STEP="${FINETUNE_STEP:-900}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

TS=$(date +%Y%m%d_%H%M%S)
TAG=$(echo "$TASKS" | tr ' ' '-')
mkdir -p logs "$OUTPUT_ROOT"
exec > >(tee -a "logs/lisa_rho_sweep_qa_${TAG}_${TS}.log") 2>&1

task_path() {
    case "$1" in
        arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
        medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
        *)      echo "" ;;
    esac
}

[[ -f "$SAFETY_DATA" ]] || { echo "safety data 없음: $SAFETY_DATA" >&2; exit 1; }
for t in $TASKS; do
    p="$(task_path "$t")"
    [[ -n "$p" ]] || { echo "알 수 없는 task: $t (arc|medqa)" >&2; exit 1; }
    [[ -f "$p" ]]  || { echo "task data 없음: $p" >&2; exit 1; }
done

echo "════════════════════════════════════════════════════════════════"
echo " LISA rho sweep   ts=${TS}"
echo "   base model  : $MODEL   (GPU $CUDA_VISIBLE_DEVICES)"
echo "   safety data : $SAFETY_DATA ($SAFETY_SAMPLES)"
echo "   tasks       : $TASKS      rhos: $RHOS      lr: $LR"
echo "   고정         : alignment_step=$ALIGNMENT_STEP finetune_step=$FINETUNE_STEP "\
"r=$LORA_R alpha=$LORA_ALPHA batch=${MICRO_BATCH}x${GRAD_ACCUM} ep=$EPOCHS"
echo "   기준선(rho=1.0): $HF_NAMESPACE/llama2_7b-chat-{task}-lisa-r16-a32-lr${LR}-cb"
echo "   output      : $OUTPUT_ROOT"
echo "════════════════════════════════════════════════════════════════"

failed=()
for task in $TASKS; do
  TASK_DATA="$(task_path "$task")"
  for rho in $RHOS; do
    out_dir="$OUTPUT_ROOT/${task}_lr${LR}_rho${rho}"
    if [[ -f "$out_dir/finetune_config.json" ]]; then
      echo "[${task} rho=$rho] 이미 완료 — skip"; continue
    fi
    echo "──────────────────────────────────────────────────────────────"
    echo "[${task} rho=$rho] start  ($(date +%H:%M:%S))"
    mkdir -p "$out_dir"
    push_args=()
    [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--upload_name \
      "${HF_NAMESPACE}/llama2_7b-chat-${task}-lisa-r16-a32-lr${LR}-cb-rho${rho}")

    if "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
        --model_path "$MODEL" \
        --output_dir "$out_dir" \
        --task_data_path "$TASK_DATA" \
        --num_eval_samples 0 \
        --safety_data_path "$SAFETY_DATA" \
        --guide_data_num "$SAFETY_SAMPLES" \
        --rho "$rho" \
        --alignment_step "$ALIGNMENT_STEP" \
        --finetune_step "$FINETUNE_STEP" \
        --lora \
        --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
        --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
        --learning_rate "$LR" --epochs "$EPOCHS" \
        --batch_size "$MICRO_BATCH" --grad_accum "$GRAD_ACCUM" \
        --max_length "$MAX_LENGTH" \
        --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
        --lr_scheduler_type cosine \
        --seed "$SEED" --bf16 --report_to none \
        "${push_args[@]}" 2>&1 | tee "$out_dir/run.log"; then
      echo "[${task} rho=$rho] done  ($(date +%H:%M:%S))"
    else
      echo "[${task} rho=$rho] FAILED"; failed+=("${task}/rho${rho}")
    fi
  done
done

echo ""
echo "════════════════════ summary ════════════════════"
for f in $(find "$OUTPUT_ROOT" -name finetune_config.json | sort); do
    "$PY" -c "
import json,sys
d=json.load(open('$f'))
print(f\"  rho={d['rho']:<8} loss=?  {d['dataset'].split('/')[-1]:<36} {'$f'.rsplit('/',2)[1]}\")"
done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "실패: ${failed[*]}"; exit 1
fi
echo "완료. 평가는 arc_eval / medqa_eval 하네스에서 별도 수행할 것."
