#!/usr/bin/env bash
#
# SafeLoRA threshold 스윕 (ARC-Challenge / MedQA) — safety↓ downstream↑ 방향 탐색.
# LISA rho 스윕(scripts/run_lisa_rho_sweep_qa.sh)의 SafeLoRA 대응판.
#
# 기준선: scripts/run_lisa_safelora_asft_qa.sh 의 SafeLoRA run (threshold=0.35)
#         → kmseong/llama2_7b-chat-{arc,medqa}-safelora-r16-a32-lr3e-4-cb
#
# SafeLoRA 에서 rho 에 해당하는 손잡이가 threshold 다:
#     각 LoRA 레이어에 대해 cos = cos((C·B)A, BA) 를 재고,
#     cos ≤ threshold 인 레이어만 B ← C·B 로 투영한다.
#   → threshold 를 낮추면 투영되는 레이어가 줄어든다 = 안전 개입 축소
#     = safety↓ downstream↑ (요청 방향). 올리면 반대.
#
# 관측된 cos 분포 (이 레포의 sst2/agnews run): 대략 0.16 ~ 0.72,
# threshold 0.35 에서 160개 중 27~34개 투영. 그래서 0.15/0.25 는 투영 레이어를
# 유의미하게 줄이는 지점이고, 0.5 는 늘리는 반대편 점이다.
#
# ⚠️ 비효율 주의: SafeLoRA 는 학습(표준 LoRA) 후 **사후** 투영이라 threshold 를 바꿔도
#    학습 부분은 완전히 동일하다. 원칙적으로는 한 번 학습한 adapter 를 재사용해
#    투영만 여러 번 하면 되는데, 현재 러너가 투영 전 adapter 를 저장하지 않아
#    (merged dense 만 저장) threshold 마다 재학습한다. ARC 1분 / MedQA 17분 수준이라
#    지금은 감수하지만, 점을 많이 찍을 거면 adapter 저장 경로를 먼저 만드는 게 낫다.
#
# 사용:
#   bash scripts/run_safelora_thr_sweep_qa.sh
#   THRS="0.15 0.25 0.5" TASKS=medqa bash scripts/run_safelora_thr_sweep_qa.sh
#   PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/run_safelora_thr_sweep_qa.sh
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
SAFELORA_BASE_MODEL="${SAFELORA_BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
SAFELORA_ALIGNED_MODEL="${SAFELORA_ALIGNED_MODEL:-$MODEL}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"

TASKS="${TASKS:-arc medqa}"
THRS="${THRS:-0.15 0.25}"       # 기준선 0.35 보다 낮은 쪽 = safety↓ downstream↑
LR="${LR:-3e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/safelora_thr_sweep_qa}"

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

# 디스크가 빠듯하면(<MIN_FREE_GB) 새 run 을 시작하지 않고 멈춘다 —
# 학습 도중 ENOSPC 로 죽어 13GB 쓰레기를 남기는 것보다 낫다.
MIN_FREE_GB="${MIN_FREE_GB:-20}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

TS=$(date +%Y%m%d_%H%M%S)
TAG=$(echo "$TASKS" | tr ' ' '-')
mkdir -p logs "$OUTPUT_ROOT"
exec > >(tee -a "logs/safelora_thr_sweep_qa_${TAG}_${TS}.log") 2>&1

task_path() {
    case "$1" in
        arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
        medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
        # 분류 태스크도 같은 축으로 돌릴 수 있게 열어 둔다 (run_lisa_safelora_cls.sh 와 동일 데이터).
        sst2)   echo "$REPO_DIR/data/sst2_train_8k_seed42.json" ;;
        agnews) echo "$REPO_DIR/data/agnews_train_8k_seed42.json" ;;
        *)      echo "" ;;
    esac
}

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

[[ -f "$SAFETY_DATA" ]] || { echo "safety data 없음: $SAFETY_DATA" >&2; exit 1; }
for t in $TASKS; do
    p="$(task_path "$t")"
    [[ -n "$p" ]] || { echo "알 수 없는 task: $t (arc|medqa|sst2|agnews)" >&2; exit 1; }
    [[ -f "$p" ]]  || { echo "task data 없음: $p" >&2; exit 1; }
done

echo "════════════════════════════════════════════════════════════════"
echo " SafeLoRA threshold sweep   ts=${TS}"
echo "   base model   : $MODEL   (GPU $CUDA_VISIBLE_DEVICES)"
echo "   safelora base: $SAFELORA_BASE_MODEL"
echo "   tasks        : $TASKS      thresholds: $THRS      lr: $LR"
echo "   기준선(thr=0.35): $HF_NAMESPACE/llama2_7b-chat-{task}-safelora-r16-a32-lr${LR}-cb"
echo "   output       : $OUTPUT_ROOT      free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

failed=()
for task in $TASKS; do
  TASK_DATA="$(task_path "$task")"
  for thr in $THRS; do
    out_dir="$OUTPUT_ROOT/${task}_lr${LR}_thr${thr}"
    if [[ -f "$out_dir/summary.json" ]]; then
      echo "[${task} thr=$thr] 이미 완료 — skip"; continue
    fi
    avail=$(free_gb)
    if [[ "$avail" -lt "$MIN_FREE_GB" ]]; then
      echo "[${task} thr=$thr] SKIP — 디스크 부족 (${avail}GB < ${MIN_FREE_GB}GB)"
      failed+=("${task}/thr${thr}(disk)"); continue
    fi
    echo "──────────────────────────────────────────────────────────────"
    echo "[${task} thr=$thr] start  ($(date +%H:%M:%S), free ${avail}GB)"
    mkdir -p "$out_dir"
    push_args=()
    [[ "$PUSH_TO_HUB" == "1" ]] && push_args=(--push_to_hub --hf_repo_id \
      "${HF_NAMESPACE}/llama2_7b-chat-${task}-safelora-r16-a32-lr${LR}-cb-thr${thr}")

    if "$PY" finetune_gsm8k_lora.py \
        --method safe_lora \
        --model_name "$MODEL" \
        --output_dir "$out_dir" \
        --task_data_path "$TASK_DATA" \
        --target_modules "$TARGET_MODULES_CSV" \
        --layer_type "$LAYER_TYPES" --target_layers all \
        --safelora_base_model "$SAFELORA_BASE_MODEL" \
        --safelora_aligned_model "$SAFELORA_ALIGNED_MODEL" \
        --safelora_select_type threshold \
        --safelora_threshold "$thr" \
        --safelora_load_dtype float32 \
        --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
        --learning_rate "$LR" --epochs "$EPOCHS" \
        --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_length "$MAX_LENGTH" \
        --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
        --seed "$SEED" --dtype bfloat16 --save_merged_model \
        "${push_args[@]}" 2>&1 | tee "$out_dir/run.log"; then
      echo "[${task} thr=$thr] done  ($(date +%H:%M:%S))"
    else
      echo "[${task} thr=$thr] FAILED"; failed+=("${task}/thr${thr}")
    fi
  done
done

echo ""
echo "════════════════════ summary (투영된 레이어 수) ════════════════════"
for f in $(find "$OUTPUT_ROOT" -name summary.json | sort); do
    "$PY" -c "
import json
d=json.load(open('$f')); s=d.get('safelora') or {}
print(f\"  thr={s.get('used_threshold'):<6} 투영 {s.get('num_layers_projected')}/{s.get('num_lora_layers')}  \"
      f\"cos[{s.get('cos_min')}, {s.get('cos_max')}]  {'$f'.rsplit('/',2)[1]}\")"
done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "실패/스킵: ${failed[*]}"; exit 1
fi
echo "완료. 평가는 arc_eval / medqa_eval 하네스에서 별도 수행할 것."
