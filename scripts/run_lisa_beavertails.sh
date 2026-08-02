#!/usr/bin/env bash
#
# LISA (rho=0, alternation only) on GSM8K — safety data 를 beavertails 로 교체.
#
# 기준선: kmseong/llama2_7b-chat-gsm8k-lisa-cb-r16a32-lr3e-4-ep3-rho0-alt
#         (scripts/run_lisa_diagnose.sh 의 "rho0-alt" run, safety=circuit_breakers)
#
# 이 스크립트는 그 run 과 **safety 데이터 하나만** 다르게 한다:
#     circuit_breakers_train.json  →  beavertails_cb_train.json
# 두 파일은 스키마(prompt / llama3_output)도 행 수(4994)도 같아 그대로 교체된다.
# 나머지 플래그(gradient_checkpointing 포함)는 기준선과 비트 단위로 동일하게 유지 —
# 하나라도 흔들면 "safety 데이터 축의 효과"라는 대조가 깨진다.
#
# rho=0 인 이유: proximal 은 rho=0.1 만으로도 무제약 drift 의 99.94% 를 제거해 downstream
# 학습을 막는다(run_lisa_align_sweep.sh 주석). 그래서 이 계열은 rho=0 으로 고정하고
# alternation 만 남긴다.
#
# 사용:
#   bash scripts/run_lisa_beavertails.sh
#   PUSH=0 bash scripts/run_lisa_beavertails.sh        # 업로드 없이
set -uo pipefail          # -e 는 쓰지 않는다: 업로드가 실패해도 학습 결과는 남긴다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
# 이 박스에는 SLURM 이 없다 → GPU 를 직접 고른다.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

NS="${HF_NAMESPACE:-kmseong}"
MODEL="${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"   # 기준선과 동일한 출발점
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/beavertails_cb_train.json}"
LR="${LR:-3e-4}"
EPOCHS="${EPOCHS:-3}"
RHO="${RHO:-0}"
TAG="${TAG:-rho0-alt}"
PUSH="${PUSH:-1}"
OUT="${OUT:-$REPO_DIR/outputs/lisa_beavertails/$TAG}"

# 기준선의 -cb- 자리에 -bt- 를 넣어 safety 데이터 축이 이름에서 바로 보이게 한다.
REPO_ID="${REPO_ID:-$NS/llama2_7b-chat-gsm8k-lisa-bt-r16a32-lr${LR}-ep${EPOCHS}-${TAG}}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUT"
exec > >(tee -a "logs/lisa_beavertails_${TS}.log") 2>&1

[ -f "$SAFETY_DATA" ] || { echo "safety data 없음: $SAFETY_DATA"; exit 1; }

echo "════════════════════════════════════════════════════════════════"
echo " LISA (rho=$RHO, alternation only) · GSM8K · safety=beavertails"
echo "   base model  : $MODEL"
echo "   safety data : $SAFETY_DATA"
echo "   lr / epochs : $LR / $EPOCHS      GPU: $CUDA_VISIBLE_DEVICES"
echo "   output      : $OUT"
echo "   HF repo     : $([ "$PUSH" = "1" ] && echo "$REPO_ID" || echo '(업로드 안 함)')"
echo "   기준선      : $NS/llama2_7b-chat-gsm8k-lisa-cb-r16a32-lr3e-4-ep3-rho0-alt"
echo "════════════════════════════════════════════════════════════════"

if [ -f "$OUT/finetune_config.json" ]; then
    echo "이미 완료 — skip ($OUT)"; exit 0
fi

push_args=()
[ "$PUSH" = "1" ] && push_args=(--upload_name "$REPO_ID")

"$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
    --model_path "$MODEL" \
    --output_dir "$OUT" \
    --dataset_name openai/gsm8k --dataset_subset main --train_split train \
    --num_train_samples 7473 --num_eval_samples 0 \
    --safety_data_path "$SAFETY_DATA" \
    --guide_data_num 4994 \
    --rho "$RHO" --alignment_step 100 --finetune_step 900 \
    --lora --lora_target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --learning_rate "$LR" --epochs "$EPOCHS" \
    --batch_size 16 --grad_accum 1 \
    --max_length 1024 --warmup_ratio 0.03 --weight_decay 0.0 \
    --lr_scheduler_type cosine --seed 42 \
    --bf16 --gradient_checkpointing --report_to none \
    "${push_args[@]}" 2>&1 | tee "$OUT/run.log"

echo "완료: $OUT"
