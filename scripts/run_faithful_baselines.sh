#!/bin/bash
# Faithful LoRA-family baselines — HANDOFF_lora_experiments.md §10.2 / §8
#
# 배경: 기존에 HF 에 올라간 SaLoRA/LISA 는 "method 간 config 를 맞춘" 버전이라
#       각 논문의 설계 동작점을 벗어나 있었다(§5). 그래서 SaLoRA 는 lr3e-4 에서 safety 가
#       붕괴(ASR 0.69)했는데, 이는 버그가 아니라 동작점 이탈의 결과였다.
#       이 스크립트는 각 method 를 **논문 동작점 그대로** 재실행해 짝비교를 만든다.
#
#   SaLoRA faithful : q_proj,v_proj 만 / alpha == r (scaling 1) / dropout 0.0
#                     (기존: q,k,v,up,down + alpha=2r → scaling 2 로 C 투영을 압도)
#   LISA   faithful : q_proj,v_proj 만 / r=8, alpha=4 / rho=1.0 align=100 ft=900
#                     (기존: q,k,v,up,down + r16/alpha32)
#
# 완료된 run 은 건너뛰므로 죽어도 재실행하면 이어서 간다.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "logs/faithful_baselines_${TS}.log") 2>&1

# ═══════════════════ config ═══════════════════
PY=${PY:-/venv/hb/bin/python}
# ⚠️ CUDA_VISIBLE_DEVICES 를 여기서 설정하지 않는다(스케줄러/단일 GPU 가 알아서 정한다).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# HANDOFF §1: 모든 baseline 의 시작점은 full-param safety FT 모델.
# (adapter_subspace_lora 만 base 모델에서 출발하므로 의미가 다르다 — 섞지 말 것)
BASE_MODEL=${BASE_MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}
SAFETY_DATA="$(pwd)/data/circuit_breakers_train.json"

EPOCHS=3; MAXLEN=1024; SEED=42
OUT_ROOT=${OUT_ROOT:-outputs/faithful_baselines}

# LoRA-family 는 full-FT LR(1e-5/5e-5)이 아니라 1e-4~3e-4 를 써야 한다 (§4).
SALORA_LRS=(1e-4 2e-4 3e-4)
# LISA 는 rho=1.0 의 proximal pull 이 설계상 강해 LR 과 무관하게 downstream 을 거의
# 배우지 못했다(§5). faithful 은 r8/alpha4 라 scaling 이 0.5 로 더 작아 underfit 이
# 심해질 가능성이 높다. 3점을 태우기 전에 1점만 찍어보고 판단한다.
LISA_LRS=(2e-4)

PUSH=${PUSH:-1}
HF_NS=kmseong

RUN_SALORA=${RUN_SALORA:-1}
RUN_LISA=${RUN_LISA:-1}

mkdir -p "$OUT_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo " faithful LoRA-family baselines   ts=${TS}"
echo "   PY          : $PY"
echo "   base model  : $BASE_MODEL"
echo "   safety data : $SAFETY_DATA"
echo "   SaLoRA lrs  : ${SALORA_LRS[*]}   LISA lrs: ${LISA_LRS[*]}"
echo "════════════════════════════════════════════════════════════════"

[ -f "$SAFETY_DATA" ] || { echo "safety data 없음: $SAFETY_DATA"; exit 1; }

# ═══════════ SaLoRA (faithful: q,v only / alpha == r / dropout 0) ═══════════
if [ "$RUN_SALORA" = "1" ]; then
for lr in "${SALORA_LRS[@]}"; do
  odir="${OUT_ROOT}/salora_faithful/lr_${lr}"
  repo="${HF_NS}/llama2_7b-chat-gsm8k-salora-faithful-qv-a16-r16-lr${lr}"
  if [ -f "${odir}/summary.json" ]; then
    echo "[salora:lr${lr}] skip — 이미 완료"; continue
  fi
  echo "──────────────────────────────────────────────────────────────"
  echo "[salora:lr${lr}] faithful SaLoRA (q,v / r16 alpha16 s=1 / dropout 0.0)"
  mkdir -p "$odir"
  push_args=()
  [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")

  "$PY" finetune_gsm8k_salora.py \
    --model_name "$BASE_MODEL" \
    --output_dir "$odir" \
    --safety_data_path "$SAFETY_DATA" \
    --target_modules q_proj,v_proj \
    --layer_type attn_q,attn_v \
    --lora_r 16 --lora_alpha 16 --lora_dropout 0.0 \
    --learning_rate "$lr" --epochs "$EPOCHS" \
    --batch_size 2 --gradient_accumulation_steps 8 \
    --max_length "$MAXLEN" --seed "$SEED" --dtype bfloat16 \
    "${push_args[@]}"
done
fi

# ═══════════ LISA (faithful: q,v only / r8 alpha4 / rho 1.0) ═══════════
# ⚠️ alignment_step/finetune_step 은 micro-batch 단위 카운터라 grad_accum 으로 나누어
#    떨어져야 상태 전환이 accumulation 창을 쪼개지 않는다. 100,900 은 grad_accum=4 에 안전.
if [ "$RUN_LISA" = "1" ]; then
for lr in "${LISA_LRS[@]}"; do
  odir="${OUT_ROOT}/lisa_faithful/lr_${lr}"
  repo="${HF_NS}/llama2_7b-chat-gsm8k-lisa-faithful-qv-r8a4-lr${lr}"
  if [ -f "${odir}/finetune_config.json" ]; then
    echo "[lisa:lr${lr}] skip — 이미 완료"; continue
  fi
  echo "──────────────────────────────────────────────────────────────"
  echo "[lisa:lr${lr}] faithful LISA (q,v / r8 alpha4 / rho 1.0)"
  mkdir -p "$odir"
  up_args=()
  [ "$PUSH" = "1" ] && up_args=(--upload_name "$repo")

  "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
    --model_path "$BASE_MODEL" \
    --output_dir "$odir" \
    --safety_data_path "$SAFETY_DATA" \
    --lora --lora_target_modules q_proj v_proj \
    --lora_r 8 --lora_alpha 4 --lora_dropout 0.05 \
    --learning_rate "$lr" --epochs "$EPOCHS" \
    --batch_size 4 --grad_accum 4 \
    --max_length "$MAXLEN" --seed "$SEED" \
    --rho 1.0 --alignment_step 100 --finetune_step 900 --guide_data_num 4994 \
    --report_to none \
    "${up_args[@]}"
done
fi

echo ""
echo "════════════════════ summary ════════════════════"
for f in $(find "$OUT_ROOT" -name summary.json -o -name finetune_config.json | sort); do
  echo "  $f"
done
echo "완료. 평가는 별도 하네스(lm-evaluation-harness / HarmBench)에서 수행할 것 — HANDOFF §9."
