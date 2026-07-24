#!/bin/bash
# LISA (Bi-State Optimization) 로 GSM8K downstream fine-tuning.
#   - base model : safety-aligned Llama2-7B-chat
#   - downstream : GSM8K (clean, no poison)
#   - alignment  : circuit_breakers_train.json 의 안전 응답(llama3_output)
#   - tuning     : LoRA
#
# 사용법:  bash run_lisa_gsm8k.sh
# GPU 는 CUDA_VISIBLE_DEVICES 로 지정 (미지정 시 스크립트 기본값 "2,3").
set -e

cd "$(dirname "$0")"

# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

MODEL_PATH=${MODEL_PATH:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}
SAFETY_DATA=${SAFETY_DATA:-/home/gokms0509/Safety-WaRP-LLM/data/circuit_breakers_train.json}
OUTPUT_DIR=${OUTPUT_DIR:-./lisa_gsm8k_llama2_7b_chat_lr5e-5}

# ---- LISA / BSO hyper-parameters ----
RHO=${RHO:-1.0}                 # proximal term 계수
ALIGN_STEP=${ALIGN_STEP:-100}   # 한 alignment status 의 스텝 수
FT_STEP=${FT_STEP:-900}         # 한 finetune status 의 스텝 수
GUIDE_NUM=${GUIDE_NUM:-4994}    # alignment 데이터 개수 (circuit_breakers 전체)

# ---- 학습 세팅 (참고: finetune_gsm8k_full_params.py) ----
LR=${LR:-5e-5}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAXLEN=${MAXLEN:-1024}

conda run -n hb --no-capture-output python finetune_gsm8k_lisa.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --safety_data_path "${SAFETY_DATA}" \
    --learning_rate "${LR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_length "${MAXLEN}" \
    --lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --rho "${RHO}" \
    --alignment_step "${ALIGN_STEP}" \
    --finetune_step "${FT_STEP}" \
    --guide_data_num "${GUIDE_NUM}" \
    --report_to none
