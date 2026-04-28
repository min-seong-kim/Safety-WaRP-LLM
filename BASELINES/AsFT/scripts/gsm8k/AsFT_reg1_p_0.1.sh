#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ASFT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${ASFT_ROOT}"

export CUDA_VISIBLE_DEVICES=1

# bash scripts/gsm8k/AsFT_reg1_p_0.1.sh> finetuned_logs/gsm8k/AsFT_reg1_p_0.1.log 2>&1 &
echo $CUDA_VISIBLE_DEVICES

BASE_MODEL="meta-llama/Llama-2-7b-hf"
ALIGNED_MODEL="kmseong/llama2_7b-Safety-FT-lr3e-5"

mkdir -p finetuned_logs/gsm8k finetuned_models/gsm8k

python -u AsFT_finetuning.py \
--batch_size_training 8 --lr 5e-5 \
--num_epochs 5 \
--dataset gsm8k_dataset \
--mode 1k_p_0.1 \
--model_name ${ALIGNED_MODEL} \
--base_model_path ${BASE_MODEL} \
--aligned_model_path ${ALIGNED_MODEL} \
--pure_bf16 \
--dist_checkpoint_root_folder finetuned_models \
--dist_checkpoint_folder gsm8k-7b-full \
--output_dir finetuned_models/gsm8k/AsFT_reg1_p_0.1 \
--use_peft True \
--lambda_reg 1 \
--gradient_accumulation_steps 1 --run_validation False --save_every_epoch False

# --enable_fsdp \