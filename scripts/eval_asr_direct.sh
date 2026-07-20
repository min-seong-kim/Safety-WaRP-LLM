#!/bin/bash
# 6개 모델 Direct(=DirectRequest, AdvBench) ASR 채점. HarmBench, harmbench conda env, GPU 1.
# HarmBench classifier(cais/HarmBench-Llama-2-13b-cls) 로 grading (논문 기준).
set -uo pipefail
source "$(conda info --base 2>/dev/null || echo /home/users/minseong/.conda)/etc/profile.d/conda.sh"
conda activate harmbench
cd /home/users/minseong/HarmBench

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS_OVERRIDE=$'llama2_7b-gsm8k-lora-lr1e-4\nllama2_7b-gsm8k-origproj-lr1e-4\nllama2_7b-gsm8k-wsr-lr1e-4\nllama2_7b-gsm8k-lora-lr2e-4\nllama2_7b-gsm8k-origproj-lr2e-4\nllama2_7b-gsm8k-wsr-lr2e-4'
export MODELS_OVERRIDE
export METHODS_OVERRIDE="DirectRequest"
export CLS_PATHS_OVERRIDE="cais/HarmBench-Llama-2-13b-cls"
export GRADING_OVERRIDE="classifier"

bash ./harmbench_eval.sh
echo "===== ASR Direct eval done ====="
