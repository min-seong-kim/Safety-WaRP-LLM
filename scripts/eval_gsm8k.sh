#!/bin/bash
# 6개 HF 모델에 대해 GSM8K (5-shot, chat template) 를 lm_eval(vllm)로 채점. hb env, GPU 1.
set -uo pipefail
cd /home/users/minseong/lm-evaluation-harness
HBPY=/home/users/minseong/.conda/envs/hb/bin/python
export CUDA_VISIBLE_DEVICES=1
export VLLM_LOGGING_LEVEL=ERROR
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

OUT=/home/users/minseong/Safety-WaRP-LLM/outputs/lora_comparison/eval_gsm8k
mkdir -p "$OUT"
MODELS=(
  "kmseong/llama2_7b-chat-gsm8k-lora-r16-lr1e-4"
  "kmseong/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr1e-4"
  "kmseong/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr1e-4"
  "kmseong/llama2_7b-chat-gsm8k-lora-r16-lr2e-4"
  "kmseong/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr2e-4"
  "kmseong/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr2e-4"
)
for m in "${MODELS[@]}"; do
  tag=$(echo "$m" | sed 's#.*/##')
  echo "======== GSM8K: $m ========"
  $HBPY -m lm_eval --model vllm \
    --model_args "pretrained=${m},dtype=bfloat16,gpu_memory_utilization=0.85,enforce_eager=True,tensor_parallel_size=1,data_parallel_size=1,max_model_len=2048" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template \
    --output_path "${OUT}/${tag}" 2>&1 | tee "${OUT}/${tag}.log" | grep -iE "exact_match|Running|error" | tail -20
done
echo "===== GSM8K eval done. results under ${OUT} ====="
