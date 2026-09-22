#!/usr/bin/env bash
# 공개된 원공간 RSN-Tune 모델 2개를 **행 버전과 완전히 동일한 조건**으로 재평가.
#   conda env        : harmbench (vLLM 0.16)  ← harmbench_eval.sh 는 스스로 activate 하지 않는다
#   sys 모드         : STRIP_SAFETY_SYSTEM_PROMPT=0, RESULT_TAG=""  (RESULTS.md 전 표의 규약)
#   GRADING=hard     : harmbench_eval.sh 기본값 (refusal keyword)
#   데이터셋/공격    : AdvBench standard · Direct/AutoDAN/PAIR/PAP · SEED=42
#   lm-eval          : conda hb · gsm8k 5-shot · exact_match,flexible-extract
#   PREFETCH=0 + HF_HUB_OFFLINE=1 : huggingface.co 차단 상태, 두 모델은 캐시에 있다
set -uo pipefail
HB=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/HarmBench
LM=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness
KEYS=$'llama2_7b-chat-gsm8k-ft_freeze-rsn-lr5e-5\nllama2_13b-chat-rsn-tune-gsm8k-lr5e-5'
REPOS="kmseong/llama2_7b_chat_gsm8k_ft_freeze_rsn_lr5e-5_new_revised wvnvwn/llama-2-13b-chat-hf-gsm8k-rsn-tuned-lr5e-5"

source "$(conda info --base)/etc/profile.d/conda.sh"
export TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.torchinductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "########## 1. HarmBench (conda harmbench, sys 모드) ##########"
conda activate harmbench
echo "[env] $(python -c 'import sys;print(sys.executable)')"
echo "[env] transformers $(python -c 'import transformers;print(transformers.__version__)') / vllm $(python -c 'import vllm;print(vllm.__version__)')"
cd "$HB"
CUDA_VISIBLE_DEVICES=0 \
  MODELS_OVERRIDE="$KEYS" \
  STRIP_SAFETY_SYSTEM_PROMPT=0 RESULT_TAG="" \
  SEED=42 RESUME=true PREFETCH=0 \
  bash ./harmbench_eval.sh
echo "=== HarmBench rc=$? ==="
conda deactivate

echo "########## 2. lm-eval GSM8K 5-shot (conda hb) ##########"
conda activate hb
cd "$LM"
for r in $REPOS; do
  tag=$(echo "$r" | tr '/' '_')
  echo "---- lm-eval: $tag"
  CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args "pretrained=$r,seed=42,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,tensor_parallel_size=1,enforce_eager=True" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto \
    --output_path "$LM/logs/published_rsn/$tag" 2>&1 | tail -12
done
echo "########## PUBLISHED RSN EVAL COMPLETE $(date -Iseconds) ##########"
