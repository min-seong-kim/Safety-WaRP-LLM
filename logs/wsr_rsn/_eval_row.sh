#!/usr/bin/env bash
# 행 버전 7B·13B 평가: HarmBench 4종(AdvBench standard, nosys) + lm-eval GSM8K
# GPU 가 비면 시작한다 (보정 검출이 끝날 때까지 대기).
set -uo pipefail
L=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM/logs/wsr_rsn
HB=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/HarmBench
LM=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness
R=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM/outputs/wsr_rsn_row
KEYS="llama2_7b-chat-wsr_rsn_row-gsm8k-lr5e-5 llama2_13b-chat-wsr_rsn_row-gsm8k-lr5e-5"

# 선행 작업(보정 검출) 종료 대기 — pid 파일만 본다
if [[ -f "$L/orch.pid" ]]; then
  p=$(cat "$L/orch.pid")
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
  echo "[wait] 보정 검출 종료"
fi
# GPU 가 실제로 비는 것까지 확인
for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ "$u" -lt 5000 ]] && break; sleep 20
done
echo "[gpu] 여유 확인 ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

echo "########## 1. HarmBench (4 attacks × 2 models) ##########"
cd "$HB"
CUDA_VISIBLE_DEVICES=0 HB_MODELS="$KEYS" RESUME=true \
  bash harmbench_eval.sh 2>&1 | tail -80
echo "=== HarmBench rc=$? ==="

echo "########## 2. lm-eval GSM8K ##########"
cd "$LM"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate hb
export TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.torchinductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
for m in "$R/llama2_7b/rsn/gsm8k" "$R/llama2_13b/rsn/gsm8k"; do
  tag=$(echo "$m" | sed 's#.*/wsr_rsn_row/##; s#/#_#g')
  echo "---- lm-eval: $tag"
  CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
      --model_args "pretrained=$m,seed=42,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,tensor_parallel_size=1,enforce_eager=True" \
      --tasks gsm8k --batch_size auto --num_fewshot 0 \
      --output_path "$LM/logs/wsr_rsn_row/$tag" 2>&1 | tail -25
done
echo "########## EVAL COMPLETE $(date -Iseconds) ##########"
