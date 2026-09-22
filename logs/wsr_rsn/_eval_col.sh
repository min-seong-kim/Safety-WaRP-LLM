#!/usr/bin/env bash
# 열 버전 학습 종료 후 models.yaml 등록 + 평가 (행 버전과 동일 조건)
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
HB=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/HarmBench
LM=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness
R=$(pwd)/outputs/wsr_rsn_col
source "$(conda info --base)/etc/profile.d/conda.sh"
export TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.torchinductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

p=$(cat "$L/retry.pid"); while kill -0 "$p" 2>/dev/null; do sleep 60; done
echo "[wait] 열 학습 종료 $(date -Iseconds)"

# 완성된 셀만 등록한다
KEYS=""; MODELS=""
for m in llama2_7b llama2_13b; do
  d="$R/$m/rsn/gsm8k"
  [[ -f "$d/config.json" ]] || { echo "[skip] $m 열 모델 없음"; continue; }
  key="${m}-chat-wsr_rsn_col-gsm8k-lr5e-5"
  if ! grep -q "^${key}:$" "$HB/configs/model_configs/models.yaml"; then
    cat >> "$HB/configs/model_configs/models.yaml" <<EOF

${key}:
  model:
    model_name_or_path: ${d}
    use_fast_tokenizer: False
    dtype: float16
    chat_template: llama-2
    max_model_len: 4096
    gpu_memory_utilization: 0.5
  num_gpus: 1
  model_type: open_source
EOF
    echo "[yaml] 등록 $key"
  fi
  KEYS="${KEYS}${key}"$'\n'; MODELS="$MODELS $d"
done
[[ -n "$KEYS" ]] || { echo "[abort] 평가할 열 모델이 없다"; exit 1; }

echo "########## HarmBench (sys, 4공격) ##########"
conda activate harmbench; cd "$HB"
CUDA_VISIBLE_DEVICES=0 MODELS_OVERRIDE="${KEYS%$'\n'}" \
  STRIP_SAFETY_SYSTEM_PROMPT=0 RESULT_TAG="" SEED=42 RESUME=true PREFETCH=0 \
  bash ./harmbench_eval.sh
conda deactivate

echo "########## lm-eval GSM8K 5-shot ##########"
conda activate hb; cd "$LM"
for d in $MODELS; do
  tag=$(echo "$d" | sed 's#.*/wsr_rsn_col/##; s#/#_#g')
  CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args "pretrained=$d,seed=42,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,tensor_parallel_size=1,enforce_eager=True" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto \
    --output_path "$LM/logs/wsr_rsn_col/$tag" 2>&1 | tail -12
done
echo "########## COL EVAL COMPLETE $(date -Iseconds) ##########"
