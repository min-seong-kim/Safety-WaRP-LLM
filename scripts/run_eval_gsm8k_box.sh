#!/bin/bash
#SBATCH -J eval_gsm8k
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/eval_gsm8k_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/eval_gsm8k_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition suma_rtx4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# GSM8K 채점 (lm-eval, vllm). eval_models.sh 와 100% 동일 설정을 재현하되, 사용자 파일은 안 건드리고
# 모델 목록을 인자로 받는다. 제출: sbatch scripts/run_eval_gsm8k_box.sh <hf_model_id> [<hf_model_id> ...]
# 결과: /home/gokms0509/lm-evaluation-harness/eval_results/ 하위 JSON. exact_match 는 로그/summary 에.

HARNESS=/home/gokms0509/lm-evaluation-harness
cd "$HARNESS"
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail

# eval_models.sh 와 동일한 재현성/로그 env
export LMEVAL_LOG_LEVEL=ERROR VLLM_LOGGING_LEVEL=ERROR TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
export VLLM_USE_STANDALONE_COMPILE=0 VLLM_USE_DEEP_GEMM=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8" PYTHONHASHSEED=42
EVAL_SEED=42
SUMMARY=/home/gokms0509/Safety-WaRP-LLM/logs/eval_gsm8k_summary_${SLURM_JOB_ID:-manual}.tsv
echo -e "model\tstrict\tflexible" > "$SUMMARY"

for m in "$@"; do
  echo "======== GSM8K: $m ========"
  # eval_models.sh run_task 의 gsm8k 호출을 그대로 재현 (chat 모델 → --apply_chat_template)
  env VLLM_BATCH_INVARIANT=1 lm_eval --model vllm \
    --model_args "pretrained=${m},seed=${EVAL_SEED},attention_backend=TRITON_ATTN,enforce_eager=True,enable_prefix_caching=False,tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=0.85" \
    --tasks gsm8k --device cuda --seed "$EVAL_SEED" --batch_size 32 \
    --output_path eval_results --num_fewshot 5 --log_samples --apply_chat_template \
    2>&1 | tee "/home/gokms0509/Safety-WaRP-LLM/logs/eval_gsm8k_$(echo "$m" | sed 's#.*/##').log" \
    | grep -iE "exact_match|error|OOM" | tail -25
  # summary 추출: 결과 테이블에서 strict-match / flexible-extract 값
  log="/home/gokms0509/Safety-WaRP-LLM/logs/eval_gsm8k_$(echo "$m" | sed 's#.*/##').log"
  strict=$(grep -iE "strict-match" "$log" | grep -oE "[0-9]\.[0-9]+" | head -1)
  flex=$(grep -iE "flexible-extract" "$log" | grep -oE "[0-9]\.[0-9]+" | head -1)
  echo -e "${m}\t${strict:-NA}\t${flex:-NA}" >> "$SUMMARY"
  echo "  → strict=${strict:-NA} flexible=${flex:-NA}"
done
echo "== eval done. summary: $SUMMARY =="
cat "$SUMMARY"
