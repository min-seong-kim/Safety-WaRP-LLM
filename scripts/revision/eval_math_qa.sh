#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  base 모델 MATH 평가 — 학습 포맷 그대로 0-shot (lm-eval task `hendrycks_math_qa`)
#
#  왜: 기본 `hendrycks_math_safe` 는 "Problem: ...\nAnswer:" 5-shot 이고 few-shot 예시가 짧은
#  답이라, base 모델이 학습한 "풀이 + Final Answer: $x$" 대신 짧은 답으로 끌려간다
#  (8B p00 algebra 에서 83%). 이 task 는 학습 프롬프트 `Question: {q}\nAnswer:` 로 0-shot
#  생성하고 "Final Answer: $...$" 를 추출한다. 정의: lm-evaluation-harness/lm_eval/tasks/hendrycks_math_qa/
#
#  lm_eval 호출 인자는 eval_models.sh 와 같다(vLLM · seed 42 · batch-invariant · FLASH_ATTN ·
#  enforce_eager · prefix caching off · util 0.4). eval_models.sh 에 key 를 추가하지 않은 이유:
#  작성 시점에 그 스크립트가 실행 중이었다(실행 중인 bash 스크립트는 편집 금지).
#
#  사용:  GPU=0 bash scripts/revision/eval_math_qa.sh <model_path_or_repo> [...]
#  결과:  lm-evaluation-harness/eval_results_math_qa/<이름>/results_*.json  (있으면 건너뜀)
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
MINSEONG=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong
LMEVAL_DIR="$MINSEONG/lm-evaluation-harness"
OUT_ROOT="${OUT_ROOT:-$LMEVAL_DIR/eval_results_math_qa}"
GPU="${GPU:-0}"
UTIL="${UTIL:-0.4}"

unset CONDA_EXE CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER
export HF_HOME="$MINSEONG/.hf_cache" TMPDIR="$MINSEONG/.tmp"
export PYTHONHASHSEED=42 VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 VLLM_USE_DEEP_GEMM=0
export VLLM_LOGGING_LEVEL=ERROR VLLM_USE_STANDALONE_COMPILE=0 CUDA_VISIBLE_DEVICES="$GPU"
LM_EVAL="$MINSEONG/miniconda3/envs/hb/bin/lm_eval"

cd "$LMEVAL_DIR"
for M in "$@"; do
  name="$(basename "$M")"
  out="$OUT_ROOT/$name"
  if ls "$out"/*/results_*.json >/dev/null 2>&1 || ls "$out"/results_*.json >/dev/null 2>&1; then
    echo "[skip] $name (결과 있음)"; continue
  fi
  mkdir -p "$out"
  echo "[$(date '+%F %T')] ▶ $name (GPU$GPU)"
  "$LM_EVAL" --model vllm \
    --model_args "pretrained=$M,seed=42,attention_backend=FLASH_ATTN,enforce_eager=True,enable_prefix_caching=False,tensor_parallel_size=1,data_parallel_size=1,dtype=auto,gpu_memory_utilization=$UTIL" \
    --tasks hendrycks_math_qa --include_path lm_eval/tasks \
    --seed 42 --batch_size 128 --log_samples --output_path "$out" > "$out/run.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] $( [ $rc -eq 0 ] && echo ✔ || echo "❌ rc=$rc" ) $name  $(grep -E '^\|hendrycks_math_qa +\|' "$out/run.log" | head -1)"
done
