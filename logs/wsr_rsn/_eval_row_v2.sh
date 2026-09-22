#!/usr/bin/env bash
# 행 버전 7B·13B 평가 — 저장소 기존 표와 **동일 조건**으로 맞춘다.
#   HarmBench : conda env `harmbench`, AdvBench standard, 4공격,
#               **sys 모드**(STRIP_SAFETY_SYSTEM_PROMPT=0, RESULT_TAG="" ),
#               GRADING=hard(keyword, harmbench_eval.sh 기본), SEED=42
#   lm-eval   : conda env `hb`, gsm8k **5-shot** (eval_models.sh 의 TASK_FEWSHOT[gsm8k]=5),
#               결과는 exact_match,flexible-extract 를 읽는다
#   PREFETCH=0 : 로컬 경로 모델이라 HF 다운로드 단계를 건너뛴다
set -uo pipefail
L=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM/logs/wsr_rsn
HB=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/HarmBench
LM=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness
R=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM/outputs/wsr_rsn_row
KEYS=$'llama2_7b-chat-wsr_rsn_row-gsm8k-lr5e-5\nllama2_13b-chat-wsr_rsn_row-gsm8k-lr5e-5'

source "$(conda info --base)/etc/profile.d/conda.sh"
# /tmp 가 noexec 라 Triton 이 .so 를 mmap 못 한다 → 캐시를 HOME 으로 (CLAUDE.md 함정)
export TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.torchinductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# 선행 작업 종료 대기 (pid 파일만 본다 — pgrep 금지)
for f in "$L/orch.pid"; do
  [[ -f "$f" ]] || continue; p=$(cat "$f")
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
  echo "[wait] $f 종료"
done
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ "$u" -lt 5000 ]] && break; sleep 20
done
echo "[gpu] $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "########## 1. HarmBench (sys 모드, 4공격 × 2모델) ##########"
conda activate harmbench
echo "[env] $(python -c 'import sys;print(sys.executable)')"
cd "$HB"
CUDA_VISIBLE_DEVICES=0 \
  MODELS_OVERRIDE="$KEYS" \
  STRIP_SAFETY_SYSTEM_PROMPT=0 RESULT_TAG="" \
  SEED=42 RESUME=true PREFETCH=0 \
  bash ./harmbench_eval.sh
echo "=== HarmBench rc=$? ==="
conda deactivate

echo "########## 2. lm-eval GSM8K (5-shot) ##########"
conda activate hb
echo "[env] $(python -c 'import sys;print(sys.executable)')"
cd "$LM"
for m in "$R/llama2_7b/rsn/gsm8k" "$R/llama2_13b/rsn/gsm8k"; do
  tag=$(echo "$m" | sed 's#.*/wsr_rsn_row/##; s#/#_#g')
  echo "---- lm-eval: $tag"
  CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
      --model_args "pretrained=$m,seed=42,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,tensor_parallel_size=1,enforce_eager=True" \
      --tasks gsm8k --num_fewshot 5 --batch_size auto \
      --output_path "$LM/logs/wsr_rsn_row/$tag" 2>&1 | tail -20
done
echo "########## EVAL COMPLETE $(date -Iseconds) ##########"
