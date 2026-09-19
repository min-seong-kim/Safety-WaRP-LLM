#!/usr/bin/env bash
# 타깃한정 2차: llama2_13b(gsm8k) / llama32_3b(math) × {AsFT, Lisa} = 4셀 → 학습 후 평가.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_tgtonly/b2.pid
COMBOS="llama2_13b:gsm8k llama32_3b:math" \
TRAIN_ONLY_TARGETS=1 \
EXP1_OUT_ROOT="$PWD/outputs/asft_lisa_tgtonly" \
EXP1_LOG_DIR="$PWD/logs/asft_lisa_tgtonly" \
EXP1_REPO_SUFFIX="_tgtonly" \
  bash scripts/revision/30_asft_lisa_fullft.sh
echo "B2_TRAIN_EXIT=$?"

for i in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-0}" -lt 2000 ] && break
  echo "[b2] GPU ${used}MiB 점유 중 — 대기"; sleep 30
done
EVAL_ROOT="$PWD/outputs/asft_lisa_tgtonly" bash scripts/revision/32_eval_cells.sh
echo "B2_EVAL_EXIT=$?"
