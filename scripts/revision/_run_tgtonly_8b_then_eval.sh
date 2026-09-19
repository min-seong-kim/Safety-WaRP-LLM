#!/usr/bin/env bash
# llama3.1-8B 타깃한정 2셀 학습이 끝나면 (7B 2셀 포함) 전체를 평가한다.
# ⚠️ 학습과 평가를 겹치면 안 된다 — HarmBench 가 llama2_7b 에 gpu_memory_utilization 0.95
#    (=174GB) 를 쓰므로 학습 중 실행하면 OOM 이다. 학습 종료를 확인하고 시작한다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_tgtonly/chain.pid

for i in $(seq 1 120); do
  grep -q "ALL_CELLS_FINISHED" logs/asft_lisa_tgtonly/orchestrator2.log 2>/dev/null && break
  sleep 60
done
echo "[chain] 학습 종료 확인 $(date -Is)"

for i in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-0}" -lt 2000 ] && break
  echo "[chain] GPU ${used}MiB 점유 중 — 대기"; sleep 30
done

EVAL_ROOT="$PWD/outputs/asft_lisa_tgtonly" bash scripts/revision/32_eval_cells.sh
echo "CHAIN_EXIT=$?"
