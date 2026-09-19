#!/usr/bin/env bash
# 재학습(_rerun) 2셀 평가. 업로드가 끝날 때까지 기다린 뒤 시작한다.
# ⚠️ 이전 평가의 자식 프로세스가 GPU 를 붙들고 있으면 vLLM 이 KV cache 부족으로 죽는다
#    (2026-09-19 실측: rc=0 으로 끝난 뒤에도 10개 프로세스가 59GB 점유). 먼저 확인한다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_rerun/eval.pid

for i in $(seq 1 60); do
  [ -f outputs/asft_lisa_rerun/llama31_8b/math/lisa/.uploaded ] && break
  grep -q "ALL_CELLS_FINISHED" logs/asft_lisa_rerun/orchestrator.log 2>/dev/null && break
  sleep 60
done
echo "[rerun-eval] 업로드 확인 완료 $(date -Is)"

# GPU 가 비었는지 확인 (점유 중이면 최대 10분 대기)
for i in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-0}" -lt 2000 ] && break
  echo "[rerun-eval] GPU ${used}MiB 점유 중 — 대기"; sleep 30
done

EVAL_ROOT="$PWD/outputs/asft_lisa_rerun" bash scripts/revision/32_eval_cells.sh
echo "RERUN_EVAL_EXIT=$?"
