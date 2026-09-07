#!/usr/bin/env bash
# 06:00(UTC) 까지 v4 가 여전히 'GPU 대기' 상태면 → v4 종료 후 v5(WSR_BASIS_BS=1, WSR_NEED 45) 로 재시작
cd ~/Safety-WaRP-LLM
while [ "$(date +%H%M)" \< "0600" ] || [ "$(date +%H%M)" \> "1200" ]; do sleep 300; done
last=$(grep -E '^\[' logs/revision_sweep/sweep.log | tail -1)
if echo "$last" | grep -q 'GPU 대기'; then
  echo "[fallback] $(date) 아직 대기 중: $last → v5(WSR_BASIS_BS=1) 로 전환"
  SP=$(awk '{print $2}' logs/revision_sweep/sweep.pid); PIDS=$(pstree -p $SP | grep -o '([0-9]*)' | tr -d '()'); for p in $PIDS; do kill -TERM $p 2>/dev/null; done; sleep 3; for p in $PIDS; do kill -KILL $p 2>/dev/null; done
  mv logs/revision_sweep/sweep.log logs/revision_sweep/sweep_v4.log
  ALLOWED_GPUS=0 WSR_BASIS_BS=1 WSR_NEED_OVERRIDE=45 MODELS_ORDER="llama2_13b gemma2_9b qwen25_7b" EVAL_PER_MODEL=0 nohup bash logs/revision_sweep/sweep_v6.sh > logs/revision_sweep/sweep.log 2>&1 &
  echo "pid $!" > logs/revision_sweep/sweep.pid; echo "[fallback] v5 pid $!"
  echo "$(date -Iseconds) v5 started with WSR_BASIS_BS=1 (gemma2_9b/llama2_13b wsr_lora cells use basis_batch_size=1)" >> logs/revision_sweep/DEVIATIONS.txt
else
  echo "[fallback] $(date) 대기 아님(진행 중): $last → 전환하지 않음"
fi
