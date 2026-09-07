#!/usr/bin/env bash
# v6 종료 후: GPU 0 이 충분히 비면(≥75GiB) 실패 조합만 RESUME 으로 다시 돈다(최대 3회). GPU 0 전용.
cd ~/Safety-WaRP-LLM
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
SP=$(awk '{print $2}' logs/revision_sweep/sweep.pid)
while kill -0 "$SP" 2>/dev/null; do sleep 120; done
log "v6 종료 감지 → 후속 gap-fill"
for pass in 1 2 3; do
  n=0
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | awk '{printf "%d",$1/1024}')
    [ "$free" -ge 75 ] && break
    n=$((n+1)); [ $((n % 30)) -eq 1 ] && log "   ⏳ GPU0 대기: free ${free}GiB < 75"
    sleep 60
  done
  log "── gap-fill pass $pass (GPU0 free ${free}GiB)"
  out=$(HB_GPU=0 LM_GPU=0 bash logs/revision_sweep/gapfill_eval.sh 2>&1 | tee -a logs/revision_sweep/gapfill.log | tail -400)
  if echo "$out" | grep -q '❌ Failed:'; then log "   실패 조합 남음: $(echo "$out" | grep '❌ Failed:' | tail -1)"; else log "   실패 조합 없음 → 종료"; break; fi
done
log "후속 gap-fill 완료"; echo done > logs/revision_sweep/gapfill.rc
