#!/usr/bin/env bash
# 공유 lustre 가 100% 라 다른 사용자 때문에 여유가 급감할 수 있다.
# 500GB 아래로 떨어지면 한 줄 찍고 종료한다(그 시점에 정리 판단).
while true; do
  avail=$(df -BG --output=avail /NHNHOME/26msit001_A/BASE/edge_ai_lab 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -z "$avail" ] && { sleep 300; continue; }
  if [ "$avail" -lt 500 ]; then
    echo "DISK_LOW ${avail}GB $(date -Is)"; exit 0
  fi
  sleep 300
done
