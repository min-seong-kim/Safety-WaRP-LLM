#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  무인 운영 워치독 — 이벤트를 stdout 으로 흘리고, 알려진 고장은 스스로 고친다.
#
#  자동 복구: `hf download` 가 N분간 단 1바이트도 못 읽으면 SIGKILL 한다.
#    (2026-09-07 실측: 소켓 11개를 쥔 채 CPU 5초만 쓰고 15분 정지 → 죽이면
#     prefetch 재시도 루프가 이어받아 11.8MiB/s 로 복구됨)
#    판정은 /proc/PID/io 의 rchar 증가분으로 한다. 프로세스 이름 매칭이 아니라
#    실제 전송량이므로 "느린 것"과 "멈춘 것"을 혼동하지 않는다.
# ════════════════════════════════════════════════════════════════════════════
R=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
STALL_TICKS=${STALL_TICKS:-10}      # 60초 × 10 = 10분 무전송이면 정지로 판정
TICK=${TICK:-60}
SELF=$$
declare -A rchar_prev stall off
EV='\[revision\] ?\[(done|fail|WARN|ERROR)|############|\[chain\]|업로드/검증 실패|실패한 셀|Traceback|CUDA out of memory|No space left|quota|401 Client|403 Client|RepositoryNotFound'

logs() { cat "$R/logs/qa_a16.logpath" "$R/logs/bt_then_eval.logpath" 2>/dev/null; }
for L in $(logs); do off[$L]=$(stat -c %s "$L" 2>/dev/null || echo 0); done

echo "[watchdog] 시작 $(date '+%F %T')  (정지판정 ${STALL_TICKS}틱 × ${TICK}초)"
while true; do
    # ── 1) 로그 신규 구간에서 이벤트만 추출 ───────────────────────────────
    for L in $(logs); do
        [ -f "$L" ] || continue
        cur=$(stat -c %s "$L" 2>/dev/null || echo 0)
        prev=${off[$L]:-0}
        if [ "$cur" -gt "$prev" ]; then
            tail -c +$((prev+1)) "$L" 2>/dev/null | tr '\r' '\n' \
              | grep -aE "$EV" | grep -avE '^\s*$' | cut -c1-220 | head -20
            off[$L]=$cur
        elif [ "$cur" -lt "$prev" ]; then
            off[$L]=$cur
        fi
    done

    # ── 2) 정지한 hf download 를 죽여 재시도를 유도 ───────────────────────
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$SELF" ] && continue
        cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
        case "$cmd" in *"hf download"*) ;; *) continue ;; esac
        r=$(awk '/^rchar/{print $2}' /proc/$pid/io 2>/dev/null); [ -z "$r" ] && continue
        if [ "${rchar_prev[$pid]:-x}" = "$r" ]; then
            stall[$pid]=$(( ${stall[$pid]:-0} + 1 ))
            if [ "${stall[$pid]}" -ge "$STALL_TICKS" ]; then
                echo "[watchdog] ⚠️ hf download(pid=$pid) $((STALL_TICKS*TICK/60))분간 전송 0바이트 → SIGKILL, 재시도 유도"
                kill -9 "$pid" 2>/dev/null
                unset 'stall[$pid]' 'rchar_prev[$pid]'
                continue
            fi
        else
            stall[$pid]=0
        fi
        rchar_prev[$pid]=$r
    done

    # ── 3) 전체 파이프라인 종료 감지 ──────────────────────────────────────
    CH="$(cat "$R/logs/bt_then_eval.logpath" 2>/dev/null)"
    if [ -n "$CH" ] && grep -qaF '############ 전체 종료' "$CH" 2>/dev/null; then
        echo "[watchdog] ✅ 전체 파이프라인 종료 감지 — 워치독 종료"; exit 0
    fi
    sleep "$TICK"
done
