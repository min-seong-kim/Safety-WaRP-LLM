#!/usr/bin/env bash
# 실험 1 백그라운드 런처. setsid 로 독립 세션을 만들어 부모 셸과 프로세스 그룹을 분리한다
# (분리하지 않으면 `kill -- -PID` 가 런처를 띄운 셸까지 죽인다 — 2026-09-19 실측).
# 감시는 PID 파일로 한다: pgrep -f <script> 는 감시자 자신도 잡는다(CLAUDE.md).
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_fullft/orch.pid
export HF_HUB_ENABLE_HF_TRANSFER=0
bash scripts/revision/30_asft_lisa_fullft.sh
echo "EXP1_ORCHESTRATOR_EXIT=$?"
