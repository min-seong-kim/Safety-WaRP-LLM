#!/usr/bin/env bash
# 실험 2 런처 — 원공간 동결 스윕, base 라인 (llama2_7b_base / llama31_8b_base × ρ 10~50%).
# setsid 로 독립 세션(부모 셸과 프로세스 그룹 분리). 감시는 PID 파일로 한다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/origspace_base/orch.pid
export HF_HUB_ENABLE_HF_TRANSFER=0
bash scripts/revision/31_origspace_base.sh
echo "EXP2_ORCHESTRATOR_EXIT=$?"
