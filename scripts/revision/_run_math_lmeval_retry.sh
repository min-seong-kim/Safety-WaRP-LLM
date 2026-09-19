#!/usr/bin/env bash
# MATH 기준행의 lm-eval 재시도. 앞선 시도는 vLLM KV cache 부족으로 실패했다
# (max_model_len=131072 → 16GiB 필요, 가용 12.14GiB). HarmBench 단계의 vLLM 이
# 메모리를 덜 놓은 상태에서 이어 붙은 것이 원인으로 보여, GPU 가 빈 상태에서 다시 돌린다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/eval_exp1/mathlm.pid
LM_ONLY=1 REPOS_ONLY="kmseong/llama3_1_8b_instruct_MATH_lr5e-5" bash scripts/revision/32_eval_cells.sh
echo "MATHLM_EXIT=$?"
