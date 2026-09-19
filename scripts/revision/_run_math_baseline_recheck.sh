#!/usr/bin/env bash
# Llama-3.1-8B-Instruct / MATH 의 Full FT 기준행을 이 박스의 같은 파이프라인으로 재측정.
# RESULTS.md 의 0.1212 는 구 박스 측정값이라, 이번 AsFT(0.1988)/Lisa(0.1692) 와 비교 가능한지
# 확인해야 한다. 같은 섹션의 LoRA 행(0.2456/0.2530)도 full FT 의 두 배라 의심스럽다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/eval_exp1/mathbase.pid
REPOS_ONLY="kmseong/llama3_1_8b_instruct_MATH_lr5e-5" bash scripts/revision/32_eval_cells.sh
echo "MATHBASE_EXIT=$?"
