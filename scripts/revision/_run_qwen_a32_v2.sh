#!/usr/bin/env bash
# Qwen2.5-7B-Instruct / GSM8K 의 AsFT · Lisa 를 α=32(scaling 2) 로 재학습한다.
# 기존 α=32 세대(8/28 asft · 9/1 lisa)와 리포명이 겹치지 않게 HF_REPO_SUFFIX=_v2 를 쓴다.
# 설정은 revisioning_wsr.tex Appendix A 의 LoRA 설정에서 alpha 만 32 로 바꾼 것이다:
#   r=16 · α=32 · dropout 0.05 · targets q,k,v,up,down · lr 3e-4 · 3ep ·
#   eff.batch 16 · max_len 1024 · seed 42 · bf16 · cosine · wd 0 · warmup 0.03
#   AsFT λ=1.0 · Lisa ρ=1.0 / align 100 · ft 900
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

export OUT_ROOT="$PWD/outputs/revision_qwen_a32_v2"
export LOG_ROOT="$PWD/logs/revision_qwen_a32_v2"
export HF_REPO_SUFFIX=_v2
export SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS="asft lisa"
export LISA_RHO=1.0
export LORA_ALPHA=32
export PUSH_TO_HUB=1
# 이 박스는 다운로드가 느리다(≈9MB/s). 평가를 로컬 merged_model 에서 하려고 남겨둔다.
export PRUNE_AFTER_UPLOAD=0 PRUNE_HF_CACHE=0 PRUNE_BASIS=0

bash scripts/revision/20_lora_family.sh
echo "QWEN_A32_V2_TRAIN_DONE rc=$?"
