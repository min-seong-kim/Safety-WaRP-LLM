#!/usr/bin/env bash
# 재현 실험: WSR-LoRA α=16 ρ=0.3 (llama2_7b / CB / gsm8k) — GPU 0 단독 사용
cd ~/Safety-WaRP-LLM
export CUDA_VISIBLE_DEVICES=0
export PY=$HOME/.conda/envs/hb/bin/python
export PATH=$HOME/.conda/envs/hb/bin:$PATH
export TRITON_CACHE_DIR=$HOME/.triton/cache TORCHINDUCTOR_CACHE_DIR=$HOME/.torchinductor_cache
export MODELS=llama2_7b SAFETY_SETS=cb TASKS=gsm8k METHODS=wsr_lora
export KEEP_RATIO=0.3 WSR_LORA_ALPHA=16
export OUT_ROOT=$PWD/outputs/revision_repro PUSH_TO_HUB=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0
export STAGES="02 20"
echo "[repro] start $(date)"
bash scripts/revision/run_all.sh
rc=$?
echo "[repro] run_all rc=$rc  $(date)"
echo "$rc" > logs/revision_repro/train.rc
