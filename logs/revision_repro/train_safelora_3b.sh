#!/usr/bin/env bash
# 재현 실험: SafeLoRA thr=0.35 α=16 (llama32_3b / CB / math) — GPU 0
cd ~/Safety-WaRP-LLM
export CUDA_VISIBLE_DEVICES=0
export PY=$HOME/.conda/envs/hb/bin/python
export PATH=$HOME/.conda/envs/hb/bin:$PATH
export TRITON_CACHE_DIR=$HOME/.triton/cache TORCHINDUCTOR_CACHE_DIR=$HOME/.torchinductor_cache
export MODELS=llama32_3b SAFETY_SETS=cb TASKS=math METHODS=safelora
export LORA_ALPHA=16 SAFELORA_THRESHOLD=0.35
export OUT_ROOT=$PWD/outputs/revision_repro PUSH_TO_HUB=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0
export STAGES="20"
echo "[repro] start $(date)"
bash scripts/revision/run_all.sh
rc=$?
echo "[repro] run_all rc=$rc  $(date)"
echo "$rc" > logs/revision_repro/train.rc
