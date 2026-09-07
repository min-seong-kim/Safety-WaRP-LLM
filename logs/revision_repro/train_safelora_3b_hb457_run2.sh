#!/usr/bin/env bash
# 재현 실험 2: 같은 셀을 원본 박스 환경(hb_repro: torch 2.10 / transformers 4.57.3 / peft 0.18.1)으로 재학습 — GPU 0
cd ~/Safety-WaRP-LLM
export CUDA_VISIBLE_DEVICES=0
export PY=$HOME/.conda/envs/hb_repro/bin/python
export PATH=$HOME/.conda/envs/hb_repro/bin:$PATH
export TRITON_CACHE_DIR=$HOME/.triton/cache_hb_repro TORCHINDUCTOR_CACHE_DIR=$HOME/.torchinductor_cache_hb_repro
export MODELS=llama32_3b SAFETY_SETS=cb TASKS=math METHODS=safelora
export LORA_ALPHA=16 SAFELORA_THRESHOLD=0.35
export OUT_ROOT=$PWD/outputs/revision_repro_hb457_run2 PUSH_TO_HUB=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0
export STAGES="20"
echo "[repro3] start $(date)"; $PY -c "import torch,transformers,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__)"
mkdir -p "$OUT_ROOT"
bash scripts/revision/run_all.sh
rc=$?
echo "[repro3] run_all rc=$rc  $(date)"
echo "$rc" > logs/revision_repro/train3.rc
