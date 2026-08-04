#!/bin/bash
# SaLoRA GSM8K sweep — same swap/setup as LISA/LoRA sweep.
#   swap axes : lr {1e-5, 5e-5} x rank {8, 16}
#   alpha     : alpha == r  (SaLoRA 원본 충실 재현, s=1)   [user choice]
#   target    : q,k,v,up,down   (layer_type: attn_q,attn_k,attn_v,ffn_up,ffn_down)
#   matched   : epochs=3, eff.batch=16 (bs4 x ga4), max_len=1024
#   salora    : rank_safe=32, rank_util=32, calib=128 (defaults)
# 4 models -> kmseong/llama2_7b-chat-gsm8k-salora-lr<LR>-r<R>
#
# Waits for the LISA/LoRA sweep (sweep_driver.log 'ALL DONE') before starting,
# so it runs sequentially on the single, compute-saturated GPU.
set -u
cd /root/Safety-WaRP-LLM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=/venv/hb/bin/python
BASE=kmseong/llama2_7b-chat-Safety-FT-lr5e-5
SAFETY=/root/Safety-WaRP-LLM/data/circuit_breakers_train.json
HF_USER=kmseong
SWEEP=/root/Safety-WaRP-LLM/gsm8k_eval/sweep
LOGDIR=/root/Safety-WaRP-LLM/gsm8k_eval/logs
DRIVERLOG="$LOGDIR/sweep_driver.log"
mkdir -p "$SWEEP" "$LOGDIR"

echo "[salora] waiting for LISA/LoRA sweep to finish (ALL DONE in $DRIVERLOG) ..."
until grep -q "ALL DONE" "$DRIVERLOG" 2>/dev/null; do sleep 30; done
echo "[salora] LISA/LoRA sweep done -> starting SaLoRA sweep"

upload() {  # $1 local_dir  $2 repo_id
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi, create_repo
d, repo = sys.argv[1], sys.argv[2]
tok = os.environ["HF_TOKEN"]
create_repo(repo, token=tok, private=False, exist_ok=True, repo_type="model")
HfApi().upload_folder(repo_id=repo, folder_path=d, token=tok, repo_type="model",
                      commit_message="GSM8K SaLoRA sweep")
print("UPLOADED", repo)
PYEOF
}

run_salora() {  # $1 LR  $2 R  $3 A
  local LR=$1 R=$2 A=$3
  local OUT="$SWEEP/salora_lr${LR}_r${R}"
  local REPO="$HF_USER/llama2_7b-chat-gsm8k-salora-lr${LR}-r${R}"
  local LOG="$LOGDIR/sweep_salora_lr${LR}_r${R}.log"
  if [ ! -f "$OUT/merged_model/config.json" ]; then
    echo "==================== TRAIN SaLoRA lr=$LR r=$R a=$A ===================="
    "$PY" finetune_gsm8k_salora.py \
      --model_name "$BASE" --output_dir "$OUT" \
      --safety_data_path "$SAFETY" \
      --learning_rate "$LR" --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --max_length 1024 \
      --lora_r "$R" --lora_alpha "$A" --lora_dropout 0.0 \
      --target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
      --layer_type attn_q,attn_k,attn_v,ffn_up,ffn_down --target_layers all \
      2>&1 | tee "$LOG"
  else
    echo "SKIP train (exists): $OUT"
  fi
  if [ -f "$OUT/merged_model/config.json" ]; then
    echo "==================== UPLOAD SaLoRA -> $REPO ===================="
    upload "$OUT/merged_model" "$REPO" >>"$LOG" 2>&1 && echo "OK upload $REPO" || echo "FAIL upload $REPO"
  else
    echo "FAIL train SaLoRA lr=$LR r=$R (no merged_model)"
  fi
}

for LR in 1e-5 5e-5; do for R in 8 16; do A=$R; run_salora "$LR" "$R" "$A"; done; done
echo "==================== SALORA ALL DONE ===================="
