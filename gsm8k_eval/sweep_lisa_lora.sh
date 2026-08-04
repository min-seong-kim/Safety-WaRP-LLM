#!/bin/bash
# LISA vs vanilla-LoRA GSM8K sweep.
#   swap axes : lr {1e-5, 5e-5} x rank {8(a16), 16(a32)}
#   target    : q,k,v,up,down
#   matched   : epochs=3, eff.batch=16 (bs4 x ga4), max_len=1024
# 8 models total -> uploaded to kmseong/llama2_7b-chat-gsm8k-<method>-lr<LR>-r<R>
set -u
cd /root/Safety-WaRP-LLM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=/venv/hb/bin/python
BASE=kmseong/llama2_7b-chat-Safety-FT-lr5e-5
SAFETY=/root/Safety-WaRP-LLM/data/circuit_breakers_train.json
HF_USER=kmseong
SWEEP=/root/Safety-WaRP-LLM/gsm8k_eval/sweep
LOGDIR=/root/Safety-WaRP-LLM/gsm8k_eval/logs
mkdir -p "$SWEEP" "$LOGDIR"

upload() {  # $1 local_dir  $2 repo_id  -> exit code reflects success
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi, create_repo
d, repo = sys.argv[1], sys.argv[2]
tok = os.environ["HF_TOKEN"]
create_repo(repo, token=tok, private=False, exist_ok=True, repo_type="model")
HfApi().upload_folder(repo_id=repo, folder_path=d, token=tok, repo_type="model",
                      commit_message="GSM8K LISA/LoRA sweep")
print("UPLOADED", repo)
PYEOF
}

run_lora() {  # $1 LR  $2 R  $3 A
  local LR=$1 R=$2 A=$3
  local OUT="$SWEEP/lora_lr${LR}_r${R}"
  local REPO="$HF_USER/llama2_7b-chat-gsm8k-lora-lr${LR}-r${R}"
  local LOG="$LOGDIR/sweep_lora_lr${LR}_r${R}.log"
  if [ ! -f "$OUT/merged_model/config.json" ]; then
    echo "==================== TRAIN LoRA lr=$LR r=$R a=$A ===================="
    "$PY" finetune_gsm8k_lora.py \
      --method lora --model_name "$BASE" --output_dir "$OUT" \
      --safety_data_path "$SAFETY" \
      --learning_rate "$LR" --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --max_length 1024 \
      --lora_r "$R" --lora_alpha "$A" --lora_dropout 0.05 \
      --target_modules q_proj,k_proj,v_proj,up_proj,down_proj 2>&1 | tee "$LOG"
  else
    echo "SKIP train (exists): $OUT"
  fi
  if [ -f "$OUT/merged_model/config.json" ]; then
    echo "==================== UPLOAD LoRA -> $REPO ===================="
    upload "$OUT/merged_model" "$REPO" >>"$LOG" 2>&1 && echo "OK upload $REPO" || echo "FAIL upload $REPO"
  else
    echo "FAIL train LoRA lr=$LR r=$R (no merged_model)"
  fi
}

run_lisa() {  # $1 LR  $2 R  $3 A
  local LR=$1 R=$2 A=$3
  local OUT="$SWEEP/lisa_lr${LR}_r${R}"
  local REPO="$HF_USER/llama2_7b-chat-gsm8k-lisa-lr${LR}-r${R}"
  local LOG="$LOGDIR/sweep_lisa_lr${LR}_r${R}.log"
  if [ ! -f "$OUT/config.json" ]; then
    echo "==================== TRAIN LISA lr=$LR r=$R a=$A ===================="
    ( cd gsm8k_eval && "$PY" finetune_gsm8k_lisa.py \
        --model_path "$BASE" --output_dir "$OUT" \
        --safety_data_path "$SAFETY" \
        --learning_rate "$LR" --epochs 3 --batch_size 4 --grad_accum 4 --max_length 1024 \
        --lora --lora_r "$R" --lora_alpha "$A" --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj up_proj down_proj \
        --rho 1.0 --alignment_step 100 --finetune_step 900 --guide_data_num 4994 \
        --report_to none ) 2>&1 | tee "$LOG"
  else
    echo "SKIP train (exists): $OUT"
  fi
  if [ -f "$OUT/config.json" ]; then
    echo "==================== UPLOAD LISA -> $REPO ===================="
    upload "$OUT" "$REPO" >>"$LOG" 2>&1 && echo "OK upload $REPO" || echo "FAIL upload $REPO"
  else
    echo "FAIL train LISA lr=$LR r=$R (no config.json)"
  fi
}

# ---- vanilla LoRA first (unvalidated script -> fail fast), then LISA ----
for LR in 1e-5 5e-5; do for R in 8 16; do A=$((R*2)); run_lora "$LR" "$R" "$A"; done; done
for LR in 1e-5 5e-5; do for R in 8 16; do A=$((R*2)); run_lisa "$LR" "$R" "$A"; done; done
echo "==================== ALL DONE ===================="
