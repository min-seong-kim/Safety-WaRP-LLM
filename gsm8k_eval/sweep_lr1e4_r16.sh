#!/bin/bash
# gsm8k LoRA-family re-sweep with PROPER LoRA learning rates.
#   fixed rank : 16   (alpha: LoRA/LISA=32 [α=2r], SaLoRA=16 [α=r])
#   lr sweep   : 1e-4, 2e-4, 3e-4
#   methods    : lora, salora, lisa   → 3 x 3 = 9 models
#   target     : q,k,v,up,down ; epochs 3 ; eff.batch 16 (bs4 x ga4) ; max_len 1024
#   base       : kmseong/llama2_7b-chat-Safety-FT-lr5e-5
# 이전 스윕(lr{1e-5,5e-5})이 LoRA엔 lr가 너무 낮아 underfit → 정상 LoRA lr로 재실행.
# upload → kmseong/llama2_7b-chat-gsm8k-<method>-lr<LR>-r16
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
R=16

upload() {  # $1 local_dir  $2 repo_id
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi, create_repo
d, repo = sys.argv[1], sys.argv[2]
tok = os.environ["HF_TOKEN"]
create_repo(repo, token=tok, private=False, exist_ok=True, repo_type="model")
HfApi().upload_folder(repo_id=repo, folder_path=d, token=tok, repo_type="model",
                      commit_message="gsm8k LoRA-family re-sweep (r16, proper LoRA lr)")
print("UPLOADED", repo)
PYEOF
}

run_lora() {  # $1 LR
  local LR=$1 A=32
  local OUT="$SWEEP/lora_lr${LR}_r${R}"
  local REPO="$HF_USER/llama2_7b-chat-gsm8k-lora-lr${LR}-r${R}"
  local LOG="$LOGDIR/sweep2_lora_lr${LR}_r${R}.log"
  if [ ! -f "$OUT/merged_model/config.json" ]; then
    echo "==================== TRAIN LoRA lr=$LR r=$R a=$A ===================="
    "$PY" finetune_gsm8k_lora.py \
      --method lora --model_name "$BASE" --output_dir "$OUT" --safety_data_path "$SAFETY" \
      --learning_rate "$LR" --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --max_length 1024 \
      --lora_r "$R" --lora_alpha "$A" --lora_dropout 0.05 \
      --target_modules q_proj,k_proj,v_proj,up_proj,down_proj 2>&1 | tee "$LOG"
  else echo "SKIP train (exists): $OUT"; fi
  if [ -f "$OUT/merged_model/config.json" ]; then
    echo "==================== UPLOAD LoRA -> $REPO ===================="
    upload "$OUT/merged_model" "$REPO" >>"$LOG" 2>&1 && echo "OK upload $REPO" || echo "FAIL upload $REPO"
  else echo "FAIL train LoRA lr=$LR (no merged_model)"; fi
}

run_salora() {  # $1 LR
  local LR=$1 A=32   # α=2r=32 (scaling 2, LoRA와 맞춤: 공정 비교)
  local OUT="$SWEEP/salora_lr${LR}_r${R}"
  local REPO="$HF_USER/llama2_7b-chat-gsm8k-salora-lr${LR}-r${R}"
  local LOG="$LOGDIR/sweep2_salora_lr${LR}_r${R}.log"
  if [ ! -f "$OUT/merged_model/config.json" ]; then
    echo "==================== TRAIN SaLoRA lr=$LR r=$R a=$A ===================="
    "$PY" finetune_gsm8k_salora.py \
      --model_name "$BASE" --output_dir "$OUT" --safety_data_path "$SAFETY" \
      --learning_rate "$LR" --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --max_length 1024 \
      --lora_r "$R" --lora_alpha "$A" --lora_dropout 0.0 \
      --target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
      --layer_type attn_q,attn_k,attn_v,ffn_up,ffn_down --target_layers all 2>&1 | tee "$LOG"
  else echo "SKIP train (exists): $OUT"; fi
  if [ -f "$OUT/merged_model/config.json" ]; then
    echo "==================== UPLOAD SaLoRA -> $REPO ===================="
    upload "$OUT/merged_model" "$REPO" >>"$LOG" 2>&1 && echo "OK upload $REPO" || echo "FAIL upload $REPO"
  else echo "FAIL train SaLoRA lr=$LR (no merged_model)"; fi
}

run_lisa() {  # $1 LR
  local LR=$1 A=32
  local OUT="$SWEEP/lisa_lr${LR}_r${R}"
  local REPO="$HF_USER/llama2_7b-chat-gsm8k-lisa-lr${LR}-r${R}"
  local LOG="$LOGDIR/sweep2_lisa_lr${LR}_r${R}.log"
  if [ ! -f "$OUT/config.json" ]; then
    echo "==================== TRAIN LISA lr=$LR r=$R a=$A ===================="
    ( cd gsm8k_eval && "$PY" finetune_gsm8k_lisa.py \
        --model_path "$BASE" --output_dir "$OUT" --safety_data_path "$SAFETY" \
        --learning_rate "$LR" --epochs 3 --batch_size 4 --grad_accum 4 --max_length 1024 \
        --lora --lora_r "$R" --lora_alpha "$A" --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj up_proj down_proj \
        --rho 1.0 --alignment_step 100 --finetune_step 900 --guide_data_num 4994 \
        --report_to none ) 2>&1 | tee "$LOG"
  else echo "SKIP train (exists): $OUT"; fi
  if [ -f "$OUT/config.json" ]; then
    echo "==================== UPLOAD LISA -> $REPO ===================="
    upload "$OUT" "$REPO" >>"$LOG" 2>&1 && echo "OK upload $REPO" || echo "FAIL upload $REPO"
  else echo "FAIL train LISA lr=$LR (no config.json)"; fi
}

# LoRA(α=32) 3개는 완료·업로드됨 → relaunch 시 재업로드 방지 위해 스킵.
# (필요 시 아래 주석 해제; resume 가드가 학습은 스킵하되 업로드는 재수행함)
# for LR in 1e-4 2e-4 3e-4; do run_lora   "$LR"; done
for LR in 1e-4 2e-4 3e-4; do run_salora "$LR"; done
for LR in 1e-4 2e-4 3e-4; do run_lisa   "$LR"; done
echo "==================== SWEEP2 ALL DONE ===================="
