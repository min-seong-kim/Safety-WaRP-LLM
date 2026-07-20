#!/bin/bash
# 6 run (3 method × 2 LR) 학습 → dense 저장 → HF push → push 성공 시 로컬 merged 삭제(디스크 절약).
# 각 run 은 summary.json 있으면 skip(resume). 한 run 실패해도 다음 run 진행.
cd /home/users/minseong/Safety-WaRP-LLM
HBPY=/home/users/minseong/.conda/envs/hb/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LT="attn_q,attn_k,attn_v,ffn_down,ffn_up"; TM="q_proj,k_proj,v_proj,down_proj,up_proj"
BASIS="checkpoints/phase1_20260713_144203/basis"
MASKS="checkpoints/phase2_20260719_145729/checkpoints/masks"
SAFECOLS="outputs/lora_comparison/artifacts/orig_safecols"
OUT="outputs/lora_comparison"; NS="kmseong"
LR_LIST=(1e-4 2e-4)

run_one() {
  local method=$1 lr=$2 repo=$3
  local odir="${OUT}/${method}/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "[skip] ${method} lr${lr} done"; return 0; fi
  mkdir -p "$odir"
  local extra=()
  [ "$method" = "wsr_lora" ] && extra=(--basis_dir "$BASIS" --mask_dir "$MASKS")
  [ "$method" = "original_projected_lora" ] && extra=(--safecols_dir "$SAFECOLS")
  echo "=========== ${method} lr=${lr} → ${repo} ==========="
  $HBPY finetune_gsm8k_lora.py --method "$method" --model_name "$MODEL" \
    --output_dir "$odir" --layer_type "$LT" --target_modules "$TM" \
    --keep_ratio 0.1 --direction_keep_ratio 0.1 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --learning_rate "$lr" --epochs 3 --batch_size 2 --gradient_accumulation_steps 8 \
    --gsm8k_samples 1000 --max_length 1024 --seed 42 --dtype bfloat16 --gradient_checkpointing \
    --save_merged_model --push_to_hub --hf_repo_id "$repo" \
    "${extra[@]}" 2>&1 | tee "${odir}/run.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = "0" ] && [ -f "${odir}/summary.json" ]; then
    # push 성공 확인 후 로컬 merged 삭제 (HF 가 source of truth)
    $HBPY -c "from huggingface_hub import HfApi; HfApi().model_info('${repo}')" 2>/dev/null \
      && { echo "[cleanup] rm ${odir}/merged_model (HF ok)"; rm -rf "${odir}/merged_model" "${odir}/trainer"; } \
      || echo "[warn] HF repo ${repo} 확인 실패 — 로컬 유지"
  else
    echo "[FAIL] ${method} lr${lr} rc=${rc}"
  fi
}

for lr in "${LR_LIST[@]}"; do
  run_one lora "$lr"                    "${NS}/llama2_7b-chat-gsm8k-lora-r16-lr${lr}"
  run_one original_projected_lora "$lr" "${NS}/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr${lr}"
  run_one wsr_lora "$lr"                "${NS}/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr${lr}"
done
echo "===== ALL TRAIN RUNS DONE ====="
