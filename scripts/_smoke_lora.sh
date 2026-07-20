#!/bin/bash
# 3 method 통합 runner 무결성 스모크 (tiny data, no push). 각 방법이 학습→dense저장→sanity gen까지 통과하는지.
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
common="--model_name $MODEL --layer_type $LT --target_modules $TM --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --learning_rate 1e-4 --epochs 1 --batch_size 2 --gradient_accumulation_steps 2 --gsm8k_samples 32 --max_length 512 --dtype bfloat16 --gradient_checkpointing --save_merged_model"
set -x
$HBPY finetune_gsm8k_lora.py --method lora $common --output_dir outputs/smoke/lora || { echo SMOKE_FAIL_lora; exit 11; }
$HBPY finetune_gsm8k_lora.py --method original_projected_lora --safecols_dir $SAFECOLS $common --output_dir outputs/smoke/orig || { echo SMOKE_FAIL_orig; exit 12; }
$HBPY finetune_gsm8k_lora.py --method wsr_lora --basis_dir $BASIS --mask_dir $MASKS $common --output_dir outputs/smoke/wsr || { echo SMOKE_FAIL_wsr; exit 13; }
echo "SMOKE_ALL_OK"
