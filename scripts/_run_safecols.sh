#!/bin/bash
cd /home/users/minseong/Safety-WaRP-LLM
HBPY=/home/users/minseong/.conda/envs/hb/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
OUT=outputs/lora_comparison/artifacts/orig_safecols
run() {
  $HBPY models/build_lora_safety_artifacts.py \
    --model_name kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --safety_data_path ./data/circuit_breakers_train.json \
    --out_dir "$OUT" --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all --direction_keep_ratio 0.1 --batch_size "$1" --max_length 1024
}
run 2 || { echo "=== batch2 실패, batch1 재시도 ==="; rm -rf "$OUT"; run 1; }
echo "=== safecols done. files: $(find $OUT -name '*_safecols.pt' | wc -l) ==="
