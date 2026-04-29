# Base Model
export CUDA_VISIBLE_DEVICES=4

echo "===== Find Foundation Neurons ======"
python foundation_neuron_detection.py 1000 \
    --model_name Qwen/Qwen2.5-7B \
    --ffn_active_fraction 0.01 \
    --attn_active_fraction 0.01

echo "===== Find Foundation Neurons ======"


echo "===== Find Foundation Neurons ======"


# Instruct Model
echo "===== Find Foundation Neurons ======"
python foundation_neuron_detection.py 1000 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --ffn_active_fraction 0.01 \
    --attn_active_fraction 0.01

echo "===== Find Foundation Neurons ======"


echo "===== Find Foundation Neurons ======"