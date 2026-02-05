#!/bin/bash

# Phase 0: Base Safety Training with LoRA
# LoRA를 사용한 효율적인 safety fine-tuning

echo "========================================="
echo "Phase 0: Base Safety Training (LoRA)"
echo "========================================="
echo "LoRA: Low-Rank Adaptation"
echo "  - 학습 파라미터: ~0.5% of full model"
echo "  - 메모리 효율적"
echo "  - 빠른 학습"
echo "========================================="

python train_lora.py \
    --phase 0 \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    --circuit_breakers_samples 4994 \
    --base_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_lr 2e-5 \
    --lora_weight_decay 0.01 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42

echo ""
echo "========================================="
echo "Phase 0 (LoRA) 완료!"
echo "========================================="
echo "출력:"
echo "  - LoRA adapters: checkpoints/phase0_lora_*/final_lora_model/"
echo "  - Merged model: checkpoints/phase0_lora_*/final_merged_model/"
echo ""
echo "다음 단계:"
echo "  1. Phase 1에서 'final_merged_model' 사용"
echo "  2. scripts/run_phase1_basis.sh 실행"
echo "     --phase0_model_dir ./checkpoints/phase0_lora_*/final_merged_model"
echo "========================================="
