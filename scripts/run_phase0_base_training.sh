#!/bin/bash

# Phase 0: Base Safety Training
# 원본 FSCIL-WaRP의 base_train()과 동일

echo "========================================="
echo "Phase 0: Base Safety Training"
echo "========================================="

python train_fixed.py \
    --phase 0 \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    --circuit_breakers_samples 4994 \
    --base_epochs 3 \
    --base_lr 2e-5 \
    --base_weight_decay 0.01 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42 \
    --use_sft

echo ""
echo "========================================="
echo "Phase 0 완료!"
echo "다음 단계: scripts/run_phase1_basis.sh 실행"
echo "========================================="
