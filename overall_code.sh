#!/bin/bash

source /home/korea_bupj/miniconda3/etc/profile.d/conda.sh
conda activate hb

PHASE0_MODEL="wvnvwn/qwen-2.5-7b-ssft-lr5e-5"
SSFT_DATASET="./data/circuit_breakers_train.json"


echo "#### SSFT Phase ####"
python models/phase0_SSFT.py \
    --model_name $PHASE0_MODEL \
    --dataset_json $SSFT_DATASET \
    --checkpoints_dir "/home/users/jongbokwon/minseong_results/checkpoints/qwen-2.5-7b/base_lr${lr}" \
    --learning_rate $lr \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_length 1024 \
    --max_samples 4994 \
    --wandb_run_name "qwen-2.5-7b-base-ssft-lr${lr}"