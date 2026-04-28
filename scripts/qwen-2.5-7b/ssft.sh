echo "Running Phase 0: SSFT_INSTRUCT"
python models/phase0_SSFT.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_json "./data/circuit_breakers_train.json" \
    --checkpoints_dir "/home/users/jongbokwon/minseong_results/checkpoints/qwen-2.5-7b/instruct" \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_length 1024 \
    --max_samples 4994 \
    --wandb_run_name "qwen-2.5-7b-instruct-ssft"
echo "========================================================================"

echo "Running Phase 0: SSFT_BASE"
python models/phase0_SSFT.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --dataset_json "./data/circuit_breakers_train.json" \
    --checkpoints_dir "/home/users/jongbokwon/minseong_results/checkpoints/qwen-2.5-7b/base" \
    --learning_rate 3e-5 \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_length 1024 \
    --max_samples 4994 \
    --wandb_run_name "qwen-2.5-7b-base-ssft"