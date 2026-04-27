echo "Running Phase 0: SSFT"
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
echo "========================================================================"

echo "Running Phase 1: Basis"
python train.py \
    --phase 1 \
echo "========================================================================"

echo "Running Phase 2: Importance Scoring"
python train.py \
    --phase 2 \
echo "========================================================================"

echo "Running Phase 3: Importance"
python train.py \
    --phase 3
echo "========================================================================"