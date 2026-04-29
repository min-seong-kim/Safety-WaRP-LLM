echo "Running Phase 0: SSFT_INSTRUCT"
python models/phase0_SSFT.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_json "/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/jongbokwon/Safety-WaRP-LLM/data/circuit_breakers_train.json" \
    --checkpoints_dir "/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/jongbokwon/minseong_results/checkpoints/qwen-2.5-3b/instruct" \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_length 1024 \
    --max_samples 4994 \
    --wandb_run_name "qwen-2.5-3b-instruct-ssft"
echo "========================================================================"

echo "Running Phase 0: SSFT_BASE"
python models/phase0_SSFT.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --dataset_json "/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/jongbokwon/Safety-WaRP-LLM/data/circuit_breakers_train.json" \
    --checkpoints_dir "/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/jongbokwon/minseong_results/checkpoints/qwen-2.5-7b/base_newlr" \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_length 1024 \
    --max_samples 4994 \
    --wandb_run_name "qwen-2.5-7b-base-ssft-newlr"