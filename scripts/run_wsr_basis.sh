#!/bin/bash
#SBATCH -J wsr_basis
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/wsr_basis_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/wsr_basis_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition gigabyte_a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# WSR keep_ratio 스윕용 Phase1 basis 생성 (이 박스엔 checkpoints/ 가 없어 재생성 필요).
#   결과를 고정 심볼릭 checkpoints/phase1_sweep_basis 로 연결 → kr 학습 잡들이 참조.
#   basis 는 keep_ratio 와 무관(모델·safety데이터에만 의존)하므로 1회만 생성.
# 제출: sbatch scripts/run_wsr_basis.sh

cd /home/gokms0509/Safety-WaRP-LLM
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
LINK="checkpoints/phase1_sweep_basis"

if [ -d "$LINK" ] && ls "$LINK"/*/layer_*_svd.pt >/dev/null 2>&1; then
  echo "skip: basis already present at $LINK"; exit 0
fi

mkdir -p checkpoints logs
python train.py --phase 1 \
  --phase0_model_dir "$MODEL" --safety_dataset circuit_breakers \
  --batch_size 4 --layer_type "$LAYER_TYPE" --target_layers all \
  --output_dir ./checkpoints/wsr_basis --log_dir ./logs \
  --device cuda --dtype bfloat16 --no_wandb

newest=$(find checkpoints/wsr_basis -type d -path '*phase1_*/basis' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
if [ -z "$newest" ]; then echo "ERROR: basis dir not produced"; exit 1; fi
ln -sfn "$(realpath "$newest")" "$LINK"
echo "== basis ready: $LINK -> $(realpath "$newest") =="
ls "$LINK" | head
