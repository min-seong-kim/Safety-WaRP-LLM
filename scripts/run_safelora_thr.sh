#!/bin/bash
#SBATCH -J safelora_thr
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/safelora_thr_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/safelora_thr_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition suma_rtx4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# SafeLoRA threshold 스윕 (트레이드오프 곡선용). 학습은 표준 LoRA(24GB OK), 사후투영만 threshold 다름.
#   고정: lr=2e-4 (가장 강한 fine-tuning 압력 지점에서 곡선을 그림). thr0.35는 기존 앵커점.
# 제출: sbatch scripts/run_safelora_thr.sh <THRESHOLD>   (예: 0.1 0.2 0.3 0.4 0.5)

THR="${1:?usage: sbatch run_safelora_thr.sh <threshold>}"
LR="${2:-2e-4}"

cd /home/gokms0509/Safety-WaRP-LLM
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
SAFELORA_BASE="meta-llama/Llama-2-7b-chat-hf"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
HF_NS=kmseong; PUSH="${PUSH:-1}"

OUT_ROOT="/scratch2/gokms0509/lora_comparison/safelora_sweep"
odir="${OUT_ROOT}/thr${THR}_lr${LR}"
mkdir -p "$odir"
if [ -f "${odir}/summary.json" ]; then echo "skip safelora thr${THR} lr${LR} (done)"; exit 0; fi
git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

repo="${HF_NS}/llama2_7b-chat-gsm8k-safelora-thr${THR}-r${LORA_R}-lr${LR}"
push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
echo "[safelora_sweep] thr=${THR} lr=${LR} → ${repo}"
python finetune_gsm8k_lora.py --method safe_lora --model_name "$MODEL" \
  --output_dir "$odir" --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" \
  --safelora_base_model "$SAFELORA_BASE" --safelora_aligned_model "$MODEL" \
  --safelora_select_type threshold --safelora_threshold "$THR" --safelora_load_dtype float32 \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --learning_rate "$LR" --epochs "$EPOCHS" --batch_size "$BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
  --dtype bfloat16 --gradient_checkpointing --save_merged_model "${push_args[@]}" \
  2>&1 | tee "${odir}/run.log"
echo "== safelora thr${THR} lr${LR} 완료 =="
