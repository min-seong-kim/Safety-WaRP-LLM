#!/bin/bash
#SBATCH -J safelora_preproj
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/safelora_preproj_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/safelora_preproj_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition suma_rtx4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# 통제 실험: SafeLoRA 의 "투영 전 표준 LoRA".
#   SafeLoRA(--method safe_lora)는 표준 LoRA 를 학습한 뒤 사후 투영만 추가한다. 즉 투영 전 base 는
#   동일 config 의 --method lora 와 (같은 seed·env 이므로) 동일. 이 박스에서 safelora 열과 완전히 같은
#   설정으로 표준 LoRA 를 학습 → 두 가지 통제를 가능케 함:
#     (a) safelora(thr0.35) vs preproj : 순수 "사후 투영" 효과 (환경 동일)
#     (b) preproj vs 옛 lora 열         : 순수 "학습 환경/RNG" 효과 (방법 동일)
# 제출: sbatch scripts/run_safelora_preproj.sh   (lr 1e-4, 2e-4 둘 다)

cd /home/gokms0509/Safety-WaRP-LLM
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── config: run_safelora_comparison.sh 와 100% 동일(단 method=lora, 투영 없음) ──
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4 2e-4)
HF_NS=kmseong; PUSH="${PUSH:-1}"

OUT_ROOT="/scratch2/gokms0509/lora_comparison/safelora_preproj"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

run_one () {
  local lr=$1 repo=$2
  local odir="${OUT_ROOT}/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "skip preproj lr${lr} (done)"; return; fi
  mkdir -p "$odir"
  local push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
  echo "[preproj] lr=${lr} → ${repo}"
  python finetune_gsm8k_lora.py --method lora --model_name "$MODEL" \
    --output_dir "$odir" --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$lr" --epochs "$EPOCHS" --batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
    --dtype bfloat16 --gradient_checkpointing --save_merged_model "${push_args[@]}" \
    2>&1 | tee "${odir}/run.log"
}

for lr in "${LR_LIST[@]}"; do
  run_one "$lr" "${HF_NS}/llama2_7b-chat-gsm8k-safelora-preproj-r${LORA_R}-lr${lr}"
done
echo "== safelora_preproj 완료 =="
