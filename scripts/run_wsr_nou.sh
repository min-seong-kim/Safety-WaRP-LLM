#!/bin/bash
#SBATCH -J wsr_nou
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/wsr_nou_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/wsr_nou_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition gigabyte_a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# WSR-LoRA 에서 rotation(U)만 제거한 ablation.
#   증분:  ΔW = (1-M) ∘ (s·BA)   (원래 weight 공간, element-wise mask, forward-내 freeze)
#   mask:  train.py --phase 2 --original_space_mask → importance=Σ|∂L_safety/∂W| 원소별, per-layer keep_ratio.
#          (WSR per_layer 와 동일 지표, basis_coeff→W 만 다름 → U 유무만 차이나는 공정 ablation)
#   학습:  finetune_gsm8k_lora.py --method wsr_lora_nou (basis 불필요, --mask_dir 만)
# 제출:  sbatch scripts/run_wsr_nou.sh <keep_ratio> [lr]      (기본 kr=0.1, lr 미지정 시 1e-4·2e-4 둘 다)
# HF:    kmseong/llama2_7b-chat-gsm8k-wsr-lora-noU-elem-kr<kr>-r16-lr<lr>

cd /home/gokms0509/Safety-WaRP-LLM
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail
# SLURM 이 --gres 로 GPU 할당 → CUDA_VISIBLE_DEVICES 선언 금지.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch2/gokms0509/hf_cache        # /home 디스크 풀 회피
export HF_HUB_DISABLE_XET=1

# ── config (wsr-lora-elem 열과 100% 동일, rotation 만 제거) ──
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
KR="${1:-0.1}"
if [ -n "${2:-}" ]; then LR_LIST=("$2"); else LR_LIST=(1e-4 2e-4); fi
HF_NS=kmseong; PUSH="${PUSH:-1}"

OUT_ROOT="/scratch2/gokms0509/lora_comparison/wsr_nou/kr_${KR}"
P2ROOT="${OUT_ROOT}/phase2"
mkdir -p "$P2ROOT"
git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

# ── 1) 원래공간 element mask (layer_type 5-way 분할 → 병합; per-layer keep_ratio 라 분할=단일실행 동일) ──
LTS=(ffn_down ffn_up attn_q attn_k attn_v)
COMBINED="${P2ROOT}/masks_combined"
mkdir -p "$COMBINED"

gen_mask () {
  local lt="$1"
  local out="${P2ROOT}/p2_${lt}"
  local md
  md=$(find "$out" -type d -path '*phase2_original_space_*/checkpoints/masks' 2>/dev/null | head -1)
  if [ -n "$md" ] && ls "$md"/"$lt"/layer_*_mask.pt >/dev/null 2>&1; then echo "$md"; return; fi
  mkdir -p "$out"
  python train.py --phase 2 --original_space_mask \
    --phase0_model_dir "$MODEL" --dataset_phase2 circuit_breakers \
    --keep_ratio "$KR" --batch_size 1 \
    --layer_type "$lt" --target_layers all --output_dir "$out" --log_dir ./logs \
    --device cuda --dtype bfloat16 --no_wandb 1>&2
  find "$out" -type d -path '*phase2_original_space_*/checkpoints/masks' 2>/dev/null | head -1
}

echo "===== [1] original-space element masks (kr=${KR}) ====="
for lt in "${LTS[@]}"; do
  md=$(gen_mask "$lt")
  if [ -z "$md" ] || ! ls "$md"/"$lt"/layer_*_mask.pt >/dev/null 2>&1; then
    echo "FATAL: mask gen failed for $lt"; exit 1
  fi
  ln -sfn "$(realpath "$md/$lt")" "$COMBINED/$lt"
  echo "  $lt : $(ls "$md/$lt"/layer_*_mask.pt | wc -l) masks → $COMBINED/$lt"
done
nmask=$(ls "$COMBINED"/*/layer_*_mask.pt 2>/dev/null | wc -l)
echo "  combined masks: $nmask (expect 160)"
[ "$nmask" -ge 160 ] || { echo "FATAL: only $nmask masks"; exit 1; }

# ── 2) 학습: wsr_lora_nou (basis 없음, mask 만) ──
run_one () {
  local lr=$1 repo=$2
  local odir="${OUT_ROOT}/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "skip wsr_nou kr${KR} lr${lr} (done)"; return; fi
  mkdir -p "$odir"
  local push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
  echo "===== [2] train wsr_lora_nou kr=${KR} lr=${lr} → ${repo} ====="
  python finetune_gsm8k_lora.py --method wsr_lora_nou --model_name "$MODEL" \
    --output_dir "$odir" --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" \
    --mask_dir "$COMBINED" --keep_ratio "$KR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$lr" --epochs "$EPOCHS" --batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
    --dtype bfloat16 --gradient_checkpointing --save_merged_model "${push_args[@]}" \
    2>&1 | tee "${odir}/run.log"
}

for lr in "${LR_LIST[@]}"; do
  run_one "$lr" "${HF_NS}/llama2_7b-chat-gsm8k-wsr-lora-noU-elem-kr${KR}-r${LORA_R}-lr${lr}"
done
echo "== wsr_nou kr=${KR} 완료 =="