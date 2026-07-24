#!/bin/bash
#SBATCH -J salora_matched
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/salora_matched_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/salora_matched_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition gigabyte_a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# SaLoRA 예산-맞춤 재실행: wsr/safelora 열과 동일 budget(α=32, 5모듈 q,k,v,down,up)으로.
#   ⚠️ 원본 SaLoRA(α=r=16, q/v)와는 다름 → 논문 표에는 "budget-matched, 원저자 권장설정과 다름" 각주.
#   5모듈이면 down_proj input-gram(11008²·fp32≈0.48GB)×레이어 → 48GB A6000 사용(24GB는 OOM 위험).
#   OOM 나면 --partition suma_a100 (80GB)로 바꿔 재제출.
# 제출: sbatch scripts/run_salora_matched.sh

cd /home/gokms0509/Safety-WaRP-LLM
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── config: 예산-맞춤 (α=32, 5모듈). SaLoRA 부분공간 랭크는 원본과 동일(rs=du=32) ──
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
SAFETY_DATA="./data/circuit_breakers_train.json"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05    # ← wsr/safelora 열과 동일
RANK_SAFE=32; RANK_UTIL=32; CALIB_SAMPLES=128; CALIB_BS=2; NITER=20
# gram OOM 근본해결: salora.py init 을 layer_type 그룹별(down_proj 먼저) 처리로 리팩터 → 한 번에
# 한 그룹 Gram(최대 down 15.5GB)만 상주 → peak ~30GB (48GB a6000 여유). CALIB_BS=2 로 복원(활성화 무관).
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4 2e-4)
HF_NS=kmseong; PUSH="${PUSH:-1}"

OUT_ROOT="/scratch2/gokms0509/lora_comparison/salora_matched"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

run_one () {
  local lr=$1 repo=$2
  local odir="${OUT_ROOT}/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "skip salora_matched lr${lr} (done)"; return; fi
  mkdir -p "$odir"
  local push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
  echo "[salora_matched] lr=${lr} → ${repo}"
  python finetune_gsm8k_salora.py --model_name "$MODEL" \
    --output_dir "$odir" --safety_data_path "$SAFETY_DATA" \
    --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" --target_layers all \
    --salora_rank_safe "$RANK_SAFE" --salora_rank_util "$RANK_UTIL" \
    --salora_calib_samples "$CALIB_SAMPLES" --salora_calib_batch_size "$CALIB_BS" --salora_niter "$NITER" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$lr" --epochs "$EPOCHS" --batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
    --dtype bfloat16 --gradient_checkpointing "${push_args[@]}" \
    2>&1 | tee "${odir}/run.log"
}

for lr in "${LR_LIST[@]}"; do
  run_one "$lr" "${HF_NS}/llama2_7b-chat-gsm8k-salora-matched-a32-r${LORA_R}-lr${lr}"
done
echo "== salora_matched 완료 =="
