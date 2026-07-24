#!/bin/bash
#SBATCH -J salora_gsm8k
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/salora_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/salora_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition suma_rtx4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# GPU 24GB로 충분: SaLoRA는 q/v만 대상 → gram ~4.3GB + model 14GB.
#
# SaLoRA (ICLR'25) baseline — 비교표의 5번째 열.
# 제출:  sbatch --dependency=afterany:<safelora_jobid> scripts/run_salora_comparison.sh
#
# 학습 중 LoRA 증분을 고정 투영 C = I − V_s V_sᵀ (safety 출력 부분공간 밖)로 통과시켜
# downstream 학습이 안전 정렬을 건드리지 못하게 한다. 시작=safety 모델(PiSSA residual).
# 결과 dense 모델을 /scratch2 저장 + HF push → eval_gsm8k.sh / HarmBench 로 표 산출.

cd /home/gokms0509/Safety-WaRP-LLM

source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb

# set -e 없음: 한 run(push 실패/OOM)이 나머지를 막지 않도록.
set -uo pipefail

# SLURM 이 --gres 로 GPU 를 할당하므로 CUDA_VISIBLE_DEVICES 를 선언하지 않는다.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── config (원본 SaLoRA 충실 재현: alpha==r → s=1, target q_proj,v_proj) ──
# 다른 비교열과 맞추는 부분: EPOCHS=3, LR=(1e-4,2e-4), batch/accum, seed, 시작 모델.
# ⚠️ wsr_lora/safelora 열은 α=32·5모듈이지만, SaLoRA는 정의상 α=r·q/v. 예산 일치가 필요하면
#    LORA_ALPHA=32, LAYER_TYPE/TARGET_MODULES 를 5종으로 바꿔라(단 원본 SaLoRA와는 달라짐).
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
SAFETY_DATA="./data/circuit_breakers_train.json"
LAYER_TYPE="attn_q,attn_v"
TARGET_MODULES="q_proj,v_proj"
LORA_R=16; LORA_ALPHA=16; LORA_DROPOUT=0.0
RANK_SAFE=32; RANK_UTIL=32; CALIB_SAMPLES=128; CALIB_BS=4; NITER=20
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4 2e-4)
HF_NS=kmseong; PUSH="${PUSH:-1}"

OUT_ROOT="/scratch2/gokms0509/lora_comparison"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "${OUT_ROOT}/git_commit_salora.txt" 2>/dev/null || true

run_one () {
  local lr=$1 repo=$2
  local odir="${OUT_ROOT}/salora/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "skip salora lr${lr} (done)"; return; fi
  mkdir -p "$odir"
  local push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
  echo "[salora] lr=${lr} → ${repo}"
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
  run_one "$lr" "${HF_NS}/llama2_7b-chat-gsm8k-salora-r${LORA_R}-lr${lr}"
done

echo "== salora 학습/업로드 완료. eval 은 scripts/eval_gsm8k.sh 의 MODELS 에 아래 2개 추가 =="
for lr in "${LR_LIST[@]}"; do
  echo "  ${HF_NS}/llama2_7b-chat-gsm8k-salora-r${LORA_R}-lr${lr}"
done
