#!/bin/bash
#SBATCH -J safelora_gsm8k
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/safelora_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/safelora_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition suma_rtx4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# GPU 24GB(RTX4090/A5000/RTX3090)로 충분: 학습 peak ~18GB, projection peak ~15GB(LoRA-only 스냅샷).
# 대안(유휴 많음): --partition gigabyte_a5000 / dell_rtx3090 / base_suma_rtx3090.
# 48GB+ 필요 없음. A6000/A100 파티션은 현재 만실이라 pending 유발.
#
# 바닐라 Safe LoRA (NeurIPS'24) baseline — wsr_lora 비교표의 4번째 열.
# 제출:  sbatch scripts/run_safelora_comparison.sh
#
# 학습 설정은 --method lora 와 완전히 동일(같은 시작 checkpoint·GSM8K·seed·r16·α32·ep3·LR grid).
# 차이는 학습 후 lora_B 를 alignment subspace 로 사후 투영하는 것뿐(재학습 아님).
#   V = W_aligned − W_base,  C = VVᵀ/‖V‖,  cos≤thr 레이어만 B←C·B.
# 결과 dense 모델을 /scratch2 에 저장 + HF push → eval_gsm8k.sh / HarmBench 로 표 산출.

cd /home/gokms0509/Safety-WaRP-LLM

# conda (torch/transformers/peft 있는 env). set -u 전에 활성화.
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb

# set -e 는 쓰지 않는다: 한 run(예: push 실패/OOM)이 나머지 run 을 막지 않도록.
set -uo pipefail

# SLURM 이 --gres 로 GPU 를 할당하므로 CUDA_VISIBLE_DEVICES 를 선언하지 않는다.
# (finetune_gsm8k_lora.py 가 gsm8k_eval import 의 하드코딩을 SLURM 할당값으로 복원함.)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# HF 인증: ~/.cache/huggingface/token (huggingface-cli login) 사용. 필요시 export HF_TOKEN=...

# ── config (wsr_lora 와 동일 budget) ──
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"          # aligned = 세 방법 공통 시작점
SAFELORA_BASE="meta-llama/Llama-2-7b-chat-hf"            # base (Safety-FT 이전) → V = Safety-FT 방향
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4 2e-4)
SAFELORA_SELECT="threshold"    # threshold | number
SAFELORA_THRESHOLD=0.35        # 코사인 ≤ 0.35 인 LoRA 레이어를 투영
SAFELORA_NUM_LAYERS=10         # (number 모드에서만 사용)
HF_NS=kmseong; PUSH="${PUSH:-1}"

OUT_ROOT="/scratch2/gokms0509/lora_comparison"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "${OUT_ROOT}/git_commit_safelora.txt" 2>/dev/null || true

run_one () {
  local lr=$1 repo=$2
  local odir="${OUT_ROOT}/safe_lora/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "skip safe_lora lr${lr} (done)"; return; fi
  mkdir -p "$odir"
  local push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
  echo "[safe_lora] lr=${lr} → ${repo}"
  python finetune_gsm8k_lora.py --method safe_lora --model_name "$MODEL" \
    --output_dir "$odir" --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" \
    --safelora_base_model "$SAFELORA_BASE" --safelora_aligned_model "$MODEL" \
    --safelora_select_type "$SAFELORA_SELECT" --safelora_threshold "$SAFELORA_THRESHOLD" \
    --safelora_num_proj_layers "$SAFELORA_NUM_LAYERS" --safelora_load_dtype float32 \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$lr" --epochs "$EPOCHS" --batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
    --dtype bfloat16 --gradient_checkpointing --save_merged_model "${push_args[@]}" \
    2>&1 | tee "${odir}/run.log"
}

for lr in "${LR_LIST[@]}"; do
  run_one "$lr" "${HF_NS}/llama2_7b-chat-gsm8k-safelora-thr${SAFELORA_THRESHOLD}-r${LORA_R}-lr${lr}"
done

echo "== safe_lora 학습/업로드 완료. eval 은 scripts/eval_gsm8k.sh 의 MODELS 에 아래 2개 추가 =="
for lr in "${LR_LIST[@]}"; do
  echo "  ${HF_NS}/llama2_7b-chat-gsm8k-safelora-thr${SAFELORA_THRESHOLD}-r${LORA_R}-lr${lr}"
done
