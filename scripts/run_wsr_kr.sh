#!/bin/bash
#SBATCH -J wsr_kr
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gokms0509/Safety-WaRP-LLM/logs/wsr_kr_%j.out
#SBATCH --error=/home/gokms0509/Safety-WaRP-LLM/logs/wsr_kr_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition gigabyte_a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# WSR keep_ratio 스윕 (트레이드오프 곡선용): Phase2 per-layer mask → wsr_lora 학습. 고정 lr=2e-4.
#   메모리: 5개 layer_type U(fp32, down_proj 11008²·32≈15.5GB 등 총 ~23.5GB)+basis_coeff+grad 가
#   48GB 초과(a100 80GB 필요하나 QOS 없음). WaRP reparam 은 forward 를 바꾸지 않으므로(정확 재param)
#   layer_type 별 gradient importance 는 서로 독립 → Phase2 를 [ffn_down] 과 [나머지 4] 로 쪼개
#   각각 48GB 안에서 돌린 뒤 mask 를 병합한다(단일 실행과 결과 동일). 학습(Phase3)은 model+U≈36GB 로 48GB 안에 들어감.
# 제출: sbatch scripts/run_wsr_kr.sh <KEEP_RATIO>   (예: 0.05 0.15 0.2 0.3)

KR="${1:?usage: sbatch run_wsr_kr.sh <keep_ratio>}"
LR="${2:-2e-4}"

cd /home/gokms0509/Safety-WaRP-LLM
source /home/gokms0509/anaconda3/etc/profile.d/conda.sh
conda activate hb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
HF_NS=kmseong; PUSH="${PUSH:-1}"
BASIS="checkpoints/phase1_sweep_basis"

OUT_ROOT="/scratch2/gokms0509/lora_comparison/wsr_sweep"
odir="${OUT_ROOT}/kr${KR}_lr${LR}"
mkdir -p "$odir"
if [ -f "${odir}/summary.json" ]; then echo "skip wsr kr${KR} lr${LR} (done)"; exit 0; fi
git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

if ! ls "$BASIS"/*/layer_*_svd.pt >/dev/null 2>&1; then
  echo "ERROR: basis missing at $BASIS — run scripts/run_wsr_basis.sh first"; exit 1
fi

# ── Phase 2: per-layer importance mask (layer_type 분할로 48GB 안에서) ──
P2ROOT="checkpoints/wsr_kr${KR}"
gen_mask () {  # $1=layer_type list  $2=tag  → stdout: masks dir path
  # ⚠️ 한 local 문에서 뒤 변수가 앞 변수를 참조하면 set -u 에서 unbound(인자 선확장) → 분리 대입.
  local lts="$1" tag="$2"
  local out="${P2ROOT}/${tag}"
  local md
  md=$(find "$out" -type d -path '*phase2_*/checkpoints/masks' 2>/dev/null | head -1)
  if [ -n "$md" ] && ls "$md"/*/layer_*_mask.pt >/dev/null 2>&1; then echo "$md"; return; fi
  echo "[wsr_kr] Phase2 mask kr=${KR} lts=${lts}" 1>&2
  python train.py --phase 2 --perlayer \
    --phase0_model_dir "$MODEL" --basis_dir "$BASIS" \
    --dataset_phase2 circuit_breakers --keep_ratio "$KR" --batch_size 1 \
    --layer_type "$lts" --target_layers all \
    --output_dir "$out" --log_dir ./logs \
    --device cuda --dtype bfloat16 --no_wandb 1>&2
  find "$out" -type d -path '*phase2_*/checkpoints/masks' 2>/dev/null | head -1
}

# Phase2 importance 는 gradient checkpointing 이 없어 (모듈 수 × 재구성 weight autograd 그래프 + 활성화)
# 가 관건 → layer_type 을 1개씩 완전분할해야 48GB 에 들어간다(ffn_down 단독은 이미 통과 확인).
LTS=(ffn_down ffn_up attn_q attn_k attn_v)
declare -A MD
for lt in "${LTS[@]}"; do
  MD[$lt]=$(gen_mask "$lt" "p2_${lt}")
  if [ -z "${MD[$lt]}" ]; then echo "ERROR: mask gen failed for $lt"; exit 1; fi
done

# 병합: 5개 layer_type subdir 를 한 dir 로 심볼릭
COMBINED="${P2ROOT}/masks_combined"
mkdir -p "$COMBINED"
for lt in "${LTS[@]}"; do
  ln -sfn "$(realpath "${MD[$lt]}/$lt")" "$COMBINED/$lt"
done
MASK_DIR="$COMBINED"
echo "[wsr_kr] combined mask_dir=$MASK_DIR"
ls -la "$MASK_DIR"

# ── Phase 3: wsr_lora 학습 ──
repo="${HF_NS}/llama2_7b-chat-gsm8k-wsr-lora-elem-kr${KR}-r${LORA_R}-lr${LR}"
push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
echo "[wsr_kr] train kr=${KR} lr=${LR} → ${repo}"
python finetune_gsm8k_lora.py --method wsr_lora --model_name "$MODEL" \
  --output_dir "$odir" --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" \
  --basis_dir "$BASIS" --mask_dir "$MASK_DIR" --keep_ratio "$KR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --learning_rate "$LR" --epochs "$EPOCHS" --batch_size "$BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
  --dtype bfloat16 --gradient_checkpointing --save_merged_model "${push_args[@]}" \
  2>&1 | tee "${odir}/run.log"
echo "== wsr kr${KR} lr${LR} 완료 =="
