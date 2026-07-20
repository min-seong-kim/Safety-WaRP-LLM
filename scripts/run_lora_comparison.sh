#!/bin/bash
# WSR-LoRA 비교: Standard LoRA / Original-space Projected LoRA / WSR-LoRA(element-wise)
# 세 방법 동일 시작 checkpoint·GSM8K·seed·rank·trainable param, 차이는 증분 좌표계/제약뿐.
# GPU 1 전용. 결과 dense 모델 HF push → HarmBench(ASR) + lm-eval(GSM8K) 로 표 산출.
set -euo pipefail

cd "$(dirname "$0")/.."
HBPY=/home/users/minseong/.conda/envs/hb/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── config ──
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
TARGET_MODULES="q_proj,k_proj,v_proj,down_proj,up_proj"
SAFETY_DATA="./data/circuit_breakers_train.json"
KEEP_RATIO=0.1          # WSR element freeze 비율
DIR_KEEP_RATIO=0.1      # original 열 보호 비율 (다른 물리량)
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4 2e-4)
HF_NS=kmseong; PUSH=1

# 기존 완성 basis 재사용 (Phase 1 스킵). 없으면 train.py --phase 1 로 생성.
BASIS_DIR="checkpoints/phase1_20260713_144203/basis"
OUT_ROOT="outputs/lora_comparison"
WSR_MASK_DIR="${OUT_ROOT}/artifacts/wsr_masks"     # train.py --phase 2 --perlayer 산출 masks 디렉토리
SAFECOLS_DIR="${OUT_ROOT}/artifacts/orig_safecols"
mkdir -p "$OUT_ROOT/artifacts"

git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

# ── 1) WSR element mask (Phase 2 per-layer) ──
if [ ! -d "$WSR_MASK_DIR" ]; then
  echo "[1] WSR per-layer masks..."
  $HBPY train.py --phase 2 --perlayer \
    --phase0_model_dir "$MODEL" --basis_dir "$BASIS_DIR" \
    --dataset_phase2 circuit_breakers --keep_ratio "$KEEP_RATIO" --batch_size 1 \
    --layer_type "$LAYER_TYPE" --target_layers all \
    --device cuda --dtype bfloat16 --no_wandb \
    --output_dir checkpoints --log_dir logs
  # train.py 는 checkpoints/phase2_TS/checkpoints/masks 로 저장 → 최신 것을 심볼릭 연결
  latest=$(find checkpoints -type d -path '*phase2_*/checkpoints/masks' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2)
  ln -sfn "$(realpath "$latest")" "$WSR_MASK_DIR"
fi

# ── 2) original-space safe_cols ──
if [ ! -d "$SAFECOLS_DIR" ]; then
  echo "[2] original-space safe_cols..."
  $HBPY models/build_lora_safety_artifacts.py \
    --model_name "$MODEL" --safety_data_path "$SAFETY_DATA" \
    --out_dir "$SAFECOLS_DIR" --layer_type "$LAYER_TYPE" --target_layers all \
    --direction_keep_ratio "$DIR_KEEP_RATIO" --batch_size 2 --max_length "$MAXLEN"
fi

# ── 3) 6 run: 3 method × 2 LR ──
run_one () {
  local method=$1 lr=$2 tag=$3 repo=$4
  local odir="${OUT_ROOT}/${method}/lr_${lr}"
  if [ -f "${odir}/summary.json" ]; then echo "skip ${method} lr${lr} (done)"; return; fi
  mkdir -p "$odir"
  local extra=()
  [ "$method" = "wsr_lora" ] && extra=(--basis_dir "$BASIS_DIR" --mask_dir "$WSR_MASK_DIR")
  [ "$method" = "original_projected_lora" ] && extra=(--safecols_dir "$SAFECOLS_DIR")
  local push_args=(); [ "$PUSH" = "1" ] && push_args=(--push_to_hub --hf_repo_id "$repo")
  echo "[3] ${method} lr=${lr} → ${repo}"
  $HBPY finetune_gsm8k_lora.py --method "$method" --model_name "$MODEL" \
    --output_dir "$odir" --layer_type "$LAYER_TYPE" --target_modules "$TARGET_MODULES" \
    --keep_ratio "$KEEP_RATIO" --direction_keep_ratio "$DIR_KEEP_RATIO" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$lr" --epochs "$EPOCHS" --batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_length "$MAXLEN" --seed "$SEED" \
    --dtype bfloat16 --gradient_checkpointing --save_merged_model "${extra[@]}" "${push_args[@]}" \
    2>&1 | tee "${odir}/run.log"
}

for lr in "${LR_LIST[@]}"; do
  lrs=${lr//e-/e-}
  run_one lora "$lr" "lora-lr${lr}"                     "${HF_NS}/llama2_7b-chat-gsm8k-lora-r${LORA_R}-lr${lr}"
  run_one original_projected_lora "$lr" "orig-lr${lr}"  "${HF_NS}/llama2_7b-chat-gsm8k-origproj-lora-kr${DIR_KEEP_RATIO}-r${LORA_R}-lr${lr}"
  run_one wsr_lora "$lr" "wsr-lr${lr}"                  "${HF_NS}/llama2_7b-chat-gsm8k-wsr-lora-elem-kr${KEEP_RATIO}-r${LORA_R}-lr${lr}"
done

echo "== 학습/업로드 완료. 이제 eval =="
echo "GSM8K : /home/users/minseong/lm-evaluation-harness/eval_models.sh 의 model_list 에 6 HF ID 추가, task_list=(gsm8k)"
echo "ASR   : /home/users/minseong/HarmBench/configs/model_configs/models.yaml 에 6 엔트리 추가 후 harmbench_eval.sh"
