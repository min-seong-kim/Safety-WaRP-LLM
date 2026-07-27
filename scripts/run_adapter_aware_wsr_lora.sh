#!/bin/bash
# Adapter-aware, column-structured WSR projection for LoRA.
# 기본은 W1~W3 artifact 생성까지만 수행한다. 전체 GSM8K C~F는 RUN_TRAINING=1 필요.
set -euo pipefail
cd "$(dirname "$0")/.."

HBPY=${HBPY:-python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}
OUT_ROOT=${OUT_ROOT:-outputs/adapter_aware_wsr_projected_lora}
SAFETY_ADAPTER=${SAFETY_ADAPTER:-outputs/adapter_subspace_lora/safety_adapter}
SAFETY_MERGED=${SAFETY_MERGED:-outputs/adapter_subspace_lora/safety_merged}
SAFETY_DATA=${SAFETY_DATA:-./data/circuit_breakers_train.json}
BASIS_LINK=${WSR_BASIS_DIR:-${OUT_ROOT}/phase1_basis}
SCORES_DIR=${OUT_ROOT}/column_scores
SUBSPACE_ROOT=${OUT_ROOT}/subspaces
TARGET_MODULES=q_proj,k_proj,v_proj,up_proj,down_proj
LAYER_TYPES=attn_q,attn_k,attn_v,ffn_up,ffn_down
EXPECTED_MODULES=${EXPECTED_MODULES:-160}
RUN_TRAINING=${RUN_TRAINING:-0}
LORA_R=${LORA_R:-16}; LORA_ALPHA=${LORA_ALPHA:-32}; LR=${LR:-1e-4}
mkdir -p "$OUT_ROOT" "$SUBSPACE_ROOT" logs

complete_report () {
  local report=$1 expected=$2
  [ -f "$report" ] || return 1
  "$HBPY" - "$report" "$expected" <<'PYEOF'
import json, sys
r=json.load(open(sys.argv[1])); expected=int(sys.argv[2])
raise SystemExit(0 if r.get("num_modules") == expected else 1)
PYEOF
}
complete_basis () {
  local root=$1 expected=$2
  [ -f "${root}/metadata.json" ] || return 1
  "$HBPY" - "${root}/metadata.json" "$expected" <<'PYEOF'
import json, sys
r=json.load(open(sys.argv[1])); expected=int(sys.argv[2])
raise SystemExit(0 if r.get("num_layers_saved") == expected else 1)
PYEOF
}

# W1: 반드시 safety LoRA merged W_safe에서 activation basis를 재생성한다.
if complete_basis "$BASIS_LINK" "$EXPECTED_MODULES"; then
  echo "[W1] skip: 완전한 basis artifact ${BASIS_LINK}"
else
  echo "[W1] safety-merged model에서 WSR activation basis 생성"
  stage_root=${OUT_ROOT}/phase1_runs
  before=$(find "$stage_root" -type d -path '*/basis' 2>/dev/null | sort || true)
  "$HBPY" train.py --phase 1 --phase0_model_dir "$SAFETY_MERGED" \
    --safety_dataset circuit_breakers --circuit_breakers_path "$SAFETY_DATA" \
    --layer_type "$LAYER_TYPES" --target_layers all --device cuda --dtype bfloat16 \
    --batch_size 1 --output_dir "$stage_root" --log_dir logs --no_wandb
  latest=$(find "$stage_root" -type d -path '*/basis' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
  [ -n "$latest" ] || { echo "Phase 1 basis를 찾지 못했습니다"; exit 1; }
  rm -f "$BASIS_LINK"
  ln -s "$(realpath "$latest")" "$BASIS_LINK"
fi

# W2: backward 1회에서 세 점수를 모두 산출. raw G는 기본 저장하지 않는다.
if complete_report "${SCORES_DIR}/report.json" "$EXPECTED_MODULES"; then
  echo "[W2] skip: 완전한 column-score artifact"
else
  "$HBPY" compute_wsr_column_scores.py \
    --safety_merged_model_path "$SAFETY_MERGED" \
    --safety_adapter_path "$SAFETY_ADAPTER" --wsr_basis_dir "$BASIS_LINK" \
    --out_dir "$SCORES_DIR" --safety_data_path "$SAFETY_DATA" \
    --target_modules "$TARGET_MODULES" --layer_type "$LAYER_TYPES" \
    --target_layers all --column_aggregation l2 --importance_chunk_size 128 \
    --importance_dtype float32 --batch_size 1 --dtype bfloat16 \
    --require_safety_adapter_for_all_targets
  complete_report "${SCORES_DIR}/report.json" "$EXPECTED_MODULES" || {
    echo "W2 incomplete: expected ${EXPECTED_MODULES} modules"; exit 1; }
fi

build_subspace () {
  local tag=$1 mode=$2; shift 2
  local out=${SUBSPACE_ROOT}/${tag}
  if complete_report "${out}/report.json" "$EXPECTED_MODULES"; then
    echo "[W3:${tag}] skip"
  else
    "$HBPY" build_adapter_wsr_subspace.py --column_scores_dir "$SCORES_DIR" \
      --wsr_basis_dir "$BASIS_LINK" --out_dir "$out" --layer_type "$LAYER_TYPES" \
      --importance_mode "$mode" "$@"
    complete_report "${out}/report.json" "$EXPECTED_MODULES" || exit 1
  fi
}
build_subspace gradient_ratio1 gradient_only --direction_keep_ratio 0.01
build_subspace taylor_ratio1 adapter_taylor --direction_keep_ratio 0.01
build_subspace taylor_top16 adapter_taylor --direction_top_k 16
build_subspace random_ratio1 adapter_taylor --direction_keep_ratio 0.01 \
  --random_control --random_seed 42 --match_k_from "${SUBSPACE_ROOT}/taylor_ratio1"

if [ "$RUN_TRAINING" != 1 ]; then
  echo "W1~W3 완료. 전체 C~F 학습은 RUN_TRAINING=1로 다시 실행하세요."
  exit 0
fi

for tag in gradient_ratio1 taylor_ratio1 taylor_top16 random_ratio1; do
  run=${OUT_ROOT}/runs/${tag}/lr_${LR}
  if [ -f "${run}/summary.json" ]; then echo "[W4:${tag}] skip"; continue; fi
  "$HBPY" finetune_gsm8k_lora.py --method adapter_aware_wsr_projected_lora \
    --model_name "$SAFETY_MERGED" --adapter_wsr_subspace_dir "${SUBSPACE_ROOT}/${tag}" \
    --output_dir "$run" --target_modules "$TARGET_MODULES" --layer_type "$LAYER_TYPES" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --learning_rate "$LR" \
    --epochs 3 --batch_size 2 --gradient_accumulation_steps 8 --max_length 1024 \
    --dtype bfloat16 --lora_param_dtype float32 --gradient_checkpointing \
    --require_safety_adapter_for_all_targets --save_merged_model
done
