#!/usr/bin/env bash
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3/envs/hb/bin/python
for m in llama2_7b:meta-llama/Llama-2-7b-chat-hf llama2_13b:meta-llama/Llama-2-13b-chat-hf; do
  key=${m%%:*}; mdl=${m#*:}
  out="outputs/wsr_sn_tune/${key}/phase1"
  if [ -f "outputs/wsr_sn_tune/${key}/BASIS_DIR" ]; then echo "[SKIP] $key basis 있음"; continue; fi
  echo "=== Phase 1 basis: $key ($mdl) ==="
  mkdir -p "$out"
  "$PY" train.py --phase 1 --phase0_model_dir "$mdl" \
      --safety_dataset circuit_breakers --circuit_breakers_path data/circuit_breakers_train.json \
      --circuit_breakers_samples_phase1 4994 --basis_save_dtype bfloat16 \
      --batch_size 2 --max_length 1024 \
      --layer_type ffn_up,ffn_down,attn_q,attn_k,attn_v --target_layers all \
      --output_dir "$out" --log_dir "$out/logs" \
      --device cuda --dtype bfloat16 --seed 42 --no_wandb
  rc=$?
  p1=$(find "$out" -maxdepth 1 -name 'phase1_*' -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ $rc -eq 0 ] && [ -d "$p1/basis" ]; then
    echo "$p1/basis" > "outputs/wsr_sn_tune/${key}/BASIS_DIR"
    echo "[DONE] $key -> $p1/basis"
  else
    echo "[FAIL] $key rc=$rc"; exit 1
  fi
done
echo "=== ALL BASIS DONE ==="
