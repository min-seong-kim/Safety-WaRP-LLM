#!/usr/bin/env bash
# AsFT full-parameter 러너 end-to-end 스모크 테스트 (~1분).
#   작은 랜덤 LLaMA 두 벌(base / aligned)로 전 경로를 태운다. 수치가 아니라
#   "파이프라인이 끝까지 도는가" 와 "해석적 그래디언트가 autograd 와 일치하는가" 를 본다.
#
#   bash asft/scripts/_smoke_asft_full.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
WORK="${WORK:-${TMPDIR:-/tmp}/asft_full_smoke_$$}"
mkdir -p "$WORK"
echo "[smoke] work dir: $WORK"

"$PY" - "$WORK" <<'PYEOF'
import json, sys, torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
work = sys.argv[1]
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=2048)
base = LlamaForCausalLM(cfg).to(torch.bfloat16)
base.save_pretrained(f"{work}/tiny-base")
# aligned = base + 작은 섭동 → V = W_aligned − W_base 가 0 이 아니게 만든다
aligned = LlamaForCausalLM(cfg).to(torch.bfloat16)
aligned.load_state_dict(base.state_dict())
with torch.no_grad():
    for p in aligned.parameters():
        p.add_(torch.randn_like(p, dtype=torch.float32).to(torch.bfloat16) * 0.01)
aligned.save_pretrained(f"{work}/tiny-aligned")
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
tok.save_pretrained(f"{work}/tiny-base"); tok.save_pretrained(f"{work}/tiny-aligned")
json.dump(json.load(open("data/gsm8k_train_task_7473.json"))[:24],
          open(f"{work}/task.json", "w"))
print("[smoke] fixtures ready")
PYEOF

run_one() {   # <label> <extra args...>
  local label="$1"; shift
  echo "──────── [smoke] $label ────────"
  "$PY" asft/finetune_asft_full.py \
    --model_path "$WORK/tiny-aligned" --base_model "$WORK/tiny-base" \
    --output_dir "$WORK/out_${label}" \
    --task_data_path "$WORK/task.json" --task_samples 24 \
    --epochs 1 --batch_size 2 --grad_accum 2 --max_length 256 \
    --learning_rate 1e-4 --logging_steps 1 --asft_check_equiv "$@"
  test -f "$WORK/out_${label}/config.json"
  test -f "$WORK/out_${label}/finetune_config.json"
  echo "[smoke] $label OK"
}

run_one gpu
run_one offload   --asft_offload_cpu
run_one fp32store --asft_store_dtype float32
run_one lam0      --asft_lambda_reg 0.0

echo ""
echo "✅ [smoke] AsFT full-param 전 경로 통과 — $WORK"
