#!/usr/bin/env bash
# SafeGrad 러너 end-to-end 스모크 테스트 (~1분).
#   작은 랜덤 LLaMA + 실제 태스크/안전 JSON 앞부분 몇십 개로 전 경로를 한 번 태운다.
#   수치가 아니라 "파이프라인이 끝까지 도는가" 만 본다.
#
#   bash safegrad/scripts/_smoke_safegrad.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$PWD"
PY="${PY:-python}"
WORK="${WORK:-${TMPDIR:-/tmp}/safegrad_smoke_$$}"
mkdir -p "$WORK"
echo "[smoke] work dir: $WORK"

# 1) 작은 랜덤 모델 + 실제 토크나이저 규약을 흉내낸 tokenizer
"$PY" - "$WORK" <<'PYEOF'
import json, sys, torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
work = sys.argv[1]
cfg = LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=2048)
m = LlamaForCausalLM(cfg).to(torch.bfloat16)
m.save_pretrained(f"{work}/tiny-model")
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
tok.save_pretrained(f"{work}/tiny-model")
# 태스크/안전 데이터 앞부분만 잘라 쓴다
task = json.load(open("data/gsm8k_train_task_7473.json"))[:24]
json.dump(task, open(f"{work}/task.json", "w"))
safe = json.load(open("data/circuit_breakers_train.json"))[:16]
json.dump(safe, open(f"{work}/safety.json", "w"))
print("[smoke] fixtures ready")
PYEOF

run_one() {   # <label> <extra args...>
  local label="$1"; shift
  echo "──────── [smoke] $label ────────"
  "$PY" safegrad/finetune_safegrad.py \
    --model_path "$WORK/tiny-model" \
    --output_dir "$WORK/out_${label}" \
    --task_data_path "$WORK/task.json" --task_samples 24 \
    --safety_data_path "$WORK/safety.json" --guide_data_num 16 \
    --epochs 1 --batch_size 2 --grad_accum 2 --max_length 256 \
    --learning_rate 1e-4 --logging_steps 1 \
    --lora_r 4 --lora_alpha 8 "$@"
  test -f "$WORK/out_${label}/config.json"
  test -f "$WORK/out_${label}/finetune_config.json"
  echo "[smoke] $label OK"
}

run_one separate    --ref_mode separate
run_one adapteroff  --ref_mode adapter_off
run_one noproj      --ref_mode adapter_off --no_projection
run_one klresponse  --ref_mode adapter_off --kl_reduction response --kl_fp32

echo ""
echo "✅ [smoke] SafeGrad 전 경로 통과 — $WORK"
