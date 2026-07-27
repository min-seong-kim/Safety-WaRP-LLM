#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WSR-Tune vs ActSVD ablation 스모크 테스트 (수 분, 소형 랜덤 LLaMA).
#
# 목적: 7B 실험을 돌리기 전에 Phase 1(입력/출력 기저) → Phase 2(arm별 마스크)
#       → Phase 3(마스킹 학습 + 복원 저장) 경로가 실제로 끝까지 도는지 확인한다.
#       정확도는 보지 않는다. 형상·예산·복원 오차만 본다.
#
# 사용: bash scripts/_smoke_wsr_actsvd.sh [ARMS...]      (기본: A B C D)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_DIR="$(pwd)"

PY=${PY:-/venv/hb/bin/python}
DEVICE=${DEVICE:-cuda}
WORK=${WORK:-${TMPDIR:-/tmp}/wsr_actsvd_smoke}
ARMS=${*:-"A B C D D_perm"}
LAYER_TYPES=attn_q,ffn_up,ffn_down
NUM_LAYERS=2
EXPECTED_MODULES=$((NUM_LAYERS * 3))

rm -rf "$WORK"; mkdir -p "$WORK"
echo "smoke workdir: $WORK   arms: $ARMS   device: $DEVICE"

# ── 1. 소형 랜덤 LLaMA + 토크나이저 ─────────────────────────────────
"$PY" - "$WORK" "$NUM_LAYERS" <<'PYEOF'
import sys, torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

work, num_layers = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained("kmseong/llama2_7b-chat-Safety-FT-lr5e-5")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = LlamaConfig(
    vocab_size=len(tok), hidden_size=128, intermediate_size=344,
    num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=4,
    max_position_embeddings=1024, torch_dtype="bfloat16",
)
model = LlamaForCausalLM(cfg).to(torch.bfloat16)
# 모듈 형상이 m≠n 인 경우(up/down)를 포함해야 입력/출력 기저 혼동을 잡을 수 있다
print("q_proj", tuple(model.model.layers[0].self_attn.q_proj.weight.shape))
print("up_proj", tuple(model.model.layers[0].mlp.up_proj.weight.shape))
print("down_proj", tuple(model.model.layers[0].mlp.down_proj.weight.shape))
model.save_pretrained(f"{work}/tiny_model")
tok.save_pretrained(f"{work}/tiny_model")
PYEOF

# ── 2. 소형 safety 데이터 (저장소 파일은 건드리지 않는다) ──
"$PY" - "$WORK" <<'PYEOF'
import json, sys
work = sys.argv[1]
src = json.load(open("data/circuit_breakers_train.json"))
json.dump(src[:8], open(f"{work}/cb_small.json", "w"))
PYEOF
CB="${WORK}/cb_small.json"

COMMON=( --phase0_model_dir "${WORK}/tiny_model"
         --circuit_breakers_path "$CB"
         --layer_type "$LAYER_TYPES" --target_layers all
         --device "$DEVICE" --dtype bfloat16 --seed 42
         --log_dir "${WORK}/logs" --no_wandb )

newest () { find "$1" -maxdepth 1 -type d -name "$2*" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-; }

# ── 3. 기저 2종 ────────────────────────────────────────────────────
for SIDE in input output; do
  echo "── Phase 1 ($SIDE basis) ──"
  "$PY" train.py --phase 1 "${COMMON[@]}" \
    --safety_dataset circuit_breakers --circuit_breakers_samples_phase1 8 \
    --basis_side "$SIDE" --basis_save_dtype float32 --gram_dtype float32 \
    --batch_size 2 --max_length 256 \
    --output_dir "${WORK}/basis_${SIDE}" > "${WORK}/p1_${SIDE}.out" 2>&1 \
    || { echo "FAIL: phase1 $SIDE"; tail -30 "${WORK}/p1_${SIDE}.out"; exit 1; }
  DIR="$(newest "${WORK}/basis_${SIDE}" "phase1_${SIDE}_")/basis"
  echo "$DIR" > "${WORK}/basis_${SIDE}.path"
  "$PY" - "$DIR" "$EXPECTED_MODULES" "$SIDE" <<'PYEOF'
import json, sys
meta = json.load(open(f"{sys.argv[1]}/metadata.json"))
assert meta["num_layers_saved"] == int(sys.argv[2]), meta["num_layers_saved"]
assert meta["basis_side"] == sys.argv[3], meta["basis_side"]
worst = max(d["orthogonality_err_after_cast"] for d in meta["diagnostics"]["per_layer"].values())
print(f"  ✓ {meta['basis_side']} basis: {meta['num_layers_saved']} modules, "
      f"worst orthogonality err = {worst:.2e}, storage = {meta['storage_gib']*1024:.1f} MiB")
assert worst < 1e-2, worst
PYEOF
done

# ── 4. arm별 Phase 2 → Phase 3 ─────────────────────────────────────
for ARM in $ARMS; do
  echo "── arm $ARM ──"
  P2_EXTRA=()
  case "$ARM" in
    A) BASIS_ARGS=() ;;
    B) BASIS_ARGS=( --basis_dir "$(cat "${WORK}/basis_output.path")" ) ;;
    *) BASIS_ARGS=( --basis_dir "$(cat "${WORK}/basis_input.path")" ) ;;
  esac

  "$PY" train.py --phase 2 "${COMMON[@]}" "${BASIS_ARGS[@]}" "${P2_EXTRA[@]}" \
    --ablation_arm "$ARM" --keep_ratio 0.10 \
    --dataset_phase2 circuit_breakers \
    --circuit_breakers_samples_phase2 8 \
    --batch_size 1 --max_length 256 \
    --output_dir "${WORK}/arm_${ARM}/p2" > "${WORK}/p2_${ARM}.out" 2>&1 \
    || { echo "FAIL: phase2 arm $ARM"; tail -30 "${WORK}/p2_${ARM}.out"; exit 1; }
  MASKS="$(newest "${WORK}/arm_${ARM}/p2" "phase2_arm${ARM}_")/checkpoints/masks"
  mkdir -p "${WORK}/arm_${ARM}"; echo "$MASKS" > "${WORK}/arm_${ARM}/masks.path"
  grep -E "total frozen|budget check" "${WORK}/p2_${ARM}.out" | sed 's/^/  /'

  "$PY" train.py --phase 3 "${COMMON[@]}" "${BASIS_ARGS[@]}" \
    --masks_dir "$MASKS" --ablation_arm "$ARM" --keep_ratio 0.10 \
    --phase3_dataset gsm8k --gsm8k_samples 8 \
    --epochs 1 --utility_lr 1e-4 --batch_size 1 --gradient_accumulation_steps 2 \
    --max_length 256 \
    --output_dir "${WORK}/arm_${ARM}/p3" > "${WORK}/p3_${ARM}.out" 2>&1 \
    || { echo "FAIL: phase3 arm $ARM"; tail -40 "${WORK}/p3_${ARM}.out"; exit 1; }
  FINAL="$(newest "${WORK}/arm_${ARM}/p3" phase3_)/final_model"
  echo "$FINAL" > "${WORK}/arm_${ARM}/final_model.path"
  grep -E "frozen scalars|reconstruction rel-err" "${WORK}/p3_${ARM}.out" | tail -2 | sed 's/^/  /'

  # 저장된 모델이 표준 nn.Linear 구조로 복원됐는지 + 실제로 학습이 반영됐는지
  "$PY" - "$FINAL" "${WORK}/tiny_model" <<'PYEOF'
import sys, torch
from safetensors.torch import load_file
import glob, os
def weights(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        out.update(load_file(f))
    return out
new, old = weights(sys.argv[1]), weights(sys.argv[2])
assert new, "저장된 safetensors 없음"
leftovers = [k for k in new if any(t in k for t in ("basis_coeff", "UT_forward", "coeff_mask"))]
assert not leftovers, f"WaRP 버퍼가 남았습니다: {leftovers[:3]}"
key = "model.layers.0.self_attn.q_proj.weight"
delta = (new[key].float() - old[key].float()).abs().max().item()
print(f"  ✓ saved as plain nn.Linear; |ΔW| max on {key} = {delta:.3e}")
assert delta > 0, "학습이 전혀 반영되지 않았습니다"
PYEOF
done

# ── 5. 리포트 ──────────────────────────────────────────────────────
for SIDE in input output; do cp "${WORK}/basis_${SIDE}.path" "${WORK}/basis_${SIDE}.path.bak"; done
mkdir -p "${WORK}/report_root"
cp "${WORK}/basis_input.path" "${WORK}/report_root/basis_input.path"
cp "${WORK}/basis_output.path" "${WORK}/report_root/basis_output.path"
for ARM in $ARMS; do
  mkdir -p "${WORK}/report_root/arm_${ARM}"
  cp "${WORK}/arm_${ARM}"/*.path "${WORK}/report_root/arm_${ARM}/" 2>/dev/null || true
done
"$PY" wsr_actsvd_ablation_report.py --root "${WORK}/report_root" --keep_ratio 0.10

echo ""
echo "✅ SMOKE TEST PASSED  (logs: $WORK)"
