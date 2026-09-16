#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Llama-3.1-8B Base 의 RESTA 행 복구: 병합 결과 검증 → 업로드 → 평가
#
#  왜 새로 만드는가
#  ----------------
#  논문 Table 7 의 Resta 행(AVG 8.37 / GSM8K 38.82)을 만든 리포
#  `kmseong/llama3.1-8b-base-lr5e-5-gsm8k-resta-gamma0.3` 와, 사용자가 지목한
#  `kmseong/llama3.1-8b-base-gsm8k-resta-gamma0.3-lr1e-5` 가 **둘 다 허브에서 404** 다
#  (2026-09-16 확인. kmseong 전체에 llama3.1-8b **base** 용 resta 리포가 하나도 없다).
#  그래서 같은 레시피로 다시 만든다:
#      W_resta = W_ft + γ·(W_align − W_base),  γ = 0.3   (논문 §4.1)
#      W_ft    = kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5   (Table 7 Full FT 행)
#      W_align = kmseong/Llama-3.1-8B-base-SSFT_lr5e-5         (SSFT 출발 모델)
#      W_base  = meta-llama/Llama-3.1-8B
#  ⚠️ 원본과 **bit-identical 하지는 않을 수 있다** — 없어진 리포가 정확히 어떤 W_ft 로
#     만들어졌는지 확인할 방법이 없다. 새 행임을 표에 밝힐 것.
#
#  사용: [WAIT_PID=<병합 PID>] bash scripts/revision/finish_resta_l31base.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_DIR"
MINSEONG="$(dirname "$REPO_DIR")"
HARMBENCH_DIR="${HARMBENCH_DIR:-$MINSEONG/HarmBench}"
CONDA_ROOT="${CONDA_ROOT:-$MINSEONG/miniconda3}"
export CONDA_SH="${CONDA_SH:-$CONDA_ROOT/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-$MINSEONG/.hf_cache}"
export PY="${PY:-$CONDA_ROOT/envs/hb/bin/python}"

CELL="$REPO_DIR/outputs/revision/cb/llama31_8b_base/gsm8k/resta"
REPO="$(bash -c 'source scripts/revision/common.sh >/dev/null 2>&1; hf_repo_id cb llama31_8b_base gsm8k resta')"
echo "[resta] 셀=$CELL"
echo "[resta] 리포=$REPO"

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "[resta] 병합 PID $WAIT_PID 종료 대기..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[resta] 병합 종료 $(date '+%F %T')"
fi

# ── 1. 병합 결과 검증 ──────────────────────────────────────────────────────
"$PY" - "$CELL" <<'PYEOF'
import json, os, sys, glob
d = sys.argv[1]
need = ["config.json", "model.safetensors.index.json", "tokenizer.json"]
missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
print(f"  safetensors 샤드 {len(shards)}개, 누락 부속파일 {missing or '없음'}")
if missing or not shards:
    sys.exit(1)
# index 가 가리키는 모든 샤드가 실제로 있는가
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
want = set(idx["weight_map"].values())
have = {os.path.basename(p) for p in shards}
if want - have:
    print(f"  ✗ index 가 가리키는데 없는 샤드: {sorted(want - have)}"); sys.exit(2)
print("  ✓ index ↔ 샤드 일치")
PYEOF
rc=$?; (( rc == 0 )) || { echo "[resta] ✗ 병합 산출물이 불완전하다. 중단."; exit $rc; }

# ── 2. 실제로 로드되는지 + 병합이 반영됐는지 ──────────────────────────────
"$PY" - "$CELL" <<'PYEOF'
import sys, torch
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open
import glob, os
d = sys.argv[1]
AutoConfig.from_pretrained(d); print("  ✓ AutoConfig 로드")
t = AutoTokenizer.from_pretrained(d)
print(f"  ✓ AutoTokenizer 로드 (chat_template={'있음' if getattr(t,'chat_template',None) else '없음'} — base 라 없는 게 정상)")
# 첫 샤드의 한 텐서가 finite 한지 (γ 덧셈이 깨지지 않았는지 최소 확인)
p = sorted(glob.glob(os.path.join(d, "*.safetensors")))[0]
with safe_open(p, framework="pt") as f:
    k = next(iter(f.keys())); v = f.get_tensor(k)
    print(f"  ✓ {k}: dtype={v.dtype} finite={bool(torch.isfinite(v.float()).all())} absmax={v.float().abs().max():.4f}")
PYEOF
rc=$?; (( rc == 0 )) || { echo "[resta] ✗ 로드 검증 실패. 중단."; exit $rc; }

date -Iseconds > "$CELL/.done"
echo "$CELL" > "$CELL/MODEL_DIR"

# ── 3. 업로드 ─────────────────────────────────────────────────────────────
echo ""; echo "▶ 업로드 → $REPO"
"$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$REPO" || {
  echo "[resta] ✗ 업로드 실패. 중단."; exit 3; }

# ── 4. GPU 대기 후 평가 ───────────────────────────────────────────────────
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ "${used:-0}" -lt 5000 ]] && break
  echo "[resta] GPU ${used}MiB 사용 중 — 대기"; sleep 120
done

cd "$HARMBENCH_DIR"
echo ""; echo "▶ models.yaml 등록"
printf '%s\n' "$REPO" | "$PY" register_models_basename.py --write || true
"$PY" - "$REPO" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/model_configs/models.yaml", encoding="utf-8"))
BASE_QA = {"llama-2-base", "llama-3-base"}
hits = [(k, v["model"]) for k, v in cfg.items()
        if isinstance(v, dict) and (v.get("model") or {}).get("model_name_or_path") == sys.argv[1]]
if not hits: print("  ✗ 미등록"); sys.exit(3)
for k, m in hits:
    ok = m.get("chat_template") in BASE_QA
    print(f"  {'✓' if ok else '✗'} {k}  tpl={m.get('chat_template')}")
    if not ok: sys.exit(4)
PYEOF
rc=$?; (( rc == 0 )) || exit $rc

exec env HB_GPU=0 LM_GPU=0 GPU_UTIL_CAP="${GPU_UTIL_CAP:-0.9}" \
  HB_VARIANTS=sys RESUME=true LMEVAL_RESUME=1 VALIDATE_REPOS=1 \
  PREFETCH_MODE=blocking SEED=42 LM_TASKS=gsm8k \
  ./run_all_eval.sh "$REPO"
