#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  재학습한 gemma2_9b SEAL(lr 1e-5) 평가 — HarmBench 4종 + lm-eval gsm8k
#
#  WAIT_PID 가 주어지면 그 학습 프로세스가 끝나고 `.uploaded` 가 생길 때까지 기다린 뒤 돈다.
#  (같은 GPU 를 쓰므로 겹치면 안 된다.)
#
#  gemma 특유의 주의점 — 전부 기존 코드가 처리하므로 여기서 재확인만 한다:
#   · HarmBench models.yaml 은 chat_template: gemma + **block_size: 32** 가 필요하다
#     (head_size 256 이라 FlashInfer 의 block_size 16 에서 죽는다).
#     add_models_to_yaml.build_block() 이 gemma 키에 자동으로 넣는다.
#   · lm-eval 도 같은 이유로 MODEL_ARGS 에 block_size 를 넣는다(eval_models.sh 의 gemma 분기).
#   · AutoDAN/PAIR test case 는 gemma2_9b_it 것을 재사용한다(GetBaseModel).
#
#  사용:
#    bash scripts/revision/eval_gemma_seal.sh                  # 즉시 평가
#    WAIT_PID=<학습 PID> bash scripts/revision/eval_gemma_seal.sh   # 학습 끝난 뒤 평가
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_DIR"
MINSEONG="$(dirname "$REPO_DIR")"
HARMBENCH_DIR="${HARMBENCH_DIR:-$MINSEONG/HarmBench}"
CONDA_ROOT="${CONDA_ROOT:-$MINSEONG/miniconda3}"
export CONDA_SH="${CONDA_SH:-$CONDA_ROOT/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-$MINSEONG/.hf_cache}"
export PY="${PY:-$CONDA_ROOT/envs/hb/bin/python}"

CELL="${CELL:-$REPO_DIR/outputs/revision_gemma_seal_lr1e-5/cb/gemma2_9b/gsm8k/seal}"
REPO="$(GEMMA2_9B_FULL_LR=1e-5 bash -c 'source scripts/revision/common.sh >/dev/null 2>&1; hf_repo_id cb gemma2_9b gsm8k seal')"
echo "[eval-gemma-seal] 대상 리포: $REPO"

# ── 학습 대기 ─────────────────────────────────────────────────────────────
if [[ -n "${WAIT_PID:-}" ]]; then
  echo "[eval-gemma-seal] 학습 PID $WAIT_PID 종료 대기..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[eval-gemma-seal] 학습 종료 $(date '+%F %T')"
fi

# ── 업로드 확인 (허브에 없으면 평가할 수 없다) ─────────────────────────────
if [[ ! -f "$CELL/.done" ]]; then
  echo "[eval-gemma-seal] ✗ 학습이 완료되지 않았다 ($CELL/.done 없음). 중단."
  exit 1
fi
if ! "$PY" - "$REPO" <<'PYEOF'
import sys
from huggingface_hub import HfApi
try:
    info = HfApi().model_info(sys.argv[1])
    n = sum(1 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    print(f"  허브 확인 OK — safetensors {n}개")
    sys.exit(0 if n > 0 else 1)
except Exception as e:
    print(f"  허브에 없다: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
then
  echo "[eval-gemma-seal] ✗ 허브에 모델이 없다 — 업로드가 끝났는지 확인할 것. 중단."
  exit 2
fi

# ── GPU 가 빌 때까지 대기 (학습 프로세스 정리 여유) ────────────────────────
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ "${used:-0}" -lt 5000 ]] && break
  echo "[eval-gemma-seal] GPU ${used}MiB 사용 중 — 대기"; sleep 60
done

cd "$HARMBENCH_DIR"
echo ""; echo "▶ models.yaml 등록 (gemma → chat_template: gemma + block_size: 32)"
printf '%s\n' "$REPO" | "$PY" register_models_basename.py --write || true
"$PY" - "$REPO" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/model_configs/models.yaml", encoding="utf-8"))
repo = sys.argv[1]
hits = [(k, v["model"]) for k, v in cfg.items()
        if isinstance(v, dict) and (v.get("model") or {}).get("model_name_or_path") == repo]
if not hits:
    print("  ✗ 미등록"); sys.exit(3)
for k, m in hits:
    tpl, bs = m.get("chat_template"), m.get("block_size")
    ok = (tpl == "gemma") and (bs == 32)
    print(f"  {'✓' if ok else '✗'} {k}  chat_template={tpl} block_size={bs}")
    if not ok:
        print("    ⚠️ gemma 는 chat_template=gemma · block_size=32 여야 한다 "
              "(head_size 256 → FlashInfer block_size 16 에서 죽는다).")
        sys.exit(4)
PYEOF
rc=$?; (( rc == 0 )) || exit $rc

exec env HB_GPU=0 LM_GPU=0 GPU_UTIL_CAP="${GPU_UTIL_CAP:-0.9}" \
  HB_VARIANTS="${HB_VARIANTS:-sys}" RESUME=true LMEVAL_RESUME=1 \
  VALIDATE_REPOS=1 PREFETCH_MODE=blocking SEED=42 LM_TASKS=gsm8k \
  ./run_all_eval.sh "$REPO"
