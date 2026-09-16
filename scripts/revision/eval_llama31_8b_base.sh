#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Table 7 · Llama-3.1-8B Base × PEFT 7종 평가
#
#    safety    : HarmBench  DirectRequest / AutoDAN / PAIR / PAP  (AdvBench standard)
#    downstream: lm-eval    gsm8k 5-shot
#
#  리포 id 는 common.sh 의 hf_repo_id() 에서 뽑는다 — **손으로 적지 않는다**.
#  학습 드라이버(run_llama31_8b_base.sh)와 같은 변수를 export 해야 같은 이름이 나온다.
#
#  ── base 모델이라 확인해야 하는 세 가지 ──────────────────────────────────
#  1) HarmBench chat_template 이 `llama-3-base` 여야 한다
#     (= "Question: {instruction}\nAnswer:", 학습 프롬프트와 글자 단위 동일).
#     register_models_basename.py 가 base 를 판정해 자동으로 붙인다 — 등록 직후
#     아래 verify 단계에서 실제 yaml 값을 다시 확인한다.
#  2) AutoDAN/PAIR 의 test case 는 `llama3_1_8b-base` 것을 재사용해야 한다
#     (Table 7 의 기존 7행이 전부 그 디렉토리에 있다). harmbench_eval.sh 의
#     GetBaseModel() 이 base 를 인식하도록 고쳐 두었다.
#  3) lm-eval 은 모델명에 chat/instruct/it 이 없으면 --apply_chat_template 을
#     붙이지 않는다. 리포명에 그 토큰이 없으므로 plain 5-shot 으로 돈다(맞는 동작).
#
#  ── 대조군(CONTROL_REPOS) ────────────────────────────────────────────────
#  Table 7 의 기존 행은 다른 박스(96GB GPU)에서 측정됐다. vLLM 의
#  gpu_memory_utilization 은 **전체 대비 비율**이라 183GB B200 에서는 KV cache
#  블록 수가 달라지고, 그러면 배치 구성이 달라져 greedy 생성 결과까지 조금
#  움직일 수 있다(register_models_basename.py 주석의 실측 사례: util 0.8 vs 0.95
#  에서 ASR 0.097 vs 0.042). 그래서 기존 행 하나를 같이 재측정해 **새 수치와 기존
#  표를 같은 잣대로 놓을 수 있는지** 먼저 확인한다. 어긋나면 새 7행만으로 표를
#  만들지 말고 기존 7행도 이 박스에서 다시 재야 한다.
#
#  사용:
#    bash scripts/revision/eval_llama31_8b_base.sh              # 전체
#    DRY_RUN=1 bash scripts/revision/eval_llama31_8b_base.sh    # 등록/매핑만 확인
#    HB_ONLY=1 ...   LM_ONLY=1 ...                              # 한쪽만
#    CONTROL_REPOS="" ...                                       # 대조군 생략
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINSEONG="$(dirname "$REPO_DIR")"
HARMBENCH_DIR="${HARMBENCH_DIR:-$MINSEONG/HarmBench}"
CONDA_ROOT="${CONDA_ROOT:-$MINSEONG/miniconda3}"
export CONDA_SH="${CONDA_SH:-$CONDA_ROOT/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-$MINSEONG/.hf_cache}"
export PY="${PY:-$CONDA_ROOT/envs/hb/bin/python}"

cd "$REPO_DIR"

# 학습 드라이버와 **같은** 값이어야 리포명이 일치한다.
export SAFETY_SETS=cb MODELS=llama31_8b_base TASKS=gsm8k
export LORA_R="${LORA_R:-16}" LORA_ALPHA="${LORA_ALPHA:-16}"
export WSR_LORA_ALPHA="${WSR_LORA_ALPHA:-16}"
export LISA_RHO="${LISA_RHO:-1.0}" KEEP_RATIO="${KEEP_RATIO:-0.3}"
export SAFELORA_THRESHOLD="${SAFELORA_THRESHOLD:-0.3}" ASFT_LAMBDA_REG="${ASFT_LAMBDA_REG:-1.0}"
export SEAL_TOPP="${SEAL_TOPP:-0.8}" SEAL_FULL_LR="${SEAL_FULL_LR:-1e-5}"

METHODS_TO_EVAL="${METHODS_TO_EVAL:-lora asft lisa seal safelora salora wsr_lora}"
# Table 7 의 WSR-Tune 행 — 이 박스에서 재측정해 기존 표와의 정합을 본다.
CONTROL_REPOS="${CONTROL_REPOS-kmseong/llama3.1-8b-base-warp-gsm8k-lr1e-5}"

# ── 1. 평가 대상 리포 id 수집 (hf_repo_id 규약에서 생성) ─────────────────────
mapfile -t REPOS < <(
  source scripts/revision/common.sh >/dev/null 2>&1
  for m in $METHODS_TO_EVAL; do
    if [[ "$m" == seal ]]; then FULL_LR="$SEAL_FULL_LR"; fi
    hf_repo_id cb llama31_8b_base gsm8k "$m"
  done
)
[[ ${#REPOS[@]} -gt 0 ]] || { echo "[eval] 리포 id 를 뽑지 못했다"; exit 1; }
for c in $CONTROL_REPOS; do REPOS+=("$c"); done

echo "════════════════════════════════════════════════════════════════"
echo "  Table 7 · Llama-3.1-8B Base 평가 대상 ${#REPOS[@]}개"
printf '    %s\n' "${REPOS[@]}"
echo "  HarmBench : $HARMBENCH_DIR"
echo "  HF_HOME   : $HF_HOME"
echo "════════════════════════════════════════════════════════════════"

# ── 2. 허브 존재 확인 (없는 걸 넣으면 배치 중간에 죽는다) ────────────────────
"$PY" - "${REPOS[@]}" <<'PYEOF'
import sys
from huggingface_hub import HfApi
api, missing = HfApi(), []
for r in sys.argv[1:]:
    try:
        api.model_info(r)
        print(f"  OK      {r}")
    except Exception as e:
        print(f"  MISSING {r}  ({type(e).__name__})")
        missing.append(r)
if missing:
    print("\n[eval] 허브에 없는 리포가 있다 — 학습/업로드가 끝났는지 확인할 것:")
    for r in missing:
        print("   -", r)
    sys.exit(2)
PYEOF
rc=$?
if (( rc != 0 )); then
  echo "[eval] 대상 모델이 모두 올라가기 전에는 평가를 시작하지 않는다."
  exit $rc
fi

# ── 3. models.yaml 등록 + base chat_template 실측 확인 ──────────────────────
cd "$HARMBENCH_DIR"
echo ""
echo "▶ models.yaml 등록"
printf '%s\n' "${REPOS[@]}" | "$PY" register_models_basename.py --write || true

echo ""
echo "▶ 등록 결과 확인 (base 모델은 chat_template 이 llama-3-base 여야 한다)"
"$PY" - "${REPOS[@]}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/model_configs/models.yaml", encoding="utf-8"))

# base QA 프롬프트("Question: {instruction}\nAnswer:")로 해석되는 chat_template 값들.
#   baselines/model_utils.py 에서 `llama-2-base` 와 `llama-3-base` 는 **둘 다**
#   Llama_3B_QA_BASE_PROMPT 로 매핑되는 별칭이다. 문자열이 아니라 실제로 어떤
#   TEMPLATE 이 되는지를 봐야 한다 — 기존 Table 7 항목들은 `llama-2-base` 로 적혀 있다.
BASE_QA_ALIASES = {"llama-2-base", "llama-3-base"}

bad = []
for repo in sys.argv[1:]:
    hits = [k for k, v in cfg.items()
            if isinstance(v, dict) and (v.get("model") or {}).get("model_name_or_path") == repo]
    if not hits:
        print(f"  ✗ 미등록  {repo}"); bad.append(repo); continue
    for k in hits:
        tpl = cfg[k]["model"].get("chat_template")
        ok = tpl in BASE_QA_ALIASES
        note = "  (= llama-3-base 와 동일 프롬프트)" if ok and tpl != "llama-3-base" else ""
        print(f"  {'✓' if ok else '✗'} {k:62s} chat_template={tpl}{note}")
        if not ok:
            bad.append(k)
if bad:
    print("\n[eval] base 모델인데 QA 템플릿이 아니다 — 학습 프롬프트와 어긋난다. 중단.")
    sys.exit(3)
PYEOF
rc=$?
(( rc == 0 )) || exit $rc

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "[dry-run] 여기까지. 실제 평가는 DRY_RUN 없이 다시 실행."
  DRY_RUN=1 ./run_all_eval.sh "${REPOS[@]}"
  exit 0
fi

# ── 4. HarmBench(4종) + lm-eval(gsm8k) ─────────────────────────────────────
#   GPU 1장 → run_all_eval.sh 가 PARALLEL=0 / LM_GPU=0 으로 자동 전환한다.
#   AUTO_TASK_FALLBACK 은 필요 없다(리포명에 gsm8k 가 들어 있어 자동 매칭된다).
exec env \
  HB_GPU="${HB_GPU:-0}" LM_GPU="${LM_GPU:-0}" \
  GPU_UTIL_CAP="${GPU_UTIL_CAP:-0.9}" \
  HB_VARIANTS="${HB_VARIANTS:-sys}" \
  RESUME="${RESUME:-true}" LMEVAL_RESUME="${LMEVAL_RESUME:-1}" \
  VALIDATE_REPOS="${VALIDATE_REPOS:-1}" \
  PREFETCH_MODE="${PREFETCH_MODE:-blocking}" \
  SEED="${SEED:-42}" \
  LM_TASKS="${LM_TASKS:-gsm8k}" \
  ./run_all_eval.sh "${REPOS[@]}"
