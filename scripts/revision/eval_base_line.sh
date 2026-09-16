#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  논문 Table 7 (base 모델) 라인 평가 — HarmBench 4종 + lm-eval gsm8k
#
#    MODEL_KEY=llama31_8b_base  bash scripts/revision/eval_base_line.sh
#    MODEL_KEY=llama2_7b_base   bash scripts/revision/eval_base_line.sh
#
#  리포 id 는 **손으로 적지 않는다**. common.sh 의 hf_repo_id() 에 셀별 하이퍼파라미터를
#  환경변수로 주입해 뽑는다(CELLS). 같은 기법을 ρ 두 값으로 돌린 경우까지 이 방식으로
#  커버된다 — llama2_7b_base 는 lisa ρ=0.0/1.0, wsr_lora ρ=0.1/0.3 이 모두 허브에 있다.
#
#  ── base 모델이라 확인해야 하는 세 가지 ──────────────────────────────────
#  1) HarmBench chat_template 이 base QA 템플릿이어야 한다.
#     `llama-2-base` 와 `llama-3-base` 는 **같은 문자열**("Question: {instruction}\nAnswer:",
#     baselines/model_utils.py 의 Llama_3B_QA_BASE_PROMPT)로 매핑되는 별칭이라 둘 다 통과.
#     register_models_basename.py 가 base 를 판정해 자동으로 붙이고, 아래에서 실측 확인한다.
#  2) AutoDAN/PAIR test case 는 `<family>-base` 것을 재사용해야 한다
#     (논문 Table 7 행들이 그 디렉토리에 있다). harmbench_eval.sh 의 GetBaseModel() 을
#     base 인식하도록 고쳐 두었다.
#  3) lm-eval 은 모델명에 chat/instruct/it 이 있으면 --apply_chat_template 을 붙인다.
#     base 리포명에 그 토큰이 없어야 plain 5-shot 으로 돈다 — 아래에서 확인한다.
#
#  주요 env:
#    MODEL_KEY       llama31_8b_base | llama2_7b_base   (common.sh 레지스트리 키)
#    CELLS           "<method>[:VAR=val[,VAR=val]] ..." 형식. 비우면 모델별 기본값.
#    CONTROL_REPOS   함께 재측정할 기존(논문) 리포. 빈 문자열이면 생략.
#    DRY_RUN=1       등록/매핑 확인까지만
#    HB_ONLY=1 / LM_ONLY=1
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

MODEL_KEY="${MODEL_KEY:-llama31_8b_base}"
export SAFETY_SETS=cb MODELS="$MODEL_KEY"

# LoRA 계열은 이 라인 전체가 α=16 이다(리포명에 _a16 이 붙는다).
export LORA_R="${LORA_R:-16}" LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ── 모델별 셀 목록 · task · 대조군 ────────────────────────────────────────
#   "<method>:VAR=val,VAR=val"  — VAR 은 hf_repo_id 가 보는 하이퍼파라미터 변수다.
case "$MODEL_KEY" in
  llama31_8b_base)
    TASK="${TASK:-gsm8k}"
    DEFAULT_CELLS="lora: asft:ASFT_LAMBDA_REG=1.0 lisa:LISA_RHO=1.0 \
seal:FULL_LR=1e-5,SEAL_TOPP=0.8 safelora:SAFELORA_THRESHOLD=0.3 \
salora:SALORA_R_S=32,SALORA_R_T=32 wsr_lora:KEEP_RATIO=0.3,WSR_LORA_ALPHA=16"
    # 논문 WSR-Tune 행의 safety 가 나온 바로 그 모델 (2026-09-16 재측정에서 4공격 전부 일치)
    DEFAULT_CONTROL="kmseong/llama3.1-8b-base-warp-gsm8k-lr1e-5" ;;
  llama2_7b_base)
    TASK="${TASK:-gsm8k}"
    # 2026-09-14~15 배치. LISA 는 ρ 두 값, WSR-LoRA 도 ρ 두 값이 허브에 있다.
    # SEAL 은 이 라인의 full-param lr 이 3e-5 다(SSFT 도 3e-5).
    DEFAULT_CELLS="lora: asft:ASFT_LAMBDA_REG=1.0 lisa:LISA_RHO=0.0 lisa:LISA_RHO=1.0 \
seal:FULL_LR=3e-5,SEAL_TOPP=0.8 safelora:SAFELORA_THRESHOLD=0.3 \
salora:SALORA_R_S=32,SALORA_R_T=32 wsr_lora:KEEP_RATIO=0.1,WSR_LORA_ALPHA=16 \
wsr_lora:KEEP_RATIO=0.3,WSR_LORA_ALPHA=16"
    DEFAULT_CONTROL="" ;;
  *)
    echo "[eval] 알 수 없는 MODEL_KEY: $MODEL_KEY (llama31_8b_base | llama2_7b_base)" >&2
    exit 1 ;;
esac
export TASKS="$TASK"
CELLS="${CELLS:-$DEFAULT_CELLS}"
CONTROL_REPOS="${CONTROL_REPOS-$DEFAULT_CONTROL}"

# ── 1. 리포 id 생성 (hf_repo_id 규약에서) ──────────────────────────────────
REPOS=()
for cell in $CELLS; do
  method="${cell%%:*}"; kvs="${cell#*:}"
  envs=()
  if [[ -n "$kvs" && "$kvs" != "$cell" ]]; then
    IFS=',' read -ra pairs <<< "$kvs"
    for kv in "${pairs[@]}"; do [[ -n "$kv" ]] && envs+=("$kv"); done
  fi
  id="$(env "${envs[@]}" bash -c '
        source scripts/revision/common.sh >/dev/null 2>&1
        hf_repo_id cb '"$MODEL_KEY"' '"$TASK"' '"$method"'')"
  [[ -n "$id" ]] && REPOS+=("$id")
done
[[ ${#REPOS[@]} -gt 0 ]] || { echo "[eval] 리포 id 를 뽑지 못했다"; exit 1; }
for c in $CONTROL_REPOS; do REPOS+=("$c"); done

echo "════════════════════════════════════════════════════════════════"
echo "  Table 7 base 라인 평가 — MODEL_KEY=$MODEL_KEY  task=$TASK"
echo "  대상 ${#REPOS[@]}개"
printf '    %s\n' "${REPOS[@]}"
echo "════════════════════════════════════════════════════════════════"

# ── 2. 허브 존재 확인 + lm-eval 프롬프트 판정 확인 ─────────────────────────
"$PY" - "${REPOS[@]}" <<'PYEOF'
import re, sys
from huggingface_hub import HfApi
api, bad = HfApi(), []
for r in sys.argv[1:]:
    name = r.split("/")[-1].lower()
    # eval_models.sh 의 판정과 **같은 규칙**(단순 부분문자열): instruct|chat|it 이 있으면
    # --apply_chat_template 이 붙는다. base 리포명에 이게 걸리면 평가 프롬프트가 학습과 달라진다.
    chatty = ("instruct" in name) or ("chat" in name) or ("it" in name)
    try:
        api.model_info(r)
        ok = "OK "
    except Exception as e:
        ok = "MISSING"; bad.append(r)
    flag = "  ⚠️ lm-eval 이 chat template 을 붙인다(base 인데!)" if chatty else ""
    print(f"  {ok:8s} {r}{flag}")
    if chatty:
        bad.append(r + " (chat-template 오판정)")
if bad:
    print("\n[eval] 진행 불가:")
    for r in bad: print("   -", r)
    sys.exit(2)
PYEOF
rc=$?; (( rc == 0 )) || { echo "[eval] 중단."; exit $rc; }

# ── 3. models.yaml 등록 + base 템플릿 실측 확인 ───────────────────────────
cd "$HARMBENCH_DIR"
echo ""; echo "▶ models.yaml 등록"
printf '%s\n' "${REPOS[@]}" | "$PY" register_models_basename.py --write || true

echo ""; echo "▶ 등록 결과 확인 (base QA 템플릿이어야 한다)"
"$PY" - "${REPOS[@]}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/model_configs/models.yaml", encoding="utf-8"))
# baselines/model_utils.py 에서 둘 다 Llama_3B_QA_BASE_PROMPT 로 매핑되는 별칭이다.
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
        note = "  (= base QA 템플릿, llama-3-base 와 동일 문자열)" if ok and tpl != "llama-3-base" else ""
        print(f"  {'✓' if ok else '✗'} {k:64s} chat_template={tpl}{note}")
        if not ok: bad.append(k)
if bad:
    print("\n[eval] base 모델인데 QA 템플릿이 아니다 — 학습 프롬프트와 어긋난다. 중단.")
    sys.exit(3)
PYEOF
rc=$?; (( rc == 0 )) || exit $rc

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""; echo "[dry-run] 등록/검증까지 확인 완료. 실제 평가는 DRY_RUN 없이 재실행."
  DRY_RUN=1 ./run_all_eval.sh "${REPOS[@]}"
  exit 0
fi

# ── 4. HarmBench(4종) + lm-eval ───────────────────────────────────────────
exec env \
  HB_GPU="${HB_GPU:-0}" LM_GPU="${LM_GPU:-0}" \
  GPU_UTIL_CAP="${GPU_UTIL_CAP:-0.9}" \
  HB_VARIANTS="${HB_VARIANTS:-sys}" \
  RESUME="${RESUME:-true}" LMEVAL_RESUME="${LMEVAL_RESUME:-1}" \
  VALIDATE_REPOS="${VALIDATE_REPOS:-1}" \
  PREFETCH_MODE="${PREFETCH_MODE:-blocking}" \
  SEED="${SEED:-42}" \
  LM_TASKS="${LM_TASKS:-$TASK}" \
  ./run_all_eval.sh "${REPOS[@]}"
