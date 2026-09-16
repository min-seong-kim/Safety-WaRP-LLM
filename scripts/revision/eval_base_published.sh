#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  논문 Table 7 의 **기존 행 모델들**을 이 박스에서 재평가한다 (base 라인 공용).
#
#    LINE=llama31_8b_base bash scripts/revision/eval_base_published.sh
#    LINE=llama2_7b_base  bash scripts/revision/eval_base_published.sh
#
#  왜 도는가
#  ---------
#  Table 7 은 safety 와 downstream 이 **서로 다른 모델**에서 나온 행이 있다
#  (llama3.1-8b: safety=lr1e-5 / GSM8K=lr5e-5 — 2026-09-16 규명, RESULTS.md 참조).
#  또 llama2-7b base 행은 이 박스의 HarmBench 결과 트리에서 아예 재현되지 않는다
#  (AutoDAN 10 × PAIR 10 디렉토리 조합을 전수로 훑어도 논문값과 맞는 키가 없다).
#  여기서 ASR 과 GSM8K 를 **한 배치에서 함께** 재서, 새로 학습한 PEFT 행과
#  "한 모델 = 한 행" 기준으로 나란히 놓을 수 있게 한다.
#
#  RESUME=true 이므로 이미 결과가 있는 (모델 × 공격) 조합은 건너뛴다. 강제로 다시
#  재려면 RESUME=false. 2026-09-16 확인: 이 박스는 옛 박스 수치를 소수점까지 재현한다
#  (WSR-Tune·Full FT 두 모델에서 4개 공격 모두 0.00 차이).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_DIR"
MINSEONG="$(dirname "$REPO_DIR")"
HARMBENCH_DIR="${HARMBENCH_DIR:-$MINSEONG/HarmBench}"
CONDA_ROOT="${CONDA_ROOT:-$MINSEONG/miniconda3}"
export CONDA_SH="${CONDA_SH:-$CONDA_ROOT/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-$MINSEONG/.hf_cache}"
export PY="${PY:-$CONDA_ROOT/envs/hb/bin/python}"

LINE="${LINE:-llama31_8b_base}"
case "$LINE" in
  llama31_8b_base)
    # 전부 lr 1e-5 계열. RESTA 는 원본 리포가 허브에서 404 라 재생성본을 쓴다
    # (finish_resta_l31base.sh 가 만든 것. RESULTS.md 의 † 각주 참조).
    REPOS=(
      "kmseong/Llama-3.1-8B-base-SSFT_lr5e-5"                      # SSFT 출발 모델
      "kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5"                # Full Params FT
      "kmseong/llama3.1-8b-base-gsm8k-safeinstr-ratio_10p-lr1e-5"  # SafeInstr
      "kmseong/llama3.1-8b-base-gsm8k-safedelta-scale0.1-lr1e-5"   # SafeDelta
      "kmseong/llama3.1-8B_base_gsm8k_ft_freeze_sn_lr1e-5"         # SN-Tune
      "kmseong/llama3.1-8B_base_gsm8k_ft_freeze_rsn_lr1e-5"        # RSN-Tune
      "kmseong/llama3.1-8b-base-warp-gsm8k-lr1e-5"                 # WSR-Tune
    ) ;;
  llama2_7b_base)
    # 전부 lr 3e-5 계열 (이 라인의 full-param lr).
    # ⚠️ 이 7개는 2026-04 배치다. 9월에 학습한 PEFT 셀들은 출발 모델이
    #    kmseong/llama2_7b-base-CB_SSFT-lr3e-5 (2026-09-14) 인데, 이 7개가 같은
    #    SSFT 체크포인트에서 나왔는지는 **확인되지 않았다**. 표에 밝힐 것.
    REPOS=(
      "kmseong/llama2_7b-base-gsm8k_ssft_lr3e-5"                   # Full Params FT
      "kmseong/llama2_7b_base-gsm8k_safelnstr_10p_lr3e-5"          # SafeInstr (repo명 오타 safelnstr)
      "kmseong/llama2_7b_base_resta_lr3e-5_y0.3"                   # RESTA γ=0.3
      "kmseong/llama2_7b_base_safedelta_gsm8k_s0.1"                # SafeDelta s=0.1
      "kmseong/llama2_7b_base_gsm8k_ft_freeze_sn_lr3e-5"           # SN-Tune
      "kmseong/llama2_7b_base_gsm8k_ft_freeze_rsn_lr3e-5"          # RSN-Tune
      "kmseong/llama2_7B_SSFT_WaRP_safety_basis_gsm8k_FT_lr3e-5"   # WSR-Tune
    ) ;;
  *) echo "[eval-pub] 알 수 없는 LINE: $LINE (llama31_8b_base | llama2_7b_base)" >&2; exit 1 ;;
esac


if [[ -n "${WAIT_PID:-}" ]]; then
  echo "[eval-pub] 앞 작업 PID $WAIT_PID 종료 대기..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[eval-pub] 앞 작업 종료 $(date '+%F %T')"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Table 7 기존 행 재평가 — LINE=$LINE · ${#REPOS[@]}개"
printf '    %s\n' "${REPOS[@]}"
echo "════════════════════════════════════════════════════════════════"

# ── 존재 확인 + base 모델인데 lm-eval 이 chat template 을 붙이지 않는지 ─────
"$PY" - "${REPOS[@]}" <<'PYEOF'
import sys
from huggingface_hub import HfApi
api, bad = HfApi(), []
for r in sys.argv[1:]:
    nm = r.split("/")[-1].lower()
    chatty = ("instruct" in nm) or ("chat" in nm) or ("it" in nm)   # eval_models.sh 와 같은 규칙
    try:
        api.model_info(r); ok = "OK "
    except Exception:
        ok = "404"; bad.append(r)
    print(f"  {ok:4s} {r}" + ("   ⚠️ base 인데 lm-eval 이 chat template 을 붙인다" if chatty else ""))
    if chatty: bad.append(r + " (chat-template 오판정)")
if bad:
    print("\n[eval-pub] 진행 불가:"); [print("   -", b) for b in bad]; sys.exit(2)
PYEOF
rc=$?; (( rc == 0 )) || exit $rc

# ── GPU 대기 ──────────────────────────────────────────────────────────────
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ "${used:-0}" -lt 5000 ]] && break
  echo "[eval-pub] GPU ${used}MiB 사용 중 — 대기"; sleep 60
done

cd "$HARMBENCH_DIR"
echo ""; echo "▶ models.yaml 등록"
printf '%s\n' "${REPOS[@]}" | "$PY" register_models_basename.py --write || true

echo ""; echo "▶ 등록 확인 (base QA 템플릿이어야 한다)"
"$PY" - "${REPOS[@]}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/model_configs/models.yaml", encoding="utf-8"))
BASE_QA = {"llama-2-base", "llama-3-base"}   # 둘 다 Llama_3B_QA_BASE_PROMPT 로 매핑되는 별칭
bad = []
for repo in sys.argv[1:]:
    hits = [k for k, v in cfg.items()
            if isinstance(v, dict) and (v.get("model") or {}).get("model_name_or_path") == repo]
    if not hits:
        print(f"  ✗ 미등록 {repo}"); bad.append(repo); continue
    for k in hits:
        tpl = cfg[k]["model"].get("chat_template")
        ok = tpl in BASE_QA
        print(f"  {'✓' if ok else '✗'} {k:52s} tpl={tpl}")
        if not ok: bad.append(k)
if bad:
    print("\n[eval-pub] base 모델인데 QA 템플릿이 아닌 항목이 있다. 중단.")
    sys.exit(3)
PYEOF
rc=$?; (( rc == 0 )) || exit $rc

exec env HB_GPU=0 LM_GPU=0 GPU_UTIL_CAP="${GPU_UTIL_CAP:-0.9}" \
  HB_VARIANTS="${HB_VARIANTS:-sys}" RESUME="${RESUME:-true}" LMEVAL_RESUME="${LMEVAL_RESUME:-1}" \
  VALIDATE_REPOS=1 PREFETCH_MODE=blocking SEED=42 LM_TASKS=gsm8k \
  ./run_all_eval.sh "${REPOS[@]}"
