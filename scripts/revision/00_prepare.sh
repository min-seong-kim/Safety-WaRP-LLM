#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 00 — 데이터 준비 + 환경 사전 점검
#
#  revision 실험의 모든 arm 은 태스크마다 **JSON 파일 하나**를 공유한다.
#  이 스크립트는 그 5개 파일이 전부 존재하는지 확인하고, 없는 것만 생성한다.
#
#    gsm8k  → data/gsm8k_train_task_7473.json        (여기서 생성)
#    math   → data/math_train_task_7500.json         (scripts/prepare_math_task_data.py)
#    arc    → data/arc_challenge_train_task_1119.json (scripts/prepare_qa_task_data.py)
#    medqa  → data/medqa_train_task_10178.json        (scripts/prepare_qa_task_data.py)
#    agnews → data/agnews_train_8k_seed42.json        (기존 8k seed42 서브셋, 생성 불필요)
#
#  사용:  bash scripts/revision/00_prepare.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/00_prepare_${TS}.log") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo " Stage 00 — 데이터 준비 / 환경 점검     ts=$TS"
echo "════════════════════════════════════════════════════════════════"

# ────────────────────────── 1. 환경 점검 ──────────────────────────
hdr "1. 환경 점검"
command -v "$PY" >/dev/null || die "python 을 찾을 수 없다 (PY=$PY). 'conda activate hb' 했는가?"
"$PY" - <<'PYEOF' || die "필수 패키지 점검 실패"
import sys
import torch, transformers, peft, accelerate, datasets
print(f"  python       {sys.version.split()[0]}  ({sys.executable})")
print(f"  torch        {torch.__version__}  cuda_build={torch.version.cuda}  avail={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu          {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}  (n={torch.cuda.device_count()})")
else:
    print("  [WARN] CUDA 를 쓸 수 없다. 학습 스테이지는 실패한다.")
print(f"  transformers {transformers.__version__}   peft {peft.__version__}   accelerate {accelerate.__version__}   datasets {datasets.__version__}")
try:
    import fire
    print(f"  fire         {fire.__version__}   (SafeDelta 에 필요)")
except ImportError:
    print("  [WARN] fire 없음 — SafeDelta(safedelta) 스테이지가 실패한다.")
PYEOF

echo ""
echo "  CUDA_VISIBLE_DEVICES = '${CUDA_VISIBLE_DEVICES:-<unset>}'  (스케줄러가 넣어준 값을 그대로 쓴다)"

# HF 토큰 — gated repo(meta-llama/*, google/gemma-*) 접근에 필요하다.
if "$PY" -c "
from huggingface_hub import HfApi
try:
    print('  hf user      ' + HfApi().whoami()['name'])
except Exception as e:
    raise SystemExit(1)
" 2>/dev/null; then :; else
  warn "HF 토큰이 설정되지 않았다. meta-llama/* 와 google/gemma-2-9b-it 는 gated 라 base 모델 로드에"
  warn "  실패한다(SafeLoRA / AsFT / RESTA 가 base 를 필요로 한다)."
  warn "  해결: 셸에서  hf auth login --token <YOUR_TOKEN>   또는  export HF_TOKEN=<...>"
fi

# SafeDelta 외부 저장소
if has_method safedelta; then
  if [[ -f "$SAFEDELTA_DIR/llama2/run_safedelta.py" ]]; then
    echo "  SafeDelta    $SAFEDELTA_DIR  (OK)"
    # 주석 처리된 원본 줄은 무시하고, 실제로 실행되는 대입문만 잡는다.
    if grep -qE '^[[:space:]]*os\.environ\["CUDA_VISIBLE_DEVICES"\][[:space:]]*=' "$SAFEDELTA_DIR/llama2/run_safedelta.py"; then
      warn "SafeDelta run_safedelta.py 에 CUDA_VISIBLE_DEVICES 하드코딩이 살아 있다."
      warn "  os.environ.setdefault(...) 로 바꿔야 이 박스(GPU 1장)에서 동작한다."
    fi
  else
    warn "SafeDelta 없음: $SAFEDELTA_DIR/llama2/run_safedelta.py  → safedelta 스테이지가 실패한다."
  fi
fi

# ────────────────────────── 2. 안전 데이터 ──────────────────────────
hdr "2. 안전 데이터"
for s in cb bt; do
  p="$(safety_json "$s")"
  if [[ -f "$p" ]]; then
    n=$("$PY" -c "import json;print(len(json.load(open('$p'))))")
    echo "  [$s]  $n rows   $p"
  else
    warn "[$s] 없음: $p"
  fi
done

# ────────────────────────── 3. 태스크 데이터 ──────────────────────────
hdr "3. 태스크 데이터 생성/점검"

# 3-1. GSM8K
gsm8k_out="$(task_json gsm8k)"
if [[ -f "$gsm8k_out" ]]; then
  echo "  [gsm8k]  이미 존재 — skip"
else
  echo "  [gsm8k]  생성 중..."
  run "$PY" scripts/revision/prepare_gsm8k_task_data.py --output "$gsm8k_out" \
    || warn "gsm8k 태스크 JSON 생성 실패"
fi

# 3-2. MATH
math_out="$(task_json math)"
if [[ -f "$math_out" ]]; then
  echo "  [math]   이미 존재 — skip"
else
  echo "  [math]   생성 중 (scripts/prepare_math_task_data.py)..."
  run "$PY" scripts/prepare_math_task_data.py || warn "math 태스크 JSON 생성 실패"
fi

# 3-3. ARC-C / MedQA
for t in arc medqa; do
  out="$(task_json "$t")"
  if [[ -f "$out" ]]; then
    echo "  [$t]$( [[ $t == arc ]] && echo "    " || echo "  ")이미 존재 — skip"
  else
    echo "  [$t] 생성 중 (scripts/prepare_qa_task_data.py --tasks $t)..."
    if [[ "$t" == "medqa" && ! -f "$REPO_DIR/data/medqa_train_10178.jsonl" \
          && ! -f "$REPO_DIR/data/medqa_train_10178.json" ]]; then
      echo "       MedQA 원본이 없다 → medqa_eval/prepare_medqa_dataset.py 먼저 실행"
      run "$PY" medqa_eval/prepare_medqa_dataset.py --output_dir "$REPO_DIR/data" \
        || warn "MedQA 원본 생성 실패"
    fi
    run "$PY" scripts/prepare_qa_task_data.py --tasks "$t" || warn "$t 태스크 JSON 생성 실패"
  fi
done

# 3-4. AG News (생성 안 함 — 고정 8k seed42 서브셋)
agnews_out="$(task_json agnews)"
if [[ -f "$agnews_out" ]]; then
  echo "  [agnews] 이미 존재 — skip"
else
  warn "[agnews] 없음: $agnews_out"
  warn "  이 파일은 data/subsets_seed42.manifest.json 으로 고정 샘플링된 8k 서브셋이다."
  warn "  git 에 force-track 되어 있어야 정상이다 (.gitignore 의 !data/*.json)."
fi

# ────────────────────────── 4. 요약 ──────────────────────────
hdr "4. 태스크 데이터 요약"
"$PY" - <<'PYEOF'
import json, os, sys
REPO = os.getcwd()
FILES = {
    "gsm8k":  "data/gsm8k_train_task_7473.json",
    "math":   "data/math_train_task_7500.json",
    "arc":    "data/arc_challenge_train_task_1119.json",
    "medqa":  "data/medqa_train_task_10178.json",
    "agnews": "data/agnews_train_8k_seed42.json",
}
missing = []
print(f"  {'task':8s} {'rows':>7s}  {'q(chars)':>9s} {'r(chars)':>9s}  file")
for task, rel in FILES.items():
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        missing.append(task); print(f"  {task:8s} {'MISSING':>7s}  {'':>9s} {'':>9s}  {rel}"); continue
    rows = json.load(open(p, encoding="utf-8"))
    bad = [i for i, r in enumerate(rows)
           if not str(r.get("question", "")).strip() or not str(r.get("response", "")).strip()]
    ql = sum(len(r["question"]) for r in rows) / max(len(rows), 1)
    rl = sum(len(r["response"]) for r in rows) / max(len(rows), 1)
    flag = f"  ⚠ 빈 행 {len(bad)}개" if bad else ""
    print(f"  {task:8s} {len(rows):7d}  {ql:9.0f} {rl:9.0f}  {rel}{flag}")
if missing:
    print(f"\n  [WARN] 없는 태스크: {', '.join(missing)} — 해당 태스크 실험은 돌릴 수 없다.")
    sys.exit(0)
print("\n  5개 태스크 JSON 모두 준비됨.")
PYEOF

echo ""
echo "다음 단계:"
echo "  bash scripts/revision/01_ssft_bt.sh          # BeaverTails 안전정렬 출발모델"
echo "  bash scripts/revision/02_warp_basis_mask.sh  # Phase1 basis + Phase2 mask"
echo "  또는  bash scripts/revision/run_all.sh       # 전체"
