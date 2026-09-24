#!/usr/bin/env bash
# =============================================================================
# eval_wsr_rsn_col.sh — 열 버전 WSR-RSN-Tune 셀 평가 (HarmBench 4공격 + GSM8K 5-shot)
#
# 저장소 RESULTS.md 의 다른 표와 **동일 조건**:
#   HarmBench : conda `harmbench`, AdvBench standard, Direct/AutoDAN/PAIR/PAP,
#               sys 모드(STRIP_SAFETY_SYSTEM_PROMPT=0, RESULT_TAG=""), GRADING=hard, SEED=42
#               (AutoDAN/PAIR test case 는 family base 것을 harmbench_eval.sh 가 자동 재사용)
#   lm-eval   : conda `hb`, gsm8k 5-shot, exact_match flexible-extract
#
# 사용
#   bash sn_tune/scripts/eval_wsr_rsn_col.sh                       # MODEL_DIR 가 있는 셀 전부
#   CELLS="outputs/wsr_rsn_col_p3/llama2_7b/k1200_200_u300_50" bash ...
#   WAIT_PID_FILE=logs/wsr_rsn_col/orch.pid bash ...              # 학습 오케스트레이터 종료까지 대기
#
# models.yaml 키 규약: <model>-chat-wsr_rsn_col_<tag>-gsm8k-lr5e-5
#   (harmbench_eval.sh 의 family 판정이 'llama27b*'/'llama213b*' 접두를 보므로 앞부분을 지키자)
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/wsr_rsn_col_p3}"
HB="${HB:-$HOME/HarmBench}"
LM="${LM:-$HOME/lm-evaluation-harness}"
YAML="$HB/configs/model_configs/models.yaml"
LOG_ROOT="${LOG_ROOT:-logs/wsr_rsn_col}"
WAIT_PID_FILE="${WAIT_PID_FILE:-}"
GPU_UTIL_7B="${GPU_UTIL_7B:-0.85}"
GPU_UTIL_13B="${GPU_UTIL_13B:-0.7}"
SKIP_HB="${SKIP_HB:-0}"; SKIP_LM="${SKIP_LM:-0}"
mkdir -p "$LOG_ROOT"

log() { echo "[$(date '+%F %T')] $*"; }

# 선행 학습 종료 대기 — pid 파일만 본다 (pgrep 금지: CLAUDE.md 함정)
if [[ -n "$WAIT_PID_FILE" && -f "$WAIT_PID_FILE" ]]; then
    p=$(cat "$WAIT_PID_FILE")
    while kill -0 "$p" 2>/dev/null; do sleep 60; done
    log "[wait] $WAIT_PID_FILE ($p) 종료"
fi

# 셀 수집
if [[ -z "${CELLS:-}" ]]; then
    CELLS="$(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type d | sort | while read -r d; do [[ -f "$d/MODEL_DIR" ]] && echo "$d"; done)"
fi
[[ -n "$CELLS" ]] || { echo "[ERROR] 평가할 셀이 없다 (MODEL_DIR 없음)"; exit 1; }

KEYS=(); DIRS=(); FAMS=()
for cell in $CELLS; do
    model="$(basename "$(dirname "$cell")")"; tag="$(basename "$cell")"
    mdir="$(readlink -f "$(cat "$cell/MODEL_DIR")")"   # HarmBench/lm-eval 은 다른 cwd 에서 돈다 → 절대경로 필수
    [[ -f "$mdir/config.json" ]] || { log "[skip] $cell: 모델 없음 ($mdir)"; continue; }
    key="${model}-chat-wsr_rsn_col_${tag}-gsm8k-lr5e-5"
    KEYS+=("$key"); DIRS+=("$mdir"); FAMS+=("$model")
done
(( ${#KEYS[@]} )) || { echo "[ERROR] 유효한 셀이 없다"; exit 1; }

# ── models.yaml 에 로컬 경로 항목 추가 (키가 이미 있으면 건너뜀; 텍스트 append 라 기존 포맷 불변) ──
cp -n "$YAML" "$YAML.bak_$(date +%Y%m%d)" 2>/dev/null || true
for i in "${!KEYS[@]}"; do
    key="${KEYS[$i]}"; mdir="${DIRS[$i]}"; fam="${FAMS[$i]}"
    if grep -q "^${key}:" "$YAML"; then log "[yaml] 있음: $key"; continue; fi
    util="$GPU_UTIL_7B"; [[ "$fam" == "llama2_13b" ]] && util="$GPU_UTIL_13B"
    cat >> "$YAML" <<EOF

${key}:
  model:
    model_name_or_path: ${mdir}
    use_fast_tokenizer: False
    dtype: float16
    chat_template: llama-2
    max_model_len: 4096
    gpu_memory_utilization: ${util}
  num_gpus: 1
  model_type: open_source
EOF
    log "[yaml] 추가: $key → $mdir"
done

source "$(conda info --base)/etc/profile.d/conda.sh"
# /tmp 가 noexec 라 Triton .so 를 mmap 못 한다 → 캐시를 HOME 으로 (CLAUDE.md 함정)
export TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.torchinductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [[ "$SKIP_HB" != "1" ]]; then
    log "########## 1. HarmBench (sys · keyword · 4공격 × ${#KEYS[@]}모델) ##########"
    conda activate harmbench
    ( cd "$HB" && CUDA_VISIBLE_DEVICES=0 MODELS_OVERRIDE="$(printf '%s\n' "${KEYS[@]}")" \
        STRIP_SAFETY_SYSTEM_PROMPT=0 RESULT_TAG="" SEED=42 RESUME=true PREFETCH=0 \
        bash ./harmbench_eval.sh )
    log "=== HarmBench rc=$? ==="
    conda deactivate
fi

if [[ "$SKIP_LM" != "1" ]]; then
    log "########## 2. lm-eval GSM8K 5-shot ##########"
    conda activate hb
    for i in "${!KEYS[@]}"; do
        key="${KEYS[$i]}"; mdir="${DIRS[$i]}"; fam="${FAMS[$i]}"
        util="$GPU_UTIL_7B"; [[ "$fam" == "llama2_13b" ]] && util="$GPU_UTIL_13B"
        out="$LM/logs/wsr_rsn_col/$key"
        if ls "$out"/*/results_*.json >/dev/null 2>&1; then log "[skip] lm-eval 있음: $key"; continue; fi
        log "---- lm-eval: $key"
        ( cd "$LM" && CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
            --model_args "pretrained=$mdir,seed=42,dtype=auto,gpu_memory_utilization=$util,max_model_len=4096,tensor_parallel_size=1,enforce_eager=True" \
            --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "$out" 2>&1 | tail -8 )
    done
    conda deactivate
fi

log "########## 3. 요약 ##########"
conda activate hb
python sn_tune/summarize_wsr_rsn_col.py --out_root "$OUT_ROOT" --hb "$HB" --lm "$LM"
log "EVAL COMPLETE"
