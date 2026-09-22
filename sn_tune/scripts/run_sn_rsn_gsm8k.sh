#!/usr/bin/env bash
# =============================================================================
# run_sn_rsn_gsm8k.sh — 원공간 SN-Tune / RSN-Tune 전체 파이프라인 (GSM8K)
#
#   0  패치 확인            (hb_sn)
#   1  safety 뉴런 검출      (hb_sn)   circuit_breakers  → N_safe
#   2  utility 뉴런 검출     (hb_sn)   wikipedia         → N_foundation
#   3  critical = 1 - 2                                  → N_robust
#   4  SN-Tune  / RSN-Tune   (hb)      safety 코퍼스로 해당 뉴런만 학습
#   5  GSM8K FT (뉴런 동결)  (hb)      나머지만 학습
#
# **환경이 두 개로 갈린다.** 검출(1,2)은 패치된 transformers 가 있는 `hb_sn` 에서,
# 학습(4,5)은 정품인 `hb` 에서 돌아야 한다. 이 스크립트가 단계마다 알아서 바꾼다.
# `hb_sn` 이 없으면 먼저: bash sn_tune/setup_hb_sn.sh
#
# 사용
#   bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
#   MODELS="llama2_7b"        bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
#   ARMS="rsn"                bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
#   DRY_RUN=1                 bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
#   STOP_AFTER_NEURONS=1      bash sn_tune/scripts/run_sn_rsn_gsm8k.sh
#
# 재실행 안전: 단계마다 `.done` 마커를 남기고 있으면 건너뛴다.
# **마커를 지워서 재실행하지 마라** — 하이퍼파라미터를 바꿨으면 OUT_ROOT 를 바꿔라.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 설정 (여기가 주 조작면)
# ---------------------------------------------------------------------------
MODELS="${MODELS:-llama2_7b llama2_13b}"
ARMS="${ARMS:-sn rsn}"                 # sn | rsn | "sn rsn"
OUT_ROOT="${OUT_ROOT:-outputs/sn_tune}"
NEURON_DIR="${NEURON_DIR:-sn_tune/output_neurons}"

DETECT_ENV="${DETECT_ENV:-hb_sn}"      # 패치된 transformers
TRAIN_ENV="${TRAIN_ENV:-hb}"           # 정품 transformers

SAFETY_JSON="${SAFETY_JSON:-data/circuit_breakers_train.json}"
SAFETY_PROMPTS="${SAFETY_PROMPTS:-4994}"
UTILITY_DOCS="${UTILITY_DOCS:-1000}"

# per-layer top-k. **safety 와 utility 가 서로 다른 값을 쓴다** — 아래 값은
# 공개된 Llama-2-7B RSN 모델을 만든 실제 실행의 로그에서 그대로 가져온 것이다
# (Safety-Neuron/neuron_detection/logs/neuron_detection/, 2026-05-03):
#
#   [Llama-2-7B-chat]
#     safety  1200/200 · circuit_breakers 4994 → 12,998 = 0.9558% 뉴런 (1.135% 파라미터)
#     utility  300/50  · wikipedia 1000        →  1,826 = 0.1343% 뉴런 (0.185% 파라미터)
#     critical                                 → 11,329            (0.967% 파라미터)
#   [Llama-2-13B-chat]  — 탐색 흔적이 로그에 남아 있다
#     safety  1200/200 → 15,828 = 0.7431%  (낮아서 버림)
#     safety  1800/300 → 20,406 = 0.9581%  ← 채택 (7B 의 0.9558% 와 나란함)
#
# utility 에 같은 1200/200 을 쓰면 foundation 집합이 과도하게 커져 critical 이
# 깎여나간다. 두 값을 같이 두지 마라.
# safety 는 모델마다 다르다 (~1% 를 맞추려면 크기에 따라 키워야 한다).
# utility 는 300/50 이 집안 표준이었다 — 7B chat/base · 3.1-8B · 3.1-8B-Instruct ·
# Qwen2.5-32B 실행이 전부 300/50 이다.
SAFETY_TOP_FFN_llama2_7b="${SAFETY_TOP_FFN_llama2_7b:-1200}"
SAFETY_TOP_ATTN_llama2_7b="${SAFETY_TOP_ATTN_llama2_7b:-200}"
SAFETY_TOP_FFN_llama2_13b="${SAFETY_TOP_FFN_llama2_13b:-1800}"
SAFETY_TOP_ATTN_llama2_13b="${SAFETY_TOP_ATTN_llama2_13b:-300}"

UTILITY_TOP_FFN="${UTILITY_TOP_FFN:-300}"
UTILITY_TOP_ATTN="${UTILITY_TOP_ATTN:-50}"

# 학습 하이퍼파라미터 — 공개된 7B/13B RSN 모델의 finetune_config.json 과 동일하다.
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"          # 유효 배치 16
MAX_LEN="${MAX_LEN:-1024}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-7473}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
DRY_RUN="${DRY_RUN:-0}"
STOP_AFTER_NEURONS="${STOP_AFTER_NEURONS:-0}"

# 출발 모델 — **plain chat 모델**이다. SSFT 모델이 아니다.
#   sn_tuning 로그 17건을 전수 확인한 결과 전부 plain 이었다
#   (Llama-2-7b-chat-hf 12건 · Llama-2-13b-chat-hf 3건 · Llama-3.2-3B-Instruct 2건).
#   설계상 당연하다 — SN-Tune 자체가 safety 정렬 단계라 SSFT 를 대체한다.
#
#   ⚠️ 이 때문에 Table 3 의 다른 행(SafeInstr / SEAL / WSR-Tune)과 출발점이 다르다.
#   그 행들은 kmseong/llama2_7b-chat-Safety-FT-lr5e-5 에서 출발한다. 이건 방법론의
#   차이지 실수가 아니지만, 표에 각주로 밝혀야 한다. 출발점을 맞추고 싶으면
#   START_llama2_7b=kmseong/llama2_7b-chat-Safety-FT-lr5e-5 로 덮어라.
START_llama2_7b="${START_llama2_7b:-meta-llama/Llama-2-7b-chat-hf}"
START_llama2_13b="${START_llama2_13b:-meta-llama/Llama-2-13b-chat-hf}"

# ---------------------------------------------------------------------------
# conda
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda 가 PATH 에 없다." >&2; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

py_of() {  # 환경 이름 -> python 경로
    local prefix; prefix="$(conda env list | awk -v n="$1" '$1==n {print $NF}')"
    [[ -n "$prefix" ]] || { echo ""; return; }
    echo "${prefix}/bin/python"
}

DETECT_PY="$(py_of "$DETECT_ENV")"
TRAIN_PY="$(py_of "$TRAIN_ENV")"

# DRY_RUN 일 때는 환경이 없어도 계획을 볼 수 있어야 한다.
if [[ -z "$DETECT_PY" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[WARN] 검출 환경 '${DETECT_ENV}' 이 없다 — DRY_RUN 이라 자리표시자로 계속한다."
        DETECT_PY="<${DETECT_ENV}>/bin/python"
    else
        echo "[ERROR] 검출 환경 '${DETECT_ENV}' 이 없다. 먼저 만들어라:" >&2
        echo "        bash sn_tune/setup_hb_sn.sh" >&2
        exit 1
    fi
fi
if [[ -z "$TRAIN_PY" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[WARN] 학습 환경 '${TRAIN_ENV}' 이 없다 — DRY_RUN 이라 자리표시자로 계속한다."
        TRAIN_PY="<${TRAIN_ENV}>/bin/python"
    else
        echo "[ERROR] 학습 환경 '${TRAIN_ENV}' 이 없다." >&2; exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
run() {   # run <마커파일> <설명> -- <명령...>
    local marker="$1"; shift
    local desc="$1"; shift
    [[ "$1" == "--" ]] && shift

    if [[ -f "$marker" ]]; then
        echo "[SKIP] ${desc}  (마커: ${marker})"
        return 0
    fi
    echo
    echo "───────────────────────────────────────────────────────────────"
    echo "[RUN ] ${desc}"
    printf '      '; printf ' %q' "$@"; echo
    echo "───────────────────────────────────────────────────────────────"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY ] 실행하지 않음"
        return 0
    fi
    mkdir -p "$(dirname "$marker")"
    "$@"
    date -Iseconds > "$marker"
}

start_model_of() {
    local key="START_$1"
    echo "${!key:-}"
}

topk_of() {  # topk_of <SAFETY_TOP_FFN|SAFETY_TOP_ATTN> <model>
    local key="$1_$2"
    echo "${!key:-}"
}

mkdir -p "$NEURON_DIR" "$OUT_ROOT"

# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
for model in $MODELS; do
    START="$(start_model_of "$model")"
    if [[ -z "$START" ]]; then
        echo "[ERROR] '${model}' 의 출발 모델이 정의되지 않았다 (START_${model})." >&2
        exit 1
    fi

    S_FFN="$(topk_of SAFETY_TOP_FFN "$model")"
    S_ATTN="$(topk_of SAFETY_TOP_ATTN "$model")"
    if [[ -z "$S_FFN" || -z "$S_ATTN" ]]; then
        echo "[ERROR] '${model}' 의 safety top-k 가 없다 " >&2
        echo "        (SAFETY_TOP_FFN_${model} / SAFETY_TOP_ATTN_${model} 을 정하라)." >&2
        echo "        새 모델이면 STOP_AFTER_NEURONS=1 로 돌려 파라미터% 를 보고 맞춰라 —" >&2
        echo "        검출 1회는 7B 기준 약 4분이다." >&2
        exit 1
    fi

    MDIR="${OUT_ROOT}/${model}"
    NDIR="${NEURON_DIR}/${model}"
    mkdir -p "$MDIR" "$NDIR"

    SAFETY_TXT="${NDIR}/safety_neurons.txt"
    UTILITY_TXT="${NDIR}/utility_neurons.txt"
    CRITICAL_TXT="${NDIR}/critical_neurons.txt"

    echo
    echo "==============================================================="
    echo " ${model}   출발 모델: ${START}"
    echo "   safety top-k ${S_FFN}/${S_ATTN}   utility top-k ${UTILITY_TOP_FFN}/${UTILITY_TOP_ATTN}"
    echo "==============================================================="

    # ── 0. 패치 확인 ────────────────────────────────────────────────────────
    if [[ "$DRY_RUN" != "1" ]]; then
        echo "[CHECK] ${DETECT_ENV} 에 패치가 걸려 있는가"
        "$DETECT_PY" -m sn_tune.verify_patch --model_name "$START" --expect present
        echo "[CHECK] ${TRAIN_ENV} 는 정품인가"
        "$TRAIN_PY"  -m sn_tune.verify_patch --model_name "$START" --expect absent
    fi

    # ── 1. safety 뉴런 ──────────────────────────────────────────────────────
    run "${NDIR}/.safety.done" "[1] safety 뉴런 검출 (${DETECT_ENV})" -- \
        "$DETECT_PY" -m sn_tune.detect_original "$SAFETY_PROMPTS" \
            --model_name "$START" \
            --dataset_file "$SAFETY_JSON" \
            --top_number_ffn "$S_FFN" --top_number_attn "$S_ATTN" \
            --safety_neuron \
            --output_file "$SAFETY_TXT"

    # ── 2. utility 뉴런 ─────────────────────────────────────────────────────
    run "${NDIR}/.utility.done" "[2] utility 뉴런 검출 (${DETECT_ENV})" -- \
        "$DETECT_PY" -m sn_tune.detect_original "$UTILITY_DOCS" \
            --model_name "$START" \
            --top_number_ffn "$UTILITY_TOP_FFN" --top_number_attn "$UTILITY_TOP_ATTN" \
            --utility_neuron \
            --output_file "$UTILITY_TXT"

    # ── 3. critical ─────────────────────────────────────────────────────────
    run "${NDIR}/.critical.done" "[3] critical = safety \\ utility" -- \
        "$TRAIN_PY" -m sn_tune.critical_neurons \
            --safety_file "$SAFETY_TXT" \
            --utility_file "$UTILITY_TXT" \
            --output_file "$CRITICAL_TXT" \
            --model_name "$START" \
            --space original

    if [[ "$STOP_AFTER_NEURONS" == "1" ]]; then
        echo "[STOP] STOP_AFTER_NEURONS=1 — 뉴런 파일까지만 만들고 멈춘다."
        continue
    fi

    # ── 4 & 5. arm 별 ───────────────────────────────────────────────────────
    for arm in $ARMS; do
        case "$arm" in
            sn)  NEURON_TXT="$SAFETY_TXT" ;;
            rsn) NEURON_TXT="$CRITICAL_TXT" ;;
            *)   echo "[ERROR] 알 수 없는 arm: ${arm}" >&2; exit 1 ;;
        esac

        ADIR="${MDIR}/${arm}"
        TUNED_DIR="${ADIR}/tuned"
        FT_DIR="${ADIR}/gsm8k"
        mkdir -p "$ADIR"

        # 4. SN-Tune / RSN-Tune — safety 코퍼스로 해당 뉴런만 학습
        run "${ADIR}/.tune.done" "[4/${arm}] ${arm^^}-Tune (${TRAIN_ENV})" -- \
            "$TRAIN_PY" -m sn_tune.sn_tune_original \
                --neuron_file "$NEURON_TXT" \
                --dataset_file "$SAFETY_JSON" \
                --model_name "$START" \
                --local_model_name "$TUNED_DIR" \
                --learning_rate "$LR" \
                --num_epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --grad_accum_steps "$GRAD_ACCUM" \
                --max_seq_length "$MAX_LEN" \
                --warmup_ratio "$WARMUP_RATIO" \
                --no_timestamp_suffix \
                --model_dir_file "${ADIR}/TUNED_MODEL_DIR"

        # 5. GSM8K FT — 위에서 쓴 그 뉴런을 이번엔 **얼린다**
        UPLOAD_ARGS=()
        if [[ "$PUSH_TO_HUB" == "1" ]]; then
            UPLOAD_ARGS=(--upload_name "${HF_NAMESPACE}/${model}-${arm}_gsm8k_lr${LR}")
        fi
        run "${ADIR}/.gsm8k.done" "[5/${arm}] GSM8K FT + ${arm^^} 동결 (${TRAIN_ENV})" -- \
            "$TRAIN_PY" -m sn_tune.finetune_freeze_sn \
                --model_path "$TUNED_DIR" \
                --safety_neurons_file "$NEURON_TXT" \
                --output_dir "$FT_DIR" \
                --num_train_samples "$GSM8K_SAMPLES" \
                --learning_rate "$LR" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --grad_accum "$GRAD_ACCUM" \
                --max_length "$MAX_LEN" \
                --weight_decay "$WEIGHT_DECAY" \
                --warmup_ratio "$WARMUP_RATIO" \
                --no_timestamp_suffix \
                --model_dir_file "${ADIR}/MODEL_DIR" \
                "${UPLOAD_ARGS[@]}"
    done
done

echo
echo "[DONE] 원공간 (R)SN-Tune 파이프라인 완료."
echo "       뉴런 파일: ${NEURON_DIR}/<model>/"
echo "       모델:      ${OUT_ROOT}/<model>/<arm>/gsm8k  (MODEL_DIR 파일에 실제 경로)"
