#!/usr/bin/env bash
# =============================================================================
# run_wsr_rsn_row_gsm8k.sh — WSR-(R)SN-Tune, **행 마스크** 버전 (GSM8K)
#
#   [1] Phase 1 basis U            (train.py;  BASIS_DIR_<model> 로 재사용 가능)
#   [2] 회전 safety 검출            score[i] = Σ|x U Wᵀ|_i   → 출력뉴런(행)
#   [3] 회전 utility 검출           동일 방식, wikipedia
#   [4] critical = [2] \ [3]
#   [5] (R)SN-Tune                 해당 뉴런만 safety 데이터로 학습
#   [6] GSM8K FT                   해당 뉴런을 얼리고 나머지 학습
#
# 검출(2,3)만 **패치된 transformers** 가 필요하다. PYTHONPATH 오버레이로 해결한다:
#     bash sn_tune/setup_patched_transformers.sh
# 학습(5,6)은 정품으로 돌아야 하므로 PYTHONPATH 를 주지 않는다.
#
# **행 마스크는 원공간 동결과 수학적으로 동일하다** (W̃ = W U 가 행을 보존).
# 그래서 학습은 원공간 도구를 그대로 쓴다 — sn_tune/RESULTS.md §B-1 참고.
# 열(column) 버전은 run_wsr_sn_rsn_gsm8k.sh 쪽이다.
#
# 사용
#   bash sn_tune/scripts/run_wsr_rsn_row_gsm8k.sh
#   MODELS=llama2_7b ARMS=rsn DRY_RUN=1 bash ...
#   STOP_AFTER_NEURONS=1 bash ...      # 파라미터% 확인용
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODELS="${MODELS:-llama2_7b llama2_13b}"
ARMS="${ARMS:-rsn}"                       # sn | rsn | "sn rsn"
OUT_ROOT="${OUT_ROOT:-outputs/wsr_rsn_row}"
NEURON_DIR="${NEURON_DIR:-sn_tune/output_neurons_warp_row}"
BASIS_ROOT="${BASIS_ROOT:-outputs/wsr_sn_tune}"   # Phase1 산출물 위치 (공유)
OVERLAY="${OVERLAY:-${REPO_ROOT}/.transformers_patched}"
TRAIN_ENV="${TRAIN_ENV:-hb}"

SAFETY_JSON="${SAFETY_JSON:-data/circuit_breakers_train.json}"
UTILITY_JSON="${UTILITY_JSON:-sn_tune/corpus/wikipedia_utility_1000.json}"
SAFETY_PROMPTS="${SAFETY_PROMPTS:-4994}"
UTILITY_DOCS="${UTILITY_DOCS:-1000}"

# 회전 공간에서도 원공간과 같은 top-k 를 쓴다 (사용자 결정 2026-09-22).
SAFETY_TOP_FFN_llama2_7b="${SAFETY_TOP_FFN_llama2_7b:-1200}"
SAFETY_TOP_ATTN_llama2_7b="${SAFETY_TOP_ATTN_llama2_7b:-200}"
SAFETY_TOP_FFN_llama2_13b="${SAFETY_TOP_FFN_llama2_13b:-1800}"
SAFETY_TOP_ATTN_llama2_13b="${SAFETY_TOP_ATTN_llama2_13b:-300}"
UTILITY_TOP_FFN="${UTILITY_TOP_FFN:-300}"
UTILITY_TOP_ATTN="${UTILITY_TOP_ATTN:-50}"

LAYER_TYPES="${LAYER_TYPES:-ffn_up,ffn_down,attn_q,attn_k,attn_v}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

LR="${LR:-5e-5}"; EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LEN="${MAX_LEN:-1024}"; WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"; GSM8K_SAMPLES="${GSM8K_SAMPLES:-7473}"

DRY_RUN="${DRY_RUN:-0}"; STOP_AFTER_NEURONS="${STOP_AFTER_NEURONS:-0}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"; HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

START_llama2_7b="${START_llama2_7b:-meta-llama/Llama-2-7b-chat-hf}"
START_llama2_13b="${START_llama2_13b:-meta-llama/Llama-2-13b-chat-hf}"

# huggingface.co 가 막혀 있다. 필요한 것은 전부 캐시에 있다.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

command -v conda >/dev/null 2>&1 || { echo "[ERROR] conda 없음" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
PREFIX="$(conda env list | awk -v n="$TRAIN_ENV" '$1==n {print $NF}')"
if [[ -n "$PREFIX" ]]; then PY="${PREFIX}/bin/python"
elif [[ "$DRY_RUN" == "1" ]]; then PY="<${TRAIN_ENV}>/bin/python"
else echo "[ERROR] 환경 '${TRAIN_ENV}' 없음" >&2; exit 1; fi

if [[ ! -d "$OVERLAY" && "$DRY_RUN" != "1" ]]; then
    echo "[ERROR] 패치 오버레이가 없다: ${OVERLAY}" >&2
    echo "        bash sn_tune/setup_patched_transformers.sh" >&2
    exit 1
fi

var_of()  { local k="$1"; echo "${!k:-}"; }
topk_of() { local k="$1_$2"; echo "${!k:-}"; }
newest_subdir() {
    find "$1" -maxdepth 1 -name "$2_*" -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}
run() {
    local marker="$1"; shift; local desc="$1"; shift; [[ "$1" == "--" ]] && shift
    if [[ -f "$marker" ]]; then echo "[SKIP] ${desc}"; return 0; fi
    echo; echo "─────────────────────────────────────────────────────────────"
    echo "[RUN ] ${desc}"; printf '      '; printf ' %q' "$@"; echo
    echo "─────────────────────────────────────────────────────────────"
    if [[ "$DRY_RUN" == "1" ]]; then echo "[DRY ] 실행하지 않음"; return 0; fi
    mkdir -p "$(dirname "$marker")"
    "$@"
    date -Iseconds > "$marker"
}

mkdir -p "$NEURON_DIR" "$OUT_ROOT"

for model in $MODELS; do
    START="$(var_of "START_${model}")"
    S_FFN="$(topk_of SAFETY_TOP_FFN "$model")"
    S_ATTN="$(topk_of SAFETY_TOP_ATTN "$model")"
    [[ -n "$START" && -n "$S_FFN" && -n "$S_ATTN" ]] || {
        echo "[ERROR] '${model}' 설정 누락 (START_/SAFETY_TOP_*_${model})" >&2; exit 1; }

    MDIR="${OUT_ROOT}/${model}"; NDIR="${NEURON_DIR}/${model}"
    mkdir -p "$MDIR" "$NDIR"
    SAFETY_TXT="${NDIR}/rot_safety_neurons.txt"
    UTILITY_TXT="${NDIR}/rot_utility_neurons.txt"
    CRITICAL_TXT="${NDIR}/rot_critical_neurons.txt"

    echo; echo "==============================================================="
    echo " ${model}   출발: ${START}"
    echo "   safety top-k ${S_FFN}/${S_ATTN}   utility top-k ${UTILITY_TOP_FFN}/${UTILITY_TOP_ATTN}"
    echo "==============================================================="

    # ── 1. Phase 1 basis (공유) ────────────────────────────────────────────
    BASIS_PTR="${BASIS_ROOT}/${model}/BASIS_DIR"
    PRESET="$(var_of "BASIS_DIR_${model}")"
    if [[ -n "$PRESET" ]]; then
        BASIS_DIR="$PRESET"
    elif [[ -s "$BASIS_PTR" ]]; then
        BASIS_DIR="$(cat "$BASIS_PTR")"
        echo "[INFO] 기존 basis 재사용: ${BASIS_DIR}"
    else
        BASIS_OUT="${BASIS_ROOT}/${model}/phase1"
        mkdir -p "$BASIS_OUT"
        run "${MDIR}/.basis.done" "[1] Phase 1 basis" -- \
            "$PY" train.py --phase 1 --phase0_model_dir "$START" \
                --safety_dataset circuit_breakers --circuit_breakers_path "$SAFETY_JSON" \
                --circuit_breakers_samples_phase1 "$SAFETY_PROMPTS" \
                --basis_save_dtype bfloat16 --batch_size 2 --max_length "$MAX_LEN" \
                --layer_type "$LAYER_TYPES" --target_layers all \
                --output_dir "$BASIS_OUT" --log_dir "${BASIS_OUT}/logs" \
                --device cuda --dtype bfloat16 --seed 42 --no_wandb
        if [[ "$DRY_RUN" != "1" ]]; then
            p1="$(newest_subdir "$BASIS_OUT" phase1)"
            [[ -n "$p1" && -d "$p1/basis" ]] || { echo "[ERROR] basis 없음" >&2; exit 1; }
            mkdir -p "$(dirname "$BASIS_PTR")"; echo "$p1/basis" > "$BASIS_PTR"
            BASIS_DIR="$p1/basis"
        else BASIS_DIR="<basis>"; fi
    fi

    # ── 2·3. 회전 검출 (패치 오버레이 필요) ────────────────────────────────
    run "${NDIR}/.safety.done" "[2] 회전 safety 검출 (패치 오버레이)" -- \
        env PYTHONPATH="$OVERLAY" "$PY" -m sn_tune.detect_warp_rotated "$SAFETY_PROMPTS" \
            --model_name "$START" --dataset_file "$SAFETY_JSON" \
            --top_number_ffn "$S_FFN" --top_number_attn "$S_ATTN" \
            --use_basis_rotation_score --basis_dir "$BASIS_DIR" \
            --attn_implementation "$ATTN_IMPL" \
            --safety_neuron --output_file "$SAFETY_TXT"

    run "${NDIR}/.utility.done" "[3] 회전 utility 검출 (패치 오버레이)" -- \
        env PYTHONPATH="$OVERLAY" "$PY" -m sn_tune.detect_warp_rotated "$UTILITY_DOCS" \
            --model_name "$START" \
            --top_number_ffn "$UTILITY_TOP_FFN" --top_number_attn "$UTILITY_TOP_ATTN" \
            --use_basis_rotation_score --basis_dir "$BASIS_DIR" \
            --attn_implementation "$ATTN_IMPL" \
            --utility_json "$UTILITY_JSON" \
            --utility_neuron --output_file "$UTILITY_TXT"

    # ── 4. critical ───────────────────────────────────────────────────────
    run "${NDIR}/.critical.done" "[4] critical = safety \\ utility" -- \
        "$PY" -m sn_tune.critical_neurons \
            --safety_file "$SAFETY_TXT" --utility_file "$UTILITY_TXT" \
            --output_file "$CRITICAL_TXT" --model_name "$START" --space warp

    [[ "$STOP_AFTER_NEURONS" == "1" ]] && { echo "[STOP] STOP_AFTER_NEURONS=1"; continue; }

    # ── 5·6. arm 별 (학습은 정품 transformers — PYTHONPATH 주지 않는다) ────
    for arm in $ARMS; do
        case "$arm" in
            sn)  NEURON_TXT="$SAFETY_TXT" ;;
            rsn) NEURON_TXT="$CRITICAL_TXT" ;;
            *) echo "[ERROR] 알 수 없는 arm: ${arm}" >&2; exit 1 ;;
        esac
        ADIR="${MDIR}/${arm}"; TUNED="${ADIR}/tuned"; FT="${ADIR}/gsm8k"
        mkdir -p "$ADIR"

        run "${ADIR}/.tune.done" "[5/${arm}] ${arm^^}-Tune (정품 transformers)" -- \
            "$PY" -m sn_tune.sn_tune_original \
                --neuron_file "$NEURON_TXT" --dataset_file "$SAFETY_JSON" \
                --model_name "$START" --local_model_name "$TUNED" \
                --learning_rate "$LR" --num_epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" \
                --max_seq_length "$MAX_LEN" --warmup_ratio "$WARMUP_RATIO" \
                --no_timestamp_suffix --model_dir_file "${ADIR}/TUNED_MODEL_DIR"

        UP=(); [[ "$PUSH_TO_HUB" == "1" ]] && \
            UP=(--upload_name "${HF_NAMESPACE}/${model}-wsr_row_${arm}_gsm8k_lr${LR}")
        run "${ADIR}/.gsm8k.done" "[6/${arm}] GSM8K FT + ${arm^^} 동결" -- \
            "$PY" -m sn_tune.finetune_freeze_sn \
                --model_path "$TUNED" --safety_neurons_file "$NEURON_TXT" \
                --output_dir "$FT" --num_train_samples "$GSM8K_SAMPLES" \
                --learning_rate "$LR" --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
                --max_length "$MAX_LEN" --weight_decay "$WEIGHT_DECAY" \
                --warmup_ratio "$WARMUP_RATIO" \
                --no_timestamp_suffix --model_dir_file "${ADIR}/MODEL_DIR" "${UP[@]}"
    done
done
echo; echo "[DONE] 뉴런: ${NEURON_DIR}/<model>/   모델: ${OUT_ROOT}/<model>/<arm>/gsm8k"
