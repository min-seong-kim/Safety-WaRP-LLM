#!/usr/bin/env bash
# =============================================================================
# run_wsr_sn_rsn_gsm8k.sh — WSR-SN-Tune / WSR-RSN-Tune 전체 파이프라인 (GSM8K)
#
# 원공간 파이프라인(run_sn_rsn_gsm8k.sh)과 **단계가 1:1 로 대응**하되, 모든 검출과
# 동결이 WaRP 재매개변수화 공간에서 일어난다. 논문 Table 3 의 "X + WSR-Tune" 행이
# 말하는 바 — 기존 방법의 보호 메커니즘을 재매개변수화된 공간 안에서 적용한다 — 를
# (R)SN-Tune 에 대해 구현한 것이다.
#
#   1  Phase 1 basis      U   (train.py --phase 1;  BASIS_DIR 를 주면 건너뜀)
#   2  WaRP safety 검출        circuit_breakers → basis_coeff 의 safety '열'
#   3  WaRP utility 검출       wikipedia        → foundation '열'
#   4  critical = 2 - 3
#   5  WSR-(R)SN-Tune          그 열들만 safety 데이터로 학습
#   6  GSM8K FT (열 동결)      나머지 열만 학습 → W = basis_coeff @ Uᵀ 로 복원 저장
#
# 원공간 파이프라인과 달리 **패치된 transformers 가 필요 없다.** 여기 검출기는
# 자기 forward hook 으로 점수를 직접 계산하므로 정품 `hb` 하나로 전부 돌아간다.
#
# ⚠️ 원공간 뉴런(행/출력뉴런)과 WaRP 뉴런(열/basis 방향)은 **다른 대상**이다.
#    critical 을 계산할 때 두 공간의 파일을 섞지 마라 — 인덱스 의미가 달라서
#    숫자만 줄어들 뿐 아무 뜻이 없다. 이 스크립트는 WaRP 것끼리만 뺀다.
#
# 사용
#   bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh
#   MODELS="llama2_7b" ARMS="rsn" bash sn_tune/scripts/run_wsr_sn_rsn_gsm8k.sh
#   BASIS_DIR_llama2_7b=./checkpoints/phase1_XXX/basis bash ...   # Phase1 재사용
#   DRY_RUN=1 bash ...
#
# 재실행 안전: `.done` 마커로 건너뛴다. 하이퍼파라미터를 바꿨으면 마커를 지우지 말고
# OUT_ROOT 를 바꿔라 (경로에 하이퍼파라미터가 안 들어 있다).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODELS="${MODELS:-llama2_7b llama2_13b}"
ARMS="${ARMS:-sn rsn}"
OUT_ROOT="${OUT_ROOT:-outputs/wsr_sn_tune}"
NEURON_DIR="${NEURON_DIR:-sn_tune/output_neurons_warp}"
TRAIN_ENV="${TRAIN_ENV:-hb}"

SAFETY_JSON="${SAFETY_JSON:-data/circuit_breakers_train.json}"
UTILITY_JSON="${UTILITY_JSON:-sn_tune/corpus/wikipedia_utility_1000.json}"
SAFETY_PROMPTS="${SAFETY_PROMPTS:-4994}"
UTILITY_DOCS="${UTILITY_DOCS:-1000}"

# ── Phase 1/WaRP 불변식 ────────────────────────────────────────────────────
# layer_type 은 Phase 1 과 검출/학습에서 **완전히 동일**해야 한다. 어긋나면 조용히
# 틀린 결과가 나온다. 그래서 변수 하나로만 쓴다.
LAYER_TYPES="${LAYER_TYPES:-ffn_up,ffn_down,attn_q,attn_k,attn_v}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"
BASIS_SAMPLES="${BASIS_SAMPLES:-4994}"
BASIS_BATCH="${BASIS_BATCH:-2}"
BASIS_SAVE_DTYPE="${BASIS_SAVE_DTYPE:-bfloat16}"

# WaRP 공간 검출 top-k (열 기준).
#
# **safety 는 원공간과 같은 값을 쓴다** (사용자 결정 2026-09-22): 7B 1200/200,
# 13B 1800/300. 과거 WaRP-SN-Tune 과 맞추기 위해서다.
#   ⚠️ 다만 이 값이 과거 실행에서 *복원된* 것은 아니다 — 살아남은
#   warp_sn_tune_config.json 에는 top_k 가 없고, run_warp_sn_pipeline.py 자체
#   기본값은 2000/350("targets ~1% safety params")이었다. 검출 후 파라미터%를
#   반드시 확인하고 기록할 것.
#
# utility 는 원공간과 동일하게 300/50 (모델 크기와 무관한 집안 표준).
SAFETY_TOP_FFN_llama2_7b="${SAFETY_TOP_FFN_llama2_7b:-1200}"
SAFETY_TOP_ATTN_llama2_7b="${SAFETY_TOP_ATTN_llama2_7b:-200}"
SAFETY_TOP_FFN_llama2_13b="${SAFETY_TOP_FFN_llama2_13b:-1800}"
SAFETY_TOP_ATTN_llama2_13b="${SAFETY_TOP_ATTN_llama2_13b:-300}"

UTILITY_TOP_FFN="${UTILITY_TOP_FFN:-300}"
UTILITY_TOP_ATTN="${UTILITY_TOP_ATTN:-50}"

# 열 검출의 선택 기준: 프롬프트 중 몇 %의 top-k 에 들어야 선택되는가.
# 1.0 = 정확한 교집합 (원공간 검출과 같은 기준).
FREQ_THRESHOLD="${FREQ_THRESHOLD:-1.0}"

LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LEN="${MAX_LEN:-1024}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-7473}"
# 태스크 데이터는 저장소에 고정된 JSON 을 쓴다. HF `openai/gsm8k` 와 byte-identical 임을
# 확인했고(question 7473/7473, response==answer 7473/7473), 오프라인에서도 안전하다.
TASK_DATA_PATH="${TASK_DATA_PATH:-data/gsm8k_train_task_7473.json}"
# 13B 는 WaRP 재파라미터화 때문에 GPU 가 빠듯하다. 쓰이지 않는 원본 weight 버퍼를
# CPU 로 내려 16.4GiB 를 비운다 (models/warp_modules.offload_weight_buffers).
OFFLOAD_WEIGHT_BUFFER="${OFFLOAD_WEIGHT_BUFFER:-1}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
DRY_RUN="${DRY_RUN:-0}"
STOP_AFTER_NEURONS="${STOP_AFTER_NEURONS:-0}"

# 출발 모델 — run_sn_rsn_gsm8k.sh 와 **반드시 동일**해야 두 arm 이 비교 가능하다.
# plain chat 모델이다 (SSFT 아님). 이유는 원공간 드라이버의 같은 블록 주석 참고.
START_llama2_7b="${START_llama2_7b:-meta-llama/Llama-2-7b-chat-hf}"
START_llama2_13b="${START_llama2_13b:-meta-llama/Llama-2-13b-chat-hf}"

# 이미 만들어 둔 Phase 1 basis 가 있으면 지정 (그러면 1단계를 건너뛴다)
BASIS_DIR_llama2_7b="${BASIS_DIR_llama2_7b:-}"
BASIS_DIR_llama2_13b="${BASIS_DIR_llama2_13b:-}"

# ---------------------------------------------------------------------------
# conda
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda 가 PATH 에 없다." >&2; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
TRAIN_PREFIX="$(conda env list | awk -v n="$TRAIN_ENV" '$1==n {print $NF}')"
if [[ -n "$TRAIN_PREFIX" ]]; then
    PY="${TRAIN_PREFIX}/bin/python"
elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[WARN] 환경 '${TRAIN_ENV}' 이 없다 — DRY_RUN 이라 자리표시자로 계속한다."
    PY="<${TRAIN_ENV}>/bin/python"
else
    echo "[ERROR] 환경 '${TRAIN_ENV}' 이 없다." >&2; exit 1
fi

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
run() {   # run <마커> <설명> -- <명령...>
    local marker="$1"; shift
    local desc="$1"; shift
    [[ "$1" == "--" ]] && shift
    if [[ -f "$marker" ]]; then
        echo "[SKIP] ${desc}  (마커: ${marker})"; return 0
    fi
    echo
    echo "───────────────────────────────────────────────────────────────"
    echo "[RUN ] ${desc}"
    printf '      '; printf ' %q' "$@"; echo
    echo "───────────────────────────────────────────────────────────────"
    if [[ "$DRY_RUN" == "1" ]]; then echo "[DRY ] 실행하지 않음"; return 0; fi
    mkdir -p "$(dirname "$marker")"
    "$@"
    date -Iseconds > "$marker"
}

newest_subdir() {  # <base> <prefix>
    find "$1" -maxdepth 1 -name "$2_*" -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

var_of() { local k="$1"; echo "${!k:-}"; }
topk_of() { local k="$1_$2"; echo "${!k:-}"; }

mkdir -p "$NEURON_DIR" "$OUT_ROOT"

# ---------------------------------------------------------------------------
# 0. utility 코퍼스 (WaRP 검출기는 JSON 만 읽으므로 미리 덤프해 둬야 한다)
# ---------------------------------------------------------------------------
if [[ ! -f "$UTILITY_JSON" ]]; then
    run "${NEURON_DIR}/.utility_corpus.done" \
        "[0] utility 코퍼스 덤프 (wikipedia → ${UTILITY_JSON})" -- \
        "$PY" -m sn_tune.build_utility_corpus \
            --num_samples "$UTILITY_DOCS" --output_file "$UTILITY_JSON"
else
    echo "[SKIP] utility 코퍼스가 이미 있다: ${UTILITY_JSON}"
fi

# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
for model in $MODELS; do
    START="$(var_of "START_${model}")"
    [[ -n "$START" ]] || { echo "[ERROR] START_${model} 이 없다." >&2; exit 1; }

    S_FFN="$(topk_of SAFETY_TOP_FFN "$model")"
    S_ATTN="$(topk_of SAFETY_TOP_ATTN "$model")"
    if [[ -z "$S_FFN" || -z "$S_ATTN" ]]; then
        echo "[ERROR] '${model}' 의 safety top-k 가 없다" >&2
        echo "        (SAFETY_TOP_FFN_${model} / SAFETY_TOP_ATTN_${model} 을 정하라)" >&2
        exit 1
    fi

    MDIR="${OUT_ROOT}/${model}"
    NDIR="${NEURON_DIR}/${model}"
    mkdir -p "$MDIR" "$NDIR"

    SAFETY_TXT="${NDIR}/warp_safety_neurons.txt"
    UTILITY_TXT="${NDIR}/warp_utility_neurons.txt"
    CRITICAL_TXT="${NDIR}/warp_critical_neurons.txt"
    BASIS_PTR="${MDIR}/BASIS_DIR"

    echo
    echo "==============================================================="
    echo " ${model}   출발 모델: ${START}"
    echo "   safety top-k ${S_FFN}/${S_ATTN}   utility top-k ${UTILITY_TOP_FFN}/${UTILITY_TOP_ATTN}"
    echo "==============================================================="

    # ── 1. Phase 1 basis ───────────────────────────────────────────────────
    PRESET_BASIS="$(var_of "BASIS_DIR_${model}")"
    if [[ -n "$PRESET_BASIS" ]]; then
        echo "[SKIP] Phase 1 — 지정된 basis 사용: ${PRESET_BASIS}"
        echo "$PRESET_BASIS" > "$BASIS_PTR"
    else
        BASIS_OUT="${MDIR}/phase1"
        if [[ ! -f "${MDIR}/.basis.done" ]]; then
            mkdir -p "$BASIS_OUT"
            run "${MDIR}/.basis.done" "[1] Phase 1 basis (train.py)" -- \
                "$PY" train.py \
                    --phase 1 \
                    --phase0_model_dir "$START" \
                    --safety_dataset circuit_breakers \
                    --circuit_breakers_path "$SAFETY_JSON" \
                    --circuit_breakers_samples_phase1 "$BASIS_SAMPLES" \
                    --basis_save_dtype "$BASIS_SAVE_DTYPE" \
                    --batch_size "$BASIS_BATCH" \
                    --max_length "$MAX_LEN" \
                    --layer_type "$LAYER_TYPES" \
                    --target_layers "$TARGET_LAYERS" \
                    --output_dir "$BASIS_OUT" \
                    --log_dir "${BASIS_OUT}/logs" \
                    --device cuda --dtype "$DTYPE" --seed "$SEED" --no_wandb
            if [[ "$DRY_RUN" != "1" ]]; then
                p1="$(newest_subdir "$BASIS_OUT" phase1)"
                [[ -n "$p1" && -d "$p1/basis" ]] || {
                    echo "[ERROR] Phase 1 결과 basis 를 찾지 못했다: ${BASIS_OUT}" >&2; exit 1; }
                echo "$p1/basis" > "$BASIS_PTR"
                echo "[INFO] basis → $(cat "$BASIS_PTR")"
            fi
        else
            echo "[SKIP] Phase 1 (마커 있음)"
        fi
    fi
    BASIS_DIR=""
    [[ -s "$BASIS_PTR" ]] && BASIS_DIR="$(cat "$BASIS_PTR")"
    if [[ "$DRY_RUN" != "1" && ( -z "$BASIS_DIR" || ! -d "$BASIS_DIR" ) ]]; then
        echo "[ERROR] basis 디렉토리가 없다: '${BASIS_DIR}'" >&2; exit 1
    fi
    [[ -n "$BASIS_DIR" ]] || BASIS_DIR="<basis>"   # DRY_RUN 표시용

    # ── 2. WaRP safety 검출 ────────────────────────────────────────────────
    run "${NDIR}/.safety.done" "[2] WaRP safety 뉴런 검출" -- \
        "$PY" sn_tune/run_warp_sn_pipeline.py \
            --model_name "$START" \
            --basis_dir "$BASIS_DIR" \
            --dataset_file "$SAFETY_JSON" \
            --output_dir "${MDIR}/detect_safety" \
            --neuron_output_file "$SAFETY_TXT" \
            --layer_types "$LAYER_TYPES" \
            --num_prompts "$SAFETY_PROMPTS" \
            --top_k_ffn "$S_FFN" --top_k_attn "$S_ATTN" \
            --freq_threshold "$FREQ_THRESHOLD" \
            --max_seq_len "$MAX_LEN" \
            --gpu "$GPU" --dtype "$DTYPE" \
            --detection_only

    # ── 3. WaRP utility 검출 ───────────────────────────────────────────────
    run "${NDIR}/.utility.done" "[3] WaRP utility 뉴런 검출" -- \
        "$PY" sn_tune/run_warp_sn_pipeline.py \
            --model_name "$START" \
            --basis_dir "$BASIS_DIR" \
            --dataset_file "$UTILITY_JSON" \
            --output_dir "${MDIR}/detect_utility" \
            --neuron_output_file "$UTILITY_TXT" \
            --layer_types "$LAYER_TYPES" \
            --num_prompts "$UTILITY_DOCS" \
            --top_k_ffn "$UTILITY_TOP_FFN" --top_k_attn "$UTILITY_TOP_ATTN" \
            --freq_threshold "$FREQ_THRESHOLD" \
            --max_seq_len "$MAX_LEN" \
            --gpu "$GPU" --dtype "$DTYPE" \
            --detection_only

    # ── 4. critical (WaRP 공간끼리만) ──────────────────────────────────────
    run "${NDIR}/.critical.done" "[4] WaRP critical = safety \\ utility" -- \
        "$PY" -m sn_tune.critical_neurons \
            --safety_file "$SAFETY_TXT" \
            --utility_file "$UTILITY_TXT" \
            --output_file "$CRITICAL_TXT" \
            --model_name "$START" \
            --space warp

    if [[ "$STOP_AFTER_NEURONS" == "1" ]]; then
        echo "[STOP] STOP_AFTER_NEURONS=1"
        continue
    fi

    # ── 5 & 6. arm 별 ──────────────────────────────────────────────────────
    for arm in $ARMS; do
        case "$arm" in
            sn)  NEURON_TXT="$SAFETY_TXT" ;;
            rsn) NEURON_TXT="$CRITICAL_TXT" ;;
            *)   echo "[ERROR] 알 수 없는 arm: ${arm}" >&2; exit 1 ;;
        esac

        ADIR="${MDIR}/${arm}"
        TUNE_OUT="${ADIR}/tuned"
        FT_DIR="${ADIR}/gsm8k"
        mkdir -p "$ADIR"

        # 5. WSR-(R)SN-Tune — 선택된 열만 safety 데이터로 학습
        run "${ADIR}/.tune.done" "[5/${arm}] WSR-${arm^^}-Tune" -- \
            "$PY" sn_tune/run_warp_sn_pipeline.py \
                --model_name "$START" \
                --basis_dir "$BASIS_DIR" \
                --dataset_file "$SAFETY_JSON" \
                --output_dir "$TUNE_OUT" \
                --existing_neuron_file "$NEURON_TXT" \
                --layer_types "$LAYER_TYPES" \
                --learning_rate "$LR" \
                --num_epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --grad_accum_steps "$GRAD_ACCUM" \
                --max_seq_len "$MAX_LEN" \
                --max_samples "$SAFETY_PROMPTS" \
                --warmup_ratio "$WARMUP_RATIO" \
                --gpu "$GPU" --dtype "$DTYPE" --seed "$SEED"

        # run_warp_sn_pipeline 은 warp_sn_tuned_lr<lr>_<ts>/ 로 저장한다 — 최신 것을 집는다.
        TUNED_DIR="<tuned>"
        if [[ "$DRY_RUN" != "1" ]]; then
            TUNED_DIR="$(newest_subdir "$TUNE_OUT" warp_sn_tuned)"
            [[ -n "$TUNED_DIR" ]] || { echo "[ERROR] WSR-SN-Tune 결과를 못 찾았다: ${TUNE_OUT}" >&2; exit 1; }
            echo "$TUNED_DIR" > "${ADIR}/TUNED_MODEL_DIR"
            echo "[INFO] tuned → ${TUNED_DIR}"
        fi

        # 6. GSM8K FT — 같은 열을 이번엔 **얼린다**
        UPLOAD_ARGS=()
        if [[ "$PUSH_TO_HUB" == "1" ]]; then
            UPLOAD_ARGS=(--upload_name "${HF_NAMESPACE}/${model}-wsr_${arm}_gsm8k_lr${LR}")
        fi
        run "${ADIR}/.gsm8k.done" "[6/${arm}] GSM8K FT + WaRP 열 동결" -- \
            "$PY" sn_tune/finetune_downstream_freeze_warp_sn.py \
                --model_name_or_path "$TUNED_DIR" \
                --basis_dir "$BASIS_DIR" \
                --neuron_file "$NEURON_TXT" \
                --layer_types "$LAYER_TYPES" \
                --task gsm8k \
                --num_train_samples "$GSM8K_SAMPLES" \
                --output_dir "$FT_DIR" \
                --learning_rate "$LR" \
                --num_epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --grad_accum "$GRAD_ACCUM" \
                --weight_decay "$WEIGHT_DECAY" \
                --warmup_ratio "$WARMUP_RATIO" \
                --max_length "$MAX_LEN" \
                --seed "$SEED" --gpu "$GPU" --dtype "$DTYPE" \
                --task_data_path "$TASK_DATA_PATH" \
                ${OFFLOAD_WEIGHT_BUFFER:+--offload_weight_buffer} \
                "${UPLOAD_ARGS[@]}"
    done
done

echo
echo "[DONE] WSR-(R)SN-Tune 파이프라인 완료."
echo "       뉴런 파일: ${NEURON_DIR}/<model>/"
echo "       모델:      ${OUT_ROOT}/<model>/<arm>/gsm8k"
