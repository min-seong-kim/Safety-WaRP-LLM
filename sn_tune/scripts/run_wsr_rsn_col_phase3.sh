#!/usr/bin/env bash
# =============================================================================
# run_wsr_rsn_col_phase3.sh — WSR-(R)SN-Tune **열(column) 버전**, Phase 3 트레이너 재사용
#
# 논문 Table 3 "(R)SN-Tune + WSR-Tune" 행의 열 버전. 뉴런 = `basis_coeff = W @ U` 의 열
# (= 안전 활성화로 만든 입력 basis 방향). 이 정의에서만 WSR 재매개변수화가 실제로
# 다른 부분공간을 얼린다 (행 버전은 원공간 동결과 수학적으로 같다 — sn_tune/RESULTS.md §B-0).
#
#   [1] Phase 1 basis U            (outputs/wsr_sn_tune/<model>/BASIS_DIR 재사용)
#   [2] safety 열 검출             warp_sn_detection (없으면 검출, 있으면 재사용)
#   [3] utility 열 검출            wikipedia 1000 (없으면 검출)
#   [4] critical = [2] \ [3]
#   [5] 마스크 2종                 warp_col_masks.py  (tune: critical 열만 학습 / freeze: critical 열 동결)
#   [6] RSN-Tune (WaRP 공간)      train.py --phase 3 --phase3_dataset safety  + masks_tune
#   [7] GSM8K FT (WaRP 공간)      train.py --phase 3 --phase3_dataset gsm8k   + masks_freeze --non_freeze
#                                  ← 출발점은 [6] 의 모델. 공개 WSR-Tune 과 같은 트레이너·플래그.
#
# 왜 Phase 3 인가: 이전 세대 전용 트레이너(finetune_downstream_freeze_warp_sn.py)는
# 모듈마다 basis_coeff@Uᵀ 를 매 step 물질화하고 전 파라미터를 학습해 10.8 s/it · 178 GB OOM.
# Phase 3 는 같은 LinearWaRP 로 검증된 경로이고, gradient checkpointing 과 4×4 배치로 7B 가
# 30여 분에 끝난다.
#
# 사용
#   bash sn_tune/scripts/run_wsr_rsn_col_phase3.sh
#   MODELS="llama2_7b" CELLS_llama2_7b="1200:200:300:50" bash ...      # safetyK:safetyA:utilK:utilA
#   STOP_AFTER_MASKS=1 bash ...     # 검출·critical·마스크까지만 (예산 확인용)
#   DRY_RUN=1 bash ...
#
# 재실행 안전: 단계별 .done 마커. 셀 경로에 top-k 가 들어가므로 값을 바꾸면 새 셀이 된다.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODELS="${MODELS:-llama2_7b llama2_13b}"
# 셀 = safetyK:safetyA:utilityK:utilityA (공백 구분 여러 개)
CELLS_llama2_7b="${CELLS_llama2_7b:-1200:200:300:50}"
CELLS_llama2_13b="${CELLS_llama2_13b:-1800:300:300:50}"

OUT_ROOT="${OUT_ROOT:-outputs/wsr_rsn_col_p3}"
NEURON_DIR="${NEURON_DIR:-sn_tune/output_neurons_warp}"
BASIS_ROOT="${BASIS_ROOT:-outputs/wsr_sn_tune}"      # <BASIS_ROOT>/<model>/BASIS_DIR
LOG_ROOT="${LOG_ROOT:-logs/wsr_rsn_col}"
TRAIN_ENV="${TRAIN_ENV:-hb}"

SAFETY_JSON="${SAFETY_JSON:-data/circuit_breakers_train.json}"
UTILITY_JSON="${UTILITY_JSON:-sn_tune/corpus/wikipedia_utility_1000.json}"
TASK_JSON="${TASK_JSON:-data/gsm8k_train_task_7473.json}"   # revision 라인과 byte-identical 프롬프트
SAFETY_PROMPTS="${SAFETY_PROMPTS:-4994}"
UTILITY_DOCS="${UTILITY_DOCS:-1000}"
FREQ_THRESHOLD="${FREQ_THRESHOLD:-1.0}"                     # 1.0 = 정확한 교집합

LAYER_TYPES="${LAYER_TYPES:-ffn_up,ffn_down,attn_q,attn_k,attn_v}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"

# 학습 동작점 = 공개 WSR-Tune / (R)SN-Tune 과 동일
LR="${LR:-5e-5}"; EPOCHS="${EPOCHS:-3}"; MAX_LEN="${MAX_LEN:-1024}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"; WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-16}"
MB_llama2_7b="${MB_llama2_7b:-4}"       # micro-batch. 7B 4×4 (Qwen 7B 실측 109 GB)
MB_llama2_13b="${MB_llama2_13b:-2}"     # 13B 2×8
DTYPE="${DTYPE:-bfloat16}"; SEED="${SEED:-42}"

START_llama2_7b="${START_llama2_7b:-meta-llama/Llama-2-7b-chat-hf}"     # SN-Tune 관례: plain chat
START_llama2_13b="${START_llama2_13b:-meta-llama/Llama-2-13b-chat-hf}"

DRY_RUN="${DRY_RUN:-0}"
STOP_AFTER_MASKS="${STOP_AFTER_MASKS:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # huggingface.co 차단 상태에서도 캐시로 진행
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
TRAIN_PREFIX="$(conda env list | awk -v n="$TRAIN_ENV" '$1==n {print $NF}')"
if [[ -z "$TRAIN_PREFIX" ]]; then
    [[ "$DRY_RUN" == "1" ]] || { echo "[ERROR] 환경 '${TRAIN_ENV}' 이 없다." >&2; exit 1; }
    PY="<${TRAIN_ENV}>/bin/python"
else
    PY="${TRAIN_PREFIX}/bin/python"
fi
mkdir -p "$LOG_ROOT"

log() { echo "[$(date '+%F %T')] $*"; }

# run <마커> <설명> -- <명령...> : 마커가 있으면 건너뛴다.
run() {
    local marker="$1" desc="$2"; shift 2
    [[ "$1" == "--" ]] && shift
    if [[ -f "$marker" ]]; then log "[skip] $desc  (마커: $marker)"; return 0; fi
    log "[run ] $desc"
    if [[ "$DRY_RUN" == "1" ]]; then printf '      %q ' "$@"; echo; return 0; fi
    "$@"
    date -Iseconds > "$marker"
}

newest_subdir() {  # newest_subdir <dir> <prefix>
    find "$1" -maxdepth 1 -type d -name "${2}*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

# 옛 이름의 검출 파일을 새 규약 이름으로 연결한다 (재검출 방지).
#   warp_safety_neurons.txt  = 7B 1200/200, 13B 1800/300   (2026-09-22 검출)
#   warp_utility_neurons.txt = 300/50
#   _calib/safety_k{K}_{A}.txt = 7B 보정 검출
link_legacy() {  # link_legacy <model> <kind:safety|utility> <K> <A> <target>
    local model="$1" kind="$2" K="$3" A="$4" target="$5"
    [[ -f "$target" ]] && return 0
    local cands=()
    if [[ "$kind" == "safety" ]]; then
        cands+=("$NEURON_DIR/_calib/safety_k${K}_${A}.txt")
        [[ "$model" == "llama2_7b"  && "$K:$A" == "1200:200" ]] && cands+=("$NEURON_DIR/$model/warp_safety_neurons.txt")
        [[ "$model" == "llama2_13b" && "$K:$A" == "1800:300" ]] && cands+=("$NEURON_DIR/$model/warp_safety_neurons.txt")
    else
        [[ "$K:$A" == "300:50" ]] && cands+=("$NEURON_DIR/$model/warp_utility_neurons.txt")
    fi
    local c
    for c in "${cands[@]}"; do
        if [[ -f "$c" ]]; then
            [[ "$DRY_RUN" == "1" ]] || cp "$c" "$target"
            log "[reuse] $kind ${K}/${A} ← $c"
            return 0
        fi
    done
    return 1
}

detect() {  # detect <model> <start> <basis> <dataset_json> <num_prompts> <K> <A> <out_txt> <work_dir>
    local model="$1" start="$2" basis="$3" data="$4" n="$5" K="$6" A="$7" out="$8" work="$9"
    "$PY" sn_tune/run_warp_sn_pipeline.py \
        --model_name "$start" --basis_dir "$basis" \
        --dataset_file "$data" --output_dir "$work" --neuron_output_file "$out" \
        --layer_types "$LAYER_TYPES" --num_prompts "$n" \
        --top_k_ffn "$K" --top_k_attn "$A" --freq_threshold "$FREQ_THRESHOLD" \
        --max_seq_len "$MAX_LEN" --gpu 0 --dtype "$DTYPE" --detection_only
}

# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
for model in $MODELS; do
    start_var="START_${model}"; START="${!start_var}"
    mb_var="MB_${model}";        MB="${!mb_var}"
    cells_var="CELLS_${model}";  CELLS="${!cells_var}"
    ACC=$(( EFFECTIVE_BATCH / MB ))
    (( MB * ACC == EFFECTIVE_BATCH )) || { echo "[ERROR] micro-batch $MB 가 $EFFECTIVE_BATCH 의 약수가 아니다" >&2; exit 1; }

    BASIS_PTR="$BASIS_ROOT/$model/BASIS_DIR"
    if [[ ! -f "$BASIS_PTR" ]]; then
        echo "[ERROR] $BASIS_PTR 가 없다 — Phase 1 basis 를 먼저 만들어라 (run_wsr_sn_rsn_gsm8k.sh [1])" >&2; exit 1
    fi
    BASIS_DIR="$(cat "$BASIS_PTR")"
    [[ -d "$BASIS_DIR" || "$DRY_RUN" == "1" ]] || { echo "[ERROR] basis 디렉토리 없음: $BASIS_DIR" >&2; exit 1; }

    NDIR="$NEURON_DIR/$model"; mkdir -p "$NDIR"

    for cell in $CELLS; do
        IFS=: read -r SK SA UK UA <<< "$cell"
        TAG="k${SK}_${SA}_u${UK}_${UA}"
        CELL="$OUT_ROOT/$model/$TAG"; mkdir -p "$CELL"
        log "################ $model / $TAG  (start=$START, batch=${MB}x${ACC}) ################"

        SAFETY_TXT="$NDIR/safety_k${SK}_${SA}.txt"
        UTILITY_TXT="$NDIR/utility_k${UK}_${UA}.txt"
        CRITICAL_TXT="$NDIR/critical_${TAG}.txt"

        # [2] safety 열 검출
        if ! link_legacy "$model" safety "$SK" "$SA" "$SAFETY_TXT"; then
            run "$SAFETY_TXT.done" "[2] safety 열 검출 ${SK}/${SA}" -- \
                detect "$model" "$START" "$BASIS_DIR" "$SAFETY_JSON" "$SAFETY_PROMPTS" "$SK" "$SA" "$SAFETY_TXT" "$CELL/detect_safety"
        fi
        # [3] utility 열 검출
        if ! link_legacy "$model" utility "$UK" "$UA" "$UTILITY_TXT"; then
            [[ -f "$UTILITY_JSON" || "$DRY_RUN" == "1" ]] || { echo "[ERROR] $UTILITY_JSON 없음 — sn_tune/build_utility_corpus.py" >&2; exit 1; }
            run "$UTILITY_TXT.done" "[3] utility 열 검출 ${UK}/${UA}" -- \
                detect "$model" "$START" "$BASIS_DIR" "$UTILITY_JSON" "$UTILITY_DOCS" "$UK" "$UA" "$UTILITY_TXT" "$CELL/detect_utility"
        fi
        # [4] critical
        run "$CRITICAL_TXT.done" "[4] critical = safety \\ utility" -- \
            "$PY" -m sn_tune.critical_neurons --safety_file "$SAFETY_TXT" --utility_file "$UTILITY_TXT" \
                --output_file "$CRITICAL_TXT" --model_name "$START" --space warp

        # [5] 마스크 2종
        run "$CELL/masks_tune/.done" "[5a] tune 마스크 (critical 열만 학습)" -- \
            "$PY" -m sn_tune.warp_col_masks --neuron_file "$CRITICAL_TXT" --model_name "$START" \
                --mode tune --output_dir "$CELL/masks_tune" --layer_types "$LAYER_TYPES"
        run "$CELL/masks_freeze/.done" "[5b] freeze 마스크 (critical 열 동결)" -- \
            "$PY" -m sn_tune.warp_col_masks --neuron_file "$CRITICAL_TXT" --model_name "$START" \
                --mode freeze --output_dir "$CELL/masks_freeze" --layer_types "$LAYER_TYPES"

        if [[ "$STOP_AFTER_MASKS" == "1" ]]; then log "[stop] STOP_AFTER_MASKS=1"; continue; fi

        # [6] RSN-Tune in WaRP space: critical 열만 safety 데이터로 학습
        #     Phase 3 기본(freeze) 변형: basis_coeff 만 학습 가능, weight decay 는 0 으로 강제됨
        #     → mask=1 원소는 grad 도 decay 도 없어 정확히 고정된다.
        # 경로에 출발 모델 이름을 넣는다: Phase 3 는 --phase0_model_dir 문자열에서 chat 여부를 판정하므로
        # [7] 이 받는 로컬 경로에 "chat" 이 없으면 chat template 없이 학습된다 (CLAUDE.md 함정).
        TUNE_OUT="$CELL/rsn_tune_$(basename "$START")"; mkdir -p "$TUNE_OUT"
        if [[ ! -f "$TUNE_OUT/.done" ]]; then
            run "$TUNE_OUT/.done" "[6] WSR-RSN-Tune (safety ${SAFETY_PROMPTS}, ${EPOCHS}ep)" -- \
                "$PY" train.py --phase 3 \
                    --phase0_model_dir "$START" --basis_dir "$BASIS_DIR" --masks_dir "$CELL/masks_tune" \
                    --phase3_dataset safety --circuit_breakers_path "$SAFETY_JSON" \
                    --circuit_breakers_samples_phase3 "$SAFETY_PROMPTS" \
                    --epochs "$EPOCHS" --utility_lr "$LR" --base_weight_decay "$WEIGHT_DECAY" \
                    --warmup_ratio "$WARMUP_RATIO" --lr_scheduler_type cosine --max_grad_norm 1.0 \
                    --max_length "$MAX_LEN" --batch_size "$MB" --gradient_accumulation_steps "$ACC" \
                    --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
                    --output_dir "$TUNE_OUT" --log_dir "$TUNE_OUT/logs" \
                    --device cuda --dtype "$DTYPE" --seed "$SEED" \
                    --gradient_checkpointing --no_wandb --profile_json "$TUNE_OUT/profile.json"
            if [[ "$DRY_RUN" != "1" ]]; then
                p3="$(newest_subdir "$TUNE_OUT" phase3_)"
                [[ -n "$p3" && -f "$p3/final_model/config.json" ]] || { echo "[ERROR] [6] final_model 없음: $TUNE_OUT" >&2; rm -f "$TUNE_OUT/.done"; exit 1; }
                echo "$p3/final_model" > "$CELL/TUNED_MODEL_DIR"
            fi
        fi
        [[ "$DRY_RUN" == "1" ]] && TUNED="<tuned>" || TUNED="$(cat "$CELL/TUNED_MODEL_DIR")"

        # [7] GSM8K FT: critical 열 동결, 나머지 학습 — 공개 WSR-Tune 과 같은 --non_freeze 트레이너
        #     (전 파라미터 학습 + mask=1 basis_coeff 는 detach + WaRPMaskRestoreCallback 로 고정)
        GSM_OUT="$CELL/gsm8k"; mkdir -p "$GSM_OUT"
        if [[ ! -f "$GSM_OUT/.done" ]]; then
            run "$GSM_OUT/.done" "[7] GSM8K FT (critical 열 동결, ${EPOCHS}ep)" -- \
                "$PY" train.py --phase 3 \
                    --phase0_model_dir "$TUNED" --basis_dir "$BASIS_DIR" --masks_dir "$CELL/masks_freeze" \
                    --phase3_dataset gsm8k --phase3_task_data_path "$TASK_JSON" --phase3_task_samples 0 \
                    --epochs "$EPOCHS" --utility_lr "$LR" --base_weight_decay "$WEIGHT_DECAY" \
                    --warmup_ratio "$WARMUP_RATIO" --lr_scheduler_type cosine --max_grad_norm 1.0 \
                    --max_length "$MAX_LEN" --batch_size "$MB" --gradient_accumulation_steps "$ACC" \
                    --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
                    --output_dir "$GSM_OUT" --log_dir "$GSM_OUT/logs" \
                    --device cuda --dtype "$DTYPE" --seed "$SEED" \
                    --non_freeze --gradient_checkpointing --no_wandb --profile_json "$GSM_OUT/profile.json"
            if [[ "$DRY_RUN" != "1" ]]; then
                p3="$(newest_subdir "$GSM_OUT" phase3_)"
                [[ -n "$p3" && -f "$p3/final_model/config.json" ]] || { echo "[ERROR] [7] final_model 없음: $GSM_OUT" >&2; rm -f "$GSM_OUT/.done"; exit 1; }
                echo "$p3/final_model" > "$CELL/MODEL_DIR"
                # 셀 요약 (평가·업로드·표 작성이 읽는다)
                "$PY" - "$CELL" "$model" "$START" "$TAG" "$SK" "$SA" "$UK" "$UA" "$BASIS_DIR" "$CRITICAL_TXT" "$LR" "$EPOCHS" "$MB" "$ACC" <<'EOF'
import json, sys, os
cell, model, start, tag, sk, sa, uk, ua, basis, crit, lr, ep, mb, acc = sys.argv[1:]
m = json.load(open(os.path.join(cell, "masks_freeze", "metadata.json")))
cfg = {
  "method": "WSR-RSN-Tune (column)", "model": model, "start_model": start, "cell": tag,
  "safety_topk": [int(sk), int(sa)], "utility_topk": [int(uk), int(ua)],
  "basis_dir": basis, "critical_file": crit,
  "critical_columns": m["neuron_counts"], "critical_param_fraction": m["selected_column_param_fraction_of_model"],
  "stage6_rsn_tune": {"trainer": "phase3 (freeze variant, basis_coeff only, wd forced 0)", "dataset": "circuit_breakers 4994"},
  "stage7_gsm8k": {"trainer": "phase3 --non_freeze (all params + WaRPMaskRestoreCallback)", "dataset": "gsm8k_train_task_7473.json"},
  "lr": float(lr), "epochs": int(ep), "batch": f"{mb}x{acc}", "max_len": 1024, "seed": 42,
  "tuned_model_dir": open(os.path.join(cell, "TUNED_MODEL_DIR")).read().strip(),
  "model_dir": open(os.path.join(cell, "MODEL_DIR")).read().strip(),
}
json.dump(cfg, open(os.path.join(cell, "cell_config.json"), "w"), indent=2)
print(json.dumps(cfg, indent=2))
EOF
            fi
        fi
        log "[done] $model / $TAG → $(cat "$CELL/MODEL_DIR" 2>/dev/null || echo '<dry-run>')"
    done
done
log "ALL DONE"
