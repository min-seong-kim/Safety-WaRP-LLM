#!/usr/bin/env bash
# =============================================================================
# run_wsr_gradcol.sh — WSR-(R)SN-Tune 열 버전, **열 선택을 기울기 중요도로** (2026-09-22 사용자 결정 "2번")
#
# 열 = basis_coeff 의 열. 선택 기준 = Phase 2 안전 손실 기울기 |∂L/∂W̃| 를 열 단위 L2 로 집계해
# 레이어별 상위 ρ (= ActSVD ablation arm C, `train.py --phase 2 --ablation_arm C`).
# 활성화 크기로 고르던 기존 열 버전(24.17)과 달리 "거부에 중요한" 방향을 고른다.
#
#   [P2] arm C 열 마스크 (freeze: 선택 열 mask=1)        ← MASKS_C 로 받는다 (먼저 만들어 둘 것)
#   [5]  tune 마스크 = invert(freeze)                     mask_ops.py invert
#   [6]  (R)SN-Tune: 선택 열만 safety 데이터로 학습        train.py --phase 3 safety (freeze 변형)
#   [7]  GSM8K: 선택 열 동결, 나머지 학습                   train.py --phase 3 gsm8k --non_freeze
#
# 사용
#   MASKS_C=<phase2_armC_*/checkpoints/masks> CELL=outputs/wsr_rsn_col_p3/llama2_7b/gradcol_rho0.1 \
#     MODEL=llama2_7b bash sn_tune/scripts/run_wsr_gradcol.sh
# 단계별 .done 마커, SKIP_TUNE=1 이면 [6] 생략.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

MODEL="${MODEL:-llama2_7b}"
MASKS_C="${MASKS_C:?arm C 마스크 디렉토리(MASKS_C) 필요}"
CELL="${CELL:?셀 디렉토리(CELL) 필요}"
SKIP_TUNE="${SKIP_TUNE:-0}"
case "$MODEL" in
    llama2_7b)  START="${START:-meta-llama/Llama-2-7b-chat-hf}";  MB="${MB:-4}" ;;
    llama2_13b) START="${START:-meta-llama/Llama-2-13b-chat-hf}"; MB="${MB:-2}" ;;
    *) echo "unknown MODEL $MODEL" >&2; exit 1 ;;
esac
ACC=$(( 16 / MB ))
BASIS_DIR="$(cat outputs/wsr_sn_tune/$MODEL/BASIS_DIR)"
SAFETY_JSON="${SAFETY_JSON:-data/circuit_breakers_train.json}"
TASK_JSON="${TASK_JSON:-data/gsm8k_train_task_7473.json}"
LAYER_TYPES="ffn_up,ffn_down,attn_q,attn_k,attn_v"
LR=5e-5; EPOCHS=3; MAX_LEN=1024; WD=0.01; WARMUP=0.1; DTYPE=bfloat16; SEED=42
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false

source "$(conda info --base)/etc/profile.d/conda.sh"
PY="$(conda env list | awk '$1=="hb"{print $NF}')/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }
run() { local marker="$1" desc="$2"; shift 2; [[ "$1" == "--" ]] && shift
        if [[ -f "$marker" ]]; then log "[skip] $desc"; return 0; fi; log "[run ] $desc"; "$@"; date -Iseconds > "$marker"; }
newest_subdir() { find "$1" -maxdepth 1 -type d -name "${2}*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

mkdir -p "$CELL"
[[ -f "$MASKS_C/metadata.json" ]] || { echo "[ERROR] arm C 마스크 없음: $MASKS_C" >&2; exit 1; }
log "################ $MODEL / $(basename "$CELL")  start=$START  masks_C=$MASKS_C  batch=${MB}x${ACC} ################"

# [5] 마스크: freeze = arm C 그대로(링크), tune = 반전. 기록용 열 파일.
if [[ ! -e "$CELL/masks_freeze" ]]; then ln -s "$(readlink -f "$MASKS_C")" "$CELL/masks_freeze"; fi
run "$CELL/masks_tune/.done" "[5] tune 마스크 = invert(arm C)" -- \
    "$PY" -m sn_tune.mask_ops invert --input "$CELL/masks_freeze" --output_dir "$CELL/masks_tune"
run "$CELL/columns.txt.done" "[5b] 열 인덱스 기록" -- \
    "$PY" -m sn_tune.mask_ops columns --input "$CELL/masks_freeze" --output_file "$CELL/columns.txt"
"$PY" -m sn_tune.mask_ops stats --input "$CELL/masks_freeze"

# [6]
TUNE_OUT="$CELL/rsn_tune_$(basename "$START")"; mkdir -p "$TUNE_OUT"
if [[ "$SKIP_TUNE" == "1" ]]; then
    log "[skip] [6] SKIP_TUNE=1"; echo "$START" > "$CELL/TUNED_MODEL_DIR"
elif [[ ! -f "$TUNE_OUT/.done" ]]; then
    run "$TUNE_OUT/.done" "[6] WSR-SN-Tune (선택 열만, safety 4994, ${EPOCHS}ep)" -- \
        "$PY" train.py --phase 3 --phase0_model_dir "$START" --basis_dir "$BASIS_DIR" --masks_dir "$CELL/masks_tune" \
            --phase3_dataset safety --circuit_breakers_path "$SAFETY_JSON" --circuit_breakers_samples_phase3 4994 \
            --epochs "$EPOCHS" --utility_lr "$LR" --base_weight_decay "$WD" --warmup_ratio "$WARMUP" \
            --lr_scheduler_type cosine --max_grad_norm 1.0 --max_length "$MAX_LEN" \
            --batch_size "$MB" --gradient_accumulation_steps "$ACC" --layer_type "$LAYER_TYPES" --target_layers all \
            --output_dir "$TUNE_OUT" --log_dir "$TUNE_OUT/logs" --device cuda --dtype "$DTYPE" --seed "$SEED" \
            --gradient_checkpointing --no_wandb --profile_json "$TUNE_OUT/profile.json"
    p3="$(newest_subdir "$TUNE_OUT" phase3_)"
    [[ -n "$p3" && -f "$p3/final_model/config.json" ]] || { echo "[ERROR] [6] final_model 없음" >&2; rm -f "$TUNE_OUT/.done"; exit 1; }
    echo "$p3/final_model" > "$CELL/TUNED_MODEL_DIR"
fi
TUNED="$(cat "$CELL/TUNED_MODEL_DIR")"

# [7]
GSM_OUT="$CELL/gsm8k"; mkdir -p "$GSM_OUT"
if [[ ! -f "$GSM_OUT/.done" ]]; then
    run "$GSM_OUT/.done" "[7] GSM8K FT (선택 열 동결, ${EPOCHS}ep)" -- \
        "$PY" train.py --phase 3 --phase0_model_dir "$TUNED" --basis_dir "$BASIS_DIR" --masks_dir "$CELL/masks_freeze" \
            --phase3_dataset gsm8k --phase3_task_data_path "$TASK_JSON" --phase3_task_samples 0 \
            --epochs "$EPOCHS" --utility_lr "$LR" --base_weight_decay "$WD" --warmup_ratio "$WARMUP" \
            --lr_scheduler_type cosine --max_grad_norm 1.0 --max_length "$MAX_LEN" \
            --batch_size "$MB" --gradient_accumulation_steps "$ACC" --layer_type "$LAYER_TYPES" --target_layers all \
            --output_dir "$GSM_OUT" --log_dir "$GSM_OUT/logs" --device cuda --dtype "$DTYPE" --seed "$SEED" \
            --non_freeze --gradient_checkpointing --no_wandb --profile_json "$GSM_OUT/profile.json"
    p3="$(newest_subdir "$GSM_OUT" phase3_)"
    [[ -n "$p3" && -f "$p3/final_model/config.json" ]] || { echo "[ERROR] [7] final_model 없음" >&2; rm -f "$GSM_OUT/.done"; exit 1; }
    echo "$p3/final_model" > "$CELL/MODEL_DIR"
    "$PY" - "$CELL" "$MODEL" "$START" "$MASKS_C" "$MB" "$ACC" "$SKIP_TUNE" <<'EOF'
import json, sys, os
cell, model, start, masks_c, mb, acc, skip = sys.argv[1:]
m = json.load(open(os.path.join(cell, "masks_freeze", "metadata.json")))
col = json.load(open(os.path.join(cell, "columns.txt.meta.json")))
cfg = {"method": "WSR-SN-Tune (gradient column, arm C)", "arm": "gradcol", "skip_tune": skip, "model": model, "start_model": start,
       "cell": os.path.basename(cell), "masks_C": masks_c, "column_rho": m.get("keep_ratio"),
       "selected_columns": col["per_module"] | {"total": col["total_selected_columns"]},
       "critical_columns": {"total": col["total_selected_columns"]}, "critical_param_fraction": m.get("keep_ratio"),
       "stage6": "phase3 freeze variant (selected columns only, wd 0)", "stage7": "phase3 --non_freeze",
       "lr": 5e-5, "epochs": 3, "batch": f"{mb}x{acc}", "seed": 42,
       "tuned_model_dir": open(os.path.join(cell, "TUNED_MODEL_DIR")).read().strip(),
       "model_dir": open(os.path.join(cell, "MODEL_DIR")).read().strip()}
json.dump(cfg, open(os.path.join(cell, "cell_config.json"), "w"), indent=2)
print(json.dumps(cfg, indent=2))
EOF
fi
log "[done] $CELL → $(cat "$CELL/MODEL_DIR")"
