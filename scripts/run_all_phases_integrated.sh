#!/bin/bash

# Safety-WaRP-LLM: Complete Training Pipeline (Integrated)
# Phase 1 (Basis) -> Phase 2 (Importance) -> Phase 3 (Learning)
# 
# run_phase1_basis.sh, run_phase2_importance.sh, run_phase3_learning.sh를 통합
# 한 번에 모든 phase를 순차적으로 실행
source /home/yonsei_jong/miniconda3/etc/profile.d/conda.sh
conda activate hb
set -e  # Exit on error
set -o pipefail  # Ensure failures are not hidden by tee pipelines
export CUDA_VISIBLE_DEVICES=0

echo "========================================================================"
echo "Safety-WaRP-LLM: Complete Training Pipeline (Integrated)"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

# Phase 0 모델
PHASE0_MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"  


# Phase 1: Basis Construction
# ==============================
# Dataset 선택 (Safety 또는 Utility)
# Options: circuit_breakers, wikipedia
PHASE1_DATASET="circuit_breakers"
PHASE1_SAMPLES=4994
# 기존 basis가 있으면 Phase 1 스킵 (빈 문자열이면 Phase 1 수행)
PHASE1_BASIS_DIR_OVERRIDE=""


# Phase 2: Importance Scoring
# ==============================
# Dataset 선택 (동일하게 사용)
PHASE2_DATASET="circuit_breakers"
PHASE2_SAMPLES=4994
KEEP_RATIO_LIST=("0.1")  

# Two-Mask 설정 (비활성화하려면 TWO_MASK="" 로 설정)
# preserve_mask AND NOT adapt_mask → adapt에 중요한 파라미터는 Phase 3에서 학습 가능
# TWO_MASK=""           # "" = 비활성화 (기본), "true" = 활성화
# ADAPT_DATASET="math" # adapt 데이터셋: gsm8k, math, metamath, wikipedia, safety
# ADAPT_SAMPLES=4994       # 0=전체

# Phase 3: Incremental Learning
# ==============================
# Dataset 선택 (Utility 또는 Safety)
PHASE3_DATASET="gsm8k" # Options: safety, gsm8k, metamath, math, agnews, medqa, mmlu

# SafeInstr: safety data mixing (0.0 = 비활성화, 0.1 = 학습 데이터의 10%)
SAFEINSTR_RATIO=0.0
CIRCUIT_BREAKERS_PATH="./data/circuit_breakers_train.json"

# Phase3=MATH 설정
MATH_SUBJECTS="all"  # 예: Algebra,Geometry
MATH_LEVELS="all"    # 예: 1,2,3,4,5

# Phase3=AGNEWS 설정
AGNEWS_DATASET_PATH="/home/yonsei_jong/Safety-WaRP-LLM/data/agnews_train_8000.jsonl"   # --agnews_dataset_path 필수 (agnews 선택 시)
AGNEWS_SAMPLES=8000      # 0=전체

# Phase3=MEDQA 설정
MEDQA_DATASET_PATH="/home/yonsei_jong/Safety-WaRP-LLM/data/medqa_train_10178.jsonl"   # --medqa_dataset_path 필수 (medqa 선택 시)
MEDQA_SAMPLES=10000      # 0=전체

# Phase3=MMLU 설정
MMLU_SUBJECT="all"                # all 또는 단일 subject
MMLU_SPLIT="auxiliary_train"      # auxiliary_train | train | validation | test | dev
MMLU_EVAL_SPLIT="validation"
MMLU_SAMPLES=10000                 # 0=전체
MMLU_EVAL_SAMPLES=0                # 0=eval 생략

if [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_SAMPLES=4994
elif [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_SAMPLES=0  # 0 = all samples
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_SAMPLES=10000  # 0 = all samples
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_SAMPLES=0  # 0 = all samples
elif [ "$PHASE3_DATASET" = "agnews" ]; then
    PHASE3_SAMPLES=$AGNEWS_SAMPLES
elif [ "$PHASE3_DATASET" = "medqa" ]; then
    PHASE3_SAMPLES=$MEDQA_SAMPLES
elif [ "$PHASE3_DATASET" = "mmlu" ]; then
    PHASE3_SAMPLES=$MMLU_SAMPLES
fi

# 공통 설정
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
DTYPE="bfloat16"
DEVICE="auto"
EPOCHS=3
# LR_LIST=("1e-5" "3e-5" "5e-5")
LR_LIST=("5e-5")  
TARGET_LAYERS="all"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"
BASE_OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BASE_OUTPUT_DIR
mkdir -p $LOG_DIR

# ========================================================================
# Resource accounting (phase별 소요 시간 / VRAM)
#   - wall-clock: 아래 bash 타이머 (파이썬 기동/모델 다운로드 포함)
#   - 시간/VRAM 상세: train.py 가 --profile_json 경로에 저장하는 요약 JSON
#     (stage별 시간, torch alloc/reserved peak, 디바이스 peak)
# ========================================================================
PROFILE_DIR="$LOG_DIR/profile_${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"
PHASE_TIMES=()   # "label|seconds|profile_json" 형식

fmt_hms() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

record_phase_time() {
    # record_phase_time <label> <start_SECONDS> <profile_json|->
    local label="$1" start="$2" pjson="$3"
    local elapsed=$((SECONDS - start))
    PHASE_TIMES+=("${label}|${elapsed}|${pjson}")
    echo ""
    echo "⏱  ${label} wall-clock: $(fmt_hms $elapsed)  (${elapsed}s)"
    if [ "$pjson" != "-" ] && [ -f "$pjson" ]; then
        python - "$pjson" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"🖥  peak VRAM: device={d['peak_device_gb']:.2f}GB  "
      f"torch_reserved={d['peak_torch_reserved_gb']:.2f}GB  "
      f"torch_alloc={d['peak_torch_alloc_gb']:.2f}GB")
PYEOF
    fi
    echo ""
}

echo "Configuration:"
echo "  Phase 0 Model: $PHASE0_MODEL"
echo "  Phase 1 Dataset: $PHASE1_DATASET (samples=$PHASE1_SAMPLES)"
echo "  Phase 2 Dataset: $PHASE2_DATASET (samples=$PHASE2_SAMPLES)"
echo "  Phase 3 Dataset: $PHASE3_DATASET (samples=$PHASE3_SAMPLES)"
echo "  SafeInstr Ratio: $SAFEINSTR_RATIO"
echo "  Keep Ratios: ${KEEP_RATIO_LIST[*]}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Output Dir: $BASE_OUTPUT_DIR"
echo ""

# Phase 3 Dataset validation
if [[ ! "$PHASE3_DATASET" =~ ^(safety|gsm8k|metamath|math|agnews|medqa|mmlu)$ ]]; then
    echo "❌ ERROR: Unknown Phase 3 dataset: $PHASE3_DATASET"
    echo "Choose from: safety, gsm8k, metamath, math, agnews, medqa, mmlu"
    exit 1
fi

# ========================================================================
# Phase 1: Basis Construction
# ========================================================================
echo ""
echo "========================================================================"
echo "PHASE 1: Basis Construction"
echo "========================================================================"
echo ""

if [ -n "$PHASE1_BASIS_DIR_OVERRIDE" ]; then
    PHASE1_BASIS_DIR="$PHASE1_BASIS_DIR_OVERRIDE"
    if [ ! -d "$PHASE1_BASIS_DIR" ]; then
        echo "❌ ERROR: PHASE1_BASIS_DIR_OVERRIDE not found: $PHASE1_BASIS_DIR"
        exit 1
    fi
    echo "✅ Phase 1 skipped (existing basis provided)"
    echo "   Using basis: $PHASE1_BASIS_DIR"
    echo ""
else
    if [ "$PHASE1_DATASET" = "circuit_breakers" ]; then
        PHASE1_DATASET_ARG="--circuit_breakers_samples_phase1 $PHASE1_SAMPLES"
    elif [ "$PHASE1_DATASET" = "wikipedia" ]; then
        PHASE1_DATASET_ARG="--wikipedia_samples_phase1 $PHASE1_SAMPLES"
    else
        echo "❌ ERROR: Unknown Phase 1 dataset: $PHASE1_DATASET"
        echo "Choose from: circuit_breakers, wikipedia"
        exit 1
    fi

    PHASE1_PROFILE_JSON="$PROFILE_DIR/phase1.json"
    PHASE1_START=$SECONDS

    python train.py \
        --phase 1 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --safety_dataset "$PHASE1_DATASET" \
        $PHASE1_DATASET_ARG \
        --batch_size $BATCH_SIZE \
        --layer_type "$LAYER_TYPE" \
        --target_layers $TARGET_LAYERS \
        --output_dir $BASE_OUTPUT_DIR \
        --log_dir $LOG_DIR \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed 42 \
        --profile_json "$PHASE1_PROFILE_JSON" \
        2>&1 | tee $LOG_DIR/phase1_${TIMESTAMP}.log

    record_phase_time "Phase 1" "$PHASE1_START" "$PHASE1_PROFILE_JSON"

    # Phase 1 출력 경로 추출 (최신 phase1 디렉토리 찾기)
    PHASE1_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase1_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    PHASE1_BASIS_DIR="$PHASE1_OUTPUT_DIR/basis"

    if [ ! -d "$PHASE1_BASIS_DIR" ]; then
        echo "❌ ERROR: Phase 1 basis directory not found: $PHASE1_BASIS_DIR"
        exit 1
    fi

    echo ""
    echo "✅ Phase 1 completed successfully"
    echo "   Basis saved to: $PHASE1_BASIS_DIR"
    echo ""
fi

# ========================================================================
# Phase 2 & Phase 3: Keep Ratio Sweep
# Phase 1 basis 공유, keep_ratio마다 Phase 2(mask 생성) → Phase 3(학습) 수행
# ========================================================================

if [ "$PHASE2_DATASET" = "circuit_breakers" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 $PHASE2_SAMPLES"
elif [ "$PHASE2_DATASET" = "wikipedia" ]; then
    PHASE2_DATASET_ARG="--dataset_phase2 wikipedia --wikipedia_samples_phase2 $PHASE2_SAMPLES"
else
    echo "ERROR: Unknown Phase 2 dataset: $PHASE2_DATASET"
    exit 1
fi

if [ -n "$TWO_MASK" ]; then
    TWO_MASK_ARG="--two_mask --adapt_dataset_phase2 $ADAPT_DATASET --adapt_samples_phase2 $ADAPT_SAMPLES"
    echo "[Two-Mask] Enabled: adapt_dataset=$ADAPT_DATASET, adapt_samples=$ADAPT_SAMPLES"
else
    TWO_MASK_ARG=""
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$PHASE3_DATASET" = "gsm8k" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "safety" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset safety --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase3 $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset metamath --metamath_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "math" ]; then
    PHASE3_DATASET_ARG="--phase3_dataset math --math_samples $PHASE3_SAMPLES --math_subjects $MATH_SUBJECTS --math_levels $MATH_LEVELS"
elif [ "$PHASE3_DATASET" = "agnews" ]; then
    if [ -z "$AGNEWS_DATASET_PATH" ]; then
        echo "ERROR: AGNEWS_DATASET_PATH must be set when PHASE3_DATASET=agnews"
        exit 1
    fi
    PHASE3_DATASET_ARG="--phase3_dataset agnews --agnews_dataset_path $AGNEWS_DATASET_PATH --agnews_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "medqa" ]; then
    if [ -z "$MEDQA_DATASET_PATH" ]; then
        echo "ERROR: MEDQA_DATASET_PATH must be set when PHASE3_DATASET=medqa"
        exit 1
    fi
    PHASE3_DATASET_ARG="--phase3_dataset medqa --medqa_dataset_path $MEDQA_DATASET_PATH --medqa_samples $PHASE3_SAMPLES"
elif [ "$PHASE3_DATASET" = "mmlu" ]; then
    PHASE3_DATASET_ARG=""
else
    echo "ERROR: Unknown Phase 3 dataset: $PHASE3_DATASET"
    exit 1
fi

PHASE3_OUTPUT_DIRS=()

for KEEP_RATIO in "${KEEP_RATIO_LIST[@]}"; do
    KR_SAFE=$(echo "$KEEP_RATIO" | sed 's/[^a-zA-Z0-9_-]/_/g')

    # ------------------------------------------------------------------
    # Phase 2: Importance Scoring (keep_ratio=$KEEP_RATIO)
    # ------------------------------------------------------------------
    echo ""
    echo "========================================================================"
    echo "PHASE 2: Importance Scoring  (keep_ratio=$KEEP_RATIO)"
    echo "========================================================================"
    echo ""

    PHASE2_PROFILE_JSON="$PROFILE_DIR/phase2_kr${KR_SAFE}.json"
    PHASE2_START=$SECONDS

    python train.py \
        --phase 2 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --basis_dir "$PHASE1_BASIS_DIR" \
        --circuit_breakers_path ./data/circuit_breakers_train.json \
        $PHASE2_DATASET_ARG \
        --keep_ratio $KEEP_RATIO \
        --batch_size $BATCH_SIZE \
        --max_length 1024 \
        --layer_type "$LAYER_TYPE" \
        --target_layers $TARGET_LAYERS \
        --output_dir $BASE_OUTPUT_DIR \
        --log_dir $LOG_DIR \
        --device $DEVICE \
        --dtype $DTYPE \
        --seed 42 \
        --perlayer \
        --profile_json "$PHASE2_PROFILE_JSON" \
        $TWO_MASK_ARG \
        2>&1 | tee $LOG_DIR/phase2_kr${KR_SAFE}_${TIMESTAMP}.log

    record_phase_time "Phase 2 (kr=$KEEP_RATIO)" "$PHASE2_START" "$PHASE2_PROFILE_JSON"

    PHASE2_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase2_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    PHASE2_MASKS_DIR="$PHASE2_OUTPUT_DIR/checkpoints/masks"

    if [ ! -d "$PHASE2_MASKS_DIR" ]; then
        echo "ERROR: Phase 2 (kr=$KEEP_RATIO) masks not found: $PHASE2_MASKS_DIR"
        exit 1
    fi

    echo ""
    echo "Phase 2 (kr=$KEEP_RATIO) completed: $PHASE2_MASKS_DIR"
    echo ""

    # ------------------------------------------------------------------
    # Phase 3: Incremental Learning (keep_ratio=$KEEP_RATIO, LR sweep)
    # ------------------------------------------------------------------
    echo "========================================================================"
    echo "PHASE 3: Incremental Learning  (keep_ratio=$KEEP_RATIO, LR: ${LR_LIST[*]})"
    echo "========================================================================"
    echo ""

    for LEARNING_RATE in "${LR_LIST[@]}"; do
        LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')

        echo "──────────────────────────────────────────────────────────────────────"
        echo "  Phase 3: keep_ratio=$KEEP_RATIO  LR=$LEARNING_RATE"
        echo "──────────────────────────────────────────────────────────────────────"

        # SafeInstr 인자 구성
        if (( $(echo "$SAFEINSTR_RATIO > 0" | bc -l) )); then
            SAFEINSTR_ARG="--safety_mix_ratio $SAFEINSTR_RATIO --circuit_breakers_path $CIRCUIT_BREAKERS_PATH"
        else
            SAFEINSTR_ARG=""
        fi

        PHASE3_PROFILE_JSON="$PROFILE_DIR/phase3_kr${KR_SAFE}_lr${LR_SAFE}.json"
        PHASE3_START=$SECONDS

        if [ "$PHASE3_DATASET" = "mmlu" ]; then
            PHASE3_OUTPUT_DIR="$BASE_OUTPUT_DIR/phase3_mmlu_kr${KR_SAFE}_lr${LR_SAFE}_${TIMESTAMP}"
            python mmlu_eval/finetune_mmlu_full_params.py \
                --model_path "$PHASE0_MODEL" \
                --mmlu_subject "$MMLU_SUBJECT" \
                --mmlu_split "$MMLU_SPLIT" \
                --mmlu_eval_split "$MMLU_EVAL_SPLIT" \
                --num_train_samples "$PHASE3_SAMPLES" \
                --num_eval_samples "$MMLU_EVAL_SAMPLES" \
                --output_dir "$PHASE3_OUTPUT_DIR" \
                --learning_rate "$LEARNING_RATE" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --grad_accum "$GRAD_ACCUM_STEPS" \
                --max_length 1024 \
                --safety_mix_ratio "$SAFEINSTR_RATIO" \
                --safety_data_path "$CIRCUIT_BREAKERS_PATH" \
                2>&1 | tee $LOG_DIR/phase3_kr${KR_SAFE}_lr${LR_SAFE}_${TIMESTAMP}.log

            # mmlu 경로는 train.py 를 쓰지 않으므로 wall-clock 만 기록
            PHASE3_PROFILE_JSON="-"
        else
            python train.py \
                --phase 3 \
                --phase0_model_dir "$PHASE0_MODEL" \
                --basis_dir "$PHASE1_BASIS_DIR" \
                --masks_dir "$PHASE2_MASKS_DIR" \
                $PHASE3_DATASET_ARG \
                --epochs $EPOCHS \
                --utility_lr $LEARNING_RATE \
                --batch_size $BATCH_SIZE \
                --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
                --layer_type "$LAYER_TYPE" \
                --target_layers $TARGET_LAYERS \
                --output_dir $BASE_OUTPUT_DIR \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --dtype $DTYPE \
                --seed 42 \
                --non_freeze \
                --constrained_sft \
                --csft_bias_factor 10 --csft_bias_length 3 --csft_first_token_bias_factor 3 \
                --profile_json "$PHASE3_PROFILE_JSON" \
                $SAFEINSTR_ARG \
                2>&1 | tee $LOG_DIR/phase3_kr${KR_SAFE}_lr${LR_SAFE}_${TIMESTAMP}.log

            PHASE3_OUTPUT_DIR=$(find $BASE_OUTPUT_DIR -maxdepth 1 -name "phase3_*" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
        fi

        record_phase_time "Phase 3 (kr=$KEEP_RATIO, lr=$LEARNING_RATE)" "$PHASE3_START" "$PHASE3_PROFILE_JSON"

        if [ ! -d "$PHASE3_OUTPUT_DIR" ]; then
            echo "WARNING: Phase 3 (kr=$KEEP_RATIO, LR=$LEARNING_RATE) output not found"
        else
            echo "Phase 3 (kr=$KEEP_RATIO, LR=$LEARNING_RATE) completed: $PHASE3_OUTPUT_DIR"
            PHASE3_OUTPUT_DIRS+=("kr${KEEP_RATIO}_lr${LEARNING_RATE}:$PHASE3_OUTPUT_DIR")
        fi
        echo ""
    done

done

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "========================================================================"
echo "Complete Pipeline Finished!"
echo "========================================================================"
echo ""
echo "Phase 1 Basis:  $PHASE1_BASIS_DIR"
echo ""
echo "Phase 3 Models (keep_ratio x LR):"
for entry in "${PHASE3_OUTPUT_DIRS[@]}"; do
    label="${entry%%:*}"
    dir="${entry#*:}"
    echo "  [$label]  $dir/final_model"
done
echo ""
echo "Logs:"
echo "  Phase 1: $LOG_DIR/phase1_${TIMESTAMP}.log"
for KEEP_RATIO in "${KEEP_RATIO_LIST[@]}"; do
    KR_SAFE=$(echo "$KEEP_RATIO" | sed 's/[^a-zA-Z0-9_-]/_/g')
    echo "  Phase 2 (kr=$KEEP_RATIO): $LOG_DIR/phase2_kr${KR_SAFE}_${TIMESTAMP}.log"
    for LEARNING_RATE in "${LR_LIST[@]}"; do
        LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')
        echo "  Phase 3 (kr=$KEEP_RATIO, LR=$LEARNING_RATE): $LOG_DIR/phase3_kr${KR_SAFE}_lr${LR_SAFE}_${TIMESTAMP}.log"
    done
done
echo ""
echo "========================================================================"
echo "Resource Summary (phase별 소요 시간 / peak VRAM)"
echo "========================================================================"
python - "$PROFILE_DIR/pipeline_resource_summary.json" "${PHASE_TIMES[@]}" <<'PYEOF'
import json, os, sys

out_path, entries = sys.argv[1], sys.argv[2:]


def hms(sec):
    sec = int(round(sec))
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


rows, total = [], 0.0
for e in entries:
    label, secs, pjson = e.split('|', 2)
    secs = float(secs)
    total += secs
    row = {'phase': label, 'wall_seconds': secs, 'wall_clock': hms(secs),
           'profile_json': None if pjson == '-' else pjson}
    if pjson != '-' and os.path.isfile(pjson):
        d = json.load(open(pjson))
        row.update({
            'peak_device_gb': d['peak_device_gb'],
            'peak_device_delta_gb': d.get('peak_device_delta_gb', d['peak_device_gb']),
            'peak_torch_reserved_gb': d['peak_torch_reserved_gb'],
            'peak_torch_alloc_gb': d['peak_torch_alloc_gb'],
            'inner_seconds': d['total_seconds'],
            'stages': [{'stage': s['stage'], 'duration': s['duration'],
                        'device_peak_gb': s['device_peak_gb']} for s in d['stages']],
        })
    rows.append(row)

hdr = (f"{'phase':<34}{'wall-clock':>12}{'device':>10}{'device-base':>13}"
       f"{'torch_resv':>13}{'torch_alloc':>13}")
print(hdr)
print('-' * len(hdr))
for r in rows:
    dev = f"{r['peak_device_gb']:.2f}G" if 'peak_device_gb' in r else '-'
    dlt = f"{r['peak_device_delta_gb']:.2f}G" if 'peak_device_delta_gb' in r else '-'
    res = f"{r['peak_torch_reserved_gb']:.2f}G" if 'peak_torch_reserved_gb' in r else '-'
    alc = f"{r['peak_torch_alloc_gb']:.2f}G" if 'peak_torch_alloc_gb' in r else '-'
    print(f"{r['phase'][:33]:<34}{r['wall_clock']:>12}{dev:>10}{dlt:>13}{res:>13}{alc:>13}")
print('-' * len(hdr))
peak = max([r.get('peak_device_gb', 0.0) for r in rows] + [0.0])
peak_delta = max([r.get('peak_device_delta_gb', 0.0) for r in rows] + [0.0])
print(f"{'TOTAL':<34}{hms(total):>12}{peak:>9.2f}G{peak_delta:>12.2f}G")
print("  device = nvidia-smi 기준 전체 사용량, device-base = 프로파일 시작 시점 대비 증가분")
print("  (GPU 단독 점유 시 두 값 차이는 CUDA context 정도)")
print()
for r in rows:
    if not r.get('stages'):
        continue
    print(f"[{r['phase']}] stage breakdown:")
    for s in r['stages']:
        print(f"    {s['stage']:<28}{s['duration']:>10}   peak {s['device_peak_gb']:.2f}GB")
    print()

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({'phases': rows, 'total_wall_seconds': total,
               'total_wall_clock': hms(total), 'peak_device_gb': peak}, f, indent=2)
print(f"Saved: {out_path}")
PYEOF
echo ""
echo "  Per-phase profile JSON: $PROFILE_DIR/"
echo ""

echo ""
echo "Configuration:"
echo "  - Phase 1/2 Dataset: $PHASE1_DATASET"
echo "  - Phase 3 Dataset:   $PHASE3_DATASET"
echo "  - Keep Ratios:       ${KEEP_RATIO_LIST[*]}"
echo "  - Learning Rates:    ${LR_LIST[*]}"
echo "  - Epochs:            $EPOCHS"
echo "  - Layer Types:       $LAYER_TYPE"
echo ""

if [ "$PHASE3_DATASET" = "safety" ]; then
    echo "🔐 Safety Training Mode:"
    echo "  - Using HuggingFace Trainer (Non-Freeze)"
    echo "  - All params trainable, WaRP masking via forward"
    echo "  - Automatic gradient blocking (mask=1)"
elif [ "$PHASE3_DATASET" = "gsm8k" ]; then
    echo "📚 Utility Training Mode (GSM8K):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Math reasoning (GSM8K) learning"
elif [ "$PHASE3_DATASET" = "metamath" ]; then
    echo "📚 Utility Training Mode (MetaMath):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Advanced math reasoning (MetaMath) learning"
elif [ "$PHASE3_DATASET" = "math" ]; then
    echo "📚 Utility Training Mode (Hendrycks MATH):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Subject filter: $MATH_SUBJECTS, Level filter: $MATH_LEVELS"
elif [ "$PHASE3_DATASET" = "agnews" ]; then
    echo "📰 Utility Training Mode (AG News):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - News classification (AG News) learning"
    echo "  - Dataset path: $AGNEWS_DATASET_PATH, Samples: $AGNEWS_SAMPLES"
elif [ "$PHASE3_DATASET" = "medqa" ]; then
    echo "🏥 Utility Training Mode (MedQA USMLE):"
    echo "  - Using HuggingFace Trainer"
    echo "  - basis_coeff only training with WaRP masking"
    echo "  - Medical QA MCQ (MedQA) learning"
    echo "  - Dataset path: $MEDQA_DATASET_PATH, Samples: $MEDQA_SAMPLES"
elif [ "$PHASE3_DATASET" = "mmlu" ]; then
    echo "🎯 Downstream Utility Training Mode (MMLU):"
    echo "  - Using mmlu_eval/finetune_mmlu_full_params.py"
    echo "  - Subject: $MMLU_SUBJECT, Split: $MMLU_SPLIT"
    echo "  - Samples: $MMLU_SAMPLES, Eval samples: $MMLU_EVAL_SAMPLES"
fi

echo ""
echo "========================================================================"
