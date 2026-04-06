#!/bin/bash

# Phase 3: Incremental Learning

echo "========================================="
echo "Phase 3: Incremental Learning (Fixed)"
echo "========================================="

# 이전 Phase 결과 경로 (로컬 디렉토리 또는 Hugging Face 모델 ID)
# PHASE0_MODEL="./checkpoints/phase0_20260213_230047"  # 로컬 디렉토리 예시
PHASE0_MODEL="meta-llama/Llama-3.2-3B"  # Hugging Face 모델 ID 예시
BASIS_DIR="./checkpoints/phase1_20260405_002504/basis"
MASKS_DIR="./checkpoints/phase2_20260405_022932/checkpoints/masks"

# ========================================
# Dataset 선택 (CONFIGURE THIS)
# ========================================
# 옵션 1: GSM8K (Utility Learning) - SFTTrainer 방식
# DATASET="gsm8k"
# GSM8K_SAMPLES=0

# 옵션 2: Safety (Safety Learning) - phase0_SSFT 커스텀 루프 방식
DATASET="safety"
CIRCUIT_BREAKERS_SAMPLES=4994

# 옵션 3: MetaMath (Utility Learning) - SFTTrainer 방식
# DATASET="metamath"
# METAMATH_SAMPLES=10000  # 0 = all samples

# 옵션 4: Hendrycks MATH (Utility Learning) - SFTTrainer 방식
# DATASET="math"
# MATH_SAMPLES=0           # 0 = all samples
# MATH_SUBJECTS="all"     # 예: Algebra,Geometry
# MATH_LEVELS="all"       # 예: 1,2,3,4,5
# 
# ========================================

# PHASE0_MODEL이 로컬 경로처럼 보일 때만 디렉토리 체크
if [[ "$PHASE0_MODEL" == ./* || "$PHASE0_MODEL" == /* ]]; then
    if [ ! -d "$PHASE0_MODEL" ]; then
        echo "ERROR: Phase 0 모델을 찾을 수 없습니다: $PHASE0_MODEL"
        exit 1
    fi
fi

if [ ! -d "$BASIS_DIR" ]; then
    echo "ERROR: Basis를 찾을 수 없습니다: $BASIS_DIR"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "ERROR: Masks를 찾을 수 없습니다: $MASKS_DIR"
    echo "먼저 scripts/run_phase2_importance.sh를 실행하세요."
    exit 1
fi

echo ""
echo "Dataset 설정:"
if [ "$DATASET" = "gsm8k" ]; then
    echo "  - Type: Utility Learning (GSM8K)"
    echo "  - Samples: $GSM8K_SAMPLES (0=all)"
    DATASET_ARG="--phase3_dataset gsm8k --gsm8k_samples $GSM8K_SAMPLES"
elif [ "$DATASET" = "safety" ]; then
    echo "  - Type: Safety Learning (Circuit Breakers)"
    echo "  - Samples: $CIRCUIT_BREAKERS_SAMPLES"
    DATASET_ARG="--phase3_dataset safety --circuit_breakers_path ./data/circuit_breakers_train.json --circuit_breakers_samples_phase3 $CIRCUIT_BREAKERS_SAMPLES"
elif [ "$DATASET" = "metamath" ]; then
    echo "  - Type: Utility Learning (MetaMath)"
    echo "  - Samples: $METAMATH_SAMPLES (0=all)"
    DATASET_ARG="--phase3_dataset metamath --metamath_samples $METAMATH_SAMPLES"
elif [ "$DATASET" = "math" ]; then
    echo "  - Type: Utility Learning (Hendrycks MATH)"
    echo "  - Samples: $MATH_SAMPLES (0=all)"
    echo "  - Subjects: $MATH_SUBJECTS"
    echo "  - Levels: $MATH_LEVELS"
    DATASET_ARG="--phase3_dataset math --math_samples $MATH_SAMPLES --math_subjects $MATH_SUBJECTS --math_levels $MATH_LEVELS"
else
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Choose from: gsm8k, safety, metamath, math"
    exit 1
fi
echo ""

#  ${NON_FREEZE_FLAG} 옵션은 Phase 3에서 WaRP이외 layer를 freeze하지 않고 학습할 때 사용.
#     --layer_type attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_down,ffn_up \
#     --non_freeze
python train.py \
    --phase 3 \
    --phase0_model_dir "$PHASE0_MODEL" \
    --basis_dir "$BASIS_DIR" \
    --masks_dir "$MASKS_DIR" \
    $DATASET_ARG \
    --epochs 3 \
    --utility_lr 1e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --target_layers all \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --dtype bfloat16 \
    --seed 42 \
    --non_freeze
    

echo ""
echo "========================================="
echo "Phase 3 완료! (Dataset: $DATASET)"
echo "최종 모델: ./checkpoints/phase3_*/final_model"
echo ""
if [ "$DATASET" = "safety" ]; then
    echo "✅ Safety dataset used with phase0_SSFT training loop"
    echo "   - Custom training loop (AdamW8bit, gradient clipping)"
    echo "   - Hyperparameters: LR=1e-5, Epochs=3, Batch=4, GradAccum=4"
    echo "   - WaRP masking: basis_coeff만 학습 가능"
elif [ "$DATASET" = "gsm8k" ]; then
    echo "✅ GSM8K dataset used with SFTTrainer"
    echo "   - HuggingFace Trainer-based training loop"
    echo "   - Hyperparameters: LR=1e-5, Epochs=3, Batch=4, GradAccum=4"
    echo "   - WaRP masking: basis_coeff만 학습 가능"
elif [ "$DATASET" = "metamath" ]; then
    echo "✅ MetaMath dataset used with SFTTrainer"
    echo "   - HuggingFace Trainer-based training loop"
    echo "   - Hyperparameters: LR=1e-5, Epochs=3, Batch=4, GradAccum=4"
    echo "   - WaRP masking: basis_coeff만 학습 가능"
elif [ "$DATASET" = "math" ]; then
    echo "✅ Hendrycks MATH dataset used with SFTTrainer"
    echo "   - HuggingFace Trainer-based training loop"
    echo "   - Hyperparameters: LR=1e-5, Epochs=3, Batch=4, GradAccum=4"
    echo "   - Subject filter: $MATH_SUBJECTS, Level filter: $MATH_LEVELS"
    echo "   - WaRP masking: basis_coeff만 학습 가능"
fi
echo "========================================="
