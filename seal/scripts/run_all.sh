#!/bin/bash
# ============================================================================
# SEAL × WaRP 통합 비교 파이프라인 (gsm8k downstream, circuit_breakers safety)
#
# 실행 결과로 두 모델을 생성한다:
#   (A) baseline : SEAL 데이터 선택 → 표준 full-param SFT
#   (B) WaRP     : SEAL 데이터 선택 → WaRP 공간 SFT (안전 방향 동결)
#
# 저장소 루트에서 실행:  bash seal/scripts/run_all.sh
#
# - 모든 stdout/stderr는 seal/logs/run_all_<timestamp>.log 로 tee 저장된다.
# - 이미 완료된 단계(selector/선택/SFT 산출물 존재)는 자동으로 건너뛴다(resume).
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."   # → 저장소 루트

# ─────────────── 설정 (환경에 맞게 수정) ───────────────
export CUDA_VISIBLE_DEVICES=0          # 사용할 GPU (이 머신은 B200 0번 하나뿐)

# ─── 실행 인터프리터 고정 ───
# tmux/PATH 상태와 무관하게 bare `python`이 항상 hb env(3.10, transformers 포함)를
# 쓰도록 강제한다. 다른 머신에서는 HB_ENV_BIN 환경변수로 오버라이드:
#   HB_ENV_BIN=/path/to/envs/<name>/bin bash seal/scripts/run_all.sh
export PATH="${HB_ENV_BIN:-/home/edgeai_lab/miniconda3/envs/hb/bin}:$PATH"

MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"   # 안전정렬 초기 모델
SAFETY_JSON="data/circuit_breakers_train.json"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_down,ffn_up"  # Phase 1/2/WaRP 전부 동일해야 함
TARGET_LAYERS="all"
KEEP_RATIO=0.1                            # WaRP freeze ratio (안전 계수 유지 비율)
TOPP=0.8                                  # SEAL 데이터 선택 비율

# 학습 하이퍼파라미터
SEL_EPOCHS=2                              # selector 학습 epoch
SFT_EPOCHS=3
LR=5e-5
BATCH=4
GRAD_ACCUM=4
MAXLEN=1024

# Phase 1 basis 재사용 (있으면 지정, 없으면 빈 문자열 → 새로 계산)
PHASE1_BASIS_OVERRIDE=""

CKPT_DIR="seal/ckpt"
OUT_DIR="seal/out"
LOG_DIR="seal/logs"
mkdir -p "$CKPT_DIR" "$OUT_DIR" "$LOG_DIR"

# ─────────────── 로그: 모든 출력을 파일로 tee ───────────────
LOG_FILE="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] 출력이 다음 파일로 저장됩니다: $LOG_FILE"

TOPP_PCT=$(python -c "print(int($TOPP*100))")
SELECTOR_NAME="gsm8k_selector"
SELECTOR_PT="${CKPT_DIR}/${SELECTOR_NAME}_softmax.pt"
SELECTED_JSON="${CKPT_DIR}/gsm8k_selected_top${TOPP_PCT}.json"
BASELINE_OUT="${OUT_DIR}/baseline_top${TOPP_PCT}"
WARP_OUT="${OUT_DIR}/warp_top${TOPP_PCT}"

echo "############################################################"
echo "# Stage 1: SEAL bilevel selector 학습"
echo "############################################################"
if [ -f "$SELECTOR_PT" ]; then
    echo "[skip] selector 이미 존재: $SELECTOR_PT"
else
    python -m seal.train_selector \
        --model_path "$MODEL" \
        --safety_data_path "$SAFETY_JSON" \
        --max_length "$MAXLEN" \
        --epochs "$SEL_EPOCHS" \
        --batch_size "$BATCH" \
        --lora \
        --out_dir "$CKPT_DIR" \
        --selector_name "$SELECTOR_NAME"
fi

echo "############################################################"
echo "# Stage 1.5: top-${TOPP} 데이터 선택"
echo "############################################################"
if [ -f "$SELECTED_JSON" ]; then
    echo "[skip] 선택 인덱스 이미 존재: $SELECTED_JSON"
else
    python -m seal.select_data \
        --selector_path "$SELECTOR_PT" \
        --topp "$TOPP" \
        --out "$SELECTED_JSON"
fi

echo "############################################################"
echo "# Stage 2-A: baseline SFT (선택 데이터, 표준 full-param)"
echo "############################################################"
if [ -f "${BASELINE_OUT}/sft_config.json" ]; then
    echo "[skip] baseline 이미 존재: $BASELINE_OUT"
else
    python -m seal.train_sft \
        --model_path "$MODEL" \
        --selected_indices "$SELECTED_JSON" \
        --max_length "$MAXLEN" \
        --epochs "$SFT_EPOCHS" --learning_rate "$LR" \
        --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
        --output_dir "$BASELINE_OUT"
fi

echo "############################################################"
echo "# WaRP 준비: Phase 1 (basis) + Phase 2 (mask)"
echo "############################################################"
if [ -n "$PHASE1_BASIS_OVERRIDE" ]; then
    BASIS_DIR="$PHASE1_BASIS_OVERRIDE"
    echo "  [skip] Phase 1 재사용: $BASIS_DIR"
else
    python train.py \
        --phase 1 \
        --phase0_model_dir "$MODEL" \
        --safety_dataset circuit_breakers \
        --circuit_breakers_samples_phase1 4994 \
        --batch_size "$BATCH" \
        --layer_type "$LAYER_TYPE" \
        --target_layers "$TARGET_LAYERS" \
        --output_dir ./checkpoints \
        --log_dir ./logs \
        --device cuda --dtype bfloat16 --seed 42
    # 최신 phase1 basis 자동 탐색
    P1=$(find ./checkpoints -maxdepth 1 -type d -name 'phase1_*' -printf '%T@ %p\n' \
         | sort -nr | head -1 | cut -d' ' -f2-)
    BASIS_DIR="${P1}/basis"
fi
echo "  BASIS_DIR=$BASIS_DIR"

python train.py \
    --phase 2 \
    --phase0_model_dir "$MODEL" \
    --basis_dir "$BASIS_DIR" \
    --circuit_breakers_path "$SAFETY_JSON" \
    --dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 4994 \
    --keep_ratio "$KEEP_RATIO" \
    --batch_size "$BATCH" --max_length "$MAXLEN" \
    --layer_type "$LAYER_TYPE" \
    --target_layers "$TARGET_LAYERS" \
    --output_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda --dtype bfloat16 --seed 42 --perlayer
# 최신 phase2 masks 자동 탐색 (checkpoints/phase2_*/checkpoints/masks)
P2=$(find ./checkpoints -maxdepth 1 -type d -name 'phase2_*' -printf '%T@ %p\n' \
     | sort -nr | head -1 | cut -d' ' -f2-)
MASKS_DIR=$(find "$P2" -type d -name masks | head -1)
echo "  MASKS_DIR=$MASKS_DIR"

echo "############################################################"
echo "# Stage 2-B: WaRP SFT (선택 데이터, basis_coeff만 학습)"
echo "############################################################"
if [ -f "${WARP_OUT}/sft_config.json" ]; then
    echo "[skip] WaRP 이미 존재: $WARP_OUT"
else
    python -m seal.train_sft \
        --model_path "$MODEL" \
        --selected_indices "$SELECTED_JSON" \
        --max_length "$MAXLEN" \
        --epochs "$SFT_EPOCHS" --learning_rate "$LR" \
        --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
        --use_warp \
        --basis_dir "$BASIS_DIR" \
        --masks_dir "$MASKS_DIR" \
        --layer_type "$LAYER_TYPE" \
        --target_layers "$TARGET_LAYERS" \
        --output_dir "$WARP_OUT"
fi

echo "############################################################"
echo "# ✅ 완료"
echo "#   baseline: ${BASELINE_OUT}"
echo "#   WaRP:     ${WARP_OUT}"
echo "#   log:      ${LOG_FILE}"
echo "############################################################"
