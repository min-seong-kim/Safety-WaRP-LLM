#!/bin/bash
# ============================================================================
# WSR-SEAL — Llama-2-13B-Chat (논문 Table 3 확장: SEAL 에 WSR-Tune 을 얹은 팔)
#
# 7B 쌍(kmseong/llama2_7b_chat_seal_5e-5  +  ..._seal_warp_5e-5)의 13B 대응이다.
# 13B 는 SEAL(baseline) 만 이미 존재하므로 **WaRP 팔만** 만든다:
#     kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5   (기존, baseline)
#   + kmseong/llama2_13b-chat-CB_SSFT-seal-warp_...               (이번에 생성)
#
# ── 비교가 성립하는 이유 ────────────────────────────────────────────────────
# SEAL 과 SEAL+WSR 은 **같은 선택 데이터**를 봐야 한다. selector 를 다시 돌리면
# (학습은 bitwise 재현되지 않으므로) 다른 부분집합이 뽑혀 데이터 차이가 교란요인이
# 된다. 그래서 selector 를 재학습하지 않고, 기존 13B SEAL 저장소의
# `sft_config.json > select_meta` 에 통째로 저장돼 있던 인덱스 5978개를 그대로 복원해
# 쓴다(seal/ckpt/revision/cb/llama2_13b/gsm8k_selected_top80.json).
# 따라서 두 팔의 차이는 **WaRP 재매개변수화 하나뿐**이다.
#
# ── baseline 과 맞춘 하이퍼파라미터 (허브 sft_config.json 에서 읽음) ────────
#   base_model  wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5
#   epochs 3 · lr 5e-5 · wd 0.01 · warmup 0.1 · cosine · max_grad_norm 1.0
#   batch 1 × grad_accum 16 (effective 16) · max_len 1024 · seed 42 · bf16
#   task JSON  data/gsm8k_train_task_7473.json   (21_seal.sh 와 동일 경로)
#
# Phase 1/2 는 7B WSR-SEAL(seal/scripts/run_all.sh)과 같은 설정:
#   circuit_breakers 전량(4994) · keep_ratio 0.1 · --perlayer ·
#   layer_type attn_q,attn_k,attn_v,ffn_up,ffn_down · target_layers all
# ⚠️ layer_type / target_layers 는 Phase 1·2·SFT 세 곳이 반드시 같아야 한다.
#
# 사용:  bash seal/scripts/run_warp_seal_13b.sh
#        SKIP_UPLOAD=1 bash seal/scripts/run_warp_seal_13b.sh   # 업로드 생략
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

M=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3
source "$M/etc/profile.d/conda.sh"; conda activate hb
export CUDA_VISIBLE_DEVICES=0
PY="$M/envs/hb/bin/python"

MODEL="wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5"
SAFETY_JSON="data/circuit_breakers_train.json"
TASK_JSON="data/gsm8k_train_task_7473.json"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_up,ffn_down"
TARGET_LAYERS="all"
KEEP_RATIO=0.1
CB_SAMPLES=4994            # circuit_breakers_train.json 전량
MAXLEN=1024; EPOCHS=3; LR=5e-5; WD=0.01; WARMUP=0.1
BATCH=1; GRAD_ACCUM=16     # 13B: effective 16 유지
SEED=42
REPO_ID="${REPO_ID:-kmseong/llama2_13b-chat-CB_SSFT-seal-warp_gsm8k_topp0.8_rho0.1_lr5e-5}"

SEL_JSON="seal/ckpt/revision/cb/llama2_13b/gsm8k_selected_top80.json"
WARP_OUT="seal/out/warp_13b_top80"
CKPT="checkpoints/seal_warp_13b"
LOG_DIR="seal/logs"; mkdir -p "$LOG_DIR" "$CKPT"
LOG_FILE="${LOG_DIR}/warp_seal_13b_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $LOG_FILE"

[ -f "$SEL_JSON" ] || { echo "❌ 선택 인덱스 없음: $SEL_JSON"; exit 1; }
$PY -c "
import json,sys; d=json.load(open('$SEL_JSON'))
assert d['num_selected']==5978 and d['total']==7473 and d['topp']==0.8, d
print(f\"[check] 선택 인덱스 OK: {d['num_selected']}/{d['total']} top-p {d['topp']}\")"
[ $? -eq 0 ] || exit 1

echo "############ Phase 1 — safety basis (13B) ############"
BASIS_DIR=$(cat "$CKPT/BASIS_DIR" 2>/dev/null || true)
if [ -n "$BASIS_DIR" ] && [ -d "$BASIS_DIR" ]; then
    echo "[skip] basis 재사용: $BASIS_DIR"
else
    $PY train.py --phase 1 \
        --phase0_model_dir "$MODEL" \
        --safety_dataset circuit_breakers \
        --circuit_breakers_path "$SAFETY_JSON" \
        --circuit_breakers_samples_phase1 "$CB_SAMPLES" \
        --batch_size 1 \
        --layer_type "$LAYER_TYPE" --target_layers "$TARGET_LAYERS" \
        --output_dir "$CKPT" --log_dir ./logs \
        --device cuda --dtype bfloat16 --seed "$SEED" || exit 1
    P1=$(find "$CKPT" -maxdepth 1 -type d -name 'phase1_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
    BASIS_DIR="${P1}/basis"; echo "$BASIS_DIR" > "$CKPT/BASIS_DIR"
fi
echo "  BASIS_DIR=$BASIS_DIR"
[ -d "$BASIS_DIR" ] || { echo "❌ basis 생성 실패"; exit 1; }

echo "############ Phase 2 — importance mask (rho=$KEEP_RATIO) ############"
MASKS_DIR=$(cat "$CKPT/MASKS_DIR" 2>/dev/null || true)
if [ -n "$MASKS_DIR" ] && [ -d "$MASKS_DIR" ]; then
    echo "[skip] masks 재사용: $MASKS_DIR"
else
    $PY train.py --phase 2 \
        --phase0_model_dir "$MODEL" \
        --basis_dir "$BASIS_DIR" \
        --circuit_breakers_path "$SAFETY_JSON" \
        --dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 "$CB_SAMPLES" \
        --keep_ratio "$KEEP_RATIO" \
        --batch_size 1 --max_length "$MAXLEN" \
        --layer_type "$LAYER_TYPE" --target_layers "$TARGET_LAYERS" \
        --output_dir "$CKPT" --log_dir ./logs \
        --device cuda --dtype bfloat16 --seed "$SEED" --perlayer || exit 1
    P2=$(find "$CKPT" -maxdepth 1 -type d -name 'phase2_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
    MASKS_DIR=$(find "$P2" -type d -name masks | head -1); echo "$MASKS_DIR" > "$CKPT/MASKS_DIR"
fi
echo "  MASKS_DIR=$MASKS_DIR"
[ -d "$MASKS_DIR" ] || { echo "❌ mask 생성 실패"; exit 1; }

echo "############ SEAL S2 — WaRP 공간 SFT (basis_coeff 만 학습) ############"
if [ -f "${WARP_OUT}/sft_config.json" ]; then
    echo "[skip] 이미 존재: $WARP_OUT"
else
    $PY -m seal.train_sft \
        --model_path "$MODEL" \
        --task_data_path "$TASK_JSON" \
        --num_train_samples 0 \
        --selected_indices "$SEL_JSON" \
        --max_length "$MAXLEN" \
        --epochs "$EPOCHS" --learning_rate "$LR" \
        --weight_decay "$WD" --warmup_ratio "$WARMUP" \
        --lr_scheduler_type cosine --max_grad_norm 1.0 \
        --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
        --gradient_checkpointing --seed "$SEED" \
        --use_warp \
        --basis_dir "$BASIS_DIR" --masks_dir "$MASKS_DIR" \
        --layer_type "$LAYER_TYPE" --target_layers "$TARGET_LAYERS" \
        --output_dir "$WARP_OUT" || exit 1
fi
$PY -c "
import json; c=json.load(open('${WARP_OUT}/sft_config.json'))
assert c['use_warp'] is True and c['mode']=='WaRP', c
assert c['select_meta']['num_selected']==5978, c['select_meta']
print('[check] WaRP 모드 OK · 선택 데이터', c['num_train_samples'], '샘플 · warp', {k:v for k,v in c['warp'].items() if k!='stats'})"

if [ "${SKIP_UPLOAD:-0}" != "1" ]; then
    echo "############ HF 업로드 + 검증 ############"
    $PY scripts/revision/upload_and_prune.py --cell_dir "$WARP_OUT" --repo_id "$REPO_ID" || exit 1
    echo "  → https://huggingface.co/$REPO_ID"
fi
echo "############ ✅ 완료 ($(date '+%F %T')) ############"
