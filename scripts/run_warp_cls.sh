#!/usr/bin/env bash
#
# WaRP (WSR-Tune) 파이프라인 — AG News / SST-2.
#
#   Phase 1 : safety 데이터(circuit_breakers) activation → SVD basis
#   Phase 2 : 같은 safety 데이터로 gradient 중요도 → keep_ratio 만큼 freeze mask
#   Phase 3 : downstream(agnews / sst2) 학습, 중요 방향은 frozen
#
# ⚠️ Phase 1/2 는 downstream 과 무관하다(안전 모델 + 안전 데이터 + layer_type 에만 의존).
#    따라서 **한 번만** 돌려 두 태스크가 공유한다. GPU 2장은 Phase 3 에서만 병렬화된다.
#
# ⚠️ 불변식: --layer_type / --target_layers 는 Phase 1/2/3 에서 반드시 동일해야 한다.
#    어긋나면 basis·mask·학습이 서로 다른 레이어를 가리켜 조용히 틀린 결과가 나온다.
#
# 사용:
#   bash scripts/run_warp_cls.sh
#   PHASE1_BASIS_DIR=checkpoints/phase1_XXXX/basis bash scripts/run_warp_cls.sh   # Phase1 재사용
#   STOP_AFTER_MASKS=1 bash scripts/run_warp_cls.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"

# Phase 1/2/3 공통 (불변식)
LAYER_TYPE="${LAYER_TYPE:-attn_q,attn_k,attn_v,ffn_up,ffn_down}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"
KEEP_RATIO="${KEEP_RATIO:-0.1}"          # freeze ratio 10%
# ⚠️ Phase 2 는 --perlayer 여부로 **다른 구현**이 선택된다(train.py:459):
#      --perlayer 有 → phase2_importance_per_layer  : keep_ratio 를 레이어별로 적용. ~5 it/s
#      --perlayer 無 → phase2_importance_whole      : 모델 전체 일괄. ~0.1 it/s (50배 느림)
#    속도 차이도 크지만 **마스크 의미가 다르다**. 레포의 기존 스윕
#    (run_beavertails_kr_sweep.sh)과 업로드된 WaRP 모델들이 모두 per-layer 이므로
#    비교 대상과 맞추려면 --perlayer 를 유지할 것.
SAFETY_SAMPLES="${SAFETY_SAMPLES:-4994}"
P2_BATCH="${P2_BATCH:-2}"                # 과거 빠른 run 과 동일

# Phase 3
TASKS="${TASKS:-agnews sst2}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"            # effective 16
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-42}"
GPU_A="${GPU_A:-0}"                      # agnews
GPU_B="${GPU_B:-1}"                      # sst2

PHASE1_BASIS_DIR="${PHASE1_BASIS_DIR:-}"
PHASE2_MASK_DIR="${PHASE2_MASK_DIR:-}"
STOP_AFTER_MASKS="${STOP_AFTER_MASKS:-0}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
exec > >(tee -a "logs/warp_cls_${TS}.log") 2>&1

newest() { find "$1" -maxdepth 0 -type d 2>/dev/null | head -1; }
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

echo "════════════════════════════════════════════════════════════════"
echo " WaRP pipeline  ts=${TS}"
echo "   safety model : $MODEL"
echo "   safety data  : $SAFETY_DATA"
echo "   layer_type   : $LAYER_TYPE   target_layers: $TARGET_LAYERS   keep_ratio: $KEEP_RATIO"
echo "   Phase3       : tasks=$TASKS lr=$LR ep=$EPOCHS batch=${BATCH_SIZE}x${GRAD_ACCUM} (eff $((BATCH_SIZE*GRAD_ACCUM)))"
echo "   GPU          : agnews=$GPU_A  sst2=$GPU_B      free: $(free_gb)GB"
echo "════════════════════════════════════════════════════════════════"

# ═══════════ Phase 1: basis ═══════════
if [[ -z "$PHASE1_BASIS_DIR" ]]; then
  echo "──────────── Phase 1 (basis) start $(date +%H:%M:%S) ────────────"
  CUDA_VISIBLE_DEVICES="$GPU_A" "$PY" train.py --phase 1 \
      --phase0_model_dir "$MODEL" \
      --safety_dataset circuit_breakers \
      --circuit_breakers_path "$SAFETY_DATA" \
      --layer_type "$LAYER_TYPE" --target_layers "$TARGET_LAYERS" \
      --output_dir "$REPO_DIR/checkpoints" --log_dir "$REPO_DIR/logs" \
      --device cuda --dtype bfloat16 || { echo "Phase1 FAILED"; exit 1; }
  PHASE1_DIR=$(ls -dt checkpoints/phase1_* 2>/dev/null | head -1)
  PHASE1_BASIS_DIR="$PHASE1_DIR/basis"
  echo "Phase 1 완료 → $PHASE1_BASIS_DIR"
else
  echo "Phase 1 skip — 지정된 basis 사용: $PHASE1_BASIS_DIR"
fi
[[ -d "$PHASE1_BASIS_DIR" ]] || { echo "basis 디렉토리 없음: $PHASE1_BASIS_DIR"; exit 1; }

# ═══════════ Phase 2: masks ═══════════
if [[ -z "$PHASE2_MASK_DIR" ]]; then
  echo "──────────── Phase 2 (mask, keep_ratio=$KEEP_RATIO) start $(date +%H:%M:%S) ────────────"
  CUDA_VISIBLE_DEVICES="$GPU_A" "$PY" train.py --phase 2 \
      --phase0_model_dir "$MODEL" \
      --basis_dir "$PHASE1_BASIS_DIR" \
      --dataset_phase2 circuit_breakers \
      --circuit_breakers_path "$SAFETY_DATA" \
      --circuit_breakers_samples_phase2 "$SAFETY_SAMPLES" \
      --keep_ratio "$KEEP_RATIO" \
      --batch_size "$P2_BATCH" --max_length "$MAX_LENGTH" \
      --layer_type "$LAYER_TYPE" --target_layers "$TARGET_LAYERS" \
      --perlayer \
      --seed "$SEED" \
      --output_dir "$REPO_DIR/checkpoints" --log_dir "$REPO_DIR/logs" \
      --device cuda --dtype bfloat16 || { echo "Phase2 FAILED"; exit 1; }
  PHASE2_DIR=$(ls -dt checkpoints/phase2_* 2>/dev/null | head -1)
  PHASE2_MASK_DIR="$PHASE2_DIR/checkpoints/masks"
  echo "Phase 2 완료 → $PHASE2_MASK_DIR"
else
  echo "Phase 2 skip — 지정된 mask 사용: $PHASE2_MASK_DIR"
fi
[[ -d "$PHASE2_MASK_DIR" ]] || { echo "mask 디렉토리 없음: $PHASE2_MASK_DIR"; exit 1; }

if [[ "$STOP_AFTER_MASKS" == "1" ]]; then
  echo "STOP_AFTER_MASKS=1 — Phase 3 건너뜀"
  echo "  basis: $PHASE1_BASIS_DIR"
  echo "  masks: $PHASE2_MASK_DIR"
  exit 0
fi

# ═══════════ Phase 3: downstream (태스크별 GPU 병렬) ═══════════
run_phase3() {   # $1=task  $2=gpu
  local task=$1 gpu=$2
  local out="outputs/warp_cls/${task}_kr${KEEP_RATIO}_lr${LR}_ep${EPOCHS}"
  if [[ -f "$out/done.marker" ]]; then echo "[$task] Phase3 이미 완료 — skip"; return 0; fi
  mkdir -p "$out"
  local extra=()
  [[ "$task" == "agnews" ]] && extra=(--agnews_dataset_path "$REPO_DIR/data/agnews_train_8k_seed42.json" --agnews_samples 0)
  [[ "$task" == "sst2" ]]   && extra=(--sst2_dataset_path  "$REPO_DIR/data/sst2_train_8k_seed42.json")
  echo "[$task] Phase3 start (GPU $gpu) $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --phase 3 \
      --phase0_model_dir "$MODEL" \
      --basis_dir "$PHASE1_BASIS_DIR" \
      --masks_dir "$PHASE2_MASK_DIR" \
      --phase3_dataset "$task" "${extra[@]}" \
      --utility_lr "$LR" --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM" \
      --max_length "$MAX_LENGTH" --seed "$SEED" \
      --layer_type "$LAYER_TYPE" --target_layers "$TARGET_LAYERS" \
      --output_dir "$REPO_DIR/checkpoints" --log_dir "$REPO_DIR/logs" \
      --device cuda --dtype bfloat16 > "$out/run.log" 2>&1 \
    && { touch "$out/done.marker"; echo "[$task] Phase3 done $(date +%H:%M:%S)"; } \
    || { echo "[$task] Phase3 FAILED — $out/run.log 확인"; return 1; }
}

pids=()
for task in $TASKS; do
  gpu=$GPU_A; [[ "$task" == "sst2" ]] && gpu=$GPU_B
  run_phase3 "$task" "$gpu" &
  pids+=("$!")
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo ""
echo "════════════════════ summary ════════════════════"
echo "  basis: $PHASE1_BASIS_DIR"
echo "  masks: $PHASE2_MASK_DIR"
ls -dt checkpoints/phase3_* 2>/dev/null | head -4 | sed 's/^/  /'
[[ "$fail" == "0" ]] && echo "완료." || { echo "일부 실패"; exit 1; }
