#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Original-space freeze sweep — 논문 Table 1 의 "FT (X% frozen)" 행 재현
#
#  reparameterization 없이(U=V=I) **원래 weight 공간**에서 safety importance 를 재고,
#  상위 ρ 비율의 weight 를 얼린 뒤 gsm8k 로 full-param FT 한다.
#  WSR-Tune 과의 차이는 "마스크가 어느 좌표계에서 매겨지는가" 하나뿐이다.
#
#    Phase 2 : --original_space_mask, circuit_breakers 4994, per-layer quantile
#              → 각 레이어에서 정확히 ρ 비율이 mask=1(freeze)
#    Phase 3 : --original_space_mask + 그 마스크, gsm8k full-param FT
#
#  ρ=0 기준행은 이미 있다: kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5
#  (동일 출발 모델 · 동일 동작점 lr5e-5 · 3ep · eff.batch 16 · wd 0.01 · warmup 0.1)
#
#  사용:
#    bash scripts/run_origspace_freeze_sweep.sh
#    KEEP_RATIOS="0.05 0.2" bash scripts/run_origspace_freeze_sweep.sh
#    DRY_RUN=1 bash scripts/run_origspace_freeze_sweep.sh
#    PUSH_TO_HUB=0 bash scripts/run_origspace_freeze_sweep.sh     # 업로드 생략
#
#  재개 가능: 비율마다 .done / .uploaded 마커를 남기므로 다시 돌리면 이어서 간다.
#  ⚠️ 실행 중인 이 스크립트를 편집하지 말 것 (bash 는 파일을 바이트 오프셋으로 읽는다).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_DIR="$PWD"

PY="${PY:-python}"
PHASE0_MODEL="${PHASE0_MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
KEEP_RATIOS="${KEEP_RATIOS:-0.05 0.2 0.3 0.4 0.5}"
PHASE3_DATASET="${PHASE3_DATASET:-gsm8k}"

# 동작점 — RESULTS.md 의 full-param 계열과 동일하게 맞춘다 (ρ=0 기준행과 짝이 되도록).
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
LAYER_TYPE="${LAYER_TYPE:-attn_q,attn_k,attn_v,ffn_down,ffn_up}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"
PHASE2_SAMPLES="${PHASE2_SAMPLES:-4994}"

OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/origspace_freeze}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/origspace_freeze}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
[ "$DRY_RUN" = "1" ] || exec > >(tee -a "$LOG_DIR/sweep_${TS}.log") 2>&1

# ρ → 리포 이름 태그. 0.05 → p05, 0.2 → p20
ratio_tag() { "$PY" - "$1" <<'PYEOF'
import sys
r = float(sys.argv[1])
pct = round(r * 100)
assert abs(r*100 - pct) < 1e-6, f"정수 퍼센트가 아닌 비율: {r}"
print(f"p{pct:02d}")
PYEOF
}

# 리포 이름의 모델 부분. base 라인에서 재사용하려고 변수로 뺐다(2026-09-19).
# ⚠️ base 모델에는 chat/instruct/-it 토큰이 이름에 들어가면 안 된다 — 러너들이 모델
#    참조 **문자열**로 chat template 사용 여부를 정하므로 plain 프롬프트가 깨진다.
MODEL_TAG="${MODEL_TAG:-llama2_7b-chat}"
repo_id() { echo "${HF_NAMESPACE}/${MODEL_TAG}-origspace-freeze-$1-${PHASE3_DATASET}-lr${LR}"; }

echo "════════════════════════════════════════════════════════════"
echo " Original-space freeze sweep"
echo "   출발 모델   : $PHASE0_MODEL"
echo "   리포 태그   : $MODEL_TAG"
echo "   keep ratios : $KEEP_RATIOS   (= 얼리는 비율)"
echo "   downstream  : $PHASE3_DATASET  lr=$LR  ep=$EPOCHS  eff.batch=$((BATCH_SIZE*GRAD_ACCUM))"
echo "   layer_type  : $LAYER_TYPE  ($TARGET_LAYERS)"
echo "   out         : $OUT_ROOT"
echo "   upload      : PUSH_TO_HUB=$PUSH_TO_HUB  ns=$HF_NAMESPACE"
[ "$DRY_RUN" = "1" ] && echo "   *** DRY RUN ***"
echo "════════════════════════════════════════════════════════════"

FAILED=()

for KR in $KEEP_RATIOS; do
  TAG="$(ratio_tag "$KR")" || { FAILED+=("$KR (bad ratio)"); continue; }
  CELL="$OUT_ROOT/$TAG"
  REPO="$(repo_id "$TAG")"
  mkdir -p "$CELL"

  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo "  ρ=$KR  ($TAG)  →  $REPO"
  echo "──────────────────────────────────────────────────────────"

  if [ -f "$CELL/.done" ]; then
    echo "  [skip] 이미 완료: $(cat "$CELL/MODEL_DIR" 2>/dev/null)"
  elif [ "$DRY_RUN" = "1" ]; then
    echo "  [dry-run] phase2 --original_space_mask --keep_ratio $KR"
    echo "  [dry-run] phase3 --original_space_mask --phase3_dataset $PHASE3_DATASET --utility_lr $LR"
    continue
  else
    # ── Phase 2: original-space importance → mask ──────────────────────────
    P2_MARK="$CELL/PHASE2_DIR"
    if [ -s "$P2_MARK" ] && [ -d "$(cat "$P2_MARK")" ]; then
      MASKS_DIR="$(cat "$P2_MARK")"
      echo "  [skip] Phase 2 재사용: $MASKS_DIR"
    else
      echo "  ▶ Phase 2 (importance, original space, keep_ratio=$KR)"
      "$PY" train.py \
        --phase 2 \
        --phase0_model_dir "$PHASE0_MODEL" \
        --original_space_mask \
        --dataset_phase2 circuit_breakers \
        --circuit_breakers_path ./data/circuit_breakers_train.json \
        --circuit_breakers_samples_phase2 "$PHASE2_SAMPLES" \
        --keep_ratio "$KR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --layer_type "$LAYER_TYPE" \
        --target_layers "$TARGET_LAYERS" \
        --output_dir "$CKPT_ROOT" \
        --log_dir "$LOG_DIR" \
        --device cuda --dtype "$DTYPE" --seed "$SEED" || { FAILED+=("$TAG (phase2)"); continue; }

      P2_DIR=$(find "$CKPT_ROOT" -maxdepth 1 -name "phase2_original_space_*" -type d \
                 -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
      MASKS_DIR="$P2_DIR/checkpoints/masks"
      [ -d "$MASKS_DIR" ] || { echo "  ❌ 마스크 없음: $MASKS_DIR"; FAILED+=("$TAG (no masks)"); continue; }
      echo "$MASKS_DIR" > "$P2_MARK"
      echo "    마스크: $MASKS_DIR"
    fi

    # ── Phase 3: gsm8k full-param FT, mask=1 은 동결 ────────────────────────
    echo "  ▶ Phase 3 (downstream FT, frozen=$KR)"
    "$PY" train.py \
      --phase 3 \
      --phase0_model_dir "$PHASE0_MODEL" \
      --masks_dir "$MASKS_DIR" \
      --original_space_mask \
      --phase3_dataset "$PHASE3_DATASET" --gsm8k_samples 0 \
      --epochs "$EPOCHS" \
      --utility_lr "$LR" \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$GRAD_ACCUM" \
      --warmup_ratio "$WARMUP_RATIO" \
      --lr_scheduler_type "$LR_SCHEDULER" \
      --base_weight_decay "$WEIGHT_DECAY" \
      --max_length "$MAX_LENGTH" \
      --layer_type "$LAYER_TYPE" \
      --target_layers "$TARGET_LAYERS" \
      --output_dir "$CKPT_ROOT" \
      --log_dir "$LOG_DIR" \
      --device cuda --dtype "$DTYPE" --seed "$SEED" || { FAILED+=("$TAG (phase3)"); continue; }

    P3_DIR=$(find "$CKPT_ROOT" -maxdepth 1 -name "phase3_original_space_*" -type d \
               -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    MODEL_DIR="$P3_DIR/final_model"
    [ -f "$MODEL_DIR/config.json" ] || { echo "  ❌ 모델 없음: $MODEL_DIR"; FAILED+=("$TAG (no model)"); continue; }
    echo "$MODEL_DIR" > "$CELL/MODEL_DIR"
    date -Iseconds > "$CELL/.done"
    echo "    모델: $MODEL_DIR"
  fi

  # ── 업로드 (4종 검증: 파일/크기/AutoConfig/chat_template) ────────────────
  if [ "$PUSH_TO_HUB" = "1" ] && [ ! -f "$CELL/.uploaded" ] && [ "$DRY_RUN" != "1" ]; then
    echo "  ▶ 업로드 → $REPO"
    # ⚠️ --cell_dir 은 **셀 디렉토리**다 (final_model 경로가 아니다).
    #    upload_and_prune.py 는 셀 안의 .done 을 확인하고 MODEL_DIR 파일을 읽어
    #    실제 가중치 위치를 찾는다. final_model 을 넘기면 ".done 이 없다" 로 중단된다.
    if "$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$REPO"; then
      echo "$REPO" > "$CELL/.uploaded"
    else
      echo "  ⚠️ 업로드 실패 (로컬 가중치는 그대로 남는다)"
      FAILED+=("$TAG (upload)")
    fi
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo " ⚠️ 실패: ${#FAILED[@]}건"; printf '   ✗ %s\n' "${FAILED[@]}"
else
  echo " ✅ 전부 완료"
fi
echo " 평가 대상 리포:"
for KR in $KEEP_RATIOS; do
  T="$(ratio_tag "$KR" 2>/dev/null)" && echo "   $(repo_id "$T")"
done
echo "════════════════════════════════════════════════════════════"
