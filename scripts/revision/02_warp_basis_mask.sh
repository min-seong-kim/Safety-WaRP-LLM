#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 02 — WaRP Phase 1 (safety-conditioned basis) + Phase 2 (importance mask)
#
#  WSR-Tune 과 WSR-LoRA 가 **둘 다** 이 산출물을 쓴다.
#    Phase 1 : 안전 데이터 활성 공분산 H Hᵀ = U Σ Uᵀ → layer 별 U        (basis)
#    Phase 2 : 안전 loss gradient |∂L/∂W̃| 로 layer 내 top-ρ 이진 마스크   (mask)
#
#  ── 반드시 지켜야 하는 불변식 (CLAUDE.md) ────────────────────────────────
#   * --layer_type 과 --target_layers 가 Phase 1/2/3 에서 **완전히 동일**해야 한다.
#     어긋나면 조용히 틀린 결과가 나온다. 여기서는 common.sh 의 LAYER_TYPES /
#     TARGET_LAYERS 하나만 쓰므로 구조적으로 어긋날 수 없다.
#   * Phase 2 는 --perlayer 가 **필수**다. 없으면 전역 threshold 를 쓰는 다른
#     구현(phase2_importance_whole)이 선택되어 ~50배 느리고 마스크도 달라진다.
#
#  산출물은 (safety, model) 마다 독립 디렉토리에 떨어지고, 경로를 파일로 고정한다:
#    checkpoints/revision/warp/<safety>/<model>/BASIS_DIR
#    checkpoints/revision/warp/<safety>/<model>/MASKS_DIR
#  (train.py 가 phase1_<타임스탬프> 로 디렉토리를 만들기 때문에, 뒤 스테이지가
#   "최신 디렉토리 찾기" 같은 취약한 방식에 의존하지 않도록 여기서 못박는다.)
#
#  사용:
#    bash scripts/revision/02_warp_basis_mask.sh
#    MODELS=llama2_7b SAFETY_SETS=cb bash scripts/revision/02_warp_basis_mask.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/02_warp_basis_mask_${TS}.log") 2>&1

preflight
print_plan

echo ""
echo "  layer_type    : $LAYER_TYPES"
echo "  target_layers : $TARGET_LAYERS"
echo "  keep_ratio ρ  : $KEEP_RATIO"
echo "  basis 저장    : dtype=$BASIS_SAVE_DTYPE  omit_UT=$BASIS_OMIT_UT  (fp32+UT 대비 1/4)"
echo "  출력          : $CKPT_ROOT/warp/<safety>/<model>/"

# train.py 는 phase{1,2}_<ts> 디렉토리를 만든다. base 아래에서 가장 최근 것을 집어
# 경로 파일에 고정한다.
newest_subdir() {  # <base> <prefix>
  find "$1" -maxdepth 1 -name "$2_*" -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

for safety in $SAFETY_SETS; do
  SAFE_DATA="$(safety_json "$safety")"
  [[ -f "$SAFE_DATA" ]] || { warn "안전 데이터 없음: $SAFE_DATA — $safety 건너뜀"; continue; }

  for mkey in $MODELS; do
    safety_applies "$safety" "$mkey" || continue
    model_cfg "$mkey" || { FAILED_CELLS+=("warp/$safety/$mkey (unknown model)"); continue; }
    ALIGNED="$(aligned_for "$mkey" "$safety")"

    # BT 출발모델이 아직 없으면 Phase 1 을 돌릴 수 없다.
    if [[ "$ALIGNED" == /* && ! -f "$ALIGNED/config.json" ]]; then
      warn "[$safety/$mkey] 출발 모델 없음: $ALIGNED  → 01_ssft_bt.sh 를 먼저 돌려라. 건너뜀."
      FAILED_CELLS+=("warp/$safety/$mkey (aligned missing)")
      continue
    fi

    # 이 조합에서 실제로 돌릴 WSR 셀이 하나도 없으면 basis 를 만들 이유가 없다.
    need=0
    for task in $(tasks_for_model "$mkey" "$safety"); do
      for m in wsr_tune wsr_lora; do
        want_cell "$safety" "$mkey" "$task" "$m" && need=1
      done
    done
    if (( need == 0 )); then
      log "[skip] basis/mask $safety/$mkey — 돌릴 WSR 셀이 없다"
      continue
    fi

    BASE_OUT="$CKPT_ROOT/warp/$safety/$mkey"
    mkdir -p "$BASE_OUT"
    BASIS_PTR="$BASE_OUT/BASIS_DIR"
    MASKS_PTR="$BASE_OUT/MASKS_DIR"

    # ═══════════════ Phase 1: basis ═══════════════
    if [[ -s "$BASIS_PTR" ]] && [[ -d "$(cat "$BASIS_PTR")" ]]; then
      log "[skip] phase1 $safety/$mkey — 이미 완료 ($(cat "$BASIS_PTR"))"
    else
      basis_ut_arg=""
      [[ "$BASIS_OMIT_UT" == "1" ]] && basis_ut_arg="--basis_omit_ut"
      deadline_passed && { log "[deadline] 마감 초과 — 시작하지 않는다"; continue; }
      hdr "Phase 1 (basis)  $safety/$mkey   model=$ALIGNED   batch=$MB_P12  dtype=$BASIS_SAVE_DTYPE omit_ut=$BASIS_OMIT_UT"
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] $PY train.py --phase 1 --phase0_model_dir $ALIGNED \\"
        echo "      --safety_dataset circuit_breakers --circuit_breakers_path $SAFE_DATA \\"
        echo "      --circuit_breakers_samples_phase1 $SAFETY_SAMPLES --batch_size $MB_P12 \\"
        echo "      --layer_type $LAYER_TYPES --target_layers $TARGET_LAYERS \\"
        echo "      --output_dir $BASE_OUT --log_dir $LOG_ROOT --device cuda --dtype $DTYPE --seed $SEED --no_wandb"
      else
        # --safety_dataset 은 '로더 종류'를 고르는 스위치다(circuit_breakers = prompt/llama3_output
        # 스키마의 JSON). BT 파일도 스키마가 같으므로 같은 로더를 쓰고, 실제 파일은
        # --circuit_breakers_path 로 지정한다.
        "$PY" train.py \
            --phase 1 \
            --phase0_model_dir "$ALIGNED" \
            --safety_dataset circuit_breakers \
            --circuit_breakers_path "$SAFE_DATA" \
            --circuit_breakers_samples_phase1 "$SAFETY_SAMPLES" \
            --basis_save_dtype "$BASIS_SAVE_DTYPE" $basis_ut_arg \
            --batch_size "$MB_P12" \
            --max_length "$MAX_LENGTH" \
            --layer_type "$LAYER_TYPES" \
            --target_layers "$TARGET_LAYERS" \
            --output_dir "$BASE_OUT" \
            --log_dir "$LOG_ROOT" \
            --device cuda --dtype "$DTYPE" --seed "$SEED" --no_wandb \
            --profile_json "$BASE_OUT/phase1_profile.json" \
            2>&1 | tee "$BASE_OUT/phase1.log"
        rc=${PIPESTATUS[0]}

        p1="$(newest_subdir "$BASE_OUT" phase1)"
        if (( rc == 0 )) && [[ -n "$p1" && -d "$p1/basis" ]]; then
          echo "$p1/basis" > "$BASIS_PTR"
          log "[done] phase1 $safety/$mkey → $p1/basis"
        else
          warn "[fail rc=$rc] phase1 $safety/$mkey — 로그: $BASE_OUT/phase1.log"
          FAILED_CELLS+=("phase1/$safety/$mkey")
          [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
          continue
        fi
      fi
    fi

    # ═══════════════ Phase 2: importance mask ═══════════════
    # mask 는 WSR-Tune(Phase 3) 만 쓴다. WSR-LoRA 는 basis(U) 만 필요하다.
    # 이 조합에 돌릴 wsr_tune 셀이 없으면 Phase 2 를 통째로 건너뛴다.
    need_mask=0
    for task in $(tasks_for_model "$mkey" "$safety"); do
      want_cell "$safety" "$mkey" "$task" wsr_tune && need_mask=1
    done
    if (( need_mask == 0 )); then
      log "[skip] phase2 $safety/$mkey — 돌릴 wsr_tune 셀이 없다 (WSR-LoRA 는 basis 만 쓴다)"
      continue
    fi

    if [[ -s "$MASKS_PTR" ]] && [[ -d "$(cat "$MASKS_PTR")" ]]; then
      log "[skip] phase2 $safety/$mkey — 이미 완료 ($(cat "$MASKS_PTR"))"
      continue
    fi

    BASIS_DIR=""
    [[ -s "$BASIS_PTR" ]] && BASIS_DIR="$(cat "$BASIS_PTR")"
    if [[ "$DRY_RUN" != "1" && ( -z "$BASIS_DIR" || ! -d "$BASIS_DIR" ) ]]; then
      warn "[$safety/$mkey] basis 가 없어 phase2 를 건너뛴다"
      FAILED_CELLS+=("phase2/$safety/$mkey (no basis)")
      continue
    fi

    hdr "Phase 2 (mask, ρ=$KEEP_RATIO)  $safety/$mkey   batch=$MB_P12"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  [dry-run] $PY train.py --phase 2 --phase0_model_dir $ALIGNED \\"
      echo "      --basis_dir <phase1 basis> --dataset_phase2 circuit_breakers \\"
      echo "      --circuit_breakers_path $SAFE_DATA --circuit_breakers_samples_phase2 $SAFETY_SAMPLES \\"
      echo "      --keep_ratio $KEEP_RATIO --perlayer --batch_size $MB_P12 --max_length $MAX_LENGTH \\"
      echo "      --layer_type $LAYER_TYPES --target_layers $TARGET_LAYERS --output_dir $BASE_OUT ..."
      continue
    fi

    # ⚠️ --perlayer 필수 (없으면 다른 구현이 선택되어 마스크 자체가 달라진다)
    "$PY" train.py \
        --phase 2 \
        --phase0_model_dir "$ALIGNED" \
        --basis_dir "$BASIS_DIR" \
        --dataset_phase2 circuit_breakers \
        --circuit_breakers_path "$SAFE_DATA" \
        --circuit_breakers_samples_phase2 "$SAFETY_SAMPLES" \
        --keep_ratio "$KEEP_RATIO" \
        --perlayer \
        --batch_size "$MB_P12" \
        --max_length "$MAX_LENGTH" \
        --layer_type "$LAYER_TYPES" \
        --target_layers "$TARGET_LAYERS" \
        --output_dir "$BASE_OUT" \
        --log_dir "$LOG_ROOT" \
        --device cuda --dtype "$DTYPE" --seed "$SEED" --no_wandb \
        --profile_json "$BASE_OUT/phase2_kr${KEEP_RATIO}_profile.json" \
        2>&1 | tee "$BASE_OUT/phase2_kr${KEEP_RATIO}.log"
    rc=${PIPESTATUS[0]}

    p2="$(newest_subdir "$BASE_OUT" phase2)"
    if (( rc == 0 )) && [[ -n "$p2" && -d "$p2/checkpoints/masks" ]]; then
      echo "$p2/checkpoints/masks" > "$MASKS_PTR"
      log "[done] phase2 $safety/$mkey → $p2/checkpoints/masks"
    else
      warn "[fail rc=$rc] phase2 $safety/$mkey — 로그: $BASE_OUT/phase2_kr${KEEP_RATIO}.log"
      FAILED_CELLS+=("phase2/$safety/$mkey")
      [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
    fi
  done
done

echo ""
echo "════════════════ Stage 02 요약 ════════════════"
for safety in $SAFETY_SETS; do
  for mkey in $MODELS; do
    b="$CKPT_ROOT/warp/$safety/$mkey/BASIS_DIR"
    m="$CKPT_ROOT/warp/$safety/$mkey/MASKS_DIR"
    bs="MISSING"; ms="MISSING"
    [[ -s "$b" ]] && [[ -d "$(cat "$b")" ]] && bs="ok"
    [[ -s "$m" ]] && [[ -d "$(cat "$m")" ]] && ms="ok"
    printf "  %-3s %-12s basis=%-8s mask=%-8s\n" "$safety" "$mkey" "$bs" "$ms"
  done
done

print_failures
