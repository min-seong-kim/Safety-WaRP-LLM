#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 12 — WSR-Tune (WaRP Phase 3)
#
#  Stage 02 가 만든 basis(U) 와 mask(M) 를 써서, 재파라미터화 공간에서
#  안전 중요 계수를 얼린 채 downstream 을 학습한다.
#      W̃ ← stopgrad(W̃⊙M) + W̃⊙(1−M),   W̃ = Vᵀ W U  (V=I)
#
#  ── 불변식 ────────────────────────────────────────────────────────────────
#   * --layer_type / --target_layers 가 Phase 1/2 와 **완전히 동일**해야 한다.
#     common.sh 의 LAYER_TYPES / TARGET_LAYERS 하나만 쓰므로 구조적으로 보장된다.
#   * --non_freeze 는 기존 sweep 전부가 쓴 표준 경로다(Phase3IncrementalLearnerNonFreeze).
#     빼면 다른 구현이 선택되어 비교가 깨진다.
#   * downstream 데이터는 --phase3_task_data_path 로 **다른 arm 과 같은 JSON**을 읽힌다.
#     (이 플래그가 없으면 Phase 3 이 태스크별 자체 로더를 쓰는데, agnews 는 셔플/
#      프롬프트 조립 방식이 baseline 러너와 달라 학습 텍스트가 어긋난다.)
#
#  하이퍼파라미터: full-param 계열과 동일 — epochs 3, effective batch 16, lr 5e-5,
#    base_weight_decay 0.01, warmup 0.1, cosine, max_grad_norm 1.0, max_len 1024,
#    seed 42, bf16, keep_ratio ρ=0.1.
#
#  사용:
#    bash scripts/revision/12_wsr_tune.sh
#    MODELS=llama2_7b SAFETY_SETS=cb bash scripts/revision/12_wsr_tune.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/12_wsr_tune_${TS}.log") 2>&1

preflight
print_plan

has_method wsr_tune || { log "wsr_tune 이 METHODS 에 없다 — 종료"; exit 0; }

echo ""
echo "  keep_ratio ρ  : $KEEP_RATIO"
echo "  layer_type    : $LAYER_TYPES   target_layers: $TARGET_LAYERS"

newest_subdir() { find "$1" -maxdepth 1 -name "$2*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

for safety in $SAFETY_SETS; do
  for mkey in $MODELS; do
    safety_applies "$safety" "$mkey" || continue
    model_cfg "$mkey" || { FAILED_CELLS+=("$safety/$mkey (unknown model)"); continue; }
    ALIGNED="$(aligned_for "$mkey" "$safety")"

    WARP_BASE="$CKPT_ROOT/warp/$safety/$mkey"
    BASIS_PTR="$WARP_BASE/BASIS_DIR"; MASKS_PTR="$WARP_BASE/MASKS_DIR"
    BASIS_DIR=""; MASKS_DIR=""
    [[ -s "$BASIS_PTR" ]] && BASIS_DIR="$(cat "$BASIS_PTR")"
    [[ -s "$MASKS_PTR" ]] && MASKS_DIR="$(cat "$MASKS_PTR")"

    # 이 조합에 돌릴 wsr_tune 셀이 하나도 없으면(=논문에 이미 있는 셀) basis/mask 도
    # 만들어지지 않는 게 정상이다. 아래 존재 검사를 그대로 태우면 가짜 실패가 기록된다.
    want_any=0
    for task in $(tasks_for_model "$mkey" "$safety"); do
      want_cell "$safety" "$mkey" "$task" wsr_tune && want_any=1
    done
    if (( want_any == 0 )); then
      log "[skip] wsr_tune $safety/$mkey — 돌릴 셀이 없다 (논문 Table 2/4/10 에 이미 있음)"
      continue
    fi

    if [[ "$DRY_RUN" != "1" ]] && { [[ ! -d "$BASIS_DIR" ]] || [[ ! -d "$MASKS_DIR" ]]; }; then
      warn "[$safety/$mkey] basis/mask 가 없다 → 02_warp_basis_mask.sh 먼저. 건너뜀."
      warn "   basis=$BASIS_DIR  mask=$MASKS_DIR"
      FAILED_CELLS+=("wsr_tune/$safety/$mkey (no basis/mask)")
      continue
    fi

    accum="$(accum_for "$MB_WARP")" || { FAILED_CELLS+=("$safety/$mkey (batch)"); continue; }

    for task in $(tasks_for_model "$mkey" "$safety"); do
      TASK_DATA="$(task_json "$task")"
      if [[ "$TASK_DATA" == "__UNKNOWN__" || ! -f "$TASK_DATA" ]]; then
        warn "[$safety/$mkey/$task] 태스크 데이터 없음: $TASK_DATA. 건너뜀."
        FAILED_CELLS+=("wsr_tune/$safety/$mkey/$task (task data)"); continue
      fi

      cell_wanted "$safety" "$mkey" "$task" wsr_tune || { log "[skip] wsr_tune  $safety/$mkey/$task  (논문/rebuttal 에 이미 있음)"; continue; }
      odir="$(out_dir "$safety" "$mkey" "$task" wsr_tune)"
      if is_done "$odir"; then log "[skip] wsr_tune  $safety/$mkey/$task  (이미 완료)"; continue; fi

      deadline_passed && { log "[deadline] 마감 초과 — 시작하지 않는다"; continue; }
      hdr "wsr_tune(ρ=$KEEP_RATIO)  $safety/$mkey/$task  lr=$FULL_LR batch=${MB_WARP}x${accum}"
      mkdir -p "$odir"

      if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] $PY train.py --phase 3 --phase0_model_dir $ALIGNED \\"
        echo "      --basis_dir $BASIS_DIR --masks_dir $MASKS_DIR \\"
        echo "      --phase3_dataset $task --phase3_task_data_path $TASK_DATA --phase3_task_samples $TASK_SAMPLES \\"
        echo "      --epochs $EPOCHS --utility_lr $FULL_LR --base_weight_decay $FULL_WEIGHT_DECAY \\"
        echo "      --warmup_ratio $FULL_WARMUP_RATIO --lr_scheduler_type $FULL_SCHEDULER \\"
        echo "      --max_grad_norm $MAX_GRAD_NORM --max_length $MAX_LENGTH \\"
        echo "      --batch_size $MB_WARP --gradient_accumulation_steps $accum \\"
        echo "      --layer_type $LAYER_TYPES --target_layers $TARGET_LAYERS \\"
        echo "      --output_dir $odir --device cuda --dtype $DTYPE --seed $SEED --non_freeze --gradient_checkpointing --no_wandb"
        continue
      fi

      "$PY" train.py \
          --phase 3 \
          --phase0_model_dir "$ALIGNED" \
          --basis_dir "$BASIS_DIR" \
          --masks_dir "$MASKS_DIR" \
          --phase3_dataset "$task" \
          --phase3_task_data_path "$TASK_DATA" \
          --phase3_task_samples "$TASK_SAMPLES" \
          --epochs "$EPOCHS" \
          --utility_lr "$FULL_LR" \
          --base_weight_decay "$FULL_WEIGHT_DECAY" \
          --warmup_ratio "$FULL_WARMUP_RATIO" \
          --lr_scheduler_type "$FULL_SCHEDULER" \
          --max_grad_norm "$MAX_GRAD_NORM" \
          --max_length "$MAX_LENGTH" \
          --batch_size "$MB_WARP" \
          --gradient_accumulation_steps "$accum" \
          --layer_type "$LAYER_TYPES" \
          --target_layers "$TARGET_LAYERS" \
          --output_dir "$odir" \
          --log_dir "$LOG_ROOT" \
          --device cuda --dtype "$DTYPE" --seed "$SEED" \
          --non_freeze --gradient_checkpointing --no_wandb \
          --profile_json "$odir/phase3_profile.json" \
          2>&1 | tee "$odir/run.log"
      rc=${PIPESTATUS[0]}

      p3="$(newest_subdir "$odir" phase3_non_freeze_)"
      final="$p3/final_model"
      if (( rc == 0 )) && [[ -n "$p3" && -f "$final/config.json" ]]; then
        # 규약: 셀 디렉토리 자체가 모델 디렉토리가 되도록 final_model 을 끌어올린다.
        # (다른 arm 은 output_dir 이 곧 모델 디렉토리라 평가 스크립트가 균일해진다.)
        mv "$final"/* "$odir"/ 2>/dev/null
        rmdir "$final" 2>/dev/null
        # metadata.json (train_seconds / peak VRAM) 은 phase3_* 안에 남으므로 옆에 복사
        [[ -f "$p3/metadata.json" ]] && cp "$p3/metadata.json" "$odir/phase3_metadata.json"
        mark_done "$odir"
        write_model_ptr "$odir" || true
        upload_cell "$safety" "$mkey" "$task" wsr_tune
        log "[done] wsr_tune  $safety/$mkey/$task → $odir"
      else
        warn "[fail rc=$rc] wsr_tune  $safety/$mkey/$task — 로그: $odir/run.log"
        warn "  OOM 이면 ${mkey^^}_MB_WARP 를 줄여라 (곱이 16 이면 결과 동일)."
        FAILED_CELLS+=("wsr_tune/$safety/$mkey/$task")
        [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
      fi
    done
    # 이 (안전데이터, 모델) 의 WSR 계열이 전부 끝났으면 basis/mask 를 지운다.
    prune_basis_if_done "$safety" "$mkey"
  done
done

print_failures
