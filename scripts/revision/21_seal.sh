#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 21 — SEAL (Safety-Enhanced Aligned LLM finetuning via bilevel data selection)
#
#  3단계로 돈다:
#    S1   selector 학습 (LoRA)            seal/train_selector.py  → <name>_softmax.pt
#    S1.5 top-p 선택                      seal/select_data.py     → selected_topNN.json
#    S2   선택된 데이터로 full-param SFT   seal/train_sft.py       → 최종 모델
#
#  ⚠️ **인덱스 정합**: selector 가 뱉는 인덱스는 downstream 데이터의 **행 순서**에
#     묶인다. S1 과 S2 가 반드시 같은 파일 · 같은 샘플 수를 봐야 한다.
#     여기서는 둘 다 --task_data_path <같은 JSON> · 샘플수 0(전체) 이므로 안전하다.
#
#  ⚠️ SEAL 의 S2 는 논문대로 **full-parameter** SFT 다. LoRA 4종과 예산이 다르므로
#     표에서 LoRA 열과 직접 비교하지 말고 별도 열로 보고할 것.
#     S2 하이퍼파라미터는 Full FT arm 과 동일하게 맞춘다
#     (lr 5e-5, wd 0.01, warmup 0.1, cosine, eff.batch 16, 3 epoch).
#
#  ⚠️ selector 산출물(seal/ckpt/**)은 .gitignore 대상이라 git 으로 넘어가지 않는다.
#
#  사용:
#    bash scripts/revision/21_seal.sh
#    MODELS=llama2_7b SAFETY_SETS=cb bash scripts/revision/21_seal.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT" "$SEAL_CKPT_ROOT"
exec > >(tee -a "$LOG_ROOT/21_seal_${TS}.log") 2>&1

preflight
print_plan

has_method seal || { log "seal 이 METHODS 에 없다 — 종료"; exit 0; }

TOPP_PCT=$("$PY" -c "print(int(round($SEAL_TOPP*100)))")
echo ""
echo "  SEAL top-p     : $SEAL_TOPP (top${TOPP_PCT})"
echo "  selector epochs: $SEAL_SEL_EPOCHS"
echo "  S2 (full-param): lr=$FULL_LR wd=$FULL_WEIGHT_DECAY warmup=$FULL_WARMUP_RATIO epochs=$EPOCHS"
echo "  selector ckpt  : $SEAL_CKPT_ROOT"

for safety in $SAFETY_SETS; do
  SAFE_DATA="$(safety_json "$safety")"

  for mkey in $MODELS; do
    safety_applies "$safety" "$mkey" || continue
    model_cfg "$mkey" || { FAILED_CELLS+=("$safety/$mkey (unknown model)"); continue; }
    ALIGNED="$(aligned_for "$mkey" "$safety")"
    if [[ "$ALIGNED" == /* && ! -f "$ALIGNED/config.json" ]]; then
      warn "[$safety/$mkey] 출발 모델 없음: $ALIGNED → 01_ssft_bt.sh 먼저. 건너뜀."
      FAILED_CELLS+=("seal/$safety/$mkey (aligned missing)"); continue
    fi
    accum_full="$(accum_for "$MB_FULL")" || { FAILED_CELLS+=("$safety/$mkey (batch)"); continue; }

    for task in $(tasks_for_model "$mkey" "$safety"); do
      TASK_DATA="$(task_json "$task")"
      if [[ "$TASK_DATA" == "__UNKNOWN__" || ! -f "$TASK_DATA" ]]; then
        warn "[$safety/$mkey/$task] 태스크 데이터 없음: $TASK_DATA. 건너뜀."
        FAILED_CELLS+=("seal/$safety/$mkey/$task (task data)"); continue
      fi

      cell_wanted "$safety" "$mkey" "$task" seal || { log "[skip] seal  $safety/$mkey/$task  (논문/rebuttal 에 이미 있음)"; continue; }
      SEL_DIR="$SEAL_CKPT_ROOT/$safety/$mkey"
      mkdir -p "$SEL_DIR"
      SEL_NAME="${task}_selector"
      SEL_PT="$SEL_DIR/${SEL_NAME}_softmax.pt"
      SEL_JSON="$SEL_DIR/${task}_selected_top${TOPP_PCT}.json"
      odir="$(out_dir "$safety" "$mkey" "$task" seal)"

      if is_done "$odir"; then log "[skip] seal  $safety/$mkey/$task  (이미 완료)"; continue; fi

      # ───────── S1: selector ─────────
      if [[ -f "$SEL_PT" ]]; then
        log "[skip] seal-S1 selector  $safety/$mkey/$task  (이미 존재: $SEL_PT)"
      else
        deadline_passed && { log "[deadline] 마감 초과 — 시작하지 않는다"; continue; }
        hdr "seal-S1 selector  $safety/$mkey/$task  batch=$MB_LORA"
        if [[ "$DRY_RUN" == "1" ]]; then
          echo "  [dry-run] $PY -m seal.train_selector --model_path $ALIGNED \\"
          echo "      --safety_data_path $SAFE_DATA --task_data_path $TASK_DATA --num_ft_samples $TASK_SAMPLES \\"
          echo "      --max_length $MAX_LENGTH --epochs $SEAL_SEL_EPOCHS --batch_size $MB_LORA --lora \\"
          echo "      --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT \\"
          echo "      --lora_target_modules ${TARGET_MODULES_LIST[*]} \\"
          echo "      --out_dir $SEL_DIR --selector_name $SEL_NAME --seed $SEED"
        else
          "$PY" -m seal.train_selector \
              --model_path "$ALIGNED" \
              --safety_data_path "$SAFE_DATA" \
              --task_data_path "$TASK_DATA" \
              --num_ft_samples "$TASK_SAMPLES" \
              --max_length "$MAX_LENGTH" \
              --epochs "$SEAL_SEL_EPOCHS" \
              --batch_size "$MB_LORA" \
              --lora \
              --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
              --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
              --seed "$SEED" \
              --out_dir "$SEL_DIR" --selector_name "$SEL_NAME" \
              2>&1 | tee "$SEL_DIR/${SEL_NAME}_s1.log"
          rc=${PIPESTATUS[0]}
          if (( rc != 0 )) || [[ ! -f "$SEL_PT" ]]; then
            warn "[fail rc=$rc] seal-S1  $safety/$mkey/$task — 로그: $SEL_DIR/${SEL_NAME}_s1.log"
            FAILED_CELLS+=("seal-S1/$safety/$mkey/$task")
            [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
            continue
          fi
        fi
      fi

      # ───────── S1.5: top-p 선택 ─────────
      if [[ -f "$SEL_JSON" ]]; then
        log "[skip] seal-S1.5 select  $safety/$mkey/$task  (이미 존재)"
      elif [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] $PY -m seal.select_data --selector_path $SEL_PT --topp $SEAL_TOPP --out $SEL_JSON"
      else
        hdr "seal-S1.5 select top-$SEAL_TOPP  $safety/$mkey/$task"
        "$PY" -m seal.select_data --selector_path "$SEL_PT" --topp "$SEAL_TOPP" --out "$SEL_JSON"
        if [[ ! -f "$SEL_JSON" ]]; then
          warn "[fail] seal-S1.5  $safety/$mkey/$task"
          FAILED_CELLS+=("seal-S1.5/$safety/$mkey/$task")
          [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
          continue
        fi
      fi

      # ───────── S2: 선택 데이터로 full-param SFT ─────────
      run_cell "$odir" "seal-S2 SFT  $safety/$mkey/$task  lr=$FULL_LR batch=${MB_FULL}x${accum_full}" -- \
        "$PY" -m seal.train_sft \
          --model_path "$ALIGNED" \
          --task_data_path "$TASK_DATA" \
          --num_train_samples "$TASK_SAMPLES" \
          --selected_indices "$SEL_JSON" \
          --max_length "$MAX_LENGTH" \
          --epochs "$EPOCHS" \
          --learning_rate "$FULL_LR" \
          --weight_decay "$FULL_WEIGHT_DECAY" \
          --warmup_ratio "$FULL_WARMUP_RATIO" \
          --lr_scheduler_type "$FULL_SCHEDULER" \
          --max_grad_norm "$MAX_GRAD_NORM" \
          --batch_size "$MB_FULL" --grad_accum "$accum_full" \
          --gradient_checkpointing \
          --seed "$SEED" \
          --output_dir "$odir"
      post_cell "$odir" "$safety" "$mkey" "$task" seal
    done
  done
done

print_failures
