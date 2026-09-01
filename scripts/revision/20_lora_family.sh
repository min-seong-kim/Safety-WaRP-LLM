#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 20 — LoRA 계열 6종
#
#    lora      Vanilla LoRA          finetune_gsm8k_lora.py --method lora
#    asft      AsFT (λ=1.0)          finetune_gsm8k_lora.py --method asft
#    safelora  SafeLoRA (thr=0.3)    finetune_gsm8k_lora.py --method safe_lora
#    lisa      LISA (ρ=1.0)          gsm8k_eval/finetune_gsm8k_lisa.py
#    salora    SaLoRA                finetune_gsm8k_salora.py
#    wsr_lora  WSR-LoRA              wsr-lora/wsr_lora.py --reparam
#
#  ── 예산 정합 (전 기법 공통) ──────────────────────────────────────────────
#    r=16, alpha=32, dropout=0.05, targets={q,k,v,up,down},
#    epochs 3, effective batch 16, max_len 1024, warmup 0.03, weight_decay 0.0,
#    cosine, max_grad_norm 1.0, seed 42, bf16.
#    lr = 3e-4 (gsm8k/math/medqa/arc) · 7e-5 (agnews)   ← 사용자 지정
#    기법 간 차이가 "안전 메커니즘" 하나로만 남게 하려는 것이다.
#
#  ── 주의 ─────────────────────────────────────────────────────────────────
#  * WSR-LoRA 는 rebuttal 에 적힌 **PiSSA 초기화 + Ã=A U** 변형이다
#    (W_0 = W_res + B_0A_0, B_0 = P_r Λ_r^{1/2}). finetune_gsm8k_lora.py 의
#    `--method wsr_lora`(옛 element-wise product-mask)와 다른 것이므로 쓰지 않는다.
#    Stage 02 의 Phase 1 basis 가 **필수**다.
#  * SaLoRA 원저자 설정은 alpha=r=16 / {q,v} 지만, 다른 열과 예산을 맞추려고
#    alpha=32 / 5모듈로 돌린다. 논문 표에 각주로 "budget-matched, 원저자 권장설정과
#    다름" 을 남길 것.
#  * SafeLoRA / AsFT 는 base 모델과 aligned 모델을 **동시에** 로드해 V=W_aligned−W_base
#    를 만든다. base 는 gated repo(meta-llama/*, google/gemma-*) 이므로 HF 토큰이 필요하고,
#    13B 는 float32 로 두 벌 = CPU RAM 이 100GB 넘게 필요하다.
#    RAM 이 모자라면 SAFELORA_LOAD_DTYPE=bfloat16 으로 낮출 것(정밀도는 조금 떨어진다).
#
#  사용:
#    bash scripts/revision/20_lora_family.sh
#    METHODS="lora safelora" MODELS=llama2_7b bash scripts/revision/20_lora_family.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/20_lora_family_${TS}.log") 2>&1

SAFELORA_LOAD_DTYPE="${SAFELORA_LOAD_DTYPE:-float32}"
ASFT_STORE_DTYPE="${ASFT_STORE_DTYPE:-float32}"

preflight
print_plan

_any=0
for m in lora asft safelora lisa salora wsr_lora; do has_method "$m" && _any=1; done
(( _any )) || { log "LoRA 계열 기법이 METHODS 에 없다 — 종료"; exit 0; }

echo ""
echo "  AsFT λ          : $ASFT_LAMBDA_REG"
echo "  SafeLoRA thr    : $SAFELORA_THRESHOLD  (load dtype=$SAFELORA_LOAD_DTYPE)"
echo "  LISA            : ρ=$LISA_RHO align_step=$LISA_ALIGNMENT_STEP ft_step=$LISA_FINETUNE_STEP"
echo "  SaLoRA          : salora/salora_lora.py · r_s=$SALORA_R_S r_t=$SALORA_R_T init=$SALORA_INIT_MODE n_harmful=$SALORA_N_HARMFUL n_task=$SALORA_N_TASK"
echo "  WSR-LoRA ρ      : $KEEP_RATIO  (Phase1 basis 필요)"

for safety in $SAFETY_SETS; do
  SAFE_DATA="$(safety_json "$safety")"

  for mkey in $MODELS; do
    safety_applies "$safety" "$mkey" || continue
    model_cfg "$mkey" || { FAILED_CELLS+=("$safety/$mkey (unknown model)"); continue; }
    ALIGNED="$(aligned_for "$mkey" "$safety")"
    if [[ "$ALIGNED" == /* && ! -f "$ALIGNED/config.json" ]]; then
      warn "[$safety/$mkey] 출발 모델 없음: $ALIGNED → 01_ssft_bt.sh 먼저. 건너뜀."
      FAILED_CELLS+=("$safety/$mkey (aligned missing)"); continue
    fi
    accum="$(accum_for "$MB_LORA")" || { FAILED_CELLS+=("$safety/$mkey (batch)"); continue; }

    # WSR-LoRA 용 Phase 1 basis
    BASIS_PTR="$CKPT_ROOT/warp/$safety/$mkey/BASIS_DIR"
    BASIS_DIR=""; [[ -s "$BASIS_PTR" ]] && BASIS_DIR="$(cat "$BASIS_PTR")"
    # dry-run 에서는 아직 basis 가 없는 게 정상이므로 빈 문자열 대신 자리표시자를 보여준다.
    [[ "$DRY_RUN" == "1" && -z "$BASIS_DIR" ]] && BASIS_DIR="<02_warp_basis_mask.sh 산출물>"

    for task in $(tasks_for_model "$mkey" "$safety"); do
      TASK_DATA="$(task_json "$task")"
      if [[ "$TASK_DATA" == "__UNKNOWN__" || ! -f "$TASK_DATA" ]]; then
        warn "[$safety/$mkey/$task] 태스크 데이터 없음: $TASK_DATA. 건너뜀."
        FAILED_CELLS+=("$safety/$mkey/$task (task data)"); continue
      fi
      LR="$(lora_lr "$task")"
      TAG="$safety/$mkey/$task lr=$LR batch=${MB_LORA}x${accum}"

      # ═══════════════ Vanilla LoRA ═══════════════
      if want_cell "$safety" "$mkey" "$task" lora; then
        odir="$(out_dir "$safety" "$mkey" "$task" lora)"
        run_cell "$odir" "lora  $TAG" -- \
          "$PY" finetune_gsm8k_lora.py --method lora \
            --model_name "$ALIGNED" --output_dir "$odir" \
            --task_data_path "$TASK_DATA" --task_samples "$TASK_SAMPLES" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$LR" --epochs "$EPOCHS" \
            --batch_size "$MB_LORA" --gradient_accumulation_steps "$accum" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$LORA_WARMUP_RATIO" --weight_decay "$LORA_WEIGHT_DECAY" \
            --seed "$SEED" --dtype "$DTYPE" --gradient_checkpointing --save_merged_model
        post_cell "$odir" "$safety" "$mkey" "$task" lora
      fi

      # ═══════════════ AsFT ═══════════════
      if want_cell "$safety" "$mkey" "$task" asft; then
        odir="$(out_dir "$safety" "$mkey" "$task" asft)"
        run_cell "$odir" "asft(λ=$ASFT_LAMBDA_REG)  $TAG" -- \
          "$PY" finetune_gsm8k_lora.py --method asft \
            --model_name "$ALIGNED" --output_dir "$odir" \
            --task_data_path "$TASK_DATA" --task_samples "$TASK_SAMPLES" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
            --asft_base_model "$BASE" --asft_aligned_model "$ALIGNED" \
            --asft_lambda_reg "$ASFT_LAMBDA_REG" --asft_store_dtype "$ASFT_STORE_DTYPE" \
            --asft_check_equiv \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$LR" --epochs "$EPOCHS" \
            --batch_size "$MB_LORA" --gradient_accumulation_steps "$accum" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$LORA_WARMUP_RATIO" --weight_decay "$LORA_WEIGHT_DECAY" \
            --seed "$SEED" --dtype "$DTYPE" --gradient_checkpointing --save_merged_model
        post_cell "$odir" "$safety" "$mkey" "$task" asft
      fi

      # ═══════════════ SafeLoRA ═══════════════
      if want_cell "$safety" "$mkey" "$task" safelora; then
        odir="$(out_dir "$safety" "$mkey" "$task" safelora)"
        run_cell "$odir" "safelora(thr=$SAFELORA_THRESHOLD)  $TAG" -- \
          "$PY" finetune_gsm8k_lora.py --method safe_lora \
            --model_name "$ALIGNED" --output_dir "$odir" \
            --task_data_path "$TASK_DATA" --task_samples "$TASK_SAMPLES" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
            --safelora_base_model "$BASE" --safelora_aligned_model "$ALIGNED" \
            --safelora_select_type threshold --safelora_threshold "$SAFELORA_THRESHOLD" \
            --safelora_load_dtype "$SAFELORA_LOAD_DTYPE" \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$LR" --epochs "$EPOCHS" \
            --batch_size "$MB_LORA" --gradient_accumulation_steps "$accum" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$LORA_WARMUP_RATIO" --weight_decay "$LORA_WEIGHT_DECAY" \
            --seed "$SEED" --dtype "$DTYPE" --gradient_checkpointing --save_merged_model
        post_cell "$odir" "$safety" "$mkey" "$task" safelora
      fi

      # ═══════════════ LISA ═══════════════
      # LISA 는 alignment step 과 finetune step 을 번갈아 돈다. 안전 데이터는
      # circuit_breakers/beavertails 의 거부응답(llama3_output)을 정답으로 학습한다.
      if want_cell "$safety" "$mkey" "$task" lisa; then
        odir="$(out_dir "$safety" "$mkey" "$task" lisa)"
        run_cell "$odir" "lisa(ρ=$LISA_RHO)  $TAG" -- \
          "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
            --model_path "$ALIGNED" --output_dir "$odir" \
            --task_data_path "$TASK_DATA" --task_samples "$TASK_SAMPLES" --num_eval_samples 0 \
            --safety_data_path "$SAFE_DATA" --guide_data_num "$SAFETY_SAMPLES" \
            --rho "$LISA_RHO" \
            --alignment_step "$LISA_ALIGNMENT_STEP" --finetune_step "$LISA_FINETUNE_STEP" \
            --lora --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$LR" --epochs "$EPOCHS" \
            --batch_size "$MB_LORA" --grad_accum "$accum" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$LORA_WARMUP_RATIO" --weight_decay "$LORA_WEIGHT_DECAY" \
            --lr_scheduler_type "$LORA_SCHEDULER" --max_grad_norm "$MAX_GRAD_NORM" \
            --seed "$SEED" --bf16 --gradient_checkpointing --report_to none
        post_cell "$odir" "$safety" "$mkey" "$task" lisa
      fi

      # ═══════════════ SaLoRA ═══════════════
      if want_cell "$safety" "$mkey" "$task" salora; then
        odir="$(out_dir "$safety" "$mkey" "$task" salora)"
        # 구현체: salora/salora_lora.py + salora/salora_impl.py (사용자 지정 기준 구현).
        #   · --gsm8k_json 은 이름과 달리 임의의 {"question","response"} JSON 을 받는다
        #     (pw.GSM8KDataset — WSR-LoRA 와 같은 로더라 프롬프트가 다른 arm 과 동일하다).
        #   · 두 파일이 서로를 상대경로로 import 하므로 PYTHONPATH 에 salora/ 와 wsr-lora/ 가 필요하다.
        #   · 이 러너에는 gradient_checkpointing 옵션이 없다(원본 그대로). B200 에서는 문제없다.
        #   · weight_decay 는 러너 안에서 0.0 으로 고정돼 있다 (= LORA_WEIGHT_DECAY 와 동일).
        run_cell "$odir" "salora(r_s=$SALORA_R_S,r_t=$SALORA_R_T)  $TAG" -- \
          env PYTHONPATH="$REPO_DIR/salora:$REPO_DIR/wsr-lora:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
          "$PY" salora/salora_lora.py \
            --model_name "$ALIGNED" --output_dir "$odir" \
            --safety_data "$SAFE_DATA" --response_field llama3_output \
            --gsm8k_json "$TASK_DATA" --train_samples "$TASK_SAMPLES" \
            --target_modules "$TARGET_MODULES_CSV" \
            --rank "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --r_s "$SALORA_R_S" --r_t "$SALORA_R_T" --init_mode "$SALORA_INIT_MODE" \
            --n_harmful "$SALORA_N_HARMFUL" --n_task "$SALORA_N_TASK" \
            --salora_max_tokens "$SALORA_MAX_TOKENS" \
            --lr "$LR" --epochs "$EPOCHS" \
            --batch_size "$MB_LORA" --grad_accum "$accum" \
            --max_length "$MAX_LENGTH" --warmup_ratio "$LORA_WARMUP_RATIO" \
            --dtype "$DTYPE" --seed "$SEED"
        post_cell "$odir" "$safety" "$mkey" "$task" salora
      fi

      # ═══════════════ WSR-LoRA ═══════════════
      if want_cell "$safety" "$mkey" "$task" wsr_lora; then
        odir="$(out_dir "$safety" "$mkey" "$task" wsr_lora)"
        if [[ "$DRY_RUN" != "1" && ( -z "$BASIS_DIR" || ! -d "$BASIS_DIR" ) ]]; then
          warn "[$safety/$mkey/$task] WSR-LoRA 에 필요한 Phase 1 basis 가 없다 → 02_warp_basis_mask.sh 먼저. 건너뜀."
          FAILED_CELLS+=("wsr_lora/$safety/$mkey/$task (no basis)")
        else
          # --gsm8k_json 은 이름과 달리 임의의 {"question","response"} JSON 을 받는다
          # (pissa_wsr_lora.GSM8KDataset). 다른 arm 과 같은 파일을 그대로 넘긴다.
          run_cell "$odir" "wsr_lora(ρ=$KEEP_RATIO)  $TAG" -- \
            "$PY" wsr-lora/wsr_lora.py \
              --model_name "$ALIGNED" --output_dir "$odir" \
              --safety_data "$SAFE_DATA" --safety_samples "$SAFETY_SAMPLES" \
              --gsm8k_json "$TASK_DATA" --train_samples "$TASK_SAMPLES" \
              --basis_dir "$BASIS_DIR" --basis_samples "$SAFETY_SAMPLES" --reparam \
              --rho "$KEEP_RATIO" --mask_B 1 --mask_A 1 \
              --target_modules "$TARGET_MODULES_CSV" \
              --rank "$LORA_R" --alpha "${WSR_LORA_ALPHA:-$LORA_ALPHA}" --dropout "$LORA_DROPOUT" \
              --lr "$LR" --epochs "$EPOCHS" \
              --batch_size "$MB_LORA" --grad_accum "$accum" \
              --max_length "$MAX_LENGTH" \
              --warmup_ratio "$LORA_WARMUP_RATIO" --weight_decay "$LORA_WEIGHT_DECAY" \
              --scheduler cosine --dtype "$DTYPE" --seed "$SEED"
          post_cell "$odir" "$safety" "$mkey" "$task" wsr_lora
        fi
      fi
    done
    # 이 (안전데이터, 모델) 의 WSR 계열이 전부 끝났으면 basis/mask 를 지운다.
    prune_basis_if_done "$safety" "$mkey"
  done
done

print_failures
