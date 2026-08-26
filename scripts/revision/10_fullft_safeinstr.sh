#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 10 — Full FT / SafeInstr  (full-parameter)
#
#    fullft    : 방어 없는 full-parameter SFT. 논문의 "Full Params FT" 기준선이고,
#                Δ_S / Δ_D 를 재는 원점이다. RESTA / SafeDelta 의 입력이기도 하다.
#    safeinstr : 같은 SFT 에 안전 데이터를 섞는다.
#                논문 §4.1 "safety examples corresponding to 10% of the downstream
#                training set" → --safety_mix_ratio 0.1
#
#  러너: finetune_task_full_params.py
#    이 스크립트는 agnews_eval/finetune_agnews_full_params.py 의 토큰화/콜레이터/
#    모델로딩/안전혼합을 **import 해서** 쓰고, 데이터만 data/local_task_dataset.py 의
#    (question, response) 페어로 바꾼다. 즉 LoRA 계열 러너 · WaRP Phase 3 과
#    프롬프트 문자열·chat template·-100 마스킹 규칙이 전부 같다.
#
#  하이퍼파라미터(사용자 지정): epochs 3, effective batch 16, lr 5e-5,
#    weight_decay 0.01, warmup 0.1, cosine, max_len 1024, max_grad_norm 1.0,
#    seed 42, bf16.
#
#  사용:
#    bash scripts/revision/10_fullft_safeinstr.sh
#    METHODS=fullft MODELS=llama2_7b SAFETY_SETS=cb bash scripts/revision/10_fullft_safeinstr.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/10_fullft_safeinstr_${TS}.log") 2>&1

preflight
print_plan

has_method fullft || has_method safeinstr || { log "fullft/safeinstr 둘 다 METHODS 에 없다 — 종료"; exit 0; }

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
    accum="$(accum_for "$MB_FULL")" || { FAILED_CELLS+=("$safety/$mkey (batch)"); continue; }

    for task in $(tasks_for_model "$mkey" "$safety"); do
      TASK_DATA="$(task_json "$task")"
      if [[ "$TASK_DATA" == "__UNKNOWN__" || ! -f "$TASK_DATA" ]]; then
        warn "[$safety/$mkey/$task] 태스크 데이터 없음: $TASK_DATA → 00_prepare.sh 먼저. 건너뜀."
        FAILED_CELLS+=("$safety/$mkey/$task (task data)"); continue
      fi

      # ═══════════ Full FT ═══════════
      if want_cell "$safety" "$mkey" "$task" fullft; then
        odir="$(out_dir "$safety" "$mkey" "$task" fullft)"
        cmd=( "$PY" finetune_task_full_params.py
              --model_path "$ALIGNED"
              --task_data_path "$TASK_DATA"
              --task_name "$task"
              --task_samples "$TASK_SAMPLES"
              --output_dir "$odir"
              --learning_rate "$FULL_LR"
              --epochs "$EPOCHS"
              --batch_size "$MB_FULL" --grad_accum "$accum"
              --max_length "$MAX_LENGTH"
              --weight_decay "$FULL_WEIGHT_DECAY"
              --warmup_ratio "$FULL_WARMUP_RATIO"
              --lr_scheduler_type "$FULL_SCHEDULER"
              --max_grad_norm "$MAX_GRAD_NORM"
              --seed "$SEED" --bf16 --gradient_checkpointing
              --report_to none )
        run_cell "$odir" "fullft  $safety/$mkey/$task  lr=$FULL_LR batch=${MB_FULL}x${accum}" -- "${cmd[@]}"
        post_cell "$odir" "$safety" "$mkey" "$task" fullft
      fi

      # ═══════════ SafeInstr ═══════════
      if want_cell "$safety" "$mkey" "$task" safeinstr; then
        odir="$(out_dir "$safety" "$mkey" "$task" safeinstr)"
        cmd=( "$PY" finetune_task_full_params.py
              --model_path "$ALIGNED"
              --task_data_path "$TASK_DATA"
              --task_name "$task"
              --task_samples "$TASK_SAMPLES"
              --output_dir "$odir"
              --safety_data_path "$SAFE_DATA"
              --safety_mix_ratio "$SAFEINSTR_RATIO"
              --learning_rate "$FULL_LR"
              --epochs "$EPOCHS"
              --batch_size "$MB_FULL" --grad_accum "$accum"
              --max_length "$MAX_LENGTH"
              --weight_decay "$FULL_WEIGHT_DECAY"
              --warmup_ratio "$FULL_WARMUP_RATIO"
              --lr_scheduler_type "$FULL_SCHEDULER"
              --max_grad_norm "$MAX_GRAD_NORM"
              --seed "$SEED" --bf16 --gradient_checkpointing
              --report_to none )
        run_cell "$odir" "safeinstr($SAFEINSTR_RATIO)  $safety/$mkey/$task  lr=$FULL_LR" -- "${cmd[@]}"
        post_cell "$odir" "$safety" "$mkey" "$task" safeinstr
      fi
    done
  done
done

print_failures
