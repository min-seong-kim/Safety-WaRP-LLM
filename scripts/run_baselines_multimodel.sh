#!/usr/bin/env bash
#
# ════════════════════════════════════════════════════════════════════════════
#  Safety-Finetuning baseline 5종 × 모델 5종 통합 러너
#     methods : LISA / SafeLoRA / AsFT / SaLoRA / SEAL
#     models  : llama2-13b-chat, llama3.2-3b-it, llama3.1-8b-it,
#               qwen2.5-7b-it, gemma2-9b-it
#
#  기존 llama2-7b-chat 실험(scripts/run_lisa_safelora_asft_qa.sh,
#  run_salora_matched.sh, seal/scripts/run_all.sh)의 동작점을 그대로 이식하고,
#  모델별로 달라져야 하는 것(출발 모델 / micro-batch)만 레지스트리로 뺐다.
#
#  ── 공정성 규약 ───────────────────────────────────────────────────────────
#  * LoRA 4종(LISA/SafeLoRA/AsFT/SaLoRA)은 **동일 예산**:
#      r=16, alpha=32, dropout=0.05, targets={q,k,v,up,down},
#      effective batch 16, 3 epochs, max_len 1024, cosine, warmup 0.03,
#      weight_decay 0, seed 42, bf16
#    → micro-batch × grad_accum 은 모델 크기에 맞춰 바꾸되 **곱은 항상 16**.
#  * SEAL 은 논문대로 full-param SFT 라 예산이 다르다(별도 lr/wd/warmup).
#    LoRA 열과 직접 비교하지 말고 "SEAL 열" 로 따로 보고할 것.
#  * 안전 데이터는 전부 circuit_breakers(4994) 로 통일.
#
#  ── 사용 ─────────────────────────────────────────────────────────────────
#    bash scripts/run_baselines_multimodel.sh                     # 전체
#    MODELS=llama31_8b METHODS="asft safelora" bash scripts/run_baselines_multimodel.sh
#    DRY_RUN=1 bash scripts/run_baselines_multimodel.sh           # 명령만 출력
#    STOP_AFTER_SSFT=1 bash scripts/run_baselines_multimodel.sh   # Stage 0 까지만
#    PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/...
#
#  완료된 run 은 건너뛴다(summary.json / finetune_config.json / sft_config.json).
#  중간에 죽어도 그대로 재실행하면 이어서 간다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# ⚠️ SLURM 환경에서는 CUDA_VISIBLE_DEVICES 를 절대 여기서 설정하지 말 것 (CLAUDE.md).
#    스케줄러가 넣어준다. SLURM 이 없는 박스에서만 아래를 주석 해제.
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODELS="${MODELS:-llama2_13b llama32_3b llama31_8b qwen25_7b gemma2_9b}"
METHODS="${METHODS:-lisa safelora asft salora seal}"
LRS="${LRS:-3e-4}"                      # LoRA 4종 공통 (SEAL 은 SEAL_LR 사용)
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/baselines_multimodel}"
SAFETY_DATA="${SAFETY_DATA:-$REPO_DIR/data/circuit_breakers_train.json}"
DRY_RUN="${DRY_RUN:-0}"
STOP_AFTER_SSFT="${STOP_AFTER_SSFT:-0}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_NAMESPACE="${HF_NAMESPACE:-}"

# ── matched 예산 (LoRA 4종 공통) ──
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
EFFECTIVE_BATCH=16
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
SEED="${SEED:-42}"
TARGET_MODULES_CSV="q_proj,k_proj,v_proj,up_proj,down_proj"
TARGET_MODULES_LIST=(q_proj k_proj v_proj up_proj down_proj)
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"

# ── method 고유 (matched 예산 밖. 각 논문 기본값) ──
SAFETY_SAMPLES="${SAFETY_SAMPLES:-4994}"
SAFELORA_THRESHOLD="${SAFELORA_THRESHOLD:-0.35}"
LISA_RHO="${LISA_RHO:-1.0}"
LISA_ALIGNMENT_STEP="${LISA_ALIGNMENT_STEP:-100}"
LISA_FINETUNE_STEP="${LISA_FINETUNE_STEP:-900}"
ASFT_LAMBDA_REG="${ASFT_LAMBDA_REG:-1.0}"
SALORA_RANK_SAFE="${SALORA_RANK_SAFE:-32}"
SALORA_RANK_UTIL="${SALORA_RANK_UTIL:-32}"
SALORA_CALIB_SAMPLES="${SALORA_CALIB_SAMPLES:-128}"
SALORA_CALIB_BS="${SALORA_CALIB_BS:-2}"
SALORA_NITER="${SALORA_NITER:-20}"
# SEAL 은 full-param SFT → 자체 예산
SEAL_TOPP="${SEAL_TOPP:-0.8}"
SEAL_SEL_EPOCHS="${SEAL_SEL_EPOCHS:-2}"
SEAL_SFT_EPOCHS="${SEAL_SFT_EPOCHS:-3}"
SEAL_LR="${SEAL_LR:-5e-5}"
SEAL_WEIGHT_DECAY="${SEAL_WEIGHT_DECAY:-0.01}"
SEAL_WARMUP_RATIO="${SEAL_WARMUP_RATIO:-0.1}"

# ── Phase 0 (SSFT) 기본값: 업로드된 llama2-7b SSFT 와 동일 ──
SSFT_LR="${SSFT_LR:-5e-5}"

# ════════════════════ 모델 레지스트리 ════════════════════
# 필드: BASE | ALIGNED(=안전정렬 출발모델, 비면 Stage 0 에서 생성) | TASK | MICRO_BATCH
#   MICRO_BATCH × GRAD_ACCUM = 16 이 되도록 GRAD_ACCUM 은 자동 계산한다.
#   MICRO_BATCH 는 A6000 48GB 기준 보수적 추정치 — OOM 이면 절반으로 줄이면 된다
#   (곱이 16 으로 유지되므로 결과는 동일하다).
model_cfg() {
  case "$1" in
    llama2_13b)
      BASE="meta-llama/Llama-2-13b-chat-hf"
      ALIGNED="${LLAMA2_13B_ALIGNED:-wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5}"
      TASK="${LLAMA2_13B_TASK:-gsm8k}"
      MICRO_BATCH="${LLAMA2_13B_MB:-2}" ;;
    llama32_3b)
      BASE="meta-llama/Llama-3.2-3B-Instruct"
      ALIGNED="${LLAMA32_3B_ALIGNED:-kmseong/llama3_2_3b-instruct-SSFT-lr5e-5}"
      TASK="${LLAMA32_3B_TASK:-math}"
      MICRO_BATCH="${LLAMA32_3B_MB:-8}" ;;
    llama31_8b)
      BASE="meta-llama/Llama-3.1-8B-Instruct"
      ALIGNED="${LLAMA31_8B_ALIGNED:-kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5}"
      TASK="${LLAMA31_8B_TASK:-math}"
      MICRO_BATCH="${LLAMA31_8B_MB:-4}" ;;
    qwen25_7b)
      BASE="Qwen/Qwen2.5-7B-Instruct"
      ALIGNED="${QWEN25_7B_ALIGNED:-wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5}"
      TASK="${QWEN25_7B_TASK:-gsm8k}"
      MICRO_BATCH="${QWEN25_7B_MB:-4}" ;;
    gemma2_9b)
      BASE="google/gemma-2-9b-it"
      # ⚠️ 이 모델만 SSFT lr 이 3e-5 다 (나머지는 5e-5)
      ALIGNED="${GEMMA2_9B_ALIGNED:-wvnvwn/gemma-2-9b-it-ssft-lr3e-5}"
      TASK="${GEMMA2_9B_TASK:-gsm8k}"
      MICRO_BATCH="${GEMMA2_9B_MB:-2}" ;;
    *) echo "알 수 없는 모델 키: $1 (llama2_13b|llama32_3b|llama31_8b|qwen25_7b|gemma2_9b)" >&2
       return 1 ;;
  esac
  GRAD_ACCUM=$(( EFFECTIVE_BATCH / MICRO_BATCH ))
  if (( MICRO_BATCH * GRAD_ACCUM != EFFECTIVE_BATCH )); then
    echo "MICRO_BATCH($MICRO_BATCH) 가 EFFECTIVE_BATCH($EFFECTIVE_BATCH) 의 약수가 아닙니다" >&2
    return 1
  fi
  return 0
}

# ── task → 로컬 JSON 경로 (gsm8k 는 HF 에서 직접 로드하므로 빈 문자열) ──
task_path() {
  case "$1" in
    gsm8k)  echo "" ;;
    math)   ls -1 "$REPO_DIR"/data/math_train_task_*.json 2>/dev/null | sort | tail -1 ;;
    arc)    echo "$REPO_DIR/data/arc_challenge_train_task_1119.json" ;;
    medqa)  echo "$REPO_DIR/data/medqa_train_task_10178.json" ;;
    sst2)   echo "$REPO_DIR/data/sst2_train_8k_seed42.json" ;;
    agnews) echo "$REPO_DIR/data/agnews_train_8k_seed42.json" ;;
    *)      echo "__UNKNOWN__" ;;
  esac
}

has_method() { [[ " $METHODS " == *" $1 "* ]]; }

run() {  # DRY_RUN 이면 출력만
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'; return 0
  fi
  "$@"
}

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUTPUT_ROOT" seal/ckpt
exec > >(tee -a "logs/baselines_multimodel_${TS}.log") 2>&1

# ════════════════════ 사전 점검 ════════════════════
[[ -f "$SAFETY_DATA" ]] || { echo "안전 데이터 없음: $SAFETY_DATA" >&2; exit 1; }
if [[ "$PUSH_TO_HUB" == "1" && -z "$HF_NAMESPACE" ]]; then
  echo "PUSH_TO_HUB=1 이면 HF_NAMESPACE 가 필요합니다" >&2; exit 1
fi
for mkey in $MODELS; do
  model_cfg "$mkey" || exit 1
  tp="$(task_path "$TASK")"
  [[ "$tp" == "__UNKNOWN__" ]] && { echo "[$mkey] 알 수 없는 task: $TASK" >&2; exit 1; }
  if [[ "$TASK" != "gsm8k" && -z "$tp" ]]; then
    echo "[$mkey] task '$TASK' 데이터가 없습니다." >&2
    [[ "$TASK" == "math" ]] && echo "  → python scripts/prepare_math_task_data.py 를 먼저 실행하세요." >&2
    exit 1
  fi
  if [[ -n "$tp" && ! -f "$tp" ]]; then echo "[$mkey] task data 없음: $tp" >&2; exit 1; fi
done

echo "════════════════════════════════════════════════════════════════"
echo " Safety-FT baselines × multi-model     ts=${TS}"
echo "   models  : $MODELS"
echo "   methods : $METHODS      lrs(LoRA): $LRS   (SEAL lr: $SEAL_LR)"
echo "   matched : r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT "\
"eff_batch=$EFFECTIVE_BATCH epochs=$EPOCHS targets=$TARGET_MODULES_CSV"
echo "   safety  : $SAFETY_DATA ($SAFETY_SAMPLES)"
echo "   output  : $OUTPUT_ROOT"
[[ "$DRY_RUN" == "1" ]] && echo "   *** DRY RUN ***"
echo "════════════════════════════════════════════════════════════════"

# ════════════════════ Stage 0: 안전정렬(SSFT) 출발 모델 ════════════════════
# LoRA/SEAL 전부 "이미 안전정렬된 모델을 downstream 으로 미세조정할 때 안전성이
# 무너지는가" 를 보는 실험이므로, 출발점이 안전정렬 모델이어야 한다.
# 기존 llama2-7b 실험의 출발점은 kmseong/llama2_7b-chat-Safety-FT-lr5e-5 였다.
declare -A ALIGNED_OF
for mkey in $MODELS; do
  model_cfg "$mkey" || exit 1
  if [[ -n "$ALIGNED" ]]; then
    echo "[Stage 0][$mkey] 안전정렬 모델 재사용: $ALIGNED"
    ALIGNED_OF[$mkey]="$ALIGNED"
    continue
  fi
  ssft_out="$REPO_DIR/checkpoints/ssft_${mkey}_lr${SSFT_LR}"
  if [[ -f "$ssft_out/config.json" ]]; then
    echo "[Stage 0][$mkey] 이미 SSFT 완료 — skip ($ssft_out)"
  else
    echo "──────────────────────────────────────────────────────────────"
    echo "[Stage 0][$mkey] Phase 0 SSFT 시작: $BASE  (lr=$SSFT_LR)  $(date +%H:%M:%S)"
    mkdir -p "$ssft_out"
    run "$PY" models/phase0_SSFT.py "$SAFETY_DATA" \
        --model_name "$BASE" --lr "$SSFT_LR" \
        --output_dir "$ssft_out" --no_wandb \
        2>&1 | tee "$ssft_out/ssft.log"
  fi
  ALIGNED_OF[$mkey]="$ssft_out"
done

if [[ "$STOP_AFTER_SSFT" == "1" ]]; then
  echo "STOP_AFTER_SSFT=1 → Stage 0 에서 종료"; exit 0
fi

# ════════════════════ Stage 1: 기법별 학습 ════════════════════
for mkey in $MODELS; do
  model_cfg "$mkey" || exit 1
  ALIGNED="${ALIGNED_OF[$mkey]}"
  TASK_DATA="$(task_path "$TASK")"

  # gsm8k → HF 로드, 그 외 → 로컬 JSON. 모든 러너가 같은 플래그를 쓴다.
  task_args=()
  [[ -n "$TASK_DATA" ]] && task_args=(--task_data_path "$TASK_DATA")

  echo ""
  echo "██████ $mkey  ($BASE)"
  echo "   aligned start : $ALIGNED"
  echo "   task          : $TASK   ${TASK_DATA:-(HF openai/gsm8k)}"
  echo "   batch         : ${MICRO_BATCH} × ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"

  for lr in $LRS; do

    # ═══════════ LISA ═══════════
    if has_method lisa; then
      out_dir="$OUTPUT_ROOT/lisa/${mkey}_${TASK}_lr${lr}"
      if [[ -f "$out_dir/finetune_config.json" ]]; then
        echo "[LISA $mkey/$TASK lr=$lr] 이미 완료 — skip"
      else
        echo "── [LISA $mkey/$TASK lr=$lr] $(date +%H:%M:%S)"
        mkdir -p "$out_dir"
        push=(); [[ "$PUSH_TO_HUB" == "1" ]] && push=(--upload_name \
          "${HF_NAMESPACE}/${mkey}-${TASK}-lisa-r${LORA_R}-a${LORA_ALPHA}-lr${lr}-cb")
        run "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
            --model_path "$ALIGNED" --output_dir "$out_dir" \
            "${task_args[@]}" --num_eval_samples 0 \
            --safety_data_path "$SAFETY_DATA" --guide_data_num "$SAFETY_SAMPLES" \
            --rho "$LISA_RHO" \
            --alignment_step "$LISA_ALIGNMENT_STEP" --finetune_step "$LISA_FINETUNE_STEP" \
            --lora --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
            --lr_scheduler_type cosine --seed "$SEED" --bf16 \
            --gradient_checkpointing --report_to none \
            "${push[@]}" 2>&1 | tee "$out_dir/run.log"
      fi
    fi

    # ═══════════ SafeLoRA ═══════════
    if has_method safelora; then
      out_dir="$OUTPUT_ROOT/safelora/${mkey}_${TASK}_lr${lr}"
      if [[ -f "$out_dir/summary.json" ]]; then
        echo "[SafeLoRA $mkey/$TASK lr=$lr] 이미 완료 — skip"
      else
        echo "── [SafeLoRA $mkey/$TASK lr=$lr] $(date +%H:%M:%S)"
        mkdir -p "$out_dir"
        push=(); [[ "$PUSH_TO_HUB" == "1" ]] && push=(--push_to_hub --hf_repo_id \
          "${HF_NAMESPACE}/${mkey}-${TASK}-safelora-r${LORA_R}-a${LORA_ALPHA}-lr${lr}-thr${SAFELORA_THRESHOLD}-cb")
        run "$PY" finetune_gsm8k_lora.py --method safe_lora \
            --model_name "$ALIGNED" --output_dir "$out_dir" \
            "${task_args[@]}" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers all \
            --safelora_base_model "$BASE" --safelora_aligned_model "$ALIGNED" \
            --safelora_select_type threshold --safelora_threshold "$SAFELORA_THRESHOLD" \
            --safelora_load_dtype float32 \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
            --seed "$SEED" --dtype bfloat16 --gradient_checkpointing --save_merged_model \
            "${push[@]}" 2>&1 | tee "$out_dir/run.log"
      fi
    fi

    # ═══════════ AsFT ═══════════
    if has_method asft; then
      out_dir="$OUTPUT_ROOT/asft/${mkey}_${TASK}_lr${lr}"
      if [[ -f "$out_dir/summary.json" ]]; then
        echo "[AsFT $mkey/$TASK lr=$lr] 이미 완료 — skip"
      else
        echo "── [AsFT $mkey/$TASK lr=$lr] $(date +%H:%M:%S)"
        mkdir -p "$out_dir"
        push=(); [[ "$PUSH_TO_HUB" == "1" ]] && push=(--push_to_hub --hf_repo_id \
          "${HF_NAMESPACE}/${mkey}-${TASK}-asft-r${LORA_R}-a${LORA_ALPHA}-lr${lr}-lam${ASFT_LAMBDA_REG}-cb")
        run "$PY" finetune_gsm8k_lora.py --method asft \
            --model_name "$ALIGNED" --output_dir "$out_dir" \
            "${task_args[@]}" \
            --target_modules "$TARGET_MODULES_CSV" \
            --layer_type "$LAYER_TYPES" --target_layers all \
            --asft_base_model "$BASE" --asft_aligned_model "$ALIGNED" \
            --asft_lambda_reg "$ASFT_LAMBDA_REG" --asft_store_dtype float32 \
            --asft_check_equiv \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" --weight_decay "$WEIGHT_DECAY" \
            --seed "$SEED" --dtype bfloat16 --gradient_checkpointing --save_merged_model \
            "${push[@]}" 2>&1 | tee "$out_dir/run.log"
      fi
    fi

    # ═══════════ SaLoRA ═══════════
    # ⚠️ 원저자 설정은 alpha=r=16 / {q,v} 인데, 여기서는 다른 열과 예산을 맞추려고
    #    alpha=32 / 5모듈로 돌린다 (run_salora_matched.sh 와 동일한 선택).
    #    논문 표에 각주로 "budget-matched, 원저자 권장설정과 다름" 을 남길 것.
    if has_method salora; then
      out_dir="$OUTPUT_ROOT/salora/${mkey}_${TASK}_lr${lr}"
      if [[ -f "$out_dir/summary.json" ]]; then
        echo "[SaLoRA $mkey/$TASK lr=$lr] 이미 완료 — skip"
      else
        echo "── [SaLoRA $mkey/$TASK lr=$lr] $(date +%H:%M:%S)"
        mkdir -p "$out_dir"
        push=(); [[ "$PUSH_TO_HUB" == "1" ]] && push=(--push_to_hub --hf_repo_id \
          "${HF_NAMESPACE}/${mkey}-${TASK}-salora-r${LORA_R}-a${LORA_ALPHA}-lr${lr}-cb")
        run "$PY" finetune_gsm8k_salora.py \
            --model_name "$ALIGNED" --output_dir "$out_dir" \
            "${task_args[@]}" --safety_data_path "$SAFETY_DATA" \
            --layer_type "$LAYER_TYPES" --target_modules "$TARGET_MODULES_CSV" \
            --target_layers all \
            --salora_rank_safe "$SALORA_RANK_SAFE" --salora_rank_util "$SALORA_RANK_UTIL" \
            --salora_calib_samples "$SALORA_CALIB_SAMPLES" \
            --salora_calib_batch_size "$SALORA_CALIB_BS" --salora_niter "$SALORA_NITER" \
            --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
            --learning_rate "$lr" --epochs "$EPOCHS" \
            --batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" --seed "$SEED" \
            --dtype bfloat16 --gradient_checkpointing \
            "${push[@]}" 2>&1 | tee "$out_dir/run.log"
      fi
    fi

  done  # lr

  # ═══════════ SEAL (full-param, lr 스윕 밖) ═══════════
  # Stage 1 selector → Stage 1.5 top-p 선택 → Stage 2 full-param SFT.
  # ⚠️ selector 인덱스는 downstream 데이터의 **행 순서**에 묶인다.
  #    Stage 1 과 Stage 2 가 반드시 같은 데이터(같은 파일/같은 num_samples)여야 한다.
  if has_method seal; then
    topp_pct=$("$PY" -c "print(int($SEAL_TOPP*100))")
    sel_name="${mkey}_${TASK}_selector"
    sel_pt="seal/ckpt/${sel_name}_softmax.pt"
    sel_json="seal/ckpt/${mkey}_${TASK}_selected_top${topp_pct}.json"
    seal_out="$OUTPUT_ROOT/seal/${mkey}_${TASK}"

    if [[ -f "$sel_pt" ]]; then
      echo "[SEAL $mkey/$TASK] selector 이미 존재 — skip"
    else
      echo "── [SEAL $mkey/$TASK] Stage 1 selector $(date +%H:%M:%S)"
      run "$PY" -m seal.train_selector \
          --model_path "$ALIGNED" --safety_data_path "$SAFETY_DATA" \
          "${task_args[@]}" \
          --max_length "$MAX_LENGTH" --epochs "$SEAL_SEL_EPOCHS" \
          --batch_size "$MICRO_BATCH" --lora \
          --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
          --lora_target_modules "${TARGET_MODULES_LIST[@]}" \
          --out_dir seal/ckpt --selector_name "$sel_name"
    fi

    if [[ -f "$sel_json" ]]; then
      echo "[SEAL $mkey/$TASK] 선택 인덱스 이미 존재 — skip"
    else
      echo "── [SEAL $mkey/$TASK] Stage 1.5 top-${SEAL_TOPP} 선택"
      run "$PY" -m seal.select_data \
          --selector_path "$sel_pt" --topp "$SEAL_TOPP" --out "$sel_json"
    fi

    if [[ -f "$seal_out/sft_config.json" ]]; then
      echo "[SEAL $mkey/$TASK] SFT 이미 완료 — skip"
    else
      echo "── [SEAL $mkey/$TASK] Stage 2 full-param SFT $(date +%H:%M:%S)"
      mkdir -p "$seal_out"
      run "$PY" -m seal.train_sft \
          --model_path "$ALIGNED" --selected_indices "$sel_json" \
          "${task_args[@]}" \
          --max_length "$MAX_LENGTH" --epochs "$SEAL_SFT_EPOCHS" \
          --learning_rate "$SEAL_LR" --weight_decay "$SEAL_WEIGHT_DECAY" \
          --warmup_ratio "$SEAL_WARMUP_RATIO" \
          --batch_size "$MICRO_BATCH" --grad_accum "$GRAD_ACCUM" \
          --gradient_checkpointing --seed "$SEED" \
          --output_dir "$seal_out" 2>&1 | tee "$seal_out/run.log"
    fi
  fi

done  # model

echo ""
echo "════════════════════ summary ════════════════════"
find "$OUTPUT_ROOT" \( -name summary.json -o -name finetune_config.json -o -name sft_config.json \) \
  | sort | sed 's/^/  /'
echo ""
echo "완료. 평가는 별도 하네스에서:"
echo "  utility : gsm8k_eval/ (gsm8k)  ·  MATH 는 최종답 'Final Answer: \$…\$' 매칭"
echo "  safety  : HarmBench ASR / beavertails-harmful 747"
