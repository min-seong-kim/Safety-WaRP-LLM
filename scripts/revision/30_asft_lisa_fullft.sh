#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  실험 1 — AsFT / Lisa 를 **full-parameter** 로 (2026-09-19, 사용자 지시)
#
#  논문 개정본 revisioning_wsr.tex 의 Table 2(498-505) / Table 4(817-824) 에서
#  AsFT·Lisa 행이 `0.00` 플레이스홀더로 비어 있다. 두 기법은 full-parameter 블록에
#  놓여 있는데 기존 측정치는 전부 LoRA(α=16/32) 라 그 자리에 넣을 수 없다.
#  그래서 같은 동작점의 full-param 판을 새로 학습한다.
#
#  8 (모델, 태스크) × 2 기법 = 16 셀
#    llama2_7b(gsm8k) llama2_13b(gsm8k) llama31_8b(math) llama32_3b(math)
#    qwen25_7b(gsm8k) gemma2_9b(gsm8k)  llama2_7b(medqa) llama2_7b(arc)
#
#  하이퍼파라미터
#    · 기법별  : AsFT λ=1.0 / Lisa ρ=1.0·align100·ft900  (revisioning_wsr.tex:1089-1090)
#    · 학습공통: 3ep · eff.batch 16 · max_len 1024 · seed 42 · bf16 · cosine
#                wd 0.01 · warmup 0.1 · lr 은 모델별(common.sh; **gemma 만 1e-5**)
#    · 안전데이터: circuit_breakers (출발 모델이 CB 로 정렬됐으므로)
#
#  ⚠️ AsFT full-param 은 원 논문에 없는 일반화다(ΔW = W − W₀).
#     models/asft_baseline.AsFTFullParamRegularizer 주석 참조. λ=1.0 은 LoRA 스케일에서
#     정해진 값이라, 학습 로그의 `asft_reg` 가 L_SFT 대비 타당한지 확인해야 한다.
#
#  사용:
#    bash scripts/revision/30_asft_lisa_fullft.sh              # 전체 (재개 가능)
#    DRY_RUN=1 bash scripts/revision/30_asft_lisa_fullft.sh    # 명령만 출력
#    CELLS="llama2_7b:gsm8k:asft" bash scripts/revision/30_asft_lisa_fullft.sh
#
#  재개: 셀마다 .done / .uploaded 마커를 남긴다. 다시 돌리면 이어서 간다.
#  ⚠️ 실행 중인 이 파일을 편집하지 말 것 (bash 는 바이트 오프셋으로 읽는다).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_DIR="$PWD"
source scripts/revision/common.sh

PY="${PY:-python}"
# ⚠️ common.sh 가 OUT_ROOT / PUSH_TO_HUB / LOG_ROOT 를 **이미 기본값으로 채운 뒤**라서
#    여기서 ${OUT_ROOT:-...} 로 쓰면 내 기본값이 절대 적용되지 않는다(2026-09-19 실측:
#    출력이 outputs/revision 으로 가고 업로드가 꺼진 채 시작됐다). 전용 변수명을 쓴다.
OUT_ROOT="${EXP1_OUT_ROOT:-$REPO_DIR/outputs/asft_lisa_fullft}"
LOG_DIR="${EXP1_LOG_DIR:-$REPO_DIR/logs/asft_lisa_fullft}"
PUSH_TO_HUB="${EXP1_PUSH_TO_HUB:-1}"
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
DRY_RUN="${DRY_RUN:-0}"
SAFETY=cb
SAFETY_JSON="$(safety_json $SAFETY)"

# 기법별 하이퍼파라미터 (개정본 부록 Table 8 과 일치)
ASFT_LAMBDA="${ASFT_LAMBDA_REG:-1.0}"
LISA_RHO_V="${LISA_RHO_FULL:-1.0}"
LISA_ALIGN="${LISA_ALIGNMENT_STEP:-100}"
LISA_FT="${LISA_FINETUNE_STEP:-900}"

# (모델키:태스크) 8 조합 — 사용자 확정 2026-09-19
COMBOS="${COMBOS:-llama2_7b:gsm8k llama2_13b:gsm8k llama31_8b:math llama32_3b:math qwen25_7b:gsm8k gemma2_9b:gsm8k llama2_7b:medqa llama2_7b:arc}"
METHODS_RUN="${METHODS_RUN:-asft lisa}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

# EXP1_REPO_SUFFIX: 같은 설정을 다시 돌려 재현성을 볼 때 리포명 충돌을 피한다(예: "_rerun").
#   ⚠️ 출력 경로(EXP1_OUT_ROOT)도 같이 바꿔야 한다. out_dir 은 하이퍼파라미터를 담지 않아서
#      같은 경로에 돌리면 .done 때문에 조용히 건너뛴다(CLAUDE.md).
REPO_SUFFIX="${EXP1_REPO_SUFFIX:-}"
repo_for() {  # <model> <task> <method>
  local mtag; mtag="$(hf_model_tag "$1")"
  case "$3" in
    asft) echo "${HF_NAMESPACE}/${mtag}-CB_SSFT-asft_${2}_lambda${ASFT_LAMBDA}_fullft_lr${FULL_LR}${REPO_SUFFIX}" ;;
    lisa) echo "${HF_NAMESPACE}/${mtag}-CB_SSFT-lisa_${2}_rho${LISA_RHO_V}_fullft_lr${FULL_LR}${REPO_SUFFIX}" ;;
  esac
}

echo "════════════════════════════════════════════════════════════════"
echo "  실험 1: AsFT / Lisa full-parameter — 16 셀"
echo "  조합   : $COMBOS"
echo "  기법   : $METHODS_RUN   (AsFT λ=$ASFT_LAMBDA / Lisa ρ=$LISA_RHO_V)"
echo "  출력   : $OUT_ROOT"
echo "  업로드 : PUSH_TO_HUB=$PUSH_TO_HUB  ns=$HF_NAMESPACE"
[ "$DRY_RUN" = "1" ] && echo "  *** DRY RUN ***"
echo "════════════════════════════════════════════════════════════════"

DONE_N=0; SKIP_N=0; FAIL_N=0; FAILED=()

for combo in $COMBOS; do
  mkey="${combo%%:*}"; task="${combo##*:}"
  model_cfg "$mkey" || { echo "[!] 알 수 없는 모델: $mkey"; continue; }
  TASK_JSON="$(task_json "$task")"
  [ -f "$TASK_JSON" ] || { echo "[!] 태스크 JSON 없음: $TASK_JSON"; continue; }

  # micro-batch: 모델별 MB_FULL, grad_accum 은 16/MB_FULL (effective batch 16 불변식)
  MB="$MB_FULL"; GA=$(( EFFECTIVE_BATCH / MB ))
  [ $(( MB * GA )) -eq "$EFFECTIVE_BATCH" ] || { echo "[!] $mkey: MB_FULL=$MB 가 16의 약수가 아님"; continue; }

  # 13B 는 anchor / V·W₀ 를 CPU 로 내린다 (full-param + 여분 사본 2벌 = OOM).
  #   단 TRAIN_ONLY_TARGETS=1 이면 학습 대상이 ~68% 로 줄어 grads/AdamW/anchor 가 모두 작아지고
  #   (13B 기준 약 119GB) GPU 에 다 올라간다 → offload 불필요(스트리밍 비용 제거).
  EXTRA_ASFT=(); EXTRA_LISA=()
  # WSR-Tune 은 basis_coeff(= q,k,v,up,down) 만 학습한다. 공정 비교를 위해 같은 범위로 맞춘다.
  #   TRAIN_ONLY_TARGETS=1 → o_proj / gate_proj / embed / lm_head 까지 동결.
  if [ "${TRAIN_ONLY_TARGETS:-0}" = "1" ]; then
    EXTRA_ASFT+=(--train_only_targets)
    EXTRA_LISA+=(--train_only_targets "$TARGET_MODULES_CSV")
  fi
  if [ "$mkey" = "llama2_13b" ] && [ "${TRAIN_ONLY_TARGETS:-0}" != "1" ]; then
    EXTRA_ASFT+=(--asft_offload_cpu)
    EXTRA_LISA+=(--anchor_device cpu)
  fi

  for method in $METHODS_RUN; do
    CELL="$OUT_ROOT/$mkey/$task/$method"
    REPO="$(repo_for "$mkey" "$task" "$method")"
    mkdir -p "$CELL"
    LOG="$LOG_DIR/${mkey}_${task}_${method}.log"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  $mkey / $task / $method"
    echo "    출발 : $ALIGNED_CB"
    echo "    lr=$FULL_LR  micro=$MB x accum=$GA  ep=$EPOCHS"
    echo "    repo : $REPO"
    echo "────────────────────────────────────────────────────────────"

    if [ -f "$CELL/.done" ]; then
      echo "  [skip] 학습 완료됨"; SKIP_N=$((SKIP_N+1))
    elif [ "$DRY_RUN" = "1" ]; then
      echo "  [dry-run] method=$method task_json=$TASK_JSON"
      continue
    else
      ST=$(date +%s)
      if [ "$method" = "asft" ]; then
        "$PY" asft/finetune_asft_full.py \
          --model_path "$ALIGNED_CB" --base_model "$BASE" \
          --task_data_path "$TASK_JSON" \
          --output_dir "$CELL/model" \
          --asft_lambda_reg "$ASFT_LAMBDA" --asft_check_equiv \
          --target_modules "$TARGET_MODULES_CSV" \
          --learning_rate "$FULL_LR" --epochs "$EPOCHS" \
          --batch_size "$MB" --grad_accum "$GA" \
          --weight_decay "$FULL_WEIGHT_DECAY" --warmup_ratio "$FULL_WARMUP_RATIO" \
          --lr_scheduler_type "$FULL_SCHEDULER" --max_grad_norm "$MAX_GRAD_NORM" \
          --max_length "$MAX_LENGTH" --seed "$SEED" --logging_steps 20 \
          "${EXTRA_ASFT[@]}" >"$LOG" 2>&1
      else
        "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
          --model_path "$ALIGNED_CB" --no_lora \
          --task_data_path "$TASK_JSON" \
          --output_dir "$CELL/model" \
          --safety_data_path "$SAFETY_JSON" --guide_data_num "$SAFETY_SAMPLES" \
          --rho "$LISA_RHO_V" --alignment_step "$LISA_ALIGN" --finetune_step "$LISA_FT" \
          --learning_rate "$FULL_LR" --epochs "$EPOCHS" \
          --batch_size "$MB" --grad_accum "$GA" \
          --weight_decay "$FULL_WEIGHT_DECAY" --warmup_ratio "$FULL_WARMUP_RATIO" \
          --lr_scheduler_type "$FULL_SCHEDULER" --max_grad_norm "$MAX_GRAD_NORM" \
          --max_length "$MAX_LENGTH" --seed "$SEED" --logging_steps 20 \
          "${EXTRA_LISA[@]}" >"$LOG" 2>&1
      fi
      RC=$?
      EL=$(( $(date +%s) - ST ))
      if [ $RC -ne 0 ] || [ ! -f "$CELL/model/config.json" ]; then
        echo "  ❌ 학습 실패 (rc=$RC, ${EL}s) — 로그: $LOG"
        tail -15 "$LOG" | sed 's/^/     | /'
        FAIL_N=$((FAIL_N+1)); FAILED+=("$mkey/$task/$method"); continue
      fi
      echo "$CELL/model" > "$CELL/MODEL_DIR"
      echo "$REPO" > "$CELL/REPO"
      touch "$CELL/.done"
      echo "  ✅ 학습 완료 ($((EL/60))분 $((EL%60))초)"
      DONE_N=$((DONE_N+1))
    fi

    # ── 업로드 ──────────────────────────────────────────────────────────
    if [ "$PUSH_TO_HUB" = "1" ] && [ -f "$CELL/.done" ] && [ ! -f "$CELL/.uploaded" ]; then
      echo "  ▶ 업로드 → $REPO"
      "$PY" scripts/revision/upload_and_prune.py --cell_dir "$CELL" --repo_id "$REPO" \
            >>"$LOG" 2>&1 \
        && { touch "$CELL/.uploaded"; echo "  ✅ 업로드 완료"; } \
        || { echo "  ⚠️ 업로드 실패 — 로그: $LOG"; tail -8 "$LOG" | sed 's/^/     | /'; }
    fi
  done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  학습 완료 $DONE_N · 건너뜀 $SKIP_N · 실패 $FAIL_N"
[ ${#FAILED[@]} -gt 0 ] && printf '  실패: %s\n' "${FAILED[@]}"
echo "════════════════════════════════════════════════════════════════"
echo "ALL_CELLS_FINISHED"
