#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WSR-Tune vs ActSVD mask-structure ablation (rebuttal 실험)
#   설계 문서: wsr_actsvd_ablation_spec.md
#   근거 논문: Wei et al. 2024, "Assessing the Brittleness of Safety Alignment
#             via Pruning and Low-Rank Modifications" (ActSVD = §2.1)
#
# 네 arm을 **완전히 같은 fine-tuning 파이프라인**에 태우고 좌표계와 마스크 단위만 바꾼다.
#
#   arm A : 원본 공간          + entry  mask   (논문 Table 5 재현 체크)
#   arm B : ActSVD 출력측 기저 + row    mask   ★ 핵심 비교 (rank-level freezing)
#   arm C : safety 입력측 기저 + column mask
#   arm D : safety 입력측 기저 + entry  mask   (WSR-Tune, 논문 Table 2 재현 체크)
#
# 모든 arm이 **safety 데이터(circuit_breakers)만** 사용한다 — WSR-Tune 본문과 동일한 셋업
# (safety-tuned 모델을 downstream FT 할 때 안전성이 보존되는가). utility corpus는 쓰지 않는다.
# spec §4의 utility-disentangled arm은 이 목적에서 벗어나므로 구현하지 않았다.
#
# 동결 스칼라 수는 arm 간 ±1%로 맞춰진다 (spec §2). 각 arm의 masks 디렉토리에
# budget_report.json 이 남고, 마지막에 wsr_actsvd_ablation_report.py 가 교차 검증한다.
#
# 사용법:
#   bash scripts/run_wsr_actsvd_ablation.sh                    # 전체 (기저 생성 + A/B/D/C)
#   ARMS="A B D" bash scripts/run_wsr_actsvd_ablation.sh       # spec §9 우선순위 1
#   STOP_AFTER_MASKS=1 bash scripts/run_wsr_actsvd_ablation.sh # 학습 전 artifact만 만들고 정지
#   WSR_INPUT_BASIS_DIR=/path/to/existing/basis bash scripts/...  # 기존 Phase 1 basis 재사용
#
# 완료된 단계는 건너뛰므로 중단 후 재실행하면 이어서 간다.
# ⚠️ CUDA_VISIBLE_DEVICES 를 설정하지 않는다 (스케줄러/단일 GPU가 정한다).
#
# 메모리 예산 (Llama-2-7B, q/k/v/up/down × 32 layers, bf16) — spec §6 질문에 대한 답:
#   기저 저장/상주   : 입력측 12.1 GiB, 출력측 12.1 GiB (동일! ActSVD가 더 비싸지 않다)
#                      per layer = 3·4096² + 11008² + 4096² = 188.4M 원소 (양쪽 동일)
#   Phase 1 Gram(fp32): 약 24 GiB 추가 → 모델 13.5 GiB 포함 ~38 GiB (97 GiB GPU에서 여유)
#   Phase 3          : 모델 13.5 + basis_coeff 9 + 기저 12.1 + grad 9 + AdamW 18 + mask 4.5
#                      ≈ 66 GiB + activations → batch 2 / accum 8 권장. OOM 시 BATCH=1 ACCUM=16
#                      또는 EXTRA_PHASE3_ARGS="--gradient_checkpointing".
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_DIR="$(pwd)"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "logs/wsr_actsvd_ablation_${TS}.log") 2>&1

# ═══════════════════════════ config ═══════════════════════════
PY=${PY:-/venv/hb/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

# safety-tuned 시작 모델 (논문 §4.1과 동일해야 arm A/D의 재현 체크가 의미를 가진다)
MODEL=${MODEL:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}
SAFETY_DATA=${SAFETY_DATA:-${REPO_DIR}/data/circuit_breakers_train.json}
OUT_ROOT=${OUT_ROOT:-${REPO_DIR}/outputs/wsr_actsvd_ablation}

# spec §2 Fixed setup — 모든 arm 공통
ARMS=${ARMS:-"A B D C"}                 # 우선순위 순서 (spec §9)
KEEP_RATIO=${KEEP_RATIO:-0.10}          # ρ = 10% (논문 기본값)
LAYER_TYPES=${LAYER_TYPES:-attn_q,attn_k,attn_v,ffn_up,ffn_down}
TARGET_LAYERS=${TARGET_LAYERS:-all}
PHASE3_DATASET=${PHASE3_DATASET:-gsm8k}
GSM8K_SAMPLES=${GSM8K_SAMPLES:-0}       # 0=전체
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-5}                          # 논문 Appendix A.1 downstream LR
BATCH=${BATCH:-2}
ACCUM=${ACCUM:-8}                       # eff. batch 16 (논문 Appendix A)
MAXLEN=${MAXLEN:-1024}
SEED=${SEED:-42}
DTYPE=${DTYPE:-bfloat16}
CB_SAMPLES=${CB_SAMPLES:-4994}          # basis/importance 용 circuit_breakers 샘플 수
PHASE1_BATCH=${PHASE1_BATCH:-4}         # Gram 누적은 배치 크기에 불변 → 속도만 고려
PHASE2_BATCH=${PHASE2_BATCH:-2}         # ⚠️ G = Σ_batch |∂L/∂W̃| 이라 배치 크기가 결과를 바꾼다.
                                        #    run_all_phases_integrated.sh(BATCH_SIZE=2)와 일치시켜
                                        #    arm D가 논문 Table 2를 재현하도록 한다.

# 기저 구성 옵션
BASIS_TOKEN_SCOPE=${BASIS_TOKEN_SCOPE:-all}   # all=기존 Phase 1과 동일(재현 체크 유지)
                                              # response=spec §2 문구 그대로 (응답 토큰만)
BASIS_SAVE_DTYPE=${BASIS_SAVE_DTYPE:-bfloat16}
GRAM_DTYPE=${GRAM_DTYPE:-float32}
STRUCTURED_AGG=${STRUCTURED_AGG:-l2}
STRUCTURED_RANK=${STRUCTURED_RANK:-grad}      # grad | spectral(ActSVD 원기준)

# 기존 산출물 재사용
WSR_INPUT_BASIS_DIR=${WSR_INPUT_BASIS_DIR:-}   # 비우면 새로 만든다
ACTSVD_OUTPUT_BASIS_DIR=${ACTSVD_OUTPUT_BASIS_DIR:-}
STOP_AFTER_MASKS=${STOP_AFTER_MASKS:-0}
EXTRA_PHASE3_ARGS=${EXTRA_PHASE3_ARGS:-}       # 예: "--gradient_checkpointing"

NUM_LAYER_TYPES=$(awk -F',' '{print NF}' <<< "$LAYER_TYPES")
EXPECTED_MODULES=${EXPECTED_MODULES:-$((32 * NUM_LAYER_TYPES))}

mkdir -p "$OUT_ROOT"

echo "═══════════════════════════════════════════════════════════════"
echo " WSR-Tune vs ActSVD ablation"
echo "   model        : $MODEL"
echo "   arms         : $ARMS"
echo "   keep_ratio ρ : $KEEP_RATIO"
echo "   layer types  : $LAYER_TYPES ($EXPECTED_MODULES modules expected)"
echo "   downstream   : $PHASE3_DATASET, ${EPOCHS}ep lr=$LR eff.batch=$((BATCH * ACCUM))"
echo "   token scope  : $BASIS_TOKEN_SCOPE"
echo "   out root     : $OUT_ROOT"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════ helpers ═══════════════════════════
newest_dir () {   # newest_dir <parent> <glob-prefix>
  find "$1" -maxdepth 1 -type d -name "$2*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

basis_complete () {   # basis_complete <basis_dir> <expected_side>
  local dir=$1 side=$2
  [ -f "${dir}/metadata.json" ] || return 1
  "$PY" - "$dir/metadata.json" "$EXPECTED_MODULES" "$side" <<'PYEOF'
import json, sys
meta = json.load(open(sys.argv[1]))
expected, side = int(sys.argv[2]), sys.argv[3]
ok = meta.get("num_layers_saved") == expected
ok = ok and (meta.get("basis_side", "input") == side)
raise SystemExit(0 if ok else 1)
PYEOF
}

masks_complete () {   # masks_complete <masks_dir> <arm>
  local dir=$1 arm=$2
  [ -f "${dir}/metadata.json" ] || return 1
  "$PY" - "$dir/metadata.json" "$EXPECTED_MODULES" "$arm" <<'PYEOF'
import json, os, sys
meta = json.load(open(sys.argv[1]))
expected, arm = int(sys.argv[2]), sys.argv[3]
if meta.get("ablation_arm") != arm:
    raise SystemExit(1)
root = os.path.dirname(sys.argv[1])
n = sum(len([f for f in os.listdir(os.path.join(root, d)) if f.endswith("_mask.pt")])
        for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
raise SystemExit(0 if n == expected else 1)
PYEOF
}

record () { echo "$2" > "$1"; }

# ═══════════════════════ W1: safety 입력측 기저 (arm C/D) ═══════════════════════
INPUT_BASIS_PTR="${OUT_ROOT}/basis_input.path"
if [ -n "$WSR_INPUT_BASIS_DIR" ]; then
  echo "[W1] 사용자가 지정한 입력측 basis 사용: $WSR_INPUT_BASIS_DIR"
  record "$INPUT_BASIS_PTR" "$WSR_INPUT_BASIS_DIR"
elif [ -f "$INPUT_BASIS_PTR" ] && basis_complete "$(cat "$INPUT_BASIS_PTR")" input; then
  echo "[W1] skip: $(cat "$INPUT_BASIS_PTR")"
else
  echo "[W1] safety 입력측(WSR) 기저 생성 — X_in X_in^T 고유기저"
  STAGE="${OUT_ROOT}/basis_input_runs"; mkdir -p "$STAGE"
  "$PY" train.py --phase 1 \
    --phase0_model_dir "$MODEL" \
    --safety_dataset circuit_breakers \
    --circuit_breakers_samples_phase1 "$CB_SAMPLES" \
    --basis_side input \
    --basis_token_scope "$BASIS_TOKEN_SCOPE" \
    --basis_save_dtype "$BASIS_SAVE_DTYPE" \
    --gram_dtype "$GRAM_DTYPE" \
    --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
    --batch_size "$PHASE1_BATCH" --max_length "$MAXLEN" \
    --device cuda --dtype "$DTYPE" --seed "$SEED" \
    --output_dir "$STAGE" --log_dir "${REPO_DIR}/logs" --no_wandb
  DIR="$(newest_dir "$STAGE" phase1_input_)/basis"
  basis_complete "$DIR" input || { echo "[W1] FAILED: $DIR 불완전"; exit 1; }
  record "$INPUT_BASIS_PTR" "$DIR"
  echo "[W1] done: $DIR"
fi
INPUT_BASIS="$(cat "$INPUT_BASIS_PTR")"

# ═══════════════════ W2: ActSVD 출력측 기저 (arm B) ═══════════════════
OUTPUT_BASIS_PTR="${OUT_ROOT}/basis_output.path"
if [ -n "$ACTSVD_OUTPUT_BASIS_DIR" ]; then
  echo "[W2] 사용자가 지정한 출력측 basis 사용: $ACTSVD_OUTPUT_BASIS_DIR"
  record "$OUTPUT_BASIS_PTR" "$ACTSVD_OUTPUT_BASIS_DIR"
elif [ -f "$OUTPUT_BASIS_PTR" ] && basis_complete "$(cat "$OUTPUT_BASIS_PTR")" output; then
  echo "[W2] skip: $(cat "$OUTPUT_BASIS_PTR")"
elif [[ " $ARMS " == *" B "* ]]; then
  echo "[W2] ActSVD 출력측 기저 생성 — U S V^T ≈ W X_in 의 left singular vectors"
  STAGE="${OUT_ROOT}/basis_output_runs"; mkdir -p "$STAGE"
  "$PY" train.py --phase 1 \
    --phase0_model_dir "$MODEL" \
    --safety_dataset circuit_breakers \
    --circuit_breakers_samples_phase1 "$CB_SAMPLES" \
    --basis_side output \
    --basis_token_scope "$BASIS_TOKEN_SCOPE" \
    --basis_save_dtype "$BASIS_SAVE_DTYPE" \
    --gram_dtype "$GRAM_DTYPE" \
    --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
    --batch_size "$PHASE1_BATCH" --max_length "$MAXLEN" \
    --device cuda --dtype "$DTYPE" --seed "$SEED" \
    --output_dir "$STAGE" --log_dir "${REPO_DIR}/logs" --no_wandb
  DIR="$(newest_dir "$STAGE" phase1_output_)/basis"
  basis_complete "$DIR" output || { echo "[W2] FAILED: $DIR 불완전"; exit 1; }
  record "$OUTPUT_BASIS_PTR" "$DIR"
  echo "[W2] done: $DIR"
else
  echo "[W2] skip: arm B가 ARMS에 없음"
fi
OUTPUT_BASIS="$( [ -f "$OUTPUT_BASIS_PTR" ] && cat "$OUTPUT_BASIS_PTR" || echo "" )"


# ═══════════════════════ arm 별 Phase 2 → Phase 3 ═══════════════════════
for ARM in $ARMS; do
  echo ""
  echo "───────────────────────────────────────────────────────────────"
  echo " ARM $ARM"
  echo "───────────────────────────────────────────────────────────────"
  ARM_DIR="${OUT_ROOT}/arm_${ARM}"; mkdir -p "$ARM_DIR"

  case "$ARM" in
    A)            BASIS_ARGS="" ;;
    B)            [ -n "$OUTPUT_BASIS" ] || { echo "arm B에는 출력측 basis가 필요합니다"; exit 1; }
                  BASIS_ARGS="--basis_dir $OUTPUT_BASIS" ;;
    C|D|D_perm)   BASIS_ARGS="--basis_dir $INPUT_BASIS" ;;
    *)            echo "Unknown arm: $ARM"; exit 1 ;;
  esac

  # ── Phase 2: importance → mask ────────────────────────────────────
  MASKS_PTR="${ARM_DIR}/masks.path"
  if [ -f "$MASKS_PTR" ] && masks_complete "$(cat "$MASKS_PTR")" "$ARM"; then
    echo "[P2/$ARM] skip: $(cat "$MASKS_PTR")"
  else
    STAGE="${ARM_DIR}/phase2_runs"; mkdir -p "$STAGE"
    # shellcheck disable=SC2086
    "$PY" train.py --phase 2 \
      --phase0_model_dir "$MODEL" \
      $BASIS_ARGS \
      --ablation_arm "$ARM" \
      --structured_agg "$STRUCTURED_AGG" \
      --structured_rank "$STRUCTURED_RANK" \
      --dataset_phase2 circuit_breakers \
      --circuit_breakers_path "$SAFETY_DATA" \
      --circuit_breakers_samples_phase2 "$CB_SAMPLES" \
      --keep_ratio "$KEEP_RATIO" \
      --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
      --batch_size "$PHASE2_BATCH" --max_length "$MAXLEN" \
      --device cuda --dtype "$DTYPE" --seed "$SEED" \
      --output_dir "$STAGE" --log_dir "${REPO_DIR}/logs" --no_wandb
    DIR="$(newest_dir "$STAGE" "phase2_arm${ARM}_")/checkpoints/masks"
    masks_complete "$DIR" "$ARM" || { echo "[P2/$ARM] FAILED: $DIR 불완전"; exit 1; }
    record "$MASKS_PTR" "$DIR"
    echo "[P2/$ARM] done: $DIR"
  fi
  MASKS_DIR="$(cat "$MASKS_PTR")"

  if [ "$STOP_AFTER_MASKS" = "1" ]; then
    echo "[P3/$ARM] skip: STOP_AFTER_MASKS=1"
    continue
  fi

  # ── Phase 3: downstream fine-tuning ───────────────────────────────
  MODEL_PTR="${ARM_DIR}/final_model.path"
  if [ -f "$MODEL_PTR" ] && [ -f "$(cat "$MODEL_PTR")/config.json" ]; then
    echo "[P3/$ARM] skip: $(cat "$MODEL_PTR")"
  else
    STAGE="${ARM_DIR}/phase3_runs"; mkdir -p "$STAGE"
    # shellcheck disable=SC2086
    "$PY" train.py --phase 3 \
      --phase0_model_dir "$MODEL" \
      $BASIS_ARGS \
      --masks_dir "$MASKS_DIR" \
      --ablation_arm "$ARM" \
      --phase3_dataset "$PHASE3_DATASET" \
      --gsm8k_samples "$GSM8K_SAMPLES" \
      --keep_ratio "$KEEP_RATIO" \
      --epochs "$EPOCHS" --utility_lr "$LR" \
      --batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --max_length "$MAXLEN" \
      --layer_type "$LAYER_TYPES" --target_layers "$TARGET_LAYERS" \
      --device cuda --dtype "$DTYPE" --seed "$SEED" \
      --output_dir "$STAGE" --log_dir "${REPO_DIR}/logs" --no_wandb \
      $EXTRA_PHASE3_ARGS
    DIR="$(newest_dir "$STAGE" phase3_)/final_model"
    [ -f "${DIR}/config.json" ] || { echo "[P3/$ARM] FAILED: $DIR 없음"; exit 1; }
    record "$MODEL_PTR" "$DIR"
    echo "[P3/$ARM] done: $DIR"
  fi
done

# ═══════════════════════ 리포트 ═══════════════════════
echo ""
"$PY" wsr_actsvd_ablation_report.py --root "$OUT_ROOT" --keep_ratio "$KEEP_RATIO" || true

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " 완료. 다음 단계 (spec §5, 이 저장소 밖):"
echo "   · GSM8K 5-shot  : lm-evaluation-harness"
echo "   · ASR           : HarmBench (Direct + PAP 우선, 최종본만 4-attack 전체)"
echo "   평가 결과는 wsr_actsvd_ablation_report.py --eval_json 으로 표에 합칠 수 있다."
echo "═══════════════════════════════════════════════════════════════"
