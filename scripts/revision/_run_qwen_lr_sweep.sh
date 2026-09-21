#!/usr/bin/env bash
# Qwen2.5-7B-Instruct (CB safety-tuned) / GSM8K — AsFT · Lisa learning-rate sweep.
#
#   LRS 에 나열한 lr 마다 2셀(AsFT, Lisa)을 학습·업로드하고, 마지막에 전부 한 번에 평가한다.
#   lr 외의 설정은 전부 동일: LoRA r16/α32 · dropout 0.05 · targets q,k,v,up,down ·
#   3ep · 유효배치 16(4×4) · max_len 1024 · seed 42 · bf16 · cosine · wd 0 · warmup 0.03
#   AsFT λ=1.0 / Lisa ρ=1.0 (align 100 · ft 900).
#
# LORA_LR_DEFAULT 가 학습 명령과 리포명(_lr<값>)을 함께 바꾼다(hf_lr_tag → lora_lr).
# OUT_ROOT 은 lr 마다 분리한다 — out_dir 은 하이퍼파라미터를 이름에 담지 않아서 같은 경로를
# 쓰면 기존 .done 에 걸려 "이미 완료"로 건너뛴다(CLAUDE.md 에 기록된 함정).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

LRS="${LRS:-5e-3 3e-5}"
PATHS=()

for LR in $LRS; do
  echo ""
  echo "═══════════ lr=$LR 학습 시작 ═══════════ $(date -Iseconds)"
  OUT="$PWD/outputs/revision_qwen_lr$LR"
  mkdir -p "$OUT" "$PWD/logs/revision_qwen_lr$LR"
  env SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS="asft lisa" \
      LORA_LR_DEFAULT="$LR" LORA_ALPHA=32 LISA_RHO=1.0 SKIP_PUBLISHED=0 \
      OUT_ROOT="$OUT" LOG_ROOT="$PWD/logs/revision_qwen_lr$LR" \
      PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0 \
      bash scripts/revision/20_lora_family.sh
  echo "LR${LR}_RC=$?"

  for m in asft lisa; do
    cell="$OUT/cb/qwen25_7b/gsm8k/$m"
    [ -f "$cell/.done" ] || { echo "  [건너뜀] $m lr=$LR — .done 없음"; continue; }
    md="$(cat "$cell/MODEL_DIR" 2>/dev/null)"
    [ -n "$md" ] && [ -d "$md" ] || { echo "  [건너뜀] $m lr=$LR — MODEL_DIR 없음"; continue; }
    case "$m" in
      asft) name="qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr$LR" ;;
      lisa) name="qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr$LR" ;;
    esac
    mkdir -p outputs/eval_local; ln -sfn "$md" "outputs/eval_local/$name"
    PATHS+=("$PWD/outputs/eval_local/$name")
  done
done

echo ""
echo "═══════════ 평가 ═══════════ $(date -Iseconds)"
[ "${#PATHS[@]}" -eq 0 ] && { echo "❌ 평가할 셀이 없다"; exit 1; }
printf '  평가 대상: %s\n' "${PATHS[@]##*/}"
PREFETCH=0 REPOS_ONLY="${PATHS[*]}" bash scripts/revision/32_eval_cells.sh
echo "QWEN_LRSWEEP_DONE rc=$?  $(date -Iseconds)"
