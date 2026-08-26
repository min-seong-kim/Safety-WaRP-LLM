#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 01 — BeaverTails 안전정렬 출발 모델 (Phase 0 SSFT)
#
#  이 논문의 모든 실험은 "**이미 안전정렬된** 모델을 downstream 으로 미세조정하면
#  안전성이 무너지는가" 를 본다. 따라서 출발점이 SSFT 모델이어야 한다.
#
#    Circuit Breakers 축 : 6개 모델 전부 이미 존재 → 이 스크립트가 할 일 없음
#    BeaverTails 축      : llama2_7b 만 존재(wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv),
#                          나머지 5개 모델을 여기서 학습한다.
#
#  설정은 CB 쪽 SSFT 와 동일: full-param, 4994 샘플, 3 epoch, effective batch 16,
#  max_len 1024, lr 5e-5 (gemma2-9b 만 3e-5 — 기존 CB 모델도 3e-5 로 만들어졌다).
#
#  사용:
#    bash scripts/revision/01_ssft_bt.sh
#    MODELS=llama2_13b bash scripts/revision/01_ssft_bt.sh
#    DRY_RUN=1 bash scripts/revision/01_ssft_bt.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/01_ssft_bt_${TS}.log") 2>&1

preflight
print_plan

BT_DATA="$(safety_json bt)"
[[ -f "$BT_DATA" ]] || die "BeaverTails 안전 데이터 없음: $BT_DATA"

echo ""
echo "  안전 데이터 : $BT_DATA ($SAFETY_SAMPLES 샘플)"
echo "  출력        : $CKPT_ROOT/ssft_bt/"

for mkey in $MODELS; do
  safety_applies bt "$mkey" || { log "[skip] ssft_bt/$mkey — BT_MODELS 밖"; continue; }
  model_cfg "$mkey" || { FAILED_CELLS+=("ssft_bt/$mkey (unknown model)"); continue; }

  if [[ -n "$ALIGNED_BT" ]]; then
    log "[skip] ssft_bt/$mkey — 기존 BT 안전정렬 모델 재사용: $ALIGNED_BT"
    continue
  fi

  out_d="$(bt_ssft_dir "$mkey")"
  if [[ -f "$out_d/config.json" ]] || is_done "$out_d"; then
    log "[skip] ssft_bt/$mkey — 이미 완료 ($out_d)"
    continue
  fi

  accum="$(accum_for "$MB_FULL")" || { FAILED_CELLS+=("ssft_bt/$mkey (batch)"); continue; }

  hdr "SSFT(BT) $mkey  base=$BASE  lr=$SSFT_LR  batch=${MB_FULL}x${accum}"
  mkdir -p "$out_d"

  # phase0_SSFT.py 는 batch/grad_accum 을 CLI 로 받지 않는다 → 환경변수로 넘긴다.
  # (models/phase0_SSFT.py 의 BATCH_SIZE / GRAD_ACCUM_STEPS 가 이 값을 읽는다.)
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] SSFT_BATCH_SIZE=$MB_FULL SSFT_GRAD_ACCUM=$accum \\"
    echo "            $PY models/phase0_SSFT.py $BT_DATA --model_name $BASE --lr $SSFT_LR \\"
    echo "            --output_dir $out_d --no_wandb --log_dir $LOG_ROOT"
    continue
  fi

  SSFT_BATCH_SIZE="$MB_FULL" SSFT_GRAD_ACCUM="$accum" \
  "$PY" models/phase0_SSFT.py "$BT_DATA" \
      --model_name "$BASE" \
      --lr "$SSFT_LR" \
      --output_dir "$out_d" \
      --no_wandb \
      --log_dir "$LOG_ROOT" \
      2>&1 | tee "$out_d/ssft.log"
  rc=${PIPESTATUS[0]}

  if (( rc == 0 )) && [[ -f "$out_d/config.json" ]]; then
    mark_done "$out_d"
    write_model_ptr "$out_d" || true
    log "[done] ssft_bt/$mkey → $out_d"
    # 새로 만든 자산이므로 허브에 올려 둔다. 단 **삭제하지는 않는다** —
    # 이 모델은 해당 (모델, BT) 의 모든 셀에서 출발점으로 계속 쓰인다.
    if [[ "$PUSH_TO_HUB" == "1" ]]; then
      repo="$(hf_ssft_repo_id "$mkey" bt)"
      hdr "HF 업로드 (SSFT-BT)  $mkey  →  $repo"
      "$PY" scripts/revision/upload_and_prune.py --cell_dir "$out_d" --repo_id "$repo" \
        2>&1 | sed 's/^/    /' || { warn "SSFT-BT 업로드 실패: $repo"; FAILED_CELLS+=("upload/ssft_bt/$mkey"); }
    fi
  else
    warn "[fail rc=$rc] ssft_bt/$mkey — 로그: $out_d/ssft.log"
    warn "  OOM 이면 ${mkey^^}_MB_FULL 을 절반으로 줄여라 (곱이 16 이면 결과 동일)."
    warn "  예: LLAMA2_13B_MB_FULL=1 bash scripts/revision/01_ssft_bt.sh"
    FAILED_CELLS+=("ssft_bt/$mkey")
    [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
  fi
done

echo ""
echo "════════════════ Stage 01 요약 ════════════════"
for mkey in $MODELS; do
  model_cfg "$mkey" || continue
  a="$(aligned_for "$mkey" bt)"
  if [[ "$a" == /* ]]; then
    if [[ -f "$a/config.json" ]]; then echo "  [ok]      $mkey  → $a"
    else                                echo "  [MISSING] $mkey  → $a"; fi
  else
    echo "  [reuse]   $mkey  → $a"
  fi
done

print_failures
