#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Stage 11 — 사후(post-hoc) 방어: RESTA / SafeDelta
#
#  둘 다 "**방어 없이 미세조정된 모델을 나중에 고친다**" 는 계열이므로 입력이
#  Stage 10 의 **fullft** 산출물이다(safeinstr 이 아니다). 논문 Table 2 도 그렇다.
#
#    RESTA     W_resta = W_ft + γ·(W_align − W_base),  γ = 0.3  (논문 §4.1)
#              scripts/resta_add_safety.py — mergekit 없이 샤드 스트리밍으로 계산.
#              가중치 합이 1.0+γ−γ=1.0 이라 mergekit 의 linear(normalize=True) 와 동일.
#
#    SafeDelta W_sd = W_align + M⊙ΔW + C,  s = 0.1  (논문 §4.1)
#              외부 구현 $SAFEDELTA_DIR/llama2/run_safedelta.py 를 그대로 호출한다.
#              ⚠️ `from configs import ...` 때문에 반드시 llama2/ 안에서 실행해야 한다.
#              ⚠️ s 는 **허용 안전손실 예산**이다. 크면 fine-tuned 델타를 더 많이 살린다
#                 = utility↑ safety↓. (원본 README 의 설명은 반대로 적혀 있으니 믿지 말 것.)
#              ⚠️ s 는 fine-tuning 종류를 가로질러 비교 가능한 값이 아니다. 여기서는
#                 full-param 델타에 거는 것이고, 논문 Table 2 와 같은 조건이다.
#              산출물은 run_safedelta.py 가 `<ft_dir>-SafeDelta-s<scale>` 에 저장하므로
#              끝나고 규약 경로로 옮긴다.
#
#  사용:
#    bash scripts/revision/11_posthoc.sh
#    METHODS=resta bash scripts/revision/11_posthoc.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/11_posthoc_${TS}.log") 2>&1

preflight
print_plan

has_method resta || has_method safedelta || { log "resta/safedelta 둘 다 METHODS 에 없다 — 종료"; exit 0; }

if has_method safedelta && [[ ! -f "$SAFEDELTA_DIR/llama2/run_safedelta.py" ]]; then
  warn "SafeDelta 구현 없음: $SAFEDELTA_DIR/llama2/run_safedelta.py — safedelta 셀을 전부 건너뛴다."
fi

echo ""
echo "  RESTA γ      : $RESTA_GAMMA"
echo "  SafeDelta s  : $SAFEDELTA_SCALE  (nsamples=$SAFEDELTA_NSAMPLES seq_len=$SAFEDELTA_SEQLEN)"
echo "  SafeDelta dir: $SAFEDELTA_DIR"

for safety in $SAFETY_SETS; do
  SAFE_DATA="$(safety_json "$safety")"

  for mkey in $MODELS; do
    safety_applies "$safety" "$mkey" || continue
    model_cfg "$mkey" || { FAILED_CELLS+=("$safety/$mkey (unknown model)"); continue; }
    ALIGNED="$(aligned_for "$mkey" "$safety")"

    for task in $(tasks_for_model "$mkey" "$safety"); do
      FT_DIR="$(out_dir "$safety" "$mkey" "$task" fullft)"

      # 두 기법 모두 fullft 산출물이 있어야 한다.
      if [[ "$DRY_RUN" != "1" ]] && { ! is_done "$FT_DIR" || [[ ! -f "$FT_DIR/config.json" ]]; }; then
        warn "[$safety/$mkey/$task] fullft 산출물이 없다 ($FT_DIR) → 10_fullft_safeinstr.sh 먼저. 건너뜀."
        FAILED_CELLS+=("posthoc/$safety/$mkey/$task (no fullft)")
        continue
      fi

      # ═══════════ RESTA ═══════════
      if want_cell "$safety" "$mkey" "$task" resta; then
        odir="$(out_dir "$safety" "$mkey" "$task" resta)"
        neg_gamma="$("$PY" -c "print(-float('$RESTA_GAMMA'))")"
        cmd=( "$PY" scripts/resta_add_safety.py
              --model1 "$FT_DIR"   --weight1 1.0
              --model2 "$ALIGNED"  --weight2 "$RESTA_GAMMA"
              --model3 "$BASE"     --weight3 "$neg_gamma"
              --output_path "$odir"
              --dtype "$DTYPE" )
        run_cell "$odir" "resta(γ=$RESTA_GAMMA)  $safety/$mkey/$task" -- "${cmd[@]}"
        post_cell "$odir" "$safety" "$mkey" "$task" resta
      fi

      # ═══════════ SafeDelta ═══════════
      if want_cell "$safety" "$mkey" "$task" safedelta; then
        odir="$(out_dir "$safety" "$mkey" "$task" safedelta)"
        if is_done "$odir"; then
          log "[skip] safedelta  $safety/$mkey/$task  (이미 완료)"
          continue
        fi
        if [[ ! -f "$SAFEDELTA_DIR/llama2/run_safedelta.py" ]]; then
          FAILED_CELLS+=("safedelta/$safety/$mkey/$task (impl missing)"); continue
        fi

        # run_safedelta.py 가 강제로 쓰는 출력 경로
        produced="${FT_DIR}-SafeDelta-s${SAFEDELTA_SCALE}"

        deadline_passed && { log "[deadline] 마감 초과 — 시작하지 않는다"; continue; }
        hdr "safedelta(s=$SAFEDELTA_SCALE)  $safety/$mkey/$task"
        mkdir -p "$odir"

        if [[ "$DRY_RUN" == "1" ]]; then
          echo "  [dry-run] ( cd $SAFEDELTA_DIR/llama2 && $PY run_safedelta.py \\"
          echo "                --model_name_align $ALIGNED --model_name_ft $FT_DIR \\"
          echo "                --scale $SAFEDELTA_SCALE --nsamples $SAFEDELTA_NSAMPLES \\"
          echo "                --seq_len $SAFEDELTA_SEQLEN --safe_data_path $SAFE_DATA )"
          echo "  [dry-run] mv $produced -> $odir"
          continue
        fi

        rm -rf "$produced"
        ( cd "$SAFEDELTA_DIR/llama2" && "$PY" run_safedelta.py \
              --model_name_align "$ALIGNED" \
              --model_name_ft "$FT_DIR" \
              --scale "$SAFEDELTA_SCALE" \
              --nsamples "$SAFEDELTA_NSAMPLES" \
              --seq_len "$SAFEDELTA_SEQLEN" \
              --safe_data_path "$SAFE_DATA" ) 2>&1 | tee "$odir/run.log"
        rc=${PIPESTATUS[0]}

        if (( rc == 0 )) && [[ -f "$produced/config.json" ]]; then
          # 규약 경로로 옮긴다 (run.log 는 유지)
          mv "$produced"/* "$odir"/ 2>/dev/null
          rmdir "$produced" 2>/dev/null
          # chat_template 가 딸려오지 않는 경우가 있어 출발 모델 것으로 보완한다.
          # (없으면 평가 시 프롬프트가 학습 때와 달라진다 — CLAUDE.md 의 함정 참조)
          if [[ ! -f "$odir/chat_template.jinja" ]] && [[ -f "$FT_DIR/chat_template.jinja" ]]; then
            cp "$FT_DIR/chat_template.jinja" "$odir/"
            log "  chat_template.jinja 를 fullft 산출물에서 복사했다"
          fi
          "$PY" - "$odir" <<'PYEOF' || warn "chat_template 검증 실패"
import sys, json, os
from transformers import AutoTokenizer
d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
print(f"  chat_template: {'OK' if tok.chat_template else 'NONE (⚠️ 평가 포맷이 어긋난다)'}")
PYEOF
          mark_done "$odir"
          write_model_ptr "$odir" || true
          upload_cell "$safety" "$mkey" "$task" safedelta
          log "[done] safedelta  $safety/$mkey/$task → $odir"
        else
          warn "[fail rc=$rc] safedelta  $safety/$mkey/$task — 로그: $odir/run.log"
          FAILED_CELLS+=("safedelta/$safety/$mkey/$task")
          [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
        fi
      fi
    done
  done
done

print_failures
