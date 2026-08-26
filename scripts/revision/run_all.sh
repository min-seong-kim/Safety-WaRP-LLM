#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Revision 실험 오케스트레이터
#
#  논문 Table 2 / Table 4 / Figure 4 를 아래로 확장한다.
#
#    안전 데이터 2종 : cb(circuit_breakers) · bt(beavertails)
#    모델 6종        : llama2_7b llama2_13b llama32_3b llama31_8b qwen25_7b gemma2_9b
#    기법 12종       : fullft safeinstr resta safedelta wsr_tune          (기존 5)
#                      lora asft lisa seal safelora salora wsr_lora       (신규 7)
#    태스크          : 모델별 primary (Table2/4) + llama2_7b 한정 medqa/arc/agnews (Figure4)
#
#  스테이지 순서(의존성 순):
#    00 데이터 준비 → 01 BT SSFT → 02 basis/mask → 10 FullFT/SafeInstr
#       → 11 RESTA/SafeDelta(10 필요) → 12 WSR-Tune(02 필요)
#       → 20 LoRA 6종(wsr_lora 는 02 필요) → 21 SEAL
#
#  전 스테이지가 **재개 가능**하다(.done 센티넬). 중간에 죽어도 그대로 다시 돌리면 이어서 간다.
#
#  ── 사용 ─────────────────────────────────────────────────────────────────
#    bash scripts/revision/run_all.sh                     # 전체
#    DRY_RUN=1 bash scripts/revision/run_all.sh           # 명령만 출력 (먼저 이걸로 확인할 것)
#    PLAN_ONLY=1 bash scripts/revision/run_all.sh         # 셀 개수/진행률만
#    SAFETY_SETS=cb bash scripts/revision/run_all.sh      # CB 축만
#    PUSH_TO_HUB=1 bash scripts/revision/run_all.sh       # 셀마다 HF 업로드 후 로컬 삭제
#    ORDER=stage bash scripts/revision/run_all.sh         # 스테이지 단위(디스크 넉넉할 때만)
#    MODELS=llama2_7b METHODS="fullft wsr_tune" bash scripts/revision/run_all.sh
#    STAGES="10 20" bash scripts/revision/run_all.sh      # 특정 스테이지만
#
#  ⚠️ 단일 GPU 순차 실행이다. 전체 매트릭스는 수 주 규모다. 먼저 PLAN_ONLY 로 규모를
#     확인하고, MODELS/SAFETY_SETS/METHODS 로 쪼개서 돌리는 것을 권한다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

PLAN_ONLY="${PLAN_ONLY:-0}"
STAGES="${STAGES:-00 01 02 10 11 12 20 21}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"

# ────────────────────────── 계획 요약 ──────────────────────────
print_plan

echo ""
echo "════════════════ 셀 목록 / 진행률 ════════════════"
declare -i n_total=0 n_done=0
declare -a PENDING=()
for safety in $SAFETY_SETS; do
  for mkey in $MODELS; do
    model_cfg "$mkey" || continue
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      for method in $METHODS; do
        cell_wanted "$safety" "$mkey" "$task" "$method" || continue
        n_total+=1
        d="$(out_dir "$safety" "$mkey" "$task" "$method")"
        if is_done "$d"; then n_done+=1; else PENDING+=("$safety/$mkey/$task/$method"); fi
      done
    done
  done
done
printf "  학습 셀: %d개 중 %d개 완료, %d개 남음\n" "$n_total" "$n_done" "$(( n_total - n_done ))"
check_disk

# 선행 산출물(BT SSFT / basis / mask) 상태
echo ""
echo "  선행 산출물:"
for safety in $SAFETY_SETS; do
  for mkey in $MODELS; do
    safety_applies "$safety" "$mkey" || continue
    model_cfg "$mkey" || continue
    a="$(aligned_for "$mkey" "$safety")"
    astat="reuse(hub)"
    [[ "$a" == /* ]] && { [[ -f "$a/config.json" ]] && astat="ok(local)" || astat="MISSING"; }
    b="$CKPT_ROOT/warp/$safety/$mkey/BASIS_DIR"; m="$CKPT_ROOT/warp/$safety/$mkey/MASKS_DIR"
    bstat="불필요"; mstat="불필요"
    for task in $(tasks_for_model "$mkey" "$safety"); do
      want_cell "$safety" "$mkey" "$task" wsr_lora && [[ "$bstat" == "불필요" ]] && bstat="MISSING"
      want_cell "$safety" "$mkey" "$task" wsr_tune && { bstat="MISSING"; mstat="MISSING"; }
    done
    [[ "$bstat" == "MISSING" && -s "$b" ]] && [[ -d "$(cat "$b")" ]] && bstat="ok"
    [[ "$mstat" == "MISSING" && -s "$m" ]] && [[ -d "$(cat "$m")" ]] && mstat="ok"
    printf "    %-3s %-12s aligned=%-11s basis=%-8s mask=%s\n" "$safety" "$mkey" "$astat" "$bstat" "$mstat"
  done
done

if (( ${#PENDING[@]} > 0 )) && [[ "${SHOW_PENDING:-0}" == "1" ]]; then
  echo ""
  echo "  남은 셀:"
  printf '    %s\n' "${PENDING[@]}"
fi

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo ""
  echo "PLAN_ONLY=1 → 여기서 종료. 실제로 돌리려면 PLAN_ONLY 를 빼라."
  exit 0
fi

# ────────────────────────── 스테이지 실행 ──────────────────────────
#  ORDER=model (기본): (안전데이터 × 모델) 하나를 **끝까지** 처리하고 다음으로 넘어간다.
#    이래야 basis/mask 를 한 조합만 디스크에 두고 바로 지울 수 있다(조합당 최대 30GB).
#    스테이지 단위로 돌면 12개 basis 가 동시에 살아 248GB 를 차지한다.
#  ORDER=stage: 예전처럼 스테이지 단위. 디스크가 넉넉할 때만 쓸 것.
ORDER="${ORDER:-model}"
declare -a STAGE_FAILED=()

run_stage() {  # <번호> <스크립트> [env 오버라이드...]
  local num="$1" script="$2"; shift 2
  [[ " $STAGES " == *" $num "* ]] || return 0
  echo ""
  echo "  ── STAGE $num  $script  $*   ($(date '+%m-%d %H:%M:%S'))"
  if env "$@" bash "$HERE/$script"; then
    log "[stage $num] 완료"
  else
    warn "[stage $num] 실패한 셀이 있다 (위 로그 확인)"
    STAGE_FAILED+=("$num:$script${*:+ ($*)}")
  fi
}

# ── 전역 1회: 데이터 준비 ──
run_stage 00 00_prepare.sh

# ── 전역 1회: BT 안전정렬 출발모델 (이후 모든 조합의 입력) ──
if [[ " $SAFETY_SETS " == *" bt "* ]]; then
  run_stage 01 01_ssft_bt.sh
else
  log "[stage 01] SAFETY_SETS 에 bt 가 없음 — 건너뜀"
fi

need_warp() { has_method wsr_tune || has_method wsr_lora; }

run_combo() {  # <safety> <model> — 이 조합을 끝까지
  local safety="$1" mkey="$2"
  if deadline_passed; then log "[deadline] $safety/$mkey — 마감 초과, 건너뜀"; return 0; fi
  echo ""
  echo "████████████████████████████████████████████████████████████████"
  echo "  ▶ $safety / $mkey     ($(date '+%m-%d %H:%M:%S'))"
  echo "     디스크 여유: $(df -BG --output=avail "$OUT_ROOT" 2>/dev/null | tail -1 | tr -d ' ')"
  echo "████████████████████████████████████████████████████████████████"
  local E=(MODELS="$mkey" SAFETY_SETS="$safety")

  need_warp && run_stage 02 02_warp_basis_mask.sh "${E[@]}"
  run_stage 10 10_fullft_safeinstr.sh "${E[@]}"   # 11 의 선행조건
  run_stage 12 12_wsr_tune.sh         "${E[@]}"
  run_stage 20 20_lora_family.sh      "${E[@]}"   # 여기서 basis 가 마지막으로 쓰이고 삭제된다
  run_stage 21 21_seal.sh             "${E[@]}"
  run_stage 11 11_posthoc.sh          "${E[@]}"   # fullft 를 소비 → 끝나면 fullft 도 삭제 가능
  # 11 이 끝나 fullft 의 prune 조건이 풀렸으므로 한 번 더 훑어 지운다.
  if [[ "$PUSH_TO_HUB" == "1" && "$PRUNE_AFTER_UPLOAD" == "1" && "$DRY_RUN" != "1" ]]; then
    local task
    for task in $(tasks_for_model "$mkey"); do
      want_cell "$safety" "$mkey" "$task" fullft && upload_cell "$safety" "$mkey" "$task" fullft
    done
  fi
}

if [[ "$ORDER" == "model" ]]; then
  # 모델을 바깥 루프로 둔다: 한 모델의 CB/BT 를 연달아 처리해야 그 모델의 HF 캐시
  # (base + aligned, 15~26GB)를 한 번만 받고 끝나면 바로 비울 수 있다.
  for mkey in $MODELS; do
    model_cfg "$mkey" || continue
    for safety in $SAFETY_SETS; do
      safety_applies "$safety" "$mkey" || continue
      run_combo "$safety" "$mkey"
    done
    prune_hf_cache_for_model "$mkey"
  done
else
  need_warp && run_stage 02 02_warp_basis_mask.sh
  run_stage 10 10_fullft_safeinstr.sh
  run_stage 11 11_posthoc.sh
  run_stage 12 12_wsr_tune.sh
  run_stage 20 20_lora_family.sh
  run_stage 21 21_seal.sh
fi

# ────────────────────────── 최종 요약 ──────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " 최종 요약    ts=$TS"
echo "════════════════════════════════════════════════════════════════"
declare -i f_total=0 f_done=0
for safety in $SAFETY_SETS; do
  for mkey in $MODELS; do
    model_cfg "$mkey" || continue
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      line="  $safety/$mkey/$task :"
      any=0
      for method in $METHODS; do
        cell_wanted "$safety" "$mkey" "$task" "$method" || continue
        any=1; f_total+=1
        d="$(out_dir "$safety" "$mkey" "$task" "$method")"
        if is_done "$d"; then f_done+=1; line+=" $method"; else line+=" -$method"; fi
      done
      (( any )) && echo "$line"
    done
  done
done
echo ""
printf "  완료 %d / %d 셀   (앞에 '-' 가 붙은 것이 미완료)\n" "$f_done" "$f_total"
echo "  산출물: $OUT_ROOT/<safety>/<model>/<task>/<method>/   (모델 경로는 각 셀의 MODEL_DIR 파일)"
echo "  로그  : $LOG_ROOT/"

if (( ${#STAGE_FAILED[@]} > 0 )); then
  echo ""
  echo "  실패한 스테이지: ${STAGE_FAILED[*]}"
  echo "  같은 명령을 그대로 다시 실행하면 완료된 셀은 건너뛰고 실패분만 재시도한다."
  exit 1
fi
exit 0
