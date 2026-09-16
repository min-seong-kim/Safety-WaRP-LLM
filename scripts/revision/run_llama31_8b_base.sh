#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  논문 Table 7 (C.1 "Results on Base Models") 의 **Llama-3.1-8B Base** 열에
#  rebuttal PEFT 7종을 추가한다.
#
#    lora · asft(λ=1.0) · lisa(ρ=1.0) · seal(top-p 0.8) ·
#    safelora(thr=0.3) · salora(r_s=r_t=32) · wsr_lora(ρ=0.3)
#
#  출발 모델 : kmseong/Llama-3.1-8B-base-SSFT_lr5e-5   (CB 로 안전정렬된 **base**)
#  downstream: gsm8k (data/gsm8k_train_task_7473.json — 전 arm 공용)
#
#  ── 하이퍼파라미터 (사용자 지정, 2026-09-16) ──────────────────────────────
#    LoRA 6종 : r=16, **alpha=16**, dropout 0.05, {q,k,v,up,down},
#               lr 3e-4, 3 epoch, effective batch 16, max_len 1024, seed 42, bf16
#    SEAL     : full-param S4, **lr 1e-5**, 3 epoch, effective batch 16
#               (Table 7 의 Llama-3.1-8B Base full-param 행들이 lr 1e-5 로 만들어졌다)
#    LISA ρ   : 1.0        WSR-LoRA ρ : 0.3        SafeLoRA thr : 0.3
#
#  허브의 llama2-7b base 라인(2026-09-14~15, `kmseong/llama2_7b-base-CB_SSFT-*`)과
#  **같은 설정·같은 리포명 규약**이다. 리포명은 common.sh 의 hf_repo_id() 가 만든다:
#    kmseong/llama3_1_8b-base-CB_SSFT-{method}_gsm8k[_{hparam}]_a16_lr3e-4
#
#  ── base 모델이라 특별히 주의할 것 ────────────────────────────────────────
#  * 출발 모델에 chat_template 이 **없다**. 모든 러너의 is_instruct_model() 은 모델
#    참조 **문자열**에 chat/instruct/it 이 있는지로 판정하므로, 리포·디렉토리 이름에
#    그 토큰이 들어가면 안 된다. 그래야 `Question: {q}\nAnswer:` plain 프롬프트가 되고,
#    이것이 HarmBench 의 chat_template: llama-3-base 와 **글자 단위로 같다**.
#  * AsFT / SafeLoRA 는 V = W_aligned − W_base 를 만들어야 해서 gated 리포
#    meta-llama/Llama-3.1-8B 를 실제로 내려받는다 → HF 토큰 + 라이선스 수락 필요.
#    그래서 이 둘을 **맨 마지막에** 돌린다(토큰이 없어도 나머지 5종은 끝난다).
#
#  사용:
#    bash scripts/revision/run_llama31_8b_base.sh              # 전체 (재실행 안전)
#    PLAN_ONLY=1 bash scripts/revision/run_llama31_8b_base.sh  # 계획만
#    DRY_RUN=1   bash scripts/revision/run_llama31_8b_base.sh  # 명령만 출력
#    STAGES="20b" bash scripts/revision/run_llama31_8b_base.sh # 특정 스테이지만
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# ────────────────────────── 이 박스(edgeai-1) 설정 ──────────────────────────
#  conda 는 lustre 에 이미 설치돼 있다. `hb` env = environment_hb_repro.yml 핀
#  (torch 2.10.0+cu128 / transformers 4.57.3 / peft 0.18.1), B200(sm_100) 동작 확인.
CONDA_ROOT="${CONDA_ROOT:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3}"
export PY="${PY:-$CONDA_ROOT/envs/hb/bin/python}"
export HF_HOME="${HF_HOME:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache}"
export PATH="$(dirname "$PY"):$PATH"
# ⚠️ CUDA_VISIBLE_DEVICES 는 설정하지 않는다 (CLAUDE.md). 이 박스는 B200 1장뿐이다.

# ────────────────────────── 실험 선택자 ──────────────────────────
export SAFETY_SETS=cb
export MODELS=llama31_8b_base
export TASKS=gsm8k

# LoRA 계열 공통: r=16 / **alpha=16** → 리포명에 _a16 이 붙는다(lora_alpha_tag).
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export WSR_LORA_ALPHA="${WSR_LORA_ALPHA:-16}"   # wsr_lora 는 별도 변수를 본다
export LISA_RHO="${LISA_RHO:-1.0}"
export KEEP_RATIO="${KEEP_RATIO:-0.3}"          # WSR-LoRA ρ
export SAFELORA_THRESHOLD="${SAFELORA_THRESHOLD:-0.3}"
export ASFT_LAMBDA_REG="${ASFT_LAMBDA_REG:-1.0}"
export SEAL_TOPP="${SEAL_TOPP:-0.8}"
export SEAL_FULL_LR="${SEAL_FULL_LR:-1e-5}"     # SEAL S4(full-param) lr — 사용자 지정

# ────────────────────────── 저장·업로드 ──────────────────────────
#  디스크 여유가 28TB 라 지우지 않는다. 로컬에 남겨야 평가 때 재다운로드
#  (이 박스 ~9MB/s) 를 피할 수 있다.
export PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
export PRUNE_AFTER_UPLOAD="${PRUNE_AFTER_UPLOAD:-0}"
export PRUNE_BASIS="${PRUNE_BASIS:-0}"
export PRUNE_HF_CACHE="${PRUNE_HF_CACHE:-0}"
export HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/revision}"
export SKIP_PUBLISHED="${SKIP_PUBLISHED:-1}"

DRY_RUN="${DRY_RUN:-0}";  export DRY_RUN
PLAN_ONLY="${PLAN_ONLY:-0}"
# 02 = Phase1 basis(WSR-LoRA 용) · 20a = 토큰 불필요 4종 · 21 = SEAL · 20b = 토큰 필요 2종
STAGES="${STAGES:-02 20a 21 20b}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/revision}"; export LOG_ROOT
mkdir -p "$LOG_ROOT" "$OUT_ROOT"
MAIN_LOG="$LOG_ROOT/llama31_8b_base_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  Table 7 · Llama-3.1-8B Base × PEFT 7종"
echo "    python    : $PY"
echo "    HF_HOME   : $HF_HOME"
echo "    출발 모델 : kmseong/Llama-3.1-8B-base-SSFT_lr5e-5"
echo "    LoRA      : r=$LORA_R alpha=$LORA_ALPHA lr=3e-4 ep3 eff.batch16"
echo "    SEAL      : full-param lr=$SEAL_FULL_LR topp=$SEAL_TOPP"
echo "    LISA ρ    : $LISA_RHO   WSR-LoRA ρ : $KEEP_RATIO   SafeLoRA thr : $SAFELORA_THRESHOLD"
echo "    stages    : $STAGES"
echo "    log       : $MAIN_LOG"
echo "════════════════════════════════════════════════════════════════"

if [[ "$PLAN_ONLY" == "1" ]]; then
  METHODS="lora asft lisa seal safelora salora wsr_lora" \
    bash -c 'source scripts/revision/common.sh; print_plan; check_disk
             echo ""; echo "  생성될 리포:"
             for m in lora asft lisa seal safelora salora wsr_lora; do
               [[ $m == seal ]] && FULL_LR='"$SEAL_FULL_LR"'
               printf "    %-9s %s\n" "$m" "$(hf_repo_id cb llama31_8b_base gsm8k $m)"
             done'
  exit 0
fi

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
rc_all=0

# ── 02: Phase 1 basis (WSR-LoRA 전용. wsr_tune 셀이 없으므로 Phase 2 는 자동 생략) ──
if has_stage 02; then
  METHODS="wsr_lora" bash scripts/revision/02_warp_basis_mask.sh || rc_all=1
fi

# ── 20a: 출발 모델만 있으면 되는 4종 ──────────────────────────────────────
if has_stage 20a; then
  METHODS="lora lisa salora wsr_lora" bash scripts/revision/20_lora_family.sh || rc_all=1
fi

# ── 21: SEAL (S1 selector → S1.5 top-p → S2 full-param SFT @ lr 1e-5) ─────
if has_stage 21; then
  METHODS="seal" FULL_LR="$SEAL_FULL_LR" bash scripts/revision/21_seal.sh || rc_all=1
fi

# ── 20b: gated base(meta-llama/Llama-3.1-8B) 가 필요한 2종 ────────────────
if has_stage 20b; then
  METHODS="asft safelora" bash scripts/revision/20_lora_family.sh || rc_all=1
fi

echo ""
echo "════════════════════ 셀 상태 ════════════════════"
for m in lora asft lisa seal safelora salora wsr_lora; do
  d="$OUT_ROOT/cb/llama31_8b_base/gsm8k/$m"
  # ⚠️ "디렉토리 존재"를 진행 신호로 쓰지 말 것. run_cell 은 DRY_RUN 분기보다 **먼저**
  #    mkdir -p 를 하므로, dry-run 한 번이면 전 셀의 빈 디렉토리가 생긴다.
  #    실제 신호는 run.log(시작) / .done(성공) / .uploaded(업로드·검증 완료) 다.
  if   [[ -f "$d/.uploaded" ]]; then st="uploaded  $(cat "$d/.uploaded")"
  elif [[ -f "$d/.done" ]];     then st="done (업로드 안 됨)"
  elif [[ -s "$d/run.log" ]];   then st="진행 중 (run.log $(wc -c <"$d/run.log") bytes)"
  else                               st="시작 안 함"; fi
  printf "  %-9s %s\n" "$m" "$st"
done
echo "전체 로그: $MAIN_LOG"
exit $rc_all
