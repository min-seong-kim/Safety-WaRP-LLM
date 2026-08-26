#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Revision 실험 공용 레지스트리 + 헬퍼   (source 전용, 직접 실행하지 않는다)
#
#  논문 Table 2 / Table 4 / Figure 4 를 12개 기법 × 6개 모델 × 2개 안전데이터로
#  확장하기 위한 단일 진실 공급원(single source of truth).
#  모든 스테이지 스크립트(00~21)가 이 파일만 참조한다. 하이퍼파라미터를 바꿀 일이
#  있으면 **여기만** 고칠 것.
#
#  ── 설계 규약 ─────────────────────────────────────────────────────────────
#  1. effective batch = 16 은 전 기법 공통 불변식이다. micro-batch 는 모델/기법
#     별로 다르되 grad_accum 은 항상 16/micro 로 자동 계산한다(나누어떨어져야 함).
#  2. 모든 arm 은 **같은 로컬 태스크 JSON**(data/*_task_*.json)을 읽는다.
#     프롬프트/정답 문자열이 한 글자라도 다르면 비교가 성립하지 않는다.
#     검증: python scripts/revision/verify_prompt_parity.py
#  3. CUDA_VISIBLE_DEVICES 를 여기서 설정하지 않는다(CLAUDE.md). SLURM 이 넣어준다.
#  4. 완료 판정은 출력 디렉토리의 `.done` 센티넬 하나로 통일한다.
#  5. **안전 데이터 축은 하나다.** 안전 데이터를 쓰는 기법(SafeInstr / SafeDelta /
#     LISA / SaLoRA / SEAL / WSR-Tune / WSR-LoRA)은 전부 "그 출발 모델을 안전정렬할 때
#     쓴 바로 그 데이터셋"을 쓴다. CB 출발모델 → circuit_breakers, BT 출발모델 →
#     beavertails. 스테이지 스크립트에서 $safety 루프 변수 하나가 aligned_for() 와
#     safety_json() 을 **동시에** 결정하므로 구조적으로 섞일 수 없다.
#     (RESTA / AsFT / SafeLoRA 는 별도 안전 코퍼스 대신 base·aligned 두 모델의 차이
#      V = W_aligned − W_base 를 쓰므로 같은 축에 자동으로 묶인다.)
# ════════════════════════════════════════════════════════════════════════════

[[ -n "${_REVISION_COMMON_SOURCED:-}" ]] && return 0
_REVISION_COMMON_SOURCED=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# ⚠️ CUDA_VISIBLE_DEVICES 설정 금지 — 스케줄러가 넣어준 값을 그대로 쓴다.

# ────────────────────────────── 경로 규약 ──────────────────────────────
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/revision}"       # 학습 산출물
CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/checkpoints/revision}" # basis / mask / SSFT
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/revision}"
SEAL_CKPT_ROOT="${SEAL_CKPT_ROOT:-$REPO_DIR/seal/ckpt/revision}"

# ────────────────────────────── 실행 제어 ──────────────────────────────
DRY_RUN="${DRY_RUN:-0}"          # 1 = 명령만 출력
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"           # 1 = 셀이 끝날 때마다 HF 업로드
HF_NAMESPACE="${HF_NAMESPACE:-kmseong}"
HF_PRIVATE="${HF_PRIVATE:-0}"
# 업로드 검증에 성공한 셀의 로컬 가중치를 지운다. 216셀 전체는 3.4TB 라 이게 없으면
# 디스크가 버티지 못한다. fullft 는 RESTA/SafeDelta 가 소비한 뒤에만 지워진다.
PRUNE_AFTER_UPLOAD="${PRUNE_AFTER_UPLOAD:-1}"
# WSR 계열이 전부 끝난 (안전데이터, 모델) 의 basis/mask 삭제 (조합당 최대 30GB).
PRUNE_BASIS="${PRUNE_BASIS:-1}"
# 한 모델의 두 안전축이 모두 끝나면 그 모델의 HF 캐시(base/aligned 스냅샷)를 지운다.
# 모델 하나가 캐시에서 15~26GB 를 먹는다. 필요하면 다시 받는다.
PRUNE_HF_CACHE="${PRUNE_HF_CACHE:-1}"
# BT SSFT 로컬 디렉토리를 업로드 검증 후 삭제하고, 이후에는 허브 리포를 출발점으로 쓴다.
PRUNE_SSFT_LOCAL="${PRUNE_SSFT_LOCAL:-0}"
# Phase 1 basis 저장 형식. bf16 + UT 생략 = 디스크 1/4.
#   Phase 2/3/WSR-LoRA 모두 로드 직후 U 를 모델 dtype(bf16)으로 내리고, UT 키는
#   이 저장소의 어떤 소비자도 읽지 않으므로 사용값이 바뀌지 않는다.
BASIS_SAVE_DTYPE="${BASIS_SAVE_DTYPE:-bfloat16}"
BASIS_OMIT_UT="${BASIS_OMIT_UT:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"   # 1 = 한 셀이 죽어도 다음 셀 진행

# 선택자 (전부 공백 구분)
MODELS="${MODELS:-llama2_7b llama2_13b llama32_3b llama31_8b qwen25_7b gemma2_9b}"
SAFETY_SETS="${SAFETY_SETS:-cb bt}"
METHODS="${METHODS:-fullft safeinstr resta safedelta wsr_tune lora asft lisa seal safelora salora wsr_lora}"
# TASKS 를 비워두면 모델별 primary task + (llama2_7b 한정) Figure4 확장 태스크를 자동 결정한다.
TASKS="${TASKS:-}"

# ── BT(BeaverTails) 축 범위 ──────────────────────────────────────────────
#  BT 는 "안전 데이터 출처를 바꿔도 결론이 유지되는가"를 보는 축이라 전 모델로 펼칠 필요가 없다.
#  rebuttal 도 llama2-7b / GSM8K 한 셀에서만 BT 를 보였다. 그 범위를 그대로 따른다.
BT_MODELS="${BT_MODELS:-llama2_7b}"
# llama2-7b 의 4개 태스크 전부를 12기법으로 돌린다 (사용자 지정).
BT_TASKS="${BT_TASKS:-gsm8k medqa arc agnews}"

# 이미 논문/rebuttal 에 실린 셀은 건너뛴다 (0 이면 전부 다시 돌린다).
SKIP_PUBLISHED="${SKIP_PUBLISHED:-1}"

# ────────────────────────── 공통 하이퍼파라미터 ──────────────────────────
EFFECTIVE_BATCH=16
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
MAX_GRAD_NORM=1.0
DTYPE=bfloat16

# Full-parameter 계열 (Full FT / SafeInstr / SEAL-S4 / WSR-Tune Phase3)
FULL_LR="${FULL_LR:-5e-5}"
FULL_WEIGHT_DECAY=0.01
FULL_WARMUP_RATIO=0.1
FULL_SCHEDULER=cosine

# LoRA 계열 (lora / asft / lisa / safelora / salora / wsr_lora)
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_WEIGHT_DECAY=0.0
LORA_WARMUP_RATIO=0.03
LORA_SCHEDULER=cosine
TARGET_MODULES_CSV="q_proj,k_proj,v_proj,up_proj,down_proj"
TARGET_MODULES_LIST=(q_proj k_proj v_proj up_proj down_proj)
LAYER_TYPES="attn_q,attn_k,attn_v,ffn_up,ffn_down"
TARGET_LAYERS=all

# ── 기법 고유 하이퍼파라미터 ──────────────────────────────────────────────
#    논문 §4.1 / rebuttal / 사용자 지정값. 근거를 주석에 남긴다.
SAFEINSTR_RATIO="${SAFEINSTR_RATIO:-0.1}"   # 논문 §4.1 "safety examples = 10% of the downstream training set"
RESTA_GAMMA="${RESTA_GAMMA:-0.3}"           # 논문 §4.1 "Resta ... scaling coefficient 0.3"
SAFEDELTA_SCALE="${SAFEDELTA_SCALE:-0.1}"   # 논문 §4.1 "SafeDelta ... s = 0.1"
SAFEDELTA_NSAMPLES="${SAFEDELTA_NSAMPLES:-512}"
SAFEDELTA_SEQLEN="${SAFEDELTA_SEQLEN:-512}"
SAFEDELTA_DIR="${SAFEDELTA_DIR:-/home/edgeai_lab/SafeDelta}"

KEEP_RATIO="${KEEP_RATIO:-0.1}"             # WSR-Tune / WSR-LoRA freeze ratio rho = 10%
ASFT_LAMBDA_REG="${ASFT_LAMBDA_REG:-1.0}"   # 사용자 지정 λ=1.0 (= 참조 구현 AsFT_reg1_p_0.1.sh)
LISA_RHO="${LISA_RHO:-1.0}"                 # 사용자 지정
LISA_ALIGNMENT_STEP="${LISA_ALIGNMENT_STEP:-100}"
LISA_FINETUNE_STEP="${LISA_FINETUNE_STEP:-900}"
SAFELORA_THRESHOLD="${SAFELORA_THRESHOLD:-0.3}"   # 사용자 지정 thr=0.3
# SaLoRA 는 salora/salora_lora.py + salora/salora_impl.py (Li et al., ICLR'25 포팅) 를 쓴다.
#   r_s : safety 부분공간 차원 — C_S = I − U_C U_Cᵀ (U_C = top-r_s left SV of W X_harmful)
#   r_t : task 부분공간 차원   — init_mode=task 일 때 B_S 를 이 부분공간으로 투영
#   둘 다 원본 포팅의 기본값 32 를 그대로 쓴다.
SALORA_R_S="${SALORA_R_S:-32}"
SALORA_R_T="${SALORA_R_T:-32}"
SALORA_INIT_MODE="${SALORA_INIT_MODE:-task}"        # full SaLoRA
SALORA_N_HARMFUL="${SALORA_N_HARMFUL:-256}"         # C_S 용 harmful 샘플 수
SALORA_N_TASK="${SALORA_N_TASK:-256}"               # task 부분공간용 downstream 샘플 수
SALORA_MAX_TOKENS="${SALORA_MAX_TOKENS:-4096}"
SEAL_TOPP="${SEAL_TOPP:-0.8}"
SEAL_SEL_EPOCHS="${SEAL_SEL_EPOCHS:-2}"

SAFETY_SAMPLES="${SAFETY_SAMPLES:-4994}"    # CB / BT 모두 4994 로 맞춰져 있다

# downstream 태스크 샘플 수. 0 = 전체(= 본 실험 설정, "train set 전부 학습").
# 스모크 테스트에서만 작은 값으로 덮어써라 (예: TASK_SAMPLES=32 EPOCHS=1).
TASK_SAMPLES="${TASK_SAMPLES:-0}"

# ────────────────────────────── 안전 데이터 ──────────────────────────────
#   BT 파일은 rebuttal 의 "BeaverTails-based safety dataset, Llama-Guard-3-8B 로
#   선별하고 Circuit Breakers 와 개수를 맞춤"에 해당한다(둘 다 4994행, 동일 스키마).
safety_json() {
  case "$1" in
    cb) echo "$REPO_DIR/data/circuit_breakers_train.json" ;;
    bt) echo "$REPO_DIR/data/beavertails_cb_train.json" ;;
    *)  echo "__UNKNOWN__" ;;
  esac
}

# ────────────────────────────── 태스크 데이터 ──────────────────────────────
#   전 기법이 이 JSON 하나를 읽는다. 생성: bash scripts/revision/00_prepare.sh
#   ⚠️ agnews 는 **8k seed42 서브셋**이다(전체 120k 아님). 기존 AGNEWS 결과가 전부
#      이 서브셋으로 나왔기 때문에 여기에 맞춘다. 전체를 쓰려면 AGNEWS_TASK_JSON 을
#      덮어쓰되, 기존 수치와는 짝이 맞지 않게 된다는 점을 알고 쓸 것.
task_json() {
  case "$1" in
    gsm8k)  echo "${GSM8K_TASK_JSON:-$REPO_DIR/data/gsm8k_train_task_7473.json}" ;;
    math)   echo "${MATH_TASK_JSON:-$REPO_DIR/data/math_train_task_7500.json}" ;;
    medqa)  echo "${MEDQA_TASK_JSON:-$REPO_DIR/data/medqa_train_task_10178.json}" ;;
    arc)    echo "${ARC_TASK_JSON:-$REPO_DIR/data/arc_challenge_train_task_1119.json}" ;;
    agnews) echo "${AGNEWS_TASK_JSON:-$REPO_DIR/data/agnews_train_8k_seed42.json}" ;;
    *)      echo "__UNKNOWN__" ;;
  esac
}

# LoRA 계열 learning rate — 사용자 지정.
#   3e-4 : gsm8k / medqa / arc / math   (math 는 gsm8k 와 동일 취급하기로 확정)
#   7e-5 : agnews
lora_lr() {
  case "$1" in
    agnews) echo "${LORA_LR_AGNEWS:-7e-5}" ;;
    *)      echo "${LORA_LR_DEFAULT:-3e-4}" ;;
  esac
}

# ══════════════════════════ 모델 레지스트리 ══════════════════════════
#  필드
#    BASE          : 원본 HF 모델 (SafeLoRA / AsFT / RESTA 가 base 로 필요)
#    ALIGNED_CB    : Circuit Breakers 로 안전정렬된 출발 모델 (전부 이미 존재)
#    ALIGNED_BT    : BeaverTails 로 안전정렬된 출발 모델. 빈 문자열이면 01_ssft_bt.sh 가 학습한다.
#    PRIMARY_TASK  : 논문 Table 2/4 에서 이 모델이 쓰는 downstream
#    SSFT_LR       : Phase 0 SSFT learning rate (gemma2 만 3e-5)
#    MB_FULL       : full-param SFT micro-batch (Full FT / SafeInstr / SEAL)
#    MB_WARP       : WSR-Tune Phase 3 micro-batch (basis_coeff + Uᵀ 버퍼 때문에 더 무겁다)
#    MB_LORA       : LoRA 계열 micro-batch
#    MB_P12        : Phase 1(basis) / Phase 2(mask) micro-batch
#  ⚠️ micro-batch 는 전부 16 의 약수여야 한다(effective batch 16 유지).
#     OOM 이면 절반으로 줄여라 — 곱이 16 으로 유지되므로 결과는 동일하다.
model_cfg() {
  case "$1" in
    llama2_7b)
      BASE="meta-llama/Llama-2-7b-chat-hf"
      ALIGNED_CB="${LLAMA2_7B_ALIGNED_CB:-kmseong/llama2_7b-chat-Safety-FT-lr5e-5}"
      ALIGNED_BT="${LLAMA2_7B_ALIGNED_BT:-wvnvwn/llama2-7b-chat-lr5e-5-ssft-bv}"
      PRIMARY_TASK="${LLAMA2_7B_TASK:-gsm8k}"
      SSFT_LR="${LLAMA2_7B_SSFT_LR:-5e-5}"
      MB_FULL="${LLAMA2_7B_MB_FULL:-4}";  MB_WARP="${LLAMA2_7B_MB_WARP:-1}"
      MB_LORA="${LLAMA2_7B_MB_LORA:-4}";  MB_P12="${LLAMA2_7B_MB_P12:-2}" ;;
    llama2_13b)
      BASE="meta-llama/Llama-2-13b-chat-hf"
      ALIGNED_CB="${LLAMA2_13B_ALIGNED_CB:-wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5}"
      ALIGNED_BT="${LLAMA2_13B_ALIGNED_BT:-}"
      PRIMARY_TASK="${LLAMA2_13B_TASK:-gsm8k}"
      SSFT_LR="${LLAMA2_13B_SSFT_LR:-5e-5}"
      MB_FULL="${LLAMA2_13B_MB_FULL:-1}"; MB_WARP="${LLAMA2_13B_MB_WARP:-1}"
      MB_LORA="${LLAMA2_13B_MB_LORA:-2}"; MB_P12="${LLAMA2_13B_MB_P12:-1}" ;;
    llama32_3b)
      BASE="meta-llama/Llama-3.2-3B-Instruct"
      ALIGNED_CB="${LLAMA32_3B_ALIGNED_CB:-kmseong/llama3_2_3b-instruct-SSFT-lr5e-5}"
      ALIGNED_BT="${LLAMA32_3B_ALIGNED_BT:-}"
      PRIMARY_TASK="${LLAMA32_3B_TASK:-math}"
      SSFT_LR="${LLAMA32_3B_SSFT_LR:-5e-5}"
      MB_FULL="${LLAMA32_3B_MB_FULL:-8}"; MB_WARP="${LLAMA32_3B_MB_WARP:-2}"
      MB_LORA="${LLAMA32_3B_MB_LORA:-8}"; MB_P12="${LLAMA32_3B_MB_P12:-2}" ;;
    llama31_8b)
      BASE="meta-llama/Llama-3.1-8B-Instruct"
      ALIGNED_CB="${LLAMA31_8B_ALIGNED_CB:-kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5}"
      ALIGNED_BT="${LLAMA31_8B_ALIGNED_BT:-}"
      PRIMARY_TASK="${LLAMA31_8B_TASK:-math}"
      SSFT_LR="${LLAMA31_8B_SSFT_LR:-5e-5}"
      MB_FULL="${LLAMA31_8B_MB_FULL:-2}"; MB_WARP="${LLAMA31_8B_MB_WARP:-1}"
      MB_LORA="${LLAMA31_8B_MB_LORA:-4}"; MB_P12="${LLAMA31_8B_MB_P12:-2}" ;;
    qwen25_7b)
      BASE="Qwen/Qwen2.5-7B-Instruct"
      ALIGNED_CB="${QWEN25_7B_ALIGNED_CB:-wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5}"
      ALIGNED_BT="${QWEN25_7B_ALIGNED_BT:-}"
      PRIMARY_TASK="${QWEN25_7B_TASK:-gsm8k}"
      SSFT_LR="${QWEN25_7B_SSFT_LR:-5e-5}"
      MB_FULL="${QWEN25_7B_MB_FULL:-2}";  MB_WARP="${QWEN25_7B_MB_WARP:-1}"
      MB_LORA="${QWEN25_7B_MB_LORA:-4}";  MB_P12="${QWEN25_7B_MB_P12:-2}" ;;
    gemma2_9b)
      BASE="google/gemma-2-9b-it"
      # ⚠️ 이 모델만 SSFT lr 이 3e-5 다 (나머지 5e-5). 기존 CB 모델도 3e-5 로 만들어졌다.
      ALIGNED_CB="${GEMMA2_9B_ALIGNED_CB:-wvnvwn/gemma-2-9b-it-ssft-lr3e-5}"
      ALIGNED_BT="${GEMMA2_9B_ALIGNED_BT:-}"
      PRIMARY_TASK="${GEMMA2_9B_TASK:-gsm8k}"
      SSFT_LR="${GEMMA2_9B_SSFT_LR:-3e-5}"
      MB_FULL="${GEMMA2_9B_MB_FULL:-1}";  MB_WARP="${GEMMA2_9B_MB_WARP:-1}"
      MB_LORA="${GEMMA2_9B_MB_LORA:-2}";  MB_P12="${GEMMA2_9B_MB_P12:-1}" ;;
    *)
      echo "[common] 알 수 없는 모델 키: $1" >&2
      echo "  선택지: llama2_7b llama2_13b llama32_3b llama31_8b qwen25_7b gemma2_9b" >&2
      return 1 ;;
  esac
  return 0
}

# Figure 4 확장: llama2_7b 만 primary(gsm8k) 외에 medqa / arc / agnews 를 추가로 돈다.
FIG4_MODEL="${FIG4_MODEL:-llama2_7b}"
FIG4_EXTRA_TASKS="${FIG4_EXTRA_TASKS:-medqa arc agnews}"

# 이 (안전축, 모델) 조합을 돌리는가?  BT 는 BT_MODELS 에 든 모델만.
safety_applies() {  # <safety> <model>
  [[ "$1" != "bt" ]] && return 0
  [[ " $BT_MODELS " == *" $2 "* ]]
}

# 모델 하나가 돌아야 할 태스크 목록. 안전축을 주면 그 축의 범위를 반영한다.
tasks_for_model() {  # <model> [<safety>]
  local mkey="$1" safety="${2:-}"
  if [[ -n "$TASKS" ]]; then echo "$TASKS"; return 0; fi
  if [[ "$safety" == "bt" ]]; then echo "$BT_TASKS"; return 0; fi
  model_cfg "$mkey" || return 1
  if [[ "$mkey" == "$FIG4_MODEL" ]]; then
    echo "$PRIMARY_TASK $FIG4_EXTRA_TASKS"
  else
    echo "$PRIMARY_TASK"
  fi
}

# ══════════════ 이미 논문에 실려 재사용하는 셀 ══════════════
#  **엄격 규칙**: 출발 모델·lr·epoch·effective batch 가 이번 설정과 일치함을 확인한
#  것만 재사용한다. 확인된 것은 논문 본문 표의 full-param 5종뿐이다.
#
#    논문 Table 2 / Table 4 / Table 10(Figure 4) 의
#       Full FT · SafeInstr · Resta · SafeDelta · WSR-Tune
#    → scripts/run_all_phases_integrated.sh 의 PHASE0_MODEL =
#       kmseong/llama2_7b-chat-Safety-FT-lr5e-5, lr 5e-5, 3 epoch, eff.batch 16.
#       RESTA γ=0.3 / SafeDelta s=0.1 도 논문 §4.1 과 일치.
#
#  ── rebuttal 의 PEFT 수치를 재사용하지 않는 이유 (2026-08-26 확인) ──────────
#   1) 출발 모델이 다르다. MedQA/ARC 의 LISA/SafeLoRA/AsFT 를 돌린
#      scripts/run_lisa_safelora_asft_qa.sh 는 MODEL=wvnvwn/llama2-7b-chat-lr5e-5-ssft-cb
#      를 썼다. 허브에서 이 리포와 kmseong/llama2_7b-chat-Safety-FT-lr5e-5 는
#      safetensors sha256 이 셋 다 다르다(총 바이트는 8바이트 차 — 같은 가중치를 다르게
#      샤딩했을 가능성도 있으나 확인되지 않았다).
#   2) SafeLoRA threshold 가 다르다. run_safelora_thr_sweep_qa.sh 는 THRS="0.15 0.25",
#      기준선 0.35 였다. **0.3 은 MedQA/ARC 에서 돌린 적이 없다.**
#   3) AGNEWS 는 동작점 자체가 다르다. rebuttal AGNEWS 표는 분류 베이스라인 라인
#      (epoch 1, lr 1e-5, weight_decay 0)에서 나왔다. 이번 설정은 epoch 3, lr 5e-5, wd 0.01.
#
#  ⚠️ 알려진 잔여 차이: 논문 MedQA 실험은 10000 샘플을 썼고(run_all_phases_integrated.sh
#     의 MEDQA_SAMPLES=10000), 이번 신규 셀은 전체 10178 샘플을 쓴다(1.7% 차이).
#     재사용하는 MedQA 기준행 5개만 그 조건이다. 맞추려면 MEDQA_TASK_SAMPLES 를 쓰거나
#     SKIP_PUBLISHED=0 으로 그 5개도 다시 돌려라.
already_published() {  # <safety> <model> <task> <method>
  [[ "$SKIP_PUBLISHED" == "1" ]] || return 1
  # BT 축은 전부 새로 돌린다 (모든 기법 적용).
  [[ "$1" == "bt" ]] && return 1
  case "$4" in fullft|safeinstr|resta|safedelta|wsr_tune) ;; *) return 1 ;; esac
  case "$1/$2/$3" in
    cb/llama2_7b/gsm8k)     return 0 ;;   # 논문 Table 2
    cb/llama2_7b/medqa)     return 0 ;;   # 논문 Table 10 / Figure 4
    cb/llama2_7b/arc)       return 0 ;;   # 논문 Table 10 / Figure 4
    cb/llama2_13b/gsm8k)    return 0 ;;   # 논문 Table 2
    cb/llama32_3b/math)     return 0 ;;   # 논문 Table 2
    cb/llama31_8b/math)     return 0 ;;   # 논문 Table 2
    cb/qwen25_7b/gsm8k)     return 0 ;;   # 논문 Table 4
    cb/gemma2_9b/gsm8k)     return 0 ;;   # 논문 Table 4
    # cb/llama2_7b/agnews 는 재사용하지 않는다 — 동작점이 다르다(위 3번).
  esac
  return 1
}

# 이 셀을 이번에 돌려야 하는가 (선택자 + 기발표 여부).
cell_wanted() {  # <safety> <model> <task> <method>
  safety_applies "$1" "$2" || return 1
  already_published "$1" "$2" "$3" "$4" && return 1
  return 0
}

# BT SSFT 출력 디렉토리.
# ⚠️ 디렉토리 **이름**에 base 모델 식별자(chat / instruct / -it)를 반드시 남긴다.
#    모든 러너의 is_instruct_model() 이 모델 참조 **문자열**로 chat template 사용 여부를
#    판정하기 때문이다. `ssft_bt/llama2_13b_lr5e-5` 처럼 태그가 없는 이름을 쓰면
#    chat template 대신 plain 프롬프트로 fallback 되어 CB arm 과 포맷이 어긋난다.
bt_ssft_dir() {
  local mkey="$1"
  model_cfg "$mkey" || return 1
  echo "$CKPT_ROOT/ssft_bt/$(basename "$BASE")-ssft-bt-lr${SSFT_LR}"
}

# (모델, 안전데이터) → 출발 모델. 없으면 01_ssft_bt.sh 가 만들 로컬 경로를 돌려준다.
aligned_for() {
  local mkey="$1" safety="$2"
  model_cfg "$mkey" || return 1
  case "$safety" in
    cb) echo "$ALIGNED_CB" ;;
    bt)
      if [[ -n "$ALIGNED_BT" ]]; then echo "$ALIGNED_BT"; return 0; fi
      local d; d="$(bt_ssft_dir "$mkey")"
      # 로컬이 정리됐지만 허브에 올라가 있으면 허브 리포를 출발점으로 쓴다.
      if [[ ! -f "$d/config.json" && -s "$d/.uploaded" ]]; then
        tr -d '\n' < "$d/.uploaded"; echo
      else
        echo "$d"
      fi ;;
    *)  echo "__UNKNOWN__"; return 1 ;;
  esac
}

# ────────────────────────────── 헬퍼 ──────────────────────────────
die()  { echo "[revision][ERROR] $*" >&2; exit 1; }
warn() { echo "[revision][WARN ] $*" >&2; }
log()  { echo "[revision] $*"; }
hdr()  { echo ""; echo "──────────────────────────────────────────────────────────────"; echo "  $*   ($(date '+%m-%d %H:%M:%S'))"; echo "──────────────────────────────────────────────────────────────"; }

# effective batch 16 을 유지하는 grad_accum 계산 (나누어떨어지지 않으면 실패)
accum_for() {
  local micro="$1"
  (( micro > 0 )) || { echo "[common] micro-batch 는 양수여야 한다: $micro" >&2; return 1; }
  (( EFFECTIVE_BATCH % micro == 0 )) || {
    echo "[common] micro-batch($micro) 가 effective batch($EFFECTIVE_BATCH) 의 약수가 아니다" >&2; return 1; }
  echo $(( EFFECTIVE_BATCH / micro ))
}

# 출력 경로 규약:  outputs/revision/<safety>/<model>/<task>/<method>
out_dir() { echo "$OUT_ROOT/$1/$2/$3/$4"; }

is_done()   { [[ -f "$1/.done" ]]; }
mark_done() { date -Iseconds > "$1/.done"; }

# 러너마다 모델을 저장하는 위치가 다르다(<out> 직접 / <out>/merged_model).
# 평가 쪽이 균일하게 읽을 수 있도록 셀마다 MODEL_DIR 파일에 실제 모델 경로를 못박는다.
#   write_model_ptr <cell_dir> [<model_dir, 기본=자동탐지>]
write_model_ptr() {
  local cell="$1" mdir="${2:-}"
  if [[ -z "$mdir" ]]; then
    if   [[ -f "$cell/merged_model/config.json" ]]; then mdir="$cell/merged_model"
    elif [[ -f "$cell/config.json" ]];              then mdir="$cell"
    else warn "모델 디렉토리를 찾지 못했다: $cell"; return 1; fi
  fi
  echo "$mdir" > "$cell/MODEL_DIR"
}

# ══════════════════════ HF 업로드 · 로컬 정리 ══════════════════════
#  리포명 규약 (사용자 지정 형식을 다듬은 것):
#     {ns}/{model}-{CB|BT}_SSFT-{method}_{task}[_{hparam}]_lr{lr}
#  · 하이퍼파라미터가 없는 기법(fullft/lora)은 그 슬롯을 **생략**한다(빈 `__` 방지).
#  · lr 은 기법에서 자동 도출한다(full-param 5e-5 / LoRA 3e-4, agnews 만 7e-5).
#  · 기법명의 `_` 는 `-` 로 바꾼다(wsr_tune → wsr-tune). 필드 구분자 `_` 와 충돌하므로.
#  · LoRA 계열은 전부 r=16/alpha=32 로 동일해 이름에 넣지 않는다(README 에 명시).
hf_model_tag() {
  case "$1" in
    llama2_7b)  echo "llama2_7b-chat" ;;
    llama2_13b) echo "llama2_13b-chat" ;;
    llama32_3b) echo "llama3_2_3b-instruct" ;;
    llama31_8b) echo "llama3_1_8b-instruct" ;;
    qwen25_7b)  echo "qwen2_5_7b-instruct" ;;
    gemma2_9b)  echo "gemma2_9b-it" ;;
    *) echo "$1" ;;
  esac
}

hf_safety_tag() { case "$1" in cb) echo CB ;; bt) echo BT ;; *) echo "${1^^}" ;; esac; }

hf_method_tag() {
  case "$1" in
    wsr_tune) echo "wsr-tune" ;;
    wsr_lora) echo "wsr-lora" ;;
    *) echo "$1" ;;
  esac
}

# 기법별 하이퍼파라미터 태그. 없으면 빈 문자열.
hf_hparam_tag() {
  case "$1" in
    safeinstr) echo "mix${SAFEINSTR_RATIO}" ;;
    resta)     echo "gamma${RESTA_GAMMA}" ;;
    safedelta) echo "s${SAFEDELTA_SCALE}" ;;
    wsr_tune)  echo "rho${KEEP_RATIO}" ;;
    wsr_lora)  echo "rho${KEEP_RATIO}" ;;
    asft)      echo "lambda${ASFT_LAMBDA_REG}" ;;
    lisa)      echo "rho${LISA_RHO}" ;;
    seal)      echo "topp${SEAL_TOPP}" ;;
    safelora)  echo "thr${SAFELORA_THRESHOLD}" ;;
    salora)    echo "rs${SALORA_R_S}rt${SALORA_R_T}" ;;
    *)         echo "" ;;   # fullft / lora
  esac
}

# 기법 × 태스크 → 실제로 쓴 learning rate.
#   full-param 계열(fullft/safeinstr/wsr_tune/seal)은 FULL_LR.
#   post-hoc(resta/safedelta)은 입력이 된 fullft 의 lr 을 따른다.
hf_lr_tag() {
  case "$1" in
    fullft|safeinstr|wsr_tune|seal|resta|safedelta) echo "$FULL_LR" ;;
    *) lora_lr "$2" ;;
  esac
}

hf_repo_id() {  # <safety> <model> <task> <method>
  local hp; hp="$(hf_hparam_tag "$4")"
  local mid="$(hf_model_tag "$2")-$(hf_safety_tag "$1")_SSFT-$(hf_method_tag "$4")_${3}"
  [[ -n "$hp" ]] && mid="${mid}_${hp}"
  echo "${HF_NAMESPACE}/${mid}_lr$(hf_lr_tag "$4" "$3")"
}

hf_ssft_repo_id() {  # <model> <safety>   BT 안전정렬 출발모델용
  echo "${HF_NAMESPACE}/$(hf_model_tag "$1")-$(hf_safety_tag "$2")_SSFT-lr${SSFT_LR}"
}

# fullft 는 RESTA/SafeDelta 의 **입력**이다. 그 둘이 아직 안 끝났으면 지우면 안 된다.
prune_allowed() {  # <safety> <model> <task> <method>
  local safety="$1" mkey="$2" task="$3" method="$4" consumer d
  [[ "$method" != "fullft" ]] && return 0
  for consumer in resta safedelta; do
    has_method "$consumer" || continue        # 이번 실행에서 안 돌리면 기다릴 필요 없다
    d="$(out_dir "$safety" "$mkey" "$task" "$consumer")"
    is_done "$d" || return 1
  done
  return 0
}

# 셀 하나를 업로드하고(성공 시) 로컬 가중치를 지운다.
#   PUSH_TO_HUB=1 일 때만 동작. 검증에 실패하면 로컬을 남긴다.
upload_cell() {  # <safety> <model> <task> <method>
  [[ "$PUSH_TO_HUB" == "1" ]] || return 0
  [[ "$DRY_RUN" == "1" ]] && return 0
  local safety="$1" mkey="$2" task="$3" method="$4"
  local cell repo prune_flag=()
  cell="$(out_dir "$safety" "$mkey" "$task" "$method")"
  is_done "$cell" || return 0
  repo="$(hf_repo_id "$safety" "$mkey" "$task" "$method")"

  if [[ -f "$cell/.uploaded" ]]; then
    # 이미 올렸다. 그때 prune 을 못 했다면(=fullft 대기) 지금 조건이 풀렸는지 본다.
    if [[ "$PRUNE_AFTER_UPLOAD" == "1" ]] && prune_allowed "$safety" "$mkey" "$task" "$method"; then
      "$PY" scripts/revision/upload_and_prune.py --cell_dir "$cell" --repo_id "$repo" \
            --verify_only --prune 2>&1 | sed 's/^/    /'
    fi
    return 0
  fi

  if [[ "$PRUNE_AFTER_UPLOAD" == "1" ]] && prune_allowed "$safety" "$mkey" "$task" "$method"; then
    prune_flag=(--prune)
  elif [[ "$PRUNE_AFTER_UPLOAD" == "1" ]]; then
    log "  ($method 은 RESTA/SafeDelta 의 입력이라 그 둘이 끝난 뒤에 삭제한다)"
  fi

  hdr "HF 업로드  $safety/$mkey/$task/$method  →  $repo"
  if "$PY" scripts/revision/upload_and_prune.py \
        --cell_dir "$cell" --repo_id "$repo" "${prune_flag[@]}" 2>&1 | sed 's/^/    /'; then
    :
  else
    warn "업로드/검증 실패: $repo  (로컬은 보존됨)"
    FAILED_CELLS+=("upload/$safety/$mkey/$task/$method")
  fi
}

# 한 모델을 다 쓴 뒤 그 모델의 HF 캐시 스냅샷을 지운다.
#   base + CB/BT aligned 가 각각 15~26GB 를 차지한다. 다시 필요하면 재다운로드된다.
prune_hf_cache_for_model() {  # <model>
  [[ "$PRUNE_HF_CACHE" == "1" ]] || return 0
  [[ "$DRY_RUN" == "1" ]] && return 0
  local mkey="$1" hub="${HF_HOME:-$HOME/.cache/huggingface}/hub" ref d freed=0
  model_cfg "$mkey" || return 0
  for ref in "$BASE" "$ALIGNED_CB" "$ALIGNED_BT"; do
    [[ -z "$ref" || "$ref" == /* ]] && continue           # 로컬 경로는 캐시가 아니다
    d="$hub/models--${ref//\//--}"
    [[ -d "$d" ]] || continue
    freed=$(( freed + $(du -sm "$d" 2>/dev/null | cut -f1) ))
    rm -rf "$d"
    log "[prune] HF 캐시 삭제: $ref"
  done
  (( freed > 0 )) && log "[prune] HF 캐시에서 ${freed}MB 회수"
  return 0
}

# WSR 계열이 전부 끝난 (safety, model) 의 basis/mask 를 지운다.
#   basis 는 조합당 최대 30GB 라 남겨두면 디스크가 버티지 못한다.
prune_basis_if_done() {  # <safety> <model>
  [[ "$PRUNE_BASIS" == "1" ]] || return 0
  [[ "$DRY_RUN" == "1" ]] && return 0
  # 이번 실행에 WSR 계열이 아예 없으면 basis 는 내 소관이 아니다 — 건드리지 않는다.
  # (없는데도 아래 루프를 돌면 "쓸 곳이 없다"로 판정되어 남의 basis 를 지워버린다.)
  has_method wsr_tune || has_method wsr_lora || return 0
  local safety="$1" mkey="$2" task d
  for task in $(tasks_for_model "$mkey"); do
    for m in wsr_tune wsr_lora; do
      has_method "$m" || continue
      d="$(out_dir "$safety" "$mkey" "$task" "$m")"
      is_done "$d" || return 0     # 아직 쓸 곳이 남았다
    done
  done
  local base="$CKPT_ROOT/warp/$safety/$mkey"
  local freed
  freed=$(du -sb "$base" 2>/dev/null | cut -f1)
  [[ -z "$freed" ]] && return 0
  # 포인터 파일과 로그는 남기고 phase1_*/phase2_* 실체만 지운다(재생성 여부 판단용).
  find "$base" -maxdepth 1 -type d \( -name 'phase1_*' -o -name 'phase2_*' \) -exec rm -rf {} + 2>/dev/null
  log "[prune] basis/mask 삭제: $safety/$mkey (~$(( freed / 1024 / 1024 / 1024 ))GB 회수)"
  echo "pruned $(date -Iseconds)" > "$base/PRUNED"
}

# 셀이 성공했을 때만 MODEL_DIR 을 쓴다. dry-run 에서는 아무것도 하지 않는다.
# (`A || B && C` 형태의 짧은 조건식은 bash 우선순위 때문에 dry-run 에서도 실행되므로 쓰지 말 것.)
#   post_cell <cell_dir> <safety> <model> <task> <method>
#   MODEL_DIR 기록 → (PUSH_TO_HUB=1 이면) HF 업로드 → 검증 성공 시 로컬 가중치 삭제.
post_cell() {
  local cell="$1"
  [[ "$DRY_RUN" == "1" ]] && return 0
  is_done "$cell" || return 0
  write_model_ptr "$cell" || true
  if (( $# >= 5 )); then
    upload_cell "$2" "$3" "$4" "$5"
  fi
}

# 셀의 모델 경로를 읽는다 (없으면 자동탐지).
read_model_ptr() {
  local cell="$1"
  if [[ -s "$cell/MODEL_DIR" ]]; then cat "$cell/MODEL_DIR"; return 0; fi
  if   [[ -f "$cell/merged_model/config.json" ]]; then echo "$cell/merged_model"
  elif [[ -f "$cell/config.json" ]];              then echo "$cell"
  else return 1; fi
}

has_method() { [[ " $METHODS " == *" $1 "* ]]; }

# 스테이지 스크립트의 기법 분기에서 쓰는 게이트.
#   METHODS 에 있고 + 이 (안전축,모델) 조합이 유효하고 + 아직 논문에 없는 셀일 때만 참.
want_cell() {  # <safety> <model> <task> <method>
  has_method "$4" || return 1
  cell_wanted "$1" "$2" "$3" "$4"
}

# dry-run 인식 실행기
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'; return 0
  fi
  "$@"
}

# 무인 실행용 마감 시각. REVISION_DEADLINE_EPOCH 를 넘기면 **새 셀을 시작하지 않는다**.
# (실행 중이던 셀은 끝까지 간다 — 중간에 죽이면 반쪽짜리 체크포인트가 남는다.)
deadline_passed() {
  [[ -n "${REVISION_DEADLINE_EPOCH:-}" ]] || return 1
  (( $(date +%s) >= REVISION_DEADLINE_EPOCH ))
}

# ── 디스크 안전장치 ───────────────────────────────────────────────────────
#  밤새 무인으로 도는 동안 가장 현실적인 사고는 디스크 고갈이다.
#  업로드가 실패하면 그 셀의 가중치가 삭제되지 않고 쌓이는데, 그게 몇 개 겹치면
#  이후 모든 셀이 저장 단계에서 죽는다. 그래서 새 셀을 시작하기 전에
#    (1) 이미 허브에 올라갔는데 로컬에 남아 있는 셀을 회수하고,
#    (2) 그래도 모자라면 그 셀을 **시작하지 않고** 건너뛴다(재실행하면 이어서 간다).
MIN_FREE_GB="${MIN_FREE_GB:-45}"

avail_gb() { df -BG --output=avail "$OUT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# 업로드 검증까지 끝났는데 아직 가중치가 남아 있는 셀을 지운다.
#  · .uploaded 가 있다 = 허브에 올라가고 4종 검증을 통과했다는 뜻이라 안전하다.
#  · fullft 은 RESTA/SafeDelta 가 아직이면 prune_allowed 가 막는다.
reclaim_disk() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local cell repo n=0
  while IFS= read -r up; do
    cell="$(dirname "$up")"
    # <out>/<safety>/<model>/<task>/<method>
    local method task mkey safety
    method="$(basename "$cell")"; task="$(basename "$(dirname "$cell")")"
    mkey="$(basename "$(dirname "$(dirname "$cell")")")"
    safety="$(basename "$(dirname "$(dirname "$(dirname "$cell")")")")"
    # 가중치가 남아 있나?
    find "$cell" \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null | grep -q . || continue
    prune_allowed "$safety" "$mkey" "$task" "$method" || continue
    repo="$(cat "$up")"
    log "[reclaim] $safety/$mkey/$task/$method — 이미 업로드됨, 로컬 가중치 회수"
    "$PY" scripts/revision/upload_and_prune.py --cell_dir "$cell" --repo_id "$repo" \
          --verify_only --prune 2>&1 | sed 's/^/    /'
    n=$((n+1))
  done < <(find "$OUT_ROOT" -name .uploaded 2>/dev/null)
  (( n > 0 )) && log "[reclaim] ${n}개 셀 정리 · 여유 $(avail_gb)GB"
  return 0
}

# 새 셀을 시작해도 되는가. 부족하면 회수 후 재확인.
disk_ok() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local a; a="$(avail_gb)"; a="${a:-0}"
  (( a >= MIN_FREE_GB )) && return 0
  warn "[disk] 여유 ${a}GB < ${MIN_FREE_GB}GB — 업로드 끝난 셀부터 회수한다"
  reclaim_disk
  a="$(avail_gb)"; a="${a:-0}"
  (( a >= MIN_FREE_GB )) && { log "[disk] 회수 후 ${a}GB — 계속 진행"; return 0; }
  warn "[disk] 회수 후에도 ${a}GB 뿐이다 — 이 셀은 시작하지 않는다(재실행하면 이어서 간다)"
  return 1
}

# 한 셀을 실행하고 성공 시 .done 을 남긴다.
#   run_cell <out_dir> <label> -- <command...>
run_cell() {
  local odir="$1"; shift
  local label="$1"; shift
  [[ "$1" == "--" ]] && shift
  if is_done "$odir"; then log "[skip] $label  (이미 완료: $odir)"; return 0; fi
  if deadline_passed; then
    log "[deadline] $label — 마감 시각을 넘겨 시작하지 않는다"
    return 0
  fi
  if ! disk_ok; then
    FAILED_CELLS+=("$label (disk)")
    return 0
  fi
  hdr "$label"
  mkdir -p "$odir"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'; return 0
  fi
  if "$@" 2>&1 | tee "$odir/run.log"; then
    # tee 뒤의 종료코드는 파이프 첫 명령의 것을 봐야 한다
    local rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then mark_done "$odir"; log "[done] $label"; return 0; fi
    warn "[fail rc=$rc] $label   로그: $odir/run.log"
  else
    warn "[fail] $label   로그: $odir/run.log"
  fi
  FAILED_CELLS+=("$label")
  [[ "$CONTINUE_ON_ERROR" == "1" ]] && return 0
  return 1
}

# HF push 인자 (PUSH_TO_HUB=1 일 때만). 리포명은 손으로 적지 않고 규약에서 생성한다.
hub_repo_name() {  # <safety> <model> <task> <method-tag>
  echo "${HF_NAMESPACE}/${2}-${3}-${4}-${1}"
}

declare -a FAILED_CELLS=()

print_failures() {
  if (( ${#FAILED_CELLS[@]} > 0 )); then
    echo ""
    echo "════════════════ 실패한 셀 (${#FAILED_CELLS[@]}) ════════════════"
    printf '  - %s\n' "${FAILED_CELLS[@]}"
    return 1
  fi
  echo ""
  echo "실패한 셀 없음."
  return 0
}

# 스테이지 스크립트가 공통으로 부르는 사전 점검
preflight() {
  [[ -x "$(command -v "$PY")" ]] || die "python 을 찾을 수 없다 (PY=$PY). conda activate hb 했는가?"
  for s in $SAFETY_SETS; do
    local p; p="$(safety_json "$s")"
    [[ "$p" == "__UNKNOWN__" ]] && die "알 수 없는 safety set: $s (cb|bt)"
    [[ -f "$p" ]] || die "안전 데이터 없음: $p"
  done
  if [[ "$PUSH_TO_HUB" == "1" && -z "$HF_NAMESPACE" ]]; then
    die "PUSH_TO_HUB=1 이면 HF_NAMESPACE 가 필요하다"
  fi
  mkdir -p "$OUT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"
}

# 모델별 셀 1개가 차지하는 대략적인 디스크(GB). bf16 safetensors 기준.
#   3B≈6, 7B/8B≈15, 9B≈19, 13B≈26.  RESTA/SafeDelta 도 같은 크기의 전체 모델을 저장한다.
cell_gb() {
  case "$1" in
    llama32_3b) echo 7 ;;
    llama2_7b|qwen25_7b|llama31_8b) echo 15 ;;
    gemma2_9b) echo 19 ;;
    llama2_13b) echo 26 ;;
    *) echo 15 ;;
  esac
}

# 모델별 Phase1 basis + Phase2 mask 디스크(GB). BASIS_SAVE_DTYPE=bfloat16 + BASIS_OMIT_UT=1 기준.
#   (fp32 + UT 저장이면 4배다. config 의 L/hidden/intermediate 로 계산한 값.)
basis_gb() {
  case "$1" in
    llama2_7b)  echo 16 ;;
    llama2_13b) echo 31 ;;
    llama32_3b) echo 8  ;;
    llama31_8b) echo 21 ;;
    qwen25_7b)  echo 26 ;;
    gemma2_9b)  echo 26 ;;
    *) echo 30 ;;
  esac
}

# 남은 셀이 필요로 하는 디스크를 추정하고 부족하면 경고한다.
#   216 셀 전체는 3TB 를 넘는다. 단일 디스크에 다 넣을 수 없으므로 반드시 확인할 것.
check_disk() {
  local need=0 mkey task method d
  for safety in $SAFETY_SETS; do
    for mkey in $MODELS; do
      model_cfg "$mkey" || continue
      local g; g="$(cell_gb "$mkey")"
      safety_applies "$safety" "$mkey" || continue
      for task in $(tasks_for_model "$mkey" "$safety"); do
        for method in $METHODS; do
          cell_wanted "$safety" "$mkey" "$task" "$method" || continue
          d="$(out_dir "$safety" "$mkey" "$task" "$method")"
          is_done "$d" || need=$(( need + g ))
        done
      done
    done
  done
  local avail; avail=$(df -BG --output=avail "$OUT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
  avail="${avail:-0}"
  local mount; mount="$(df --output=target "$OUT_ROOT" 2>/dev/null | tail -1)"
  echo ""
  if [[ "$PUSH_TO_HUB" == "1" && "$PRUNE_AFTER_UPLOAD" == "1" ]]; then
    # 업로드 후 삭제 모드: 누적이 아니라 **동시 최대**가 기준이다.
    #   한 (모델, 안전축) 조합을 처리하는 동안 디스크에 동시에 존재하는 것 =
    #   HF 캐시(base+aligned) + basis/mask + fullft(post-hoc 대기) + 현재 셀.
    local peak=0 g b pk mkey tight=""
    printf "  디스크: 업로드-후-삭제 모드 · %s 여유 %dGB   (셀 누적 %dGB 는 허브로 나간다)\n" \
           "$mount" "$avail" "$need"
    printf "          %-12s %8s %8s %8s %9s\n" model "HF캐시" "basis" "셀×2" "동시최대"
    for mkey in $MODELS; do
      model_cfg "$mkey" || continue
      g="$(cell_gb "$mkey")"; b="$(basis_gb "$mkey")"
      pk=$(( 3*g + b + 2*g ))          # base+CB+BT 캐시 3벌 + basis/mask + (fullft 대기 + 현재 셀)
      (( pk > peak )) && peak=$pk
      local flag=""
      if (( pk > avail )); then flag="  ← 빠듯"; tight="$tight $mkey"; fi
      printf "          %-12s %7dG %7dG %7dG %8dG%s\n" "$mkey" $((3*g)) "$b" $((2*g)) "$pk" "$flag"
    done
    if [[ -n "$tight" ]]; then
      warn "다음 모델은 현재 여유로는 빠듯하다:$tight"
      warn "  · 그 모델만 단독으로 돌리고(MODELS=<key>), 다른 모델 캐시를 먼저 비워라."
      warn "  · 또는 METHODS 를 둘로 쪼개 돌려라 (예: 먼저 full-param 5종, 다음 LoRA 7종)."
    fi
  else
    printf "  디스크: 남은 셀 추정 %dGB 필요 · %s 여유 %dGB\n" "$need" "$mount" "$avail"
    if (( need > avail )); then
      warn "디스크가 부족하다 (필요 ~${need}GB > 여유 ${avail}GB)."
      warn "  · PUSH_TO_HUB=1 로 켜면 셀마다 허브에 올리고 로컬을 지운다(권장)."
      warn "  · 또는 OUT_ROOT 를 큰 볼륨으로 옮기거나 MODELS/METHODS 로 쪼개 돌려라."
    fi
  fi
}

# 이번 실행이 다룰 (safety, model, task) 조합을 점검하고 요약 출력
print_plan() {
  echo "════════════════════════════════════════════════════════════════"
  echo " Revision 실험 계획"
  echo "   repo      : $REPO_DIR"
  echo "   python    : $($PY -c 'import sys;print(sys.executable)' 2>/dev/null || echo "$PY")"
  echo "   safety    : $SAFETY_SETS"
  echo "   models    : $MODELS"
  echo "   methods   : $METHODS"
  echo "   tasks     : ${TASKS:-<모델별 자동>}"
  echo "   eff.batch : $EFFECTIVE_BATCH   epochs: $EPOCHS   max_len: $MAX_LENGTH   seed: $SEED"
  if [[ "$TASK_SAMPLES" != "0" ]]; then
    echo "   ⚠️ TASK_SAMPLES=$TASK_SAMPLES  (전체가 아니다 — 스모크 테스트 설정)"
  fi
  if [[ "$SAFETY_SAMPLES" != "4994" ]]; then
    echo "   ⚠️ SAFETY_SAMPLES=$SAFETY_SAMPLES  (기본 4994 가 아니다 — 스모크 테스트 설정)"
  fi
  echo "   full-param: lr=$FULL_LR wd=$FULL_WEIGHT_DECAY warmup=$FULL_WARMUP_RATIO"
  echo "   lora      : r=$LORA_R a=$LORA_ALPHA drop=$LORA_DROPOUT wd=$LORA_WEIGHT_DECAY warmup=$LORA_WARMUP_RATIO"
  echo "               lr=$(lora_lr gsm8k) (gsm8k/math/medqa/arc) · $(lora_lr agnews) (agnews)"
  echo "   out       : $OUT_ROOT"
  if [[ "$PUSH_TO_HUB" == "1" ]]; then
    echo "   hf        : $HF_NAMESPACE  (업로드 후 로컬 삭제=$PRUNE_AFTER_UPLOAD, basis 삭제=$PRUNE_BASIS)"
  else
    echo "   hf        : 업로드 안 함 (PUSH_TO_HUB=0) — 디스크 3.4TB 필요"
  fi
  [[ "$DRY_RUN" == "1" ]] && echo "   *** DRY RUN ***"
  echo "════════════════════════════════════════════════════════════════"
}
