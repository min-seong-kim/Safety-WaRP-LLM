#!/usr/bin/env bash
# Qwen2.5-7B-Instruct / GSM8K 를 WSR-Tune(full-parameter, 재파라미터화 공간) 으로 학습·평가한다.
#
#   Stage 02  Phase 1 basis(U) + Phase 2 mask(M)   ← Qwen 은 아직 없다
#   Stage 12  Phase 3 (basis_coeff 만 학습)
#   평가      HarmBench 4공격(sys, GRADING=hard) + lm-eval GSM8K 5-shot
#
# ⚠️ SKIP_PUBLISHED=0 이 필수다. common.sh 의 already_published() 가
#    cb/qwen25_7b/gsm8k 의 full-param 5종(wsr_tune 포함)을 "논문 Table 4 에 이미 있다"
#    며 건너뛴다. 여기서는 그 행을 이 박스에서 다시 만드는 것이 목적이다.
#
# 비교 대상: wvnvwn/qwen-2.5-7B-Instruct-WaRP-lr5e-5 (AVG 0.0289 / GSM8K 0.6945)
# 기준행   : wvnvwn/qwen-2.5-7B-Instruct-SSFT-gsm8k-lr5e-5 (Full FT, AVG 0.0362 / 0.6732)
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb

export SAFETY_SETS=cb MODELS=qwen25_7b TASKS=gsm8k METHODS=wsr_tune
export SKIP_PUBLISHED=0
export PUSH_TO_HUB=1
# 평가를 로컬 가중치로 한다(이 박스는 다운로드 ≈9MB/s). basis 도 남겨 재사용한다.
export PRUNE_AFTER_UPLOAD=0 PRUNE_BASIS=0 PRUNE_HF_CACHE=0
export LOG_ROOT="$PWD/logs/revision_qwen_wsrtune"

echo "═══ [1/3] Stage 02 — Phase 1 basis + Phase 2 mask ═══ $(date -Iseconds)"
bash scripts/revision/02_warp_basis_mask.sh; rc02=$?
echo "STAGE02_RC=$rc02"

B="$PWD/checkpoints/revision/warp/cb/qwen25_7b/BASIS_DIR"
M="$PWD/checkpoints/revision/warp/cb/qwen25_7b/MASKS_DIR"
[ -s "$M" ] || M="$PWD/checkpoints/revision/warp/cb/qwen25_7b/MASK_DIR"
if [ ! -s "$B" ] || [ ! -s "$M" ]; then
  echo "❌ basis/mask 포인터가 없다 — Phase 3 를 시작하지 않는다"; exit 1
fi
echo "  basis: $(cat "$B")"
echo "  mask : $(cat "$M")"

echo "═══ [2/3] Stage 12 — Phase 3 (WSR-Tune) ═══ $(date -Iseconds)"
bash scripts/revision/12_wsr_tune.sh; rc12=$?
echo "STAGE12_RC=$rc12"

CELL="outputs/revision/cb/qwen25_7b/gsm8k/wsr_tune"
if [ ! -f "$CELL/.done" ]; then echo "❌ 학습 미완료 — 평가 생략"; exit 1; fi

echo "═══ [3/3] 평가 ═══ $(date -Iseconds)"
# 로컬 가중치를 허브 리포명과 같은 이름으로 링크한다(평가 key 가 허브 평가와 같아진다).
NAME="qwen2_5_7b-instruct-CB_SSFT-wsr-tune_gsm8k_rho0.1_lr5e-5"
MD="$(cat "$CELL/MODEL_DIR" 2>/dev/null)"
if [ -z "$MD" ] || [ ! -d "$MD" ]; then echo "❌ MODEL_DIR 없음"; exit 1; fi
mkdir -p outputs/eval_local
ln -sfn "$MD" "outputs/eval_local/$NAME"
PREFETCH=0 REPOS_ONLY="$PWD/outputs/eval_local/$NAME" bash scripts/revision/32_eval_cells.sh
echo "QWEN_WSRTUNE_DONE rc=$?  $(date -Iseconds)"
