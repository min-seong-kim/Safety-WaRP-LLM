#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  CB(medqa/arc) 학습이 끝나면 → BT(gsm8k) 학습 → 21개 셀 전부 한 번에 평가
#
#  BT/GSM8K 7종은 RESULTS.md 326~331행(LoRA 계열 6종) + 324행(SEAL, 기존 "없음")에
#  해당한다. 기존 행은 α=32 세대이고 AsFT 는 원본 ASR 기록이 충돌, Lisa 는 ρ 미상이라
#  전부 α=16(scaling 1.0) · LISA ρ=1.0 으로 다시 만든다.
#
#  ⚠️ 대기 판정은 **로그의 완료 마커**로 한다. `pgrep -f <script>` 는 이 스크립트
#     자신과 감시 셸까지 잡아서 무한 대기에 빠진다(CLAUDE.md 에 기록된 트랩).
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
M=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3
B=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong
R="$B/Safety-WaRP-LLM"
source "$M/etc/profile.d/conda.sh"
cd "$R"

CB_LOG="$(cat "$R/logs/qa_a16.logpath")"
DONE_MARK='############ 종료:'

# ── 1) CB 학습 완료 대기 ────────────────────────────────────────────────────
echo "[chain] CB 학습 완료 대기 시작 ($(date '+%F %T')) — 마커='$DONE_MARK'"
waited=0
while ! grep -qaF "$DONE_MARK" "$CB_LOG"; do
    sleep 120; waited=$((waited+120))
    if (( waited % 1800 == 0 )); then
        echo "[chain] 대기 중 $((waited/60))분 — 완료 셀 $(find "$R/outputs/revision_qa_a16" -name .uploaded 2>/dev/null | wc -l)/14"
    fi
    (( waited > 86400 )) && { echo "[chain] 24시간 초과 — 중단"; exit 1; }
done
echo "[chain] CB 학습 완료 감지 ($(date '+%F %T')) — CB 업로드 $(find "$R/outputs/revision_qa_a16" -name .uploaded | wc -l)개"

# ── 2) BT 축 학습 (gsm8k) ───────────────────────────────────────────────────
conda activate hb
export CUDA_VISIBLE_DEVICES=0
export SAFETY_SETS=bt MODELS=llama2_7b TASKS="gsm8k" BT_TASKS="gsm8k"
export LORA_ALPHA=16 WSR_LORA_ALPHA=16
export LISA_RHO=1.0 SAFELORA_THRESHOLD=0.3 KEEP_RATIO=0.3 EPOCHS=3
export PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 HF_NAMESPACE=kmseong
export OUT_ROOT="$R/outputs/revision_qa_a16"
mkdir -p "$OUT_ROOT"

echo "############ BT Stage 02 — Phase1 basis (bt/llama2_7b) ############"
METHODS="wsr_lora" bash scripts/revision/02_warp_basis_mask.sh
echo "############ BT Stage 20 — LoRA 계열 6종 ############"
METHODS="lora asft lisa safelora salora wsr_lora" bash scripts/revision/20_lora_family.sh
echo "############ BT Stage 21 — SEAL ############"
METHODS="seal" bash scripts/revision/21_seal.sh
conda deactivate

# ── 3) 21개 셀 일괄 평가 ────────────────────────────────────────────────────
mapfile -t REPOS < <(find "$OUT_ROOT" -name .uploaded -exec cat {} \; | sort -u)
echo "############ 평가 대상 ${#REPOS[@]}개 ############"
printf '   - %s\n' "${REPOS[@]}"
if (( ${#REPOS[@]} == 0 )); then echo "[chain] 업로드된 셀이 없다 — 평가 생략"; exit 1; fi

# HF 캐시를 lustre 로: 21 × 13GB ≈ 270GB 이고 $HOME(overlay)은 191GB 뿐이다.
export HF_HOME="$B/.hf_cache" HF_HUB_CACHE="$B/.hf_cache/hub"
mkdir -p "$HF_HUB_CACHE"
cd "$B/HarmBench"
CONDA_SH="$M/etc/profile.d/conda.sh" ./run_all_eval.sh "${REPOS[@]}"
echo "############ 전체 종료 ($(date '+%F %T')) ############"
