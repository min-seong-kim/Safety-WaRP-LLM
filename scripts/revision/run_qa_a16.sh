#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  MedQA / ARC-C × {lora asft lisa safelora salora wsr_lora seal} — llama2_7b, CB 축
#
#  사용자 지정 (2026-09-07):
#    · LoRA scaling = 1.0  → LORA_ALPHA=16, WSR_LORA_ALPHA=16 (r=16 고정)
#      ⚠️ 기존 revision 셀은 alpha=32(scaling 2.0) 이므로 직접 비교 불가.
#    · LISA rho=1.0 · SafeLoRA thr=0.3 · WSR-LoRA rho=0.3
#    · SEAL 은 full-param 이므로 lr=FULL_LR(5e-5), epochs=3 (21_seal.sh 기본과 동일)
#
#  OUT_ROOT 를 따로 쓰는 이유: out_dir 은 하이퍼파라미터를 인코딩하지 않아서
#  (CLAUDE.md) 같은 셀을 다른 alpha/rho 로 돌리면 "이미 완료"로 건너뛴다.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
M=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3
R=/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source "$M/etc/profile.d/conda.sh"; conda activate hb
export CUDA_VISIBLE_DEVICES=0
cd "$R"

export SAFETY_SETS=cb MODELS=llama2_7b TASKS="medqa arc"
export LORA_ALPHA=16 WSR_LORA_ALPHA=16
export LISA_RHO=1.0 SAFELORA_THRESHOLD=0.3 KEEP_RATIO=0.3
export EPOCHS=3
export PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 HF_NAMESPACE=kmseong
export OUT_ROOT="$R/outputs/revision_qa_a16"
mkdir -p "$OUT_ROOT"

echo "############ Stage 20 — LoRA 계열 6종 ############"
METHODS="lora asft lisa safelora salora wsr_lora" bash scripts/revision/20_lora_family.sh
rc20=$?
echo "############ Stage 21 — SEAL ############"
METHODS="seal" bash scripts/revision/21_seal.sh
rc21=$?
echo "############ 종료: stage20 rc=$rc20 / stage21 rc=$rc21 ############"
