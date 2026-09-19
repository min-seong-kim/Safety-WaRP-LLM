#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  batch3: 학습을 전부 끝낸 뒤 마지막에 한 번만 평가한다 (사용자 지시 2026-09-19).
#
#  (A) 타깃한정(q,k,v,up,down) AsFT·Lisa 8셀
#        qwen25_7b:gsm8k  gemma2_9b:gsm8k  llama2_7b:medqa  llama2_7b:arc
#  (B) 실험 2 — 원공간 동결 스윕 base 라인 (남은 9셀; p10 llama2_7b_base 는 완료)
#  (C) 전체 평가 — (A) 8개 + (B) 신규 + 기준행(원본 base / SSFT / ρ=0)
#
#  ⚠️ 학습과 평가를 겹치지 않는다. HarmBench 가 llama2_7b 에 gpu_memory_utilization 0.95
#     (=174GB) 를 쓰므로 학습 중 평가를 걸면 OOM 이다.
#  ⚠️ 평가 사이에 GPU 점유를 확인한다 — run_all_eval.sh 가 rc=0 으로 끝난 뒤에도 자식
#     프로세스가 GPU 를 붙들고 남아 다음 평가의 vLLM KV cache 를 굶긴 적이 있다.
# ════════════════════════════════════════════════════════════════════════════
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/batch3.pid
mkdir -p logs/batch3

wait_gpu() {
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-0}" -lt 2000 ] && return 0
    echo "[batch3] GPU ${used}MiB 점유 중 — 대기"; sleep 30
  done
}

echo "########## (A) 타깃한정 AsFT·Lisa 8셀 ##########  $(date -Is)"
COMBOS="qwen25_7b:gsm8k gemma2_9b:gsm8k llama2_7b:medqa llama2_7b:arc" \
TRAIN_ONLY_TARGETS=1 \
EXP1_OUT_ROOT="$PWD/outputs/asft_lisa_tgtonly" \
EXP1_LOG_DIR="$PWD/logs/asft_lisa_tgtonly" \
EXP1_REPO_SUFFIX="_tgtonly" \
  bash scripts/revision/30_asft_lisa_fullft.sh
echo "BATCH3_A_EXIT=$?"

echo "########## (B) 실험 2 — 원공간 동결 base 라인 ##########  $(date -Is)"
wait_gpu
bash scripts/revision/31_origspace_base.sh
echo "BATCH3_B_EXIT=$?"

echo "########## (C) 평가 ##########  $(date -Is)"
wait_gpu
echo "--- (C1) 타깃한정 셀 ---"
EVAL_ROOT="$PWD/outputs/asft_lisa_tgtonly" bash scripts/revision/32_eval_cells.sh
echo "BATCH3_C1_EXIT=$?"

wait_gpu
echo "--- (C2) 실험 2 셀 ---"
EVAL_ROOT="$PWD/outputs/origspace_base" bash scripts/revision/32_eval_cells.sh
echo "BATCH3_C2_EXIT=$?"

wait_gpu
echo "--- (C3) 실험 2 기준행 (원본 base / SSFT / ρ=0) ---"
REPOS_ONLY="meta-llama/Llama-2-7b-hf meta-llama/Llama-3.1-8B \
kmseong/llama2_7b-base-CB_SSFT-lr3e-5 kmseong/Llama-3.1-8B-base-SSFT_lr5e-5 \
kmseong/llama2_7b-base-gsm8k_ssft_lr3e-5 kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5" \
  bash scripts/revision/32_eval_cells.sh
echo "BATCH3_C3_EXIT=$?"
echo "BATCH3_ALL_DONE $(date -Is)"
