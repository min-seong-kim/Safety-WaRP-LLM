#!/usr/bin/env bash
# 7B AsFT 벌점 독립 검증. offload 를 쓰는 13B 셀이 **전부 끝난 뒤**에 시작한다.
# (offload 학습 중 CPU 작업을 붙이면 메모리 대역폭을 뺏어 학습이 4배 느려진다 — 실측)
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_fullft/penaltycheck.pid
# 13B 두 셀(asft/lisa)이 모두 .done 될 때까지 대기
while [ ! -f outputs/asft_lisa_fullft/llama2_13b/gsm8k/lisa/.done ]; do
    grep -q "ALL_CELLS_FINISHED" logs/asft_lisa_fullft/orchestrator.log 2>/dev/null && break
    sleep 120
done
echo "[deferred] 13B 셀 종료 확인 — 벌점 검증 시작 $(date -Is)"
CUDA_VISIBLE_DEVICES="" python scripts/revision/_check_asft_penalty.py \
  --base meta-llama/Llama-2-7b-chat-hf \
  --aligned kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
  --tuned outputs/asft_lisa_fullft/llama2_7b/gsm8k/asft/model
echo "[deferred] 완료"
