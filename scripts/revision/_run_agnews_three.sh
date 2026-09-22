#!/usr/bin/env bash
# agnews 로 학습한 SEAL / AsFT / Lisa 세 모델을 평가한다.
#   1) 안전성: HarmBench 4공격 · sys 모드 · refusal keyword(GRADING=hard)
#   2) downstream: agnews_eval/evaluate_agnews_sst2.py (1k seed42 테스트셋)
#
# 주의: evaluate_agnews_sst2.py 의 REPO_ROOT 는 parents[2] 라 저장소 밖을 가리키고
#       기본 데이터 경로도 dataset/classification/ 이다 → --agnews-data 를 명시한다.
#       run_agnews_sst2_eval.sh 는 다른 사용자의 conda python 이 하드코딩돼 있어
#       PYTHON_BIN 을 넘기거나 평가기를 직접 호출해야 한다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
# 허브(huggingface.co)가 끊기면 vLLM 이 조합마다 5회 재시도로 수 분씩 멈춘다.
# 세 모델은 캐시에 완전히 들어 있으므로(가중치 3샤드 + chat_template) 오프라인으로 돌린다.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REPOS="kmseong/llama2_7b-chat-CB_SSFT-seal_agnews_topp0.8_lr5e-5
kmseong/llama2_7b-chat-CB_SSFT-asft_agnews_lambda1.0_lr7e-5
kmseong/llama2_7b-chat-CB_SSFT-lisa_agnews_rho1.0_lr7e-5"

echo "════════ 1) HarmBench 안전성 (refusal keyword) ════════ $(date -Iseconds)"
# VALIDATE_REPOS=0: check_repos_exist.py 가 model_info 에서 무한 대기하는 일이 있다
#   (3개 확인에 4분 넘게 멈춤). 세 리포는 list_repo_files 로 이미 직접 확인했다.
HB_ONLY=1 RESUME=false GRADING_OVERRIDE=hard VALIDATE_REPOS=0 \
  REPOS_ONLY="$(echo $REPOS)" bash scripts/revision/32_eval_cells.sh
echo "SAFETY_DONE rc=$?"

echo "════════ 2) AGNews 정확도 ════════ $(date -Iseconds)"
source scripts/env.sh hb
export HF_HOME=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF="expandable_segments:True"
for r in $REPOS; do
  echo "---- $r"
  python -u agnews_eval/evaluate_agnews_sst2.py "$r" \
    --task agnews \
    --agnews-data "$PWD/data/agnews_test_1k_seed42.json" \
    --output-root "$PWD/evaluation_results/agnews_sst2" \
    --batch-size 64 --max-length 1024 --max-new-tokens 32
  echo "   rc=$?"
done
echo "AGNEWS_THREE_DONE $(date -Iseconds)"
