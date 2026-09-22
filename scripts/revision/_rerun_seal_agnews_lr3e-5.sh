#!/usr/bin/env bash
# SEAL / agnews / llama2-7b-chat 를 **lr 3e-5** 로 다시 학습한다.
# S2(full-param SFT)의 나머지 설정은 Full FT arm 과 동일하다
#   (wd 0.01 · warmup 0.1 · cosine · eff.batch 16 · 3 epoch · seed 42).
#
# ⚠️ out_dir 은 하이퍼파라미터를 인코딩하지 않는다 → 같은 셀을 다른 lr 로 돌리면
#    ".done 이 있다" 며 건너뛴다. 반드시 OUT_ROOT 로 격리한다(.done 을 지우지 말 것).
# ⚠️ 허브가 끊겨 있어 업로드는 끈다(PUSH_TO_HUB=0). 출발 모델은 캐시에 있다.
# ⚠️ agnews selector(S1) 산출물이 없으므로 S1 부터 돈다(medqa/arc 만 존재).
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache
# common.sh 는 PY=python 을 그대로 쓴다 → conda 활성화가 없으면
# ModuleNotFoundError: No module named 'transformers' 로 S1 이 즉사한다.
source scripts/env.sh hb

FULL_LR=3e-5 \
OUT_ROOT="$PWD/outputs/revision_seal_lr3e-5" \
SAFETY_SETS=cb MODELS=llama2_7b TASKS=agnews METHODS=seal \
PUSH_TO_HUB=0 PRUNE_AFTER_UPLOAD=0 \
  bash scripts/revision/21_seal.sh
echo "SEAL_LR3E5_DONE rc=$? $(date -Iseconds)"
