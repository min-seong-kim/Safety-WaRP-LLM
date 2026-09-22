#!/usr/bin/env bash
# 13B 검증에 필요한 모델을 미리 내려받는다. 네트워크/디스크만 쓰므로 GPU 작업과 병렬 가능.
# 이 박스의 허브 다운로드는 느리고 IncompleteRead 로 끊기므로 모델당 재시도한다.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
export HF_HOME=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.hf_cache
REPOS="wvnvwn/llama-2-13b-chat-hf-lr5e-5-gsm8k-lr5e-5
wvnvwn/llama-2-13b-chat-hf-lr5e-5-safeinstr-0.1
wvnvwn/llama-2-13b-chat-hf-lr5e-5-resta-0.1
wvnvwn/llama-2-13b-chat-hf-lr5e-5-safedelta-scale0.1
wvnvwn/llama-2-13b-chat-hf-gsm8k-sn-tuned-lr5e-5
wvnvwn/llama-2-13b-chat-hf-gsm8k-rsn-tuned-lr5e-5
wvnvwn/llama-2-13b-chat-hf-WaRP-lr5e-5
kmseong/llama2_13b-chat-CB_SSFT-seal_gsm8k_topp0.8_lr5e-5
kmseong/llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4
kmseong/llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4"
for r in $REPOS; do
  for try in 1 2 3 4 5; do
    echo "[prefetch] $r (시도 $try) $(date -Iseconds)"
    if python - "$r" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","*.model","*.jinja","*.txt"])
print("[ok]", sys.argv[1], p)
PY
    then echo "[prefetch] DONE $r"; break; fi
    echo "[prefetch] RETRY $r"; sleep 20
  done
done
echo "PREFETCH_13B_DONE $(date -Iseconds)"
