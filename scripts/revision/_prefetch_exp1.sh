#!/usr/bin/env bash
# 실험 1 에 필요한 출발/base 모델을 미리 받아둔다(학습과 병렬).
# 셀마다 다운로드를 기다리면 GPU 가 논다. 순차로 받아 네트워크를 독점하지 않는다.
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh hb
echo $$ > logs/asft_lisa_fullft/prefetch.pid
python - <<'PY'
from huggingface_hub import snapshot_download
import time, traceback
REPOS = [
    "kmseong/llama3_2_3b-instruct-SSFT-lr5e-5", "meta-llama/Llama-3.2-3B-Instruct",
    "kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5", "meta-llama/Llama-3.1-8B-Instruct",
    "wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5", "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5", "meta-llama/Llama-2-13b-chat-hf",
]
IGNORE = ["*.pth", "*.bin", "*.msgpack", "*.h5", "original/*"]
for r in REPOS:
    for attempt in (1, 2, 3):
        t0 = time.time()
        try:
            snapshot_download(r, ignore_patterns=IGNORE, max_workers=4)
            print(f"[prefetch] OK  {r}  ({time.time()-t0:.0f}s)", flush=True)
            break
        except Exception as e:
            print(f"[prefetch] FAIL({attempt}) {r}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(20)
    else:
        print(f"[prefetch] GAVE UP {r}", flush=True)
print("PREFETCH_DONE", flush=True)
PY
