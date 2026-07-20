import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

ids = [
    'kmseong/llama2_7b-chat-gsm8k-lora-r16-lr1e-4',
    'kmseong/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr1e-4',
    'kmseong/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr1e-4',
    'kmseong/llama2_7b-chat-gsm8k-lora-r16-lr2e-4',
    'kmseong/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr2e-4',
    'kmseong/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr2e-4',
]
for i in ids:
    print(f"prefetch {i} ...", flush=True)
    p = snapshot_download(i, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"])
    print(f"  done -> {p}", flush=True)
print("ALL PREFETCHED")
