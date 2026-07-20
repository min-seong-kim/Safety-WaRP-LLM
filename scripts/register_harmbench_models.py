"""HarmBench models.yaml 에 6개 WSR-LoRA 비교 모델 엔트리를 추가(멱등)."""
import os

YAML = "/home/users/minseong/HarmBench/configs/model_configs/models.yaml"
NS = "kmseong"
MODELS = {
    "llama2_7b-gsm8k-lora-lr1e-4":     f"{NS}/llama2_7b-chat-gsm8k-lora-r16-lr1e-4",
    "llama2_7b-gsm8k-origproj-lr1e-4": f"{NS}/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr1e-4",
    "llama2_7b-gsm8k-wsr-lr1e-4":      f"{NS}/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr1e-4",
    "llama2_7b-gsm8k-lora-lr2e-4":     f"{NS}/llama2_7b-chat-gsm8k-lora-r16-lr2e-4",
    "llama2_7b-gsm8k-origproj-lr2e-4": f"{NS}/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr2e-4",
    "llama2_7b-gsm8k-wsr-lr2e-4":      f"{NS}/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr2e-4",
}

ENTRY = """
{name}:
  model:
    model_name_or_path: {hf}
    use_fast_tokenizer: True
    dtype: bfloat16
    max_model_len: 4096
    block_size: 32
    gpu_memory_utilization: 0.5
  num_gpus: 1
  model_type: open_source
"""

def main():
    existing = open(YAML).read()
    added = []
    with open(YAML, "a") as f:
        for name, hf in MODELS.items():
            if f"\n{name}:" in existing or existing.startswith(f"{name}:"):
                print(f"[skip] {name} already present")
                continue
            f.write(ENTRY.format(name=name, hf=hf))
            added.append(name)
    print(f"added {len(added)}: {added}")
    print("short_names:")
    for n in MODELS:
        print(" ", n)

if __name__ == "__main__":
    main()
