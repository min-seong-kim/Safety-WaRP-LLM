"""저장된 SafeLoRA merged 모델(/scratch2)을 HF Hub 에 (재)업로드.

학습 중 push 가 실패(토큰 무효/네트워크)했을 때, 재학습 없이 저장본에서 바로 올린다.
유효한 write 토큰이 필요: `hf auth login` 또는 `export HF_TOKEN=...` 후 실행.

사용:
    python scripts/push_safelora_from_scratch.py
    python scripts/push_safelora_from_scratch.py --lr_list 1e-4 2e-4 --threshold 0.35 --r 16
"""
import argparse
import os

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/scratch2/gokms0509/lora_comparison")
    ap.add_argument("--hf_ns", default="kmseong")
    ap.add_argument("--lr_list", nargs="+", default=["1e-4", "2e-4"])
    ap.add_argument("--threshold", default="0.35")
    ap.add_argument("--r", default="16")
    ap.add_argument("--use_upload_folder", action="store_true",
                    help="from_pretrained 없이 폴더를 그대로 업로드(더 가벼움)")
    args = ap.parse_args()

    api = HfApi()
    who = api.whoami()  # 토큰 무효면 여기서 즉시 에러
    print(f"[push] authenticated as: {who.get('name')}")

    for lr in args.lr_list:
        merged_dir = os.path.join(args.out_root, "safe_lora", f"lr_{lr}", "merged_model")
        repo = f"{args.hf_ns}/llama2_7b-chat-gsm8k-safelora-thr{args.threshold}-r{args.r}-lr{lr}"
        if not os.path.isdir(merged_dir):
            print(f"[push] SKIP (없음): {merged_dir}")
            continue
        print(f"[push] {merged_dir} → {repo}")
        api.create_repo(repo, exist_ok=True)
        if args.use_upload_folder:
            api.upload_folder(folder_path=merged_dir, repo_id=repo, repo_type="model")
        else:
            model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.bfloat16)
            tok = AutoTokenizer.from_pretrained(merged_dir)
            model.push_to_hub(repo)
            tok.push_to_hub(repo)
            del model
        print(f"[push] ✓ https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
