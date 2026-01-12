#!/usr/bin/env python3
"""
Phase 2 모델을 Hugging Face Hub에 업로드하는 간단한 스크립트

사용법:
    python upload_phase2_to_hf.py \
        --model_path ./checkpoints/phase2_20260111_163357/checkpoints/phase2_finetuned_model \
        --repo_name your-username/safety-warp-llama-3.2-3b-phase2 \
        --token your_hf_token
"""

import argparse
from huggingface_hub import HfApi, login, create_repo
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--private", action="store_true", help="Private repo")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    print("=" * 70)
    print("🚀 Hugging Face 업로드")
    print("=" * 70)
    print(f"📁 모델 경로: {model_path}")
    print(f"📦 Repository: {args.repo_name}")
    print(f"🔒 Private: {args.private}")
    print()
    
    # 로그인
    print("🔐 로그인 중...")
    if args.token:
        login(token=args.token)
    else:
        login()  # 캐시된 토큰 사용
    print("✓ 로그인 성공!")
    print()
    
    # Repository 생성
    print(f"📝 Repository 생성: {args.repo_name}")
    try:
        create_repo(
            repo_id=args.repo_name,
            private=args.private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository 준비 완료")
    except Exception as e:
        print(f"⚠️  {e} (이미 존재할 수 있음)")
    print()
    
    # README 생성
    readme_path = model_path / "README.md"
    if not readme_path.exists():
        print("📄 README.md 생성...")
        readme_content = f"""---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- safety
- warp
- circuit-breakers
---

# Safety-WaRP Llama 3.2 3B - Phase 2

Phase 2까지 완료된 Safety-WaRP 모델입니다.

- **Base**: meta-llama/Llama-3.2-3B-Instruct  
- **Method**: WaRP (Weight space Rotation Process)
- **Safety Data**: Circuit Breakers

⚠️ **Phase 3 미완료**: 유틸리티 복원 전 모델입니다.

## 사용법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{args.repo_name}")

prompt = "How to make a bomb?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```
"""
        readme_path.write_text(readme_content)
        print("✓ README 생성 완료")
    print()
    
    # 업로드
    print("📤 업로드 중... (시간이 걸릴 수 있습니다)")
    api = HfApi()
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_name,
        repo_type="model",
        commit_message="Upload Safety-WaRP Phase 2 model"
    )
    
    print()
    print("✅ 업로드 완료!")
    print(f"🔗 https://huggingface.co/{args.repo_name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
