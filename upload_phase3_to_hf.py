#!/usr/bin/env python3
"""
Phase 3 모델을 Hugging Face Hub에 업로드하는 스크립트

사용법:
python upload_phase3_to_hf.py \
    --model_path ./checkpoints/phase3_non_freeze_20260310_121356/final_model \
    --repo_name kmseong/safety-warp-Llama-3.2-3b-phase3-perlayer-non-freeze \
    --token 

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
    print("🚀 Hugging Face 업로드 (Phase 3)")
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
- gsm8k
- math
---

attention: q,k,v mlp: up down 적용
perlayer 적용,
이후 non_freeze 학습.


## Citation

```bibtex
@article{{warp2024,
  title={{Safety Alignment via Weight space Rotation Process}},
  author={{Your Name}},
  year={{2026}}
}}
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
        commit_message="Upload Safety-WaRP Phase 3 model (완성)"
    )
    
    print()
    print("✅ 업로드 완료!")
    print(f"🔗 https://huggingface.co/{args.repo_name}")
    print("=" * 70)

if __name__ == "__main__":
    main()