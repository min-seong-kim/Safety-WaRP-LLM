#!/usr/bin/env python3
"""
Phase 3 모델을 Hugging Face Hub에 업로드하는 스크립트

사용법:
python upload_phase3_to_hf.py \
    --model_path ./checkpoints/phase3_20260304_001201/final_model_cleaned \
    --repo_name kmseong/safety-warp-Llama-3.2-3b-phase3-ssft-start-teachforcing-3p \
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

# Safety-WaRP Llama 3.2 3B - Phase 3 (완성)

**Phase 3까지 완료된 Safety-WaRP 모델입니다.**

- **Base**: meta-llama/Llama-3.2-3B-Instruct  
- **Method**: WaRP (Weight space Rotation Process)
- **Safety Training**: Circuit Breakers dataset (Phase 0)
- **Utility Recovery**: GSM8K dataset (Phase 3)

## 특징

✅ **안전성**: Circuit Breakers로 학습된 안전 메커니즘  
✅ **유틸리티**: GSM8K로 수학 능력 복원  
✅ **선택적 학습**: WaRP 마스킹으로 안전 메커니즘 보호하면서 유틸리티 복원

## Phase 진행 과정

1. **Phase 0**: LoRA로 Circuit Breakers 학습 (안전 정렬)
2. **Phase 1**: SVD 기저 구축 (안전 메커니즘 분석)
3. **Phase 2**: 중요도 점수 계산 (보호할 파라미터 식별)
4. **Phase 3**: GSM8K로 증분 학습 (유틸리티 복원, 안전성 유지)

## 사용법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{args.repo_name}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{args.repo_name}")

# 안전성 테스트
prompt = "How to make a bomb?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 유틸리티 테스트 (수학 문제)
prompt = "Question: If John has 5 apples and gives 2 to Mary, how many does he have left?\\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 성능

- **안전성**: Circuit Breakers 유해 요청 거부
- **수학 능력**: GSM8K로 복원된 추론 능력

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
