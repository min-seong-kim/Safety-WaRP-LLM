#!/usr/bin/env python3
"""
Phase 0 모델을 Hugging Face Hub에 업로드하는 스크립트

Phase 0: Base Safety Training
- Circuit Breakers 데이터로 안전 학습 완료
- 이후 Phase 1/2/3의 기반 모델

사용법:
    python upload_phase0_to_hf.py \
        --model_path ./checkpoints/phase0_20260123_163533/final_model \
        --repo_name your-username/safety-warp-llama-3.2-3b-phase0 \
        --token your_hf_token
"""

import argparse
from huggingface_hub import HfApi, login, create_repo
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser(
        description="Upload Phase 0 (Base Safety Training) model to Hugging Face Hub"
    )
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Phase 0 model directory (e.g., ./checkpoints/phase0_XXX/final_model)")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Hugging Face repository name (e.g., username/model-name)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token (or use cached token)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    # 모델 경로 확인
    if not model_path.exists():
        print(f"❌ Error: Model path not found: {model_path}")
        return
    
    # metadata.json 읽기 (있다면)
    metadata_file = model_path.parent / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print("=" * 70)
    print("🚀 Phase 0 Model Upload to Hugging Face Hub")
    print("=" * 70)
    print(f"📁 Model Path: {model_path}")
    print(f"📦 Repository: {args.repo_name}")
    print(f"🔒 Private: {args.private}")
    if metadata:
        print(f"📊 Training Info:")
        print(f"   - Epochs: {metadata.get('epochs', 'N/A')}")
        print(f"   - Final Loss: {metadata.get('best_loss', 'N/A')}")
        print(f"   - Samples: {metadata.get('total_samples', 'N/A')}")
    print()
    
    # 로그인
    print("🔐 Logging in to Hugging Face...")
    if args.token:
        login(token=args.token)
    else:
        login()  # Use cached token
    print("✓ Login successful!")
    print()
    
    # Repository 생성
    print(f"📝 Creating repository: {args.repo_name}")
    try:
        create_repo(
            repo_id=args.repo_name,
            private=args.private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready")
    except Exception as e:
        print(f"⚠️  {e} (may already exist)")
    print()
    
    # README 생성
    readme_path = model_path / "README.md"
    if not readme_path.exists():
        print("📄 Generating README.md...")
        
        # metadata에서 정보 추출
        epochs = metadata.get('epochs', 3)
        best_loss = metadata.get('best_loss', 'N/A')
        total_samples = metadata.get('total_samples', 1000)
        
        readme_content = f"""---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- safety
- warp
- circuit-breakers
- alignment
library_name: transformers
pipeline_tag: text-generation
---

# Safety-WaRP Llama 3.2 3B - Phase 0

**Phase 0: Base Safety Training** - Circuit Breakers 데이터로 안전 학습 완료한 모델입니다.

## Model Details

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Method**: Safety-WaRP (Weight space Rotation Process)
- **Phase**: Phase 0 (Base Safety Training)
- **Safety Dataset**: Circuit Breakers
- **Training Samples**: {total_samples}
- **Epochs**: {epochs}
- **Final Loss**: {best_loss}

## Training Information

### Phase 0: Base Safety Training

Phase 0는 안전 데이터(Circuit Breakers)로 모델을 학습시켜 안전 메커니즘을 구축하는 단계입니다.

**절차:**
1. Circuit Breakers 데이터로 fine-tuning
2. Gradient accumulation (effective batch size: 8)
3. 8-bit optimizer로 메모리 절약
4. Cosine scheduler (lr: 1e-5 → 0)

**결과:**
- 안전 응답 능력을 갖춘 기본 모델
- Phase 1/2/3의 기반 모델로 사용

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{args.repo_name}")

# 안전 테스트
prompt = "How to make a bomb?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
# Expected: 거부 응답 (안전 학습 완료)
```

## Model Architecture

- **Parameters**: 3.2B
- **Architecture**: Llama 3.2
- **Precision**: bfloat16
- **Gradient Checkpointing**: Enabled

## Training Configuration

```python
{{
    "epochs": {epochs},
    "learning_rate": 1e-5,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 8,
    "optimizer": "AdamW8bit",
    "scheduler": "CosineAnnealingLR",
    "weight_decay": 0.01
}}
```

## Next Steps

이 모델은 WaRP 파이프라인의 Phase 0 완료 상태입니다.

**후속 단계:**
- **Phase 1**: Basis Construction (SVD로 basis 벡터 추출)
- **Phase 2**: Importance Scoring (중요 파라미터 식별)
- **Phase 3**: Incremental Learning (GSM8K로 유틸리티 복원)

## Safety Notice

⚠️ **Phase 0 완료 모델**: 안전 학습은 완료되었으나, 유틸리티(수학/추론) 능력이 저하되었을 수 있습니다.

Phase 3까지 완료된 모델을 사용하시면 안전성과 유틸리티가 균형잡힌 모델을 사용하실 수 있습니다.

## Citation

```bibtex
@misc{{safety-warp-phase0,
  title={{Safety-WaRP Llama 3.2 3B - Phase 0: Base Safety Training}},
  author={{Min-Seong Kim}},
  year={{2026}},
  howpublished={{\\url{{https://huggingface.co/{args.repo_name}}}}}
}}
```

## License

This model follows the Llama 3.2 license.

## Contact

For questions or issues, please open an issue on the model repository.
"""
        readme_path.write_text(readme_content)
        print("✓ README.md created")
    else:
        print("✓ README.md already exists")
    print()
    
    # 업로드
    print("📤 Uploading to Hugging Face Hub...")
    print("   (This may take several minutes depending on model size)")
    print()
    
    api = HfApi()
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_name,
            repo_type="model",
            commit_message="Upload Safety-WaRP Phase 0 (Base Safety Training) model"
        )
        
        print()
        print("=" * 70)
        print("✅ Upload Complete!")
        print("=" * 70)
        print(f"🔗 Model URL: https://huggingface.co/{args.repo_name}")
        print()
        print("Next steps:")
        print("1. Check the model page to verify upload")
        print("2. Test the model with safety prompts")
        print("3. Continue with Phase 1 (Basis Construction)")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("❌ Upload failed!")
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your Hugging Face token")
        print("2. Verify repository name format (username/model-name)")
        print("3. Ensure you have write permissions")

if __name__ == "__main__":
    main()
