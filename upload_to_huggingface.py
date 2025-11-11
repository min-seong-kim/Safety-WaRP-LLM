#!/usr/bin/env python3
"""
Upload Fine-tuned Model to HuggingFace Hub

Phase 3 완료 후 만들어진 모델을 HuggingFace Hub에 업로드하는 스크립트

Usage:
    python upload_to_huggingface.py \
        --model_path ./checkpoints/phase3_20251109_190831/checkpoints/checkpoints/phase3_best.pt \
        --hf_model_id kmseong/WaRP-Safety-Llama3_8B_Instruct \
        --hf_token your_huggingface_token
"""

import argparse
import os
import logging
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [HF-Upload] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def upload_model_to_huggingface(
    model_path: str,
    hf_model_id: str,
    hf_token: str = None,
    private: bool = False,
    commit_message: str = None
):
    """
    미세조정된 모델을 HuggingFace Hub에 업로드
    
    Args:
        model_path: 저장된 모델 경로 (.pt 파일)
        hf_model_id: HuggingFace 모델 ID (e.g., "kmseong/WaRP-Safety-Llama3_8B_Instruct")
        hf_token: HuggingFace API 토큰 (None이면 환경변수에서 읽음)
        private: 비공개 모델 여부
        commit_message: 커밋 메시지
    """
    
    logger.info("="*60)
    logger.info("UPLOADING MODEL TO HUGGINGFACE HUB")
    logger.info("="*60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model ID: {hf_model_id}")
    logger.info(f"Private: {private}")
    
    # Step 1: 모델 파일 확인
    logger.info("\n[Step 1] Checking model file...")
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    model_size_gb = model_path.stat().st_size / (1024**3)
    logger.info(f"✓ Model file found: {model_path.name} ({model_size_gb:.2f}GB)")
    
    # Step 2: HuggingFace 토큰 확인
    logger.info("\n[Step 2] Checking HuggingFace token...")
    
    if hf_token is None:
        # 환경변수에서 읽기
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        
        if hf_token is None:
            logger.error("HuggingFace token not found!")
            logger.error("Set it via:")
            logger.error("  1. --hf_token argument")
            logger.error("  2. HF_TOKEN environment variable")
            logger.error("  3. HUGGINGFACE_TOKEN environment variable")
            return False
    
    logger.info("✓ HuggingFace token configured")
    
    # Step 3: HuggingFace 라이브러리 설정
    logger.info("\n[Step 3] Configuring HuggingFace Hub...")
    
    try:
        from huggingface_hub import login, HfApi
        
        # 로그인
        login(token=hf_token, add_to_git_credential=True)
        logger.info("✓ Logged in to HuggingFace Hub")
        
        # API 클라이언트
        api = HfApi()
        
    except ImportError:
        logger.error("huggingface_hub is not installed!")
        logger.error("Install it with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to login to HuggingFace: {e}")
        return False
    
    # Step 4: 로컬 저장소 준비
    logger.info("\n[Step 4] Preparing local repository...")
    
    # 임시 디렉토리에 모든 파일 복사
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 모델 파일 처리 (state_dict만 추출해서 저장)
        logger.info(f"Processing model file (extracting state_dict)...")
        
        # Phase 3 checkpoint 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # state_dict 추출
        # Phase 3 checkpoint 형식: {'model_state_dict': {...}, 'epoch': int, 'config': {...}}
        if isinstance(checkpoint, dict):
            # 우선 'model_state_dict' 확인
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("✓ Extracted state_dict from 'model_state_dict' key")
            # 그 다음 'params' 확인
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
                logger.info("✓ Extracted state_dict from 'params' key")
            # 그 외 경우: 전체 checkpoint가 state_dict
            else:
                state_dict = checkpoint
                logger.info("✓ Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            logger.info("✓ Using checkpoint directly as state_dict")
        
        # HuggingFace 호환 형식으로 저장 (state_dict만)
        torch.save(state_dict, temp_path / "pytorch_model.bin")
        logger.info(f"✓ State dict saved to pytorch_model.bin")
        
        # 기본 LLaMA 3 모델에서 config, tokenizer 등 복사
        logger.info(f"\nLoading base model config and tokenizer...")
        
        try:
            # 기본 모델에서 필요한 파일들 다운로드
            base_model = "meta-llama/Llama-3.1-8B-Instruct"
            
            config = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True
            ).config
            
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True
            )
            
            # config 저장
            config.save_pretrained(temp_path)
            logger.info("✓ Config saved")
            
            # tokenizer 저장
            tokenizer.save_pretrained(temp_path)
            logger.info("✓ Tokenizer saved")
            
        except Exception as e:
            logger.warning(f"Failed to load base model: {e}")
            logger.warning("Continuing without config and tokenizer files...")
        
        # Step 5: README 생성
        logger.info(f"\nGenerating model README...")
        
        readme_content = f"""---
license: llama3.1
language:
- en
library_name: transformers
tags:
- llama
- safety
- alignment
- warp
---

# WaRP-Safety-Llama3_8B_Instruct

Fine-tuned Llama 3.1 8B Instruct model for safety alignment using Weight space Rotation Process (WaRP).

## Model Details

- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Training Method**: Safety-First WaRP (3-Phase pipeline)
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training Procedure

### Phase 1: Basis Construction
- Collected activations from FFN layers using safety data
- Computed SVD to obtain orthonormal basis vectors
- Identified 419 important neurons in layer 31

### Phase 2: Importance Scoring
- Calculated importance scores using gradient-based methods
- Generated masks for important directions
- Used teacher forcing on safety responses

### Phase 3: Incremental Learning
- Fine-tuned on utility task (GSM8K) with gradient masking
- Protected important directions to maintain safety
- Improved utility while preserving safety mechanisms

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "kmseong/WaRP-Safety-Llama3_8B_Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# Generate text
inputs = tokenizer("What is machine learning?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Safety Features

- ✅ Protected safety mechanisms through gradient masking
- ✅ Maintained refusal capability for harmful requests
- ✅ Improved utility on reasoning tasks
- ✅ Balanced safety-utility tradeoff

## Datasets

- **Safety Data**: LibrAI/do-not-answer
- **Utility Data**: openai/gsm8k

## Citation

```
@article{{warp-safety,
  title={{Safety-First WaRP: Weight space Rotation Process for LLM Safety Alignment}},
  author={{Min-Seong Kim}},
  year={{{datetime.now().year}}}
}}
```

## License

This model is built on Llama 3.1 8B Instruct and follows the same license.

## Disclaimer

This model is fine-tuned for improved safety. Users should evaluate model outputs for their specific use cases and apply additional safety measures as needed.
"""
        
        readme_path = temp_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info("✓ README.md generated")
        
        # 메타데이터 파일 생성
        metadata = {
            "model_id": hf_model_id,
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "training_method": "Safety-First WaRP",
            "upload_date": datetime.now().isoformat(),
            "model_size_gb": model_size_gb,
            "important_neurons_layer31": 419,
            "keep_ratio": 0.1,
            "framework": "pytorch",
            "task": "text-generation"
        }
        
        metadata_path = temp_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("✓ Metadata file generated")
        
        # Step 5.5: Repository 생성 (아직 없으면)
        logger.info(f"\n[Step 5.5] Creating repository on HuggingFace Hub...")
        
        try:
            from huggingface_hub import HfApi, repo_exists
            
            api = HfApi(token=hf_token)
            
            # repo가 이미 존재하는지 확인
            repo_exists_bool = repo_exists(
                repo_id=hf_model_id,
                repo_type="model",
                token=hf_token
            )
            
            if not repo_exists_bool:
                logger.info(f"Creating new repository: {hf_model_id}")
                repo_url = api.create_repo(
                    repo_id=hf_model_id,
                    repo_type="model",
                    private=private,
                    exist_ok=True,  # 이미 존재해도 오류 안 남
                )
                logger.info(f"✓ Repository created: {repo_url}")
            else:
                logger.info(f"✓ Repository already exists: {hf_model_id}")
        
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False
        
        # Step 6: HuggingFace에 업로드
        logger.info(f"\n[Step 6] Uploading to HuggingFace Hub...")
        logger.info(f"Repository: {hf_model_id}")
        
        try:
            from huggingface_hub import upload_folder, HfApi
            
            # upload_folder() 호출 (private 파라미터 제거)
            repo_url = upload_folder(
                folder_path=str(temp_path),
                repo_id=hf_model_id,
                repo_type="model",
                commit_message=commit_message or f"Upload WaRP fine-tuned model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                token=hf_token,
            )
            
            logger.info(f"✓ Model uploaded successfully!")
            logger.info(f"✓ Repository URL: {repo_url}")
            
            # 프라이버시 설정 (private=True일 경우)
            if private:
                try:
                    api = HfApi(token=hf_token)
                    api.update_repo_visibility(repo_id=hf_model_id, private=True, repo_type="model")
                    logger.info(f"✓ Repository set to private")
                except Exception as e:
                    logger.warning(f"Failed to set repository to private: {e}")
                    logger.info("You can manually set it in HuggingFace settings")
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False
    
    # Step 7: 완료
    logger.info("\n" + "="*60)
    logger.info("UPLOAD COMPLETED")
    logger.info("="*60)
    logger.info(f"Model ID: {hf_model_id}")
    logger.info(f"Access URL: https://huggingface.co/{hf_model_id}")
    logger.info("="*60 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Upload fine-tuned model to HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with token
  python upload_to_huggingface.py \\
    --model_path ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \\
    --hf_model_id kmseong/WaRP-Safety-Llama3_8B_Instruct \\
    --hf_token your_token_here

  # Using environment variable for token
  export HF_TOKEN=your_token_here
  python upload_to_huggingface.py \\
    --model_path ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \\
    --hf_model_id kmseong/WaRP-Safety-Llama3_8B_Instruct

  # Make model private
  python upload_to_huggingface.py \\
    --model_path ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \\
    --hf_model_id kmseong/WaRP-Safety-Llama3_8B_Instruct \\
    --private
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the fine-tuned model (.pt file)'
    )
    parser.add_argument(
        '--hf_model_id',
        type=str,
        required=True,
        help='HuggingFace model ID (e.g., kmseong/WaRP-Safety-Llama3_8B_Instruct). Timestamp will be appended automatically.'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the model private on HuggingFace Hub'
    )
    parser.add_argument(
        '--commit_message',
        type=str,
        default=None,
        help='Custom commit message'
    )
    parser.add_argument(
        '--no_timestamp',
        action='store_true',
        help='Do not append timestamp to model ID'
    )
    
    args = parser.parse_args()
    
    # 모델 ID에 타임스탐프 추가 (--no_timestamp 옵션으로 비활성화 가능)
    hf_model_id = args.hf_model_id
    if not args.no_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hf_model_id = f"{args.hf_model_id}-{timestamp}"
        logger.info(f"Appending timestamp to model ID: {hf_model_id}")
    
    # 업로드 실행
    success = upload_model_to_huggingface(
        model_path=args.model_path,
        hf_model_id=hf_model_id,
        hf_token=args.hf_token,
        private=args.private,
        commit_message=args.commit_message
    )
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
