#!/usr/bin/env python3
"""
Upload Fine-tuned Model to HuggingFace Hub

Phase 3 완료 후 만들어진 모델을 HuggingFace Hub에 업로드하는 스크립트

Usage:
export HF_TOKEN=hf_xxx

python upload_to_huggingface.py \
    --model_path /home/yonsei_jong/Safety-WaRP-LLM/medqa_eval/llama2_7b_chat_SSFT_medqa_FT_lr3e-5 \
    --hf_model_id kmseong/llama2_7b_chat-SSFT-MEDQA-FT-lr3e-5 \
    --no_timestamp

python upload_to_huggingface.py \
    --model_path ./checkpoints/warp_safelora_20260427_000614/merged_model \
    --hf_model_id kmseong/llama2_7b_base-WaRP-safelora-freeze_lr2e-4
    --no_timestamp

python upload_to_huggingface.py \
    --model_path ./checkpoints/phase3_non_freeze_20260501_150227/final_model \
    --hf_model_id kmseong/llama2_7b-SSFT-WaRP_medqa_FT_lr1e-5_fix \
    --no_timestamp


"""

import argparse
import os
import logging
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoConfig, AutoTokenizer

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
    commit_message: str = None,
    base_model: str = None,
):
    """
    미세조정된 모델을 HuggingFace Hub에 업로드
    
    Args:
        model_path: 저장된 모델 경로 (.pt 파일)
        hf_model_id: HuggingFace 모델 ID (e.g., "kmseong/WaRP-Safety-Llama3_3B_Instruct")
        hf_token: HuggingFace API 토큰 (None이면 환경변수에서 읽음)
        private: 비공개 모델 여부
        commit_message: 커밋 메시지
        base_model: tokenizer/config가 없을 때 사용할 베이스 모델 ID
    """
    
    logger.info("="*60)
    logger.info("UPLOADING MODEL TO HUGGINGFACE HUB")
    logger.info("="*60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model ID: {hf_model_id}")
    logger.info(f"Private: {private}")

    def _get_path_size_gb(path: Path) -> float:
        if path.is_file():
            return path.stat().st_size / (1024 ** 3)
        total_bytes = 0
        for p in path.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size
        return total_bytes / (1024 ** 3)

    def _has_config_and_tokenizer(path: Path) -> bool:
        has_config = (path / "config.json").exists()
        has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()
        return has_config and has_tokenizer

    def _find_weight_shape(path: Path, key_candidates: list[str]):
        try:
            from safetensors import safe_open
        except Exception:
            return None

        index_path = path / "model.safetensors.index.json"
        single_path = path / "model.safetensors"

        if index_path.exists():
            idx = json.loads(index_path.read_text())
            weight_map = idx.get("weight_map", {})
            for key in key_candidates:
                shard_name = weight_map.get(key)
                if not shard_name:
                    continue
                shard_path = path / shard_name
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    if key in f.keys():
                        return tuple(f.get_slice(key).get_shape())
            return None

        if single_path.exists():
            for key in key_candidates:
                with safe_open(str(single_path), framework="pt", device="cpu") as f:
                    if key in f.keys():
                        return tuple(f.get_slice(key).get_shape())
            return None

        return None

    def _validate_vocab_alignment(path: Path) -> bool:
        logger.info("\n[Step 4.5] Validating tokenizer/config/weight vocab alignment...")
        try:
            tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
            cfg = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer/config from prepared upload folder: {e}")
            return False

        embed_shape = _find_weight_shape(path, [
            "model.embed_tokens.weight",
            "model.model.embed_tokens.weight",
            "tok_embeddings.weight",
            "transformer.wte.weight",
        ])
        lm_head_shape = _find_weight_shape(path, [
            "lm_head.weight",
            "model.lm_head.weight",
            "output.weight",
        ])

        tok_len = len(tok)
        cfg_vocab = int(getattr(cfg, "vocab_size", -1))

        logger.info(f"tokenizer_len={tok_len}, config_vocab={cfg_vocab}, embed={embed_shape}, lm_head={lm_head_shape}")

        if embed_shape is not None and embed_shape[0] not in (tok_len, cfg_vocab):
            logger.error("Embedding vocab dimension mismatches tokenizer/config vocab size")
            return False
        if lm_head_shape is not None and lm_head_shape[0] not in (tok_len, cfg_vocab):
            logger.error("LM head vocab dimension mismatches tokenizer/config vocab size")
            return False

        logger.info("✓ Vocab alignment check passed")
        return True
    
    # Step 1: 모델 파일 확인
    logger.info("\n[Step 1] Checking model file...")
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    model_size_gb = _get_path_size_gb(model_path)
    if model_path.is_dir():
        logger.info(f"✓ Model directory found: {model_path} ({model_size_gb:.2f}GB)")
    else:
        logger.info(f"✓ Model file found: {model_path.name} ({model_size_gb:.2f}GB)")
    
    # Step 2: HuggingFace 토큰 확인
    logger.info("\n[Step 2] Checking HuggingFace token...")

    # HF_TOKEN/HUGGINGFACE_TOKEN 우선 사용
    env_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    hf_token = env_token or hf_token

    if hf_token is None:
        logger.error("HuggingFace token not found!")
        logger.error("Set it via:")
        logger.error("  1. export HF_TOKEN=your_token")
        logger.error("  2. export HUGGINGFACE_TOKEN=your_token")
        return False
    
    logger.info("✓ HuggingFace token configured")
    
    # Step 3: HuggingFace 라이브러리 설정
    logger.info("\n[Step 3] Configuring HuggingFace Hub...")
    
    try:
        from huggingface_hub import HfApi

        # 토큰을 API 호출 시 직접 사용 (interactive login 방지)
        api = HfApi(token=hf_token)
        logger.info("✓ HuggingFace Hub client initialized with token")
        
    except ImportError:
        logger.error("huggingface_hub is not installed!")
        logger.error("Install it with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace client: {e}")
        return False
    
    # Step 4: 로컬 저장소 준비
    logger.info("\n[Step 4] Preparing local repository...")
    
    # 임시 디렉토리에 모든 파일 복사
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 모델 경로 처리
        if model_path.is_dir():
            logger.info("Processing model directory (copying all model artifacts)...")
            for src in model_path.iterdir():
                dst = temp_path / src.name
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            logger.info("✓ Model directory contents copied")
        else:
            # 단일 checkpoint 파일 처리 (state_dict만 추출해서 저장)
            logger.info("Processing checkpoint file (extracting state_dict)...")

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
            logger.info("✓ State dict saved to pytorch_model.bin")
        
        # 디렉토리 업로드는 로컬 아티팩트 우선 사용하고, 없을 때만 base_model로 보강
        if _has_config_and_tokenizer(temp_path):
            logger.info("\n✓ Using config/tokenizer from local model artifacts")
        elif base_model:
            logger.info(f"\nLocal config/tokenizer not found. Fetching from base model: {base_model}")
            try:
                config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                config.save_pretrained(temp_path)
                tokenizer.save_pretrained(temp_path)
                logger.info("✓ Config/tokenizer populated from base model")
            except Exception as e:
                logger.error(f"Failed to load base model artifacts: {e}")
                return False
        else:
            logger.error("config/tokenizer files are missing. Provide a complete model directory or set --base_model.")
            return False

        if not _validate_vocab_alignment(temp_path):
            logger.error("Aborting upload due to vocab alignment mismatch.")
            return False
        
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
            "base_model": base_model or "unknown",
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
    # Recommended: use environment variable token
    export HF_TOKEN=your_token_here
  python upload_to_huggingface.py \\
    --model_path ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \\
        --hf_model_id kmseong/WaRP-Safety-Llama3_8B_Instruct

    # Legacy: pass token argument directly
  python upload_to_huggingface.py \\
    --model_path ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \\
        --hf_model_id kmseong/WaRP-Safety-Llama3_8B_Instruct \
        --hf_token your_token_here

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
        help='Path to fine-tuned model directory or checkpoint file (.pt)'
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
        help='(Optional) HuggingFace API token. By default uses HF_TOKEN/HUGGINGFACE_TOKEN env var.'
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
    parser.add_argument(
        '--base_model',
        type=str,
        default=None,
        help='Optional base model ID used only when model_path lacks config/tokenizer files'
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
        commit_message=args.commit_message,
        base_model=args.base_model,
    )
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
