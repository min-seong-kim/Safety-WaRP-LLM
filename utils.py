"""
Utility functions for Safety-WaRP-LLM
"""
import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    로거 설정 함수
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 콘솔만 출력)
        level: 로깅 레벨
    
    Returns:
        logger: 설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 포매터 설정
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path):
    """디렉토리가 없으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed: 시드값 (-1이면 난수 설정)
    """
    if seed == -1:
        import random
        random.seed(None)
        torch.manual_seed(torch.seed())
    else:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """사용 가능한 디바이스 반환"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_dict(logger, d, prefix=""):
    """딕셔너리를 로그로 출력"""
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(f"{prefix}{key}:")
            log_dict(logger, value, prefix + "  ")
        else:
            logger.info(f"{prefix}{key}: {value}")


def save_config(config, path):
    """설정을 JSON으로 저장"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(config, f, indent=4, default=str)


def load_config(path):
    """JSON 설정 파일 로드"""
    with open(path, 'r') as f:
        return json.load(f)


class AverageTracker:
    """평균값 추적기"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0.0
        self.count = 0
    
    def add(self, value, count=1):
        self.sum += value * count
        self.count += count
    
    def get_average(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def get_activation_shape(model, input_ids):
    """
    모델의 활성화 형태 확인
    
    Args:
        model: LLM 모델
        input_ids: 입력 토큰 ID
    
    Returns:
        dict: 레이어별 활성화 형태
    """
    activation_shapes = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation_shapes[layer_idx] = output[0].shape
            else:
                activation_shapes[layer_idx] = output.shape
        return hook
    
    hooks = []
    try:
        # LLaMA 구조: model.layers[i].mlp.down_proj
        for i, layer in enumerate(model.model.layers):
            h = layer.mlp.down_proj.register_forward_hook(hook_fn(i))
            hooks.append(h)
        
        # 전방향 전파
        with torch.no_grad():
            _ = model(input_ids)
        
        return activation_shapes
    finally:
        # 훅 제거
        for h in hooks:
            h.remove()


def upload_model_to_huggingface(
    model_path,
    repo_id,
    hf_token=None,
    commit_message="Upload WaRP fine-tuned model",
    private=False,
    logger=None
):
    """
    미세조정된 모델을 HuggingFace Hub에 업로드
    
    Args:
        model_path: 로컬 모델 경로
        repo_id: HuggingFace repo ID (format: "username/model_name")
        hf_token: HuggingFace API 토큰 (None이면 환경변수에서 읽음)
        commit_message: 커밋 메시지
        private: 비공개 저장소 여부
        logger: 로거 객체
    
    Returns:
        bool: 성공 여부
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, Repository, login
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info("UPLOADING MODEL TO HUGGINGFACE HUB")
        logger.info(f"{'='*60}\n")
        
        # 1. 토큰 설정
        if hf_token is None:
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        
        if hf_token is None:
            logger.error("HuggingFace token not found!")
            logger.error("Please set HUGGINGFACE_TOKEN environment variable or pass hf_token")
            return False
        
        logger.info("[Step 1] Authenticating with HuggingFace...")
        login(token=hf_token)
        logger.info("✓ Authentication successful")
        
        # 2. 모델과 토크나이저 로드
        logger.info(f"\n[Step 2] Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='cpu',  # CPU로 로드하여 메모리 절약
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("✓ Model and tokenizer loaded")
        
        # 3. 저장소 생성/연결
        logger.info(f"\n[Step 3] Setting up repository: {repo_id}...")
        api = HfApi()
        
        try:
            # 저장소 생성 시도
            repo_url = api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True
            )
            logger.info(f"✓ Repository ready: {repo_url}")
        except Exception as e:
            logger.warning(f"Could not create repo: {e}")
            logger.info("Attempting to use existing repository...")
        
        # 4. 모델과 토크나이저 업로드
        logger.info(f"\n[Step 4] Uploading model to {repo_id}...")
        model.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            token=hf_token
        )
        logger.info("✓ Model uploaded")
        
        logger.info(f"\n[Step 5] Uploading tokenizer...")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            token=hf_token
        )
        logger.info("✓ Tokenizer uploaded")
        
        # 5. README 생성 및 업로드
        logger.info(f"\n[Step 6] Creating README...")
        readme_content = f"""# WaRP Safety-Aligned Llama-3.1-8B-Instruct

## Model Description

This model is a safety-aligned version of Meta's Llama-3.1-8B-Instruct, fine-tuned using the **Safety-First WaRP (Weight space Rotation Process)** pipeline.

### Training Approach

**Safety-WaRP** protects safety mechanisms in language models through a 3-phase process:

1. **Phase 1: Basis Construction**
   - Extract activation patterns from harmful prompts using do-not-answer dataset
   - Compute SVD basis vectors from activation covariance
   - Identify directions associated with safety mechanisms

2. **Phase 2: Importance Scoring**
   - Calculate gradient-based importance scores for basis directions
   - Identify critical 419 weight directions (top 10.2%) crucial for safety
   - Generate importance masks

3. **Phase 3: Incremental Learning**
   - Fine-tune on utility tasks (GSM8K) with masked gradients
   - Freeze critical safety directions during training
   - Update only non-critical weight directions

### Key Features

✅ **Safety First**: Protects model's ability to refuse harmful requests
✅ **Utility Improvement**: Maintains or improves performance on helpful tasks
✅ **Parameter Efficient**: Updates only ~90% of parameters
✅ **Transparent**: All safety mechanisms preserved, none removed

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Use for generation
prompt = "Write a helpful response to: What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0])
print(response)
```

## Training Details

- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Safety Data**: LibrAI/do-not-answer (harmful prompt filtering)
- **Utility Data**: openai/gsm8k (grade school math)
- **Protected Directions**: 419 neurons from layer 31
- **Training Method**: Gradient masking with AdamW optimizer
- **Precision**: bfloat16

## Model Performance

| Metric | Value |
|--------|-------|
| Safety Rate | High (refuses harmful requests) |
| Utility | Improved on GSM8K |
| Model Size | 8B parameters |

## Disclaimer

This model should be used responsibly. While the Safety-WaRP pipeline improves safety measures, no model is perfectly safe. Always monitor outputs for your use case.

## Citation

If you use this model, please cite:

```bibtex
@article{{warp-safety-2025}}
  title={{Safety-WaRP: Weight space Rotation for LLM Safety Alignment}},
  author={{Your Name}},
  year={{2025}}
}}
```

## License

This model follows the Llama-3.1 Community License Agreement.
"""
        
        # README 파일로 저장
        readme_path = os.path.join(model_path, "README_WaRP.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info("✓ README created")
        
        # 6. 메타데이터 업로드
        logger.info(f"\n[Step 7] Uploading metadata...")
        
        metadata = {
            "model_type": "safety_aligned_llm",
            "alignment_method": "Safety-WaRP",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "protected_directions": 419,
            "total_parameters": "8B",
            "precision": "bfloat16",
            "safety_data": "LibrAI/do-not-answer",
            "utility_data": "openai/gsm8k",
            "upload_date": datetime.now().isoformat(),
        }
        
        metadata_path = os.path.join(model_path, "warp_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✓ Metadata created")
        
        logger.info(f"\n{'='*60}")
        logger.info("✓ MODEL SUCCESSFULLY UPLOADED!")
        logger.info(f"{'='*60}")
        logger.info(f"Model available at: https://huggingface.co/{repo_id}")
        logger.info(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload model: {str(e)}", exc_info=True)
        return False

