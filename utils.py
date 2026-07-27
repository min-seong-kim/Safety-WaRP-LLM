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



# ============================================================================
# Resource profiling: phase/stage별 소요 시간 + VRAM 측정
# ----------------------------------------------------------------------------
#   두 종류의 메모리를 함께 기록한다.
#     1) torch allocator 기준 (프로세스 정확값)
#          - alloc    : torch.cuda.max_memory_allocated  (실제 텐서)
#          - reserved : torch.cuda.max_memory_reserved   (caching allocator 예약)
#     2) 디바이스 기준 (백그라운드 샘플러, total - free)
#          - CUDA context / cuBLAS workspace / NCCL 버퍼 등 torch 밖 사용량 포함
#            → nvidia-smi 로 보이는 값과 대응. 실험 재현 시 필요한 GPU 용량 기준.
#   사용 예:
#       prof = ResourceProfiler(logger, label="Phase 1", json_path="logs/p1_profile.json")
#       with prof.stage("load_model"):
#           builder.load_model()
#       ...
#       prof.finalize()   # 요약 테이블 로깅 + JSON 저장
# ============================================================================

import time as _time
import threading as _threading
from contextlib import contextmanager as _contextmanager

_GB = 1024.0 ** 3


def format_duration(seconds):
    """초 → 'HH:MM:SS' 문자열"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _GPUMemorySampler:
    """
    백그라운드 스레드로 각 visible GPU의 사용량(total - free)을 주기적으로 기록.
    (t, {device: used_bytes}) 타임라인을 남겨서, 임의 구간의 peak를 사후 계산한다.
    """

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []          # [(timestamp, {dev: used_bytes})]
        self.total_bytes = {}      # {dev: total_bytes}
        self._stop = _threading.Event()
        self._thread = None
        self.available = torch.cuda.is_available()

    def _sample_once(self):
        snapshot = {}
        for dev in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(dev)
            except Exception:
                continue
            snapshot[dev] = total - free
            self.total_bytes[dev] = total
        if snapshot:
            self.samples.append((_time.time(), snapshot))

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if not self.available or self._thread is not None:
            return
        self._sample_once()
        self._thread = _threading.Thread(target=self._run, daemon=True,
                                         name='gpu-mem-sampler')
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2 * self.interval + 1.0)
        self._thread = None
        try:
            self._sample_once()
        except Exception:
            pass

    def peak_between(self, t0, t1):
        """
        구간 [t0, t1] 내 디바이스별 peak 사용량 (bytes).
        구간이 샘플링 주기보다 짧아 샘플이 하나도 없으면, 시간적으로 가장 가까운
        샘플 하나로 대체한다 (0으로 보고되는 것을 방지).
        """
        peak = {}
        for ts, snapshot in self.samples:
            if ts < t0 or ts > t1:
                continue
            for dev, used in snapshot.items():
                if used > peak.get(dev, 0):
                    peak[dev] = used
        if not peak and self.samples:
            mid = (t0 + t1) / 2.0
            _, nearest = min(self.samples, key=lambda x: abs(x[0] - mid))
            peak = dict(nearest)
        return peak


class ResourceProfiler:
    """
    Phase 단위 실행 시간 / VRAM 프로파일러.

    Args:
        logger: 로거 (None이면 print)
        label: 프로파일 대상 이름 (예: "Phase 1")
        json_path: 요약 JSON 저장 경로 (None이면 저장 안 함)
        sample_interval: 디바이스 메모리 샘플링 주기(초)
        meta: JSON에 함께 남길 메타데이터 dict (설정값 등)
    """

    def __init__(self, logger=None, label="run", json_path=None,
                 sample_interval=0.5, meta=None):
        self.logger = logger
        self.label = label
        self.json_path = json_path
        self.meta = dict(meta or {})
        self.stages = []
        self.t_start = _time.time()
        self.cuda = torch.cuda.is_available()
        self.sampler = _GPUMemorySampler(interval=sample_interval)
        self.sampler.start()
        # 프로파일 시작 시점 디바이스 사용량 (다른 프로세스 + CUDA context).
        # GPU를 단독 점유하지 않는 환경에서 "이 실험이 추가로 쓴 VRAM"을 보려면
        # peak_device - baseline (= *_delta_gb) 을 보면 된다.
        self.device_baseline = max(
            self.sampler.peak_between(self.t_start - 5.0, _time.time()).values(),
            default=0,
        )
        self._gpu_names = {}
        if self.cuda:
            for dev in range(torch.cuda.device_count()):
                try:
                    self._gpu_names[dev] = torch.cuda.get_device_name(dev)
                except Exception:
                    self._gpu_names[dev] = f"cuda:{dev}"

    # ---------------------------------------------------------------- log
    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    # -------------------------------------------------------------- stage
    @_contextmanager
    def stage(self, name):
        """한 단계(stage)의 wall-clock 시간과 VRAM peak을 기록하는 컨텍스트 매니저."""
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = _time.time()
        self._log(f"[PROFILE] ▶ {self.label} / {name} 시작")
        status = 'ok'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            if self.cuda:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            t1 = _time.time()
            record = self._record(name, t0, t1, status)
            self.stages.append(record)
            self._log(
                f"[PROFILE] ■ {self.label} / {name} {status} | "
                f"time={format_duration(record['seconds'])} ({record['seconds']:.1f}s) | "
                f"torch_alloc_peak={record['torch_alloc_peak_gb']:.2f}GB | "
                f"torch_reserved_peak={record['torch_reserved_peak_gb']:.2f}GB | "
                f"device_peak={record['device_peak_gb']:.2f}GB"
            )

    def _record(self, name, t0, t1, status):
        alloc_peak, reserved_peak = {}, {}
        if self.cuda:
            for dev in range(torch.cuda.device_count()):
                try:
                    alloc_peak[dev] = torch.cuda.max_memory_allocated(dev)
                    reserved_peak[dev] = torch.cuda.max_memory_reserved(dev)
                except Exception:
                    pass
        device_peak = self.sampler.peak_between(t0, t1)
        return {
            'stage': name,
            'status': status,
            'seconds': t1 - t0,
            'duration': format_duration(t1 - t0),
            'torch_alloc_peak_gb': max(alloc_peak.values(), default=0) / _GB,
            'torch_reserved_peak_gb': max(reserved_peak.values(), default=0) / _GB,
            'device_peak_gb': max(device_peak.values(), default=0) / _GB,
            'device_delta_gb': max(
                max(device_peak.values(), default=0) - self.device_baseline, 0) / _GB,
            'per_device': {
                str(dev): {
                    'torch_alloc_peak_gb': alloc_peak.get(dev, 0) / _GB,
                    'torch_reserved_peak_gb': reserved_peak.get(dev, 0) / _GB,
                    'device_peak_gb': device_peak.get(dev, 0) / _GB,
                }
                for dev in set(list(alloc_peak.keys()) + list(device_peak.keys()))
            },
        }

    # ------------------------------------------------------------ summary
    def summary(self):
        total_seconds = _time.time() - self.t_start
        device_peak_all = self.sampler.peak_between(self.t_start, _time.time())
        return {
            'label': self.label,
            'total_seconds': total_seconds,
            'total_duration': format_duration(total_seconds),
            'peak_torch_alloc_gb': max(
                (s['torch_alloc_peak_gb'] for s in self.stages), default=0.0),
            'peak_torch_reserved_gb': max(
                (s['torch_reserved_peak_gb'] for s in self.stages), default=0.0),
            'peak_device_gb': max(device_peak_all.values(), default=0) / _GB,
            'device_baseline_gb': self.device_baseline / _GB,
            'peak_device_delta_gb': max(
                max(device_peak_all.values(), default=0) - self.device_baseline, 0) / _GB,
            'gpu_total_gb': {
                str(dev): tot / _GB for dev, tot in self.sampler.total_bytes.items()
            },
            'gpu_names': {str(k): v for k, v in self._gpu_names.items()},
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'stages': self.stages,
            'meta': self.meta,
        }

    def log_summary(self):
        s = self.summary()
        self._log("=" * 78)
        self._log(f"[PROFILE] {self.label} 리소스 요약 (시간 / VRAM)")
        self._log("=" * 78)
        self._log(f"{'stage':<28}{'time':>12}{'torch_alloc':>14}"
                  f"{'torch_resv':>13}{'device':>11}{'device-base':>13}")
        self._log("-" * 91)
        for st in self.stages:
            flag = '' if st['status'] == 'ok' else ' (failed)'
            self._log(
                f"{st['stage'][:26] + flag:<28}"
                f"{st['duration']:>12}"
                f"{st['torch_alloc_peak_gb']:>13.2f}G"
                f"{st['torch_reserved_peak_gb']:>12.2f}G"
                f"{st['device_peak_gb']:>10.2f}G"
                f"{st['device_delta_gb']:>12.2f}G"
            )
        self._log("-" * 91)
        self._log(
            f"{'TOTAL':<28}{s['total_duration']:>12}"
            f"{s['peak_torch_alloc_gb']:>13.2f}G"
            f"{s['peak_torch_reserved_gb']:>12.2f}G"
            f"{s['peak_device_gb']:>10.2f}G"
            f"{s['peak_device_delta_gb']:>12.2f}G"
        )
        for dev, tot in s['gpu_total_gb'].items():
            self._log(f"  GPU {dev} ({s['gpu_names'].get(dev, '?')}): "
                      f"capacity {tot:.1f}GB")
        self._log(f"  device baseline (프로파일 시작 시점 사용량, 타 프로세스 포함): "
                  f"{s['device_baseline_gb']:.2f}GB")
        self._log("  torch_alloc/torch_resv = 이 프로세스의 torch allocator 기준, "
                  "device = nvidia-smi 기준 전체 사용량")
        self._log("=" * 78)
        return s

    def save(self):
        if not self.json_path:
            return None
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.json_path)), exist_ok=True)
            with open(self.json_path, 'w') as f:
                json.dump(self.summary(), f, indent=2)
            self._log(f"[PROFILE] 요약 저장: {self.json_path}")
            return self.json_path
        except Exception as e:
            self._log(f"[PROFILE] 요약 저장 실패: {e}")
            return None

    def finalize(self, wandb_log=True):
        """샘플러 정지 → 요약 로깅 → JSON 저장 (+ W&B summary 기록)."""
        self.sampler.stop()
        s = self.log_summary()
        self.save()
        if wandb_log:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary[f"{self.label}/total_seconds"] = s['total_seconds']
                    wandb.run.summary[f"{self.label}/peak_torch_alloc_gb"] = s['peak_torch_alloc_gb']
                    wandb.run.summary[f"{self.label}/peak_torch_reserved_gb"] = s['peak_torch_reserved_gb']
                    wandb.run.summary[f"{self.label}/peak_device_gb"] = s['peak_device_gb']
                    for st in self.stages:
                        wandb.run.summary[f"{self.label}/stage/{st['stage']}/seconds"] = st['seconds']
                        wandb.run.summary[f"{self.label}/stage/{st['stage']}/device_peak_gb"] = st['device_peak_gb']
            except Exception:
                pass
        return s
