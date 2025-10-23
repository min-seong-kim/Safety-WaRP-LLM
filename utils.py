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
