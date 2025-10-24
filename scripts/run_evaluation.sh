#!/bin/bash

# Safety Evaluation Script
# Phase 3 완료 후 모델의 안전성과 유틸리티를 평가

set -e

echo "=================================================="
echo "Safety-WaRP-LLM: Safety Evaluation"
echo "=================================================="

# 설정
MODEL_PATH="${1:-}"
DEVICE="cuda:0"
SAFETY_SAMPLES=100    # do-not-answer validation 샘플 수
UTILITY_SAMPLES=100   # GSM8K test 샘플 수
REFUSAL_METHOD="keyword"  # 거절 판정 방법

# 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [ -z "$MODEL_PATH" ]; then
    echo "[*] Finding most recent Phase 3 model..."
    PHASE3_DIR=$(ls -td "$PROJECT_ROOT/checkpoints"/phase3_* 2>/dev/null | head -1)
    if [ -z "$PHASE3_DIR" ]; then
        echo "ERROR: No Phase 3 model directory found!"
        echo "Please run Phase 3 first: bash scripts/run_phase3.sh"
        exit 1
    fi
    MODEL_PATH="$PHASE3_DIR/checkpoints/phase3_best.pt"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Device: $DEVICE"
echo "  Safety Samples: $SAFETY_SAMPLES"
echo "  Utility Samples: $UTILITY_SAMPLES"
echo "  Refusal Method: $REFUSAL_METHOD"
echo ""

echo "[*] Running evaluation..."
echo "=================================================="

cd "$PROJECT_ROOT"
python -c "
import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.safety_evaluator import SafetyEvaluator
from utils import setup_logger
import logging

# 로거 설정
logger = setup_logger('safety_eval', './logs/safety_evaluation.log')

# 모델 로드
print('[*] Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='$DEVICE',
    trust_remote_code=True
)

# 체크포인트 로드 (있으면)
try:
    checkpoint = torch.load('$MODEL_PATH', map_location='$DEVICE')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Loaded checkpoint from $MODEL_PATH')
except Exception as e:
    logger.warning(f'Could not load checkpoint: {e}')

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 평가기 생성
evaluator = SafetyEvaluator(
    model=model,
    tokenizer=tokenizer,
    device='$DEVICE',
    logger=logger
)

# 평가 실행
print()
print('='*60)
print('Running Evaluation')
print('='*60)
print()

results = evaluator.evaluate_all(
    safety_samples=$SAFETY_SAMPLES,
    utility_samples=$UTILITY_SAMPLES,
    refusal_method='$REFUSAL_METHOD'
)

# 결과 저장
import json
with open('./logs/evaluation_results.json', 'w') as f:
    # numpy float32를 일반 float로 변환
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.floating, np.integer)):
            results_serializable[k] = float(v)
        else:
            results_serializable[k] = v
    json.dump(results_serializable, f, indent=2)

print()
print('='*60)
print('Results saved to ./logs/evaluation_results.json')
print('='*60)
" --safety-samples "$SAFETY_SAMPLES" --utility-samples "$UTILITY_SAMPLES"

echo ""
echo "=================================================="
echo "✓ Evaluation completed!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - JSON: ./logs/evaluation_results.json"
echo "  - Log: ./logs/safety_evaluation.log"
echo ""
