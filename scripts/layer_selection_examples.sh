#!/bin/bash

# Safety-WaRP-LLM: 레이어 범위 지정 예제

echo "======================================"
echo "Layer Selection Examples"
echo "======================================"

# 예제 1: 마지막 레이어(31번)만
echo ""
echo "[Example 1] Last layer only (layer 31)"
echo "Command: python train.py --phase 1 --target_layers last --safety_samples 100 --batch_size 4"
# python train.py --phase 1 --target_layers last --safety_samples 100 --batch_size 4

# 예제 2: 특정 레이어만
echo ""
echo "[Example 2] Single layer (layer 31)"
echo "Command: python train.py --phase 1 --target_layers 31 --safety_samples 100 --batch_size 4"
# python train.py --phase 1 --target_layers 31 --safety_samples 100 --batch_size 4

# 예제 3: 범위 지정 (마지막 2개)
echo ""
echo "[Example 3] Range (layers 30-31)"
echo "Command: python train.py --phase 1 --target_layers 30-31 --safety_samples 100 --batch_size 4"
# python train.py --phase 1 --target_layers 30-31 --safety_samples 100 --batch_size 4

# 예제 4: 범위 지정 (초반)
echo ""
echo "[Example 4] Range (layers 0-5)"
echo "Command: python train.py --phase 1 --target_layers 0-5 --safety_samples 100 --batch_size 4"
# python train.py --phase 1 --target_layers 0-5 --safety_samples 100 --batch_size 4

# 예제 5: 사전정의 범위
echo ""
echo "[Example 5] Predefined ranges"
echo "  - early (0-10): python train.py --phase 1 --target_layers early"
echo "  - middle (11-21): python train.py --phase 1 --target_layers middle"
echo "  - late (22-31): python train.py --phase 1 --target_layers late"
echo "  - all (0-31): python train.py --phase 1 --target_layers all"

echo ""
echo "======================================"
echo "Format Guide:"
echo "======================================"
echo "  all              : All layers (0-31)"
echo "  early            : Early layers (0-10)"
echo "  middle           : Middle layers (11-21)"
echo "  late             : Late layers (22-31)"
echo "  last             : Last layer only (31)"
echo "  31               : Specific layer (layer 31)"
echo "  30-31            : Range (layers 30-31)"
echo "  0-5              : Range (layers 0-5)"
echo ""
