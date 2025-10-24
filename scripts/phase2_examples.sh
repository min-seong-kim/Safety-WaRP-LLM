#!/bin/bash

# Safety-WaRP-LLM: Phase 2 - Quick Test Cases
# 다양한 시나리오에 따른 Phase 2 실행 예제

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "=================================================="
echo "Safety-WaRP-LLM: Phase 2 - Example Test Cases"
echo "=================================================="
echo ""
echo "Usage: source scripts/phase2_examples.sh"
echo "Then run one of the example commands below"
echo ""

# ============================================================
# 예제 1: 최소 설정 (기본값)
# ============================================================
cat << 'EOF'
## Example 1: Minimal Configuration (Default Settings)
## Use most recent Phase 1 basis automatically
EXAMPLE1="bash $SCRIPT_DIR/run_phase2.sh"

## 또는 직접 실행:
EXAMPLE1_DIRECT="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --batch_size 4 \
    --keep_ratio 0.1"
EOF
echo ""

# ============================================================
# 예제 2: 완전 설정 (모든 옵션)
# ============================================================
cat << 'EOF'
## Example 2: Full Configuration (All Options)
EXAMPLE2="python $PROJECT_ROOT/train.py --phase 2 \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --basis_dir ./checkpoints/phase1_20251023_200936/checkpoints/basis \
    --safety_samples 100 \
    --batch_size 2 \
    --target_layers 'last' \
    --layer_type 'ffn_down' \
    --keep_ratio 0.15 \
    --device 'cuda:0' \
    --dtype 'bfloat16' \
    --seed 42 \
    --debug"
EOF
echo ""

# ============================================================
# 예제 3: 빠른 테스트 (적은 샘플)
# ============================================================
cat << 'EOF'
## Example 3: Quick Test (Few Samples - Fast)
EXAMPLE3="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 10 \
    --batch_size 8 \
    --target_layers 'last' \
    --keep_ratio 0.1"
EOF
echo ""

# ============================================================
# 예제 4: 다양한 keep_ratio 테스트
# ============================================================
cat << 'EOF'
## Example 4: Test Different Keep Ratios
## Sparse (더 많이 제거)
EXAMPLE4A="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --keep_ratio 0.05"

## Dense (더 적게 제거)
EXAMPLE4B="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --keep_ratio 0.3"
EOF
echo ""

# ============================================================
# 예제 5: 특정 레이어만 테스트
# ============================================================
cat << 'EOF'
## Example 5: Specific Layers
## Last layer only
EXAMPLE5A="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --target_layers 'last' \
    --keep_ratio 0.1"

## Last 4 layers
EXAMPLE5B="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --target_layers '28-31' \
    --keep_ratio 0.1"

## Single layer (31)
EXAMPLE5C="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --target_layers '31' \
    --keep_ratio 0.1"
EOF
echo ""

# ============================================================
# 예제 6: 다양한 배치 크기
# ============================================================
cat << 'EOF'
## Example 6: Different Batch Sizes
## Smaller batch (more memory efficient)
EXAMPLE6A="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --batch_size 2 \
    --safety_samples 100 \
    --keep_ratio 0.1"

## Larger batch (faster)
EXAMPLE6B="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --batch_size 8 \
    --safety_samples 100 \
    --keep_ratio 0.1"
EOF
echo ""

# ============================================================
# 예제 7: 디버깅 모드
# ============================================================
cat << 'EOF'
## Example 7: Debug Mode (verbose output + small dataset)
EXAMPLE7="python $PROJECT_ROOT/train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 5 \
    --batch_size 1 \
    --target_layers 'last' \
    --keep_ratio 0.1 \
    --debug"
EOF
echo ""

# ============================================================
# 예제 8: 최신 Phase 1 자동 감지 사용
# ============================================================
cat << 'EOF'
## Example 8: Auto-detect Latest Phase 1 Results
EXAMPLE8="bash $SCRIPT_DIR/run_phase2.sh"
EOF
echo ""

# ============================================================
# 주요 옵션 설명
# ============================================================
cat << 'EOF'
================================================
Key Parameters:
================================================
--phase 2                    : Always use phase 2
--basis_dir                  : Path to Phase 1 basis checkpoint (required)
--safety_samples             : Number of safety samples to use for importance scoring
--batch_size                 : Batch size for processing
--target_layers              : Layer selection (all/early/middle/late/last or custom)
--layer_type                 : Layer type to analyze (default: ffn_down)
--keep_ratio                 : Fraction of directions to keep (default: 0.1)
  * 0.1 = Keep top 10% (remove 90%)
  * 0.2 = Keep top 20% (remove 80%)
  * 0.05 = Keep top 5% (more aggressive pruning)
--device                     : Device (cuda:0, cuda:1, etc.)
--dtype                      : Data type (bfloat16, float32)
--debug                      : Enable debug logging

================================================
Common Workflows:
================================================

1. Quick Test (< 1 min):
   python train.py --phase 2 --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --safety_samples 10 --target_layers 'last'

2. Full Run (5-10 min):
   python train.py --phase 2 --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --safety_samples 100

3. Aggressive Pruning (keep only 5%):
   python train.py --phase 2 --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --safety_samples 50 --keep_ratio 0.05

4. Conservative Pruning (keep 30%):
   python train.py --phase 2 --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --safety_samples 50 --keep_ratio 0.3

5. Multi-GPU Debug:
   python train.py --phase 2 --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --safety_samples 5 --batch_size 1 --debug

================================================
Expected Outputs:
================================================
Checkpoints/
  phase2_TIMESTAMP/
    checkpoints/
      masks/
        - layer_*.pt (importance masks for each layer)
        - metadata.json (keep_ratio, thresholds, etc.)
    logs/
      - phase2_TIMESTAMP.log

================================================
EOF

echo ""
echo "To run an example, copy and paste the command above"
echo "=================================================="
