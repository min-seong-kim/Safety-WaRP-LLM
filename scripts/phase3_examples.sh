#!/bin/bash

# Safety-WaRP-LLM: Phase 3 - Example Test Cases
# 다양한 시나리오에 따른 Phase 3 실행 예제

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "=================================================="
echo "Safety-WaRP-LLM: Phase 3 - Example Test Cases"
echo "=================================================="
echo ""
echo "Usage: source scripts/phase3_examples.sh"
echo "Then run one of the example commands below"
echo ""

# ============================================================
# 예제 1: 최소 설정 (기본값)
# ============================================================
cat << 'EOF'
## Example 1: Minimal Configuration (Auto-detect basis/masks)
## Uses most recent Phase 1 basis and Phase 2 masks automatically
EXAMPLE1="bash $SCRIPT_DIR/run_phase3.sh"

## Or direct Python command:
EXAMPLE1_DIRECT="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 100 \
    --epochs 1 \
    --batch_size 2"
EOF
echo ""

# ============================================================
# 예제 2: 완전 설정 (모든 옵션)
# ============================================================
cat << 'EOF'
## Example 2: Full Configuration (All Options)
EXAMPLE2="python $PROJECT_ROOT/train.py --phase 3 \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --basis_dir ./checkpoints/phase1_20251023_200936/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_20251023_222910/checkpoints/masks \
    --utility_samples 1000 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --device 'cuda:0' \
    --dtype 'bfloat16' \
    --seed 42 \
    --debug"
EOF
echo ""

# ============================================================
# 예제 3: 빠른 테스트 (적은 샘플, 적은 에포크)
# ============================================================
cat << 'EOF'
## Example 3: Quick Test (Few Samples - Fast)
EXAMPLE3="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 10 \
    --epochs 1 \
    --batch_size 2"
EOF
echo ""

# ============================================================
# 예제 4: 메모리 효율적 설정
# ============================================================
cat << 'EOF'
## Example 4: Memory-Efficient (Smaller batch size)
EXAMPLE4="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 500 \
    --epochs 2 \
    --batch_size 1 \
    --learning_rate 1e-5"
EOF
echo ""

# ============================================================
# 예제 5: 다양한 학습률 테스트
# ============================================================
cat << 'EOF'
## Example 5: Different Learning Rates
## Lower learning rate (more conservative)
EXAMPLE5A="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 500 \
    --epochs 3 \
    --learning_rate 1e-5"

## Higher learning rate (faster learning)
EXAMPLE5B="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 500 \
    --epochs 3 \
    --learning_rate 1e-4"
EOF
echo ""

# ============================================================
# 예제 6: 다양한 에포크 수
# ============================================================
cat << 'EOF'
## Example 6: Different Number of Epochs
## Short training (1 epoch)
EXAMPLE6A="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 500 \
    --epochs 1"

## Long training (5 epochs)
EXAMPLE6B="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 500 \
    --epochs 5"
EOF
echo ""

# ============================================================
# 예제 7: 디버깅 모드
# ============================================================
cat << 'EOF'
## Example 7: Debug Mode (verbose output + small dataset)
EXAMPLE7="python $PROJECT_ROOT/train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 10 \
    --epochs 1 \
    --batch_size 1 \
    --debug"
EOF
echo ""

# ============================================================
# 예제 8: 자동 감지 사용
# ============================================================
cat << 'EOF'
## Example 8: Auto-detect Latest Phase 1/2 Results
EXAMPLE8="bash $SCRIPT_DIR/run_phase3.sh"
EOF
echo ""

# ============================================================
# 주요 옵션 설명
# ============================================================
cat << 'EOF'
================================================
Key Parameters:
================================================
--phase 3                    : Always use phase 3
--basis_dir                  : Path to Phase 1 basis checkpoint (required)
--masks_dir                  : Path to Phase 2 masks checkpoint (required)
--utility_samples            : Number of GSM8K train samples to use
--epochs                     : Number of training epochs
--batch_size                 : Batch size for training
--learning_rate              : Learning rate (default: 5e-5)
--weight_decay               : Weight decay for AdamW (default: 0.01)
--device                     : Device (cuda:0, cuda:1, cpu, etc.)
--dtype                      : Data type (bfloat16, float32, float16)
--debug                      : Enable debug logging

================================================
Common Workflows:
================================================

1. Quick Test (< 5 min):
   bash $SCRIPT_DIR/run_phase3.sh
   # or
   python $PROJECT_ROOT/train.py --phase 3 \
       --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
       --utility_samples 50 --epochs 1 --batch_size 2

2. Normal Training (10-20 min):
   python $PROJECT_ROOT/train.py --phase 3 \
       --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
       --utility_samples 500 --epochs 3

3. Full Training (30-60 min, requires 40GB VRAM):
   python $PROJECT_ROOT/train.py --phase 3 \
       --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
       --utility_samples 5000 --epochs 5 --batch_size 4

4. Memory-Constrained (GPU with < 40GB):
   python $PROJECT_ROOT/train.py --phase 3 \
       --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
       --utility_samples 200 --epochs 2 --batch_size 1

5. Debug Run:
   python $PROJECT_ROOT/train.py --phase 3 \
       --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
       --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
       --utility_samples 5 --epochs 1 --batch_size 1 --debug

================================================
Expected Training Time:
================================================
- Quick Test: ~1-2 min
- Normal Training: ~15-30 min
- Full Training: ~60-120 min
(Varies by GPU, batch size, number of samples)

================================================
Post-Training Evaluation:
================================================

After Phase 3 completes, evaluate the model:

EVALUATE: bash $SCRIPT_DIR/run_evaluation.sh

This will measure:
  - Safety ASR (Attack Success Rate) ↓ lower is better
  - Safety Rate (Refusal Rate) ↑ higher is better
  - Utility Accuracy (GSM8K) ↑ higher is better

================================================
EOF

echo ""
echo "To run an example, copy and paste the command above"
echo "=================================================="
