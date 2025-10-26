# Safety-WaRP-LLM: Weight Space Rotation for LLM Safety Alignment

**Safety-First WaRP for Large Language Model Alignment**

## Overview

This project implements a **3-phase safety alignment pipeline** for LLMs:

1. **Phase 1: Basis Construction** 🔄
   - Collect activations from FFN down_proj layers using harmful prompts from do-not-answer dataset
   - Compute activation covariance matrices
   - Perform SVD to obtain orthonormal basis vectors

2. **Phase 2: Importance Scoring** ⚖️
   - Identify important directions in weight space using safety data
   - Compute gradient-based importance scores on basis coefficients
   - Generate importance masks for each layer

3. **Phase 3: Downstream Tuning** 📚
   - Fine-tune model on utility task (GSM8K) with gradient masking
   - Freeze important directions identified in Phase 1-2
   - Update only flat directions to maintain safety guarantees

## Quick Start

### Installation

#### Option 1: Conda (Recommended)

```bash
# Create conda environment with Python 3.11
conda create -n safety-warp python=3.11 -y
conda activate safety-warp

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
or 12.8 nightly
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


# Install project dependencies
cd /path/to/Safety-WaRP-LLM
pip install -r requirements.txt

# (Optional) For LLaMA weights access
huggingface-cli login
```

#### Option 2: venv

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check transformers
python -c "from transformers import AutoModelForCausalLM; print('✓ Transformers OK')"

# Check do-not-answer dataset
python -c "from datasets import load_dataset; d = load_dataset('LibrAI/do-not-answer', split='train'); print(f'✓ Dataset OK: {len(d)} samples')"
```

## Usage

### Phase 1: Basis Construction

**Goal**: Compute SVD basis from safety data activations

```bash
# Using shell script (recommended for first time)
bash scripts/run_phase1.sh --samples 100 --batch-size 4

# Or direct Python command
python train.py \
    --phase 1 \
    --model_name meta-llama/Llama-3-8B \
    --safety_samples 100 \
    --batch_size 4 \
    --target_layers all \
    --layer_type ffn_down \
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 42
```

**Parameters**:
- `--safety_samples`: Number of samples from do-not-answer dataset (default: 100)
- `--batch_size`: Batch size for activation collection (default: 4)
- `--target_layers`: Layer range to process
  - Predefined: `all` (0-31), `early` (0-10), `middle` (11-21), `late` (22-31), `last` (31)
  - Custom: single layer `31`, range `30-31`, `0-5`
- `--layer_type`: Layer component (ffn_down/ffn_up/attn_q/attn_k/attn_v)
- `--dtype`: Model precision (float32/float16/bfloat16)

**Output**:
```
checkpoints/phase1_YYYYMMDD_HHMMSS/
├── basis/
│   ├── layer_00_svd.pt      # SVD: U, S, Vh
│   ├── layer_01_svd.pt
│   └── metadata.json        # Configuration & statistics
└── config.json              # Run configuration
```

**Expected Results**:
- Log file: `logs/phase1_*.log`
- Basis files: 32 layers (one per transformer layer)
- Processing time: ~5-10 minutes for 100 samples (depending on GPU)
- Memory usage: ~15-20GB

### Phase 2: Importance Scoring

*(Implementation coming in next iteration)*

```bash
# Coming soon
python train.py \
    --phase 2 \
    --basis_dir ./checkpoints/phase1_TIMESTAMP/checkpoints/basis \
    --safety_samples 100 \
    --batch_size 4 \
    --keep_ratio 0.1 \
    --device cuda:0 \
    --dtype bfloat16
```

**Parameters**:
- `--basis_dir`: Path to Phase 1 basis checkpoint (required)
- `--safety_samples`: Number of samples for importance scoring (default: 50)
- `--batch_size`: Batch size for processing (default: 4)
- `--keep_ratio`: Fraction of directions to keep as "important" (default: 0.1 = top 10%)
- `--target_layers`: Which layers to score (default: all)

**Output**: Importance masks saved to `checkpoints/phase2_TIMESTAMP/checkpoints/masks/`

**Example - Quick Test**:
```bash
python train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 10 \
    --target_layers last
```

**Example - Full Run**:
```bash
python train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 100 \
    --batch_size 2 \
    --keep_ratio 0.15
```

**Example - Aggressive Pruning (top 5%)**:
```bash
python train.py --phase 2 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --safety_samples 50 \
    --keep_ratio 0.05
```

Or use the shell script:
```bash
bash scripts/run_phase2.sh  # Auto-detects latest Phase 1 basis
```

See `scripts/phase2_examples.sh` for more examples.

### Layer Selection Examples

You can control which layers to process using `--target_layers`:

```bash
# Predefined ranges
python train.py --phase 1 --target_layers all       # All layers (0-31)
python train.py --phase 1 --target_layers early     # Early layers (0-10)
python train.py --phase 1 --target_layers middle    # Middle layers (11-21)
python train.py --phase 1 --target_layers late      # Late layers (22-31)
python train.py --phase 1 --target_layers last      # Last layer (31)

# Custom ranges
python train.py --phase 1 --target_layers 31        # Single layer (layer 31 only)
python train.py --phase 1 --target_layers 30-31     # Range (layers 30-31)
python train.py --phase 1 --target_layers 0-5       # Range (layers 0-5)
```

### Phase 3: Downstream Learning

**Goal**: Fine-tune on GSM8K with masked gradient updates to protect safety

```bash
# Using shell script (recommended)
bash scripts/run_phase3.sh  # Auto-detects Phase 1/2 results

# Or direct Python command
python train.py \
    --phase 3 \
    --basis_dir ./checkpoints/phase1_TIMESTAMP/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_TIMESTAMP/checkpoints/masks \
    --utility_samples 1000 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --device cuda:0 \
    --dtype bfloat16
```

**Parameters**:
- `--basis_dir`: Path to Phase 1 basis checkpoint (required)
- `--masks_dir`: Path to Phase 2 masks checkpoint (required)
- `--utility_samples`: Number of GSM8K train samples (default: 1000)
- `--epochs`: Training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 2)
- `--learning_rate`: Learning rate (default: 5e-5)

**Output**: Fine-tuned model saved to `checkpoints/phase3_TIMESTAMP/checkpoints/`

**Example - Quick Test**:
```bash
python train.py --phase 3 \
    --basis_dir ./checkpoints/phase1_latest/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_latest/checkpoints/masks \
    --utility_samples 50 \
    --epochs 1
```

See `scripts/phase3_examples.sh` for more examples.

