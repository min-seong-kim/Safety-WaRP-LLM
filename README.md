# Safety-WaRP-LLM: Weight Space Rotation for LLM Safety Alignment

**Safety-First WaRP for Large Language Model Alignment**

## Overview

This project implements a **3-phase safety alignment pipeline** for LLMs:

1. **Phase 1: Basis Construction** 🔄
   - Collect activations from transformer layers using harmful prompts from do-not-answer dataset
   - Compute activation covariance matrices
   - Perform SVD to obtain orthonormal basis vectors

2. **Phase 2: Importance Scoring** ⚖️
   - Identify important directions in weight space using safety dataset
   - Compute gradient-based importance scores on basis coefficients
   - Generate importance masks for each layer

3. **Phase 3: Downstream Tuning** 📚
   - Fine-tune model on utility task (GSM8K) with gradient masking
   - Freeze important directions identified in Phase 1-2
   - Update only flat directions to maintain safety guarantees

## Quick Start

### Installation


```bash
# Create conda environment with Python 3.11
conda create -n safety-warp python=3.11 -y
conda activate safety-warp

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
# or 12.8 nightly
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


# Install project dependencies
cd /path/to/Safety-WaRP-LLM
pip install -r requirements.txt


## Usage

### Phase 1: Basis Construction

**Goal**: Compute SVD basis from safety data activations

```bash
python train.py \
    --phase 1 \
    --model_name meta-llama/Llama-3-8B \
    --safety_samples 100 \
    --batch_size 4 \
    --target_layers all \
    --layer_type ffn_down \
    --device cuda:0 \
    --dtype bfloat16 \
```

**Parameters**:
- `--safety_samples`: Number of samples from do-not-answer dataset (default: 100)
- `--batch_size`: Batch size for activation collection (default: 4)
- `--target_layers`: Layer range to process
  - Predefined: `all` (0-31), `early` (0-10), `middle` (11-21), `late` (22-31), `last` (31)
  - Custom: single layer `31`, range `30-31`, `0-5`
- `--layer_type`: Layer component to analyze and mask
- `--dtype`: Model precision (float32/float16/bfloat16)

### Supported Layer Types

Safety-WaRP now supports **5 different layer types** that can be specified via `--layer_type`:

| Layer Type | Component | Shape | Example |
|-----------|-----------|--------|---------|
| `ffn_down` | MLP Down Projection | (4096, 14336) | `--layer_type ffn_down` |
| `ffn_up` | MLP Up Projection | (14336, 4096) | `--layer_type ffn_up` |
| `attn_q` | Self-Attention Q | (4096, 4096) | `--layer_type attn_q` |
| `attn_k` | Self-Attention K | (4096, 4096) | `--layer_type attn_k` |
| `attn_v` | Self-Attention V | (4096, 4096) | `--layer_type attn_v` |


**Output**:
```
checkpoints/phase1_YYYYMMDD_HHMMSS/
├── basis/
│   ├── layer_00_svd.pt      # SVD: U, S, Vh
│   ├── layer_01_svd.pt
│   └── metadata.json        # Configuration & statistics
└── config.json              # Run configuration
```
### Phase 2: Importance Scoring

**Goal**: Identify important weight directions using safety data

```bash
# Using previously computed Phase 1 basis
python train.py \
    --phase 2 \
    --basis_dir ./checkpoints/phase1_TIMESTAMP/checkpoints/basis \
    --safety_samples 100 \
    --batch_size 4 \
    --keep_ratio 0.1 \
    --layer_type ffn_down \
    --device cuda:0 \
    --dtype bfloat16
```

**Parameters**:
- `--basis_dir`: Path to Phase 1 basis checkpoint (required)
- `--safety_samples`: Number of samples for importance scoring (default: 50)
- `--batch_size`: Batch size for processing (default: 4)
- `--keep_ratio`: Fraction of directions to keep as "important" (default: 0.1 = top 10%)
- `--layer_type`: **MUST match Phase 1 layer_type** (ffn_down/ffn_up/attn_q/attn_k/attn_v)
- `--target_layers`: Which layers to score (default: all, must match Phase 1)

**Output**: Importance masks saved to `checkpoints/phase2_TIMESTAMP/checkpoints/masks/`


### Phase 3: Downstream Learning

**Goal**: Fine-tune on GSM8K with masked gradient updates to protect safety

```bash
# Using shell script (recommended)
bash scripts/run_phase3.sh  # Auto-detects Phase 1/2 results

# Or direct Python command with matching layer_type
python train.py \
    --phase 3 \
    --basis_dir ./checkpoints/phase1_TIMESTAMP/checkpoints/basis \
    --masks_dir ./checkpoints/phase2_TIMESTAMP/checkpoints/masks \
    --utility_samples 1000 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --layer_type ffn_down \
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
- `--layer_type`: **MUST match Phase 1 & 2 layer_type**

**Output**: Fine-tuned model saved to `checkpoints/phase3_TIMESTAMP/checkpoints/`