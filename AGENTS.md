# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research codebase for safety-preserving LLM fine-tuning. `train.py` is the main dispatcher for the numbered WaRP phases; phase implementations and alternative methods live in `models/`. Shared data loading is under `data/`, while `scripts/` contains integrated pipelines, ablations, and evaluation helpers. Task-specific experiments are grouped in directories such as `gsm8k_eval/`, `mmlu_eval/`, `medqa_eval/`, and `agnews_eval/`. The standalone SN-Tune baseline is in `sn_tune/`, and the SEAL integration is documented in `seal/README.md`. Unit tests live in `tests/`; plots, logs, and generated checkpoints are experiment artifacts, not core modules.

## Build, Test, and Development Commands

- `conda env create -f environment.yml && conda activate safety-warp` creates the Python 3.11/CUDA environment.
- `pip install -r requirements.txt` installs dependencies into an existing environment.
- `pytest -q tests/test_lora_wsr.py` runs the CPU-only WSR-LoRA mathematical tests.
- `python train.py --phase 1 --phase0_model_dir <model> --layer_type ffn_down --target_layers all --device cuda --dtype bfloat16` builds a WaRP basis.
- `bash scripts/run_all_phases_integrated.sh` runs the primary multi-phase experiment; review its model paths, conda environment, GPU selection, and hyperparameters first.

Most training commands require a CUDA GPU, Hugging Face access, and substantial storage.

## Coding Style & Naming Conventions

Use four-space indentation and conventional PEP 8 naming: `snake_case` for files, functions, and variables; `PascalCase` for classes; uppercase names for constants. Keep tensor shapes and coordinate spaces explicit in comments and docstrings. Format Python with `black` and check it with `flake8` (both are listed in `environment.yml`). Preserve the invariant that `layer_type` and `target_layers` match across Phases 1–3.

## Testing Guidelines

Tests use `pytest` and should be named `tests/test_<feature>.py`, with functions named `test_<behavior>`. Prefer small deterministic CPU tests; seed randomness and use explicit numerical tolerances. Add regression coverage for changes to masking, basis rotation, folding, or trainable-parameter selection.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, subsystem-focused subjects (for example, `seal update` and `wsr-lora stage1`). Keep each commit narrowly scoped and avoid committing checkpoints, logs, credentials, or machine-specific paths. Pull requests should explain the experiment or behavior changed, list commands run, identify model/data assumptions, and report relevant metrics. Link issues when applicable; include plots or screenshots only when results or visualizations change.
