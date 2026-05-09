"""
Custom MBPP pass@1 evaluator using EXACT training prompt format.

Usage:
python mbpp_eval/eval_mbpp_custom.py \
    --model kmseong/llama2-7b-chat-mbpp-safedelta-scale0.3_2t \
    --device cuda: \
    --batch_size 32 \
    --output_dir mbpp_eval/results


  # compare multiple models
    python mbpp_eval/eval_mbpp_custom.py \
        --model kmseong/llama2-7b-chat-mbpp-safedelta-scale0.3_2t \
        --device cuda:6 \
        --batch_size 32 \
        --output_dir mbpp_eval/results
"""

import argparse
import json
import math
import os
import random
import re
import signal
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (must match finetune_mbpp_full_params.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
MBPP_INSTRUCTION = (
    "Write a Python function to solve the following programming problem. "
    "Provide only the complete, runnable Python code without any explanation.\n"
)

MBPP_DATASET    = "google-research-datasets/mbpp"
MBPP_SUBSET     = "full"
MBPP_TEST_SPLIT = "test"   # official MBPP test split (500 problems)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def is_chat_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(k in lower for k in ("chat", "instruct"))


def build_prompt(problem: str, test_list: list, tokenizer, is_chat: bool) -> str:
    """Build the exact same prompt used during fine-tuning."""
    tests_str = "\n".join(f"  {t}" for t in test_list[:3]) if test_list else ""
    if tests_str:
        problem_with_tests = f"{problem}\n\nYour code should pass the following tests:\n{tests_str}"
    else:
        problem_with_tests = problem

    if is_chat:
        user_content = f"{MBPP_INSTRUCTION}\n{problem_with_tests}"
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"### Instruction:\n{MBPP_INSTRUCTION}\n\n"
            f"### Input:\n{problem_with_tests}\n\n"
            f"### Response:\n"
        )
    return prompt


def extract_code(text: str) -> str:
    """Extract Python code from model output.
    
    Handles:
    - raw code (no markdown)
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    Trims leading/trailing whitespace.
    """
    # Try markdown code block first
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return raw text (trim leading spaces/newlines)
    return text.strip()


@contextmanager
def time_limit(seconds: int):
    """SIGALRM-based timeout for code execution."""
    def handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {seconds}s")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_tests(code: str, test_list: list, timeout: int = 5) -> bool:
    """Execute generated code + test assertions.
    Returns True if all tests pass, False otherwise."""
    try:
        with time_limit(timeout):
            exec_globals: dict = {}
            exec(compile(code, "<generated>", "exec"), exec_globals)
            for test in test_list:
                exec(compile(test, "<test>", "exec"), exec_globals)
        return True
    except TimeoutError:
        return False
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    examples: list,
    device: str,
    max_new_tokens: int,
    batch_size: int,
    output_dir: Path,
) -> dict:
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    # Load model & tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    is_chat = is_chat_model(model_name)
    print(f"Chat model: {is_chat}")

    # Build prompts
    prompts = [
        build_prompt(ex["text"], ex["test_list"], tokenizer, is_chat)
        for ex in examples
    ]

    # Generate in batches
    all_outputs = []
    n = len(prompts)
    for start in range(0, n, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        print(f"  Generating [{start+1}–{min(start+batch_size, n)}/{n}]...", end="\r")

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3584,   # leave room for generation (4096 - 512)
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        input_len = enc["input_ids"].shape[1]
        for ids in out_ids:
            new_ids = ids[input_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            all_outputs.append(text)

    print()  # newline after \r progress

    # Evaluate pass@1
    records = []
    passed = 0
    for ex, output in zip(examples, all_outputs):
        code = extract_code(output)
        ok = run_tests(code, ex["test_list"])
        passed += int(ok)
        records.append({
            "task_id": ex.get("task_id"),
            "problem": ex["text"],
            "test_list": ex["test_list"],
            "generated_raw": output,
            "generated_code": code,
            "passed": ok,
        })

    pass_at_1 = passed / n
    print(f"  pass@1 = {pass_at_1:.4f}  ({passed}/{n})")

    # Save results
    safe_name = model_name.replace("/", "__")
    out_path = output_dir / f"{safe_name}_mbpp_custom.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved → {out_path}")

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()

    return {"model": model_name, "pass_at_1": pass_at_1, "passed": passed, "total": n}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Custom MBPP pass@1 evaluator (training-format-aligned)"
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Model name(s) on HuggingFace Hub or local path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="mbpp_eval/results",
        help="Directory to save per-model JSONL result files"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:6",
        help="PyTorch device (e.g. cuda:6, cuda:0, cpu)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens to generate per problem"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help="Number of test examples to evaluate (0 = all ~500)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    fix_seed(args.seed)
    print(f"Seed fixed: {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MBPP test split
    print(f"Loading MBPP test split from {MBPP_DATASET} ...")
    ds = load_dataset(MBPP_DATASET, MBPP_SUBSET, split=MBPP_TEST_SPLIT, cache_dir=args.cache_dir)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    examples = [dict(ex) for ex in ds]
    print(f"Total test examples: {len(examples)}")

    # Evaluate each model
    summary = []
    for model_name in args.model:
        result = evaluate_model(
            model_name=model_name,
            examples=examples,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
        summary.append(result)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<55} {'pass@1':>8}  {'passed/total'}")
    print("-" * 70)
    for r in summary:
        print(f"{r['model']:<55} {r['pass_at_1']:>8.4f}  {r['passed']}/{r['total']}")
    print()

    # Save summary JSON
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    summary_path = output_dir / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
