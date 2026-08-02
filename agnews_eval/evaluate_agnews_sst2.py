#!/usr/bin/env python3
"""Evaluate a merged causal LM on AGNews and/or SST-2.

Only the model path (or Hugging Face model ID) is required.  In ``auto`` mode,
the task is inferred from the model path.  If neither task name is present,
both datasets are evaluated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation_results" / "agnews_sst2"
TASKS = {
    "agnews": {
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "data": REPO_ROOT / "dataset/classification/agnews_test_1k_seed42.json",
    },
    "sst2": {
        "labels": ["negative", "positive"],
        "data": REPO_ROOT / "dataset/classification/sst2_validation_full.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one model on AGNews or SST-2 with its chat template."
    )
    parser.add_argument("model", help="Local merged-model directory or Hugging Face model ID")
    parser.add_argument(
        "--task",
        choices=["auto", "agnews", "sst2", "both"],
        default="auto",
        help="auto infers the task from the model path; unknown paths run both",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--agnews-data", type=Path, default=TASKS["agnews"]["data"])
    parser.add_argument("--sst2-data", type=Path, default=TASKS["sst2"]["data"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 evaluates every row; a positive value is useful for a smoke test",
    )
    return parser.parse_args()


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip())
    return slug.strip("._-") or "model"


def infer_tasks(model: str, requested: str) -> list[str]:
    if requested == "both":
        return ["agnews", "sst2"]
    if requested != "auto":
        return [requested]

    searchable = model.lower()
    local_model = Path(model).expanduser()
    if local_model.is_dir():
        for filename in (
            "wsrlora_run_config.json",
            "salora_run_config.json",
            "vanilla_lora_run_config.json",
        ):
            metadata_path = local_model / filename
            if metadata_path.is_file():
                searchable += " " + metadata_path.read_text(encoding="utf-8").lower()

    detected = [task for task in TASKS if task in searchable]
    if len(detected) == 1:
        return detected
    return ["agnews", "sst2"]


def normalize_prediction(text: str, labels: list[str]) -> str | None:
    """Return the first supported label mentioned in newly generated text."""
    lowered = text.strip().lower()
    aliases = {
        "sci/tech": (r"\bsci\s*/\s*tech\b", r"\bscience\s*/\s*technology\b"),
    }
    hits: list[tuple[int, str]] = []
    for label in labels:
        patterns = aliases.get(label.lower(), (rf"\b{re.escape(label.lower())}\b",))
        positions = [
            match.start()
            for pattern in patterns
            for match in re.finditer(pattern, lowered)
        ]
        if positions:
            hits.append((min(positions), label))
    return min(hits)[1] if hits else None


def macro_f1(golds: list[str], predictions: list[str | None], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, predictions))
        fp = sum(g != label and p == label for g, p in zip(golds, predictions))
        fn = sum(g == label and p != label for g, p in zip(golds, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / len(scores)


def load_rows(path: Path, max_samples: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation dataset does not exist: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"evaluation dataset must be a non-empty JSON list: {path}")
    required = {"question", "response"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise KeyError(f"{path}: row {index} is missing {sorted(missing)}")
    return rows[:max_samples] if max_samples > 0 else rows


def evaluate_task(
    task: str,
    data_path: Path,
    args: argparse.Namespace,
    model,
    tokenizer,
) -> dict:
    labels = TASKS[task]["labels"]
    rows = load_rows(data_path, args.max_samples)
    records = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(row["question"]).strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        prompt_length = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        for row, output in zip(batch, outputs):
            gold = str(row.get("label_text", row["response"])).strip()
            prediction = normalize_prediction(output, labels)
            records.append(
                {
                    "source_index": row.get("source_index"),
                    "gold": gold,
                    "prediction": prediction,
                    "generation": output.strip(),
                    "correct": prediction == gold,
                }
            )
        completed = min(start + len(batch), len(rows))
        print(f"[{task}] {completed}/{len(rows)}", flush=True)

    golds = [record["gold"] for record in records]
    predictions = [record["prediction"] for record in records]
    confusion = {
        gold: {prediction: 0 for prediction in labels + ["INVALID"]}
        for gold in labels
    }
    for gold, prediction in zip(golds, predictions):
        confusion[gold][prediction or "INVALID"] += 1

    correct = sum(record["correct"] for record in records)
    invalid = sum(prediction is None for prediction in predictions)
    return {
        "task": task,
        "model": args.model,
        "data_path": str(data_path.resolve()),
        "samples": len(records),
        "accuracy": correct / len(records),
        "macro_f1": macro_f1(golds, predictions, labels),
        "invalid_count": invalid,
        "invalid_rate": invalid / len(records),
        "gold_distribution": dict(Counter(golds)),
        "prediction_distribution": dict(
            Counter(prediction or "INVALID" for prediction in predictions)
        ),
        "confusion_matrix": confusion,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "prompt_format": "tokenizer_chat_template",
        },
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")

    tasks = infer_tasks(args.model, args.task)
    output_dir = args.output_root.expanduser().resolve() / model_slug(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {
        "agnews": args.agnews_data.expanduser().resolve(),
        "sst2": args.sst2_data.expanduser().resolve(),
    }
    print(f"[setup] model={args.model}", flush=True)
    print(f"[setup] tasks={tasks}", flush=True)
    print(f"[setup] output_dir={output_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise ValueError("the tokenizer has no chat template")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    ).eval()

    summaries = {}
    for task in tasks:
        result = evaluate_task(task, data_paths[task], args, model, tokenizer)
        (output_dir / f"{task}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[task] = {
            key: value for key, value in result.items() if key != "records"
        }
        print(
            f"[{task}] accuracy={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} "
            f"invalid={result['invalid_count']}",
            flush=True,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
