import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent
BASE_TEST_FILE = DATA_DIR / "agnews_qwen2_5_7b_test_200.jsonl"
CHAT_TEST_FILE = DATA_DIR / "agnews_qwen2_5_7b_instruct_test_200.jsonl"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
DEFAULT_FORMAT_INSTRUCTION = (
    "Output format: respond with exactly one label from "
    "World, Sports, Business, Sci/Tech. Do not explain."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Hugging Face Qwen/Llama causal LM models on AG News JSONL."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--prompt_format",
        choices=["auto", "base", "chat"],
        default="auto",
        help="base uses prompt/completion JSONL; chat uses messages JSONL.",
    )
    parser.add_argument(
        "--test_jsonl_path",
        type=Path,
        default=None,
        help="Override the default test JSONL path.",
    )
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional PEFT/LoRA adapter path to load on top of model_name_or_path.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument(
        "--format_instruction",
        default=DEFAULT_FORMAT_INSTRUCTION,
        help="Instruction appended to the prompt to control output format.",
    )
    parser.add_argument(
        "--no_format_instruction",
        action="store_true",
        help="Do not append an output-format instruction.",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def resolve_prompt_format(model_name_or_path: str, prompt_format: str) -> str:
    if prompt_format != "auto":
        return prompt_format
    model_ref = str(model_name_or_path).lower()
    if "instruct" in model_ref or "chat" in model_ref:
        return "chat"
    return "base"


def resolve_test_path(prompt_format: str, test_jsonl_path: Optional[Path]) -> Path:
    if test_jsonl_path is not None:
        return test_jsonl_path
    return CHAT_TEST_FILE if prompt_format == "chat" else BASE_TEST_FILE


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_jsonl(path: Path, max_samples: int = 0) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def chunks(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def apply_base_format_instruction(prompt: str, format_instruction: str) -> str:
    if not format_instruction:
        return prompt
    marker = "### Response:\n"
    if marker in prompt:
        return prompt.replace(
            marker,
            f"### Output Format:\n{format_instruction}\n\n{marker}",
            1,
        )
    return prompt.rstrip() + f"\n\n### Output Format:\n{format_instruction}\n\n### Response:\n"


def make_base_prompt(row: Dict, format_instruction: str = "") -> str:
    if row.get("prompt"):
        return apply_base_format_instruction(row["prompt"], format_instruction)
    prompt = (
        "### Instruction:\n"
        f"{row['instruction']}\n"
        "### Input:\n"
        f"{row['input']}\n\n"
        "### Response:\n"
    )
    return apply_base_format_instruction(prompt, format_instruction)


def append_chat_format_instruction(messages: List[Dict[str, str]], format_instruction: str) -> List[Dict[str, str]]:
    if not format_instruction:
        return messages
    formatted = [dict(message) for message in messages]
    for idx in range(len(formatted) - 1, -1, -1):
        if formatted[idx].get("role") == "user":
            formatted[idx]["content"] = formatted[idx]["content"].rstrip() + "\n\n" + format_instruction
            break
    return formatted


def make_chat_messages(row: Dict, format_instruction: str = "") -> List[Dict[str, str]]:
    if row.get("messages"):
        return append_chat_format_instruction(row["messages"][:-1], format_instruction)
    messages = [
        {
            "role": "user",
            "content": f"{row['instruction']}\nNews article:\n{row['input']}",
        }
    ]
    return append_chat_format_instruction(messages, format_instruction)


def llama2_chat_fallback(messages: List[Dict[str, str]]) -> str:
    user_content = messages[0]["content"].strip()
    return f"<s>[INST] {user_content} [/INST]"


def make_chat_prompt(tokenizer, row: Dict, model_name_or_path: str, format_instruction: str) -> str:
    messages = make_chat_messages(row, format_instruction)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        if "llama-2" in str(model_name_or_path).lower():
            return llama2_chat_fallback(messages)
        raise


def build_prompts(
    tokenizer,
    rows: List[Dict],
    prompt_format: str,
    model_name_or_path: str,
    format_instruction: str,
):
    if prompt_format == "chat":
        return [
            make_chat_prompt(tokenizer, row, model_name_or_path, format_instruction)
            for row in rows
        ]
    return [make_base_prompt(row, format_instruction) for row in rows]


def extract_label(text: str) -> Optional[str]:
    normalized = text.strip()
    patterns = [
        ("Sci/Tech", r"\b(?:sci\s*/\s*tech|sci-tech|science\s*(?:and|&|/)\s*technology)\b"),
        ("Business", r"\bbusiness\b"),
        ("Sports", r"\bsports?\b"),
        ("World", r"\bworld\b"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return label
    first_token = re.split(r"[\s,.;:\n]+", normalized, maxsplit=1)[0].strip()
    for label in LABELS:
        if first_token.lower() == label.lower():
            return label
    return None


def prediction_is_correct(prediction: Optional[str], gold: str) -> bool:
    return prediction is not None and prediction.lower() == gold.lower()


def main():
    args = parse_args()
    prompt_format = resolve_prompt_format(args.model_name_or_path, args.prompt_format)
    test_path = resolve_test_path(prompt_format, args.test_jsonl_path)
    rows = load_jsonl(test_path, args.max_samples)
    format_instruction = "" if args.no_format_instruction else args.format_instruction

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(args.torch_dtype),
        device_map=args.device_map,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    results = []
    correct = 0
    total = 0

    for batch_rows in tqdm(list(chunks(rows, args.batch_size)), desc="Evaluating"):
        prompts = build_prompts(
            tokenizer,
            batch_rows,
            prompt_format,
            args.model_name_or_path,
            format_instruction,
        )
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        input_width = encoded["input_ids"].shape[1]

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for row, output_ids in zip(batch_rows, generated):
            new_tokens = output_ids[input_width:]
            raw_generation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            prediction = extract_label(raw_generation)
            gold = row["label_text"]
            is_correct = prediction_is_correct(prediction, gold)
            correct += int(is_correct)
            total += 1

            result = dict(row)
            result["raw_generation"] = raw_generation
            result["prediction"] = prediction
            result["correct"] = is_correct
            results.append(result)

    accuracy = correct / total if total else 0.0
    print(f"model_name_or_path: {args.model_name_or_path}")
    if args.adapter_path:
        print(f"adapter_path: {args.adapter_path}")
    print(f"prompt_format: {prompt_format}")
    print(f"format_instruction: {format_instruction}")
    print(f"test_jsonl_path: {test_path}")
    print(f"accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"saved_predictions: {args.output_path}")


if __name__ == "__main__":
    main()
