"""
MedQA (USMLE) 평가 스크립트.

정답 추출: 모델 출력에서 가장 먼저 나오는 A/B/C/D 옵션 레터를 추출하여
           JSONL 의 correct_option 과 비교 → Accuracy 계산.

Example usage:

python medqa_eval/evaluate_medqa.py \
    --model_name_or_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --test_jsonl_path /home/yonsei_jong/Safety-WaRP-LLM/data/medqa_test_1273.jsonl \
    --output_path ./medqa_eval/predictions/llama2_7b_chat_ssft_medqa.jsonl

python medqa_eval/evaluate_medqa.py \
    --model_name_or_path ./medqa_eval/llama2_7b_chat_SSFT_medqa_FT_lr3e-5 \
    --test_jsonl_path /home/yonsei_jong/Safety-WaRP-LLM/data/medqa_test_1273.jsonl \
    --output_path ./medqa_eval/predictions/llama2_7b_chat_ft_medqa.jsonl

dev set:
python medqa_eval/evaluate_medqa.py \
    --model_name_or_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --test_jsonl_path /home/yonsei_jong/Safety-WaRP-LLM/data/medqa_dev_1272.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_OPTIONS = {"A", "B", "C", "D"}
FORMAT_INSTRUCTION = (
    "Reply with only the option letter (A, B, C, or D) followed by a period and the answer text. "
    "Example: A. Aspirin"
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a causal LM on MedQA JSONL.")

    p.add_argument("--model_name_or_path", required=True,
                   help="HF model ID 또는 로컬 경로")
    p.add_argument("--test_jsonl_path", type=Path, required=True,
                   help="평가용 MedQA JSONL 경로 (prepare_medqa_dataset.py 출력)")
    p.add_argument("--output_path", type=Path, default=None,
                   help="예측 결과 저장 JSONL 경로 (생략 시 저장 안 함)")

    p.add_argument("--prompt_format", choices=["auto", "base", "chat"], default="auto",
                   help="auto: 모델명으로 자동 판별")
    p.add_argument("--no_format_instruction", action="store_true",
                   help="출력 포맷 안내 문구를 프롬프트에 추가하지 않음")

    p.add_argument("--adapter_path", type=str, default=None,
                   help="PEFT/LoRA 어댑터 경로 (베이스 모델에 추가로 로드)")

    p.add_argument("--max_samples",     type=int,   default=0,
                   help="평가 샘플 수 제한 (0=전체)")
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--max_input_length",type=int,   default=768,
                   help="입력 프롬프트 최대 토큰 수")
    p.add_argument("--max_new_tokens",  type=int,   default=32,
                   help="생성 토큰 수 (정답이 짧으므로 32 충분)")
    p.add_argument("--cache_dir",       type=str,   default=None)
    p.add_argument("--device_map",      type=str,   default="auto")
    p.add_argument("--torch_dtype",
                   choices=["auto", "bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code",
                   action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_dtype(name: str):
    return {
        "auto":    "auto",
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[name]


def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


def resolve_prompt_format(model_ref: str, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    return "chat" if is_instruct_model(model_ref) else "base"


def load_jsonl(path: Path, max_samples: int = 0) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def chunks(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i: i + n]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────────────────────

def _append_format_instruction(text: str, instruction: str) -> str:
    """### Response: 마커 앞에 포맷 안내를 삽입하거나, 끝에 추가."""
    if not instruction:
        return text
    marker = "### Response:\n"
    if marker in text:
        return text.replace(
            marker,
            f"### Output Format:\n{instruction}\n\n{marker}",
            1,
        )
    return text.rstrip() + f"\n\n{instruction}\n\n### Response:\n"


def make_base_prompt(row: Dict, fmt_instruction: str) -> str:
    if row.get("prompt"):
        return _append_format_instruction(str(row["prompt"]), fmt_instruction)
    instruction = row.get("instruction", "")
    input_text  = row.get("input", "")
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    return _append_format_instruction(prompt, fmt_instruction)


def make_chat_prompt(tokenizer, row: Dict, model_ref: str, fmt_instruction: str) -> str:
    instruction = row.get("instruction", "")
    input_text  = row.get("input", "")
    user_content = f"{instruction}\n\n{input_text}".strip()
    if fmt_instruction:
        user_content = user_content + "\n\n" + fmt_instruction

    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Llama-2 chat fallback
        if "llama-2" in str(model_ref).lower():
            return f"<s>[INST] {user_content} [/INST]"
        raise


def build_prompts(tokenizer, rows: List[Dict], prompt_format: str,
                  model_ref: str, fmt_instruction: str) -> List[str]:
    if prompt_format == "chat":
        return [make_chat_prompt(tokenizer, r, model_ref, fmt_instruction) for r in rows]
    return [make_base_prompt(r, fmt_instruction) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_option(text: str) -> Optional[str]:
    """
    생성된 텍스트에서 옵션 레터(A/B/C/D)를 추출.
    우선순위:
    1. 텍스트 앞쪽에서 'A.' / 'A)' / 'A ' 형태
    2. 앞쪽 단어가 단독 A/B/C/D
    3. 텍스트 내 첫 번째로 등장하는 A/B/C/D
    """
    text = text.strip()
    if not text:
        return None

    # 1. "A." / "A)" / "(A)" 패턴 — 텍스트 앞부분
    m = re.search(r"(?:^|\()\s*([A-D])\s*[.)]\s", text)
    if m:
        return m.group(1).upper()

    # 2. 첫 단어가 단독 옵션 레터
    first_word = re.split(r"[\s.,;:\n]+", text, maxsplit=1)[0].strip().upper()
    if first_word in VALID_OPTIONS:
        return first_word

    # 3. 텍스트 내 첫 번째 A~D (단어 경계)
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()

    return None


def prediction_is_correct(pred: Optional[str], gold: str) -> bool:
    return pred is not None and pred.upper() == gold.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.test_jsonl_path.exists():
        print(f"ERROR: test JSONL not found: {args.test_jsonl_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(args.test_jsonl_path, args.max_samples)
    prompt_format = resolve_prompt_format(args.model_name_or_path, args.prompt_format)
    fmt_instruction = "" if args.no_format_instruction else FORMAT_INSTRUCTION

    print(f"model          : {args.model_name_or_path}")
    print(f"test_jsonl     : {args.test_jsonl_path}")
    print(f"prompt_format  : {prompt_format}")
    print(f"samples        : {len(rows)}")
    print(f"batch_size     : {args.batch_size}")
    print(f"max_new_tokens : {args.max_new_tokens}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
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

    # ── Model ──────────────────────────────────────────────────────────────────
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
        print(f"adapter_path   : {args.adapter_path}")
    model.eval()

    # ── Inference ──────────────────────────────────────────────────────────────
    results = []
    correct = 0
    total   = 0
    # 옵션별 정답 추적
    per_option_correct = {opt: 0 for opt in VALID_OPTIONS}
    per_option_total   = {opt: 0 for opt in VALID_OPTIONS}

    for batch_rows in tqdm(list(chunks(rows, args.batch_size)), desc="Evaluating"):
        prompts = build_prompts(
            tokenizer, batch_rows, prompt_format, args.model_name_or_path, fmt_instruction
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

        for row, out_ids in zip(batch_rows, generated):
            new_tokens  = out_ids[input_width:]
            raw_gen     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred_option = extract_option(raw_gen)
            gold_option = str(row.get("correct_option", "")).strip().upper()
            is_correct  = prediction_is_correct(pred_option, gold_option)

            correct += int(is_correct)
            total   += 1
            if gold_option in VALID_OPTIONS:
                per_option_total[gold_option] += 1
                if is_correct:
                    per_option_correct[gold_option] += 1

            result = dict(row)
            result["raw_generation"] = raw_gen
            result["pred_option"]    = pred_option
            result["gold_option"]    = gold_option
            result["correct"]        = is_correct
            results.append(result)

    # ── Results ────────────────────────────────────────────────────────────────
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 50)
    print(f"model          : {args.model_name_or_path}")
    print(f"test_jsonl     : {args.test_jsonl_path}")
    print(f"prompt_format  : {prompt_format}")
    print(f"accuracy       : {accuracy:.4f}  ({correct}/{total})")
    print("per-option accuracy:")
    for opt in sorted(VALID_OPTIONS):
        n = per_option_total[opt]
        c = per_option_correct[opt]
        pct = c / n if n > 0 else 0.0
        print(f"  {opt}: {pct:.4f} ({c}/{n})")
    print("=" * 50)

    # 잘못 예측된 샘플 상위 5개 출력
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\nWrong predictions sample (first 3 / {len(wrong)} total):")
        for r in wrong[:3]:
            print(f"  gold={r['gold_option']} pred={r['pred_option']}  raw='{r['raw_generation'][:60]}'")

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 요약 JSON 저장
        summary_path = args.output_path.with_suffix(".summary.json")
        summary = {
            "model":         args.model_name_or_path,
            "test_jsonl":    str(args.test_jsonl_path),
            "prompt_format": prompt_format,
            "total":         total,
            "correct":       correct,
            "accuracy":      round(accuracy, 6),
            "per_option": {
                opt: {
                    "correct": per_option_correct[opt],
                    "total":   per_option_total[opt],
                    "accuracy": round(per_option_correct[opt] / per_option_total[opt], 6)
                             if per_option_total[opt] > 0 else 0.0,
                }
                for opt in sorted(VALID_OPTIONS)
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nsaved predictions : {args.output_path}")
        print(f"saved summary     : {summary_path}")


if __name__ == "__main__":
    main()
