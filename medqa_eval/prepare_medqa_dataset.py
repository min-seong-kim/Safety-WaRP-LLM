"""
MedQA 데이터셋 다운로드 및 JSONL 변환 스크립트

출처: openlifescienceai/medqa (USMLE 영어 문제, 동일 데이터)
     bigbio/med_qa 는 custom loading script 방식으로 현재 datasets 라이브러리에서
     차단되어 있으므로, Parquet 포맷으로 변환된 동일 데이터를 사용.

포맷:
  - instruction: 문제 풀이 지시문
  - input: 문제 + 보기 A~D
  - output: "A. {정답 텍스트}" 형식
  - prompt: instruction + input (### 포맷)
  - completion: output

Usage:
  python prepare_medqa_dataset.py [--output_dir PATH] [--train_samples N]
                                  [--test_samples N] [--cache_dir PATH]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

MEDQA_INSTRUCTION = (
    "Answer the following multiple-choice medical question by selecting the single best answer. "
    "Reply with only the option letter (A, B, C, or D) followed by a period and the answer text.\n"
    "Example: A. Aspirin"
)


def parse_args():
    p = argparse.ArgumentParser(description="Download and convert MedQA dataset to JSONL")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/yonsei_jong/Safety-WaRP-LLM/data",
        help="JSONL 저장 디렉토리",
    )
    p.add_argument(
        "--train_samples",
        type=int,
        default=0,
        help="train split 저장 개수 (0=전체, 전체=10178)",
    )
    p.add_argument(
        "--test_samples",
        type=int,
        default=0,
        help="test split 저장 개수 (0=전체, 전체=1273)",
    )
    p.add_argument(
        "--dev_samples",
        type=int,
        default=0,
        help="dev split 저장 개수 (0=전체, 전체=1272)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default="./cache")
    return p.parse_args()


def build_input_text(question: str, options: dict) -> str:
    """질문 + 보기 A~D를 하나의 텍스트로 조합"""
    option_lines = "\n".join(
        f"{letter}. {text}"
        for letter, text in sorted(options.items())
    )
    return f"{question.strip()}\n\n{option_lines}"


def build_output_text(correct_option: str, correct_answer: str) -> str:
    """정답 포맷: 'A. Nitrofurantoin'"""
    return f"{correct_option.strip()}. {correct_answer.strip()}"


def row_to_record(row: dict, idx: int, split: str) -> dict:
    """HF row → agnews JSONL 포맷과 동일한 구조로 변환"""
    data = row["data"]
    question = data["Question"]
    options = data["Options"]           # {'A': ..., 'B': ..., 'C': ..., 'D': ...}
    correct_option = data["Correct Option"]
    correct_answer = data["Correct Answer"]

    input_text = build_input_text(question, options)
    output_text = build_output_text(correct_option, correct_answer)
    prompt = (
        f"### Instruction:\n{MEDQA_INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    return {
        "id": row.get("id") or f"medqa_{split}_{idx}",
        "source": "openlifescienceai/medqa",
        "subject_name": row.get("subject_name", ""),
        "instruction": MEDQA_INSTRUCTION,
        "input": input_text,
        "output": output_text,
        "correct_option": correct_option,
        "correct_answer": correct_answer,
        "options": options,
        "prompt": prompt,
        "completion": output_text,
    }


def save_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets 라이브러리가 없습니다. 'pip install datasets' 를 실행하세요.")
        sys.exit(1)

    print("=" * 60)
    print("MedQA 데이터셋 다운로드 및 변환")
    print("  Source : openlifescienceai/medqa  (USMLE 영어)")
    print(f"  Output : {args.output_dir}")
    print("=" * 60)

    print("\n[1/2] 데이터셋 로딩 중...")
    ds = load_dataset("openlifescienceai/medqa", cache_dir=args.cache_dir)
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} examples")

    rng = random.Random(args.seed)

    print("\n[2/2] JSONL 변환 및 저장 중...")

    split_config = {
        "train": args.train_samples,
        "test":  args.test_samples,
        "dev":   args.dev_samples,
    }

    output_paths = {}
    for split, max_n in split_config.items():
        if split not in ds:
            print(f"  [{split}] split 없음, 건너뜀")
            continue

        rows = list(ds[split])
        total = len(rows)

        if max_n > 0 and total > max_n:
            rows = rng.sample(rows, max_n)
            print(f"  [{split}] {total:,} → {max_n:,} 샘플 랜덤 추출")
        else:
            print(f"  [{split}] 전체 {total:,} 샘플 사용")

        records = [row_to_record(row, idx, split) for idx, row in enumerate(rows)]

        n = len(records)
        fname = f"medqa_{split}_{n}.jsonl"
        out_path = os.path.join(args.output_dir, fname)
        save_jsonl(records, out_path)
        output_paths[split] = out_path

    print("\n완료!")
    print("\n저장된 파일:")
    for split, path in output_paths.items():
        print(f"  [{split}] {path}")

    print("\n샘플 레코드 (train 첫 번째 항목):")
    if "train" in output_paths:
        with open(output_paths["train"], "r") as f:
            sample = json.loads(f.readline())
        print(f"  id           : {sample['id']}")
        print(f"  instruction  : {sample['instruction'][:60]}...")
        print(f"  input (80c)  : {sample['input'][:80]}...")
        print(f"  output       : {sample['output']}")
        print(f"  correct_opt  : {sample['correct_option']}")

    print("\n사용 예시 (finetune 스크립트):")
    train_path = output_paths.get("train", "<train_path>")
    test_path = output_paths.get("test", output_paths.get("dev", "<test_path>"))
    print(f"  python finetune_medqa_full_params.py \\")
    print(f"      --model_path <MODEL_ID> \\")
    print(f"      --medqa_train_path {train_path} \\")
    print(f"      --medqa_eval_path  {test_path} \\")
    print(f"      --output_dir ./output_medqa_finetune")


if __name__ == "__main__":
    main()
