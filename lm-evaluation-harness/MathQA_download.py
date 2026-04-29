import os
import json
from datasets import load_dataset


OUTPUT_DIR = "./mathqa_sft_data"
TRAIN_OUT = os.path.join(OUTPUT_DIR, "train.jsonl")
VALID_OUT = os.path.join(OUTPUT_DIR, "validation.jsonl")
TEST_OUT = os.path.join(OUTPUT_DIR, "test.jsonl")


def build_messages(example):
    """
    MathQA sample -> chat-format SFT sample
    입력: 문제 + 보기
    출력: 간단한 풀이 + 정답 옵션 문자
    """

    problem = example["Problem"].strip()
    options = example["options"].strip()
    rationale = example["Rationale"].strip()
    correct = example["correct"].strip()

    user_content = (
        "Solve the following multiple-choice math problem.\n\n"
        f"Question: {problem}\n"
        f"Options: {options}\n\n"
        "First explain the reasoning briefly, then give the final answer as a single option letter."
    )

    assistant_content = (
        f"Reasoning: {rationale}\n"
        f"Final Answer: {correct}"
    )

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def save_jsonl(dataset_split, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for row in dataset_split:
            sample = build_messages(row)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # MathQA 다운로드
    dataset = load_dataset("allenai/math_qa")

    print(dataset)
    print("Train size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))
    print("Test size:", len(dataset["test"]))

    save_jsonl(dataset["train"], TRAIN_OUT)
    save_jsonl(dataset["validation"], VALID_OUT)
    save_jsonl(dataset["test"], TEST_OUT)

    print(f"Saved: {TRAIN_OUT}")
    print(f"Saved: {VALID_OUT}")
    print(f"Saved: {TEST_OUT}")


if __name__ == "__main__":
    main()