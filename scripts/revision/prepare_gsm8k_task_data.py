#!/usr/bin/env python3
"""GSM8K train split → 로컬 태스크 JSON([{"question","response"}]).

왜 필요한가
-----------
revision 실험은 12개 기법을 같은 셀에서 비교한다. 기법마다 러너가 다르고, 러너마다
GSM8K 를 각자 `load_dataset("openai/gsm8k", "main", split="train")` 으로 읽으면
"같은 데이터를 읽었다"는 것이 코드 리뷰로만 보장된다. 태스크 JSON 하나로 고정해
두면 **모든 arm 이 물리적으로 같은 파일**을 읽으므로 그 보장이 필요 없어진다.
(math/arc/medqa/agnews 는 이미 이 형태로 만들어져 있다.)

프롬프트/정답은 기존 GSM8K 하네스와 **완전히 동일**하다:
  gsm8k_eval/finetune_gsm8k_full_params.py:597-598
      question = ex["question"];  answer = ex["answer"]
즉 가공 없이 원본 필드를 그대로 옮긴다. 행 순서도 HF split 순서를 유지한다
(SEAL selector 인덱스가 행 순서에 묶여 있으므로 섞으면 안 된다).

사용:
  python scripts/revision/prepare_gsm8k_task_data.py
  python scripts/revision/prepare_gsm8k_task_data.py --output data/gsm8k_train_task_7473.json
"""
import argparse
import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="openai/gsm8k")
    p.add_argument("--dataset_subset", default="main")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default=os.path.join(REPO_DIR, "data", "gsm8k_train_task_7473.json"))
    p.add_argument("--cache_dir", default=os.path.join(REPO_DIR, "cache"))
    p.add_argument("--max_samples", type=int, default=0, help="0 = 전체")
    p.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.force:
        with open(args.output, encoding="utf-8") as f:
            rows = json.load(f)
        print(f"[gsm8k] 이미 존재 — skip ({args.output}, {len(rows)} rows). "
              f"다시 만들려면 --force")
        return

    from datasets import load_dataset

    ds = load_dataset(args.dataset_name, args.dataset_subset,
                      split=args.split, cache_dir=args.cache_dir)
    print(f"[gsm8k] loaded {len(ds)} rows from {args.dataset_name}/{args.dataset_subset}:{args.split}")

    rows = []
    for ex in ds:
        # 원본 필드를 가공 없이 그대로. strip 도 하지 않는다 — 러너 쪽
        # tokenize 함수가 동일하게 strip 하므로 여기서 손대면 오히려 갈라진다.
        q = ex["question"]
        a = ex["answer"]
        if not q or not a:
            raise ValueError(f"빈 행 발견: {ex}")
        rows.append({"question": q, "response": a})

    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"[gsm8k] wrote {len(rows)} rows → {args.output}")

    # 왕복 검증: 러너들이 실제로 쓰는 로더로 다시 읽어 첫 행이 원본과 같은지 본다.
    from data.local_task_dataset import load_task_pairs
    pairs = load_task_pairs(args.output, 1)
    assert pairs[0][0] == ds[0]["question"].strip(), "question 불일치"
    assert pairs[0][1] == ds[0]["answer"].strip(), "response 불일치"
    print("[gsm8k] round-trip 검증 OK (data.local_task_dataset.load_task_pairs)")


if __name__ == "__main__":
    main()
