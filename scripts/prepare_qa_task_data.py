"""ARC-Challenge / MedQA → 로컬 태스크 JSON({"question","response"}) 변환.

`data/local_task_dataset.py` 가 읽는 포맷으로 떨어뜨려서 LISA / SafeLoRA / AsFT 러너가
sst2·agnews 와 완전히 동일한 경로(`--task_data_path`)로 학습할 수 있게 한다.

프롬프트/정답 포맷은 각 태스크의 기존 full-FT 하네스에서 **함수를 직접 import** 해서 쓴다.
따라서 여기서 만든 데이터로 학습한 모델은 기존 평가 스크립트와 포맷이 일치한다.
  - ARC-C : arc_eval/finetune_arc_full_params.format_arc_question
            정답 = "The best answer is {A|B|C|D}"
  - MedQA : medqa_eval/finetune_medqa_full_params.medqa_prompt_response(prefer_chat=True)
            (출발 모델이 llama2-*-chat 이므로 chat 경로 = instruction + "\n\n" + input)

사용:
  python scripts/prepare_qa_task_data.py                    # 둘 다
  python scripts/prepare_qa_task_data.py --tasks arc        # arc 만
"""
import argparse
import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="arc,medqa", help="arc,medqa 중 콤마 구분")
    p.add_argument("--output_dir", default=os.path.join(REPO_DIR, "data"))
    p.add_argument("--cache_dir", default=os.path.join(REPO_DIR, "cache"))
    # ARC
    p.add_argument("--arc_dataset", default="allenai/ai2_arc")
    p.add_argument("--arc_subset", default="ARC-Challenge")
    p.add_argument("--arc_split", default="train")
    # MedQA (prepare_medqa_dataset.py 출력 JSONL)
    p.add_argument("--medqa_jsonl",
                   default=os.path.join(REPO_DIR, "data", "medqa_train_10178.jsonl"))
    p.add_argument("--max_samples", type=int, default=0, help="0=전체")
    return p.parse_args()


def build_arc(args):
    """ARC-Challenge → [(question, response)]. arc_eval 하네스와 동일 포맷."""
    from datasets import load_dataset
    from arc_eval.finetune_arc_full_params import (
        ARC_GEN_PREFIX, format_arc_question, get_arc_answer_letter,
    )

    ds = load_dataset(args.arc_dataset, args.arc_subset,
                      split=args.arc_split, cache_dir=args.cache_dir)
    rows = []
    skipped = 0
    for ex in ds:
        try:
            q = format_arc_question(ex["question"], ex["choices"])
            a = f"{ARC_GEN_PREFIX} {get_arc_answer_letter(ex['answerKey'])}"
        except Exception:
            skipped += 1
            continue
        rows.append({"question": q, "response": a,
                     "id": ex.get("id", ""), "answer_key": ex.get("answerKey", "")})
    if skipped:
        print(f"  [arc] {skipped} rows skipped")
    return rows, f"arc_challenge_train_task_{len(rows)}.json"


def build_medqa(args):
    """MedQA JSONL → [(question, response)]. medqa_eval 하네스의 chat 경로와 동일 포맷."""
    from medqa_eval.finetune_medqa_full_params import medqa_prompt_response

    if not os.path.exists(args.medqa_jsonl):
        raise FileNotFoundError(
            f"{args.medqa_jsonl} 없음 — 먼저 "
            f"`python medqa_eval/prepare_medqa_dataset.py --output_dir ./data` 를 실행하세요")

    rows = []
    with open(args.medqa_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            # 출발 모델이 llama2-7b-chat 계열 → prefer_chat=True 경로가 실제 학습 포맷이다.
            q, a = medqa_prompt_response(raw, prefer_chat=True)
            rows.append({"question": q, "response": a,
                         "id": raw.get("id", ""),
                         "correct_option": raw.get("correct_option", "")})
    return rows, f"medqa_train_task_{len(rows)}.json"


BUILDERS = {"arc": build_arc, "medqa": build_medqa}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task not in BUILDERS:
            raise ValueError(f"알 수 없는 task: {task} (arc|medqa)")
        print(f"[{task}] 생성 중 ...")
        rows, fname = BUILDERS[task](args)
        if args.max_samples > 0:
            rows = rows[:args.max_samples]
            fname = fname.replace(".json", f"_first{args.max_samples}.json")
        out = os.path.join(args.output_dir, fname)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"  {len(rows):,} rows → {out}")
        print(f"  sample question: {rows[0]['question'][:160]!r}")
        print(f"  sample response: {rows[0]['response'][:80]!r}")


if __name__ == "__main__":
    main()
