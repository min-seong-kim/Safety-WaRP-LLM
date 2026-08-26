#!/usr/bin/env python
"""
Hendrycks MATH → `data/math_train_task_<N>.json`  ([{question, response}, ...])

왜 필요한가
-----------
WaRP Phase 3 는 `--phase3_dataset math` 로 MATH 를 직접 로드하지만,
baseline 러너들(LISA / SafeLoRA / AsFT / SaLoRA / SEAL)은 gsm8k 아니면
`--task_data_path` 로 넘어온 로컬 JSON 만 읽는다. 이 스크립트가 그 JSON 을 만든다.

전처리는 `data/math_task_format.py` 를 그대로 쓰고, 그 모듈은
`models/phase3_extra_learning.py::_load_hendrycks_math` 도 import 한다.
→ WaRP arm 과 baseline arm 이 **바이트 단위로 같은 학습 텍스트**를 본다.
   (arc/medqa 가 scripts/prepare_qa_task_data.py 에서 eval 하네스의 프롬프트
    빌더를 import 하는 것과 같은 장치다.)

사용:
    python scripts/prepare_math_task_data.py                       # 전체(7500), 기본 경로
    python scripts/prepare_math_task_data.py --max_samples 7500
    python scripts/prepare_math_task_data.py --subjects Algebra,Geometry --levels 1,2,3
    python scripts/prepare_math_task_data.py --dataset_source flat  # qwedsacf/competition_math

주의: `--seed` 는 Phase 3 의 shuffle seed(기본 42)와 맞춰야 두 arm 의 샘플 순서/부분집합이
      같아진다. 기본값이 이미 42 다.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.math_task_format import (  # noqa: E402
    SUBJECT_TO_CONFIG,
    VALID_LEVELS,
    build_target,
    normalize_csv_arg,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Hendrycks MATH → task JSON")
    ap.add_argument("--output", default=None,
                    help="출력 경로 (기본: data/math_train_task_<N>.json)")
    ap.add_argument("--subjects", default="all",
                    help=f"쉼표 구분. 가능: {', '.join(SUBJECT_TO_CONFIG)}")
    ap.add_argument("--levels", default="all",
                    help="쉼표 구분 (1~5 또는 'Level 3')")
    ap.add_argument("--max_samples", type=int, default=0, help="0=전체")
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed — Phase 3 의 --seed 와 같아야 한다")
    ap.add_argument("--dataset_source", default="official", choices=["official", "flat"],
                    help="official=EleutherAI/hendrycks_math(과목별 config), flat=qwedsacf/competition_math")
    ap.add_argument("--official_dataset_path", default="EleutherAI/hendrycks_math")
    ap.add_argument("--flat_dataset_path", default="qwedsacf/competition_math")
    ap.add_argument("--train_on_mixed_formats", action="store_true",
                    help="Phase 3 의 --math_train_on_mixed_formats 와 동일 (기본 off = long 포맷만)")
    ap.add_argument("--cache_dir", default=None)
    return ap.parse_args()


def resolve_subjects(subjects_arg: str):
    subjects_arg = normalize_csv_arg(subjects_arg)
    if subjects_arg.lower() == "all":
        return list(SUBJECT_TO_CONFIG.keys())
    subjects = [normalize_csv_arg(s) for s in subjects_arg.split(",") if normalize_csv_arg(s)]
    invalid = [s for s in subjects if s not in SUBJECT_TO_CONFIG]
    if invalid:
        raise ValueError(f"Invalid math subjects: {invalid}")
    return subjects


def resolve_levels(levels_arg: str):
    levels_arg = normalize_csv_arg(levels_arg)
    if levels_arg.lower() == "all":
        return None
    levels = []
    for item in levels_arg.split(","):
        item = normalize_csv_arg(item)
        if not item:
            continue
        lvl = item if item.startswith("Level ") else f"Level {int(item)}"
        if lvl not in VALID_LEVELS:
            raise ValueError(f"Invalid math level: {item}")
        levels.append(lvl)
    return set(levels)


def main():
    args = parse_args()
    from datasets import load_dataset, concatenate_datasets

    subjects = resolve_subjects(args.subjects)
    allowed_levels = resolve_levels(args.levels)

    # ── Phase 3 와 동일한 순서: 과목별 로드 → concat → level 필터 → shuffle → subsample ──
    if args.dataset_source == "official":
        parts = []
        for subject in subjects:
            ds = load_dataset(args.official_dataset_path, SUBJECT_TO_CONFIG[subject],
                              split="train", cache_dir=args.cache_dir)
            ds = ds.map(lambda ex, subject=subject: {"type": subject})
            parts.append(ds)
        dataset = concatenate_datasets(parts)
    else:
        dataset = load_dataset(args.flat_dataset_path, split="train", cache_dir=args.cache_dir)
        subject_set = set(subjects)
        dataset = dataset.filter(lambda ex: ex.get("type") in subject_set)

    if allowed_levels is not None:
        dataset = dataset.filter(lambda ex: ex.get("level") in allowed_levels)

    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    rows, skipped = [], 0
    for idx, ex in enumerate(dataset):
        problem = (ex.get("problem") or "").strip()
        solution = (ex.get("solution") or "").strip()
        if not problem or not solution:
            skipped += 1
            continue
        try:
            # rng 는 Phase 3 와 동일하게 seed+idx (mixed 모드에서만 실제로 쓰인다)
            target = build_target(solution, random.Random(args.seed + idx),
                                  args.train_on_mixed_formats)
        except ValueError:
            skipped += 1          # boxed 답이 없는 행 — Phase 3 도 여기서 예외를 낸다
            continue
        rows.append({"question": problem, "response": target})

    out_path = args.output or os.path.join("data", f"math_train_task_{len(rows)}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    print(f"✓ {out_path}  ({len(rows)} rows, skipped {skipped})")
    print(f"  subjects={subjects}")
    print(f"  levels={'all' if allowed_levels is None else sorted(allowed_levels)}")
    print(f"  source={args.dataset_source}  seed={args.seed}  mixed={args.train_on_mixed_formats}")
    if rows:
        print("\n  [샘플 0]")
        print("  Q:", rows[0]["question"][:160].replace("\n", " "))
        print("  A:", rows[0]["response"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
