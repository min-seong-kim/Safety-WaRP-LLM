#!/usr/bin/env python3
"""
Phase 3 혼합용 harmful 부분집합을 만들어 data/ 에 저장한다.

왜 따로 저장하는가:
  models/phase3_extra_learning.py 의 _mix_safety_data() 는 매 실행마다
  rng.sample() 로 원본 4994개에서 뽑는다. 부분집합을 파일로 고정해두면
  - 어떤 747개를 썼는지 실험 기록으로 남고,
  - 여러 arm(유해 응답 / 거부 응답)이 **완전히 동일한 프롬프트 집합**을 쓰게 되어
    응답 종류만 바뀌는 깨끗한 대조가 된다.
  (파일 크기가 요청 개수와 같으면 _mix_safety_data 는 재샘플링 없이 전량 사용한다.)

출력 스키마 (항목당):
  category  : 원본 카테고리
  prompt    : 유해 질문
  response  : 유해 응답      (원본 'output')        → harmful arm
  refusal   : 거부 응답      (원본 'llama3_output') → control arm

사용:
  python scripts/build_harmful_mix_subset.py            # 기본 747개
  python scripts/build_harmful_mix_subset.py --n 1000 --out data/xxx.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/beavertails_cb_train.json")
    p.add_argument("--out", default="data/beavertails_harmful_747.json")
    p.add_argument("--n", type=int, default=747,
                   help="추출 개수 (기본 747 = gsm8k train 7473 의 10%%)")
    p.add_argument("--seed", type=int, default=42,
                   help="_mix_safety_data 기본 seed 와 동일하게 42")
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--harmful_field", default="output")
    p.add_argument("--refusal_field", default="llama3_output")
    p.add_argument("--stratified", action="store_true",
                   help="category 비율을 원본과 맞춰 층화 추출 (기본: 단순 무작위)")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    data = json.loads(src.read_text(encoding="utf-8"))

    valid = [
        d for d in data
        if d.get(args.prompt_field) and d.get(args.harmful_field) and d.get(args.refusal_field)
    ]
    print(f"원본 {src}: {len(data)}개 (필드 3개 모두 유효: {len(valid)}개)")
    if len(valid) < args.n:
        raise SystemExit(f"❌ 유효 샘플 {len(valid)}개 < 요청 {args.n}개")

    rng = random.Random(args.seed)
    if args.stratified:
        # category 별로 원본 비율에 비례해 뽑는다 (largest-remainder 로 총합 보정)
        by_cat = {}
        for d in valid:
            by_cat.setdefault(d.get("category", ""), []).append(d)
        quota, rema = {}, []
        for cat, items in by_cat.items():
            exact = len(items) / len(valid) * args.n
            quota[cat] = int(exact)
            rema.append((exact - int(exact), cat))
        for _, cat in sorted(rema, reverse=True)[: args.n - sum(quota.values())]:
            quota[cat] += 1
        picked = []
        for cat, k in quota.items():
            picked.extend(rng.sample(by_cat[cat], min(k, len(by_cat[cat]))))
        rng.shuffle(picked)
        sample = picked[: args.n]
    else:
        sample = rng.sample(valid, args.n)

    out_items = [
        {
            "category": d.get("category", ""),
            "prompt": d[args.prompt_field],
            "response": d[args.harmful_field],
            "refusal": d[args.refusal_field],
        }
        for d in sample
    ]

    out = Path(args.out)
    out.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")

    n_dup = len(out_items) - len({d["prompt"] for d in out_items})
    print(f"\n저장: {out}  ({len(out_items)}개, {out.stat().st_size / 2**20:.2f} MiB)")
    print(f"  프롬프트 중복: {n_dup}")
    print(f"  seed={args.seed}, 방식={'층화' if args.stratified else '단순 무작위'}")

    print("\n카테고리 분포 (상위 10, 괄호는 원본 비율):")
    cnt = Counter(c for d in out_items for c in d["category"].split(","))
    base = Counter(c for d in valid for c in d.get("category", "").split(","))
    tot_s = sum(cnt.values()) or 1
    tot_b = sum(base.values()) or 1
    for cat, k in cnt.most_common(10):
        print(f"  {cat:<40}{k:>5} ({k/tot_s*100:5.2f}%  원본 {base[cat]/tot_b*100:5.2f}%)")


if __name__ == "__main__":
    main()
