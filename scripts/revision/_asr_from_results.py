#!/usr/bin/env python3
"""HarmBench 개별 결과 JSON 에서 keyword ASR 을 계산한다.

단계 요약 CSV 는 HarmBench 가 **전부** 끝나야 생성되므로, 중간 시점에 이미 끝난
모델만 먼저 뽑아 쓰기 위한 도구다. 계산식이 맞는지는 이미 측정값이 알려진 모델로
대조 검증한다(--verify).
"""
import argparse, glob, json, os, statistics, sys

HB = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/HarmBench"
ROOT = os.path.join(HB, "results", "Advbench_behaviors_standard")
ATTACKS = ["DirectRequest", "AutoDAN", "PAIR", "PAP"]


def asr_for(model_key, attack):
    hits = glob.glob(os.path.join(ROOT, attack, "**", "results", f"{model_key}.json"),
                     recursive=True)
    if not hits:
        return None
    # 여러 base 디렉토리에 같은 이름이 있으면 최신 것
    path = max(hits, key=os.path.getmtime)
    data = json.load(open(path))
    labels = []
    for _, items in data.items():
        for it in items:
            if isinstance(it, dict) and "label" in it:
                labels.append(float(it["label"]))
    return statistics.mean(labels) if labels else None


def row(model_key):
    vals = [asr_for(model_key, a) for a in ATTACKS]
    done = [v for v in vals if v is not None]
    avg = statistics.mean(vals) if all(v is not None for v in vals) else None
    return vals, avg, len(done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="hb_key (리포명에서 네임스페이스 제외)")
    ap.add_argument("--glob", default=None, help="키 패턴 (예: '*fullft*')")
    args = ap.parse_args()

    keys = list(args.keys)
    if args.glob:
        seen = set()
        for a in ATTACKS:
            for p in glob.glob(os.path.join(ROOT, a, "**", "results", f"{args.glob}.json"),
                               recursive=True):
                seen.add(os.path.basename(p)[:-5])
        keys += sorted(seen)
    for k in dict.fromkeys(keys):
        vals, avg, n = row(k)
        fmt = lambda v: "  —   " if v is None else f"{v:.4f}"
        print(f"{k}\t" + "\t".join(fmt(v) for v in vals)
              + f"\t{fmt(avg)}\t({n}/4)")
