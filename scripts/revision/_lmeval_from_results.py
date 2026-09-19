#!/usr/bin/env python3
"""lm-eval 개별 results_*.json 에서 downstream 점수를 뽑는다.

요약 CSV 는 lm-eval 이 **전부** 끝나야 생성되므로 중간 확인용.
지표 선택 규칙은 RESULTS.md 의 다른 표와 같다:
  gsm8k → flexible-extract / MATH → exact_match / MedQA·ARC → acc
"""
import glob, json, os, sys

ROOT = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness/eval_results"
PREF = {  # task 이름 조각 -> 선호 지표 키 조각 (앞선 것 우선)
    "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match"],
    "math":  ["exact_match"],
    "medqa": ["acc,none", "acc"],
    "arc":   ["acc,none", "acc"],
}

def pick(res):
    for tname, metrics in res.items():
        low = tname.lower()
        for frag, keys in PREF.items():
            if frag in low:
                for k in keys:
                    for mk, mv in metrics.items():
                        if mk.startswith(k) and isinstance(mv, (int, float)):
                            return tname, mk, float(mv)
    # 못 찾으면 첫 숫자 지표
    for tname, metrics in res.items():
        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)) and "stderr" not in mk:
                return tname, mk, float(mv)
    return None, None, None

for d in sorted(glob.glob(os.path.join(ROOT, "*fullft*"))):
    files = sorted(glob.glob(os.path.join(d, "results_*.json")), key=os.path.getmtime)
    if not files:
        continue
    data = json.load(open(files[-1]))
    res = data.get("results", {})
    t, mk, v = pick(res)
    key = os.path.basename(d).replace("kmseong__", "")
    print(f"{key}\t{t}\t{mk}\t{v:.4f}" if v is not None else f"{key}\t-\t-\t-")
