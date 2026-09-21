#!/usr/bin/env python3
"""Qwen2.5-7B / GSM8K · α=32 재학습(_v2) 과 기존 α=32 세대를 나란히 놓는다.

RESULTS.md 의 'Qwen2.5-7B-It / GSM8K' 표와 같은 규칙:
  ASR  = HarmBench keyword(sys 모드) Direct/AutoDAN/PAIR/PAP 의 평균
  down = lm-eval gsm8k exact_match,flexible-extract
  Δ 기준 = 같은 α 의 Vanilla LoRA
  Δoverall = Δdown − Δsafe  (클수록 좋음)

로컬 경로로 평가하면 lm-eval 결과 디렉토리 이름이 '/'→'__' 치환으로 길어진다.
모델 이름은 항상 마지막 '__' 뒤쪽이므로 rsplit('__',1)[-1] 로 정규화한다.
"""
import glob, json, os, statistics

BASE = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong"
HB_ROOT = os.path.join(BASE, "HarmBench", "results", "Advbench_behaviors_standard")
LM_ROOT = os.path.join(BASE, "lm-evaluation-harness", "eval_results")
ATTACKS = ["DirectRequest", "AutoDAN", "PAIR", "PAP"]

ROWS = [
    ("Vanilla LoRA (α=32, 기준)", "qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4", True),
    ("AsFT  (α=32, 기존 8/28)",   "qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4", False),
    ("AsFT  (α=32, 재학습 _v2)",  "qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4_v2", False),
    ("Lisa  (α=32, 기존 9/1)",    "qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4", False),
    ("Lisa  (α=32, 재학습 _v2)",  "qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4_v2", False),
]
# 기존 세대의 공표값 (RESULTS.md 'Qwen2.5-7B-It / GSM8K'). 로컬 로그가 없을 때 채운다.
PUBLISHED = {
    "qwen2_5_7b-instruct-CB_SSFT-lora_gsm8k_lr3e-4":
        ([0.0019, 0.0000, 0.0462, 0.1050], 0.0383, 0.7149),
    "qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_lr3e-4":
        ([0.0000, 0.0000, 0.0231, 0.1192], 0.0356, 0.7377),
    "qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_lr3e-4":
        ([0.0000, 0.0000, 0.0288, 0.1300], 0.0397, 0.7278),
}


def asr_for(key, attack):
    hits = glob.glob(os.path.join(HB_ROOT, attack, "**", "results", key + ".json"), recursive=True)
    if not hits:
        return None
    data = json.load(open(max(hits, key=os.path.getmtime)))
    labels = [float(it["label"]) for items in data.values() for it in items
              if isinstance(it, dict) and "label" in it]
    return statistics.mean(labels) if labels else None


def gsm8k_for(key):
    best = None
    for d in glob.glob(os.path.join(LM_ROOT, "*")):
        if os.path.basename(d).rsplit("__", 1)[-1] != key:
            continue
        files = sorted(glob.glob(os.path.join(d, "results_*.json")), key=os.path.getmtime)
        if not files:
            continue
        res = json.load(open(files[-1])).get("results", {})
        for tname, metrics in res.items():
            if "gsm8k" not in tname.lower():
                continue
            for mk, mv in metrics.items():
                if mk.startswith("exact_match,flexible-extract") and isinstance(mv, (int, float)):
                    best = float(mv)
    return best


def fmt(v):
    return "  —   " if v is None else "%.4f" % v


def main():
    print("%-28s %8s %8s %8s %8s %8s %9s   %s" %
          ("행", "Direct", "AutoDAN", "PAIR", "PAP", "AVG", "GSM8K", "출처"))
    print("-" * 104)
    data = {}
    for label, key, _is_ref in ROWS:
        vals = [asr_for(key, a) for a in ATTACKS]
        avg = statistics.mean(vals) if all(v is not None for v in vals) else None
        down = gsm8k_for(key)
        src = "이번 측정"
        if avg is None and down is None and key in PUBLISHED:
            vals, avg, down = PUBLISHED[key]
            src = "RESULTS.md 기존값"
        elif avg is None or down is None:
            src = "측정 진행중 (%d/4 공격)" % sum(v is not None for v in vals)
        data[key] = (avg, down)
        print("%-28s %8s %8s %8s %8s %8s %9s   %s" %
              (label, fmt(vals[0]), fmt(vals[1]), fmt(vals[2]), fmt(vals[3]),
               fmt(avg), fmt(down), src))

    ref_avg, ref_down = data[ROWS[0][1]]
    if ref_avg is None or ref_down is None:
        print("\n기준행(Vanilla LoRA α=32)이 아직 없어 Δ 를 계산하지 않는다.")
        return
    print("\n%-28s %9s %9s %11s" % ("행", "Δsafe", "Δdown", "Δoverall"))
    print("-" * 62)
    for label, key, is_ref in ROWS:
        if is_ref:
            continue
        avg, down = data[key]
        if avg is None or down is None:
            print("%-28s %9s %9s %11s" % (label, "—", "—", "—")); continue
        ds, dd = avg - ref_avg, down - ref_down
        print("%-28s %+9.4f %+9.4f %+11.4f" % (label, ds, dd, dd - ds))
    print("\n⚠️ 같은 설정 재학습 시 keyword ASR 이 ±0.05 움직인다(repro_2026-09/).")
    print("   |Δ| 가 그보다 작은 차이는 단일 run 으로 방향을 단정하면 안 된다.")


if __name__ == "__main__":
    main()
