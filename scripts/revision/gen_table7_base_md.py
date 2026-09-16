#!/usr/bin/env python
"""논문 Table 7 (base 모델) 에 rebuttal PEFT 7종을 붙인 표를 만든다. 손으로 적지 않는다.

  ASR        : HarmBench  results/evaluation_summary_*.csv  (sys 모드 = STRIP_SAFETY_SYSTEM_PROMPT=0)
  downstream : lm-evaluation-harness/logs/eval_*_results.log
  리포명     : scripts/revision/common.sh 의 hf_repo_id() 규칙과 동일하게 조립한다

왜 gen_results_md.py 와 따로 두는가
-----------------------------------
그쪽 MODELS/METHODS 는 instruct 6모델 표 전용이고, LoRA alpha 기본값(α=32)을 전제로
리포명을 만든다. base 라인은 **전 기법이 α=16** 이라 패턴이 달라서, 거기에 끼워 넣으면
기존 행들의 이름 규칙이 깨진다.

사용:
  python scripts/revision/gen_table7_base_md.py --out RESULTS_table7_base.md
  python scripts/revision/gen_table7_base_md.py --model llama3_1_8b-base   # 한 모델만
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_sibling(name, env):
    """HarmBench / lm-evaluation-harness 위치. 박스마다 다르므로 하드코딩하지 않는다."""
    cand = [os.environ.get(env)] if os.environ.get(env) else []
    repo = os.path.dirname(os.path.dirname(HERE))
    cand += [
        os.path.join(os.path.dirname(repo), name),
        os.path.join(os.path.expanduser("~"), name),
        os.path.join("/home/edgeai_lab", name),
    ]
    for c in cand:
        if c and os.path.isdir(c):
            return c
    return cand[0]


HB = _find_sibling("HarmBench", "HARMBENCH_DIR")
LM = _find_sibling("lm-evaluation-harness", "LMEVAL_DIR")
NS = os.environ.get("HF_NAMESPACE", "kmseong")

# (prefix, task, 표시명, downstream 지표명, full-param lr)
#   full-param lr 은 그 라인의 SSFT/FullFT 가 쓴 값이고 SEAL 도 거기에 맞춘다.
BASE_MODELS = [
    ("llama3_1_8b-base", "gsm8k", "Llama-3.1-8B Base", "GSM8K", "1e-5"),
    ("llama2_7b-base",   "gsm8k", "Llama-2-7B Base",   "GSM8K", "3e-5"),
]

# 이번 라인은 **전 기법 α=16** 이라 LoRA 계열 이름에 _a16 이 붙는다.
PEFT_METHODS = [
    ("Vanilla LoRA",      "lora_{t}_a16_lr{lr}"),
    ("AsFT (λ=1.0)",      "asft_{t}_lambda1.0_a16_lr{lr}"),
    ("LISA (ρ=1.0)",      "lisa_{t}_rho1.0_a16_lr{lr}"),
    ("SEAL (top-p 0.8)",  "seal_{t}_topp0.8_lr{flr}"),
    ("SafeLoRA (thr=0.3)", "safelora_{t}_thr0.3_a16_lr{lr}"),
    ("SaLoRA (r_s=r_t=32)", "salora_{t}_rs32rt32_a16_lr{lr}"),
    ("WSR-LoRA (ρ=0.3)",  "wsr-lora_{t}_rho0.3_a16_lr{lr}"),
]

# 논문 Table 7 에 이미 실린 행. 값은 HarmBench **결과 key** 다(리포 이름이 아니다).
#   ⚠️ 키 이름의 lr 과 실제 리포의 lr 이 다르다. 예: `...-SSFT-lr5e-5` 키의 리포는
#      kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5 다. 손으로 유추하지 말 것.
#   ⚠️ 대소문자만 다른 쌍둥이 키가 있다. `llama3_1_8b-base-gsm8k-warp-lr5e-5`(AVG 4.46,
#      논문 행) vs `Llama-3_1-8B-base-gsm8k-warp-lr5e-5`(AVG 81.43, 구형 배치).
#      아래 값은 2026-09-16 에 논문 수치와 대조해 확정한 것이다.
PUBLISHED = {
    "llama3_1_8b-base": [
        ("Full Params FT (논문)", "llama3_1_8b-base-gsm8k-SSFT-lr5e-5"),
        ("SafeInstr (논문)",      "llama3_1_8b-base-gsm8k-safeinstr-ratio0.1-lr5e-5"),
        ("Resta (논문)",          "llama3_1_8b-base-gsm8k-resta-gamma0.3-lr5e-5"),
        ("SafeDelta (논문)",      "llama3_1_8b-base-gsm8k-safedelta-scale0.1-lr5e-5"),
        ("SN-Tune (논문)",        "llama3_1_8b-base-gsm8k-freeze-sn-lr5e-5"),
        ("RSN-Tune (논문)",       "llama3_1_8b-base-gsm8k-freeze-rsn-lr5e-5"),
        ("WSR-Tune (논문)",       "llama3_1_8b-base-gsm8k-warp-lr5e-5"),
    ],
    "llama2_7b-base": [],
}

# 논문 Table 7 의 JB AVG — 생성된 표가 그 값을 재현하는지 자동 대조한다.
PAPER_AVG = {
    "llama3_1_8b-base-gsm8k-SSFT-lr5e-5": 8.81,
    "llama3_1_8b-base-gsm8k-safeinstr-ratio0.1-lr5e-5": 6.56,
    "llama3_1_8b-base-gsm8k-resta-gamma0.3-lr5e-5": 8.37,
    "llama3_1_8b-base-gsm8k-safedelta-scale0.1-lr5e-5": 3.99,
    "llama3_1_8b-base-gsm8k-freeze-sn-lr5e-5": 36.85,
    "llama3_1_8b-base-gsm8k-freeze-rsn-lr5e-5": 28.54,
    "llama3_1_8b-base-gsm8k-warp-lr5e-5": 4.46,
}


def lora_lr(task):
    """common.sh 의 lora_lr() 과 동일 규칙."""
    return "7e-5" if task == "agnews" else "3e-4"


# HarmBench 결과 트리에서 (공격 → 결과 디렉토리). base 모델은 base 전용 test case 를 쓴다.
#   DirectRequest / PAP 는 모든 모델이 같은 프롬프트라 base/instruct 구분이 없다.
ATTACK_DIRS = {
    "llama3_1_8b-base": {
        "Direct":  "DirectRequest/default",
        "AutoDAN": "AutoDAN/llama3_1_8b-base",
        "PAIR":    "PAIR/llama3_1_8b-base",
        "PAP":     "PAP/top_5",
    },
    "llama2_7b-base": {
        "Direct":  "DirectRequest/default",
        "AutoDAN": "AutoDAN/llama2_7b-base",
        "PAIR":    "PAIR/llama2_7b-base",
        "PAP":     "PAP/top_5",
    },
}
DATASET = "Advbench_behaviors_standard"


def asr_from_results(path):
    """결과 JSON 의 label 평균 = keyword ASR(%).

    ⚠️ evaluation_summary_*.csv 를 읽지 않고 **결과 JSON 을 직접 집계**한다.
       요약 CSV 는 실행마다 새로 생겨 옛 행이 빠지기 쉬운데, 결과 JSON 은 모델마다
       남아 있다. 이 방식이 논문 Table 7 의 7개 행을 **정확히 재현**함을 확인했다
       (2026-09-16: FullFT 8.81 / SafeInstr 6.56 / Resta 8.37 / SafeDelta 3.99 /
        SN 36.85 / RSN 28.54 / WSR-Tune 4.46 — 논문값과 소수점까지 일치).
    """
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    vals = []
    for _, v in d.items():
        for it in (v if isinstance(v, list) else [v]):
            if isinstance(it, dict) and it.get("label") is not None:
                vals.append(float(it["label"]))
    return 100.0 * sum(vals) / len(vals) if vals else None


def collect_asr(model_prefix):
    """key → {Direct, AutoDAN, PAIR, PAP, AVG}. 4개가 다 있어야 AVG 를 낸다."""
    dirs = ATTACK_DIRS.get(model_prefix)
    if not dirs:
        return {}
    keys = set()
    for sub in dirs.values():
        rd = os.path.join(HB, "results", DATASET, sub, "results")
        if os.path.isdir(rd):
            keys |= {f[:-5] for f in os.listdir(rd)
                     if f.endswith(".json") and "__" not in f}   # __harmbench/__wildguard 제외
    out = {}
    for k in sorted(keys):
        row = {a: asr_from_results(os.path.join(HB, "results", DATASET, sub, "results", k + ".json"))
               for a, sub in dirs.items()}
        got = [v for v in row.values() if v is not None]
        row["AVG"] = sum(got) / 4 if len(got) == 4 else None
        out[k] = row
    return out


def _log_order_key(path):
    """로그 파일의 **실행 시각**. 파일명 eval_YYYYMMDD_HHMMSS 를 우선 쓴다.

    ⚠️ mtime 으로 정렬하면 안 된다. 이 박스의 옛 로그는 다른 박스에서 **복사**돼 와서
       mtime 이 전부 복사 시각(2026-09-07)이고 실행 순서와 무관하다. 2026-09-16 실측:
       gemma SEAL(lr5e-5)은 2026-09-01 에 두 번 측정돼 flexible 0.2820(16:59) / 0.1865(20:42)
       인데, mtime 정렬로는 16:59 판이 "최신" 으로 잡혀 RESULTS.md(0.1865)와 어긋났다.
    """
    m = re.search(r"eval_(\d{8})_(\d{6})", os.path.basename(path))
    return (m.group(1) + m.group(2)) if m else str(int(os.path.getmtime(path)))


def collect_ds():
    """repo basename → flexible-extract 값(0~1). 나중에 실행된 로그가 이긴다.

    ⚠️ "pretrained 뒤 N자 안에서 찾기" 식 정규식을 쓰지 말 것. lm-eval 로그는 vLLM
       진행바가 **한 줄에 수천 자**라 모델 사이 간격이 쉽게 3000자를 넘고, 그러면
       `(.{0,3000}?)(?=pretrained|\\Z)` 같은 패턴은 매치가 통째로 실패해 **오류 없이
       빈 결과**가 나온다(2026-09-16 실측: 새 로그 7개 중 0개 파싱).
       대신 위치를 모아 "각 지표 줄을 바로 앞의 pretrained 에 귀속"시킨다.
    """
    out = {}
    for f in sorted(glob.glob(f"{LM}/logs/*.log"), key=_log_order_key):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        models = [(m.start(), m.group(1).split("/")[-1])
                  for m in re.finditer(r"pretrained': '([^']+)'", txt)]
        if not models:
            continue
        for mm in re.finditer(r"flexible-extract\|\s*5\|exact_match\|↑\s*\|([0-9.]+)", txt):
            prev = [name for pos, name in models if pos < mm.start()]
            if prev:
                out[prev[-1]] = float(mm.group(1))
    return {k: (v, 0) for k, v in out.items()}
def fmt(v, nd=2):
    """ASR/정확도 표시. asr_from_results 는 이미 % 단위라 다시 100 을 곱하지 않는다."""
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS_table7_base.md")
    ap.add_argument("--model", default="", help="한 모델만 (예: llama3_1_8b-base)")
    a = ap.parse_args()

    ds = collect_ds()
    if not ds:
        print(f"[warn] lm-eval 결과 로그를 찾지 못했다: {LM}/logs/eval_*_results.log",
              file=sys.stderr)

    lines = ["# Table 7 (base 모델) + rebuttal PEFT 7종", "",
             f"- HarmBench : `{HB}`  ({DATASET}, keyword ASR)",
             f"- lm-eval   : `{LM}`",
             "- 생성: `python scripts/revision/gen_table7_base_md.py` — 손으로 고치지 말 것",
             "- ASR 은 결과 JSON 의 label 평균이다. 이 계산이 논문 Table 7 의 7개 행을",
             "  소수점까지 재현함을 확인했다(아래 '논문 대조' 절).", ""]

    n_missing, mismatches = 0, []
    for pre, task, disp, dsname, flr in BASE_MODELS:
        if a.model and a.model != pre:
            continue
        asr = collect_asr(pre)
        if not asr:
            print(f"[warn] {pre}: HarmBench 결과를 찾지 못했다 — ASR 열이 빈다", file=sys.stderr)
        lr = lora_lr(task)
        lines += [f"## {disp}  ({dsname}, CB 축)", "",
                  f"| Method | Direct | AutoDAN | PAIR | PAP | JB AVG ↓ | {dsname} ↑ | source |",
                  "|---|---|---|---|---|---|---|---|"]
        rows = [(n, f"{pre}-CB_SSFT-" + p.format(t=task, lr=lr, flr=flr), "new")
                for n, p in PEFT_METHODS]
        rows += [(n, k, "paper") for n, k in PUBLISHED.get(pre, [])]
        for name, key, kind in rows:
            s_ = asr.get(key)
            d_ = ds.get(key)
            if s_ is None or s_.get("AVG") is None or d_ is None:
                n_missing += 1
            src = f"`{NS}/{key}`" if kind == "new" else "논문 Table 7"
            lines.append(
                f"| {name} | {fmt(s_['Direct']) if s_ else '—'} | {fmt(s_['AutoDAN']) if s_ else '—'} | "
                f"{fmt(s_['PAIR']) if s_ else '—'} | {fmt(s_['PAP']) if s_ else '—'} | "
                f"{fmt(s_['AVG']) if s_ else '—'} | "
                f"{f'{d_[0] * 100:.2f}' if d_ else '—'} | {src} |")
            if kind == "paper" and key in PAPER_AVG and s_ and s_.get("AVG") is not None:
                if abs(s_["AVG"] - PAPER_AVG[key]) > 0.01:
                    mismatches.append((key, s_["AVG"], PAPER_AVG[key]))
        lines.append("")

    # ── 논문 대조: 재계산값이 Table 7 과 같은지 ────────────────────────────
    lines += ["## 논문 대조 (재계산 vs Table 7 게재값)", ""]
    if mismatches:
        lines.append("| key | 재계산 | 논문 | 차이 |")
        lines.append("|---|---|---|---|")
        for k, got, want in mismatches:
            lines.append(f"| `{k}` | {got:.2f} | {want:.2f} | {got - want:+.2f} |")
        lines += ["", "⚠️ 불일치가 있다. 같은 잣대가 아니므로 새 행을 논문 표에 그대로 붙이지 말 것.", ""]
    else:
        lines += ["재계산한 논문 행이 게재값과 **전부 일치**한다 "
                  "(허용오차 0.01%p). 새 행과 논문 행을 같은 표에 놓아도 된다.", ""]

    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{a.out} 작성 — 빈칸 {n_missing}개, 논문 불일치 {len(mismatches)}건")
    if n_missing:
        print("  빈칸 = 아직 평가 안 됐거나(진행 중) 결과 key 가 다르다.", file=sys.stderr)


if __name__ == "__main__":
    main()
