#!/usr/bin/env python3
"""추가 실험 5 (AsFT/Lisa full-param) 의 RESULTS.md 섹션을 생성한다.

ASR 은 HarmBench 개별 결과 JSON 에서(_asr_from_results), downstream 은 lm-eval 개별
results_*.json 에서(_lmeval_from_results) 직접 읽는다. 두 요약 CSV 모두 각 단계가 **전부**
끝나야 생성되므로, 진행 중에도 나온 것까지 반영하려면 이렇게 해야 한다.
아직 안 나온 값은 ⏳ 로 둔다.
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
asrmod = _load("asrmod", os.path.join(HERE, "_asr_from_results.py"))

import glob, json
LM_ROOT = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness/eval_results"
PREF = {"gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match"],
        "math": ["exact_match"], "medqa": ["acc,none", "acc"], "arc": ["acc,none", "exact_match"]}

def lm_score(key):
    d = os.path.join(LM_ROOT, "kmseong__" + key)
    files = sorted(glob.glob(os.path.join(d, "results_*.json")), key=os.path.getmtime)
    if not files: return None
    res = json.load(open(files[-1])).get("results", {})
    for tname, metrics in res.items():
        low = tname.lower()
        for frag, keys in PREF.items():
            if frag in low:
                for k in keys:
                    for mk, mv in metrics.items():
                        if mk.startswith(k) and isinstance(mv, (int, float)):
                            return float(mv)
    for tname, metrics in res.items():
        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)) and "stderr" not in mk:
                return float(mv)
    return None

# (제목, 기준ASR, 기준downstream, 기준리포설명, [(hb_key, 기법)])
GROUPS = [
 ("Llama-2-7B-Chat / GSM8K", 0.2078, 0.4117, "`kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5`", [
   ("llama2_7b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("llama2_7b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
 ("Llama-2-13B-Chat / GSM8K", 0.1004, 0.4594, "`wvnvwn/llama-2-13b-chat-hf-lr5e-5-gsm8k-lr5e-5`", [
   ("llama2_13b-chat-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("llama2_13b-chat-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
 ("Llama-3.1-8B-Instruct / MATH", 0.0928, 0.1212, "`kmseong/llama3_1_8b_instruct_MATH_lr5e-5`", [
   ("llama3_1_8b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("llama3_1_8b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
 ("Llama-3.2-3B-Instruct / MATH", 0.0865, 0.2152, "`kmseong/llama3_2_3b_instruct_MATH_lr5e-5`", [
   ("llama3_2_3b-instruct-CB_SSFT-asft_math_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("llama3_2_3b-instruct-CB_SSFT-lisa_math_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
 ("Qwen2.5-7B-Instruct / GSM8K", 0.0362, 0.6732, "`wvnvwn/qwen-2.5-7B-Instruct-SSFT-gsm8k-lr5e-5`", [
   ("qwen2_5_7b-instruct-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("qwen2_5_7b-instruct-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
 ("Gemma-2-9B-IT / GSM8K", 0.0537, 0.6975, "`wvnvwn/gemma-2-9b-it-lr3e-5-gsm8k-lr1e-5`", [
   ("gemma2_9b-it-CB_SSFT-asft_gsm8k_lambda1.0_fullft_lr1e-5", "AsFT λ=1.0 (full)"),
   ("gemma2_9b-it-CB_SSFT-lisa_gsm8k_rho1.0_fullft_lr1e-5", "Lisa ρ=1.0 (full)")]),
 ("Llama-2-7B-Chat / MedQA", None, None,
  "**없음** — medqa full-param 은 lr 3e-5 / 1e-5 만 존재하고 이 라인(lr 5e-5)과 맞는 것이 없다 "
  "(사용자 결정 2026-09-19: raw 만 싣는다)", [
   ("llama2_7b-chat-CB_SSFT-asft_medqa_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("llama2_7b-chat-CB_SSFT-lisa_medqa_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
 ("Llama-2-7B-Chat / ARC-C", None, None,
  "`kmseong/llama2_7b-chat-arc_ssft_lr5e-5` (동작점 일치 확인, **측정 예정**)", [
   ("llama2_7b-chat-CB_SSFT-asft_arc_lambda1.0_fullft_lr5e-5", "AsFT λ=1.0 (full)"),
   ("llama2_7b-chat-CB_SSFT-lisa_arc_rho1.0_fullft_lr5e-5", "Lisa ρ=1.0 (full)")]),
]

out = []
ndone = 0; ntotal = 0
for title, bs, bd, bref, rows in GROUPS:
    out.append(f"### {title}")
    out.append(f"Δ 기준행: {bref}"
               + (f" — AVG {bs:.4f} / downstream {bd:.4f}" if bs is not None else ""))
    out.append("")
    out.append("| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, lab in rows:
        ntotal += 1
        vals, avg, n = asrmod.row(k)
        dv = lm_score(k)
        if dv is not None: ndone += 1
        f = lambda v: "⏳" if v is None else f"{v:.4f}"
        sgn = lambda v: "⏳" if v is None else f"{v:+.4f}"
        ds = (avg - bs) if (bs is not None and avg is not None) else None
        dd = (dv - bd) if (bd is not None and dv is not None) else None
        do = (dd - ds) if (ds is not None and dd is not None) else None
        if bs is None: ds = dd = do = "NA"
        g = lambda v: "—" if v == "NA" else sgn(v)
        out.append(f"| [`{k}`](https://huggingface.co/kmseong/{k}) | {lab} | "
                   f"{f(vals[0])} | {f(vals[1])} | {f(vals[2])} | {f(vals[3])} | **{f(avg)}** | "
                   f"{f(dv)} | {g(ds)} | {g(dd)} | {g(do)} |")
    out.append("")
print(f"<!-- downstream {ndone}/{ntotal} -->")
print("\n".join(out))
