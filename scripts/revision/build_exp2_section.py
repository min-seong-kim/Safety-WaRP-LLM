#!/usr/bin/env python3
"""실험 2 (base 모델 원공간 동결 스윕) 의 RESULTS.md 섹션을 생성한다.

논문 Table 1 의 "FT (X% frozen)" 행을 **base 모델 라인**으로 재현한 것.
ASR 은 HarmBench 개별 결과 JSON, downstream 은 lm-eval 개별 results_*.json 에서
직접 읽는다(요약 CSV 는 모든 단계가 끝나야 생성되므로 진행 중에는 못 쓴다).
아직 안 나온 값은 ⏳.

⚠️ 로컬 경로로 평가한 셀은 lm-eval 결과 디렉토리가 `kmseong__<key>` 가 아니라
   `__NHNHOME__...__eval_local__<key>` 다(경로의 `/` 가 `__` 로 치환된다).
   그래서 디렉토리를 **key 로 끝나는 것**으로 찾는다 — 허브/로컬 양쪽을 한 번에 덮는다.

사용:  python scripts/revision/build_exp2_section.py [--out -]
"""
import argparse, glob, importlib.util, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
asrmod = _load("asrmod", os.path.join(HERE, "_asr_from_results.py"))

LM_ROOT = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/lm-evaluation-harness/eval_results"
MODELS_YAML = ("/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/HarmBench/"
               "configs/model_configs/models.yaml")

_HBKEY = None
def hb_keys_for(repo):
    """repo 에 대응하는 HarmBench experiment key 들을 models.yaml 에서 역조회한다.

    ⚠️ key 가 항상 basename 인 것이 아니다. `register_models_basename.py` 는 같은 repo 가
       **다른 key 로 이미 등록돼 있으면 그 key 를 그대로 재사용**한다(같은 모델이 서로 다른
       gpu_memory_utilization 으로 두 번 등록되는 것을 막기 위해서다). 실제로
       meta-llama/Llama-2-7b-hf → `llama2_7b-base`,
       kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5 → `Llama-3_1-8B-base-gsm8k-SSFT_lr1e-5`
       처럼 basename 과 다르다. basename 으로 찾으면 결과가 있는데도 ⏳ 로 보인다.
    """
    global _HBKEY
    if _HBKEY is None:
        import yaml
        _HBKEY = {}
        try:
            cfg = yaml.safe_load(open(MODELS_YAML, encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        for k, v in cfg.items():
            try:
                _HBKEY.setdefault(v["model"]["model_name_or_path"], []).append(k)
            except Exception:
                pass
    keys = list(_HBKEY.get(repo, []))
    bn = repo.split("/")[-1]
    if bn not in keys:
        keys.append(bn)          # 아직 등록 전이면 basename 으로도 시도
    return keys
GSM8K_PREF = ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match"]

def lm_score(key):
    """key 로 끝나는 결과 디렉토리를 찾아 gsm8k 점수를 돌려준다(허브/로컬 공용)."""
    cands = [d for d in glob.glob(os.path.join(LM_ROOT, "*"))
             if os.path.isdir(d) and os.path.basename(d).endswith(key)]
    files = []
    for d in cands:
        files += glob.glob(os.path.join(d, "results_*.json"))
    if not files:
        return None
    res = json.load(open(max(files, key=os.path.getmtime))).get("results", {})
    for tname, metrics in res.items():
        if "gsm8k" not in tname.lower():
            continue
        for pref in GSM8K_PREF:
            for mk, mv in metrics.items():
                if mk.startswith(pref) and isinstance(mv, (int, float)):
                    return float(mv)
    return None

# (표시이름, HarmBench key, HF 리포 or None)
GROUPS = [
    ("Llama-2-7B (base) / GSM8K · lr 3e-5", [
        ("원본 base",           "Llama-2-7b-hf",                                   "meta-llama/Llama-2-7b-hf"),
        ("+ CB safety SFT",     "llama2_7b-base-CB_SSFT-lr3e-5",                   "kmseong/llama2_7b-base-CB_SSFT-lr3e-5"),
        ("FT (0% frozen)",      "llama2_7b-base-gsm8k_ssft_lr3e-5",                "kmseong/llama2_7b-base-gsm8k_ssft_lr3e-5"),
    ] + [(f"FT ({p}% frozen)", f"llama2_7b-base-origspace-freeze-p{p:02d}-gsm8k-lr3e-5",
          f"kmseong/llama2_7b-base-origspace-freeze-p{p:02d}-gsm8k-lr3e-5") for p in (10,20,30,40,50)]),
    ("Llama-3.1-8B (base) / GSM8K · lr 1e-5", [
        ("원본 base",           "Llama-3.1-8B",                                    "meta-llama/Llama-3.1-8B"),
        ("+ CB safety SFT",     "Llama-3.1-8B-base-SSFT_lr5e-5",                   "kmseong/Llama-3.1-8B-base-SSFT_lr5e-5"),
        ("FT (0% frozen)",      "Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5",             "kmseong/Llama-3.1-8B-base-gsm8k-SSFT_lr1e-5"),
    ] + [(f"FT ({p}% frozen)", f"llama3_1_8b-base-origspace-freeze-p{p:02d}-gsm8k-lr1e-5",
          f"kmseong/llama3_1_8b-base-origspace-freeze-p{p:02d}-gsm8k-lr1e-5") for p in (10,20,30,40,50)]),
]

def f(v, nd=4):
    return "⏳" if v is None else f"{v:.{nd}f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="-")
    a = ap.parse_args()
    out = []
    for title, rows in GROUPS:
        out += [f"### {title}", "",
                "| 모델 | 조건 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        base = None
        data = []
        for label, key, repo in rows:
            # HarmBench key 는 models.yaml 역조회로 얻는다(basename 이 아닐 수 있다).
            cand = hb_keys_for(repo) if repo else [key]
            vals, avg, n = max((asrmod.row(c) for c in cand), key=lambda r: r[2])
            down = None
            for c in cand + [key]:
                down = lm_score(c)
                if down is not None:
                    break
            data.append((label, key, repo, vals, avg, n, down))
            if label == "FT (0% frozen)":
                base = (avg, down)
        for label, key, repo, vals, avg, n, down in data:
            name = f"[`{key}`](https://huggingface.co/{repo})" if repo else f"`{key}`"
            ds = dd = "—"
            if base and base[0] is not None and avg is not None and label.startswith("FT ("):
                ds = f"{avg - base[0]:+.4f}"
            if base and base[1] is not None and down is not None and label.startswith("FT ("):
                dd = f"{down - base[1]:+.4f}"
            cells = [f(v) for v in vals]
            out.append(f"| {name} | {label} | " + " | ".join(cells) +
                       f" | **{f(avg)}** | {f(down)} | {ds} | {dd} |")
        out.append("")
    txt = "\n".join(out)
    if a.out == "-":
        print(txt)
    else:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
