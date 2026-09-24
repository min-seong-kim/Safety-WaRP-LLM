"""
열 버전 WSR-RSN-Tune 셀들의 HarmBench(keyword ASR) + GSM8K 결과를 한 표로 모은다.

    python sn_tune/summarize_wsr_rsn_col.py [--out_root outputs/wsr_rsn_col_p3]

읽는 곳
  - <cell>/cell_config.json, <cell>/MODEL_DIR
  - <HB>/results/evaluation_summary_*.csv  (키 = <model>-chat-wsr_rsn_col_<tag>-gsm8k-lr5e-5,
    같은 키가 여러 CSV 에 있으면 **가장 최근 파일**을 쓴다)
  - <LM>/logs/wsr_rsn_col/<key>/*/results_*.json  (exact_match,flexible-extract; strict 도 같이 기록)

쓰는 곳: <out_root>/RESULTS.md (덮어씀) + stdout 표.
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

# 논문/RESULTS.md 의 기준행 — 같은 표에서 Δ 를 볼 수 있게 함께 찍는다 (plain chat 출발, 2026-09-22 재측정 포함)
REFERENCE = {
    "llama2_7b": {
        "RSN-Tune (논문 Table 2)": (0.2073, 0.4026),
        "RSN-Tune (09-22 재측정, flexible)": (0.2068, 0.3336),
        "WSR-RSN 행 버전 (09-22)": (0.1525, 0.3980),
        "WSR-Tune (논문, SSFT 출발)": (0.0690, 0.3899),
    },
    "llama2_13b": {
        "RSN-Tune (논문 Table 2)": (0.2879, 0.4996),
        "RSN-Tune (09-22 재측정, flexible)": (0.2878, 0.4594),
        "WSR-RSN 행 버전 (09-22)": (0.2104, 0.4845),
        "WSR-Tune (논문, SSFT 출발)": (0.0135, 0.4958),
    },
}


def read_hb(hb_root, keys):
    """키별 (Direct, AutoDAN, PAIR, PAP, AVG, csv_path) — 최신 CSV 우선."""
    out = {}
    files = sorted(glob.glob(os.path.join(hb_root, "results", "evaluation_summary_*.csv")))
    for f in files:  # 오래된 것 → 최신 순으로 덮어쓴다
        with open(f, encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#"):
                    continue
                if row[0] in keys and len(row) >= 7:
                    try:
                        vals = [float(x) if x != "" else None for x in row[2:7]]
                    except ValueError:
                        continue
                    if vals[4] is None and None not in vals[:4]:
                        vals[4] = sum(vals[:4]) / 4
                    out[row[0]] = (*vals, os.path.basename(f))
    return out


def read_lm(lm_root, key):
    paths = glob.glob(os.path.join(lm_root, "logs", "wsr_rsn_col", key, "*", "results_*.json"))
    if not paths:
        return None
    p = sorted(paths)[-1]
    r = json.load(open(p))["results"]["gsm8k"]
    return r.get("exact_match,flexible-extract"), r.get("exact_match,strict-match"), os.path.basename(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="outputs/wsr_rsn_col_p3")
    ap.add_argument("--hb", default=os.path.expanduser("~/HarmBench"))
    ap.add_argument("--lm", default=os.path.expanduser("~/lm-evaluation-harness"))
    args = ap.parse_args()

    cells = []
    for model_dir in sorted(glob.glob(os.path.join(args.out_root, "*"))):
        model = os.path.basename(model_dir)
        for cell in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if not os.path.isfile(os.path.join(cell, "MODEL_DIR")):
                continue
            tag = os.path.basename(cell)
            key = f"{model}-chat-wsr_rsn_col_{tag}-gsm8k-lr5e-5"
            cfg = {}
            cp = os.path.join(cell, "cell_config.json")
            if os.path.isfile(cp):
                cfg = json.load(open(cp))
            cells.append((model, tag, key, cell, cfg))

    hb = read_hb(args.hb, {c[2] for c in cells})
    lines = []
    lines.append(f"# WSR-RSN-Tune 열(column) 버전 결과 — {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append("측정: HarmBench AdvBench standard · sys 모드 · GRADING=hard(keyword) · Direct/AutoDAN/PAIR/PAP · seed 42; "
                 "lm-eval GSM8K 5-shot flexible-extract. 출발 모델 plain Llama-2-chat (SN-Tune 관례). "
                 "Δ 는 같은 모델의 RSN-Tune 재측정값(09-22) 기준, Δoverall = Δdown − Δsafe.")
    lines.append("")
    for model in ("llama2_7b", "llama2_13b"):
        rows = [c for c in cells if c[0] == model]
        if not rows:
            continue
        ref = REFERENCE[model]["RSN-Tune (09-22 재측정, flexible)"]
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| 기법 | critical 열 | 파라미터% | Direct | AutoDAN | PAIR | PAP | **AVG ↓** | GSM8K flex ↑ | (strict) | Δsafe | Δdown | Δoverall |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, (avg, acc) in REFERENCE[model].items():
            ds = avg - ref[0]; dd = acc - ref[1]
            lines.append(f"| {name} | — | — | | | | | {avg:.4f} | {acc:.4f} | | {ds:+.4f} | {dd:+.4f} | {dd - ds:+.4f} |")
        for _, tag, key, cell, cfg in rows:
            ncol = cfg.get("critical_columns", {}).get("total", "?")
            pf = cfg.get("critical_param_fraction")
            pf_s = f"{100 * pf:.2f}%" if isinstance(pf, float) else "?"
            h = hb.get(key)
            lm = read_lm(args.lm, key)
            d, a, p, pp, avg = (h[:5] if h else (None,) * 5)
            acc, strict = (lm[0], lm[1]) if lm else (None, None)
            f = lambda x: "" if x is None else f"{x:.4f}"
            if avg is not None and acc is not None:
                ds = avg - ref[0]; dd = acc - ref[1]
                delta = f"{ds:+.4f} | {dd:+.4f} | **{dd - ds:+.4f}**"
            else:
                delta = " | | "
            lines.append(f"| **WSR-RSN-Tune 열 {tag}** | {ncol} | {pf_s} | {f(d)} | {f(a)} | {f(p)} | {f(pp)} | **{f(avg)}** | **{f(acc)}** | {f(strict)} | {delta} |")
        lines.append("")
        for _, tag, key, cell, cfg in rows:
            h = hb.get(key); lm = read_lm(args.lm, key)
            lines.append(f"- `{key}` → `{open(os.path.join(cell, 'MODEL_DIR')).read().strip()}`"
                         f"  (HB: {h[5] if h else '미측정'} · lm-eval: {lm[2] if lm else '미측정'})")
        lines.append("")

    # 기계용 요약 (야간 대안 체인이 읽는다)
    summary = {}
    for model, tag, key, cell, cfg in cells:
        h = hb.get(key); lm = read_lm(args.lm, key)
        summary.setdefault(model, {})[tag] = {
            "key": key, "avg": (h[4] if h else None), "attacks": (list(h[:4]) if h else None),
            "gsm8k_flex": (lm[0] if lm else None), "gsm8k_strict": (lm[1] if lm else None),
            "critical_param_fraction": cfg.get("critical_param_fraction"), "arm": cfg.get("arm", "rsn"),
            "skip_tune": cfg.get("skip_tune", "0"), "model_dir": open(os.path.join(cell, "MODEL_DIR")).read().strip(),
        }
    with open(os.path.join(args.out_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    text = "\n".join(lines)
    out = os.path.join(args.out_root, "RESULTS.md")
    os.makedirs(args.out_root, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
