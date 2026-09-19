#!/usr/bin/env python3
"""평가가 끝난 모델들의 RESULTS.md 표 행을 만든다.

HarmBench 의 run_all_*_summary.csv 는 ASR 4종과 downstream 점수를 한 줄에 담고 있어
그걸 단일 출처로 쓴다. 같은 모델이 여러 CSV 에 있으면 **가장 최근 파일**을 택한다.

  python scripts/revision/gen_exp_results_rows.py --repos kmseong/a kmseong/b
  python scripts/revision/gen_exp_results_rows.py --from_cells outputs/asft_lisa_fullft \
      --baseline kmseong/llama2_7b-chat_gsm8k_full_ft_lr5e-5 --label-map ...

Δ 열은 --baseline 을 준 경우에만 계산한다 (Δsafe = AVG−기준, Δdown = down−기준,
Δoverall = Δdown − Δsafe). 기준 행이 표에 없으면 빈 칸으로 둔다.
"""
import argparse, csv, glob, json, os, re, sys

BASE = "/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong"
HB_LOGS = os.path.join(BASE, "HarmBench", "logs")

ATTACKS = ["Direct", "AutoDAN", "PAIR", "PAP", "AVG"]
# downstream 점수 열 찾기.
#   ⚠️ 열 이름을 하드코딩하지 말 것. 실제 CSV 는 `arc_challenge_chat_remove_whitespace`,
#      `medqa_4options_acc` 처럼 harness 설정이 이름에 섞여 들어온다(2026-09-19 실측).
#      MATH 열은 이 박스에 아직 샘플이 없어 이름을 모른다 — 그래서 정규식으로 찾는다.
#   각 항목: (표시이름, 정규식, 선호 접미사 우선순위)
DOWN_PATTERNS = [
    ("gsm8k",  re.compile(r"^gsm8k", re.I),                      ["flexible", "strict"]),
    ("MATH",   re.compile(r"(hendrycks_?math|^math_|^math$)", re.I), ["exact_match", ""]),
    ("MedQA",  re.compile(r"^medqa", re.I),                      ["_acc", "_acc_norm"]),
    ("ARC-C",  re.compile(r"^arc_challenge", re.I),              [""]),
    ("AGNews", re.compile(r"^(agnews|sst2)", re.I),              [""]),
]


def load_hb_stage_csv(path, grading="keyword"):
    """HarmBench **단계** 요약 CSV(results/evaluation_summary_*.csv)를 읽는다.

    최종 병합본(logs/run_all_*_summary.csv)과 형식이 다르다. 다중 행 헤더를 쓴다:
        line: ',,ASR(keyword),,,,,ASR(Harmbench-CLS),...'   <- 그룹
        line: 'hb_주소,,Direct,AutoDAN,PAIR,PAP,AVG,...'     <- 열
    이후 행의 0번 칸이 hb_key(= 리포명에서 네임스페이스를 뺀 것)다.
    lm-eval 이 끝나기 전에 ASR 만 먼저 뽑을 때 쓴다.
    """
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        lines = list(csv.reader(f))
    gi = ci = None
    for i, r in enumerate(lines):
        if any(c.startswith(f"ASR({grading})") for c in r):
            gi = i
        if r and r[0].strip().startswith("hb_") and "Direct" in r:
            ci = i
            break
    if gi is None or ci is None:
        raise ValueError(f"헤더를 찾지 못했다: {path}")
    # ASR(grading) 그룹이 시작하는 열 위치
    start = next(j for j, c in enumerate(lines[gi]) if c.startswith(f"ASR({grading})"))
    for r in lines[ci + 1:]:
        if not r or not r[0].strip() or r[0].startswith("#"):
            continue
        key = r[0].strip()
        vals = {}
        for off, name in enumerate(ATTACKS):
            j = start + off
            vals[f"sys_ASR({grading})_{name}"] = r[j] if j < len(r) else ""
        vals["model"] = key
        rows[key] = vals
    return rows


def load_rows(variant="sys", grading="keyword"):
    """{repo: (mtime, rowdict, csvpath)} — 최신 CSV 우선."""
    best = {}
    for path in sorted(glob.glob(os.path.join(HB_LOGS, "run_all_*_summary.csv")),
                       key=os.path.getmtime):
        mt = os.path.getmtime(path)
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    m = (row.get("model") or "").strip()
                    if not m:
                        continue
                    prev = best.get(m)
                    # 값이 실제로 채워진 행만 채택 (빈 행이 최신이라고 덮어쓰지 않도록)
                    key = f"{variant}_ASR({grading})_AVG"
                    if not (row.get(key) or "").strip():
                        if prev is not None:
                            continue
                    if prev is None or mt >= prev[0]:
                        best[m] = (mt, row, path)
        except Exception as e:  # CSV 가 깨져 있어도 나머지는 살린다
            print(f"[warn] {path}: {type(e).__name__}: {e}", file=sys.stderr)
    return best


def pick_down(row):
    """ASR 이 아닌 열 중 downstream 점수를 하나 고른다. 이름을 가정하지 않는다."""
    cols = [c for c in row.keys() if c and not c.startswith(("sys_ASR", "nosys_ASR"))
            and c not in ("model", "hb_key")]
    for label, pat, prefs in DOWN_PATTERNS:
        hits = [c for c in cols if pat.search(c) and (row.get(c) or "").strip()]
        if not hits:
            continue
        for pref in prefs:                      # 선호 접미사가 있으면 그쪽을 먼저
            for c in hits:
                if pref and pref in c:
                    try:
                        return label, float(row[c])
                    except ValueError:
                        pass
        for c in hits:                          # 아니면 아무거나
            try:
                return label, float(row[c])
            except ValueError:
                pass
    return None, None


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=[])
    ap.add_argument("--from_cells", default=None,
                    help="셀 트리 루트. 각 셀의 REPO 파일에서 리포명을 모은다.")
    ap.add_argument("--baseline", default=None, help="Δ 계산 기준 리포")
    ap.add_argument("--variant", default="sys", choices=["sys", "nosys"])
    ap.add_argument("--grading", default="keyword")
    ap.add_argument("--labels", default=None,
                    help='JSON: {"repo": "표시할 기법명"}')
    ap.add_argument("--hb_summary", default=None,
                    help="HarmBench 단계 요약 CSV(results/evaluation_summary_*.csv). "
                         "lm-eval 전에 ASR 만 먼저 뽑을 때 쓴다.")
    args = ap.parse_args()

    repos = list(args.repos)
    if args.from_cells:
        for f in sorted(glob.glob(os.path.join(args.from_cells, "**", "REPO"), recursive=True)):
            if os.path.exists(os.path.join(os.path.dirname(f), ".uploaded")):
                repos.append(open(f).read().strip())
    if args.baseline and args.baseline not in repos:
        repos.insert(0, args.baseline)
    seen, ordered = set(), []
    for r in repos:
        if r and r not in seen:
            seen.add(r); ordered.append(r)

    labels = json.load(open(args.labels)) if args.labels else {}
    if args.hb_summary:
        stage = load_hb_stage_csv(args.hb_summary, args.grading)
        # hb_key 는 네임스페이스가 빠진 이름이다 → 리포명 뒷부분으로 맞춘다
        best = {}
        for repo in ordered:
            k = repo.split("/", 1)[-1]
            if k in stage:
                best[repo] = (0, stage[k], args.hb_summary)
    else:
        best = load_rows(args.variant, args.grading)

    base_avg = base_down = None
    if args.baseline and args.baseline in best:
        br = best[args.baseline][1]
        try:
            base_avg = float(br[f"{args.variant}_ASR({args.grading})_AVG"])
        except (KeyError, ValueError, TypeError):
            pass
        _, base_down = pick_down(br)

    print("| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | downstream | Δsafe | Δdown | Δoverall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    missing = []
    for repo in ordered:
        hit = best.get(repo)
        if not hit:
            missing.append(repo)
            continue
        row = hit[1]
        vals = []
        for a in ATTACKS:
            v = (row.get(f"{args.variant}_ASR({args.grading})_{a}") or "").strip()
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)
        dlabel, dval = pick_down(row)
        avg = vals[-1]
        ds = dd = do = None
        if base_avg is not None and avg is not None and repo != args.baseline:
            ds = avg - base_avg
        if base_down is not None and dval is not None and repo != args.baseline:
            dd = dval - base_down
        if ds is not None and dd is not None:
            do = dd - ds
        short = repo.split("/", 1)[-1]
        lab = labels.get(repo, "")
        sgn = lambda x: "—" if x is None else f"{x:+.4f}"
        print(f"| [`{short}`](https://huggingface.co/{repo}) | {lab} | "
              + " | ".join(fmt(v) for v in vals)
              + f" | {fmt(dval)}{'' if not dlabel else ''} | {sgn(ds)} | {sgn(dd)} | {sgn(do)} |")
    if missing:
        print("\n<!-- 평가 결과를 찾지 못한 리포:", file=sys.stderr)
        for m in missing:
            print(f"     {m}", file=sys.stderr)
        print("  -->", file=sys.stderr)


if __name__ == "__main__":
    main()
