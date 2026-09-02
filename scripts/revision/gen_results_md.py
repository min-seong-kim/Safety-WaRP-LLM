#!/usr/bin/env python
"""평가 결과를 하나의 md 로 정리한다 (손으로 적지 않는다).

  ASR        : HarmBench results/evaluation_summary_*.csv  (sys 모드, ASR(keyword) 블록)
  downstream : lm-evaluation-harness/logs/eval_*_results.log
  리포명     : scripts/revision/common.sh 의 hf_repo_id 규칙과 동일한 구성

  python scripts/revision/gen_results_md.py --out RESULTS.md
"""
import argparse, glob, os, re, subprocess, sys

HB = "/home/edgeai_lab/HarmBench"
LM = "/home/edgeai_lab/lm-evaluation-harness"
HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [  # (prefix, task, 표시명, downstream 지표명)
    ("llama2_7b-chat",       "gsm8k",  "Llama-2-7B-chat",   "GSM8K"),
    ("llama2_13b-chat",      "gsm8k",  "Llama-2-13B-chat",  "GSM8K"),
    ("llama3_2_3b-instruct", "math",   "Llama-3.2-3B-It",   "MATH"),
    ("llama3_1_8b-instruct", "math",   "Llama-3.1-8B-It",   "MATH"),
    ("qwen2_5_7b-instruct",  "gsm8k",  "Qwen2.5-7B-It",     "GSM8K"),
    ("gemma2_9b-it",         "gsm8k",  "Gemma-2-9B-IT",     "GSM8K"),
]
METHODS = [
    ("Vanilla LoRA", "lora_{t}_lr{lr}"),
    ("AsFT",         "asft_{t}_lambda1.0_lr{lr}"),
    ("LISA (ρ=0.0)", "lisa_{t}_rho0.0_lr{lr}"),
    ("LISA (ρ=1.0)", "lisa_{t}_rho1.0_lr{lr}"),
    ("SEAL",         "seal_{t}_topp0.8_lr5e-5"),
    ("SafeLoRA",     "safelora_{t}_thr0.3_lr{lr}"),
    ("SaLoRA",       "salora_{t}_rs32rt32_lr{lr}"),
    ("WSR-LoRA (α=32)", "wsr-lora_{t}_rho0.1_lr{lr}"),
    ("WSR-LoRA (α=16)", "wsr-lora_{t}_rho0.1_a16_lr{lr}"),
]
NS = "kmseong"

def lora_lr(task):  # common.sh 의 lora_lr() 과 동일 규칙
    return "7e-5" if task == "agnews" else "3e-4"

def collect_asr():
    """가장 최근 값이 이기도록 시간순으로 덮어쓴다. sys 모드 CSV 만 읽는다."""
    out = {}
    for f in sorted(glob.glob(f"{HB}/results/evaluation_summary_*.csv"), key=os.path.getmtime):
        head = open(f, encoding="utf-8", errors="ignore").read(400)
        if "STRIP_SAFETY_SYSTEM_PROMPT=0" not in head:
            continue          # nosys / 구형식 제외
        for line in open(f, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split(",")
            if len(p) > 7 and p[0] and not p[0].startswith("#") and p[2] not in ("", "Direct"):
                try:
                    out[p[0]] = dict(D=p[2], AD=p[3], PAIR=p[4], PAP=p[5], AVG=float(p[6]))
                except ValueError:
                    pass
    return out

def collect_ds():
    """(값, 측정시각) 을 함께 모은다 — 재학습된 모델의 옛 수치를 걸러내기 위해."""
    out = {}
    for f in sorted(glob.glob(f"{LM}/logs/eval_*_results.log"), key=os.path.getmtime):
        ts = os.path.getmtime(f)
        cur = None
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = re.search(r"모델 평가 중:\s*(\S+)", line)
            if m:
                cur = m.group(1).split("/")[-1]; continue
            if cur and ("flexible-extract" in line or "hendrycks_math_safe" in line):
                v = re.findall(r"\|\s*([0-9.]+)\s*\|±", line)
                if v:
                    out[cur] = (float(v[0]), ts); cur = None
    return out


def hub_mtimes(keys):
    """허브의 lastModified. 측정 로그가 이보다 오래됐으면 그 수치는 옛 모델 것이다."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        got = {}
        # ⚠️ list_models() 는 lastModified 를 채우지 않는다(전부 None) — expand 로 명시해야 한다.
        #    빠뜨리면 신선도 검증이 조용히 전부 건너뛰어져 옛 수치를 그대로 싣게 된다.
        for m in api.list_models(author=NS, expand=["lastModified"]):
            n = m.id.split("/")[-1]
            if n in keys and m.lastModified:
                got[n] = m.lastModified.timestamp()
        if not got:
            print("[warn] 허브 lastModified 를 하나도 못 받음 — 신선도 검증이 무력화됨", file=sys.stderr)
        return got
    except Exception as e:
        print(f"[warn] 허브 조회 실패 — 신선도 검증 생략: {e}", file=sys.stderr)
        return {}

def fmt(v, w=6):
    return f"{v:.4f}" if isinstance(v, float) else "—"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args()
    asr, ds = collect_asr(), collect_ds()
    all_keys = {f"{pre}-CB_SSFT-" + pat.format(t=task, lr=lora_lr(task))
                for pre, task, _, _ in MODELS for _, pat in METHODS}
    hub = hub_mtimes(all_keys)
    STALE = set()
    for k, (_v, ts) in ds.items():
        h = hub.get(k)
        if h and ts < h - 300:      # 모델이 측정 이후 갱신됨 = 재학습됨
            STALE.add(k)

    L = []
    L.append("# Revision 실험 결과 — CB 축 / Llama-2-7B·13B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B")
    L.append("")
    L.append("**자동 생성 파일입니다.** 손으로 고치지 마십시오 — 다시 만들려면:")
    L.append("")
    L.append("```bash")
    L.append("python scripts/revision/gen_results_md.py --out RESULTS.md")
    L.append("```")
    L.append("")
    L.append("## 측정 조건")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append("| 安全 평가 | HarmBench · Advbench_behaviors_standard · **sys 모드**(llama-2 `<<SYS>>` 안전 프롬프트 포함) |")
    L.append("| ASR 채점 | `GRADING=hard` (refusal keyword) — 표의 값은 ASR(keyword) |")
    L.append("| 공격 | DirectRequest · AutoDAN · PAIR · PAP (AVG = 4종 평균) |")
    L.append("| downstream | lm-evaluation-harness, 5-shot, `--apply_chat_template`, GSM8K=flexible-extract / MATH=`hendrycks_math_safe` |")
    L.append("| 학습 공통 | epochs 3 · 유효 batch 16 · max_len 1024 · seed 42 · bf16 · cosine · warmup 0.03 · wd 0.0 |")
    L.append("| LoRA 계열 | r=16 · α=32 · dropout 0.05 · targets `q,k,v,up,down` · lr 3e-4 (agnews 7e-5) |")
    L.append("| 출발 모델 | 각 base 의 CB(circuit_breakers) safety-tuned 체크포인트 |")
    L.append("")
    L.append("**JB AVG 는 낮을수록 안전**하고, downstream 은 높을수록 좋습니다.")
    L.append("")

    for pre, task, disp, dsname in MODELS:
        rows = []
        for md, pat in METHODS:
            key = f"{pre}-CB_SSFT-" + pat.format(t=task, lr=lora_lr(task))
            aa = asr.get(key)
            raw = ds.get(key)
            dd = None if (raw is None or key in STALE) else raw[0]
            if aa is None and dd is None and key not in STALE:
                continue
            rows.append((md, key, aa, dd, key in STALE))
        if not rows:
            continue
        L.append(f"## {disp} / {dsname}")
        L.append("")
        L.append(f"| 기법 | {dsname} | JB AVG | Direct | AutoDAN | PAIR | PAP | 모델 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for md, key, aa, dd, stale in rows:
            url = f"https://huggingface.co/{NS}/{key}"
            note = " ⟲재학습" if stale else ""
            L.append(f"| {md}{note} | {fmt(dd)} | {fmt(aa['AVG']) if aa else '—'} | "
                     f"{aa['D'] if aa else '—'} | {aa['AD'] if aa else '—'} | "
                     f"{aa['PAIR'] if aa else '—'} | {aa['PAP'] if aa else '—'} | "
                     f"[`{key}`]({url}) |")
        L.append("")

    n_asr = sum(1 for _ in asr)
    L.append("---")
    L.append("")
    L.append(f"수집된 ASR 레코드 {n_asr}건 · downstream 레코드 {len(ds)}건.")
    L.append("빈칸(—)은 아직 측정되지 않았거나 실행이 실패한 항목입니다.")
    L.append("")
    L.append("`⟲재학습` 표시는 **모델이 마지막 측정 이후 다시 학습되어 기존 수치를 버린** 항목입니다 "
             "(허브 `lastModified` 와 측정 로그 시각을 대조해 자동 판정). 재평가 대기 중입니다.")
    L.append("")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{a.out} 작성 — 모델 {len(MODELS)}개 · 기법 {len(METHODS)}종")

if __name__ == "__main__":
    main()
