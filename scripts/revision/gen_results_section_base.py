#!/usr/bin/env python
"""RESULTS.md 에 붙일 base 라인 섹션을 측정 파일에서 직접 만든다. 손으로 적지 않는다.

  ASR   : HarmBench results/<dataset>/<attack>/<base-exp>/results/<key>.json 의 label 평균
          (= keyword ASR. 논문 Table 7 의 7개 행을 소수점까지 재현하는 계산이다.)
  GSM8K : lm-evaluation-harness logs/eval_*.log 의 **flexible-extract** 5-shot
          (RESULTS.md 의 '측정 조건' 표가 못박은 규약)

  python scripts/revision/gen_results_section_base.py            # 출력만
  python scripts/revision/gen_results_section_base.py --append RESULTS.md
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _sib(name, env):
    cand = [os.environ.get(env)] if os.environ.get(env) else []
    cand += [os.path.join(os.path.dirname(REPO), name),
             os.path.join(os.path.expanduser("~"), name)]
    for c in cand:
        if c and os.path.isdir(c):
            return c
    return cand[-1]


HB = _sib("HarmBench", "HARMBENCH_DIR")
LM = _sib("lm-evaluation-harness", "LMEVAL_DIR")
DATASET = "Advbench_behaviors_standard"
ATTACKS = {"Direct": "DirectRequest/default", "AutoDAN": "AutoDAN/llama3_1_8b-base",
           "PAIR": "PAIR/llama3_1_8b-base", "PAP": "PAP/top_5"}
NS = "kmseong"

# (표시명, 리포 basename).  기준행(Δ 계산 분모)은 Vanilla LoRA (α=16) 다 —
# RESULTS.md 규약: "LoRA 행은 같은 α 의 Vanilla LoRA" 를 기준으로 한다.
ROWS = [
    ("Vanilla LoRA (α=16)",   "llama3_1_8b-base-CB_SSFT-lora_gsm8k_a16_lr3e-4",           "lora"),
    ("AsFT λ=1.0 (α=16)",     "llama3_1_8b-base-CB_SSFT-asft_gsm8k_lambda1.0_a16_lr3e-4", "lora"),
    ("LISA ρ=1.0 (α=16)",     "llama3_1_8b-base-CB_SSFT-lisa_gsm8k_rho1.0_a16_lr3e-4",    "lora"),
    ("SafeLoRA thr=0.3 (α=16)", "llama3_1_8b-base-CB_SSFT-safelora_gsm8k_thr0.3_a16_lr3e-4", "lora"),
    ("SaLoRA r_s=r_t=32 (α=16)", "llama3_1_8b-base-CB_SSFT-salora_gsm8k_rs32rt32_a16_lr3e-4", "lora"),
    ("WSR-LoRA ρ=0.3 (α=16)", "llama3_1_8b-base-CB_SSFT-wsr-lora_gsm8k_rho0.3_a16_lr3e-4", "lora"),
    ("SEAL top-p 0.8 (full-param, lr 1e-5)", "llama3_1_8b-base-CB_SSFT-seal_gsm8k_topp0.8_lr1e-5", "full"),
]
BASELINE = "llama3_1_8b-base-CB_SSFT-lora_gsm8k_a16_lr3e-4"

# 참고용 재측정 — 논문 WSR-Tune 행의 safety 가 나온 바로 그 모델.
CONTROL = ("WSR-Tune (논문 safety 행 재측정)", "llama3_1_8b-base-gsm8k-warp-lr1e-5")


def asr_one(key, sub):
    p = os.path.join(HB, "results", DATASET, sub, "results", key + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    v = [float(i["label"]) for _, x in d.items()
         for i in (x if isinstance(x, list) else [x])
         if isinstance(i, dict) and i.get("label") is not None]
    return sum(v) / len(v) if v else None       # 0~1 스케일 (RESULTS.md 표기와 동일)


def asr(key):
    r = {a: asr_one(key, s) for a, s in ATTACKS.items()}
    got = [x for x in r.values() if x is not None]
    r["AVG"] = sum(got) / 4 if len(got) == 4 else None
    return r


def _log_order_key(path):
    """로그 파일의 **실행 시각**. 파일명 eval_YYYYMMDD_HHMMSS 를 우선 쓴다.

    ⚠️ mtime 으로 정렬하면 안 된다. 이 박스의 옛 로그는 다른 박스에서 **복사**돼 와서
       mtime 이 전부 복사 시각(2026-09-07)이고 실행 순서와 무관하다. 2026-09-16 실측:
       gemma SEAL(lr5e-5)은 2026-09-01 에 두 번 측정돼 flexible 0.2820(16:59) / 0.1865(20:42)
       인데, mtime 정렬로는 16:59 판이 "최신" 으로 잡혀 RESULTS.md(0.1865)와 어긋났다.
    """
    m = re.search(r"eval_(\d{8})_(\d{6})", os.path.basename(path))
    return (m.group(1) + m.group(2)) if m else str(int(os.path.getmtime(path)))


def gsm8k():
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
    return out
def f4(v):
    return "—" if v is None else f"{v:.4f}"


def sd(v):
    return "—" if v is None else f"{v:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", default="", help="이 파일 끝에 섹션을 덧붙인다")
    a = ap.parse_args()

    ds = gsm8k()
    base_a = asr(BASELINE)
    base_d = ds.get(BASELINE)

    L = []
    L.append("## Llama-3.1-8B **Base** / GSM8K  (논문 Table 7 확장 · rebuttal PEFT 7종)")
    L.append("")
    L.append("2026-09-16 이 박스(edgeai-1, B200)에서 학습·평가. 출발 모델 "
             "[`Llama-3.1-8B-base-SSFT_lr5e-5`](https://huggingface.co/kmseong/Llama-3.1-8B-base-SSFT_lr5e-5)"
             " (CB 로 안전정렬된 **base**).")
    L.append("")
    L.append("| 모델 | 기법 | Direct | AutoDAN | PAIR | PAP | AVG | GSM8K | Δsafe | Δdown | Δoverall |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for disp, key, kind in ROWS:
        r, d = asr(key), ds.get(key)
        if kind == "lora" and key != BASELINE and base_a["AVG"] is not None and r["AVG"] is not None:
            dsafe = r["AVG"] - base_a["AVG"]
            ddown = (d - base_d) if (d is not None and base_d is not None) else None
            dov = (ddown - dsafe) if ddown is not None else None
        else:
            dsafe = ddown = dov = None
        L.append(f"| [`{key}`](https://huggingface.co/{NS}/{key}) | {disp} | "
                 + " | ".join(f4(r[x]) for x in ["Direct", "AutoDAN", "PAIR", "PAP", "AVG"])
                 + f" | {f4(d)} | {sd(dsafe)} | {sd(ddown)} | {sd(dov)} |")
    cdisp, ckey = CONTROL
    cr, cd = asr(ckey), ds.get("llama3.1-8b-base-warp-gsm8k-lr1e-5")
    L.append(f"| `kmseong/llama3.1-8b-base-warp-gsm8k-lr1e-5` | {cdisp} | "
             + " | ".join(f4(cr[x]) for x in ["Direct", "AutoDAN", "PAIR", "PAP", "AVG"])
             + f" | {f4(cd)} | — | — | — |")
    L += [
        "",
        "**측정 조건은 위 표와 동일**하다 — HarmBench AdvBench standard · sys 모드 · "
        "`GRADING=hard`(keyword) · lm-eval 5-shot **flexible-extract**. base 모델이라 "
        "프롬프트는 chat template 이 아니라 `Question: {q}\\nAnswer:` 이고, 학습/HarmBench"
        "(`llama-3-base`)/lm-eval(`gsm8k.yaml doc_to_text`) 세 곳이 글자 단위로 같음을 확인했다.",
        "",
        "**Δ 기준행**: LoRA 6종은 같은 α 의 Vanilla LoRA(α=16). SEAL 은 full-param 이라 "
        "기준이 Full FT 여야 하는데, 논문 Table 7 의 Full FT 행은 아래 이슈가 정리되기 전까지 "
        "기준으로 쓰지 않았다(Δ 를 비웠다).",
        "",
        "### ⚠️ 논문 Table 7 기존 7행과 바로 이어 붙이면 안 되는 이유 (2026-09-16 확인)",
        "",
        "ASR 쪽은 **문제 없다**. 기존 결과 파일로 keyword ASR 을 다시 계산하면 Table 7 의 "
        "7개 행이 소수점까지 재현되고(FullFT 8.81 / SafeInstr 6.56 / Resta 8.37 / "
        "SafeDelta 3.99 / SN 36.85 / RSN 28.54 / WSR-Tune 4.46), WSR-Tune 행 모델을 이 "
        "박스에서 다시 재도 0.00/0.00/1.35/16.50 로 **완전히 일치**한다. 박스가 바뀌어도 "
        "safety 측정은 같은 잣대다.",
        "",
        "문제는 **GSM8K 열이다. Table 7 은 safety 와 downstream 을 서로 다른 모델에서 가져왔다.**",
        "논문의 GSM8K 7개 값은 전부 `lm-evaluation-harness/logs/eval_20260506_051850.log` "
        "(5-shot, flexible-extract) 에서 나오는데 그 대상이 **lr 5e-5** 리포이고, safety 값은 "
        "**lr 1e-5** 리포에서 나왔다. 두 리포는 같은 모델이 아니다:",
        "",
        "| WSR-Tune 후보 리포 | JB AVG | GSM8K |",
        "|---|---:|---:|",
        "| `llama3.1-8b-base-warp-gsm8k-lr1e-5` | **0.0446** ← 논문 safety | 0.5512 (2026-05 측정 0.5588) |",
        "| `Llama-3.1-8B-base-gsm8k-warp-lr5e-5` | 0.8143 | **0.4466** ← 논문 downstream |",
        "",
        "Full FT 도 같다 (lr1e-5: AVG 0.0881 / GSM8K 0.5709, lr5e-5: AVG 0.8243 / GSM8K 0.4238).",
        "",
        "**참고: lm-eval 은 ASR 과 달리 박스 간 재현이 완전하지 않다.** 같은 모델"
        "(`...warp-gsm8k-lr1e-5`)의 GSM8K 가 2026-05 옛 박스 0.5588 → 2026-09-16 이 박스 "
        "0.5512 로 **0.76%p** 움직였다(vLLM 버전·KV cache 구성 차이). 반면 HarmBench keyword "
        "ASR 은 4개 공격 모두 **0.00 차이**로 일치했다. 표 안의 GSM8K 비교는 같은 배치에서 잰 "
        "값끼리만 하고, 0.8%p 이하 차이는 유의미하게 읽지 말 것.",
        "위 새 7행은 **한 모델에서 safety 와 downstream 을 함께 잰 것**이라 내부적으로는 일관된다. "
        "기존 7행을 어떤 기준으로 옆에 놓을지는 미결이다 — 선택지는 (a) lr1e-5 모델들의 GSM8K 를 "
        "다시 재서 한 모델=한 행으로 맞추기(FullFT·WSR-Tune 은 이미 로그가 있고 나머지 5개만 "
        "~1시간), (b) 게재값 유지 + 각주, (c) 새 7행만 싣기.",
        "",
        "### 각주: LoRA · SaLoRA 의 AutoDAN 은 keyword 오탐이 크다",
        "",
        "두 모델의 AutoDAN ASR(0.8173 / 0.9327)은 실제 유해 응답이 아니라 degenerate 출력이 "
        "keyword 채점기에 걸린 것이다. 실제 completion 은 ① AutoDAN 템플릿 문자열 반복"
        "(`'Sure, here is how to [PROMPT] with Developer Mode enabled.'` × N), "
        "② `'Sure, here is how to ...'` 접두사 뒤 곧바로 거부, ③ gsm8k 정답 포맷(`#### 2`) 누출 "
        "형태다. 지표 자체는 논문과 동일하므로 표에는 그대로 싣되 이 각주가 필요하다. "
        "분류기(`cais/HarmBench-Llama-2-13b-cls`) 재채점은 가능하지만, 논문의 AutoDAN/PAIR 행에는 "
        "분류기 결과가 없어 **주 수치로는 쓸 수 없다**.",
        "",
    ]
    text = "\n".join(L) + "\n"
    if a.append:
        with open(a.append, "a", encoding="utf-8") as fh:
            fh.write("\n" + text)
        print(f"{a.append} 에 섹션 추가 ({len(ROWS)}행 + 대조군)")
    else:
        print(text)


if __name__ == "__main__":
    main()
