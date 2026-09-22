#!/usr/bin/env python3
"""동결이 실제로 걸렸는지 잰다 (bf16 보정 포함).

원공간 이진 마스크면 mask=1 위치는 출발 모델과 bit-identical 이어야 한다. 그런데 bf16 은
가수가 8비트라 **학습된 파라미터도 상당수가 반올림으로 값이 그대로**다. 그래서 단순
일치율을 ρ 로 읽으면 안 되고, 마스크가 없는(mask=0) 파라미터의 일치율을 바닥값 r 로 삼아
    f = (관측 - r) / (1 - r)
로 보정한다. (RESULTS.md 의 chat 라인 검증과 같은 방법.)

사용:
  python scripts/revision/_check_freeze_drift.py <start_model> <tuned_model> <masks_dir> [--max-layers N]
"""
import argparse, glob, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

SUF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
       "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start"); ap.add_argument("tuned"); ap.add_argument("masks_dir")
    ap.add_argument("--max-layers", type=int, default=6)
    a = ap.parse_args()
    kw = dict(dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    s = AutoModelForCausalLM.from_pretrained(a.start, **kw).state_dict()
    t = AutoModelForCausalLM.from_pretrained(a.tuned, **kw).state_dict()
    tot_on = tot_on_same = tot_off = tot_off_same = 0
    files = sorted(glob.glob(os.path.join(a.masks_dir, "*", "*.pt")))
    per_layer = {}
    for f in files:
        mod = os.path.basename(os.path.dirname(f))
        li = int(os.path.basename(f).split("_")[1])
        if a.max_layers and li >= a.max_layers: continue
        key = "model.layers.%d.%s.weight" % (li, SUF[mod])
        if key not in s or key not in t: continue
        obj = torch.load(f, map_location="cpu", weights_only=False)
        m = obj.get("mask") if isinstance(obj, dict) else obj
        m = np.asarray(m).astype(bool)
        sv = s[key].float().numpy(); tv = t[key].float().numpy()
        if sv.shape != m.shape: continue
        same = (sv == tv)
        on, off = m, ~m
        tot_on += on.sum(); tot_on_same += (same & on).sum()
        tot_off += off.sum(); tot_off_same += (same & off).sum()
    if tot_on == 0: print("마스크 매칭 실패"); return
    p_on = tot_on_same / tot_on; r = tot_off_same / tot_off
    f_corr = (p_on - r) / (1 - r) if r < 1 else float("nan")
    print("mask=1 일치율 (관측)      : %.4f  (%d / %d)" % (p_on, tot_on_same, tot_on))
    print("mask=0 일치율 (bf16 바닥값): %.4f" % r)
    print("보정 동결률 f = (관측-r)/(1-r): %.4f" % f_corr)
    print("→ 복원 콜백이 있었다면 f≈1.0, gradient hook 만이면 f<1.0 (드리프트)")

if __name__ == "__main__":
    main()
