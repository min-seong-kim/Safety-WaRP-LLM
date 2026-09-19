#!/usr/bin/env python3
"""학습이 끝난 AsFT 모델의 벌점 Σ‖(I−Ĉ)ΔW‖²_F 를 저장된 가중치로 독립 계산한다.

러너가 로그에 남기는 '최종 벌점 값' 이 맞는지 대조하기 위한 것. CPU 로 돌려
학습 중인 GPU 를 건드리지 않는다.

  python scripts/revision/_check_asft_penalty.py --base <hf> --aligned <hf> --tuned <dir>
"""
import argparse, torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--aligned", required=True)
ap.add_argument("--tuned", required=True)
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,up_proj,down_proj")
a = ap.parse_args()
T = [x.strip() for x in a.targets.split(",")]

def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32,
                                                low_cpu_mem_usage=True, device_map="cpu")
print("loading base/aligned/tuned on CPU ...", flush=True)
mb, ma, mt = load(a.base), load(a.aligned), load(a.tuned)
db, da, dt = dict(mb.named_parameters()), dict(ma.named_parameters()), dict(mt.named_parameters())

total, n = 0.0, 0
for name in da:
    if not any(t in name for t in T) or da[name].ndim != 2:
        continue
    V = (da[name].detach() - db[name].detach())
    vnorm = torch.norm(V)
    dW = (dt[name].detach() - da[name].detach())
    Y = dW - (V @ (V.t() @ dW)) / vnorm            # (I−Ĉ)ΔW
    total += float((Y * Y).sum()); n += 1
print(f"대상 weight {n} 개")
print(f"Σ‖(I−Ĉ)ΔW‖²_F = {total:.6e}")
