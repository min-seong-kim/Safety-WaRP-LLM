import torch, glob, os, math, collections
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
def load(p):
    if not os.path.isdir(p): p=snapshot_download(p, allow_patterns=["*.safetensors","*.json"])
    sd={}
    for f in sorted(glob.glob(os.path.join(p,"*.safetensors"))): sd.update(load_file(f))
    return sd
ali=load("kmseong/llama3_2_3b-instruct-SSFT-lr5e-5")
M={"orig":load("kmseong/llama3_2_3b-instruct-CB_SSFT-safelora_math_thr0.35_a16_lr3e-4"),
   "repro1(tf5.13)":load("outputs/revision_repro/cb/llama32_3b/math/safelora/merged_model"),
   "repro2(tf4.57)":load("outputs/revision_repro_hb457/cb/llama32_3b/math/safelora/merged_model"),
   "repro3(tf4.57 rerun)":load("outputs/revision_repro_hb457_run2/cb/llama32_3b/math/safelora/merged_model")}
keys=[k for k in ali if "proj" in k]
D={n:{k:(m[k].float()-ali[k].float()).flatten() for k in keys} for n,m in M.items()}
norm={n:math.sqrt(sum(d.pow(2).sum().item() for d in D[n].values())) for n in D}
print("‖Δ‖:", {n:round(v,3) for n,v in norm.items()})
names=list(D)
print(f"{'pair':32s} {'cos':>7s} {'‖Δa−Δb‖/‖Δa‖':>14s}")
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a,b=names[i],names[j]
        num=sum((D[a][k]*D[b][k]).sum().item() for k in keys)
        dd=math.sqrt(sum((D[a][k]-D[b][k]).pow(2).sum().item() for k in keys))
        print(f"{a+' vs '+b:32s} {num/(norm[a]*norm[b]):7.3f} {dd/norm[a]:14.3f}")
# per-module cos
mods=["q_proj","k_proj","v_proj","up_proj","down_proj"]
print("\nper-module cos:", " | ".join(f"{a} vs {b}" for i,a in enumerate(names) for b in names[i+1:]))
for mod in mods:
    ks=[k for k in keys if mod in k]; row=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            num=sum((D[a][k]*D[b][k]).sum().item() for k in ks); na=math.sqrt(sum(D[a][k].pow(2).sum().item() for k in ks)); nb=math.sqrt(sum(D[b][k].pow(2).sum().item() for k in ks))
            row.append(f"{num/(na*nb):.3f}")
    print(f"  {mod:10s} " + "  ".join(row))
