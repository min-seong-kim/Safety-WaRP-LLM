import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-3B"  # 또는 meta-llama/Llama-3.2-3B-Instruct

def num_params(module):
    return sum(p.numel() for p in module.parameters())

def dedup_total_params(model):
    # tied weight(공유 파라미터) 중복 카운트 방지
    seen = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total

def fmt(n):
    return f"{n:,} ({n/1e6:.2f}M)"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

# ===== 전체 파라미터 =====
total_raw = sum(p.numel() for p in model.parameters())
total_dedup = dedup_total_params(model)

print("=" * 90)
print(f"Model: {MODEL_ID}")
print(f"Total params (raw):   {fmt(total_raw)}")
print(f"Total params (dedup): {fmt(total_dedup)}")
print("=" * 90)

# ===== 상위 블록별 =====
emb = num_params(model.model.embed_tokens)
lm_head = num_params(model.lm_head) if hasattr(model, "lm_head") else 0
final_norm = num_params(model.model.norm) if hasattr(model.model, "norm") else 0
layers_total = sum(num_params(layer) for layer in model.model.layers)

print("[Top-level breakdown]")
for name, v in [
    ("embed_tokens", emb),
    ("layers_total", layers_total),
    ("final_norm", final_norm),
    ("lm_head", lm_head),
]:
    print(f"{name:15s}: {fmt(v):>20s}  ({100*v/total_dedup:6.2f}%)")

print("-" * 90)

# ===== 레이어 타입별 합계 =====
sum_q = sum_k = sum_v = sum_o = 0
sum_gate = sum_up = sum_down = 0
sum_in_ln = sum_post_ln = 0

for layer in model.model.layers:
    attn = layer.self_attn
    mlp = layer.mlp

    sum_q += attn.q_proj.weight.numel()
    sum_k += attn.k_proj.weight.numel()
    sum_v += attn.v_proj.weight.numel()
    sum_o += attn.o_proj.weight.numel()

    sum_gate += mlp.gate_proj.weight.numel()
    sum_up += mlp.up_proj.weight.numel()
    sum_down += mlp.down_proj.weight.numel()

    sum_in_ln += sum(p.numel() for p in layer.input_layernorm.parameters())
    sum_post_ln += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

print("[Layer-type totals across all transformer layers]")
for name, v in [
    ("attn_q_proj", sum_q),
    ("attn_k_proj", sum_k),
    ("attn_v_proj", sum_v),
    ("attn_o_proj", sum_o),
    ("mlp_gate_proj", sum_gate),
    ("mlp_up_proj", sum_up),
    ("mlp_down_proj", sum_down),
    ("input_layernorm", sum_in_ln),
    ("post_attn_layernorm", sum_post_ln),
]:
    print(f"{name:20s}: {fmt(v):>20s}  ({100*v/total_dedup:6.2f}%)")

print("-" * 90)

# ===== 레이어별 상세 =====
print("[Per-layer detail]")
header = (
    f"{'L':>2} | {'q_proj':>10} {'k_proj':>10} {'v_proj':>10} {'o_proj':>10} | "
    f"{'gate':>10} {'up':>10} {'down':>10} | {'layer_total':>12} {'%total':>8}"
)
print(header)
print("-" * len(header))

for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    mlp = layer.mlp

    q = attn.q_proj.weight.numel()
    k = attn.k_proj.weight.numel()
    v = attn.v_proj.weight.numel()
    o = attn.o_proj.weight.numel()

    g = mlp.gate_proj.weight.numel()
    u = mlp.up_proj.weight.numel()
    d = mlp.down_proj.weight.numel()

    ln = sum(p.numel() for p in layer.input_layernorm.parameters())
    pln = sum(p.numel() for p in layer.post_attention_layernorm.parameters())

    layer_total = q + k + v + o + g + u + d + ln + pln

    print(
        f"{i:2d} | "
        f"{q/1e6:10.2f} {k/1e6:10.2f} {v/1e6:10.2f} {o/1e6:10.2f} | "
        f"{g/1e6:10.2f} {u/1e6:10.2f} {d/1e6:10.2f} | "
        f"{layer_total/1e6:12.2f} {100*layer_total/total_dedup:8.2f}"
    )