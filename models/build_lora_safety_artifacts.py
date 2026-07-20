"""Original-space safety importance → per-layer safe input-columns (original_projected_lora 용).

G_l^orig = Σ_{x∈D_safe} |∂L_safe/∂W_l|   (element-wise abs 누적)
s_{l,j}  = ‖G_l^orig[:, j]‖_2            (입력 좌표 j 의 열 점수)
safe_cols = top-k_l 열,  k_l = max(1, round(ρ · n_l))

저장: <out_dir>/<layer_type>/layer_NN_safecols.pt = {'safe_cols': LongTensor, 'n': int, 'k': int}
safety loss 는 refusal(llama3_output) 토큰에만, harmful prompt/pad 는 -100.
"""
import argparse
import json
import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_safe_cols")

_LAYER_TYPE_TO_ATTR = {
    "ffn_down": ("mlp", "down_proj"), "ffn_up": ("mlp", "up_proj"), "ffn_gate": ("mlp", "gate_proj"),
    "attn_q": ("self_attn", "q_proj"), "attn_k": ("self_attn", "k_proj"),
    "attn_v": ("self_attn", "v_proj"), "attn_o": ("self_attn", "o_proj"),
}


def _resolve_layers(num, spec):
    if spec == "all":
        return list(range(num))
    if "-" in spec:
        s, e = map(int, spec.split("-"))
        return list(range(s, e + 1))
    return [int(spec)]


def _tokenize_cb(example, tokenizer, max_length):
    prompt = str(example.get("prompt", "")).strip()
    response = str(example.get("llama3_output", "")).strip()
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False)
    except Exception:
        prompt_text = f"Question: {prompt}\nAnswer:"
        full_text = prompt_text + " " + response
    p_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    f_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = f_ids.copy()
    for i in range(min(len(p_ids), len(labels))):
        labels[i] = -100
    return f_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--safety_data_path", default="./data/circuit_breakers_train.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer_type", default="attn_q,attn_k,attn_v,ffn_up,ffn_down")
    ap.add_argument("--target_layers", default="all")
    ap.add_argument("--direction_keep_ratio", type=float, default=0.1)
    ap.add_argument("--samples", type=int, default=4994)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map={"": 0})
    model.eval()

    layer_types = [x.strip() for x in args.layer_type.split(",")]
    indices = _resolve_layers(len(model.model.layers), args.target_layers)

    # 타겟 weight 만 requires_grad
    targets = {}  # (layer_idx, lt) -> weight Parameter
    for p in model.parameters():
        p.requires_grad_(False)
    for li in indices:
        layer = model.model.layers[li]
        for lt in layer_types:
            pn, attr = _LAYER_TYPE_TO_ATTR[lt]
            w = getattr(getattr(layer, pn), attr).weight
            w.requires_grad_(True)
            targets[(li, lt)] = w
    logger.info(f"Target weights: {len(targets)}")

    data = json.load(open(args.safety_data_path))
    if args.samples > 0:
        data = data[: args.samples]
    logger.info(f"Safety samples: {len(data)}")

    # importance 를 GPU 에 누적 (매 배치 CPU 전송 = 17GB/batch 로 극도로 느림 → GPU 유지)
    importance = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in targets.items()}

    def collate(batch):
        toks = [_tokenize_cb(e, tok, args.max_length) for e in batch]
        maxlen = max(len(f) for f, _ in toks)
        pad = tok.pad_token_id
        ii, am, lb = [], [], []
        for f, l in toks:
            n = maxlen - len(f)
            ii.append(f + [pad] * n); am.append([1] * len(f) + [0] * n); lb.append(l + [-100] * n)
        return (torch.tensor(ii), torch.tensor(am), torch.tensor(lb))

    from torch.utils.data import DataLoader
    dl = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    n_batches = 0
    for ii, am, lb in dl:
        ii, am, lb = ii.to(0), am.to(0), lb.to(0)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ii, attention_mask=am, labels=lb)
        if not torch.isfinite(out.loss):
            continue
        out.loss.backward()
        for k, w in targets.items():
            if w.grad is not None:
                importance[k] += w.grad.detach().abs().float()  # GPU 누적
        model.zero_grad(set_to_none=True)
        n_batches += 1
        if n_batches % 50 == 0:
            logger.info(f"  processed {n_batches} batches")
    logger.info(f"Accumulated over {n_batches} batches")

    os.makedirs(args.out_dir, exist_ok=True)
    rho = args.direction_keep_ratio
    for (li, lt), G in importance.items():
        n = G.shape[1]
        col_score = G.norm(dim=0)  # ‖G[:,j]‖_2  (len n)
        k = max(1, round(rho * n))
        safe_cols = torch.topk(col_score, k).indices.sort().values.cpu()
        d = os.path.join(args.out_dir, lt)
        os.makedirs(d, exist_ok=True)
        torch.save({"safe_cols": safe_cols, "n": n, "k": k},
                   os.path.join(d, f"layer_{li:02d}_safecols.pt"))
    meta = {"model_name": args.model_name, "layer_type": args.layer_type,
            "target_layers": args.target_layers, "direction_keep_ratio": rho,
            "samples": len(data), "n_batches": n_batches}
    json.dump(meta, open(os.path.join(args.out_dir, "metadata.json"), "w"), indent=2)
    logger.info(f"✓ Saved safe_cols to {args.out_dir}")


if __name__ == "__main__":
    main()
