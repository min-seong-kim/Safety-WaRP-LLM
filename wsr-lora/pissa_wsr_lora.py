#!/usr/bin/env python3
"""PiSSA-WSR-LoRA — WaRP/WSR's safety-subspace masking applied to a PiSSA LoRA.

Standalone. Fits the all.sh environment/params (same conda env `hb`, same
PHASE0_MODEL / SAFETY_DATA / GSM8K data), but is NOT wired into all.sh.

Method (V = I; U from safety activations):
  * W_0  : the model being fine-tuned (default = SSFT checkpoint).
  * PiSSA: per target Linear, SVD(W_0)=P Σ Qᵀ, B_0=P_r Σ_r^{1/2}, A_0=Σ_r^{1/2} Q_rᵀ.
           Keep W_0 frozen; train B_t,A_t (init B_0,A_0). Product delta
           D_t = B_tA_t − B_0A_0 (=0 at init).
  * U    : per layer, SVD of the input-activation Gram Σ x xᵀ over the safety set
           (this is exactly WaRP Phase-1's basis).  W̃ = Wᵀ… with V=I → W̃ = W U.
  * Mask : S = Σ_x |∂L_safe/∂W̃|_{W̃0} = Σ_x |(∂L_safe/∂W) U|  (elementwise).
           M = TopMask_ρ(S) freezes the top-ρ safety-important coordinates.
  * Train: effective rotated weight  W̃_t^eff = W̃_0 + (1−M)⊙D̃_t,  D̃_t = D_t U.
           Back in the original space:  W_eff = W_0 + [(1−M)⊙(D_t U)] Uᵀ, so
             y = W_0 x + [(1−M)⊙(D_t U)] (Uᵀ x).
           Masked coordinates satisfy M⊙(W̃_t^eff − W̃_0)=0 (safety preserved).

Elementwise mask ⇒ the dense d_out×d_in delta (D_t U) is materialized each step
(chosen deliberately; a column mask would keep it low-rank — see notes).
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

TARGET_ALIASES = {
    "attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj",
    "attn_o": "o_proj", "ffn_up": "up_proj", "ffn_down": "down_proj",
    "ffn_gate": "gate_proj",
}


def resolve_targets(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(TARGET_ALIASES.get(tok, tok))
    return out


def is_target(name, targets):
    return any(name.endswith(t) for t in targets)


# ───────────────────────── datasets ──────────────────────────
def _chat_ids(tokenizer, prompt, response, max_length):
    """Instruct chat-template tokenization with the prompt tokens masked out.

    Render the template to text first, then tokenize (add_special_tokens=False so
    the template's own BOS isn't duplicated) — robust across transformers versions
    where apply_chat_template(tokenize=True) may return a tokenizers.Encoding.
    """
    p_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False)
    p_ids = tokenizer(p_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    full = full[:max_length]
    labels = list(full)
    for i in range(min(len(p_ids), len(full))):
        labels[i] = -100
    return full, labels


class SafetyDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, max_samples):
        data = json.load(open(path))
        if max_samples:
            data = data[:max_samples]
        self.rows = [(d.get("prompt", ""), d.get("llama3_output", "")) for d in data]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, r = self.rows[i]
        ids, labels = _chat_ids(self.tok, p, r, self.max_length)
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, max_length, max_samples, json_path=None):
        if json_path and os.path.isfile(json_path):
            data = json.load(open(json_path))
            self.rows = [
                (d["question"], d.get("answer", d.get("response"))) for d in data
            ]
            if any(answer is None for _, answer in self.rows):
                raise KeyError("downstream JSON rows require answer or response")
        else:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="train")
            self.rows = [(r["question"], r["answer"]) for r in ds]
        if max_samples:
            self.rows = self.rows[:max_samples]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        q, a = self.rows[i]
        ids, labels = _chat_ids(self.tok, q, a, self.max_length)
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append(b["attention_mask"] + [0] * n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


# ───────────────────────── PiSSA-WSR module ──────────────────────────
class PissaWsrLinear(nn.Module):
    """y = W0 x + [(1-M) ⊙ ((B A − B0 A0) U)] (Uᵀ x)."""

    def __init__(self, weight, bias, r, U, keep_mask, dtype):
        super().__init__()
        d_out, d_in = weight.shape
        self.register_buffer("W0", weight.detach().to(dtype))
        self.register_buffer("bias", bias.detach().to(dtype) if bias is not None else None)
        # PiSSA init from W0: only the top-r singular components are needed, so
        # use randomized low-rank SVD (much faster than full SVD on big matrices).
        Wf = weight.detach().float()
        q = min(r + 8, min(Wf.shape))
        Uw, Sw, Vw = torch.svd_lowrank(Wf, q=q, niter=4)   # W ≈ Uw diag(Sw) Vwᵀ
        r = min(r, Sw.numel())
        s_sqrt = Sw[:r].clamp_min(0).sqrt()
        B0 = (Uw[:, :r] * s_sqrt.unsqueeze(0))            # d_out × r
        A0 = (s_sqrt.unsqueeze(1) * Vw[:, :r].t())         # r × d_in  (Vw is d_in×q)
        self.B = nn.Parameter(B0.to(dtype).clone())
        self.A = nn.Parameter(A0.to(dtype).clone())
        self.register_buffer("B0", B0.to(dtype))
        self.register_buffer("A0", A0.to(dtype))
        self.register_buffer("U", U.to(dtype))            # d_in × d_in
        self.register_buffer("A0U", (A0 @ U.float()).to(dtype))   # r × d_in (precomputed)
        self.register_buffer("keep", keep_mask.to(dtype))  # (1-M), d_out × d_in

    def forward(self, x):
        base = F.linear(x, self.W0, self.bias)
        AU = self.A @ self.U                               # r × d_in  (= A U)
        D_rot = self.B @ AU - self.B0 @ self.A0U           # d_out × d_in  (= (B A − B0 A0) U)
        masked = self.keep * D_rot                          # (1-M) ⊙ (D U)
        x_rot = torch.matmul(x, self.U)                    # (…, d_in)  (= Uᵀ x for row vecs)
        delta = torch.matmul(x_rot, masked.transpose(-1, -2))
        return base + delta


# ───────────────────────── U basis + importance ──────────────────────────
@torch.no_grad()
def compute_U_basis(model, target_layers, dl, device, max_batches):
    grams = {n: None for n in target_layers}
    handles = []

    def mk_hook(name):
        def hook(mod, inp):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            g = x.t() @ x
            grams[name] = g if grams[name] is None else grams[name] + g
        return hook

    for name, mod in model.named_modules():
        if name in target_layers:
            handles.append(mod.register_forward_pre_hook(mk_hook(name)))
    model.eval()
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    for h in handles:
        h.remove()
    Us = {}
    for name in list(grams.keys()):
        g = grams[name]
        if g is None:
            raise RuntimeError(f"no activations captured for {name}")
        # G = XᵀX is symmetric PSD → eigh (faster & exact than SVD). Eigenvectors
        # are the input principal directions (= right singular vectors of X).
        # eigh returns ascending eigenvalues; flip to descending (high-variance first).
        _, evecs = torch.linalg.eigh(g.float())
        Us[name] = evecs.flip(dims=[-1]).contiguous()  # d_in × d_in
        grams[name] = None
        del g
        torch.cuda.empty_cache()
    return Us


def compute_importance(model, target_mods, Us, dl, device, max_batches):
    """S_name = Σ_batch |(∂L_safe/∂W) U|  (elementwise), evaluated at W0."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in target_mods.items():
        mod.weight.requires_grad_(True)
    S = {name: torch.zeros_like(mod.weight, dtype=torch.float32)
         for name, mod in target_mods.items()}
    model.eval()
    n = 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        out.loss.backward()
        with torch.no_grad():
            for name, mod in target_mods.items():
                if mod.weight.grad is not None:
                    S[name] += (mod.weight.grad.float() @ Us[name].float()).abs()
        n += 1
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return S


def topmask(S, rho):
    """Binary mask M with the top-ρ fraction of |S| entries set to 1 (frozen)."""
    flat = S.flatten()
    k = int(rho * flat.numel())
    M = torch.zeros_like(flat)
    if k > 0:
        idx = torch.topk(flat.abs(), k).indices
        M[idx] = 1.0
    return M.view_as(S)


# ───────────────────────── main ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="W_0 (SSFT checkpoint dir)")
    ap.add_argument("--safety_data", required=True)
    ap.add_argument("--gsm8k_json", default=None, help="clean gsm8k {question,answer} json; else openai/gsm8k")
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.1, help="fraction of safety coords frozen")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--safety_samples", type=int, default=512, help="safety rows for U + importance")
    ap.add_argument("--safety_batches", type=int, default=0, help="0 = derive from safety_samples/batch")
    ap.add_argument("--train_samples", type=int, default=0, help="0 = all gsm8k")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--basis_batch_size", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    device = "cuda"
    targets = resolve_targets(args.target_modules)
    print(f"[pwsr] targets={targets} r={args.rank} rho={args.rho} W0={args.model_name}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False

    target_mods = {n: m for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and is_target(n, targets)}
    print(f"[pwsr] {len(target_mods)} target Linear layers", flush=True)

    # data for U + importance (safety) and training (gsm8k)
    safe_ds = SafetyDataset(args.safety_data, tok, args.max_length, args.safety_samples)
    from torch.utils.data import DataLoader
    coll = lambda b: collate(b, tok.pad_token_id)
    safe_dl = DataLoader(safe_ds, batch_size=args.basis_batch_size, shuffle=False, collate_fn=coll)
    n_safe_batches = args.safety_batches or max(1, math.ceil(len(safe_ds) / args.basis_batch_size))

    print("[pwsr] computing U (safety-activation basis)...", flush=True)
    Us = compute_U_basis(model, set(target_mods), safe_dl, device, n_safe_batches)
    print("[pwsr] computing elementwise safety importance S and masks M...", flush=True)
    S = compute_importance(model, target_mods, Us, safe_dl, device, n_safe_batches)
    masks = {n: topmask(S[n], args.rho) for n in target_mods}
    frozen_frac = {n: masks[n].mean().item() for n in list(masks)[:1]}
    print(f"[pwsr] mask frozen fraction (sample): {frozen_frac}", flush=True)

    # replace target Linears with PiSSA-WSR modules
    for name, mod in list(target_mods.items()):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        new = PissaWsrLinear(mod.weight.data, getattr(mod, "bias", None),
                             args.rank, Us[name], 1.0 - masks[name], dtype).to(device)
        setattr(parent, child, new)
        del Us[name]

    # freeze everything except B,A
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for n, p in model.named_parameters():
        if n.endswith(".B") or n.endswith(".A"):
            p.requires_grad_(True)
            n_train += p.numel()
    print(f"[pwsr] trainable params (B,A only): {n_train/1e6:.2f}M", flush=True)

    train_ds = GSM8KDataset(tok, args.max_length, args.train_samples, args.gsm8k_json)
    print(f"[pwsr] gsm8k train rows: {len(train_ds)}", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        weight_decay=0.0, logging_steps=10, save_strategy="no",
        bf16=(args.dtype == "bfloat16"), seed=args.seed, report_to=[],
        gradient_checkpointing=False, remove_unused_columns=False,
    )
    callbacks = []
    try:
        if os.environ.get("OVERHEAD_LOG_DIR"):
            import importlib.util as ilu
            root = os.environ.get("WARP_SAFETY_ROOT", os.getcwd())
            spec = ilu.spec_from_file_location("overhead_logger", os.path.join(root, "overhead_logger.py"))
            ol = ilu.module_from_spec(spec); spec.loader.exec_module(ol)
            cb = ol.maybe_make_callback("pissa_wsr")
            if cb is not None:
                callbacks.append(cb)
    except Exception as e:
        print(f"[overhead] pwsr instrumentation skipped: {e}")

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=coll, processing_class=tok, callbacks=callbacks)
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    adapter = {n: p.detach().cpu() for n, p in model.named_parameters()
               if n.endswith(".B") or n.endswith(".A")}
    torch.save({"adapter": adapter,
                "masks": {n: masks[n].to(torch.bool).cpu() for n in masks},
                "config": vars(args)},
               os.path.join(args.output_dir, "pissa_wsr_adapter.pt"))
    print(f"[pwsr] saved adapter+masks -> {args.output_dir}/pissa_wsr_adapter.pt", flush=True)


if __name__ == "__main__":
    main()
