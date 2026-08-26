#!/usr/bin/env python3
"""WSR-LoRA — the PEFT-LoRA instantiation of WSR-Tune (factor-level safety mask).

One module, ``PiSSAWSRLinear``, with a ``reparam`` flag selects the two variants:

  * PiSSAWSRLinear  (reparam=False):  no reparameterization.
        y = W_res x + B A x                          (W_res = W0 - B0 A0, frozen)
    PiSSA-init B,A; compute a per-ENTRY safety-importance mask directly on the
    factors B and A (gradient of the safety loss w.r.t. each factor, at init);
    freeze the top-rho important entries at their PiSSA-init value during GSM8K
    training. The update stays low-rank -> vanilla-LoRA cost (NO dense d_out x d_in
    materialization, unlike the product-mask WSR in pissa_wsr_lora.py).

  * WSR-LoRA       (reparam=True):   WaRP reparameterization (V=I).
        Ã = A U   ->   ΔW = B Ã Uᵀ                   y = W_res x + B Ã (Uᵀ x)
    Same recipe, but the mask is computed on the ROTATED factors (B, Ã), i.e. in
    the activation-covariance basis U — exactly mirroring WSR-Tune's "freeze the
    safety-important coefficients in the U basis", but at the factor level.
    Training runs in rotated coordinates; the forward's Uᵀ maps back to weight
    space automatically ( "original space로의 복귀" == the forward, NOT rotating
    the elementwise mask back — that would make it dense ).

Only the factor GRADIENT is masked (frozen entries never update, so they stay at
their safety-aligned PiSSA init). The forward always uses the full factors, so the
frozen entries keep contributing their init value. This is an entry-wise *affine*
freeze on the low-rank factors — mechanistically NOT a subspace projection, hence
distinct from ActSVD/SaLoRA (both are P_out·(BA)·P_in projections).

Reuses SafetyDataset / GSM8KDataset / collate / topmask from pissa_wsr_lora.py
so the environment and safety/gsm8k data handling match exactly.  The rotated
variant can directly consume the exact Phase-1 SVD basis produced by legacy WaRP.
"""
import argparse
import gc
import json
import math
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from pissa_wsr_lora import (resolve_targets, is_target, topmask,
                            SafetyDataset, GSM8KDataset, collate)

TARGET_TO_BASIS = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}
MASK_CACHE_FORMAT = "wsrlora_factor_keep_v1"


# ───────────────────────── PiSSA-WSR module (factor mask) ──────────────────────────
class PiSSAWSRLinear(nn.Module):
    """PiSSA LoRA with a per-entry safety mask on the low-rank FACTORS.

    reparam=False:  y = W_res x + B  A  x        (A : r×d_in trainable)
    reparam=True :  y = W_res x + B  Ã (Uᵀ x)    (Ã = A U : r×d_in trainable)

    W_res = W0 - scaling*B0 A0 is frozen. Frozen factor entries (keep==0)
    get zero gradient and stay at their PiSSA init; the forward still uses them.
    """

    def __init__(self, weight, bias, r, alpha, dropout, U, dtype, reparam):
        super().__init__()
        self.reparam = bool(reparam)
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.rank = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        Wf = weight.detach().float()
        q = min(r + 8, min(Wf.shape))
        Uw, Sw, Vw = torch.svd_lowrank(Wf, q=q, niter=4)      # W ≈ Uw diag(Sw) Vwᵀ
        r = min(r, Sw.numel())
        if r != self.rank:
            raise ValueError(f"rank {self.rank} exceeds matrix rank bound {r}")
        # PEFT PiSSA convention: scaling*B0@A0 reconstructs the selected SVD
        # components, so singular values are divided by scaling before sqrt.
        s_sqrt = (Sw[:r].clamp_min(0) / self.scaling).sqrt()
        B0 = (Uw[:, :r] * s_sqrt.unsqueeze(0))                # d_out × r
        A0 = (s_sqrt.unsqueeze(1) * Vw[:, :r].t())            # r × d_in  (pre-rotation)
        W_res = Wf - self.scaling * (B0 @ A0)
        if self.reparam:
            if U is None:
                raise ValueError("rotated WSR-LoRA requires an input basis U")
            Uf = U.detach().to(device=Wf.device, dtype=torch.float32)
            if tuple(Uf.shape) != (Wf.shape[1], Wf.shape[1]):
                raise ValueError(
                    f"basis shape {tuple(Uf.shape)} does not match input dim {Wf.shape[1]}"
                )
            A_param0 = A0 @ Uf                                # Ã0 = A0 U
            self.register_buffer("U", Uf.to(dtype))
        else:
            A_param0 = A0
        self.register_buffer("W_res", W_res.to(dtype))
        self.register_buffer("bias", bias.detach().to(dtype) if bias is not None else None)
        self.B = nn.Parameter(B0.to(dtype).clone())
        self.A = nn.Parameter(A_param0.to(dtype).clone())     # A (base) or Ã (reparam)
        # keep = (1 - M); default keeps everything until set_masks() is called
        self.register_buffer("keep_B", torch.ones_like(self.B))
        self.register_buffer("keep_A", torch.ones_like(self.A))
        self._hooks_on = False

    def set_masks(self, keep_B, keep_A):
        """Install the factor keep-masks and gradient hooks (frozen entries -> grad 0)."""
        self.keep_B.copy_(keep_B.to(self.keep_B.dtype).to(self.keep_B.device))
        self.keep_A.copy_(keep_A.to(self.keep_A.dtype).to(self.keep_A.device))
        if not self._hooks_on:
            # hooks require grad-tracking tensors; B,A train from here on anyway
            self.B.requires_grad_(True)
            self.A.requires_grad_(True)
            self.B.register_hook(lambda g: g * self.keep_B.to(g.dtype))
            self.A.register_hook(lambda g: g * self.keep_A.to(g.dtype))
            self._hooks_on = True

    def forward(self, x):
        base = F.linear(x, self.W_res, self.bias)
        x = self.lora_dropout(x)
        if self.reparam:
            # Associate x U Ã^T as x (Ã U^T)^T.  This avoids a token-wise
            # d_in×d_in multiply while retaining gradients in rotated Ã.
            A_original = self.A @ self.U.t()
            delta = (x @ A_original.t()) @ self.B.t()
        else:
            delta = (x @ self.A.t()) @ self.B.t()             # x Aᵀ Bᵀ    ->  ΔW = B A
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self):
        if self.reparam:
            dW = self.B.float() @ self.A.float() @ self.U.float().t()   # B Ã Uᵀ
        else:
            dW = self.B.float() @ self.A.float()                        # B A
        return self.W_res.float() + self.scaling * dW


def _basis_location(basis_dir, module_name):
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    if match is None:
        raise ValueError(f"cannot infer Llama layer index from module name: {module_name}")
    suffix = module_name.rsplit(".", 1)[-1]
    basis_type = TARGET_TO_BASIS.get(suffix)
    if basis_type is None:
        raise ValueError(f"no legacy WaRP basis mapping for target module: {module_name}")
    layer_index = int(match.group(1))
    return Path(basis_dir) / basis_type / f"layer_{layer_index:02d}_svd.pt"


def validate_shared_basis(basis_dir, module_names, expected_samples):
    root = Path(basis_dir).resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing shared-basis metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("decomp") != "svd":
        raise ValueError(f"shared basis must use svd, got {metadata.get('decomp')!r}")
    if int(metadata.get("total_samples", -1)) != int(expected_samples):
        raise ValueError(
            f"shared basis used {metadata.get('total_samples')} samples; "
            f"expected {expected_samples}"
        )
    missing = [str(_basis_location(root, name)) for name in module_names
               if not _basis_location(root, name).is_file()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} shared-basis files are missing:\n{preview}")
    expected_layers = set(TARGET_TO_BASIS.values())
    actual_layers = set(metadata.get("layer_types", []))
    if not expected_layers.issubset(actual_layers):
        raise ValueError(
            f"shared basis layer_types={sorted(actual_layers)} do not cover "
            f"{sorted(expected_layers)}"
        )
    return root, metadata


def load_shared_U(basis_dir, module_name, input_dim):
    path = _basis_location(basis_dir, module_name)
    # mmap keeps the unused S and UT tensors from being faulted into RAM.  The
    # legacy files store both U and UT; WSR-LoRA needs only U.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "U" not in payload:
        raise KeyError(f"shared basis file has no U tensor: {path}")
    U = payload["U"]
    if tuple(U.shape) != (input_dim, input_dim):
        raise ValueError(
            f"{path}: U shape {tuple(U.shape)} does not match ({input_dim}, {input_dim})"
        )
    return U


def mask_cache_context(args, variant, basis_dir):
    return {
        "format": MASK_CACHE_FORMAT,
        "variant": variant,
        "model_name": str(Path(args.model_name).resolve()),
        "rank": args.rank,
        "alpha": args.alpha,
        "rho": args.rho,
        "mask_B": bool(args.mask_B),
        "mask_A": bool(args.mask_A),
        "safety_data": str(Path(args.safety_data).resolve()),
        "safety_samples": args.safety_samples,
        "basis_dir": str(Path(basis_dir).resolve()) if basis_dir else None,
        "seed": args.seed,
    }


def load_mask_cache(path, expected_context, wsr_mods):
    cache_path = Path(path)
    if not cache_path.is_file():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    context = payload.get("context")
    if context != expected_context:
        raise ValueError(
            f"mask cache provenance mismatch at {cache_path}\n"
            f"expected={expected_context}\nactual={context}"
        )
    masks = payload.get("masks", {})
    if set(masks) != set(wsr_mods):
        raise ValueError(
            f"mask cache module set mismatch: cache={len(masks)}, model={len(wsr_mods)}"
        )
    for name, module in wsr_mods.items():
        keep_B = masks[name]["keep_B"]
        keep_A = masks[name]["keep_A"]
        if tuple(keep_B.shape) != tuple(module.B.shape):
            raise ValueError(f"{name}: cached B mask shape mismatch")
        if tuple(keep_A.shape) != tuple(module.A.shape):
            raise ValueError(f"{name}: cached A mask shape mismatch")
        module.set_masks(keep_B, keep_A)
    return payload


def save_mask_cache(path, context, masks):
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"context": context, "masks": masks}, temporary)
    os.replace(temporary, cache_path)


# ───────────────────────── factor importance ──────────────────────────
def compute_factor_importance(model, wsr_mods, dl, device, max_batches):
    """S_B = Σ |∂L_safe/∂B|,  S_A = Σ |∂L_safe/∂A|  at the PiSSA init (elementwise).

    A is the reparameterized factor Ã when reparam=True, so S_A lives in the U basis.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for m in wsr_mods.values():
        m.B.requires_grad_(True)
        m.A.requires_grad_(True)
    SB = {n: torch.zeros_like(m.B, dtype=torch.float32) for n, m in wsr_mods.items()}
    SA = {n: torch.zeros_like(m.A, dtype=torch.float32) for n, m in wsr_mods.items()}
    model.eval()
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        out.loss.backward()
        with torch.no_grad():
            for n, m in wsr_mods.items():
                if m.B.grad is not None:
                    SB[n] += m.B.grad.float().abs()
                if m.A.grad is not None:
                    SA[n] += m.A.grad.float().abs()
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return SB, SA


# ───────────────────────── merge / save ──────────────────────────
@torch.no_grad()
def merge_and_save(model, wsr_mods, tok, output_dir, dtype):
    """Replace each PiSSAWSRLinear by a plain Linear carrying W_res + ΔW, then save."""
    module_names = list(wsr_mods)
    for index, name in enumerate(module_names, 1):
        m = wsr_mods.pop(name)
        merged = m.merged_weight().to(dtype)
        lin = nn.Linear(
            merged.shape[1],
            merged.shape[0],
            bias=(m.bias is not None),
            device=merged.device,
            dtype=dtype,
        )
        lin.weight = nn.Parameter(merged.contiguous(), requires_grad=False)
        if m.bias is not None:
            lin.bias = nn.Parameter(m.bias.detach().clone(), requires_grad=False)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], lin)
        del m, merged, lin
        if index % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    del module_names
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(output_dir)


# ───────────────────────── main ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="W_0 (SSFT checkpoint dir)")
    ap.add_argument(
        "--safety_data",
        default=None,
        help="safety dataset used for factor importance; required unless --no_freeze",
    )
    ap.add_argument("--gsm8k_json", default=None)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.1, help="fraction of factor entries frozen")
    ap.add_argument("--reparam", action="store_true", help="WSR-LoRA (rotate Ã=A U); else PiSSAWSRLinear base")
    ap.add_argument("--mask_B", type=int, default=1, help="1=mask factor B, 0=leave B fully trainable")
    ap.add_argument("--mask_A", type=int, default=1, help="1=mask factor A/Ã, 0=leave it fully trainable")
    ap.add_argument(
        "--no_freeze",
        action="store_true",
        help=(
            "PiSSA-LoRA baseline: skip safety-importance computation and leave "
            "both factors fully trainable (default: WSR-LoRA importance + freeze)"
        ),
    )
    ap.add_argument("--basis_dir", default=None,
                    help="legacy WaRP Phase-1 basis dir; required with --reparam")
    ap.add_argument("--mask_cache", default=None,
                    help="reuse one factor-importance mask across the LR sweep")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine"])
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--safety_samples", type=int, default=4994,
                    help="safety rows for factor importance")
    ap.add_argument("--basis_samples", type=int, default=4994,
                    help="expected rows used to create the shared Phase-1 basis")
    ap.add_argument("--safety_batches", type=int, default=0, help="0 = derive from safety_samples/batch")
    ap.add_argument("--basis_batch_size", type=int, default=2)
    ap.add_argument("--train_samples", type=int, default=0, help="0 = all gsm8k")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mask_B not in (0, 1) or args.mask_A not in (0, 1):
        ap.error("--mask_B and --mask_A must each be 0 or 1")
    freeze_enabled = not args.no_freeze and bool(args.mask_B or args.mask_A)
    effective_mask_B = freeze_enabled and bool(args.mask_B)
    effective_mask_A = freeze_enabled and bool(args.mask_A)
    if args.reparam and not args.basis_dir:
        ap.error("--basis_dir is required with --reparam so the legacy Phase-1 SVD is reused")
    if freeze_enabled and not args.safety_data:
        ap.error("--safety_data is required when WSR-LoRA importance/freeze is enabled")
    if not freeze_enabled and args.mask_cache:
        ap.error("--mask_cache cannot be used with --no_freeze or when both masks are disabled")
    if args.weight_decay != 0.0:
        ap.error("--weight_decay must be 0 so frozen factor entries remain exactly frozen")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        ap.error("--batch_size and --grad_accum must be positive")
    if freeze_enabled and args.safety_samples <= 0:
        ap.error("--safety_samples must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    device = "cuda"
    variant_key = "rotation" if args.reparam else "no_rotation"
    variant = "WSR-LoRA(rotation)" if args.reparam else "WSR-LoRA(no_rotation)"
    training_mode = "wsr_lora" if freeze_enabled else "pissa_lora_no_freeze"
    targets = resolve_targets(args.target_modules)
    effective_batch = args.batch_size * args.grad_accum
    print(
        f"[wsrlora] {variant} mode={training_mode} targets={targets} "
        f"r={args.rank} alpha={args.alpha} dropout={args.dropout} "
        f"rho={args.rho if freeze_enabled else 0.0} "
        f"mask_B={int(effective_mask_B)} mask_A={int(effective_mask_A)} "
        f"lr={args.lr} effective_batch={effective_batch} "
        f"W0={args.model_name}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False

    target_mods = {n: m for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and is_target(n, targets)}
    print(f"[wsrlora] {len(target_mods)} target Linear layers", flush=True)
    if not target_mods:
        raise RuntimeError("no target Linear layers found")

    shared_basis_metadata = None
    shared_basis_root = None
    if args.reparam:
        shared_basis_root, shared_basis_metadata = validate_shared_basis(
            args.basis_dir, target_mods, args.basis_samples
        )
        print(
            f"[wsrlora] reusing shared Phase-1 SVD basis -> {shared_basis_root} "
            f"(samples={shared_basis_metadata['total_samples']})",
            flush=True,
        )

    # wrap target Linears with the PiSSA-WSR factor module
    wsr_mods = {}
    target_names = list(target_mods)
    for index, name in enumerate(target_names, 1):
        mod = target_mods.pop(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        U = load_shared_U(shared_basis_root, name, mod.in_features) if args.reparam else None
        new = PiSSAWSRLinear(
            mod.weight.data,
            getattr(mod, "bias", None),
            args.rank,
            args.alpha,
            args.dropout,
            U,
            dtype,
            args.reparam,
        ).to(device)
        setattr(parent, child, new)
        wsr_mods[name] = new
        del mod, U
        if index % 16 == 0 or index == len(target_names):
            print(f"[wsrlora] PiSSA wrapped {index}/{len(target_names)} layers", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    del target_mods, target_names
    gc.collect()
    torch.cuda.empty_cache()

    if not freeze_enabled:
        cached = None
        freeze_off_reason = (
            "--no_freeze active" if args.no_freeze else "both factor masks disabled"
        )
        print(
            f"[wsrlora] {freeze_off_reason}: skipping safety dataset, factor "
            "importance, mask construction, and gradient-mask hooks",
            flush=True,
        )
    else:
        cache_context = mask_cache_context(args, variant_key, shared_basis_root)
        cached = (
            load_mask_cache(args.mask_cache, cache_context, wsr_mods)
            if args.mask_cache else None
        )
        if cached is not None:
            print(f"[wsrlora] loaded factor importance masks -> {args.mask_cache}", flush=True)
        else:
            safe_ds = SafetyDataset(args.safety_data, tok, args.max_length, args.safety_samples)
            coll = lambda batch: collate(batch, tok.pad_token_id)
            safe_dl = DataLoader(
                safe_ds,
                batch_size=args.basis_batch_size,
                shuffle=False,
                collate_fn=coll,
            )
            n_safe_batches = args.safety_batches or max(
                1, math.ceil(len(safe_ds) / args.basis_batch_size)
            )
            print(
                f"[wsrlora] computing factor safety importance on {len(safe_ds)} rows "
                f"({n_safe_batches} batches)...",
                flush=True,
            )
            SB, SA = compute_factor_importance(
                model, wsr_mods, safe_dl, device, n_safe_batches
            )
            masks_to_cache = {}
            for name, module in wsr_mods.items():
                keep_B = (
                    1.0 - topmask(SB[name], args.rho)
                    if effective_mask_B else torch.ones_like(SB[name])
                )
                keep_A = (
                    1.0 - topmask(SA[name], args.rho)
                    if effective_mask_A else torch.ones_like(SA[name])
                )
                module.set_masks(keep_B, keep_A)
                masks_to_cache[name] = {
                    "keep_B": keep_B.detach().to("cpu", dtype=torch.uint8),
                    "keep_A": keep_A.detach().to("cpu", dtype=torch.uint8),
                }
                del SB[name], SA[name], keep_B, keep_A
            if args.mask_cache:
                save_mask_cache(args.mask_cache, cache_context, masks_to_cache)
                print(f"[wsrlora] saved factor importance masks -> {args.mask_cache}", flush=True)
            del masks_to_cache, SB, SA, safe_dl, safe_ds
            gc.collect()
            torch.cuda.empty_cache()

    sample_module = next(iter(wsr_mods.values()))
    frac_report = {
        "B_frozen": 1.0 - sample_module.keep_B.float().mean().item(),
        "A_frozen": 1.0 - sample_module.keep_A.float().mean().item(),
    }
    print(f"[wsrlora] frozen fraction (sample layer): {frac_report}", flush=True)

    # only B,A of the wrapped modules train (W_res + everything else frozen)
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for m in wsr_mods.values():
        m.B.requires_grad_(True)
        m.A.requires_grad_(True)
        n_train += m.B.numel() + m.A.numel()
    print(f"[wsrlora] trainable params (B,A only): {n_train/1e6:.2f}M", flush=True)

    coll = lambda batch: collate(batch, tok.pad_token_id)
    train_ds = GSM8KDataset(tok, args.max_length, args.train_samples, args.gsm8k_json)
    print(f"[wsrlora] gsm8k train rows: {len(train_ds)}", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, logging_steps=10, save_strategy="no",
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
            cb = ol.maybe_make_callback(
                "wsrlora_rotation" if args.reparam else "wsrlora_no_rotation"
            )
            if cb is not None:
                callbacks.append(cb)
    except Exception as e:
        print(f"[overhead] wsrlora instrumentation skipped: {e}")

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=coll, processing_class=tok, callbacks=callbacks)
    trainer.train()

    print("[wsrlora] merging W_res + ΔW into a full model and saving (5GB shards)...", flush=True)
    merge_and_save(model, wsr_mods, tok, args.output_dir, dtype)
    run_metadata = {
        "method": "WSR-LoRA",
        "training_mode": training_mode,
        "variant": variant_key,
        "basis_rotation": bool(args.reparam),
        "model_name": args.model_name,
        "shared_basis_dir": str(shared_basis_root) if shared_basis_root else None,
        "shared_basis_metadata": shared_basis_metadata,
        "safety_data": str(Path(args.safety_data).resolve()) if args.safety_data else None,
        "safety_importance_samples": args.safety_samples if freeze_enabled else 0,
        "importance_skipped": not freeze_enabled,
        "downstream_dataset": Path(args.gsm8k_json).stem if args.gsm8k_json else "gsm8k",
        "gsm8k_json": str(Path(args.gsm8k_json).resolve()) if args.gsm8k_json else None,
        "target_modules": targets,
        "rank": args.rank,
        "alpha": args.alpha,
        "scaling": args.alpha / args.rank,
        "dropout": args.dropout,
        "freeze_enabled": freeze_enabled,
        "freeze_ratio": args.rho if freeze_enabled else 0.0,
        "configured_freeze_ratio": args.rho,
        "mask_B": effective_mask_B,
        "mask_A": effective_mask_A,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": effective_batch,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "seed": args.seed,
        "mask_cache": (
            str(Path(args.mask_cache).resolve())
            if freeze_enabled and args.mask_cache else None
        ),
        "trainable_factor_parameters": n_train,
    }
    metadata_path = Path(args.output_dir) / "wsrlora_run_config.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")
    print(f"[wsrlora] DONE -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
