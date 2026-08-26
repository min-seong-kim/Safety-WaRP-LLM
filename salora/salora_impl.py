"""SaLoRA — Safety-alignment preserved Low-Rank Adaptation (Li et al., ICLR 2025) ported to
federated SFT.

Unlike the server-side safety methods in this package (safe_lora / agc / fedgrasp / ...),
which transform the FedAvg aggregate every round, SaLoRA is a *training-time LoRA
reparameterization* applied ONCE before the round loop. base.py explicitly notes that such
methods "do not fit" the SafetyMethod hook interface, so SaLoRA is NOT registered in
``SAFETY_METHODS`` and is driven by its own entry point (``main_salora.py``); it reuses the
shared FL data / aggregation / trainer / eval infrastructure unchanged.

Method (per target Linear module with weight W ∈ R^{d_out × d_in}, paper §4):

  * Fixed safety module  C_S = I − U_C U_Cᵀ , where U_C = top-r_s left singular vectors of
    W X_h and X_h are the input features of *harmful prompts paired with their safe
    responses* (here circuit_breakers: prompt + the refusal field). C_S projects the adapter's
    contribution onto the subspace orthogonal to the model's safety features, preserving them.
    We never materialize the d_out×d_out matrix C_S: U_C (d_out × r_s) is stored and the
    projection is applied as  C_S y = y − U_C (U_Cᵀ y)  (idempotent; U_C orthonormal).

  * Task-specific init (paper eqs 10–12):  with U = top-r_t left singular vectors of W X_t
    (X_t = task/downstream input features) and W = Ū S̄ V̄ᵀ,
        B_S = U Uᵀ Ū_{[:r]} √S̄_{[:r]}            (d_out × r)
        A_S = √S̄_{[:r]} V̄_{[:r]}ᵀ               (r × d_in)

  * Residual reparameterization (paper eq 13):  W' = W − s·C_S B_S A_S  (s = LoRA scaling),
    written into the frozen base weight so the model reproduces its pretrained output at init.

Forward during training (per PEFT's  result += scaling·lora_B(lora_A(x))): a forward hook on
each ``lora_B`` left-projects its output by C_S, giving the effective update s·C_S B A. A/B
train as ordinary LoRA factors; C_S and W' are frozen and identical across clients, so plain
FedAvg over A/B is exact.

Saving / eval. The trained adapter merges as W' + s·C_S B A, which standard PEFT merge would
get wrong (it omits C_S and would target the original W, not W'). Two correct save paths:

  * Retained per-round adapters → a rank-2r LoRA on the ORIGINAL base W (no W' persistence):
        A₂ = [A ; A₀],  B₂ = [C_S B , −C_S B₀],  r→2r,  alpha→2·alpha,
    so merging onto W gives  W + s·C_S(BA − B₀A₀) = W' + s·C_S B A. The existing
    merge-then-eval pipeline (which reads the base id from adapter_config) then works unchanged.
  * Final merged model → bake C_S into the live ``lora_B`` (B ← C_S B), drop the hooks, and
    ``merge_and_unload`` over the live W' base.
"""

import json
import math
import os
import random
from pathlib import Path

import torch

# Data helpers ported from FLS fsl/src/sft/data.py to this repo's pipeline. Identical
# behaviour: chat-template tokenization with prompt tokens masked out (labels=-100),
# and padding of input_ids/attention_mask/labels — implemented by pissa_wsr_lora's
# _chat_ids and collate (same as build_sft_rows / ResponseOnlyDataCollator).
from pissa_wsr_lora import _chat_ids
from pissa_wsr_lora import collate as _collate


def build_rows_from_records(records, tokenizer, model_name=None, max_length=1024):
    """[{prompt, output/response}] -> [{input_ids, attention_mask, labels}] (response-only)."""
    rows = []
    for r in records:
        prompt = r.get("prompt", "")
        response = r.get("output") or r.get("response") or r.get("llama3_output") or ""
        ids, labels = _chat_ids(tokenizer, str(prompt), str(response), max_length)
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return rows


class ResponseOnlyDataCollator:
    """Pads input_ids (pad_id), attention_mask (0), labels (-100). Right-padding; the
    label!=-100 mask still selects the correct response tokens for activation capture."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        return _collate(features, self.pad_id)


def _is_lora_layer(module):
    la = getattr(module, "lora_A", None)
    return la is not None and "default" in la


def _module_name_from_key(key):
    """``...q_proj.lora_A.weight`` -> ``...q_proj`` (the named_modules() name)."""
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return None, None


class _InputCapture:
    """Forward hooks on each target module's ``base_layer`` recording its *input* features at
    response-token positions (labels != -100), matching the disentangled activation capture of
    the reference SaLoRA implementation. Inputs (not outputs) are recorded so W X is formed from
    the original weight and any module bias is excluded; tokens per module are capped to bound
    memory and the records are kept on CPU (fp16)."""

    def __init__(self, model, module_names, max_tokens):
        self.max_tokens = int(max_tokens)
        self.buffers = {name: [] for name in module_names}
        self.counts = {name: 0 for name in module_names}
        self._mask = None
        self._handles = []
        targets = set(module_names)
        for name, module in model.named_modules():
            if name in targets:
                self._handles.append(
                    module.base_layer.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name):
        def hook(_module, inputs, _output):
            if self.counts[name] >= self.max_tokens:
                return
            x = inputs[0]
            if x.dim() == 3:
                x = x[self._mask] if self._mask is not None else x.reshape(-1, x.shape[-1])
            elif x.dim() != 2:
                x = x.reshape(-1, x.shape[-1])
            remaining = self.max_tokens - self.counts[name]
            if x.shape[0] > remaining:
                x = x[:remaining]
            if x.shape[0] > 0:
                self.buffers[name].append(x.detach().to("cpu", dtype=torch.float16))
                self.counts[name] += x.shape[0]

        return hook

    def set_mask(self, mask):
        self._mask = mask

    def get(self, name):
        chunks = self.buffers.get(name)
        return torch.cat(chunks, dim=0) if chunks else None

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class SaLoRA:
    """Builds and owns the SaLoRA reparameterization over a LoRA-wrapped model."""

    name = "salora"

    def __init__(self, args):
        self.args = args
        self.r_s = int(getattr(args, "salora_r_s", 32))
        self.r_t = int(getattr(args, "salora_r_t", 32))
        # init_mode selects the LoRA initialization. C_S (the safety module) and its forward
        # projection apply in ALL modes — only the trainable A/B starting point differs:
        #   "task"    -> (a) full SaLoRA: PiSSA init projected onto the task subspace U Uᵀ
        #                   (paper eq 12). Needs task features X_t -> reads TRAINING data.
        #   "pissa"   -> (b) PiSSA init, no task projection (paper §5.4 ablation "w.o our init").
        #                   Uses only W's SVD -> no training-data access (FL-safe). [default]
        #   "vanilla" -> standard LoRA init (A ~ kaiming, B = 0): no PiSSA, no residual reparam
        #                   (W' = W); C_S still applied in the forward. The weakest init the paper
        #                   warns against (§4.2), kept as an ablation.
        # DEFAULT "pissa": "task" would have the server read client TRAINING data (an FL
        # violation); pissa/vanilla touch only W (+ the public circuit_breakers for C_S).
        self.init_mode = str(getattr(args, "salora_init_mode", "pissa")).strip().lower()
        if self.init_mode not in ("task", "pissa", "vanilla"):
            raise ValueError(f"salora_init_mode must be task|pissa|vanilla, got {self.init_mode!r}")
        self.n_harmful = int(getattr(args, "salora_n_harmful", 256))
        self.n_task = int(getattr(args, "salora_n_task", 256))
        self.max_tokens = int(getattr(args, "salora_max_tokens", 4096))
        self.niter = int(getattr(args, "salora_svd_niter", 10))
        self.harmful_path = getattr(args, "salora_harmful_path", None)
        self.response_field = getattr(args, "salora_response_field", "llama3_output")
        self.batch_size = int(getattr(args, "salora_capture_batch_size", 4))
        # Per-module artifacts (keyed by named_modules() name):
        #   U_C  (d_out × r_s, fp32, GPU) — safety basis, used by the projection hook + saving
        #   B0/A0 (fp32, CPU)            — task-init factors, the −C_S B₀A₀ anchor for rank-2r save
        self.modules = {}
        self._proj_handles = []
        self._uc_cache = {}  # name -> {device,dtype: cast U_C} for the forward hook

    # ----------------------------------------------------------------- data ----
    def _load_harmful_records(self):
        with open(self.harmful_path, encoding="utf-8") as handle:
            data = json.load(handle)
        rng = random.Random(self.args.seed)
        if len(data) > self.n_harmful:
            data = [data[i] for i in sorted(rng.sample(range(len(data)), self.n_harmful))]
        records = []
        for row in data:
            prompt = row.get("prompt")
            response = row.get(self.response_field)
            if prompt is None or response is None:
                continue
            # _build_user_content reads "prompt"; _response_text reads "output".
            records.append({"prompt": str(prompt), "output": str(response)})
        if not records:
            raise RuntimeError(
                f"salora: no harmful records with fields prompt/{self.response_field} "
                f"in {self.harmful_path}"
            )
        return records

    def _rows(self, records, n=None):
        if n is not None and len(records) > n:
            rng = random.Random(self.args.seed)
            records = [records[i] for i in sorted(rng.sample(range(len(records)), n))]
        return build_rows_from_records(
            records,
            tokenizer=self.args._tokenizer,
            model_name=self.args.model_name,
            max_length=self.args.max_length,
        )

    # -------------------------------------------------------------- capture ----
    @torch.no_grad()
    def _capture(self, model, rows, module_names, device):
        capture = _InputCapture(model, module_names, self.max_tokens)
        collate = ResponseOnlyDataCollator(self.args._tokenizer)
        was_training = model.training
        model.eval()
        try:
            for start in range(0, len(rows), self.batch_size):
                # build_rows_from_records carries extra string fields (prompt/response); the
                # collator pads only the tokenized fields (mirrors rows_to_dataset).
                batch_rows = [
                    {k: row[k] for k in ("input_ids", "attention_mask", "labels")}
                    for row in rows[start : start + self.batch_size]
                ]
                batch = collate(batch_rows)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                capture.set_mask(batch["labels"].to(device).ne(-100))
                model(input_ids=input_ids, attention_mask=attention_mask)
                if all(capture.counts[n] >= self.max_tokens for n in module_names):
                    break
        finally:
            if was_training:
                model.train()
        out = {name: capture.get(name) for name in module_names}
        capture.remove()
        return out

    # ---------------------------------------------------------------- build ----
    def build(self, model, tokenizer, task_records):
        """Compute C_S and the task init per target module, write W', set A_S/B_S, and install
        the C_S projection hooks. Idempotent + deterministic (seeded) so a resumed run rebuilds
        an identical W'/init before the saved A/B are loaded over it."""
        torch.manual_seed(self.args.seed)
        device = next(model.parameters()).device
        if device.type != "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)

        module_names = [name for name, module in model.named_modules() if _is_lora_layer(module)]
        if not module_names:
            raise RuntimeError("salora: no LoRA layers found on the model.")
        modules_by_name = dict(model.named_modules())

        want_task = self.init_mode == "task"
        harmful_rows = self._rows(self._load_harmful_records())
        task_rows = self._rows(task_records, n=self.n_task) if want_task else []
        print(
            f"[salora] init_mode={self.init_mode} "
            f"capturing activations: harmful_rows={len(harmful_rows)} "
            f"task_rows={len(task_rows)} modules={len(module_names)} "
            f"r_s={self.r_s} r_t={self.r_t} max_tokens={self.max_tokens}",
            flush=True,
        )
        harmful_X = self._capture(model, harmful_rows, module_names, device)
        # Only "task" needs task-feature capture; "pissa"/"vanilla" skip it (no training data).
        task_X = self._capture(model, task_rows, module_names, device) if want_task else None

        scaling = self._scaling(modules_by_name[module_names[0]])
        n_built = 0
        for name in module_names:
            module = modules_by_name[name]
            Xh = harmful_X.get(name)
            if Xh is None:
                raise RuntimeError(f"salora: no captured harmful activations for module {name}.")
            Xt = task_X.get(name) if task_X is not None else None
            if want_task and Xt is None:
                raise RuntimeError(f"salora: no captured task activations for module {name}.")
            self._build_module(module, name, Xh, Xt, scaling, device)
            harmful_X[name] = None
            if task_X is not None:
                task_X[name] = None
            n_built += 1
        print(f"[salora] built reparameterization for {n_built} modules (scaling={scaling}).", flush=True)
        return model

    def save_initialization(self, path, context):
        """Cache the LR-independent SaLoRA reparameterization on CPU."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "salora_init_v1",
            "context": context,
            "reference_logits": self._initialization_reference(model=None),
            "modules": {
                name: {
                    key: value.detach().to(device="cpu", dtype=torch.float32).clone()
                    for key, value in data.items()
                    if key in ("U_C", "B0", "A0")
                }
                for name, data in self.modules.items()
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _initialization_reference(self, model=None):
        model = model or getattr(self, "_reference_model", None)
        if model is None:
            raise RuntimeError("SaLoRA reference model is not available")
        tokenizer = self.args._tokenizer
        encoded = tokenizer(
            "SaLoRA initialization cache verification.",
            return_tensors="pt",
            add_special_tokens=True,
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(**encoded).logits[:, -1, :].float().cpu()
        finally:
            if was_training:
                model.train()
        return logits

    def load_initialization(self, model, path, expected_context):
        """Restore cached U_C/A0/B0 and deterministically reconstruct W' + hooks."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "salora_init_v1":
            raise ValueError(f"unsupported SaLoRA initialization cache: {path}")
        if payload.get("context") != expected_context:
            raise ValueError(
                f"SaLoRA initialization cache provenance mismatch at {path}\n"
                f"expected={expected_context}\nactual={payload.get('context')}"
            )
        modules_by_name = dict(model.named_modules())
        module_names = [name for name, module in model.named_modules() if _is_lora_layer(module)]
        cached = payload.get("modules", {})
        if set(cached) != set(module_names):
            raise ValueError(
                f"SaLoRA initialization cache module mismatch: "
                f"cache={len(cached)} model={len(module_names)}"
            )
        scaling = self._scaling(modules_by_name[module_names[0]])
        device = next(model.parameters()).device
        for name in module_names:
            module = modules_by_name[name]
            data = cached[name]
            U_C = data["U_C"].to(device=device, dtype=torch.float32)
            B0 = data["B0"].to(device=device, dtype=torch.float32)
            A0 = data["A0"].to(device=device, dtype=torch.float32)
            weight = module.base_layer.weight
            Wf = weight.data.float()
            CB0 = B0 - U_C @ (U_C.t() @ B0)
            weight.data = (Wf - scaling * (CB0 @ A0)).to(weight.dtype).clone()
            module.lora_A["default"].weight.data = A0.to(weight.dtype).clone()
            module.lora_B["default"].weight.data = B0.to(weight.dtype).clone()
            self.modules[name] = {
                "U_C": U_C.detach().clone(),
                "B0": B0.detach().to("cpu").clone(),
                "A0": A0.detach().to("cpu").clone(),
            }
            self._uc_cache[name] = {}
            self._proj_handles.append(
                module.lora_B["default"].register_forward_hook(self._make_proj_hook(name))
            )
        reference = payload.get("reference_logits")
        if reference is None:
            raise ValueError(f"SaLoRA cache has no reference logits: {path}")
        actual = self._initialization_reference(model)
        max_abs_error = (actual - reference.float()).abs().max().item()
        if max_abs_error > 1e-5:
            raise ValueError(
                f"SaLoRA cache restoration failed logits verification: "
                f"max_abs_error={max_abs_error:.8g}"
            )
        print(
            f"[salora] restored cached reparameterization for {len(module_names)} "
            f"modules (scaling={scaling}, logits_max_abs_error={max_abs_error:.3g}) "
            f"<- {path}",
            flush=True,
        )
        return model

    def _scaling(self, module):
        scaling = module.scaling
        if isinstance(scaling, dict):
            scaling = scaling.get("default")
        return float(scaling)

    def _build_module(self, module, name, Xh, Xt, scaling, device):
        weight = module.base_layer.weight
        dtype = weight.dtype
        Wf = weight.data.float()
        d_out, d_in = Wf.shape

        def left_singular(X, q):
            score = X.to(device=device, dtype=torch.float32) @ Wf.t()  # (N, d_out), rows = W x_i
            q = max(1, min(q, score.shape[0], score.shape[1]))
            _, _, V = torch.svd_lowrank(score, q=q, niter=self.niter)  # V: (d_out, q)
            return V

        U_C = left_singular(Xh, self.r_s)        # (d_out, r_s) — safety basis (all modes)

        if self.init_mode == "vanilla":
            # Standard LoRA init (A ~ kaiming, B = 0): keep PEFT's defaults, no PiSSA, no residual
            # reparam. B_S = 0 ⇒ delta = s·C_S·0·A0 = 0 ⇒ W' = W (base untouched). C_S still hooks.
            A_S = module.lora_A["default"].weight.data.float()
            B_S = module.lora_B["default"].weight.data.float()
        else:
            r = min(int(self.args.lora_r), d_out, d_in)
            Uw, Sw, Vw = torch.svd_lowrank(Wf, q=r, niter=self.niter)  # Uw(d_out,r) Sw(r) Vw(d_in,r)
            sqrtS = torch.sqrt(Sw.clamp_min(0.0))
            B_S = Uw * sqrtS.unsqueeze(0)             # (d_out, r) = Ū_{[:r]} √S̄ (PiSSA)
            if self.init_mode == "task":
                # (a) project the PiSSA init onto the task subspace U Uᵀ (paper eq 12).
                U_task = left_singular(Xt, self.r_t)  # (d_out, r_t) — task subspace
                B_S = U_task @ (U_task.t() @ B_S)
            A_S = sqrtS.unsqueeze(1) * Vw.t()         # (r, d_in) = √S̄ V̄ᵀ

            # C_S B_S = B_S − U_C (U_Cᵀ B_S); residual W' = W − s·C_S B_S A_S.
            CB = B_S - U_C @ (U_C.t() @ B_S)
            delta = scaling * (CB @ A_S)
            module.base_layer.weight.data = (Wf - delta).to(dtype).clone()
            module.lora_A["default"].weight.data = A_S.to(dtype).clone()
            module.lora_B["default"].weight.data = B_S.to(dtype).clone()

        # NB: .to(dtype)/.to("cpu") are no-ops returning the SAME tensor when dtype/device
        # already match, which would alias these init anchors to the (now trainable) live
        # weights. Clone so the −C_S B₀ A₀ anchor stays fixed as A/B train.
        self.modules[name] = {
            "U_C": U_C.detach().clone(),              # fp32, model device (hook + save)
            "B0": B_S.detach().to("cpu", dtype=torch.float32).clone(),  # rank-2r anchor
            "A0": A_S.detach().to("cpu", dtype=torch.float32).clone(),
        }
        self._uc_cache[name] = {}
        self._proj_handles.append(
            module.lora_B["default"].register_forward_hook(self._make_proj_hook(name))
        )

    # --------------------------------------------------------- projection ----
    def _make_proj_hook(self, name):
        def hook(_module, _inputs, output):
            cache = self._uc_cache[name]
            key = (output.device, output.dtype)
            uc = cache.get(key)
            if uc is None:
                uc = self.modules[name]["U_C"].to(device=output.device, dtype=output.dtype)
                cache[key] = uc
            # C_S y = y − U_C (U_Cᵀ y)  over the last dim (d_out).
            return output - (output @ uc) @ uc.t()

        return hook

    def remove_hooks(self):
        for handle in self._proj_handles:
            handle.remove()
        self._proj_handles = []

    # -------------------------------------------------------------- saving ----
    def _lookup(self, module_name):
        data = self.modules.get(module_name)
        if data is not None:
            return data
        # Robust fallback if the state-dict prefix differs from named_modules().
        for key, value in self.modules.items():
            if module_name.endswith(key) or key.endswith(module_name):
                return value
        raise KeyError(f"salora: no module data for {module_name!r}")

    def effective_rank2r_state(self, round_state_fp32):
        """Rank-2r factors of the EFFECTIVE update over the ORIGINAL base:
        s·(C_S B A − C_S B₀ A₀) — the exact drift the deployed model carries (the −C_S B₀A₀
        anchor cancels the W' fold, so round 0 encodes zero drift). Used both for saving
        retained adapters and for feeding alignment-energy diagnostics C_S-corrected factors."""
        new_state = {}
        for key, value in round_state_fp32.items():
            # global_dict lives on the model device (e.g. cuda); the stored U_C/A0/B0 are on CPU.
            # Do the whole rank-2r construction on CPU so devices always match (a save-time op).
            value = value.detach().to(device="cpu", dtype=torch.float32)
            module_name, suffix = _module_name_from_key(key)
            if module_name is None:
                new_state[key] = value
                continue
            data = self._lookup(module_name)
            U_C = data["U_C"].to(device="cpu", dtype=torch.float32)
            if suffix == ".lora_A.weight":
                A0 = data["A0"].to(device="cpu", dtype=torch.float32)
                new_state[key] = torch.cat([value, A0], dim=0).contiguous()       # (2r, d_in)
            else:  # .lora_B.weight
                B0 = data["B0"].to(device="cpu", dtype=torch.float32)
                CB = value - U_C @ (U_C.t() @ value)                              # (d_out, r)
                CB0 = B0 - U_C @ (U_C.t() @ B0)
                new_state[key] = torch.cat([CB, -CB0], dim=1).contiguous()        # (d_out, 2r)
        return new_state

    def save_round_adapter(self, model, round_state_fp32, out_dir, base_model_id):
        """Write the round's adapter as a rank-2r LoRA on the ORIGINAL base (see module
        docstring). The result loads + merges with the stock merge-then-eval pipeline."""
        from safetensors.torch import save_file

        os.makedirs(out_dir, exist_ok=True)
        new_state = self.effective_rank2r_state(round_state_fp32)
        save_dtype = self._save_dtype(model)
        new_state = {k: v.to(save_dtype) if v.is_floating_point() else v for k, v in new_state.items()}
        save_file(new_state, os.path.join(out_dir, "adapter_model.safetensors"),
                  metadata={"format": "pt"})
        self._write_adapter_config(model, out_dir, base_model_id)

    def _save_dtype(self, model):
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    def _write_adapter_config(self, model, out_dir, base_model_id):
        cfg = dict(model.peft_config["default"].to_dict())
        cfg["r"] = 2 * int(self.args.lora_r)
        cfg["lora_alpha"] = 2 * int(self.args.lora_alpha)
        cfg["base_model_name_or_path"] = base_model_id
        for field in ("rank_pattern", "alpha_pattern"):
            if cfg.get(field):
                # rank-2r is uniform; per-module overrides from the rank-r config don't apply.
                cfg[field] = {}
        serializable = {}
        for k, v in cfg.items():
            if isinstance(v, set):
                v = sorted(v)
            serializable[k] = v
        serializable.setdefault("peft_type", "LORA")
        with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)

    def finalize_merge(self, model):
        """Bake C_S into the live ``lora_B`` (B ← C_S B), drop the hooks, and return the model
        ready for ``merge_and_unload`` over the live W' base. Call after the final global A/B
        have been loaded onto the model."""
        self.remove_hooks()
        for name, data in self.modules.items():
            module = dict(model.named_modules())[name]
            B = module.lora_B["default"].weight
            U_C = data["U_C"].to(device=B.device, dtype=torch.float32)
            Bf = B.data.float()
            B.data = (Bf - U_C @ (U_C.t() @ Bf)).to(B.dtype)
        return model

    # ----------------------------------------------------------- bookkeeping ----
    def describe(self):
        return {
            "method": "salora",
            "init_mode": self.init_mode,
            "r_s": self.r_s,
            "r_t": self.r_t,
            "n_harmful": self.n_harmful,
            "n_task": self.n_task,
            "max_tokens": self.max_tokens,
            "harmful_path": self.harmful_path,
            "response_field": self.response_field,
            "modules": len(self.modules),
        }
