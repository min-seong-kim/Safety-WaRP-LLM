"""
WaRP-Space Safety Neuron Tuning (WaRP-SN-Tune)

This tuner consumes the WaRP-SN detector output, whose indices are COLUMN
indices of basis_coeff = W @ U.  It fine-tunes only those selected columns on
safety-aligned data and restores the final checkpoint to standard HuggingFace
nn.Linear modules.

Key correctness safeguards
--------------------------
1. Training forward uses basis_coeff:
      module.flag = True
      output = F.linear(input, basis_coeff @ U^T)
2. All model parameters are frozen first.
3. Only basis_coeff tensors for modules with selected columns are unfrozen.
4. A backward hook zeroes gradients outside selected columns.
5. AdamW weight_decay is set to 0 by default, because decoupled weight decay
   would modify masked columns even when their gradients are zero.
6. Debug diagnostics verify:
   - number of trainable basis_coeff tensors
   - nonzero gradient after the first backward pass
   - actual parameter update after the first optimizer step
   - masked-column gradient leakage
"""

import os
import gc
import json
import math
import logging
from typing import Dict, Set, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .warp_modules import LinearWaRP, switch_to_warp_module, restore_weight, restore_to_linear

logger = logging.getLogger(__name__)


_LAYER_TYPE_TO_PATH: Dict[str, Tuple[str, str]] = {
    "ffn_up":   ("mlp", "up_proj"),
    "ffn_down": ("mlp", "down_proj"),
    "attn_q":   ("self_attn", "q_proj"),
    "attn_k":   ("self_attn", "k_proj"),
    "attn_v":   ("self_attn", "v_proj"),
    "ffn_gate": ("mlp", "gate_proj"),
}

_LAYER_TYPE_TO_SN_KEY: Dict[str, str] = {
    "ffn_up":   "ffn_up",
    "ffn_down": "ffn_down",
    "attn_q":   "q",
    "attn_k":   "k",
    "attn_v":   "v",
    "ffn_gate": "ffn_gate",
}


class CircuitBreakersDataset(Dataset):
    """
    Circuit Breakers-style dataset for safety tuning.

    Expected common schema:
      {"prompt": harmful_prompt, "llama3_output": safe_response}

    Labels are masked for user/system prompt tokens and enabled only for the
    assistant safe response tokens.
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_samples: Optional[int] = None,
        max_length: int = 1024,
        is_instruct: bool = True,
        response_key: str = "llama3_output",
        prompt_key: str = "prompt",
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if not isinstance(self.data, list):
            raise ValueError(f"Dataset must be a JSON list: {json_path}")
        if max_samples:
            self.data = self.data[: min(int(max_samples), len(self.data))]

        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.is_instruct = is_instruct
        self.response_key = response_key
        self.prompt_key = prompt_key
        self._logged_first = False

        logger.info(f"[Dataset] Loaded {len(self.data)} samples from {json_path}")
        logger.info(f"[Dataset] is_instruct={self.is_instruct}, max_length={self.max_length}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        harmful_prompt = str(item.get(self.prompt_key, ""))
        safe_response = str(item.get(self.response_key, ""))

        if not harmful_prompt:
            harmful_prompt = str(item.get("instruction", item.get("input", "")))
        if not safe_response:
            safe_response = str(item.get("output", item.get("response", "")))

        if self.is_instruct:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": harmful_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": harmful_prompt},
                    {"role": "assistant", "content": safe_response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        else:
            prompt_text = f"Question: {harmful_prompt}\nAnswer:"
            full_text = f"{prompt_text} {safe_response}"
            prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)["input_ids"]
            full_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_length)["input_ids"]

        full_ids = full_ids[: self.max_length]
        seq_len = len(full_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        pad_len = self.max_length - seq_len
        input_ids = full_ids + [pad_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), self.max_length)
        for i in range(prompt_len):
            labels[i] = -100
        for i in range(self.max_length):
            if attention_mask[i] == 0:
                labels[i] = -100

        # If truncation removed the entire response, keep at least last non-pad token trainable
        # to avoid NaN/zero-loss batches.
        if all(l == -100 for l in labels):
            last = max(0, seq_len - 1)
            labels[last] = input_ids[last]

        if not self._logged_first:
            self._logged_first = True
            learned = [t for t, l in zip(input_ids, labels) if l != -100]
            logger.info("\n[Dataset first sample]")
            logger.info(f"  prompt_length : {prompt_len}")
            logger.info(f"  full_length   : {seq_len}")
            logger.info(f"  learned_tokens: {len(learned)}")
            preview = self.tokenizer.decode(learned[:60], skip_special_tokens=True)
            logger.info(f"  learned preview: {preview[:250]}...")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class WaRPSNTuner:
    """
    Fine-tune only selected WaRP-space basis columns.

    safety_neurons format:
      {'ffn_up': {layer_idx: {col_idx,...}}, 'q': ..., 'k': ..., 'v': ...}
    """

    def __init__(
        self,
        model,
        tokenizer,
        basis_data: Dict[Tuple[int, str], dict],
        safety_neurons: Dict[str, Dict[int, Set[int]]],
        layer_types: List[str],
        num_layers: int,
        is_instruct: bool = True,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        max_seq_len: int = 1024,
        max_samples: Optional[int] = 4994,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        debug_first_steps: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.basis_data = basis_data
        self.safety_neurons = safety_neurons
        self.layer_types = [lt.strip() for lt in layer_types if lt.strip()]
        self.num_layers = int(num_layers)
        self.is_instruct = is_instruct
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.grad_accum_steps = int(grad_accum_steps)
        self.max_seq_len = int(max_seq_len)
        self.max_samples = max_samples
        self.warmup_ratio = float(warmup_ratio)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.debug_first_steps = int(debug_first_steps)

        self._hooks: List = []
        self._train_specs: List[Tuple[str, LinearWaRP, torch.Tensor]] = []  # name, module, col_mask_cpu
        self._reparameterized = False
        self._masking_ready = False

    def reparameterize(self) -> None:
        logger.info("=" * 80)
        logger.info("[WaRP-SN Tune] Reparameterizing for training")
        logger.info("=" * 80)

        self.model = switch_to_warp_module(self.model, self.layer_types, target_layers="all")

        reparameterized, skipped = 0, 0
        max_restore_err = 0.0

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                key = (layer_idx, layer_type)
                if key not in self.basis_data:
                    skipped += 1
                    continue

                module = self._get_module(layer, layer_type)
                if not isinstance(module, LinearWaRP):
                    skipped += 1
                    continue

                W = module.weight.data  # [d_out, d_in]
                U = self.basis_data[key]["U"].to(dtype=W.dtype, device=W.device)
                if W.shape[1] != U.shape[0]:
                    logger.warning(f"Layer {layer_idx} {layer_type}: W.in={W.shape[1]} != U.dim={U.shape[0]}; skip")
                    skipped += 1
                    continue

                basis = W @ U
                module.basis_coeff.data.copy_(basis)
                module.UT_forward = U.detach().clone()
                module.UT_backward = torch.empty(0, dtype=W.dtype, device=W.device)
                module.flag = True                    # crucial: forward uses basis_coeff
                module.coeff_mask.data.zero_()
                if hasattr(module, "mask_mode"):
                    module.mask_mode.fill_(1)         # no internal detach masking

                if reparameterized < 3:
                    restored = basis @ U.t()
                    err = (restored.float() - W.float()).abs().max().item()
                    max_restore_err = max(max_restore_err, err)
                    logger.info(f"  Restore check layer {layer_idx:02d} {layer_type:10}: max|WU U^T - W|={err:.3e}")

                reparameterized += 1

        logger.info(f"  Reparameterized modules : {reparameterized}")
        logger.info(f"  Skipped modules         : {skipped}")
        logger.info(f"  Max restore error sample: {max_restore_err:.3e}")
        logger.info("  flag=True: training forward uses basis_coeff @ U^T.")
        logger.info("=" * 80)

        if reparameterized == 0:
            raise RuntimeError("No modules were reparameterized. Check basis_data/layer_types.")
        self._reparameterized = True

    def setup_gradient_masking(self) -> None:
        """
        Freeze all parameters, then unfreeze selected basis_coeff tensors.
        Hook keeps gradients only for selected columns.
        """
        if not self._reparameterized:
            raise RuntimeError("Call reparameterize() first.")

        logger.info("=" * 80)
        logger.info("[WaRP-SN Tune] Setting up column-wise gradient masking")
        logger.info("=" * 80)

        # Freeze everything first.
        for _, p in self.model.named_parameters():
            p.requires_grad_(False)

        modules_with_cols = 0
        invalid_cols_total = 0
        effective_trainable = 0
        total_params = sum(p.numel() for p in self.model.parameters())

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                if (layer_idx, layer_type) not in self.basis_data:
                    continue

                module = self._get_module(layer, layer_type)
                if not isinstance(module, LinearWaRP):
                    continue

                sn_key = _LAYER_TYPE_TO_SN_KEY.get(layer_type)
                if sn_key is None or sn_key not in self.safety_neurons:
                    continue

                raw_cols = set(self.safety_neurons.get(sn_key, {}).get(layer_idx, set()))
                d_out, d_in = module.basis_coeff.shape
                valid_cols = sorted({int(c) for c in raw_cols if 0 <= int(c) < d_in})
                invalid_cols = len(raw_cols) - len(valid_cols)
                invalid_cols_total += max(0, invalid_cols)

                if not valid_cols:
                    continue

                col_mask = torch.zeros(d_in, dtype=torch.bool)
                col_mask[valid_cols] = True

                module.basis_coeff.requires_grad_(True)
                handle = module.basis_coeff.register_hook(self._make_col_mask_hook(col_mask, layer_idx, layer_type))
                self._hooks.append(handle)

                module_name = f"model.layers.{layer_idx}.{layer_type}"
                self._train_specs.append((module_name, module, col_mask))

                modules_with_cols += 1
                effective_trainable += int(col_mask.sum().item()) * d_out

                if layer_idx < 2:
                    logger.info(f"  Layer {layer_idx:02d} {layer_type:10}: "
                                f"selected_cols={int(col_mask.sum())}/{d_in}, "
                                f"effective_trainable={int(col_mask.sum()) * d_out:,}")

        trainable_tensors = [(n, p.numel()) for n, p in self.model.named_parameters() if p.requires_grad]
        basis_trainable = [(n, p.numel()) for n, p in self.model.named_parameters()
                           if "basis_coeff" in n and p.requires_grad]

        logger.info("")
        logger.info(f"  Modules with selected columns : {modules_with_cols}")
        logger.info(f"  Gradient hooks registered     : {len(self._hooks)}")
        logger.info(f"  Invalid selected columns      : {invalid_cols_total}")
        logger.info(f"  Trainable tensors             : {len(trainable_tensors)}")
        logger.info(f"  Trainable basis_coeff tensors : {len(basis_trainable)}")
        logger.info(f"  Effective trainable elements  : {effective_trainable:,}")
        logger.info(f"  Total model parameters        : {total_params:,}")
        logger.info(f"  Effective trainable ratio     : {effective_trainable / max(total_params,1) * 100:.4f}%")
        logger.info(f"  Optimizer tensor elements     : {sum(n for _, n in trainable_tensors):,}")
        logger.info("  First trainable tensors:")
        for n, sz in trainable_tensors[:8]:
            logger.info(f"    {n}: {sz:,}")

        if not basis_trainable:
            raise RuntimeError("No trainable basis_coeff tensors. Check safety_neurons file and layer_types.")
        if self.weight_decay != 0:
            logger.warning("weight_decay is non-zero. Decoupled AdamW weight decay can alter masked columns.")

        logger.info("=" * 80)
        self._masking_ready = True

    @staticmethod
    def _make_col_mask_hook(col_mask_cpu: torch.Tensor, layer_idx: int, layer_type: str):
        """
        Keeps gradient only at selected columns.
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            mask = col_mask_cpu.to(device=grad.device)
            # Use multiplication to avoid in-place autograd surprises.
            return grad * mask.view(1, -1).to(dtype=grad.dtype)
        return hook

    def train(self, dataset_path: str) -> None:
        if not self._masking_ready:
            raise RuntimeError("Call setup_gradient_masking() before train().")

        logger.info("=" * 80)
        logger.info("[WaRP-SN Tune] Starting safety fine-tuning")
        logger.info(f"  dataset      : {dataset_path}")
        logger.info(f"  lr           : {self.learning_rate}")
        logger.info(f"  weight_decay : {self.weight_decay}")
        logger.info(f"  epochs       : {self.num_epochs}")
        logger.info(f"  batch_size   : {self.batch_size}")
        logger.info(f"  grad_accum   : {self.grad_accum_steps}")
        logger.info(f"  max_samples  : {self.max_samples}")
        logger.info("=" * 80)

        dataset = CircuitBreakersDataset(
            json_path=dataset_path,
            tokenizer=self.tokenizer,
            max_samples=self.max_samples,
            max_length=self.max_seq_len,
            is_instruct=self.is_instruct,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=torch.Generator().manual_seed(112),
        )
        logger.info(f"  DataLoader batches per epoch: {len(loader)}")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")

        optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        total_opt_steps = self.num_epochs * math.ceil(len(loader) / self.grad_accum_steps)
        warmup_steps = int(total_opt_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps,
        )
        logger.info(f"  Optimizer steps: {total_opt_steps}, warmup_steps={warmup_steps}")

        device = next(self.model.parameters()).device
        self.model.train()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        optimizer.zero_grad(set_to_none=True)
        optimizer_step = 0
        global_backward = 0
        first_update_checked = False
        tracked_before = self._snapshot_tracked_columns(max_modules=2)

        for epoch in range(self.num_epochs):
            epoch_loss, epoch_batches = 0.0, 0
            logger.info(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            pbar = tqdm(loader, desc=f"WaRP-SN epoch {epoch+1}")

            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_cache=False,
                    )
                    loss_raw = outputs.loss
                    loss = loss_raw / self.grad_accum_steps

                if not torch.isfinite(loss):
                    logger.warning(f"[Non-finite loss] epoch={epoch+1}, batch={batch_idx}, loss={loss_raw.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                global_backward += 1

                if global_backward <= self.debug_first_steps:
                    self._log_gradient_diagnostics(prefix=f"after backward {global_backward}")

                is_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))
                if is_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.max_grad_norm)
                    if optimizer_step < self.debug_first_steps:
                        logger.info(f"[DEBUG] grad_norm before opt_step {optimizer_step+1}: {float(grad_norm):.4e}")

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    if not first_update_checked:
                        first_update_checked = True
                        self._log_update_diagnostics(tracked_before)

                    if optimizer_step % 20 == 0:
                        lr_now = scheduler.get_last_lr()[0]
                        logger.info(f"[opt_step {optimizer_step}] "
                                    f"loss={loss_raw.item():.4f}, lr={lr_now:.2e}, grad_norm={float(grad_norm):.3e}")

                loss_val = float(loss_raw.detach().cpu())
                epoch_loss += loss_val
                epoch_batches += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "opt": optimizer_step})

                if (batch_idx + 1) % 50 == 0 and device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

            logger.info(f"Epoch {epoch+1} done: avg_loss={epoch_loss / max(epoch_batches,1):.4f}, "
                        f"optimizer_steps={optimizer_step}")

        logger.info("=" * 80)
        logger.info("[WaRP-SN Tune] Training complete")
        logger.info(f"  Total optimizer steps: {optimizer_step}")
        logger.info(f"  Trainable basis_coeff tensors: {sum(1 for _, p in self.model.named_parameters() if 'basis_coeff' in _ and p.requires_grad)}")
        logger.info("=" * 80)

    def _snapshot_tracked_columns(self, max_modules: int = 2):
        """
        Snapshot selected and non-selected columns for a few modules to verify
        first optimizer step actually changes selected columns.
        """
        snaps = []
        for name, module, col_mask_cpu in self._train_specs[:max_modules]:
            with torch.no_grad():
                mask = col_mask_cpu.to(module.basis_coeff.device)
                selected_idx = torch.nonzero(mask, as_tuple=False).view(-1)
                non_idx = torch.nonzero(~mask, as_tuple=False).view(-1)
                sel = selected_idx[: min(8, selected_idx.numel())]
                non = non_idx[: min(8, non_idx.numel())]
                snaps.append({
                    "name": name,
                    "module": module,
                    "sel_idx": sel.detach().cpu(),
                    "non_idx": non.detach().cpu(),
                    "sel_before": module.basis_coeff[:, sel].detach().float().cpu().clone() if sel.numel() else None,
                    "non_before": module.basis_coeff[:, non].detach().float().cpu().clone() if non.numel() else None,
                })
        logger.info(f"[DEBUG] Snapshotted {len(snaps)} module(s) for first-update diagnostics.")
        return snaps

    def _log_gradient_diagnostics(self, prefix: str = "") -> None:
        """
        Logs gradient existence, selected-column grad, and leakage into masked columns.
        Run immediately after backward and before optimizer.zero_grad().
        """
        modules_with_grad = 0
        modules_with_nonzero_selected = 0
        max_leak = 0.0
        rows = []

        for name, module, col_mask_cpu in self._train_specs:
            g = module.basis_coeff.grad
            if g is None:
                continue
            modules_with_grad += 1
            mask = col_mask_cpu.to(g.device)
            sel_sum = g[:, mask].abs().sum().item() if mask.any() else 0.0
            non_sum = g[:, ~mask].abs().sum().item() if (~mask).any() else 0.0
            max_leak = max(max_leak, non_sum)
            if sel_sum > 0:
                modules_with_nonzero_selected += 1
            if len(rows) < 5:
                rows.append((name, sel_sum, non_sum, g.abs().mean().item(), g.abs().max().item()))

        logger.info(f"[DEBUG grad {prefix}] modules_with_grad={modules_with_grad}/{len(self._train_specs)}, "
                    f"modules_with_nonzero_selected={modules_with_nonzero_selected}, max_masked_grad_sum={max_leak:.3e}")
        for name, sel_sum, non_sum, mean_g, max_g in rows:
            logger.info(f"  [grad] {name}: selected_sum={sel_sum:.3e}, masked_sum={non_sum:.3e}, "
                        f"mean={mean_g:.3e}, max={max_g:.3e}")

        if modules_with_grad == 0:
            logger.error("[DEBUG] No basis_coeff gradient reached any WaRP module. "
                         "Check module.flag=True, requires_grad=True, and optimizer param groups.")
        elif modules_with_nonzero_selected == 0:
            logger.error("[DEBUG] Gradients exist but selected-column gradient is zero. "
                         "Check loss labels, selected columns, and hooks.")

    def _log_update_diagnostics(self, snaps) -> None:
        """
        Checks actual parameter movement after the first optimizer step.
        """
        logger.info("[DEBUG update] Checking parameter movement after first optimizer.step()")
        for snap in snaps:
            module = snap["module"]
            name = snap["name"]
            sel_idx = snap["sel_idx"].to(module.basis_coeff.device)
            non_idx = snap["non_idx"].to(module.basis_coeff.device)
            with torch.no_grad():
                if snap["sel_before"] is not None and sel_idx.numel():
                    sel_after = module.basis_coeff[:, sel_idx].detach().float().cpu()
                    sel_delta = (sel_after - snap["sel_before"]).abs().max().item()
                else:
                    sel_delta = 0.0
                if snap["non_before"] is not None and non_idx.numel():
                    non_after = module.basis_coeff[:, non_idx].detach().float().cpu()
                    non_delta = (non_after - snap["non_before"]).abs().max().item()
                else:
                    non_delta = 0.0

            logger.info(f"  [update] {name}: selected_delta_max={sel_delta:.3e}, "
                        f"masked_delta_max={non_delta:.3e}")
            if sel_delta == 0.0:
                logger.error(f"  [update] {name}: selected columns did not change after optimizer.step().")
            if non_delta > 1e-8:
                logger.warning(f"  [update] {name}: masked columns changed. "
                               f"Use weight_decay=0 and verify hooks.")

    def save(self, output_dir: str, tokenizer=None) -> None:
        logger.info("=" * 80)
        logger.info("[WaRP-SN Tune] Restoring weights and saving")
        logger.info(f"  output_dir: {output_dir}")
        logger.info("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("  Removed gradient hooks.")

        # Move to CPU before restoration to reduce GPU memory pressure.
        self.model = self.model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        restore_weight(self.model)
        logger.info("  Restored original-space weights: W = basis_coeff @ U^T")

        restore_to_linear(self.model)
        logger.info("  Converted LinearWaRP modules back to nn.Linear")

        self.model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
        logger.info("  Saved model checkpoint.")

        tok = tokenizer or self.tokenizer
        if tok is not None:
            tok.save_pretrained(output_dir)
            logger.info("  Saved tokenizer.")

        cfg = {
            "method": "WaRP-SN-Tune",
            "layer_types": self.layer_types,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "max_seq_len": self.max_seq_len,
            "max_samples": self.max_samples,
            "is_instruct": self.is_instruct,
            "index_semantics": "COLUMN indices of basis_coeff = W @ U",
        }
        with open(os.path.join(output_dir, "warp_sn_tune_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logger.info("  Saved warp_sn_tune_config.json")

        files = os.listdir(output_dir)
        logger.info(f"  Output files: {len(files)} total; "
                    f"{sum(f.endswith('.safetensors') for f in files)} safetensors shard(s).")
        logger.info("=" * 80)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_module(self, layer, layer_type: str):
        sub, attr = _LAYER_TYPE_TO_PATH[layer_type]
        return getattr(getattr(layer, sub), attr)
