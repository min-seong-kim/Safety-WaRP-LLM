"""
WaRP-Space Safety Neuron Detection

This module detects safety-relevant *input-basis directions* after WaRP/WSR
reparameterization.  The detected indices are COLUMN indices of basis_coeff
where basis_coeff = W @ U and forward restoration is W = basis_coeff @ U^T.

Important distinction
---------------------
Original SN-Tune usually scores output/intermediate neurons in the original
parameterization.  This detector scores rotated input-basis columns.  Therefore
the produced indices must be consumed by a WaRP-aware tuner that masks columns
of basis_coeff, not rows of the original weight matrix.

Design choices in this version
------------------------------
1. Detection forward is function-preserving: module.flag=False, so the model
   uses the original W during detection.
2. Hooks project each module input as x @ U and score columns in the safety
   basis.
3. The default score is output-aware:
      score_k = sum_t |(x_t U)_k| * sum_i |basis_coeff_{i,k}|
   This avoids selecting only high-variance generic activation directions.
4. Selection is frequency-based by default.  A direction is selected if it is
   among per-prompt top-k directions in at least freq_threshold fraction of
   prompts.  Setting freq_threshold=1.0 recovers exact intersection.
5. The output file remains the original 5-line JSON format for compatibility:
      ffn_up, ffn_down, q, k, v
"""

import os
import json
import logging
from typing import Dict, Set, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

from .warp_modules import LinearWaRP, switch_to_warp_module

logger = logging.getLogger(__name__)


_LAYER_TYPE_TO_PATH: Dict[str, Tuple[str, str]] = {
    "ffn_up":   ("mlp", "up_proj"),
    "ffn_down": ("mlp", "down_proj"),
    "attn_q":   ("self_attn", "q_proj"),
    "attn_k":   ("self_attn", "k_proj"),
    "attn_v":   ("self_attn", "v_proj"),
    # Keep this optional. It is skipped automatically when the basis is absent.
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

_SN_KEY_ORDER_5LINE = ["ffn_up", "ffn_down", "q", "k", "v"]


class WaRPSNDetector:
    """
    Safety-neuron detection in WaRP-reparameterized space.

    Parameters
    ----------
    model, tokenizer:
        HuggingFace causal LM and tokenizer.
    basis_dir:
        Directory containing Phase-1 SVD basis files:
          basis_dir/ffn_up/layer_00_svd.pt, ...
    layer_types:
        Target module types, e.g. ['ffn_up','ffn_down','attn_q','attn_k','attn_v'].
    top_k_ffn, top_k_attn:
        Per-prompt top-k selected for FFN/attention modules.
    score_mode:
        'activation'          : sum |xU|
        'activation_weighted' : sum |xU| * sum_i |Wtilde_i,k|  (default)
    freq_threshold:
        Select directions appearing in top-k for at least this fraction of
        successful prompts. 1.0 is exact intersection.
    """

    def __init__(
        self,
        model,
        tokenizer,
        basis_dir: str,
        layer_types: List[str],
        num_layers: int,
        is_instruct: bool = True,
        top_k_ffn: int = 300,
        top_k_attn: int = 50,
        max_seq_len: int = 1024,
        score_mode: str = "activation_weighted",
        freq_threshold: float = 0.80,
        log_score_stats: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.basis_dir = basis_dir
        self.layer_types = [lt.strip() for lt in layer_types if lt.strip()]
        self.num_layers = num_layers
        self.is_instruct = is_instruct
        self.top_k_ffn = int(top_k_ffn)
        self.top_k_attn = int(top_k_attn)
        self.max_seq_len = int(max_seq_len)
        self.score_mode = score_mode
        self.freq_threshold = float(freq_threshold)
        self.log_score_stats = log_score_stats

        if self.score_mode not in {"activation", "activation_weighted"}:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")
        if not (0 < self.freq_threshold <= 1.0):
            raise ValueError("freq_threshold must be in (0, 1].")

        self.basis_data: Dict[Tuple[int, str], dict] = {}
        self._reparameterized = False

    def load_basis(self) -> None:
        logger.info("=" * 80)
        logger.info("[WaRP-SN Detection] Loading Phase-1 safety basis")
        logger.info(f"  basis_dir      : {self.basis_dir}")
        logger.info(f"  layer_types    : {self.layer_types}")
        logger.info(f"  score_mode     : {self.score_mode}")
        logger.info(f"  freq_threshold : {self.freq_threshold}")
        logger.info("=" * 80)

        loaded, missing = 0, []
        for layer_type in self.layer_types:
            if layer_type not in _LAYER_TYPE_TO_PATH:
                logger.warning(f"[WARN] Unknown layer_type '{layer_type}', skipping.")
                continue
            lt_dir = os.path.join(self.basis_dir, layer_type)
            if not os.path.isdir(lt_dir):
                logger.warning(f"[WARN] No basis directory for {layer_type}: {lt_dir}")
                continue

            for layer_idx in range(self.num_layers):
                path = os.path.join(lt_dir, f"layer_{layer_idx:02d}_svd.pt")
                if not os.path.exists(path):
                    missing.append((layer_idx, layer_type))
                    continue

                data = torch.load(path, map_location="cpu", weights_only=False)
                if "U" not in data:
                    raise KeyError(f"Basis file lacks key 'U': {path}")
                U = data["U"]
                if not isinstance(U, torch.Tensor):
                    U = torch.from_numpy(np.asarray(U))
                if U.dim() != 2 or U.shape[0] != U.shape[1]:
                    raise ValueError(f"U must be square. Got {tuple(U.shape)} in {path}")

                # Also load singular values S for score normalization.
                S = data.get("S")
                if S is not None and not isinstance(S, torch.Tensor):
                    S = torch.from_numpy(np.asarray(S))
                self.basis_data[(layer_idx, layer_type)] = {
                    "U": U.contiguous(),
                    "S": S.contiguous() if S is not None else None,
                }
                loaded += 1

        logger.info(f"  Loaded basis files : {loaded}")
        if missing:
            logger.info(f"  Missing entries    : {len(missing)} "
                        f"(first 8: {missing[:8]}{'...' if len(missing)>8 else ''})")
        for lt in self.layer_types:
            count = sum(1 for (li, t) in self.basis_data if t == lt)
            logger.info(f"    {lt:12}: {count:3d} / {self.num_layers}")

        if loaded == 0:
            raise RuntimeError(f"No basis files found under {self.basis_dir}")
        logger.info("=" * 80)

    def reparameterize(self) -> None:
        """
        Convert target nn.Linear modules to LinearWaRP and initialize:
            basis_coeff = W @ U
            UT_forward = U
            flag        = False   # detection forward uses original W
        """
        logger.info("=" * 80)
        logger.info("[WaRP-SN Detection] Reparameterizing model for detection")
        logger.info("=" * 80)

        self.model = switch_to_warp_module(self.model, self.layer_types, target_layers="all")

        reparameterized, skipped, ortho_warn = 0, 0, 0
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                key = (layer_idx, layer_type)
                if key not in self.basis_data:
                    skipped += 1
                    continue

                module = self._get_module(layer, layer_type)
                if not isinstance(module, LinearWaRP):
                    logger.warning(f"Layer {layer_idx} {layer_type}: expected LinearWaRP, got {type(module)}")
                    skipped += 1
                    continue

                W = module.weight.data  # [d_out, d_in]
                U = self.basis_data[key]["U"].to(device=W.device, dtype=W.dtype)
                if W.shape[1] != U.shape[0]:
                    logger.warning(f"Layer {layer_idx} {layer_type}: W.in={W.shape[1]} != U.dim={U.shape[0]}")
                    skipped += 1
                    continue

                # Lightweight orthogonality diagnostic only for first few modules.
                if reparameterized < 3:
                    eye_err = (U.float().t() @ U.float() - torch.eye(U.shape[0], device=U.device)).abs().max().item()
                    logger.info(f"  Ortho check layer {layer_idx:02d} {layer_type:10}: max|U^T U-I|={eye_err:.3e}")
                    if eye_err > 5e-2:
                        ortho_warn += 1

                module.basis_coeff.data.copy_(W @ U)
                module.UT_forward = U.detach().clone()
                module.UT_backward = torch.empty(0, dtype=W.dtype, device=W.device)
                module.flag = False                      # original forward during detection
                module.coeff_mask.data.zero_()
                if hasattr(module, "mask_mode"):
                    module.mask_mode.fill_(1)

                # Store sqrt(S) for score normalization. S has same dim as U's columns.
                S = self.basis_data[key]["S"]
                if S is not None:
                    module._warp_sn_S_sqrt = S.to(device=W.device, dtype=torch.float32).clamp(min=1.0).sqrt()
                else:
                    module._warp_sn_S_sqrt = None

                reparameterized += 1
                if layer_idx < 2:
                    logger.info(f"  Layer {layer_idx:02d} {layer_type:10}: "
                                f"W{tuple(W.shape)} @ U{tuple(U.shape)} -> basis_coeff{tuple(module.basis_coeff.shape)}")

        logger.info(f"  Reparameterized modules : {reparameterized}")
        logger.info(f"  Skipped modules         : {skipped}")
        if ortho_warn:
            logger.warning(f"  Orthogonality warnings  : {ortho_warn}")
        logger.info("  flag=False: detection forward is identical to the original model.")
        logger.info("=" * 80)
        self._reparameterized = True

    @torch.no_grad()
    def detect(self, prompts: List[str]) -> Dict[str, Dict[int, Set[int]]]:
        """
        For each prompt, select per-module top-k basis columns, then select columns
        whose frequency across prompts is >= freq_threshold.
        """
        if not self._reparameterized:
            raise RuntimeError("Call reparameterize() before detect().")

        logger.info("=" * 80)
        logger.info("[WaRP-SN Detection] Starting detection")
        logger.info(f"  prompts       : {len(prompts)}")
        logger.info(f"  top_k_ffn     : {self.top_k_ffn}")
        logger.info(f"  top_k_attn    : {self.top_k_attn}")
        logger.info(f"  score_mode    : {self.score_mode}")
        logger.info(f"  freq_threshold: {self.freq_threshold}")
        logger.info("=" * 80)

        self.model.eval()
        freq: Dict[Tuple[int, str], torch.Tensor] = {}
        success, failed = 0, 0

        first_stats_logged = False
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="WaRP-SN detection")):
            scores = self._forward_and_score(prompt, prompt_idx)
            if not scores:
                failed += 1
                continue
            success += 1

            for key, score in scores.items():
                top_k = self.top_k_ffn if "ffn" in key[1] else self.top_k_attn
                k = min(int(top_k), score.numel())
                if k <= 0:
                    continue
                idx = torch.topk(score, k=k).indices
                if key not in freq:
                    freq[key] = torch.zeros(score.numel(), dtype=torch.int32)
                freq[key][idx] += 1

            if self.log_score_stats and not first_stats_logged:
                first_stats_logged = True
                for key in list(scores.keys())[:5]:
                    s = scores[key]
                    logger.info(f"  [Score stats prompt 0] {key}: "
                                f"min={s.min().item():.3e}, mean={s.mean().item():.3e}, "
                                f"max={s.max().item():.3e}, nonzero={(s>0).sum().item()}/{s.numel()}")

            if (prompt_idx + 1) % 100 == 0 or prompt_idx == 0:
                logger.info(f"  [{prompt_idx+1:>5}/{len(prompts)}] success={success}, failed={failed}")

        if success == 0:
            raise RuntimeError("All detection forwards failed. Check tokenizer/model/device.")

        logger.info(f"  Detection forward complete: success={success}, failed={failed}")

        min_count = int(np.ceil(self.freq_threshold * success))
        logger.info(f"  Frequency selection: count >= {min_count}/{success}")

        selected: Dict[Tuple[int, str], Set[int]] = {}
        for key, counts in sorted(freq.items()):
            cols = torch.nonzero(counts >= min_count, as_tuple=False).view(-1).tolist()
            selected[key] = set(int(c) for c in cols)

        output = self._to_output_format(selected)

        logger.info("\nWaRP-SN Detection Results (COLUMN indices of basis_coeff)")
        logger.info("=" * 80)
        total = 0
        for lt in self.layer_types:
            sn_key = _LAYER_TYPE_TO_SN_KEY.get(lt, lt)
            if sn_key not in output:
                continue
            type_total = sum(len(output[sn_key].get(li, set())) for li in range(self.num_layers))
            total += type_total
            logger.info(f"  {lt:12}: {type_total:8d} selected columns across {self.num_layers} layers")
        logger.info(f"  {'TOTAL':12}: {total:8d}")
        if total == 0:
            logger.warning("  No safety columns selected. Lower freq_threshold or increase top_k.")
        logger.info("=" * 80)
        return output

    def _forward_and_score(self, prompt: str, prompt_idx: int) -> Optional[Dict[Tuple[int, str], torch.Tensor]]:
        scores: Dict[Tuple[int, str], torch.Tensor] = {}
        hooks = []

        def make_hook(key: Tuple[int, str]):
            def hook_fn(module: LinearWaRP, inp, out):
                try:
                    if module.UT_forward is None or module.UT_forward.numel() == 0:
                        return
                    x = inp[0]  # [B,T,d_in] or [BT,d_in]
                    U = module.UT_forward
                    d_in = x.shape[-1]
                    if d_in != U.shape[0]:
                        logger.debug(f"[{key}] shape mismatch: x={d_in}, U={U.shape[0]}")
                        return

                    h = x.detach().float().reshape(-1, d_in)
                    # Project input activations to safety basis.
                    h_basis = h @ U.float()  # [T, d_in]
                    act_score = h_basis.abs().sum(dim=0)  # [d_in]

                    # Normalize by sqrt(S_k) to remove eigenvalue ordering.
                    # Without this, U's columns are sorted by singular value so
                    # column 0 always dominates → top-k always returns 0,1,2,...
                    # After normalization each column has equal expected baseline,
                    # so detection finds directions activated *above expectation*.
                    S_sqrt = getattr(module, "_warp_sn_S_sqrt", None)
                    if S_sqrt is not None:
                        act_score = act_score / S_sqrt.to(act_score.device)

                    if self.score_mode == "activation":
                        score = act_score
                    else:
                        # Output-aware contribution proxy. basis_coeff is [d_out, d_in].
                        # Also normalize w_col by S_sqrt so it does not re-introduce ordering.
                        w_col = module.basis_coeff.detach().float().abs().sum(dim=0)
                        if S_sqrt is not None:
                            w_col = w_col / S_sqrt.to(w_col.device)
                        score = act_score * w_col

                    scores[key] = score.detach().cpu()
                except Exception as e:
                    logger.warning(f"[Hook failure] {key}: {repr(e)}")
            return hook_fn

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                key = (layer_idx, layer_type)
                if key not in self.basis_data:
                    continue
                module = self._get_module(layer, layer_type)
                if isinstance(module, LinearWaRP):
                    hooks.append(module.register_forward_hook(make_hook(key)))

        try:
            enc = self._tokenize(prompt)
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            dev = next(self.model.parameters()).device
            enc = {k: v.to(dev) for k, v in enc.items()}
            _ = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        except Exception as e:
            logger.warning(f"[Prompt {prompt_idx}] forward failed: {repr(e)}")
            return None
        finally:
            for h in hooks:
                h.remove()

        return scores if scores else None

    def _tokenize(self, prompt: str) -> dict:
        if self.is_instruct:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            )
            return {"input_ids": input_ids}
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )

    def _to_output_format(self, selected: Dict[Tuple[int, str], Set[int]]) -> Dict[str, Dict[int, Set[int]]]:
        output: Dict[str, Dict[int, Set[int]]] = {
            "ffn_up":   {i: set() for i in range(self.num_layers)},
            "ffn_down": {i: set() for i in range(self.num_layers)},
            "q":        {i: set() for i in range(self.num_layers)},
            "k":        {i: set() for i in range(self.num_layers)},
            "v":        {i: set() for i in range(self.num_layers)},
        }
        # ffn_gate is supported internally but not written to 5-line format unless pipeline handles it.
        for (layer_idx, layer_type), cols in selected.items():
            sn_key = _LAYER_TYPE_TO_SN_KEY.get(layer_type)
            if sn_key in output:
                output[sn_key][layer_idx] = set(cols)
        return output

    def _get_module(self, layer, layer_type: str):
        sub, attr = _LAYER_TYPE_TO_PATH[layer_type]
        return getattr(getattr(layer, sub), attr)

    @staticmethod
    def save_safety_neurons(safety_neurons: Dict[str, Dict[int, Set[int]]], output_file: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for key in _SN_KEY_ORDER_5LINE:
                d = safety_neurons.get(key, {})
                serialized = {str(li): sorted([int(x) for x in s]) for li, s in d.items()}
                f.write(json.dumps(serialized) + "\n")
        meta_file = output_file + ".meta.json"
        total = sum(len(s) for d in safety_neurons.values() for s in d.values())
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "format": "5-line WaRP-SN column-index file",
                "keys": _SN_KEY_ORDER_5LINE,
                "total_selected_columns": total,
                "note": "Indices are COLUMN indices of basis_coeff = W @ U, not original SN row neurons."
            }, f, indent=2)
        logger.info(f"[WaRP-SN] Safety columns saved -> {output_file}")
        logger.info(f"[WaRP-SN] Metadata saved       -> {meta_file}")

    @staticmethod
    def load_safety_neurons(neuron_file: str) -> Dict[str, Dict[int, Set[int]]]:
        with open(neuron_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 5:
            raise ValueError(f"Expected at least 5 lines in neuron file, got {len(lines)}: {neuron_file}")
        result: Dict[str, Dict[int, Set[int]]] = {}
        for i, key in enumerate(_SN_KEY_ORDER_5LINE):
            raw = json.loads(lines[i])
            result[key] = {int(li): set(int(x) for x in v) for li, v in raw.items()}
        return result
