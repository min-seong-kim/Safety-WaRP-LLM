"""
WaRP-Rotated Safe LoRA

Safe LoRA에 WaRP Phase 1 Safety Basis Rotation을 적용.

핵심 아이디어
─────────────
원본 Safe LoRA:
    V      = W_aligned − W_base          (alignment delta)
    C      = V V^T / ‖V‖_F              (output-space projector)
    B_proj = C @ B                       (lora_B에 projection 적용)

WaRP-Rotated Safe LoRA:
    U_k    = WaRP Phase-1 basis의 top-k columns  ∈ R^[in_dim, k]
             (safety data가 가장 많이 활성화하는 input 방향들)
    V_rot  = V @ U_k                     ∈ R^[out_dim, k]
             (alignment delta를 safety-relevant input subspace로 필터링)
    C_rot  = V_rot V_rot^T / ‖V_rot‖_F  (rotated output-space projector)
    B_proj = C_rot @ B

일관성 (전 과정 동일 rotated space)
────────────────────────────────────
  1) alignment delta  V_rot = V @ U_k              ∈ R^[d_out, k]
  2) LoRA update      A_rot = A @ U_k  (원본 공간 훈련 시)
                      A_rot = A        (rotated 공간 훈련 시, A.shape[1]==k)
                      ΔW_rot = B @ A_rot             ∈ R^[d_out, k]
  3) cosine (선택 기준)
     cos_rot = cos(C_rot @ B @ A_rot,  B @ A_rot)  ← 동일 rotated space
     cos_orig= cos(C_orig @ B @ A,     B @ A)       ← 원본 space (비교용)
  4) projection       B_proj = C_rot @ B            (output space, 동일)

로그로 확인하는 것들
────────────────────
- ‖C_rot − C_orig‖_F : 두 projector의 차이
- cos_orig: 원본 공간 cosine  (비교용)
- cos_rot:  rotated 공간 cosine  ← C_rot와 동일한 공간, 완전 일관성
- delta_shift: rotated 공간 기준 projection 이동량
- 어떤 레이어가 선택되었는지 (layer-wise 비교표)
"""

import os
import sys
import gc
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# LoRA module name → WaRP layer_type 매핑
# ─────────────────────────────────────────────────────────────────────────────
LORA_TO_WARP_TYPE: Dict[str, str] = {
    "q_proj":    "attn_q",
    "k_proj":    "attn_k",
    "v_proj":    "attn_v",
    "o_proj":    "attn_o",
    "gate_proj": "ffn_gate",
    "up_proj":   "ffn_up",
    "down_proj": "ffn_down",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WaRPSafeLoRAConfig:
    """
    WaRP-Rotated Safe LoRA 설정.

    base_model_path    : unaligned base model (e.g. meta-llama/Llama-2-7b-hf)
    aligned_model_path : safety-tuned model   (e.g. kmseong/llama2_7b-Safety-FT-lr3e-5)
    basis_dir          : WaRP Phase 1 basis 저장 경로
                         (phase1_<ts>/basis/ 형식, 내부에 attn_q/, ffn_up/ 등의 서브디렉토리)
    top_k              : 사용할 basis vector 수. None이면 top_k_ratio로 결정.
    top_k_ratio        : top_k가 None일 때 in_dim * top_k_ratio개 사용 (기본 0.5)
    select_layers_type : "threshold" or "number"
    threshold          : cosine similarity 기준 (threshold 모드)
    num_proj_layers    : 투영할 레이어 수 (number 모드)
    use_approximation  : True → approximate projector V V^T / ‖V‖
                         False → exact pinv-based projector
    projection_eps     : numerical stability
    devices            : "cuda" or "cpu"
    hf_token           : HuggingFace auth token (private 모델용)
    """
    base_model_path:    str = field(default=None)
    aligned_model_path: str = field(default=None)
    basis_dir:          str = field(default=None)
    top_k:              Optional[int]   = field(default=None)
    top_k_ratio:        float           = field(default=0.5)
    select_layers_type: str             = field(default="threshold")
    threshold:          float           = field(default=0.5)
    num_proj_layers:    int             = field(default=10)
    use_approximation:  bool            = field(default=True)
    projection_eps:     float           = field(default=1e-8)
    devices:            str             = field(default="cuda")
    hf_token:           Optional[str]   = field(default=None)

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")
        if self.basis_dir is None:
            raise ValueError("basis_dir cannot be None.")
        if not os.path.isdir(self.basis_dir):
            raise ValueError(f"basis_dir does not exist: {self.basis_dir}")
        if self.select_layers_type not in {"threshold", "number"}:
            raise ValueError("select_layers_type must be 'threshold' or 'number'.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _hf_auth_kwargs(token: Optional[str]) -> Dict:
    return {"token": token} if token else {}


def _build_projection_matrix(
    delta: torch.Tensor,
    use_approximation: bool,
    eps: float,
) -> torch.Tensor:
    """
    delta ∈ R^[out, any]
    반환: C ∈ R^[out, out]

    approximate: C = delta @ delta^T / ‖delta‖_F
    exact:       C = delta @ pinv(delta^T @ delta) @ delta^T
    """
    delta = delta.float()
    norm = torch.linalg.matrix_norm(delta, ord="fro")
    if norm.item() <= eps:
        raise ValueError("Near-zero delta; cannot build projection matrix.")

    if use_approximation:
        return (delta @ delta.T) / (norm + eps)
    else:
        gram = delta.T @ delta
        gram_pinv = torch.linalg.pinv(gram)
        return delta @ gram_pinv @ delta.T


def _cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), eps=eps).item()


def _parse_lora_prefix(lora_prefix: str) -> Tuple[Optional[int], Optional[str]]:
    """
    lora_prefix 예시:
        "base_model.model.model.layers.10.self_attn.q_proj"
        "base_model.model.model.layers.0.mlp.gate_proj"

    반환: (layer_idx, warp_layer_type)
    """
    parts = lora_prefix.split(".")
    layer_idx = None
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                break
            except ValueError:
                pass
    if layer_idx is None:
        return None, None

    module_name = parts[-1]
    warp_type = LORA_TO_WARP_TYPE.get(module_name)
    return layer_idx, warp_type


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class WaRPSafeLoRA:
    """
    Safe LoRA + WaRP Basis Rotation.

    사용법:
        config = WaRPSafeLoRAConfig(
            base_model_path    = "meta-llama/Llama-2-7b-hf",
            aligned_model_path = "kmseong/llama2_7b-Safety-FT-lr3e-5",
            basis_dir          = "./checkpoints/phase1_<ts>/basis",
            top_k_ratio        = 0.5,
            select_layers_type = "threshold",
            threshold          = 0.5,
        )
        warp_safelora = WaRPSafeLoRA(peft_model, config, logger=logger)
        safe_model = warp_safelora.model
    """

    def __init__(
        self,
        peft_model: torch.nn.Module,
        config: WaRPSafeLoRAConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.peft_model = peft_model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device(
            config.devices
            if config.devices == "cpu" or torch.cuda.is_available()
            else "cpu"
        )

        # ── 1. WaRP basis 로드 ──────────────────────────────────────────────
        self.basis: Dict[Tuple[int, str], torch.Tensor] = {}
        self._load_warp_basis()

        # ── 2. LoRA 파라미터 수집 ───────────────────────────────────────────
        self.original_lora_params: Dict[str, torch.Tensor] = {}
        self.lora_modules: Dict[str, Dict[str, str]] = {}
        self._collect_lora_info()

        # ── 3. Alignment delta & projector 계산 ─────────────────────────────
        # 원본 projector (비교용)
        self.orig_projectors:    Dict[str, torch.Tensor] = {}
        # WaRP-rotated projector (실제 적용)
        self.rotated_projectors: Dict[str, torch.Tensor] = {}
        # cosine 일관성을 위해 _compute_metrics에서 사용할 U_k (per lora_prefix)
        self.U_k_per_layer:      Dict[str, torch.Tensor] = {}
        self._build_projectors()

        # ── 4. Layer metrics 계산 (cosine, delta_shift) ─────────────────────
        self.metrics: Dict[str, Dict] = {}
        self._compute_metrics()

        # ── 5. Layer 선택 ───────────────────────────────────────────────────
        self.selected_modules: List[str] = []
        self._select_modules()

        # ── 6. Projection 적용 ──────────────────────────────────────────────
        self._apply_projection()

        # ── 7. 최종 요약 로그 ───────────────────────────────────────────────
        self._log_summary()

        self.model = peft_model

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: WaRP basis 로드
    # ─────────────────────────────────────────────────────────────────────────
    def _load_warp_basis(self):
        """
        basis_dir/{layer_type}/layer_{idx:02d}_svd.pt 로드.
        각 파일의 'U' 키 = WaRP rotation matrix ∈ R^[in_dim, in_dim].
        columns은 descending eigenvalue 순으로 정렬되어 있음.
        """
        self.logger.info("=" * 70)
        self.logger.info("[WaRPSafeLoRA] Loading WaRP basis from: %s", self.config.basis_dir)

        # metadata
        meta_path = os.path.join(self.config.basis_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.logger.info("  Basis metadata: model=%s, layer_types=%s",
                             meta.get("model_name", "?"), meta.get("layer_types", "?"))

        loaded = 0
        for layer_type in os.listdir(self.config.basis_dir):
            layer_type_dir = os.path.join(self.config.basis_dir, layer_type)
            if not os.path.isdir(layer_type_dir):
                continue
            for fname in sorted(os.listdir(layer_type_dir)):
                if not fname.endswith("_svd.pt"):
                    continue
                m = re.match(r"layer_(\d+)_svd\.pt", fname)
                if m is None:
                    continue
                layer_idx = int(m.group(1))
                svd_data = torch.load(
                    os.path.join(layer_type_dir, fname), map_location="cpu"
                )
                # 'U' = WaRP의 rotation matrix V (right singular vectors)
                U = svd_data["U"]
                self.basis[(layer_idx, layer_type)] = U
                loaded += 1

        self.logger.info("  Loaded %d basis matrices.", loaded)
        if loaded == 0:
            raise RuntimeError(
                f"No basis matrices found in {self.config.basis_dir}. "
                "Run Phase 1 first."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: LoRA 파라미터 수집
    # ─────────────────────────────────────────────────────────────────────────
    def _collect_lora_info(self):
        for name, param in self.peft_model.named_parameters():
            if ".lora_" in name and name.endswith(".weight"):
                self.original_lora_params[name] = param.detach().clone().cpu()

            if ".lora_A." in name and name.endswith(".weight"):
                prefix = name.split(".lora_A.", 1)[0]
                self.lora_modules.setdefault(prefix, {})["A"] = name
            elif ".lora_B." in name and name.endswith(".weight"):
                prefix = name.split(".lora_B.", 1)[0]
                self.lora_modules.setdefault(prefix, {})["B"] = name

        self.logger.info("  Collected %d LoRA module pairs.", len(self.lora_modules))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Projector 계산
    # ─────────────────────────────────────────────────────────────────────────
    def _build_projectors(self):
        """
        base / aligned 모델을 CPU로 로드하여 alignment delta V를 계산.
        각 LoRA 레이어에 대해:
          - 원본 projector : C_orig = V V^T / ‖V‖_F
          - 회전 projector : C_rot  = (V U_k)(V U_k)^T / ‖V U_k‖_F
        """
        self.logger.info("")
        self.logger.info("[WaRPSafeLoRA] Building alignment projectors...")

        load_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        self.logger.info("  Loading base model from %s (cpu, %s)...",
                         self.config.base_model_path, load_dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=load_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(self.config.hf_token),
        )
        self.logger.info("  Loading aligned model from %s (cpu, %s)...",
                         self.config.aligned_model_path, load_dtype)
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            torch_dtype=load_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(self.config.hf_token),
        )

        # weight map 구축
        peft_config   = self.peft_model.peft_config["default"]
        target_modules = set(peft_config.target_modules)

        base_weights = {}
        for name, param in base_model.named_parameters():
            if name.endswith(".weight") and name.split(".")[-2] in target_modules:
                base_weights[name[:-len(".weight")]] = param.detach().cpu()

        aligned_weights = {}
        for name, param in aligned_model.named_parameters():
            if name.endswith(".weight") and name.split(".")[-2] in target_modules:
                aligned_weights[name[:-len(".weight")]] = param.detach().cpu()

        del base_model, aligned_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("  Computing alignment deltas & projectors...")
        self.logger.info(
            "  top_k=%s, top_k_ratio=%.2f, use_approximation=%s",
            self.config.top_k, self.config.top_k_ratio, self.config.use_approximation,
        )
        self.logger.info("")
        self.logger.info(
            "  %-65s  %8s  %8s  %8s  %10s",
            "Module", "in_dim", "k", "‖V‖_F", "‖C_rot-C_orig‖_F",
        )
        self.logger.info("  " + "-" * 105)

        n_matched = 0
        n_no_basis = 0

        for lora_prefix in sorted(self.lora_modules):
            layer_idx, warp_type = _parse_lora_prefix(lora_prefix)
            if layer_idx is None or warp_type is None:
                self.logger.warning("  Cannot parse prefix: %s  → skipping", lora_prefix)
                continue

            # alignment delta V
            module_name_short = lora_prefix.split(".")[-1]  # e.g. "q_proj"
            # 매핑 탐색: base_weights key는 full path (without .weight)
            aligned_w = None
            base_w    = None
            for key in aligned_weights:
                if key.endswith("." + module_name_short) and f".layers.{layer_idx}." in key:
                    aligned_w = aligned_weights[key]
                    base_w    = base_weights.get(key)
                    break
            if aligned_w is None or base_w is None:
                self.logger.warning("  Weight not found for %s  → skipping", lora_prefix)
                continue

            V_align = (aligned_w - base_w).float()  # [out_dim, in_dim]
            in_dim  = V_align.shape[1]
            out_dim = V_align.shape[0]

            # WaRP basis U for this layer
            basis_key = (layer_idx, warp_type)
            has_basis  = basis_key in self.basis
            if not has_basis:
                # fallback: try without the warp_type suffix (e.g. some basis dirs)
                n_no_basis += 1
                self.logger.warning(
                    "  No WaRP basis for (%d, %s) — falling back to original projector",
                    layer_idx, warp_type,
                )
                C_orig = _build_projection_matrix(V_align, self.config.use_approximation, self.config.projection_eps)
                self.orig_projectors[lora_prefix]    = C_orig.cpu()
                self.rotated_projectors[lora_prefix] = C_orig.cpu()  # same as orig
                continue

            U_full = self.basis[basis_key].float()  # [in_dim, in_dim]
            assert U_full.shape[0] == in_dim, (
                f"Basis in_dim mismatch for {lora_prefix}: "
                f"U.shape={U_full.shape}, weight in_dim={in_dim}"
            )

            # top-k 결정
            k = self.config.top_k
            if k is None:
                k = max(1, int(in_dim * self.config.top_k_ratio))
            k = min(k, in_dim)

            U_k = U_full[:, :k]  # [in_dim, k]  (top-k columns)
            self.U_k_per_layer[lora_prefix] = U_k.cpu()  # _compute_metrics 에서 사용

            # 원본 projector
            C_orig = _build_projection_matrix(
                V_align, self.config.use_approximation, self.config.projection_eps
            )
            # 회전 projector
            V_rot = V_align @ U_k              # [out_dim, k]
            C_rot = _build_projection_matrix(
                V_rot, self.config.use_approximation, self.config.projection_eps
            )

            projector_diff = torch.linalg.matrix_norm(C_rot - C_orig, ord="fro").item()
            norm_V = torch.linalg.matrix_norm(V_align, ord="fro").item()

            self.logger.info(
                "  %-65s  %8d  %8d  %8.4f  %10.6f",
                lora_prefix[-65:], in_dim, k, norm_V, projector_diff,
            )

            self.orig_projectors[lora_prefix]    = C_orig.cpu()
            self.rotated_projectors[lora_prefix] = C_rot.cpu()
            n_matched += 1

        self.logger.info("  " + "-" * 105)
        self.logger.info(
            "  Projectors computed: %d matched, %d fallback (no basis).",
            n_matched, n_no_basis,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Metrics (cosine, delta_shift) 계산
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_metrics(self):
        """
        각 LoRA 레이어에 대해 두 projector 기준 cosine similarity를 계산.

        cos_orig : 원본 공간  cos(C_orig @ B @ A,     B @ A)            (비교용)
        cos_rot  : rotated 공간  cos(C_rot @ B @ A_rot,  B @ A_rot)     (선택 기준)

        A_rot 결정:
          - A.shape[1] == k  → 이미 rotated 공간 훈련 (WaRP rotated training 사용)
          - A.shape[1] == d_in → 원본 공간 훈련  →  A_rot = A @ U_k  로 투영

        → cos_rot와 C_rot가 동일한 rotated space에서 계산되어 완전 일관성 유지.
        """
        self.logger.info("")
        self.logger.info("[WaRPSafeLoRA] Computing per-layer metrics (rotated space)...")
        self.logger.info(
            "  %-65s  %10s  %10s  %10s  %10s  %8s",
            "Module", "cos_orig", "cos_rot", "Δcos", "delta_shift", "A_space",
        )
        self.logger.info("  " + "-" * 118)

        for lora_prefix in sorted(self.lora_modules):
            if lora_prefix not in self.rotated_projectors:
                continue

            A_name = self.lora_modules[lora_prefix]["A"]
            B_name = self.lora_modules[lora_prefix]["B"]
            A = self.original_lora_params[A_name].to(self.device).float()
            B = self.original_lora_params[B_name].to(self.device).float()

            C_orig = self.orig_projectors[lora_prefix].to(self.device).float()
            C_rot  = self.rotated_projectors[lora_prefix].to(self.device).float()

            # ── cos_orig: 원본 공간 (비교용) ──────────────────────────────────
            delta_W      = B @ A              # [d_out, d_in]
            delta_W_orig = (C_orig @ B) @ A   # [d_out, d_in]
            cos_orig = _cosine_similarity_flat(delta_W_orig, delta_W)

            # ── cos_rot: rotated 공간 (선택 기준, C_rot와 동일 공간) ───────────
            U_k = self.U_k_per_layer.get(lora_prefix)
            if U_k is not None:
                U_k = U_k.to(self.device).float()  # [d_in, k]
                k   = U_k.shape[1]

                # A가 이미 rotated 공간인지 확인 (WaRP rotated training 사용 시)
                if A.shape[1] == k:
                    A_rot     = A          # [r, k]  ← rotated 공간 훈련
                    a_space   = "rotated"
                else:
                    A_rot     = A @ U_k    # [r, k]  ← 원본 공간 훈련 → 투영
                    a_space   = f"orig→k{k}"

                delta_W_rot = B @ A_rot            # [d_out, k]
                delta_rot   = (C_rot @ B) @ A_rot  # [d_out, k]
                cos_rot     = _cosine_similarity_flat(delta_rot, delta_W_rot)
                delta_shift = torch.norm(delta_rot - delta_W_rot).item()
            else:
                # basis 없는 fallback: 원본 공간
                delta_rot   = (C_rot @ B) @ A
                cos_rot     = _cosine_similarity_flat(delta_rot, delta_W)
                delta_shift = torch.norm(delta_rot - delta_W).item()
                a_space     = "orig(no basis)"

            self.metrics[lora_prefix] = {
                "cos_orig":    cos_orig,
                "cos_rot":     cos_rot,
                "delta_cos":   cos_rot - cos_orig,
                "delta_shift": delta_shift,
            }

            self.logger.info(
                "  %-65s  %10.6f  %10.6f  %+10.6f  %10.4f  %8s",
                lora_prefix[-65:],
                cos_orig, cos_rot, cos_rot - cos_orig, delta_shift, a_space,
            )

        self.logger.info("  " + "-" * 118)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Layer 선택
    # ─────────────────────────────────────────────────────────────────────────
    def _select_modules(self):
        """cos_rot 기준으로 projection 대상 레이어 선택."""
        sorted_by_cos = sorted(
            self.metrics, key=lambda name: self.metrics[name]["cos_rot"]
        )
        if self.config.select_layers_type == "threshold":
            self.selected_modules = [
                name for name in sorted_by_cos
                if self.metrics[name]["cos_rot"] < self.config.threshold
            ]
        else:
            n = min(self.config.num_proj_layers, len(sorted_by_cos))
            self.selected_modules = sorted_by_cos[:n]

        self.logger.info("")
        self.logger.info(
            "[WaRPSafeLoRA] Layer selection: %d / %d selected  (mode=%s, %s)",
            len(self.selected_modules),
            len(self.metrics),
            self.config.select_layers_type,
            (f"threshold={self.config.threshold}"
             if self.config.select_layers_type == "threshold"
             else f"num_proj_layers={self.config.num_proj_layers}"),
        )
        self.logger.info(
            "  %-65s  %10s  %10s  %10s",
            "Module", "cos_rot", "cos_orig", "Δcos",
        )
        self.logger.info("  " + "-" * 90)
        for name in sorted_by_cos[:10]:  # 상위 10개만 출력
            m = self.metrics[name]
            selected_mark = "* SELECTED" if name in self.selected_modules else ""
            self.logger.info(
                "  %-65s  %10.6f  %10.6f  %+10.6f  %s",
                name[-65:], m["cos_rot"], m["cos_orig"], m["delta_cos"], selected_mark,
            )
        if len(sorted_by_cos) > 10:
            self.logger.info("  ... (%d more layers)", len(sorted_by_cos) - 10)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Projection 적용
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_projection(self):
        """
        선택된 레이어의 lora_B에 C_rot projection을 적용.
        선택되지 않은 레이어는 원본 값을 복원.
        """
        self.logger.info("")
        self.logger.info("[WaRPSafeLoRA] Applying WaRP-rotated projection to lora_B...")

        selected_set = set(self.selected_modules)
        n_projected  = 0
        n_restored   = 0

        with torch.no_grad():
            for name, param in self.peft_model.named_parameters():
                if name not in self.original_lora_params:
                    continue

                original = self.original_lora_params[name].to(
                    dtype=param.dtype, device=param.device
                )

                # lora_A: 항상 원본 복원
                if ".lora_A." in name:
                    param.copy_(original)
                    n_restored += 1
                    continue

                # lora_B: 선택된 레이어는 C_rot 적용
                if ".lora_B." in name:
                    lora_prefix = name.split(".lora_B.", 1)[0]

                    if lora_prefix not in selected_set:
                        param.copy_(original)
                        n_restored += 1
                        continue

                    if lora_prefix not in self.rotated_projectors:
                        param.copy_(original)
                        n_restored += 1
                        continue

                    C_rot = self.rotated_projectors[lora_prefix].to(
                        dtype=torch.float32, device=param.device
                    )
                    projected = C_rot @ original.to(dtype=torch.float32, device=param.device)
                    param.copy_(projected.to(dtype=param.dtype))
                    n_projected += 1

                    self.logger.debug(
                        "  PROJECTED  %-70s  ‖ΔB‖=%.6f",
                        name[-70:],
                        torch.norm(projected - original.float()).item(),
                    )

        self.logger.info(
            "  lora_B projected: %d, lora_A/others restored: %d",
            n_projected, n_restored,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Summary 로그
    # ─────────────────────────────────────────────────────────────────────────
    def _log_summary(self):
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("[WaRPSafeLoRA] ===== SUMMARY =====")
        self.logger.info("=" * 70)
        self.logger.info("  Base model    : %s", self.config.base_model_path)
        self.logger.info("  Aligned model : %s", self.config.aligned_model_path)
        self.logger.info("  WaRP basis    : %s", self.config.basis_dir)
        self.logger.info(
            "  top_k         : %s (ratio=%.2f)",
            self.config.top_k, self.config.top_k_ratio,
        )
        self.logger.info("  Selection     : %s", self.config.select_layers_type)

        cos_orig_vals = [m["cos_orig"] for m in self.metrics.values()]
        cos_rot_vals  = [m["cos_rot"]  for m in self.metrics.values()]
        delta_cos_vals = [m["delta_cos"] for m in self.metrics.values()]

        if cos_orig_vals:
            self.logger.info("")
            self.logger.info("  Cosine similarity (across all %d layers):", len(cos_orig_vals))
            self.logger.info(
                "    original  — mean=%.4f  min=%.4f  max=%.4f",
                sum(cos_orig_vals) / len(cos_orig_vals),
                min(cos_orig_vals), max(cos_orig_vals),
            )
            self.logger.info(
                "    rotated   — mean=%.4f  min=%.4f  max=%.4f",
                sum(cos_rot_vals) / len(cos_rot_vals),
                min(cos_rot_vals), max(cos_rot_vals),
            )
            self.logger.info(
                "    Δ(rot−orig) mean=%.4f  (positive → rotated is more conservative)",
                sum(delta_cos_vals) / len(delta_cos_vals),
            )

        self.logger.info("")
        self.logger.info(
            "  Projected layers : %d / %d",
            len(self.selected_modules), len(self.metrics),
        )

        if self.selected_modules:
            self.logger.info("  Selected (lowest cos_rot first):")
            for name in self.selected_modules:
                m = self.metrics[name]
                self.logger.info(
                    "    %s  cos_rot=%.6f  cos_orig=%.6f  Δcos=%+.6f  shift=%.4f",
                    name[-60:], m["cos_rot"], m["cos_orig"], m["delta_cos"], m["delta_shift"],
                )
        else:
            self.logger.info("  No layers selected for projection.")

        # Projector diff statistics
        diffs = []
        for lp in self.rotated_projectors:
            if lp in self.orig_projectors:
                d = torch.linalg.matrix_norm(
                    self.rotated_projectors[lp].float() - self.orig_projectors[lp].float(),
                    ord="fro",
                ).item()
                diffs.append(d)
        if diffs:
            self.logger.info("")
            self.logger.info("  ‖C_rot − C_orig‖_F  — mean=%.6f  max=%.6f  min=%.6f",
                             sum(diffs) / len(diffs), max(diffs), min(diffs))
            n_different = sum(1 for d in diffs if d > 1e-4)
            self.logger.info(
                "  Projectors significantly different (‖diff‖>1e-4): %d / %d",
                n_different, len(diffs),
            )
            if n_different == 0:
                self.logger.warning(
                    "  ⚠  ALL projectors are identical to original! "
                    "Check if basis U is the correct rotation matrix."
                )

        self.logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # Public: stats dict (for saving logs)
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def stats(self) -> Dict:
        return {
            "selected_modules": self.selected_modules,
            "metrics": self.metrics,
            "num_projected_layers": len(self.selected_modules),
            "num_candidate_layers": len(self.metrics),
            "selection_mode": self.config.select_layers_type,
            "threshold": self.config.threshold,
            "num_proj_layers": self.config.num_proj_layers,
            "top_k": self.config.top_k,
            "top_k_ratio": self.config.top_k_ratio,
        }
