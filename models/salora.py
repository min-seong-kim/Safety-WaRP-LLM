"""SaLoRA (ICLR'25, "SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation") module.

원본 구현 /home/gokms0509/SaLoRA (lora_train_act.py + peft fork) 의 알고리즘을,
이 레포의 lora_wsr_elementwise.LinearWSRLoRA 관례에 맞춰 커스텀 nn.Module 로 재구현한다.
표준 peft 를 오염시키지 않는 self-contained 구현이며, finetune_gsm8k_salora.py 가 이를 사용한다.

핵심 아이디어 (target linear W ∈ R^{out×in}):
    1. safety 데이터의 입력 activation X_s 로부터 "안전 출력 부분공간" V_s (out×rs) 를 구해
       고정 투영행렬  C = I − V_s V_sᵀ  (out×out) 를 만든다. LoRA 증분의 출력을 C 로 통과시키면
       증분이 safety 부분공간에 성분을 가질 수 없어, downstream 학습이 안전 정렬을 못 건드린다.
    2. utility 데이터로 "유용 출력 부분공간" V_u (out×du) 를 구한다.
    3. PiSSA: W 의 top-r SVD  W ≈ U2 S2 V2ᵀ 로
           B = V_u V_uᵀ · (U2 √S2)   (out×r, utility 부분공간으로 투영)
           A = √S2 · V2ᵀ             (r×in)
       를 초기화하고, residual  W_res = W − s·C·B·A  로 base 를 치환한다 (s=α/r).
       → 시작 시 유효 weight = W_res + s·C·B·A = W (정확히 시작 모델).
    forward:  y = W_res·x + s·(B·A·x)·Cᵀ
    학습:     A, B 만 학습. W_res, C 는 frozen.
    저장:     restore_salora_to_linear() 가 W = W_res + s·C·B·A 로 dense fold → nn.Linear.

원본과의 차이(수학적으로 동등):
- 원본은 activation 을 CPU 에 전부 쌓아 svd_lowrank(X·Wᵀ) 로 V 를 구한다. 여기서는 Gram
  G = Xᵀ X 를 online 누적해  V = top-eig(W·G·Wᵀ) = svd_lowrank(W·G·Wᵀ) 로 구한다.
  (X·Wᵀ)ᵀ(X·Wᵀ) = W·(Xᵀ X)·Wᵀ 이므로 좌특이/우특이 벡터가 동일 → 결과 동일, 메모리는 크게 절약.
- 원본은 α=r(s=1) 을 가정한다. 여기서는 s=α/r 을 residual/forward/fold 에 일관 적용하므로
  임의의 α 에서도 시작 유효 weight = W 가 보장된다. 원본을 그대로 재현하려면 --lora_alpha == --lora_r.

주의(메모리): Gram 은 module 당 in×in fp32 (7B 의 4096² ≈ 67MB) 를 GPU 에 상주시킨다. target
module 이 많으면(layer_type 5종 × 32층) 무거워지므로, 원본처럼 attn_q,attn_v 정도로 좁히길 권장한다.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# layer_type -> (parent attr path, module attr name)  (LLaMA) — lora_wsr_elementwise 와 동일
_LAYER_TYPE_TO_ATTR = {
    "ffn_down": ("mlp", "down_proj"),
    "ffn_up": ("mlp", "up_proj"),
    "ffn_gate": ("mlp", "gate_proj"),
    "attn_q": ("self_attn", "q_proj"),
    "attn_k": ("self_attn", "k_proj"),
    "attn_v": ("self_attn", "v_proj"),
    "attn_o": ("self_attn", "o_proj"),
}


class LinearSaLoRA(nn.Module):
    """SaLoRA linear: y = W_res·x + s·(B·A·x)·Cᵀ.

    trainable: lora_A (r×in), lora_B (out×r). weight(W_res), lora_C(out×out) 는 frozen buffer.
    교체 직후(초기)에는 W_res=W, C=I, B=0 이라 forward 가 원본과 정확히 동일하다 →
    init_salora_from_activations() 가 이 상태에서 activation(Gram) 을 수집한 뒤 C/A/B/W_res 를 채운다.
    """

    def __init__(self, linear_layer: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert isinstance(linear_layer, nn.Linear), "Layer must be nn.Linear"

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        weight = linear_layer.weight.data.clone()
        dtype = weight.dtype
        device = weight.device

        # W_res (frozen). 초기엔 원본 W. init 후 residual 로 교체된다.
        self.register_buffer("weight", weight)
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

        # 고정 safety 투영행렬 C (out×out). 초기 항등.
        self.register_buffer("lora_C", torch.eye(self.out_features, dtype=dtype, device=device))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

        # LoRA factors. 초기 B=0 → ΔW=0. init_salora_from_activations 에서 PiSSA 값으로 덮어씀.
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self._initialized = False

        # ── activation(Gram) 수집용 상태 ──
        self._record = False
        self._mask = None            # (B,T) bool. response 토큰만 집계.
        self._gram = None            # (in×in) fp32, GPU 누적

    # ───────────── activation(Gram) 수집 ─────────────
    def clear_gram(self):
        self._gram = None

    @torch.no_grad()
    def _accumulate_gram(self, x: torch.Tensor):
        if self._mask is not None:
            x_ = x[self._mask]                       # (n, in)
        else:
            x_ = x.reshape(-1, x.shape[-1])          # (B*T, in)
        if x_.numel() == 0:
            return
        xf = x_.float()
        g = xf.t() @ xf                              # (in, in)
        if self._gram is None:
            self._gram = g
        else:
            self._gram += g

    @torch.no_grad()
    def set_salora_weights(self, weight_res, lora_C, A, B):
        """init 에서 계산한 residual base / 투영행렬 / PiSSA factor 주입."""
        self.weight.copy_(weight_res.to(self.weight.dtype))
        self.lora_C = lora_C.to(dtype=self.weight.dtype, device=self.weight.device)
        self.lora_A.data.copy_(A.to(self.lora_A.dtype))
        self.lora_B.data.copy_(B.to(self.lora_B.dtype))
        self._initialized = True

    # ───────────── forward / fold ─────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._record:
            self._accumulate_gram(x)
        base = F.linear(x, self.weight, self.bias)
        xd = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter = F.linear(F.linear(xd, self.lora_A), self.lora_B) * self.scaling   # s·(B·A·x)
        adapter = adapter @ self.lora_C.t().to(adapter.dtype)                       # ·Cᵀ
        return base + adapter.to(base.dtype)

    @torch.no_grad()
    def folded_weight(self) -> torch.Tensor:
        """저장용 dense weight: W_res + s·C·B·A."""
        delta = self.scaling * (self.lora_C.float() @ self.lora_B.float() @ self.lora_A.float())
        return (self.weight.float() + delta).to(self.weight.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}"
        )


def _resolve_target_layers(num_layers, target_layers):
    if isinstance(target_layers, str):
        if target_layers == "all":
            return list(range(num_layers))
        if "-" in target_layers:
            s, e = map(int, target_layers.split("-"))
            return list(range(s, e + 1))
        return [int(target_layers)]
    return list(target_layers)


def switch_to_salora(model, layer_types, target_layers, r, alpha, dropout=0.0):
    """지정 레이어의 nn.Linear 를 LinearSaLoRA 로 교체. (C/A/B 는 이후 init 에서 주입)

    반환: {(layer_idx, layer_type): LinearSaLoRA}
    """
    num_layers = len(model.model.layers)
    indices = _resolve_target_layers(num_layers, target_layers)
    converted = {}
    for layer_idx in indices:
        layer = model.model.layers[layer_idx]
        for lt in layer_types:
            if lt not in _LAYER_TYPE_TO_ATTR:
                raise ValueError(f"Unknown layer type: {lt}")
            parent_name, attr = _LAYER_TYPE_TO_ATTR[lt]
            parent = getattr(layer, parent_name)
            orig = getattr(parent, attr)
            sa = LinearSaLoRA(orig, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, sa)
            converted[(layer_idx, lt)] = sa
    logger.info(f"✓ Converted {len(converted)} modules to LinearSaLoRA")
    return converted


@torch.no_grad()
def _collect_gram(model, modules, batches, device):
    """batches 를 forward 하며 각 module 의 Gram(=response 토큰 XᵀX) 을 누적."""
    for m in modules:
        m._record = True
        m.clear_gram()
    for batch in batches:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask = batch["labels"].to(device).ne(-100)      # response 토큰만
        for m in modules:
            m._mask = mask
        model(input_ids=ids, attention_mask=attn)
    for m in modules:
        m._record = False
        m._mask = None


@torch.no_grad()
def _top_eigvecs(gram_sym, q, niter=20):
    """대칭 PSD 행렬 M 의 top-q 고유벡터 (out×q). M = W·G·Wᵀ 에 사용."""
    q = min(q, gram_sym.shape[0] - 1)
    U, S, V = torch.svd_lowrank(gram_sym, q=q, niter=niter)
    return U[:, :q].contiguous()


@torch.no_grad()
def init_salora_from_activations(model, converted, safety_batches, util_batches,
                                 r, rank_safe, rank_util, device, niter=20, logger=None):
    """SaLoRA 초기화: safety→C, utility+PiSSA→A,B, residual W_res 주입.

    safety_batches / util_batches: collated dict({input_ids, attention_mask, labels}) 리스트.
    labels != -100 (response) 토큰의 activation 만 부분공간 추정에 사용한다(원본 disentangle 모드).
    """
    log = logger or logging.getLogger(__name__)

    # ── 메모리 절감: layer_type 그룹별로 순차 처리 ──
    #   5모듈 예산-맞춤 시 모든 모듈 Gram(down_proj in=11008 → 484MB×32≈15.5GB 등 총 ~23.5GB fp32)을
    #   동시에 GPU 에 올리면 48GB 도 초과(OOM). 그룹별로 나눠 한 번에 한 그룹 Gram 만 상주시킨다.
    #   SaLoRA 불변식(초기화 후 각 모듈 forward == 원본 W·x; C 대칭 + PiSSA residual)이 성립하므로
    #   앞 그룹을 먼저 초기화해도 뒤 그룹 calibration activation 이 바뀌지 않아 순차 처리가 정답을 보존한다.
    #   Gram(∝in²)이 가장 큰 그룹(ffn_down)을 먼저 처리해 뒤 그룹 C 버퍼 누적과 peak 이 겹치지 않게 한다.
    from collections import OrderedDict
    groups = OrderedDict()
    for key, m in converted.items():
        groups.setdefault(key[1], {})[key] = m
    ordered_lts = sorted(groups.keys(),
                         key=lambda lt: next(iter(groups[lt].values())).in_features, reverse=True)
    log.info(f"[SaLoRA] group-wise init order (in_features desc): {ordered_lts}")

    for lt in ordered_lts:
        group = groups[lt]
        gmods = list(group.values())
        # 1) safety pass → V_s → C = I − V_s V_sᵀ
        log.info(f"[SaLoRA] group '{lt}': {len(gmods)} modules — safety pass ({len(safety_batches)} batches)")
        _collect_gram(model, gmods, safety_batches, device)
        safety_V = {}
        for key, m in group.items():
            W = m.weight.data.float()                        # 아직 원본 W (residual 전)
            M = W @ m._gram.to(device) @ W.t()               # (out×out) = (X·Wᵀ)ᵀ(X·Wᵀ)
            safety_V[key] = _top_eigvecs(M, rank_safe, niter).float()
            m.clear_gram()
            del M

        # 2) utility pass → V_u, PiSSA → A,B,W_res
        log.info(f"[SaLoRA] group '{lt}': utility pass ({len(util_batches)} batches)")
        _collect_gram(model, gmods, util_batches, device)
        for key, m in group.items():
            W = m.weight.data.float()
            Mu = W @ m._gram.to(device) @ W.t()              # (out×out)
            Vu = _top_eigvecs(Mu, rank_util, niter).float()  # (out×du) utility 출력 부분공간
            del Mu

            # PiSSA: W 의 top-r SVD
            U2, S2, V2 = torch.svd_lowrank(W, q=r, niter=niter)   # U2(out×r) S2(r) V2(in×r)
            sqrtS = torch.sqrt(S2)
            B = U2 @ torch.diag(sqrtS)                       # (out×r)
            B = Vu @ (Vu.t() @ B)                            # utility 부분공간으로 투영
            A = torch.diag(sqrtS) @ V2.t()                   # (r×in)

            Vs = safety_V[key]
            C = torch.eye(m.out_features, device=device) - Vs @ Vs.t()   # (out×out)
            W_res = W - m.scaling * (C @ B @ A)              # residual base

            m.set_salora_weights(W_res, C, A, B)
            m.clear_gram()
            del W, Vu, U2, S2, V2, B, A, C, W_res
        del safety_V
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(f"✓ SaLoRA initialized: rank_safe={rank_safe}, rank_util={rank_util}, r={r}")


def mark_only_salora_trainable(model):
    """lora_A/lora_B 만 학습, 나머지 전부 freeze. trainable param 수 반환."""
    trainable = 0
    for name, p in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B") or ".lora_A" in name or ".lora_B" in name:
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
    return trainable


@torch.no_grad()
def restore_salora_to_linear(model):
    """LinearSaLoRA → nn.Linear (dense fold: W_res + s·C·B·A). 저장 전 호출."""
    replacements = []
    for module in model.modules():
        for attr, child in module.named_children():
            if isinstance(child, LinearSaLoRA):
                replacements.append((module, attr, child))
    for parent, attr, sa in replacements:
        w = sa.folded_weight()
        new_linear = nn.Linear(
            sa.in_features, sa.out_features,
            bias=sa.bias is not None, dtype=w.dtype, device=w.device,
        )
        new_linear.weight = nn.Parameter(w.clone())
        if sa.bias is not None:
            new_linear.bias = nn.Parameter(sa.bias.data.clone())
        setattr(parent, attr, new_linear)
    logger.info(f"✓ Restored {len(replacements)} LinearSaLoRA → nn.Linear")
    return model
