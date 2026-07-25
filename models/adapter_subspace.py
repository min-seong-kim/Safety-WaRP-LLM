"""adapter_subspace_lora (= safety_adapter_projected_lora) 의 핵심 수치 모듈.

아이디어
────────
safety-tuned 전체 모델에서 WSR importance 를 다시 재는 대신, 이미 학습된 **safety LoRA
adapter 가 만든 low-rank update 의 input-side subspace** 를 직접 추출하고, downstream LoRA 가
그 subspace 를 건드리지 못하게 한다.

    ΔW_s   = s_s · B_s A_s = P_s Σ_s Q_sᵀ        (compact SVD)
    Q_S    = Q_s[:, :k]                           (보호할 right singular vectors)
    ΔW_d^⊥ = s_d · B_d A_d (I − Q_S Q_Sᵀ)
    W_final = W_base + ΔW_s + ΔW_d^⊥

    ⇒  ΔW_d^⊥ Q_S = 0   ⇒  W_final Q_S = W_safe Q_S   (정확)

right singular vectors 를 쓰는 이유는 projection 을 오른쪽(입력 쪽)에 걸기 때문이다.
left singular vectors P_s 는 출력 쪽 방향이라 이 제약에는 쓰이지 않는다.

safety adapter (B_s, A_s) 는 고정한다. 이 방법은 safety adapter 를 수정하거나 pruning 하지
않으며, 새로 학습하는 것은 downstream adapter (B_d, A_d) 뿐이다.

수치 원칙
─────────
dense ΔW_s (m×n) 는 **어디에서도 만들지 않는다.** thin QR 두 번 + r×r SVD 한 번으로 exact
compact SVD 를 얻고, 검증용 Frobenius 노름도 r×r Gram 만으로 exact 하게 계산한다
(`relative_reconstruction_error`, `lowrank_fro_norm` 참조).
모든 분해 연산은 fp32 에서 수행한다 (bf16 adapter 를 그대로 SVD 하지 않는다).
"""

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

EPS = 1e-12

# LLaMA projection 이름 ↔ 이 레포의 layer_type 표기 (finetune_gsm8k_lora.py 와 동일)
PROJ_TO_LT = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
              "up_proj": "ffn_up", "down_proj": "ffn_down",
              "gate_proj": "ffn_gate", "o_proj": "attn_o"}
LT_TO_PROJ = {v: k for k, v in PROJ_TO_LT.items()}


def module_name_to_key(name):
    """'...model.layers.12.self_attn.q_proj[...]' → (12, 'attn_q'). 실패 시 None."""
    parts = name.split(".")
    if "layers" not in parts:
        return None
    try:
        layer_idx = int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None
    for proj, lt in PROJ_TO_LT.items():
        if proj in parts:
            return (layer_idx, lt)
    return None


# ═══════════════════════════════════════════════════════════════════════
# 1. Compact SVD of a LoRA update (dense ΔW 를 만들지 않음)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compact_svd_from_lora(B, A, scaling, rank_tol=1e-6, dtype=torch.float32):
    """ΔW = scaling · B @ A 의 exact compact SVD. dense ΔW 를 materialize 하지 않는다.

        B_s = Q_B R_B                    (thin QR,  B ∈ R^{m×r})
        A_sᵀ = Q_A R_A                   (thin QR,  A ∈ R^{r×n})
        ⇒ A_s = R_Aᵀ Q_Aᵀ
        ⇒ B_s A_s = Q_B (R_B R_Aᵀ) Q_Aᵀ
        K = scaling · R_B R_Aᵀ ∈ R^{r×r} 에만 SVD:  K = U_K Σ V_Kᵀ
        ⇒ P_s = Q_B U_K,   Q_s = Q_A V_K

    numerical rank 는 σ_i > σ_max · rank_tol 로 판별한다.

    Args:
        B: (m, r) lora_B weight
        A: (r, n) lora_A weight
        scaling: ΔW 에 곱해지는 실효 scalar (표준 LoRA 면 alpha / r)
        rank_tol: relative singular-value threshold
        dtype: 분해 연산 dtype. bf16 adapter 를 그대로 SVD 하지 않도록 최소 fp32.
            in_features 가 큰 층(4096, 11008)에서는 fp32 QR 의 직교성 오차가
            ~sqrt(n)·eps ≈ 6e-6 까지 커지므로 float64 를 권장한다 (이 크기에서는 비용 무시).

    Returns:
        P: (m, r_eff), S: (r_eff,), Q: (n, r_eff), r_nominal: int
        (r_eff == 0 이면 adapter 업데이트가 수치적으로 0 이라는 뜻)
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"compact_svd_from_lora: dtype must be fp32/fp64, got {dtype}")
    B = B.detach().to(dtype)
    A = A.detach().to(dtype)

    r_nominal = int(B.shape[1])
    if int(A.shape[0]) != r_nominal:
        raise ValueError(f"rank mismatch: B {tuple(B.shape)} vs A {tuple(A.shape)}")

    Q_B, R_B = torch.linalg.qr(B, mode="reduced")                  # (m,r), (r,r)
    Q_A, R_A = torch.linalg.qr(A.transpose(0, 1), mode="reduced")  # (n,r), (r,r)

    K = scaling * (R_B @ R_A.transpose(0, 1))                      # (r,r)
    U_K, S, V_Kh = torch.linalg.svd(K)                             # K = U_K diag(S) V_Kh
    V_K = V_Kh.transpose(0, 1)

    P = Q_B @ U_K                                                  # (m,r)
    Q = Q_A @ V_K                                                  # (n,r)

    smax = float(S.max()) if S.numel() > 0 else 0.0
    r_eff = int((S > smax * rank_tol).sum()) if smax > 0 else 0

    return (P[:, :r_eff].contiguous(), S[:r_eff].contiguous(),
            Q[:, :r_eff].contiguous(), r_nominal)


@torch.no_grad()
def lowrank_fro_norm(B, A, scaling=1.0):
    """‖scaling·B A‖_F 를 dense 없이 exact 하게. = scaling·sqrt(tr[(BᵀB)(A Aᵀ)])."""
    B = B.detach().to(torch.float64)
    A = A.detach().to(torch.float64)
    G_B = B.transpose(0, 1) @ B          # (r,r)
    G_A = A @ A.transpose(0, 1)          # (r,r)
    val = float((G_B * G_A.transpose(0, 1)).sum())   # tr(G_B G_A)
    return abs(scaling) * (max(val, 0.0) ** 0.5)


@torch.no_grad()
def relative_reconstruction_error(B, A, scaling, P, S, Q):
    """‖ΔW_s − P Σ Qᵀ‖_F / (‖ΔW_s‖_F + eps). dense 없이 exact.

    ‖X−Y‖² = ‖X‖² + ‖Y‖² − 2⟨X,Y⟩ 를 low-rank 인수만으로 전개한다:
        ‖X‖²  = s²·tr[(BᵀB)(A Aᵀ)]
        ‖Y‖²  = Σ σ_i²
        ⟨X,Y⟩ = s·tr[(BᵀP) Σ (Qᵀ Aᵀ)]

    세 항이 거의 상쇄되므로(x2 ≈ y2 ≈ cross) 두 가지에 주의해야 한다.

    1) 산술은 float64 로 한다. fp32 로 하면 catastrophic cancellation 때문에 상대오차가
       sqrt(eps_fp32) ≈ 3e-4 밑으로 내려가지 않아, "정확한 분해"와 "1e-4 만큼 틀린 분해"를
       구분하지 못한다. 측정 도구는 측정 대상보다 정밀해야 한다.

    2) ‖Y‖² 를 Σσ_i² 로 계산하면 **안 된다.** 그 등식은 P, Q 가 정확히 정규직교일 때만
       성립하는데, fp32 QR/SVD 는 ‖QᵀQ−I‖ ~ 1e-6 수준이다. 그 1e-6 오차가 x2 대비
       ~5e-9 의 절대 오차를 만들어, 우리가 재려는 실제 잔차(~1e-15·x2)를 완전히 덮어버린다
       (증상: 정확한 분해인데도 2e-4 같은 값이 나오거나 diff2 가 음수로 나옴).
       대신 직교성을 가정하지 않는 정확한 형태를 쓴다:
           ‖PΣQᵀ‖²_F = tr(Σ (PᵀP) Σ (QᵀQ))
    """
    B = B.detach().to(torch.float64)
    A = A.detach().to(torch.float64)
    P = P.detach().to(torch.float64)
    S = S.detach().to(torch.float64)
    Q = Q.detach().to(torch.float64)

    x2 = (scaling ** 2) * float((( B.transpose(0, 1) @ B) *
                                 (A @ A.transpose(0, 1)).transpose(0, 1)).sum())
    if S.numel() == 0:
        return (max(x2, 0.0) ** 0.5) / ((max(x2, 0.0) ** 0.5) + EPS)

    G_P = P.transpose(0, 1) @ P                      # (k, k)
    G_Q = Q.transpose(0, 1) @ Q                      # (k, k)
    y2 = float(((G_P * G_Q.transpose(0, 1)) * (S[:, None] * S[None, :])).sum())

    M1 = B.transpose(0, 1) @ P                       # (r, k)
    M2 = Q.transpose(0, 1) @ A.transpose(0, 1)       # (k, r)
    cross = scaling * float((torch.einsum("ij,ji->j", M1, M2) * S).sum())

    diff2 = max(x2 + y2 - 2.0 * cross, 0.0)
    return (diff2 ** 0.5) / ((max(x2, 0.0) ** 0.5) + EPS)


@torch.no_grad()
def orthogonality_error(Q):
    """‖QᵀQ − I‖_F. (fp64 — fp32 Q 의 실제 직교성 오차를 재기 위함)"""
    if Q.numel() == 0:
        return 0.0
    Q = Q.to(torch.float64)
    G = Q.transpose(0, 1) @ Q
    return float(torch.linalg.matrix_norm(G - torch.eye(G.shape[0], device=G.device,
                                                        dtype=G.dtype)))


def select_protected_rank(S, top_k=None, energy=None):
    """보호할 singular direction 개수 k 를 고른다.

    우선순위: top_k → energy(누적 에너지) → all-effective.
    energy 모드에서는 sum_{i≤k} σ_i² / sum_i σ_i² ≥ τ 인 최소 k 를 고르므로
    layer 마다 k 가 달라질 수 있다.

    Returns: (k, mode_str)
    """
    r_eff = int(S.numel())
    if r_eff == 0:
        return 0, "empty"
    if top_k is not None:
        return max(0, min(int(top_k), r_eff)), "top_k"
    if energy is not None:
        e = (S.to(torch.float32) ** 2)
        total = float(e.sum())
        if total <= 0:
            return r_eff, "all_effective"
        cum = torch.cumsum(e, dim=0) / total
        k = int(torch.searchsorted(cum, torch.tensor(float(energy),
                                                     device=cum.device)).item()) + 1
        return max(1, min(k, r_eff)), "energy"
    return r_eff, "all_effective"


def cumulative_energy(S):
    """누적 singular-value 에너지 리스트 (σ² 기준)."""
    if S.numel() == 0:
        return []
    e = (S.to(torch.float32) ** 2)
    total = float(e.sum())
    if total <= 0:
        return [0.0] * int(S.numel())
    return (torch.cumsum(e, dim=0) / total).tolist()


# ═══════════════════════════════════════════════════════════════════════
# 2. Artifact I/O
# ═══════════════════════════════════════════════════════════════════════

def subspace_path(root, layer_type, layer_idx):
    return os.path.join(root, layer_type, f"layer_{layer_idx:02d}_subspace.pt")


def save_subspace(root, key, payload):
    layer_idx, layer_type = key
    os.makedirs(os.path.join(root, layer_type), exist_ok=True)
    torch.save(payload, subspace_path(root, layer_type, layer_idx))


def load_subspaces(root, layer_types=None):
    """{(layer_idx, layer_type): payload} 로 로드. payload['Q_S'] 가 보호 subspace."""
    out = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"adapter subspace dir not found: {root}")
    types = layer_types or [d for d in sorted(os.listdir(root))
                            if os.path.isdir(os.path.join(root, d))]
    for lt in types:
        d = os.path.join(root, lt)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith("layer_") and fname.endswith("_subspace.pt"):
                li = int(fname.split("_")[1])
                out[(li, lt)] = torch.load(os.path.join(d, fname), map_location="cpu",
                                           weights_only=False)
    return out


def read_report(root):
    p = os.path.join(root, "report.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# 3. Downstream LoRA 에 제약을 거는 projector
# ═══════════════════════════════════════════════════════════════════════

class AdapterSubspaceProjector:
    """downstream LoRA 의 lora_A 가 항상 A_d Q_S = 0 을 만족하도록 유지한다.

    두 단계를 모두 수행한다:

      (1) gradient projection — `register_post_accumulate_grad_hook` 으로 grad 가 쌓이는
          즉시 g ← g − (g Q_S) Q_Sᵀ. projection 은 선형이므로 gradient accumulation 과
          교환된다 (부분합을 각각 투영 == 총합을 투영).

      (2) parameter projection — optimizer.step() 이후 A ← A − (A Q_S) Q_Sᵀ.

    (2) 는 생략할 수 없다. gradient 를 완벽히 투영해도 **AdamW 의 update 는 부분공간을
    벗어난다**: exp_avg 는 grad 의 선형결합이라 자동으로 부분공간 안에 남지만, 실제 step 은
    exp_avg / sqrt(exp_avg_sq) 이고 exp_avg_sq 는 원소별 제곱이라 투영이라는 개념 자체가
    없다. 원소별 나눗셈이 선형성을 깨므로 update 방향이 부분공간 밖으로 새어 나간다.
    따라서 제약을 실제로 보장하는 장치는 post-step 재투영이다.
    (decoupled weight decay 는 A ← (1−λη)A 라 부분공간을 보존하므로 문제되지 않는다.)

    n×n projector (I − Q_S Q_Sᵀ) 는 절대 만들지 않는다. 항상 (·)Q_S 로 r×k 만 거친다.
    """

    def __init__(self, model, subspaces, adapter_name="default",
                 project_exp_avg=False, logger_=None):
        self.model = model
        self.adapter_name = adapter_name
        self.project_exp_avg = project_exp_avg
        self.log = logger_ or logger
        self.count = 0            # ProjectionCallback 과 동일한 인터페이스
        self._hooks = []
        self._entries = []        # [(key, module_name, A_param, Q_S_tensor)]
        self._unconstrained = []  # subspace artifact 가 없어 제약 미적용인 target

        for name, module in model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            lora_A = getattr(module, "lora_A")
            if self.adapter_name not in lora_A:
                continue
            key = module_name_to_key(name)
            if key is None:
                continue
            payload = subspaces.get(key)
            if payload is None or payload.get("Q_S") is None or payload["Q_S"].numel() == 0:
                self._unconstrained.append((key, name))
                continue

            A_param = lora_A[self.adapter_name].weight
            Q = payload["Q_S"].to(device=A_param.device, dtype=torch.float32).contiguous()
            if Q.shape[0] != A_param.shape[1]:
                raise ValueError(
                    f"{name}: Q_S in_dim {Q.shape[0]} != lora_A in_dim {A_param.shape[1]}")
            self._entries.append((key, name, A_param, Q))

        if not self._entries:
            raise ValueError(
                "AdapterSubspaceProjector: 제약을 걸 수 있는 module 이 하나도 없습니다 "
                f"(adapter_name={adapter_name}). subspace dir 와 target_modules 를 확인하세요.")

    # ── 내부 헬퍼 ────────────────────────────────────────────────────
    @staticmethod
    @torch.no_grad()
    def _project_inplace(mat, Q):
        """mat ← mat − (mat Q) Qᵀ. mat: (r, n), Q: (n, k). fp32 로 계산 후 원 dtype 복귀."""
        work = mat.detach().to(torch.float32)
        work -= (work @ Q) @ Q.transpose(0, 1)
        mat.copy_(work.to(mat.dtype))

    # ── 공개 API ─────────────────────────────────────────────────────
    @property
    def num_constrained(self):
        return len(self._entries)

    @property
    def unconstrained_targets(self):
        return list(self._unconstrained)

    def register_grad_hooks(self):
        """(1) gradient projection 훅 등록."""
        for _, _, A_param, Q in self._entries:
            def _hook(param, _Q=Q):
                if param.grad is not None:
                    self._project_inplace(param.grad, _Q)
            self._hooks.append(A_param.register_post_accumulate_grad_hook(_hook))
        self.log.info(f"[adapter_subspace] gradient projection hooks: {len(self._hooks)}")

    def remove_grad_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def project(self, optimizer=None):
        """(2) parameter (+선택적으로 Adam exp_avg) 재투영."""
        for _, _, A_param, Q in self._entries:
            self._project_inplace(A_param.data, Q)
            if self.project_exp_avg and optimizer is not None:
                state = optimizer.state.get(A_param)
                if state and "exp_avg" in state:
                    self._project_inplace(state["exp_avg"], Q)
        self.count += 1

    # ── 검증 (§10 A–C, E) ────────────────────────────────────────────
    @torch.no_grad()
    def verify(self, subspaces=None, scaling_d=None, check_mapping=True):
        """레이어별 제약 지표를 계산해 (per_layer_list, aggregate_dict) 로 반환.

        A. constraint_A     = ‖A_d Q_S‖_F / ‖A_d‖_F
        B. constraint_delta = ‖B_d A_d Q_S‖_F / ‖B_d A_d‖_F
        C. mapping_drift    = ‖W_final Q_S − W_safe Q_S‖_F / ‖W_safe Q_S‖_F
                            ( = ‖s_d B_d (A_d Q_S)‖_F / ‖W_safe Q_S‖_F )
        E. delta_norm       = ‖B_d A_d‖_F   (제약이 너무 세서 업데이트가 죽었는지 확인)
        """
        rows = []
        for key, name, A_param, Q in self._entries:
            module = self.model.get_submodule(name)
            A = A_param.detach().to(torch.float32)                        # (r_d, n)
            B = module.lora_B[self.adapter_name].weight.detach().to(torch.float32)  # (m, r_d)
            s_d = float(scaling_d) if scaling_d is not None else float(
                module.scaling.get(self.adapter_name, 1.0)
                if isinstance(getattr(module, "scaling", None), dict) else 1.0)

            AQ = A @ Q                                                     # (r_d, k)
            a_norm = float(torch.linalg.matrix_norm(A))
            aq_norm = float(torch.linalg.matrix_norm(AQ))
            delta_norm = lowrank_fro_norm(B, A, 1.0)
            bq_norm = float(torch.linalg.matrix_norm(B @ AQ))

            row = {
                "layer": key[0], "layer_type": key[1],
                "constraint_A": aq_norm / (a_norm + EPS),
                "constraint_delta": bq_norm / (delta_norm + EPS),
                "delta_norm": delta_norm,
                "protected_k": int(Q.shape[1]),
            }

            if check_mapping:
                # W_safe Q_S: merge 모드면 base_layer.weight 가 이미 W_safe.
                # keep 모드면 W_base + s_s B_s A_s 를 더해 구성한다.
                W = module.base_layer.weight.detach().to(torch.float32)
                WQ = W @ Q
                for other in module.lora_A:
                    if other == self.adapter_name:
                        continue
                    s_o = float(module.scaling.get(other, 1.0))
                    A_o = module.lora_A[other].weight.detach().to(torch.float32)
                    B_o = module.lora_B[other].weight.detach().to(torch.float32)
                    WQ = WQ + s_o * (B_o @ (A_o @ Q))
                drift = s_d * bq_norm
                row["mapping_drift"] = drift / (float(torch.linalg.matrix_norm(WQ)) + EPS)

            if subspaces is not None:
                payload = subspaces.get(key, {})
                row["safety_recon_error"] = payload.get("reconstruction_error")
                row["orthogonality_error"] = payload.get("orthogonality_error")
            rows.append(row)

        def _agg(field):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return {"max": max(vals), "mean": sum(vals) / len(vals)} if vals else None

        aggregate = {
            "num_constrained_modules": len(rows),
            "num_unconstrained_targets": len(self._unconstrained),
            "constraint_A": _agg("constraint_A"),
            "constraint_delta": _agg("constraint_delta"),
            "mapping_drift": _agg("mapping_drift"),
            "delta_norm": _agg("delta_norm"),
            "safety_recon_error": _agg("safety_recon_error"),
            "orthogonality_error": _agg("orthogonality_error"),
            "projection_calls": self.count,
        }
        return rows, aggregate
