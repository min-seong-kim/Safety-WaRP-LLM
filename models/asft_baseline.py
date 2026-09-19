"""AsFT (Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin, arXiv:2506.08473).

참조 구현: /home/edgeai_lab/AsFT
  - `AsFT_finetuning.py`   : SafeLoRA 클래스로 project_matrix(= Ĉ 리스트) 생성
  - `utils/AsFT_train_utils.py:99-119` : 학습 루프에서 정규화항을 loss 에 더하는 부분

알고리즘 (참조 구현과 수식 동일):
    각 target module 마다
        V   = W_aligned − W_base                      (alignment direction, out×in)
        Ĉ   = (V Vᵀ) / ‖V‖_F                          (out×out)
    매 step
        L = L_SFT + λ · Σ_l ‖ (I − Ĉ_l) · (B_l A_l) ‖²_F

즉 LoRA 업데이트 ΔW=BA 를 "alignment direction 이 만드는 부분공간"과 그 여집합으로 쪼개고,
여집합(=narrow safety basin 을 벗어나는 성분)에만 페널티를 준다. SafeLoRA 가 학습이 끝난 뒤
lora_B ← Ĉ·B 로 한 번 투영하는 사후 방식인 데 반해, AsFT 는 학습 내내 연속적인 벌점으로 건다.
Ĉ 는 SafeLoRA 와 **완전히 같은 행렬**이며(`models/safelora_baseline._build_projectors` 참조),
AsFT 는 여기에 cos 임계값 선택 없이 전 레이어에 정규화를 적용한다.

참조 구현과 의도적으로 같게 유지한 부분
  - ΔW 로 `B @ A` 를 쓴다. PEFT 의 실제 업데이트는 s·BA (s=alpha/r) 이지만 참조 구현이
    scaling 을 곱하지 않으며, λ=1 이라는 기본값이 그 정의 위에서 정해진 값이다.
  - Ĉ = VVᵀ/‖V‖_F 는 (‖V‖² 가 아닌 ‖V‖ 로 나누므로) 엄밀한 사영행렬이 아니다.
    SafeLoRA 원 구현에서 그대로 이어져 온 정의이며 바꾸지 않았다.
  - 정규화항은 fp32 로 계산한다.

참조 구현과 다른 부분 (수학적으로 동치, 비용만 다름)
  참조 구현은 Ĉ (out×out) 를 통째로 들고 있다가 매 step `(I−Ĉ) @ (B@A)` 를 계산한다.
  up_proj 의 Ĉ 는 11008×11008 → 32 레이어면 fp32 로 15GB 가 넘고, out²·in FLOP 이 든다.
  여기서는 대신 V 를 들고 다음 항등식을 쓴다.
      X = (I − Ĉ) B = B − V (Vᵀ B) / ‖V‖_F           (out×r)
      ‖(I − Ĉ) B A‖²_F = ‖X A‖²_F = trace( (Xᵀ X) (A Aᵀ) )
  r=16 이므로 r×r 두 개의 곱으로 끝난다. 값은 참조 구현과 부동소수점 오차 내에서 동일하며,
  `--asft_check_equiv` 로 첫 step 에 naive 식과 대조 검증할 수 있다.
"""
import contextlib
import gc

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def build_alignment_dirs(base_path, aligned_path, target_modules, device,
                         load_dtype=torch.float32, store_dtype=torch.float32, logger=None):
    """base/aligned 를 로드해 module 별 alignment direction V 와 ‖V‖_F 를 만든다.

    반환: List[(V, ‖V‖_F)] — base model 파라미터 순회 순서.
          V 는 `device` 위에 `store_dtype` 으로 상주한다(매 step 재전송 방지).
    """
    assert logger is not None
    logger.info(f"[AsFT] loading base   : {base_path} (dtype={load_dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, return_dict=True, low_cpu_mem_usage=True,
        torch_dtype=load_dtype, device_map="cpu")
    logger.info(f"[AsFT] loading aligned: {aligned_path} (dtype={load_dtype})")
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_path, return_dict=True, low_cpu_mem_usage=True,
        torch_dtype=load_dtype, device_map="cpu")

    dirs = []
    total_bytes = 0
    for (b_name, b_param), (a_name, a_param) in zip(base_model.named_parameters(),
                                                    aligned_model.named_parameters()):
        if not any(m in a_name for m in target_modules):
            continue
        # ⚠️ 이름 부분일치만으로 고르면 **bias 도 걸린다.** Qwen2.5 는 q/k/v_proj 에 bias 가
        # 있어서 층당 5개가 아니라 8개(q.w,q.b,k.w,k.b,v.w,v.b,up.w,down.w)가 잡혔고,
        # positional 대응이 어긋나 28층×8=224 개가 만들어졌다(정상은 140).
        # alignment direction 은 가중치 행렬 V = W_aligned − W_base 로만 정의된다.
        if b_param.ndim != 2:
            continue
        assert b_param.shape == a_param.shape, (
            f"base/aligned weight shape mismatch: {b_name} {tuple(b_param.shape)} "
            f"vs {a_name} {tuple(a_param.shape)}")
        vec = (a_param.detach() - b_param.detach()).to(device, dtype=torch.float32)
        norm = torch.norm(vec)                       # ‖V‖_F (참조 구현의 torch.norm 기본값)
        v = vec.to(store_dtype).contiguous()
        dirs.append((v, norm.to(torch.float32)))
        total_bytes += v.numel() * v.element_size()

    logger.info(f"[AsFT] built {len(dirs)} alignment directions on {device} "
                f"({total_bytes / 2**30:.2f} GiB, dtype={store_dtype})")
    if dirs:
        norms = torch.stack([n for _, n in dirs])
        logger.info(f"[AsFT] ‖V‖_F range: [{norms.min():.4f}, {norms.max():.4f}] "
                    f"mean={norms.mean():.4f}")

    del base_model, aligned_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dirs


class AsFTRegularizer:
    """LoRA (A, B) 쌍마다 λ·‖(I−Ĉ)BA‖²_F 를 합산해 돌려준다.

    peft 모델의 lora_A 순회 순서와 `dirs` 의 순서가 positional 로 1:1 대응한다고 가정한다
    (표준 LLaMA + 표준 target_modules 에서 성립 — SafeLoRA 구현과 동일한 가정).
    """

    def __init__(self, peft_model, dirs, lambda_reg, logger):
        self.lambda_reg = float(lambda_reg)
        self.logger = logger
        self.pairs = []          # [(lora_A param, lora_B param, V, ‖V‖_F)]
        self._checked = False

        named = dict(peft_model.named_parameters())
        idx = 0
        for name, param in peft_model.named_parameters():
            if "lora_A" not in name:
                continue
            b_name = name.replace("lora_A", "lora_B")
            if b_name not in named:
                raise ValueError(f"[AsFT] {name} 에 대응하는 lora_B 가 없습니다: {b_name}")
            if idx >= len(dirs):
                raise ValueError(
                    f"[AsFT] alignment direction 개수({len(dirs)}) 보다 LoRA 레이어가 많습니다. "
                    "--target_modules 와 base/aligned 모델이 일치하는지 확인하세요.")
            v, vnorm = dirs[idx]
            b_param = named[b_name]
            if v.shape[0] != b_param.shape[0]:
                raise ValueError(
                    f"[AsFT] 순서 불일치: {b_name} out={b_param.shape[0]} vs V out={v.shape[0]}")
            self.pairs.append((param, b_param, v, vnorm))
            idx += 1

        if idx != len(dirs):
            raise ValueError(
                f"[AsFT] LoRA 레이어 {idx} 개 ≠ alignment direction {len(dirs)} 개 — "
                "positional 대응이 깨졌습니다.")
        logger.info(f"[AsFT] regularizer 연결 완료: {len(self.pairs)} LoRA 레이어, λ={self.lambda_reg}")

    def _naive_term(self, a, b, v, vnorm):
        """참조 구현 그대로의 식 — 등가성 검증용(느리고 메모리를 많이 쓴다)."""
        c_hat = (v.float() @ v.float().t()) / vnorm
        identity = torch.eye(c_hat.shape[0], device=c_hat.device, dtype=torch.float32)
        return torch.norm((identity - c_hat) @ (b.float() @ a.float()), p="fro") ** 2

    def loss(self, check_equiv=False):
        total = None
        for a, b, v, vnorm in self.pairs:
            a32, b32 = a.float(), b.float()
            vf = v.float() if v.dtype != torch.float32 else v
            # X = (I − Ĉ) B = B − V (Vᵀ B) / ‖V‖_F        (out×r)
            x = b32 - (vf @ (vf.t() @ b32)) / vnorm
            # ‖X A‖²_F = trace( (Xᵀ X)(A Aᵀ) ) = Σ (XᵀX) ∘ (A Aᵀ)   (둘 다 대칭 r×r)
            term = ((x.t() @ x) * (a32 @ a32.t())).sum()
            total = term if total is None else total + term

        if check_equiv and not self._checked:
            with torch.no_grad():
                a, b, v, vnorm = self.pairs[0]
                naive = self._naive_term(a, b, v, vnorm)
                if naive.item() == 0.0:
                    # LoRA 초기화 직후에는 B=0 이라 양쪽 다 0 → 검증 의미가 없다. 다음 step 에 재시도.
                    return self.lambda_reg * total
                self._checked = True
                fast = ((lambda x: ((x.t() @ x) * (a.float() @ a.float().t())).sum())(
                    b.float() - (v.float() @ (v.float().t() @ b.float())) / vnorm))
                rel = (fast - naive).abs() / naive.abs().clamp_min(1e-12)
                self.logger.info(
                    f"[AsFT] equivalence check (layer 0): naive={naive.item():.6e} "
                    f"fast={fast.item():.6e} rel_err={rel.item():.3e}")
                if rel.item() > 1e-3:
                    self.logger.warning("[AsFT] 등가성 검증 오차가 큽니다 — 순서/수식을 확인하세요.")

        return self.lambda_reg * total

    def free(self):
        self.pairs = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  Full-parameter 판 AsFT
# ═══════════════════════════════════════════════════════════════════════════
#  원 논문(arXiv:2506.08473)과 참조 구현(/home/edgeai_lab/AsFT)은 **LoRA 전용**이다
#  (ΔW = B A 를 가정). 논문 개정본 Table 2/4 는 AsFT 를 full-parameter 계열 행에
#  싣기 때문에, 같은 벌점을 ΔW = W − W₀ 로 일반화한 것이 아래 클래스다.
#
#      L = L_SFT + λ · Σ_l ‖ (I − Ĉ_l) ΔW_l ‖²_F ,   ΔW_l = W_l − W_l⁽⁰⁾
#      Ĉ_l = V_l V_lᵀ / ‖V_l‖_F ,  V_l = W_l^aligned − W_l^base
#
#  W₀ 는 **출발 모델(=aligned)** 의 가중치다. 따라서 학습 시작 시점에 ΔW=0 이고
#  벌점도 0 이며, LoRA 판이 B=0 에서 출발하는 것과 같은 초기조건이다.
#
#  ⚠️ λ 의 의미가 LoRA 판과 같지 않다. 참조 구현의 λ=1.0 은 ΔW=BA (r=16, 초기 0)
#     스케일 위에서 정해진 값이고, full-param 의 ΔW 는 노름이 전혀 다르다.
#     그래도 "기법별 하이퍼파라미터는 동일하게" 라는 실험 설계에 따라 λ=1.0 을 쓴다
#     (2026-09-19 사용자 결정). 학습 로그의 `asft_reg` 값을 반드시 확인할 것 —
#     L_SFT 대비 지나치게 크거나(모델이 얼어붙음) 0 에 가까우면(벌점 무효) λ 재조정이 필요하다.
#
#  ── 왜 autograd 가 아니라 해석적 그래디언트인가 ──────────────────────────────
#  벌점을 loss 에 얹어 autograd 로 미분하면, 모듈마다 G=VᵀΔW (n×n) 와 X=(I−Ĉ)ΔW
#  (m×n) 가 backward 까지 그래프에 남는다. down_proj 하나가 fp32 로 484MB+180MB 이고
#  32 층이면 21GB 라 7B 조차 감당이 안 된다. 그래서 no_grad 로 아래 항등식을 직접 쓴다.
#
#      Y = (I − Ĉ) ΔW = ΔW − V (Vᵀ ΔW) / ‖V‖_F
#      ∇_W [ λ‖(I−Ĉ)ΔW‖²_F ] = 2λ (I − Ĉ)ᵀ(I − Ĉ) ΔW = 2λ (I − Ĉ) Y
#                             = 2λ ( Y − V (Vᵀ Y) / ‖V‖_F )          (Ĉ 는 대칭)
#
#  모듈 하나를 끝내면 임시버퍼가 바로 해제되므로 피크가 m×n 몇 장으로 끝난다.
#  `--asft_check_equiv` 로 autograd 와 대조 검증한다.
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _tf32_matmul():
    """벌점 계산 구간에서만 TF32 matmul 을 켠다.

    왜 필요한가: 벌점 그래디언트는 모듈마다 (n×m)(m×n) 과 (m×n)(n×n) 두 번의 큰
    행렬곱이다. fp32(비-TF32)로 돌리면 Llama-2-7B 기준 step 당 2.6 초가 붙어 학습이
    4.24 s/it 로 느려졌다(일반 full FT 는 1.67 s/it). TF32 는 가수 10비트라 상대오차가
    ~1e-3 이고, 애초에 V 와 W 가 bf16(가수 8비트)으로 저장돼 있어 fp32 의 여분 정밀도는
    의미가 없다. 반면 ΔW = W − W₀ 의 **뺄셈**은 두 bf16 의 차라 fp32 로 해야 한다
    (W ≈ W₀ 라 bf16 로 빼면 자리수 소실이 일어난다) — 그래서 뺄셈은 그대로 fp32 다.

    ⚠️ 전역 플래그라 반드시 원복한다. 모델 본체의 forward/backward 까지 TF32 가 되면
       다른 arm 과 수치가 달라져 비교가 깨진다.
    """
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


class AsFTFullParamRegularizer:
    """ΔW = W − W₀ 위에서 AsFT 벌점의 그래디언트를 직접 param.grad 에 더한다.

    dirs 는 `build_alignment_dirs` 가 돌려준 것을 그대로 받는다. 대응은 positional 이며,
    모델을 같은 필터(target_modules 부분일치 + ndim==2)로 순회하므로 순서가 일치한다.
    """

    def __init__(self, model, dirs, lambda_reg, target_modules, logger,
                 offload_cpu=False):
        self.lambda_reg = float(lambda_reg)
        self.logger = logger
        self.offload_cpu = bool(offload_cpu)
        self.entries = []        # [(param, W0, V, ‖V‖_F)]
        self._checked = False
        self.last_penalty = 0.0

        idx = 0
        n_bytes = 0
        for name, param in model.named_parameters():
            if not any(m in name for m in target_modules):
                continue
            if param.ndim != 2:          # bias 제외 (Qwen2.5 q/k/v 에 bias 가 있다)
                continue
            if idx >= len(dirs):
                raise ValueError(
                    f"[AsFT-full] alignment direction 개수({len(dirs)}) 보다 대상 weight 가 많다. "
                    "--target_modules 와 base/aligned 모델이 일치하는지 확인하라.")
            v, vnorm = dirs[idx]
            if tuple(v.shape) != tuple(param.shape):
                raise ValueError(
                    f"[AsFT-full] 순서 불일치: {name} {tuple(param.shape)} vs V {tuple(v.shape)}")
            store_dev = "cpu" if self.offload_cpu else param.device
            w0 = param.data.detach().to(store_dev, copy=True)
            vv = v.to(store_dev)
            self.entries.append((param, w0, vv, vnorm.to(param.device)))
            n_bytes += w0.numel() * w0.element_size() + vv.numel() * vv.element_size()
            idx += 1

        if idx != len(dirs):
            raise ValueError(
                f"[AsFT-full] 대상 weight {idx} 개 ≠ alignment direction {len(dirs)} 개 — "
                "positional 대응이 깨졌다.")

        where = "CPU (step 마다 스트리밍)" if self.offload_cpu else "GPU"
        logger.info(f"[AsFT-full] regularizer 연결: {len(self.entries)} 개 weight, "
                    f"λ={self.lambda_reg}, W₀+V 저장 {n_bytes/2**30:.2f} GiB on {where}")

    @torch.no_grad()
    def add_grad_(self):
        """벌점 그래디언트를 param.grad 에 **더하고** 벌점 값을 돌려준다.

        ⚠️ 대입(=)이 아니라 누적(+=)이다. SFT 그래디언트를 지우면 안 된다.
        """
        with _tf32_matmul():
            return self._add_grad_impl()

    def _add_grad_impl(self):
        total = 0.0
        for param, w0, v, vnorm in self.entries:
            dev = param.device
            w0d = w0.to(dev, non_blocking=True) if w0.device != dev else w0
            vd = v.to(dev, non_blocking=True) if v.device != dev else v
            vf = vd.float()
            dw = param.data.float() - w0d.float()

            y = dw - (vf @ (vf.t() @ dw)) / vnorm           # Y = (I−Ĉ)ΔW
            total += float((y * y).sum())
            z = y - (vf @ (vf.t() @ y)) / vnorm             # (I−Ĉ)Y
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.add_((2.0 * self.lambda_reg) * z.to(param.grad.dtype))
            del dw, y, z, vf

        self.last_penalty = self.lambda_reg * total
        return self.last_penalty

    @torch.no_grad()
    def penalty(self):
        """그래디언트를 건드리지 않고 벌점 값만 계산한다(로깅/검증용)."""
        with _tf32_matmul():
            return self._penalty_impl()

    def _penalty_impl(self):
        total = 0.0
        for param, w0, v, vnorm in self.entries:
            dev = param.device
            w0d = w0.to(dev) if w0.device != dev else w0
            vf = (v.to(dev) if v.device != dev else v).float()
            dw = param.data.float() - w0d.float()
            y = dw - (vf @ (vf.t() @ dw)) / vnorm
            total += float((y * y).sum())
        return self.lambda_reg * total

    def check_equiv(self):
        """첫 번째 weight 하나에서 해석적 그래디언트를 autograd 와 대조한다."""
        if self._checked or not self.entries:
            return
        with _tf32_matmul():
            self._check_equiv_impl()

    def _check_equiv_impl(self):
        param, w0, v, vnorm = self.entries[0]
        dev = param.device
        w0d = (w0.to(dev) if w0.device != dev else w0).float()
        vf = (v.to(dev) if v.device != dev else v).float()

        w = param.data.float().clone().requires_grad_(True)
        dw = w - w0d
        c_free = dw - (vf @ (vf.t() @ dw)) / vnorm
        obj = self.lambda_reg * (c_free * c_free).sum()
        if float(obj.detach()) == 0.0:
            return                      # ΔW=0 (학습 시작 직후) — 다음 step 에 다시 시도
        (auto,) = torch.autograd.grad(obj, w)

        with torch.no_grad():
            dw2 = param.data.float() - w0d
            y = dw2 - (vf @ (vf.t() @ dw2)) / vnorm
            manual = (2.0 * self.lambda_reg) * (y - (vf @ (vf.t() @ y)) / vnorm)

        rel = float((manual - auto).norm() / auto.norm().clamp_min(1e-12))
        self._checked = True
        self.logger.info(f"[AsFT-full] gradient 검증 (weight 0): "
                         f"‖auto‖={float(auto.norm()):.6e} ‖manual‖={float(manual.norm()):.6e} "
                         f"rel_err={rel:.3e}")
        if rel > 1e-3:
            self.logger.warning("[AsFT-full] 해석적 그래디언트가 autograd 와 어긋난다 — 수식을 확인하라.")

    def free(self):
        self.entries = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
