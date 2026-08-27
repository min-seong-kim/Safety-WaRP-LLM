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
