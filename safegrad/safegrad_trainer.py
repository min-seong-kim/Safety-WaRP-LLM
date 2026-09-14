"""SafeGrad trainer — gradient surgery + KL-divergence alignment loss.

논문: "SafeGrad: Gradient Surgery for Safe LLM Fine-tuning" (arXiv:2508.07172)
참조 구현: /NHNHOME/.../minseong/SafeGrad/safegrad_trainer.py  (`SafeGrad_KLD_Trainer`)

알고리즘 (논문 Algorithm 1, Eq. 3~6)
-----------------------------------
매 step 마다
    1. 유저 태스크 배치로 g_user = ∇_θ L_user
    2. alignment 배치로  g_align = ∇_θ L_align,
         L_align = D_KL( P_θ0(·|x^a) ‖ P_θ(·|x^a) )        ... Eq. 6
         (θ0 = 얼려둔 안전정렬 reference 모델)
    3. 충돌 판정은 **모델 전체를 하나의 벡터로 본 전역 내적** 하나로 한다.
         g_user · g_align < 0  이면
             g'_user = g_user − (g_user·g_align / ‖g_align‖²) · g_align   ... Eq. 4
         아니면 g'_user = g_user
    4. g_final = g'_user + ρ · g_align                                    ... Eq. 5

즉 "충돌할 때만" 유저 그래디언트에서 안전 그래디언트와 정면으로 반대되는 성분을
깎아낸다. 투영 후 cos(g'_user, g_align) = 0 이 되는 것이 이 방법의 전부다.

참조 구현과 의도적으로 같게 유지한 부분
--------------------------------------
* **전역(모델 전체) 내적** 하나로 충돌을 판정한다. 레이어별로 따로 투영하지 않는다.
  (참조 구현 주석의 "DiGraP" 가 이것을 가리킨다.)
* 내적/노름 누적만 float32 로 올리고, 실제 그래디언트 합성은 파라미터 dtype(bf16)에서 한다.
* KL 은 PyTorch `F.kl_div(log_p_theta, p_ref)` = **KL(P_ref ‖ P_theta)** 로, 논문 Eq. 6 의
  D_KL(P_θ0 ‖ P_θ) 와 방향이 일치한다.
* `kl_reduction="ref"`(기본): `reduction='sum'` 으로 **패딩·프롬프트 위치까지 포함해** 전부 더한
  뒤 `(labels != -100)` 개수로 나눈다. 분자와 분모의 토큰 집합이 다른, 참조 구현 그대로의
  정의다. 배치 패딩량에 값이 의존하므로 `kl_reduction="response"` 로 응답 토큰만 쓰도록
  바꿀 수 있게 해두었지만, baseline 비교에는 기본값(참조 재현)을 쓴다.
* KL 계산 dtype 도 기본은 참조와 같은 모델 dtype(bf16). `kl_fp32=True` 로 올릴 수 있다.

참조 구현에서 **고친** 부분 (이 저장소에서 필요한 수정)
-----------------------------------------------------
* **gradient accumulation.** 참조 구현은 `param.grad = final_grad` 로 **덮어쓴다**.
  참조 스크립트가 항상 `gradient_accumulation_steps=1` 이라 드러나지 않았을 뿐,
  accum>1 이면 마지막 micro-batch 의 그래디언트만 남는 버그다. 이 저장소의 LoRA 계열은
  effective batch 16 을 micro×accum 으로 쪼개 쓰므로(예: 4×4) 그대로 두면 안 된다.
  여기서는 micro-step 진입 시점의 누적 그래디언트를 보관했다가
  `param.grad = prev + final_grad` 로 **더한다**. g_user·g_align 도 각 micro-batch 안에서
  판정하므로, accum=1 이면 참조 구현과 완전히 동일하게 동작한다.
* **로깅.** 참조 구현은 매 step `self.log()` 를 호출한다(로그가 수백 배로 불어난다).
  여기서는 통계를 모아 두었다가 Trainer 가 `logging_steps` 주기로 로깅할 때 같이 실어 보낸다.
* **loss 스케일.** 두 backward 모두 `self.accelerator.backward` 로 태운다. accelerate 가
  두 번 다 1/accum 을 똑같이 곱하므로 투영 스칼라(dot/‖·‖²)는 스케일 불변이고,
  최종 그래디언트는 micro-batch 평균이 된다.

메모리 주의
-----------
한 step 에 그래디언트 사본이 최대 3벌(prev, g_user, g_align) 필요하다. LoRA(학습 파라미터
수천만 개)에서는 무시할 수준이지만 full-param 이면 모델 크기의 3배가 더 든다. 그래서 이
저장소의 SafeGrad arm 은 다른 LoRA 계열 baseline 과 마찬가지로 LoRA 로 돌린다.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer

IGNORE_INDEX = -100


class SafeGradTrainer(Trainer):
    """SafeGrad (gradient surgery + KL alignment) 용 HF Trainer.

    Parameters
    ----------
    ref_model : nn.Module, optional
        얼려둔 안전정렬 reference 모델 θ0. `ref_mode="separate"` 일 때 필수.
    ref_mode : {"separate", "adapter_off"}
        θ0 를 얻는 방법.
          * "separate"    — 별도 모델 인스턴스(참조 구현과 동일). 메모리 한 벌 더 든다.
          * "adapter_off" — LoRA 어댑터를 끈 현재 모델. **어댑터를 출발 모델 위에 바로
            얹은 경우에 한해** θ0 와 수학적으로 완전히 동일하며 메모리를 아낀다.
            (먼저 다른 어댑터를 merge 한 뒤 학습하는 흐름에서는 쓰면 안 된다.)
    projection : bool
        False 면 gradient surgery 없이 g_final = g_user + ρ·g_align (= 가중합 baseline).
        논문의 ablation("투영을 껐을 때") 을 재현할 때 쓴다.
    rho : float
        Eq. 5 의 ρ. 논문 기본값 1.0.
    kl_reduction : {"ref", "response"}
        위 docstring 참고. 기본 "ref" = 참조 구현 재현.
    kl_fp32 : bool
        True 면 softmax/log_softmax 를 float32 로 계산한다(메모리 2배).
    """

    def __init__(self, *args,
                 ref_model: Optional[nn.Module] = None,
                 ref_mode: str = "separate",
                 projection: bool = True,
                 rho: float = 1.0,
                 kl_reduction: str = "ref",
                 kl_fp32: bool = False,
                 align_batch_size: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if ref_mode not in ("separate", "adapter_off"):
            raise ValueError(f"ref_mode 는 'separate' 또는 'adapter_off' 여야 한다: {ref_mode}")
        if kl_reduction not in ("ref", "response"):
            raise ValueError(f"kl_reduction 은 'ref' 또는 'response' 여야 한다: {kl_reduction}")

        self.ref_mode = ref_mode
        self.projection = projection
        self.rho = float(rho)
        self.kl_reduction = kl_reduction
        self.kl_fp32 = bool(kl_fp32)
        self.align_batch_size = align_batch_size

        if ref_mode == "separate":
            if ref_model is None:
                raise ValueError("ref_mode='separate' 이면 ref_model 을 넘겨야 한다.")
            ref_model.to(self.accelerator.device)
            ref_model.eval()
            ref_model.requires_grad_(False)
            self.ref_model = ref_model
        else:
            unwrapped = self.accelerator.unwrap_model(self.model)
            if not hasattr(unwrapped, "disable_adapter"):
                raise ValueError(
                    "ref_mode='adapter_off' 는 PEFT 모델에서만 쓸 수 있다 "
                    "(disable_adapter 가 없다). full-param 학습이면 'separate' 를 써라.")
            self.ref_model = None

        # 로깅용 누적 버퍼
        self._sg_sums: Dict[str, float] = {}
        self._sg_count = 0
        self._sg_conflicts = 0

        self.alignment_dataloader = None
        self.data_iter = None

    # ------------------------------------------------------------------
    # alignment 데이터
    # ------------------------------------------------------------------
    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        from transformers.trainer_utils import seed_worker

        batch_size = self.align_batch_size or self._train_batch_size
        sampler = RandomSampler(alignment_dataset)
        params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            params["sampler"] = sampler
            params["drop_last"] = self.args.dataloader_drop_last
            params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(alignment_dataset, **params))

    def init(self, alignment_dataset):
        """학습 시작 전에 반드시 호출. alignment 데이터로더를 만든다."""
        if alignment_dataset is None or len(alignment_dataset) == 0:
            raise ValueError("SafeGrad 는 alignment 데이터가 반드시 필요하다 (guide_data_num > 0).")
        self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
        self.data_iter = iter(self.alignment_dataloader)

    def sample_from_alignment(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.alignment_dataloader)
            return next(self.data_iter)

    # ------------------------------------------------------------------
    # reference 모델
    # ------------------------------------------------------------------
    @contextmanager
    def _ref_model_context(self):
        """θ0 로 forward 할 수 있는 모델을 내준다."""
        if self.ref_mode == "separate":
            yield self.ref_model
        else:
            unwrapped = self.accelerator.unwrap_model(self.model)
            with unwrapped.disable_adapter():
                yield self.model

    # ------------------------------------------------------------------
    # KL alignment loss
    # ------------------------------------------------------------------
    def compute_alignment_loss(self, model, inputs) -> torch.Tensor:
        """L_align = KL( P_ref ‖ P_theta ) on the alignment batch (논문 Eq. 6)."""
        fwd = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        # labels 는 넘기지 않는다. 모델이 쓰지도 않는 CE loss 를 계산할 뿐 logits 은 동일하다.

        with torch.no_grad():
            with self._ref_model_context() as ref:
                logits_ref = ref(**fwd).logits

        logits_theta = model(**fwd).logits

        if self.kl_fp32:
            logits_theta = logits_theta.float()
            logits_ref = logits_ref.float()

        log_p_theta = F.log_softmax(logits_theta, dim=-1)
        with torch.no_grad():
            p_ref = F.softmax(logits_ref, dim=-1)

        labels = inputs["labels"]
        valid = labels.ne(IGNORE_INDEX)
        num_valid = valid.sum()

        if self.kl_reduction == "ref":
            # 참조 구현 그대로: 패딩/프롬프트 위치까지 전부 더한 뒤 응답 토큰 수로 나눈다.
            total = F.kl_div(log_p_theta, p_ref, reduction="sum", log_target=False)
            if num_valid > 0:
                return total / num_valid
            return torch.zeros((), device=logits_theta.device, dtype=logits_theta.dtype)

        # "response": 응답 토큰 위치만 평균낸다 (패딩량에 값이 의존하지 않는다).
        per_token = F.kl_div(log_p_theta, p_ref, reduction="none", log_target=False).sum(dim=-1)
        masked = per_token * valid.to(per_token.dtype)
        if num_valid > 0:
            return masked.sum() / num_valid
        return torch.zeros((), device=logits_theta.device, dtype=logits_theta.dtype)

    # ------------------------------------------------------------------
    # gradient helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _trainables(model):
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    @staticmethod
    def _snapshot_grads(trainables) -> Dict[str, torch.Tensor]:
        return {n: p.grad.detach().clone() for n, p in trainables if p.grad is not None}

    @staticmethod
    def _clear_grads(trainables) -> None:
        for _, p in trainables:
            p.grad = None

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------
    def training_step(self, model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None) -> torch.Tensor:
        model.train()
        trainables = self._trainables(model)

        # ── 0. 이 optimizer 주기에서 지금까지 쌓인 그래디언트를 보관하고 비운다 ──────
        #    (참조 구현은 이 단계가 없어 accum>1 에서 앞선 micro-batch 를 덮어썼다)
        prev = self._snapshot_grads(trainables)
        self._clear_grads(trainables)

        # ── 1. 유저 태스크 그래디언트 g_user ──────────────────────────────────
        task_inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss_task = self.compute_loss(model, task_inputs, return_outputs=False)
        if self.args.n_gpu > 1:
            loss_task = loss_task.mean()
        self.accelerator.backward(loss_task)
        g_task = self._snapshot_grads(trainables)
        self._clear_grads(trainables)

        # ── 2. alignment 그래디언트 g_align (KL) ──────────────────────────────
        align_inputs = self._prepare_inputs(self.sample_from_alignment())
        with self.compute_loss_context_manager():
            loss_align = self.compute_alignment_loss(model, align_inputs)
        if self.args.n_gpu > 1:
            loss_align = loss_align.mean()
        self.accelerator.backward(loss_align)
        g_align = self._snapshot_grads(trainables)
        self._clear_grads(trainables)

        # ── 3. gradient surgery ──────────────────────────────────────────────
        projection_scalar, dot, align_norm_sq = self._projection_scalar(g_task, g_align)

        for name, param in trainables:
            gt = g_task.get(name)
            ga = g_align.get(name)
            if gt is None and ga is None:
                new_grad = None
            elif ga is None:
                new_grad = gt
            elif gt is None:
                # 태스크 쪽에 그래디언트가 없는 파라미터(거의 없다). alignment 항만 반영.
                new_grad = self.rho * ga
            else:
                projected = gt - projection_scalar.to(gt.dtype) * ga
                new_grad = projected + self.rho * ga

            if new_grad is None:
                continue
            base = prev.get(name)
            param.grad = new_grad if base is None else base + new_grad

        # prev 는 param.grad 로 넘어갔으니 즉시 놓아준다
        prev.clear()
        g_task.clear()
        g_align.clear()

        # ── 4. 통계 누적 (Trainer 의 logging_steps 주기에 실려 나간다) ──────────
        self._sg_accumulate(loss_task, loss_align, dot, align_norm_sq, projection_scalar)

        return loss_task.detach() / self.args.gradient_accumulation_steps

    def _projection_scalar(self, g_task, g_align):
        """전역 내적으로 충돌을 판정하고 Eq. 4 의 투영 계수를 돌려준다."""
        device = self.accelerator.device
        dot = torch.zeros((), device=device, dtype=torch.float32)
        align_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        scalar = torch.zeros((), device=device, dtype=torch.float32)

        if not self.projection:
            return scalar, dot, align_norm_sq

        for name, gt in g_task.items():
            ga = g_align.get(name)
            if ga is None:
                continue
            # 누적만 float32 로 올린다 (bf16 으로 수천만 항을 더하면 값이 무너진다).
            gt32 = gt.to(torch.float32)
            ga32 = ga.to(torch.float32)
            dot += torch.sum(gt32 * ga32)
            align_norm_sq += torch.sum(ga32 * ga32)

        if dot < 0 and align_norm_sq > 1e-9:
            scalar = dot / align_norm_sq
        return scalar, dot, align_norm_sq

    # ------------------------------------------------------------------
    # 로깅
    # ------------------------------------------------------------------
    def _sg_accumulate(self, loss_task, loss_align, dot, align_norm_sq, projection_scalar):
        dot_v = float(dot)
        stats = {
            "safegrad/loss_task": float(loss_task.detach()),
            "safegrad/loss_align_kl": float(loss_align.detach()),
            "safegrad/grad_dot": dot_v,
            "safegrad/grad_align_norm_sq": float(align_norm_sq),
            "safegrad/projection_scalar": float(projection_scalar),
        }
        for k, v in stats.items():
            self._sg_sums[k] = self._sg_sums.get(k, 0.0) + v
        self._sg_count += 1
        if dot_v < 0:
            self._sg_conflicts += 1

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        if self._sg_count:
            for k, v in self._sg_sums.items():
                logs[k] = v / self._sg_count
            logs["safegrad/conflict_rate"] = self._sg_conflicts / self._sg_count
            self._sg_sums.clear()
            self._sg_count = 0
            self._sg_conflicts = 0
        try:
            super().log(logs, start_time)
        except TypeError:      # transformers < 4.46 은 start_time 인자가 없다
            super().log(logs)
