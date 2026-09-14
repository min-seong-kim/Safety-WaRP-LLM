"""SafeGrad 구현 검증 — gradient surgery 수학 + 참조 구현이 놓친 accumulation 처리.

빠르게 돌아간다(작은 랜덤 LLaMA, GPU 없으면 CPU):

    python safegrad/test_safegrad.py

검사 항목
---------
1. 충돌이 없을 때(g_user·g_align ≥ 0) → 투영하지 않고 g_final = g_user + ρ·g_align
2. 충돌할 때 → 투영된 유저 그래디언트가 g_align 과 **직교**한다 (논문 Fig.4(a) 의
   "After Gradient Surgery" 가 0 으로 클램프되는 것과 같은 성질)
3. projection=False 면 surgery 가 전혀 일어나지 않는다
4. θ = θ0 이면 KL alignment loss ≈ 0 이고 g_align ≈ 0
5. gradient accumulation: accum=2 로 micro-step 을 두 번 돌리면 param.grad 가
   두 micro-step 의 최종 그래디언트 **합**이 된다
   (참조 구현은 여기서 마지막 것만 남겼다 — 이 저장소는 accum>1 로 돌리므로 치명적)
6. ref_mode='adapter_off'(어댑터를 끈 현재 모델) 가 'separate'(θ0 를 따로 로드) 와
   같은 KL / g_align 을 준다
"""

import copy
import os
import sys
import tempfile

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import Dataset as HFDataset
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from safegrad.safegrad_trainer import SafeGradTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, SEQ = 64, 12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64)
    return LlamaForCausalLM(cfg).to(DEVICE, torch.float32)


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (SEQ,), generator=g).tolist()
        labels = [-100] * (SEQ // 2) + ids[SEQ // 2:]
        rows.append({"input_ids": ids, "attention_mask": [1] * SEQ, "labels": labels})
    return HFDataset.from_list(rows)


class _Collator:
    def __call__(self, feats):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in feats], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in feats], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in feats], dtype=torch.long),
        }


def _make_trainer(tmpdir, grad_accum=1, projection=True, rho=1.0, ref_model=None,
                  train_rows=None, align_rows=None):
    model = _tiny_model(seed=0)
    ref = ref_model if ref_model is not None else copy.deepcopy(model)
    args = TrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=1,
        learning_rate=1e-3,
        logging_steps=1000,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=(DEVICE == "cpu"),
        seed=0,
    )
    trainer = SafeGradTrainer(
        model=model, args=args,
        train_dataset=train_rows if train_rows is not None else _rows(8, seed=1),
        data_collator=_Collator(),
        ref_model=ref, ref_mode="separate",
        projection=projection, rho=rho,
    )
    trainer.init(align_rows if align_rows is not None else _rows(8, seed=2))
    return trainer


def _flat_dot(a, b):
    """이름이 겹치는 항끼리만 내적 (trainer 내부 전역 내적과 같은 규칙)."""
    return sum(float((a[n].double() * b[n].double()).sum()) for n in a if n in b)


# ---------------------------------------------------------------------------
# 1~3. surgery 수학 (합성 그래디언트로 직접 검증)
# ---------------------------------------------------------------------------
def test_projection_math():
    tmp = tempfile.mkdtemp()
    trainer = _make_trainer(tmp, rho=1.0)

    torch.manual_seed(7)
    names = ["a", "b"]
    g_align = {n: torch.randn(5, 3, device=DEVICE) for n in names}

    # (a) 충돌 없음: g_user = +g_align → dot > 0 → 투영 없음
    g_task = {n: g_align[n].clone() for n in names}
    scalar, dot, nsq = trainer._projection_scalar(g_task, g_align)
    assert dot > 0, dot
    assert float(scalar) == 0.0, f"충돌이 없는데 투영했다: {float(scalar)}"
    print("  [1] 비충돌 → projection_scalar = 0 ✓")

    # (b) 충돌: g_user 에 -g_align 성분을 크게 섞는다
    g_task = {n: torch.randn(5, 3, device=DEVICE) - 3.0 * g_align[n] for n in names}
    scalar, dot, nsq = trainer._projection_scalar(g_task, g_align)
    assert dot < 0, dot
    expected = _flat_dot(g_task, g_align) / sum(float((g_align[n].double() ** 2).sum()) for n in names)
    assert abs(float(scalar) - expected) < 1e-5, (float(scalar), expected)

    projected = {n: g_task[n] - float(scalar) * g_align[n] for n in names}
    resid = _flat_dot(projected, g_align)
    scale = (sum(float((projected[n].double() ** 2).sum()) for n in names) ** 0.5 *
             sum(float((g_align[n].double() ** 2).sum()) for n in names) ** 0.5)
    cos = resid / scale
    assert abs(cos) < 1e-6, f"투영 후에도 직교하지 않는다: cos={cos}"
    print(f"  [2] 충돌 → 투영 후 cos(g'_user, g_align) = {cos:.2e} ≈ 0 ✓")

    # (c) projection=False 면 아무것도 깎지 않는다
    trainer.projection = False
    scalar, dot, nsq = trainer._projection_scalar(g_task, g_align)
    assert float(scalar) == 0.0 and float(dot) == 0.0
    print("  [3] projection=False → surgery 미적용 ✓")


# ---------------------------------------------------------------------------
# 4. θ = θ0 이면 KL = 0
# ---------------------------------------------------------------------------
def test_kl_zero_when_model_equals_ref():
    tmp = tempfile.mkdtemp()
    trainer = _make_trainer(tmp)
    model = trainer.model
    batch = trainer._prepare_inputs(trainer.sample_from_alignment())

    loss = trainer.compute_alignment_loss(model, batch)
    assert float(loss) < 1e-6, f"θ=θ0 인데 KL 이 0 이 아니다: {float(loss)}"
    print(f"  [4] θ=θ0 → KL = {float(loss):.3e} ≈ 0 ✓")

    loss.backward()
    gmax = max(float(p.grad.abs().max()) for _, p in trainer._trainables(model) if p.grad is not None)
    assert gmax < 1e-5, f"θ=θ0 인데 g_align 이 0 이 아니다: {gmax}"
    print(f"  [4] θ=θ0 → max|g_align| = {gmax:.3e} ≈ 0 ✓")


# ---------------------------------------------------------------------------
# 5. gradient accumulation (참조 구현의 버그를 이 구현이 고쳤는지)
# ---------------------------------------------------------------------------
def test_gradient_accumulation_accumulates():
    tmp = tempfile.mkdtemp()
    # ref 를 살짝 흔들어 KL ≠ 0 → g_align ≠ 0 이 되게 한다
    ref = _tiny_model(seed=0)
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.05 * torch.randn_like(p))

    rows = _rows(8, seed=1)
    aligns = _rows(8, seed=2)

    def fresh():
        torch.manual_seed(123)
        t = _make_trainer(tmp, grad_accum=2, ref_model=copy.deepcopy(ref),
                          train_rows=rows, align_rows=aligns)
        return t

    collator = _Collator()
    micro = [collator([rows[0], rows[1]]), collator([rows[2], rows[3]])]

    # (a) micro-step 을 하나씩 따로 돌려 최종 그래디언트를 각각 받는다
    singles = []
    for mb in micro:
        t = fresh()
        t.data_iter = iter(t.alignment_dataloader)     # alignment 배치 순서 고정
        t.training_step(t.model, {k: v.clone() for k, v in mb.items()})
        singles.append({n: p.grad.detach().clone()
                        for n, p in t._trainables(t.model) if p.grad is not None})

    # (b) 같은 trainer 에서 두 micro-step 을 연속으로 (= 한 optimizer 주기)
    t = fresh()
    t.data_iter = iter(t.alignment_dataloader)
    for mb in micro:
        t.training_step(t.model, {k: v.clone() for k, v in mb.items()})
    accumulated = {n: p.grad.detach().clone()
                   for n, p in t._trainables(t.model) if p.grad is not None}

    # 주의: (a) 의 두 번째 실행은 alignment 이터레이터가 첫 배치부터 다시 시작하므로
    #       g_align 이 (b) 의 두 번째 micro-step 과 다르다. 따라서 값 일치가 아니라
    #       "덮어쓰기가 아니라 누적이 일어났는가" 를 본다.
    name = next(iter(accumulated))
    only_last = torch.allclose(accumulated[name], singles[-1][name], atol=1e-6)
    assert not only_last, ("param.grad 가 마지막 micro-step 값과 같다 = 덮어쓰기. "
                           "참조 구현의 accumulation 버그가 재발했다.")

    # 엄밀 검증: alignment 배치를 고정하면 합과 정확히 일치해야 한다
    fixed_align = collator([aligns[0], aligns[1]])

    class _FixedAlign(SafeGradTrainer):
        def sample_from_alignment(self):
            return {k: v.clone() for k, v in fixed_align.items()}

    def fresh_fixed():
        torch.manual_seed(123)
        m = _tiny_model(seed=0)
        args = TrainingArguments(
            output_dir=tmp, per_device_train_batch_size=2, gradient_accumulation_steps=2,
            num_train_epochs=1, learning_rate=1e-3, logging_steps=1000, save_strategy="no",
            eval_strategy="no", report_to=[], remove_unused_columns=False,
            dataloader_pin_memory=False, use_cpu=(DEVICE == "cpu"), seed=0)
        tt = _FixedAlign(model=m, args=args, train_dataset=rows, data_collator=collator,
                         ref_model=copy.deepcopy(ref), ref_mode="separate", rho=1.0)
        tt.init(aligns)
        return tt

    parts = []
    for mb in micro:
        tt = fresh_fixed()
        tt.training_step(tt.model, {k: v.clone() for k, v in mb.items()})
        parts.append({n: p.grad.detach().clone()
                      for n, p in tt._trainables(tt.model) if p.grad is not None})

    tt = fresh_fixed()
    for mb in micro:
        tt.training_step(tt.model, {k: v.clone() for k, v in mb.items()})
    both = {n: p.grad.detach().clone() for n, p in tt._trainables(tt.model) if p.grad is not None}

    worst = 0.0
    for n in both:
        expect = parts[0][n] + parts[1][n]
        worst = max(worst, float((both[n] - expect).abs().max()))
    assert worst < 1e-4, f"누적값이 두 micro-step 의 합과 다르다 (max diff {worst})"
    print(f"  [5] accum=2 → param.grad = g(mb1) + g(mb2), max diff {worst:.2e} ✓")


# ---------------------------------------------------------------------------
# 6. ref_mode='adapter_off' 가 'separate' 와 수학적으로 같은가
# ---------------------------------------------------------------------------
def test_ref_mode_equivalence():
    """LoRA 를 출발 모델 위에 바로 얹은 경우 '어댑터를 끈 모델' = θ0 임을 확인한다.

    B=0 인 초기 상태에서는 두 모드 모두 KL=0 이라 구분이 안 되므로, LoRA B 를
    일부러 흔들어 θ ≠ θ0 인 상태에서 비교한다.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        print("  [6] peft 없음 — 건너뜀")
        return

    tmp = tempfile.mkdtemp()
    base = _tiny_model(seed=0)
    ref = copy.deepcopy(base)                    # θ0 = 어댑터 얹기 전의 base

    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                     target_modules=["q_proj", "v_proj"], bias="none")
    peft_model = get_peft_model(base, cfg)
    torch.manual_seed(11)
    with torch.no_grad():                        # B=0 이면 θ=θ0 라 구분이 안 된다
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.add_(0.1 * torch.randn_like(p))

    args = TrainingArguments(
        output_dir=tmp, per_device_train_batch_size=2, gradient_accumulation_steps=1,
        num_train_epochs=1, learning_rate=1e-3, logging_steps=1000, save_strategy="no",
        eval_strategy="no", report_to=[], remove_unused_columns=False,
        dataloader_pin_memory=False, use_cpu=(DEVICE == "cpu"), seed=0)
    rows, aligns = _rows(8, seed=1), _rows(8, seed=2)

    losses, grads = {}, {}
    for mode, ref_arg in (("separate", ref), ("adapter_off", None)):
        t = SafeGradTrainer(model=peft_model, args=args, train_dataset=rows,
                            data_collator=_Collator(), ref_model=ref_arg,
                            ref_mode=mode, rho=1.0)
        t.init(aligns)
        batch = t._prepare_inputs(_Collator()([aligns[0], aligns[1]]))
        t._clear_grads(t._trainables(peft_model))
        loss = t.compute_alignment_loss(peft_model, batch)
        loss.backward()
        losses[mode] = float(loss.detach())
        grads[mode] = {n: p.grad.detach().clone()
                       for n, p in t._trainables(peft_model) if p.grad is not None}
        t._clear_grads(t._trainables(peft_model))

    assert losses["separate"] > 1e-6, "θ≠θ0 로 만들지 못했다 — 검사가 무의미하다"
    dl = abs(losses["separate"] - losses["adapter_off"])
    rel = dl / max(losses["separate"], 1e-12)
    assert rel < 1e-5, f"두 ref_mode 의 KL 이 다르다: {losses} (rel={rel})"

    worst = max(float((grads["separate"][n] - grads["adapter_off"][n]).abs().max())
                for n in grads["separate"])
    assert worst < 1e-5, f"두 ref_mode 의 g_align 이 다르다 (max diff {worst})"
    print(f"  [6] separate vs adapter_off: KL rel.diff {rel:.2e}, "
          f"g_align max diff {worst:.2e} ✓")


def main():
    print(f"SafeGrad tests on {DEVICE}")
    print("- surgery 수학")
    test_projection_math()
    print("- KL alignment")
    test_kl_zero_when_model_equals_ref()
    print("- gradient accumulation")
    test_gradient_accumulation_accumulates()
    print("- reference mode 동치성")
    test_ref_mode_equivalence()
    print("\n✅ 전부 통과")


if __name__ == "__main__":
    main()
