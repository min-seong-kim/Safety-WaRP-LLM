"""통합 runner: GSM8K 위 Standard LoRA / Original-space Projected LoRA / WSR-LoRA(element-wise) 비교.

세 방법은 동일 시작 checkpoint·GSM8K data·seed·rank·trainable param 수를 쓰고,
차이는 증분이 허용되는 좌표계/제약뿐이다 (wsr_lora_comparison.md 참조).

  lora                     : ΔW = s·BA
  original_projected_lora  : ΔW = s·BA(I−EEᵀ)          (A[:, safe_cols]=0, optimizer.step 후 재투영)
  wsr_lora                 : ΔW = [(1−M)∘(s·BA)] Uᵀ    (basis 공간 element freeze, forward 사전제약)
  safe_lora                : 표준 LoRA 학습 후 lora_B ← C·B (C=VVᵀ/‖V‖, cos≤thr 레이어만) 사후 투영
  adapter_subspace_lora    : ΔW_d^⊥ = s·B_d A_d (I−Q_S Q_Sᵀ)
                             Q_S = safety LoRA adapter ΔW_s=s_s B_s A_s 의 right singular vectors.
                             safety adapter 는 고정, downstream adapter 만 학습하며 A_d Q_S = 0 유지.
                             ⇒ W_final Q_S = W_safe Q_S (정확). Q_S 는 build_adapter_subspace.py 산출물.

⚠️ method=adapter_subspace_lora 일 때만 --model_name 은 **base** 모델(예: Llama-2-7b-chat-hf)이고
   safety 는 --safety_adapter_path 로 얹는다. 다른 method 는 --model_name 이 safety 모델이다.

저장은 dense: lora/orig/safe_lora/adapter_subspace → merge_and_unload,
wsr → restore_wsr_lora_to_linear. HF push.
"""
import argparse
import json
import logging
import os
import sys

# gsm8k_eval.finetune_gsm8k_full_params 의 import-time CUDA_VISIBLE_DEVICES 하드코딩은
# 현재 주석 처리되어 있으므로 아래 캡처/복원은 사실상 no-op 이다(셸/SLURM 값을 그대로 보존).
# 누군가 다시 하드코딩을 살릴 경우를 대비한 방어 코드로 남겨 둔다.
_INTENDED_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, TrainerCallback)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_eval.finetune_gsm8k_full_params import (  # noqa: E402
    tokenize_sft_example, DataCollatorForCausalLMWithPadding, _select_first_n)

# gsm8k_eval import 가 덮어쓴 CUDA_VISIBLE_DEVICES 복원 (torch cuda init 전에 수행)
if _INTENDED_CVD is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _INTENDED_CVD
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
from models.lora_wsr_elementwise import (  # noqa: E402
    switch_to_wsr_lora, mark_only_lora_trainable, restore_wsr_lora_to_linear)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lora_runner")

PROJ_TO_LT = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
              "up_proj": "ffn_up", "down_proj": "ffn_down", "gate_proj": "ffn_gate", "o_proj": "attn_o"}
LT_TO_PROJ = {v: k for k, v in PROJ_TO_LT.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["lora", "original_projected_lora", "wsr_lora", "wsr_lora_nou",
                             "safe_lora", "adapter_subspace_lora"])
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--safety_data_path", default="./data/circuit_breakers_train.json")
    ap.add_argument("--basis_dir", default=None)          # wsr_lora
    ap.add_argument("--mask_dir", default=None)           # wsr_lora
    ap.add_argument("--safecols_dir", default=None)       # original_projected_lora
    # safe_lora (바닐라 Safe LoRA, NeurIPS'24) — 사후 projection 전용
    ap.add_argument("--safelora_base_model", default="meta-llama/Llama-2-7b-chat-hf",
                    help="alignment delta V=W_aligned−W_base 의 base(비정렬 참조) 모델")
    ap.add_argument("--safelora_aligned_model", default=None,
                    help="aligned 모델 (기본: --model_name, 세 방법 공통 시작점 = safety 모델)")
    ap.add_argument("--safelora_select_type", default="threshold", choices=["threshold", "number"])
    ap.add_argument("--safelora_threshold", type=float, default=0.35)
    ap.add_argument("--safelora_num_proj_layers", type=int, default=10)
    ap.add_argument("--safelora_load_dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="base/aligned 로드 dtype (기본 float32=공식 구현과 동일)")
    # adapter_subspace_lora
    ap.add_argument("--safety_adapter_path", default=None,
                    help="safety LoRA adapter 디렉토리 (B_s, A_s). --model_name 은 base 모델이어야 함")
    ap.add_argument("--adapter_subspace_dir", default=None,
                    help="build_adapter_subspace.py 가 만든 Q_S artifact 디렉토리")
    ap.add_argument("--safety_adapter_mode", default="merge", choices=["merge", "keep"],
                    help="merge=safety adapter 를 base 에 병합 후 downstream adapter 1개 학습 / "
                         "keep=safety adapter 를 frozen 으로 유지하고 두 adapter 를 동시 활성화")
    ap.add_argument("--require_safety_adapter_for_all_targets", action="store_true",
                    help="Q_S 가 없는 target module 이 하나라도 있으면 error")
    ap.add_argument("--lora_param_dtype", default="float32", choices=["float32", "bfloat16"],
                    help="downstream LoRA 파라미터 dtype. 제약 A_d·Q_S=0 의 달성 정밀도를 좌우한다 "
                         "(fp32≈1e-7, bf16≈1e-3 상대오차). 다른 baseline 과 dtype 을 맞추려면 bfloat16")
    ap.add_argument("--project_optimizer_exp_avg", action="store_true",
                    help="AdamW exp_avg 도 매 step 투영 (grad 투영 시 대체로 중복이지만 안전장치)")
    ap.add_argument("--verify_every_steps", type=int, default=0,
                    help=">0 이면 N step 마다 제약 지표를 로깅")
    ap.add_argument("--keep_ratio", type=float, default=0.1)
    ap.add_argument("--direction_keep_ratio", type=float, default=0.1)
    ap.add_argument("--layer_type", default="attn_q,attn_k,attn_v,ffn_up,ffn_down")
    ap.add_argument("--target_layers", default="all")
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dataset_name", default="openai/gsm8k")
    ap.add_argument("--dataset_subset", default="main")
    ap.add_argument("--gsm8k_samples", type=int, default=0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_merged_model", action="store_true")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id", default=None)
    return ap.parse_args()


def _resolve_layers(num, spec):
    if spec == "all":
        return list(range(num))
    if "-" in spec:
        s, e = map(int, spec.split("-"))
        return list(range(s, e + 1))
    return [int(spec)]


def _name_to_key(name):
    parts = name.split(".")
    if "layers" not in parts:
        return None
    li = int(parts[parts.index("layers") + 1])
    for p, lt in PROJ_TO_LT.items():
        if p in parts:
            return (li, lt)
    return None


# ───────────────── WSR-LoRA basis/mask 로드 ─────────────────
def _load_basis(basis_dir, layer_types):
    basis = {}
    for lt in layer_types:
        d = os.path.join(basis_dir, lt)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("layer_") and f.endswith("_svd.pt"):
                li = int(f.split("_")[1])
                basis[(li, lt)] = torch.load(os.path.join(d, f), map_location="cpu")["U"]
    return basis


def _load_masks(mask_dir, layer_types):
    masks = {}
    for lt in layer_types:
        d = os.path.join(mask_dir, lt)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("layer_") and f.endswith("_mask.pt"):
                li = int(f.split("_")[1])
                masks[(li, lt)] = torch.load(os.path.join(d, f), weights_only=False)["mask"]
    return masks


def _load_safecols(safecols_dir, layer_types):
    sc = {}
    for lt in layer_types:
        d = os.path.join(safecols_dir, lt)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("layer_") and f.endswith("_safecols.pt"):
                li = int(f.split("_")[1])
                sc[(li, lt)] = torch.load(os.path.join(d, f))["safe_cols"]
    return sc


# ───────────────── original_projected 투영 콜백 ─────────────────
class ProjectionCallback(TrainerCallback):
    """optimizer.step 후 각 LoRA lora_A 의 safe_cols 열을 0으로 재투영."""
    def __init__(self, model, safecols):
        self.model = model
        self.safecols = safecols
        self.count = 0

    @torch.no_grad()
    def project(self):
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            key = _name_to_key(name)
            if key is None or key not in self.safecols:
                continue
            cols = self.safecols[key].to(next(iter(module.lora_A.values())).weight.device)
            for adapter_name, A in module.lora_A.items():
                A.weight.data[:, cols] = 0.0
        self.count += 1

    def on_train_begin(self, args, state, control, **kwargs):
        self.project()

    def on_step_end(self, args, state, control, **kwargs):
        self.project()


# ───────────────── adapter_subspace 투영 콜백 ─────────────────
class AdapterSubspaceCallback(TrainerCallback):
    """downstream lora_A 를 매 step 후 Q_S 의 직교보공간으로 재투영.

    gradient projection 은 projector 가 등록한 post-accumulate-grad hook 이 담당하고,
    이 콜백은 optimizer.step() 이후의 parameter 재투영을 담당한다. 둘 다 필요한 이유는
    models/adapter_subspace.AdapterSubspaceProjector docstring 참조 (AdamW 의 원소별
    normalization 이 선형성을 깨서 grad 만 투영해서는 제약이 유지되지 않는다).
    """

    def __init__(self, projector, verify_every_steps=0, logger_=None):
        self.projector = projector
        self.verify_every_steps = verify_every_steps
        self.log = logger_ or logger
        self._optimizer = None

    @property
    def count(self):
        return self.projector.count

    def project(self):
        self.projector.project(self._optimizer)

    def on_train_begin(self, args, state, control, **kwargs):
        self._optimizer = kwargs.get("optimizer", self._optimizer)
        self.project()

    def on_step_end(self, args, state, control, **kwargs):
        self._optimizer = kwargs.get("optimizer", self._optimizer)
        self.project()
        if self.verify_every_steps and state.global_step % self.verify_every_steps == 0:
            _, agg = self.projector.verify(check_mapping=False)
            self.log.info(
                f"[adapter_subspace] step {state.global_step} "
                f"constraint_A(max)={agg['constraint_A']['max']:.3e} "
                f"constraint_delta(max)={agg['constraint_delta']['max']:.3e} "
                f"delta_norm(mean)={agg['delta_norm']['mean']:.4f}")


def _setup_adapter_subspace(model, args, target_modules, dtype):
    """base model 위에 safety adapter + 제약된 downstream adapter 를 구성.

    Returns: (model, projector, downstream_adapter_name, info_dict)
    """
    from peft import LoraConfig, PeftModel, get_peft_model
    from models.adapter_subspace import AdapterSubspaceProjector, load_subspaces, read_report

    if not args.safety_adapter_path:
        raise ValueError("adapter_subspace_lora requires --safety_adapter_path (no fallback)")
    if not args.adapter_subspace_dir:
        raise ValueError("adapter_subspace_lora requires --adapter_subspace_dir "
                         "(build_adapter_subspace.py 로 먼저 생성)")

    subspaces = load_subspaces(args.adapter_subspace_dir)
    if not subspaces:
        raise ValueError(f"no Q_S artifacts found in {args.adapter_subspace_dir}")
    report = read_report(args.adapter_subspace_dir)
    logger.info(f"[adapter_subspace] loaded Q_S for {len(subspaces)} modules from "
                f"{args.adapter_subspace_dir}")
    if report:
        logger.info(f"  protected_k: min={report['protected_k_min']} max={report['protected_k_max']} "
                    f"mean={report['protected_k_mean']:.2f} | "
                    f"max recon err={report['max_reconstruction_error']:.3e}")

    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     bias="none", task_type="CAUSAL_LM", target_modules=target_modules)

    if args.safety_adapter_mode == "merge":
        logger.info(f"[adapter_subspace] merging safety adapter into base: "
                    f"{args.safety_adapter_path}")
        model = PeftModel.from_pretrained(model, args.safety_adapter_path)
        model = model.merge_and_unload()          # W_safe = W_base + s_s B_s A_s
        model = get_peft_model(model, cfg)
        downstream_name = "default"
    else:
        logger.info(f"[adapter_subspace] keeping safety adapter frozen: "
                    f"{args.safety_adapter_path}")
        model = PeftModel.from_pretrained(model, args.safety_adapter_path,
                                          adapter_name="safety", is_trainable=False)
        model.add_adapter("downstream", cfg)
        # PeftModel.set_adapter 는 단일 이름만 받으므로 LoraModel 쪽 API 로 둘 다 활성화한다.
        model.base_model.set_adapter(["safety", "downstream"])
        downstream_name = "downstream"
        frozen = 0
        for n, p in model.named_parameters():
            if ".safety." in n:
                p.requires_grad_(False)
                frozen += p.numel()
        logger.info(f"[adapter_subspace] safety adapter frozen: {frozen:,} params")

    # LoRA 파라미터 dtype: 제약 달성 정밀도를 좌우하므로 명시적으로 로깅한다.
    if args.lora_param_dtype == "float32" and dtype != torch.float32:
        n_cast = 0
        for n, p in model.named_parameters():
            if "lora_" in n and p.requires_grad:
                p.data = p.data.float()
                n_cast += 1
        logger.info(f"[adapter_subspace] downstream LoRA params → fp32 ({n_cast} tensors). "
                    f"제약 A_d·Q_S=0 을 ~1e-7 상대오차로 유지하기 위함 "
                    f"(bf16 이면 ~1e-3 에서 멈춘다).")

    model.print_trainable_parameters()

    projector = AdapterSubspaceProjector(
        model, subspaces, adapter_name=downstream_name,
        project_exp_avg=args.project_optimizer_exp_avg, logger_=logger)

    if projector.unconstrained_targets:
        msg = (f"Q_S 가 없어 제약이 걸리지 않는 target module {len(projector.unconstrained_targets)} 개: "
               f"{projector.unconstrained_targets[:5]}{' ...' if len(projector.unconstrained_targets) > 5 else ''}")
        if args.require_safety_adapter_for_all_targets:
            raise ValueError(msg + " (--require_safety_adapter_for_all_targets)")
        logger.warning(msg + " → 일반 LoRA 로 학습됩니다.")

    projector.register_grad_hooks()
    logger.info(f"[adapter_subspace] constrained modules: {projector.num_constrained}, "
                f"unconstrained: {len(projector.unconstrained_targets)}")

    info = {
        "safety_adapter_path": args.safety_adapter_path,
        "adapter_subspace_dir": args.adapter_subspace_dir,
        "safety_adapter_mode": args.safety_adapter_mode,
        "downstream_adapter_name": downstream_name,
        "lora_param_dtype": args.lora_param_dtype,
        "num_constrained_modules": projector.num_constrained,
        "num_unconstrained_targets": len(projector.unconstrained_targets),
        "subspace_report": {k: report[k] for k in
                            ("num_modules", "protected_k_min", "protected_k_max",
                             "protected_k_mean", "max_reconstruction_error",
                             "max_orthogonality_error")} if report else None,
    }
    return model, projector, downstream_name, info


def build_gsm8k(tokenizer, args):
    ds = load_dataset(args.dataset_name, args.dataset_subset, split="train")
    if args.gsm8k_samples > 0:
        ds = _select_first_n(ds, args.gsm8k_samples)

    def preprocess(ex):
        return tokenize_sft_example(ex["question"], ex["answer"], tokenizer, args.max_length, args.model_name)

    tok_ds = ds.map(preprocess, remove_columns=ds.column_names, desc="tokenizing gsm8k")
    return tok_ds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"=== method={args.method} lr={args.learning_rate} r={args.lora_r} ===")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    layer_types = [x.strip() for x in args.layer_type.split(",")]
    target_modules = [x.strip() for x in args.target_modules.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map={"": 0})
    model.config.use_cache = False

    projection_cb = None
    projector = None
    subspace_info = None
    subspace_verify = None

    if args.method == "adapter_subspace_lora":
        logger.info("⚠️ method=adapter_subspace_lora: --model_name 은 BASE 모델이어야 하며 "
                    f"safety 는 --safety_adapter_path 로 얹습니다 (model_name={args.model_name})")

    # ───────────── method setup ─────────────
    if args.method == "wsr_lora":
        if not args.basis_dir or not args.mask_dir:
            raise ValueError("wsr_lora requires --basis_dir and --mask_dir (no fallback)")
        basis = _load_basis(args.basis_dir, layer_types)
        masks = _load_masks(args.mask_dir, layer_types)
        target_layers = _resolve_layers(len(model.model.layers), args.target_layers)
        converted = switch_to_wsr_lora(model, layer_types, args.target_layers,
                                       r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        n_set = 0
        for key, mod in converted.items():
            if key not in basis or key not in masks:
                raise ValueError(f"missing basis/mask for {key}")
            U = basis[key]
            M = torch.as_tensor(masks[key])
            # 검증: U (n×n), mask (m×n)
            assert U.shape[0] == mod.in_features, f"{key}: U {tuple(U.shape)} vs in {mod.in_features}"
            assert tuple(M.shape) == (mod.out_features, mod.in_features), f"{key}: mask {tuple(M.shape)}"
            mod.set_basis_and_mask(U, M)
            n_set += 1
        logger.info(f"✓ WSR-LoRA basis/mask set for {n_set} modules")
        model = model.to(0)
        trainable = mark_only_lora_trainable(model)
        logger.info(f"trainable params (WSR-LoRA): {trainable:,}")
    elif args.method == "wsr_lora_nou":
        # WSR-LoRA 에서 rotation(U)만 제거한 ablation:
        #   ΔW = (1-M) ∘ (s·BA)  (원래 weight 공간, element-wise mask, forward-내 freeze).
        #   mask 는 원래공간 element importance |∂L/∂W| 로 계산(train.py --phase 2 --original_space_mask).
        if not args.mask_dir:
            raise ValueError("wsr_lora_nou requires --mask_dir (original-space element mask; no basis)")
        masks = _load_masks(args.mask_dir, layer_types)
        converted = switch_to_wsr_lora(model, layer_types, args.target_layers,
                                       r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        n_set = 0
        for key, mod in converted.items():
            if key not in masks:
                raise ValueError(f"missing mask for {key}")
            M = torch.as_tensor(masks[key])
            assert tuple(M.shape) == (mod.out_features, mod.in_features), f"{key}: mask {tuple(M.shape)}"
            mod.set_basis_and_mask(None, M)   # U=None → no rotation
            n_set += 1
        logger.info(f"✓ WSR-LoRA(no-rotation) mask set for {n_set} modules")
        model = model.to(0)
        trainable = mark_only_lora_trainable(model)
        logger.info(f"trainable params (WSR-LoRA-noU): {trainable:,}")
    elif args.method == "adapter_subspace_lora":
        model, projector, downstream_adapter, subspace_info = _setup_adapter_subspace(
            model, args, target_modules, dtype)
        projection_cb = AdapterSubspaceCallback(projector, args.verify_every_steps, logger)
    else:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        if args.method == "original_projected_lora":
            if not args.safecols_dir:
                raise ValueError("original_projected_lora requires --safecols_dir (no fallback)")
            safecols = _load_safecols(args.safecols_dir, layer_types)
            if not safecols:
                raise ValueError(f"no safe_cols found in {args.safecols_dir}")
            projection_cb = ProjectionCallback(model, safecols)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_ds = build_gsm8k(tok, args)
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tok)

    targs = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator,
                      callbacks=[projection_cb] if projection_cb else None)
    trainer.train()
    logger.info("✓ training done")

    # ───────────── dense 저장 ─────────────
    if projection_cb is not None:
        projection_cb.project()  # 마지막 재투영

    # ───────────── adapter_subspace 제약 검증 (§10 A–E) ─────────────
    if projector is not None:
        from models.adapter_subspace import load_subspaces
        subspaces = load_subspaces(args.adapter_subspace_dir)
        rows, agg = projector.verify(subspaces=subspaces, check_mapping=True)
        subspace_verify = {"aggregate": agg, "per_layer": rows}
        with open(os.path.join(args.output_dir, "subspace_verification.json"), "w") as f:
            json.dump(subspace_verify, f, indent=2)
        logger.info("=" * 70)
        logger.info("[adapter_subspace] constraint verification")
        for field, label in (("constraint_A", "A  ‖A_d Q_S‖/‖A_d‖        "),
                             ("constraint_delta", "B  ‖ΔW_d Q_S‖/‖ΔW_d‖     "),
                             ("mapping_drift", "C  ‖W_final Q−W_safe Q‖/‖·‖"),
                             ("safety_recon_error", "D  safety ΔW_s recon err  "),
                             ("delta_norm", "E  ‖B_d A_d‖_F            ")):
            v = agg.get(field)
            if v:
                logger.info(f"  {label} max={v['max']:.3e}  mean={v['mean']:.3e}")
        if agg["constraint_A"] and agg["constraint_A"]["max"] > 1e-2:
            logger.warning("constraint_A 가 큽니다 — 제약이 제대로 유지되지 않았습니다. "
                           "--lora_param_dtype float32 인지, grad hook 이 등록됐는지 확인하세요.")
        if agg["delta_norm"] and agg["delta_norm"]["mean"] < 1e-4:
            logger.warning("downstream update 가 거의 0 입니다 — 보호 subspace 가 너무 커서 "
                           "학습 여지가 없을 수 있습니다 (top_k/energy 를 낮춰보세요).")
        logger.info("=" * 70)
        projector.remove_grad_hooks()

    safelora_stats = None
    if args.method == "safe_lora":
        from models.safelora_baseline import apply_safelora
        aligned = args.safelora_aligned_model or args.model_name
        sl_dtype = {"float32": torch.float32, "float16": torch.float16,
                    "bfloat16": torch.bfloat16}[args.safelora_load_dtype]
        logger.info(f"[SafeLoRA] projection: base={args.safelora_base_model} aligned={aligned} "
                    f"select={args.safelora_select_type} thr={args.safelora_threshold}")
        safelora_stats = apply_safelora(
            model, base_path=args.safelora_base_model, aligned_path=aligned,
            target_modules=target_modules, r=args.lora_r,
            select_layers_type=args.safelora_select_type,
            threshold=args.safelora_threshold, num_proj_layers=args.safelora_num_proj_layers,
            compute_device=("cuda" if torch.cuda.is_available() else "cpu"),
            load_dtype=sl_dtype, logger=logger)
        logger.info(f"[SafeLoRA] stats: {safelora_stats}")

    if args.method in ("wsr_lora", "wsr_lora_nou"):
        restore_wsr_lora_to_linear(model)
        merged = model
    else:
        merged = model.merge_and_unload()

    merged_dir = os.path.join(args.output_dir, "merged_model")
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(merged_dir)
    logger.info(f"✓ merged model saved: {merged_dir}")

    # sanity generation
    try:
        merged.eval()
        q = "Natalia sold clips to 48 friends in April, and half as many in May. How many total?"
        msgs = [{"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii = tok(text, return_tensors="pt").to(next(merged.parameters()).device)
        with torch.no_grad():
            out = merged.generate(**ii, max_new_tokens=64, do_sample=False)
        logger.info("sanity gen: " + tok.decode(out[0][ii["input_ids"].shape[1]:], skip_special_tokens=True)[:200])
    except Exception as e:
        logger.warning(f"sanity gen failed: {e}")

    summary = {"method": args.method, "model_name": args.model_name, "lr": args.learning_rate,
               "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "keep_ratio": args.keep_ratio,
               "direction_keep_ratio": args.direction_keep_ratio, "epochs": args.epochs,
               "merged_dir": merged_dir, "hf_repo_id": args.hf_repo_id,
               "projection_calls": projection_cb.count if projection_cb else None,
               "safelora": safelora_stats,
               "adapter_subspace": subspace_info,
               "adapter_subspace_verify": subspace_verify["aggregate"] if subspace_verify else None}
    json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--push_to_hub requires --hf_repo_id")
        # ⚠️ push 실패(토큰 무효/네트워크 등)가 학습 파이프라인을 중단시키지 않도록 non-fatal.
        #    merged 모델은 이미 merged_dir(/scratch2)에 저장되어 있으므로, 실패 시
        #    나중에 scripts/push_safelora_from_scratch.py 로 재업로드하면 됨.
        try:
            logger.info(f"pushing to hub: {args.hf_repo_id}")
            merged.push_to_hub(args.hf_repo_id)
            tok.push_to_hub(args.hf_repo_id)
            logger.info(f"✓ pushed: https://huggingface.co/{args.hf_repo_id}")
        except Exception as e:
            logger.error(f"PUSH_FAILED repo={args.hf_repo_id} merged_dir={merged_dir} "
                         f"err={type(e).__name__}: {str(e)[:200]}")
            logger.error("→ 모델은 저장됨. 유효 HF 토큰으로 나중에 재업로드 필요.")

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
