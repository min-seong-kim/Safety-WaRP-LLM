#!/usr/bin/env python
"""Stage W2 — safety gradient importance G_l 로부터 **column score** 를 산출한다.

명세 `adapter_aware_column_wsr_lora_implementation.md` §3~§5.

    G_l   = Σ_b |G_W^(b) U_l|                     ← 기존 Phase 2 가 이미 이 형태로 누적한다
    ΔW̃_s  = s_s · B_s (A_s U_l)                    ← chunk 단위로만 만든다
    T_l   = |ΔW̃_s| ⊙ G_l
    c_l,j = ‖T_l[:,j]‖₂  (또는 L1)

왜 backward 를 한 번만 도는가
-----------------------------
`adapter_taylor` 는 원소별 곱 후 열 축약(L2)이라 G 의 열 노름만으로 계산할 수 없고,
L2 는 배치 누적과 교환되지 않으므로 **원소별 G_l 텐서가 반드시 필요**하다.
그런데 G_l 이 메모리에 있는 시점에 세 mode 의 score 를 모두 뽑으면 backward 1회로 끝난다.
raw G_l 은 160 모듈 fp32 기준 약 18GB 라 디스크에 쓰지 않는 것이 이득이다
(`--save_raw_importance` 로 원하면 덤프 가능).

⚠️ `--wsr_basis_dir` 의 basis U 는 반드시 **safety adapter 가 반영된 W_safe** 에서
   수집한 activation 으로 만든 것이어야 한다 (명세 §2.1). base model 로 만든 basis 를
   넣으면 이 방법의 전제가 깨진다. 이 스크립트는 basis metadata 의 model 기록과
   `--safety_merged_model_path` 를 대조해 경고한다.

산출물 (작다 — 모듈당 n floats × 3 mode):
    <out_dir>/<layer_type>/layer_NN_scores.pt
    <out_dir>/report.json
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.adapter_wsr_column import (  # noqa: E402
    VALID_AGGREGATIONS, VALID_IMPORTANCE_MODES, compute_column_scores,
    save_scores, spearman,
)
from models.adapter_subspace import PROJ_TO_LT  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wsr_column_scores")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety_merged_model_path", required=True,
                    help="W_safe = W_base + ΔW_s (dense merged). G_l 은 여기서 잰다.")
    ap.add_argument("--safety_adapter_path", required=True,
                    help="safety LoRA adapter (B_s, A_s). ΔW̃_s 계산에 필요.")
    ap.add_argument("--wsr_basis_dir", required=True,
                    help="Phase 1 basis dir (W_safe 에서 만든 것이어야 함)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--safety_data_path", default="./data/circuit_breakers_train.json")

    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--layer_type", default="attn_q,attn_k,attn_v,ffn_up,ffn_down")
    ap.add_argument("--target_layers", default="all")

    ap.add_argument("--column_aggregation", default="l2", choices=list(VALID_AGGREGATIONS))
    ap.add_argument("--importance_chunk_size", type=int, default=128)
    ap.add_argument("--importance_dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--save_raw_importance", default=None,
                    help="지정하면 raw G_l 도 이 디렉토리에 덤프한다 (fp16, 약 9GB)")
    ap.add_argument("--require_safety_adapter_for_all_targets", action="store_true")
    ap.add_argument("--allow_missing_safety_adapter_modules", action="store_true",
                    help="명세 §12: safety adapter 에 없는 module 을 gradient_only 로 처리")
    return ap.parse_args()


def _file_sha1(path, limit=1 << 20):
    """artifact 추적용 부분 해시 (전체 해시는 13GB 모델에 과하다)."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(limit))
        return h.hexdigest()[:16]
    except Exception:
        return None


def load_safety_adapter(path, target_projs, allow_missing):
    """safety adapter state_dict → {(layer_idx, layer_type): {'A':…, 'B':…}} + scaling."""
    from safetensors.torch import load_file

    cfg_path = os.path.join(path, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    r_s = int(cfg["r"])
    alpha_s = float(cfg["lora_alpha"])
    scaling = alpha_s / r_s
    if cfg.get("use_rslora"):
        import math
        scaling = alpha_s / math.sqrt(r_s)

    st_path = os.path.join(path, "adapter_model.safetensors")
    state = load_file(st_path) if os.path.isfile(st_path) else torch.load(
        os.path.join(path, "adapter_model.bin"), map_location="cpu")

    pairs = {}
    for name, tensor in state.items():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        li = int(parts[parts.index("layers") + 1])
        lt = next((PROJ_TO_LT[p] for p in PROJ_TO_LT if p in parts), None)
        if lt is None:
            continue
        slot = pairs.setdefault((li, lt), {})
        slot["A" if "lora_A" in name else "B"] = tensor

    cfg_targets = set(cfg.get("target_modules") or [])
    missing = [p for p in target_projs if cfg_targets and p not in cfg_targets]
    if missing and not allow_missing:
        raise ValueError(
            f"safety adapter 에 없는 target module: {missing}. "
            f"--allow_missing_safety_adapter_modules 를 주면 gradient_only 로 처리합니다.")
    return pairs, scaling, r_s, alpha_s, missing


def compute_gradient_importance(args):
    """기존 Phase 2 를 그대로 재사용해 G_l 을 얻는다.

    재구현하지 않는 이유: importance 정의(`Σ_b |G_W^(b) U|`)가 기존 WSR 실험과
    **비트 단위로 같은 코드 경로**를 타야 비교가 성립한다.
    """
    from models.phase2_importance_per_layer import Phase2ImportanceScorerPerLayer

    p2_args = SimpleNamespace(
        batch_size=args.batch_size,
        circuit_breakers_path=args.safety_data_path,
        device=args.device,
        dtype=args.dtype,
        layer_type=args.layer_type,
        target_layers=args.target_layers,
        max_length=args.max_length,
        dataset_phase2="circuit_breakers",
        keep_ratio=0.1,          # mask 는 만들지 않으므로 사용되지 않음
        output_dir=args.out_dir,
    )
    scorer = Phase2ImportanceScorerPerLayer(
        p2_args, logger, basis_dir=args.wsr_basis_dir,
        phase0_model_dir=args.safety_merged_model_path)

    scorer.load_basis()
    scorer.load_model()
    scorer.load_safety_data()
    scorer.convert_to_warp_modules()
    scorer.reparameterize_weights()
    scorer.compute_importance()          # → scorer.importances: {(li,lt): np.ndarray (m,n) fp32}

    basis = {k: v["U"] for k, v in scorer.basis_data.items()}
    stats = dict(scorer.stats)

    # 모델을 즉시 해제해 GPU 를 score 계산에 내준다.
    del scorer.model
    scorer.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scorer.importances, basis, stats


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    target_projs = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    layer_types = [x.strip() for x in args.layer_type.split(",") if x.strip()]

    # ── target_modules ↔ layer_type 정합성 (조용한 불일치 방지) ──
    implied = {PROJ_TO_LT[p] for p in target_projs if p in PROJ_TO_LT}
    if implied != set(layer_types):
        raise ValueError(f"--target_modules {target_projs} → {sorted(implied)} 가 "
                         f"--layer_type {layer_types} 와 다릅니다.")

    logger.info("=" * 70)
    logger.info("Stage W2: adapter-aware WSR column scores")
    logger.info(f"  W_safe : {args.safety_merged_model_path}")
    logger.info(f"  adapter: {args.safety_adapter_path}")
    logger.info(f"  basis  : {args.wsr_basis_dir}")
    logger.info(f"  agg={args.column_aggregation} chunk={args.importance_chunk_size} "
                f"dtype={args.importance_dtype}")
    logger.info("=" * 70)

    # ── basis 출처 확인 (명세 §2.1) ──
    meta_path = os.path.join(args.wsr_basis_dir, "metadata.json")
    basis_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            basis_meta = json.load(f)
        src = str(basis_meta.get("model_name") or basis_meta.get("phase0_model_dir") or "")
        if src and os.path.basename(src.rstrip("/")) != os.path.basename(
                args.safety_merged_model_path.rstrip("/")):
            logger.warning(
                f"⚠️ basis metadata 의 모델({src}) 이 --safety_merged_model_path "
                f"({args.safety_merged_model_path}) 와 다릅니다. 명세 §2.1 은 U 를 "
                f"W_safe 에서 만들 것을 요구합니다. 의도한 것이 아니면 basis 를 재생성하세요.")

    adapter_pairs, s_s, r_s, alpha_s, missing = load_safety_adapter(
        args.safety_adapter_path, target_projs, args.allow_missing_safety_adapter_modules)
    logger.info(f"safety adapter: r={r_s} alpha={alpha_s} scaling={s_s:.6f} "
                f"modules={len(adapter_pairs)}")

    importances, basis, stats = compute_gradient_importance(args)
    logger.info(f"gradient importance: {len(importances)} modules, "
                f"samples={stats['total_samples']} tokens={stats['total_tokens']}")

    compute_dtype = {"float32": torch.float32, "float64": torch.float64}[args.importance_dtype]
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.save_raw_importance:
        os.makedirs(args.save_raw_importance, exist_ok=True)

    rows = []
    keys = sorted(set(importances) | set(adapter_pairs))
    for key in keys:
        li, lt = key
        if lt not in layer_types:
            continue
        G_np = importances.get(key)
        slot = adapter_pairs.get(key)
        U = basis.get(key)

        if U is None:
            logger.warning(f"layer {li} {lt}: basis 없음 → skip")
            continue
        if G_np is None:
            logger.warning(f"layer {li} {lt}: gradient importance 없음 → skip")
            continue

        G = torch.as_tensor(G_np)
        modes = list(VALID_IMPORTANCE_MODES)
        if slot is None or "A" not in slot or "B" not in slot:
            if not args.allow_missing_safety_adapter_modules:
                raise ValueError(f"layer {li} {lt}: safety adapter 짝이 없습니다. "
                                 f"--allow_missing_safety_adapter_modules 참조.")
            logger.warning(f"layer {li} {lt}: safety adapter 없음 → gradient_only 만 계산")
            modes = ["gradient_only"]
            B_s = A_s = None
        else:
            A_s, B_s = slot["A"], slot["B"]

        U_dev = U.to(device=device, dtype=compute_dtype)
        scores = compute_column_scores(
            B_s if B_s is not None else torch.empty(0),
            A_s if A_s is not None else torch.empty(0),
            s_s, U_dev, G, modes=modes,
            aggregation=args.column_aggregation,
            chunk_size=args.importance_chunk_size,
            compute_dtype=compute_dtype)

        payload = {
            "layer": li, "layer_type": lt,
            "scores": {m: scores[m] for m in scores},
            "column_aggregation": args.column_aggregation,
            "n": int(U.shape[1]), "m": int(G.shape[0]),
            "safety_adapter_scaling": s_s,
            "safety_adapter_r": r_s,
            "has_safety_adapter": B_s is not None,
        }
        save_scores(args.out_dir, key, payload)

        if args.save_raw_importance:
            d = os.path.join(args.save_raw_importance, lt)
            os.makedirs(d, exist_ok=True)
            torch.save({"G": G.to(torch.float16)},
                       os.path.join(d, f"layer_{li:02d}_G.pt"))

        # ── 진단: mode 간 Spearman (명세 §15.6) ──
        row = {"layer": li, "layer_type": lt, "n": int(U.shape[1]),
               "has_safety_adapter": B_s is not None}
        if len(scores) == 3:
            row["spearman_grad_vs_taylor"] = spearman(scores["gradient_only"],
                                                      scores["adapter_taylor"])
            row["spearman_mag_vs_taylor"] = spearman(scores["adapter_magnitude_only"],
                                                     scores["adapter_taylor"])
            row["spearman_grad_vs_mag"] = spearman(scores["gradient_only"],
                                                   scores["adapter_magnitude_only"])
        for m_, c in scores.items():
            row[f"{m_}_max"] = float(c.max())
            row[f"{m_}_mean"] = float(c.mean())
        rows.append(row)

        del G, U_dev, scores
        importances[key] = None          # 처리 끝난 모듈은 즉시 해제 (18GB 피크 억제)
        if len(rows) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"  … {len(rows)}/{len(keys)} modules")

    def _mean(f):
        v = [r[f] for r in rows if r.get(f) is not None]
        return sum(v) / len(v) if v else None

    report = {
        "num_modules": len(rows),
        "safety_merged_model_path": args.safety_merged_model_path,
        "safety_adapter_path": args.safety_adapter_path,
        "safety_adapter_sha1": _file_sha1(
            os.path.join(args.safety_adapter_path, "adapter_model.safetensors")),
        "wsr_basis_dir": args.wsr_basis_dir,
        "basis_metadata": basis_meta,
        "safety_data_path": args.safety_data_path,
        "importance_samples": stats["total_samples"],
        "importance_tokens": stats["total_tokens"],
        "column_aggregation": args.column_aggregation,
        "importance_dtype": args.importance_dtype,
        "chunk_size": args.importance_chunk_size,
        "safety_adapter_r": r_s, "safety_adapter_alpha": alpha_s, "scaling_s": s_s,
        "missing_target_modules": missing,
        "mean_spearman_grad_vs_taylor": _mean("spearman_grad_vs_taylor"),
        "mean_spearman_mag_vs_taylor": _mean("spearman_mag_vs_taylor"),
        "mean_spearman_grad_vs_mag": _mean("spearman_grad_vs_mag"),
        "layers": rows,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"✓ {len(rows)} modules → {args.out_dir}")
    logger.info(f"  Spearman  grad↔taylor {report['mean_spearman_grad_vs_taylor']}")
    logger.info(f"            mag ↔taylor {report['mean_spearman_mag_vs_taylor']}")
    logger.info(f"            grad↔mag    {report['mean_spearman_grad_vs_mag']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
