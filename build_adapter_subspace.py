"""Safety LoRA adapter → 보호할 input-side subspace Q_S artifact 생성.

adapter_subspace_lora 의 Stage 1. base model 을 **로드하지 않는다** — 필요한 것은
safety adapter 의 (B_s, A_s) 뿐이므로 adapter_model.safetensors 만 읽으면 된다.

    ΔW_s = s_s B_s A_s = P_s Σ_s Q_sᵀ      (compact SVD, dense ΔW_s 미생성)
    Q_S  = Q_s[:, :k],   k = top-k | cumulative-energy | all-effective

출력:
    <out_dir>/<layer_type>/layer_NN_subspace.pt   {'Q_S', 'singular_values', ...}
    <out_dir>/report.json                          레이어별 진단 전체
    <out_dir>/metadata.json                        빌드 설정

사용 예:
    python build_adapter_subspace.py \
        --safety_adapter_path checkpoints/phase0_lora_.../ \
        --out_dir checkpoints/adapter_subspace/all_effective \
        --target_modules q_proj,k_proj,v_proj,up_proj,down_proj
"""
import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.adapter_subspace import (  # noqa: E402
    PROJ_TO_LT, compact_svd_from_lora, cumulative_energy, lowrank_fro_norm,
    orthogonality_error, relative_reconstruction_error, save_subspace,
    select_protected_rank)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_adapter_subspace")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety_adapter_path", required=True,
                    help="peft 로 저장된 safety LoRA adapter 디렉토리 (adapter_model.safetensors)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    ap.add_argument("--target_layers", default="all", help="all | N | A-B")
    # ── 보호 subspace 선택 (우선순위: top_k → energy → all-effective) ──
    ap.add_argument("--adapter_subspace_top_k", type=int, default=None,
                    help="상위 k 개 singular direction 보호")
    ap.add_argument("--adapter_subspace_energy", type=float, default=None,
                    help="누적 σ² 에너지 τ 를 만족하는 최소 k 보호 (예: 0.90/0.95/0.99)")
    ap.add_argument("--svd_rank_tol", type=float, default=1e-6,
                    help="numerical rank 판정: σ_i > σ_max · tol")
    ap.add_argument("--svd_dtype", default="float64", choices=["float32", "float64"],
                    help="QR/SVD 연산 dtype. in_features 가 큰 층에서 fp32 QR 은 직교성 오차가 "
                         "~6e-6 까지 커진다. 이 크기(m×r)에서는 fp64 비용이 무시할 만하므로 기본값 fp64")
    # ── 검증 ──
    ap.add_argument("--require_safety_adapter_for_all_targets", action="store_true",
                    help="target module 중 safety adapter 가 없는 것이 있으면 error")
    ap.add_argument("--allow_nonstandard_lora", action="store_true",
                    help="rank_pattern/alpha_pattern 이 있어도 진행 (DoRA 는 항상 거부)")
    ap.add_argument("--verify_dense_layers", type=int, default=0,
                    help="처음 N 개 module 에 한해 dense ΔW_s 를 만들어 low-rank 공식과 교차검증")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _resolve_layers(spec):
    if spec == "all":
        return None
    if "-" in spec:
        s, e = map(int, spec.split("-"))
        return set(range(s, e + 1))
    return {int(spec)}


def _load_adapter_state(path):
    st_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file
        return load_file(st_path)
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"adapter weights not found in {path} (adapter_model.safetensors|.bin)")


def _load_adapter_config(path):
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {path}")
    with open(cfg_path) as f:
        return json.load(f)


def _key_of(param_name):
    """peft state_dict key → ((layer_idx, layer_type), proj_name). 실패 시 (None, None)."""
    parts = param_name.split(".")
    if "layers" not in parts:
        return None, None
    try:
        layer_idx = int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None, None
    for proj, lt in PROJ_TO_LT.items():
        if proj in parts:
            return (layer_idx, lt), proj
    return None, None


def _collect_pairs(state, target_projs, layer_filter):
    """state_dict → {(layer_idx, layer_type): {'A': tensor, 'B': tensor}}"""
    pairs = {}
    for pname, tensor in state.items():
        if ".lora_A" not in pname and ".lora_B" not in pname:
            continue
        key, proj = _key_of(pname)
        if key is None or proj not in target_projs:
            continue
        if layer_filter is not None and key[0] not in layer_filter:
            continue
        slot = "A" if ".lora_A" in pname else "B"
        pairs.setdefault(key, {})[slot] = tensor
    return pairs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_projs = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    unknown = [p for p in target_projs if p not in PROJ_TO_LT]
    if unknown:
        raise ValueError(f"unknown target_modules: {unknown} (지원: {list(PROJ_TO_LT)})")
    layer_filter = _resolve_layers(args.target_layers)

    cfg = _load_adapter_config(args.safety_adapter_path)
    state = _load_adapter_state(args.safety_adapter_path)

    # ── 표준 LoRA 여부 검사 (§9) ────────────────────────────────────
    if cfg.get("use_dora", False):
        raise ValueError("DoRA adapter 는 지원하지 않습니다 (ΔW 가 B A 형태가 아님). "
                         "표준 LoRA 로 safety tuning 을 다시 수행하세요.")
    if cfg.get("lora_bias", False) or cfg.get("bias", "none") not in ("none", None):
        logger.warning(f"adapter bias={cfg.get('bias')} — bias 항은 subspace 계산에서 무시됩니다.")
    r_s = int(cfg["r"])
    alpha_s = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    rank_pattern = cfg.get("rank_pattern") or {}
    alpha_pattern = cfg.get("alpha_pattern") or {}
    if (rank_pattern or alpha_pattern) and not args.allow_nonstandard_lora:
        raise ValueError(
            "rank_pattern/alpha_pattern 이 설정된 adapter 입니다 (per-layer scaling). "
            "현재 구현은 균일 r/alpha 만 지원합니다. --allow_nonstandard_lora 로 강행 가능하나 "
            "layer 별 scaling 이 틀어질 수 있습니다.")

    if use_rslora:
        scaling_s = alpha_s / (r_s ** 0.5)
        logger.info(f"rank-stabilized LoRA 감지: scaling = alpha/sqrt(r) = {scaling_s:.6f}")
    else:
        scaling_s = alpha_s / r_s
    logger.info(f"safety adapter: r={r_s} alpha={alpha_s} scaling={scaling_s:.6f} "
                f"target_modules(cfg)={cfg.get('target_modules')}")

    pairs = _collect_pairs(state, target_projs, layer_filter)
    if not pairs:
        raise ValueError(f"safety adapter 에서 target module 을 하나도 찾지 못했습니다 "
                         f"(target_modules={target_projs}).")

    # ── safety adapter 가 커버하지 못하는 target 확인 (§8) ──────────
    cfg_targets = set(cfg.get("target_modules") or [])
    missing_projs = [p for p in target_projs if cfg_targets and p not in cfg_targets]
    if missing_projs:
        msg = (f"safety adapter 에 없는 target module: {missing_projs} → 해당 layer 는 "
               f"subspace 제약 없이 일반 LoRA 로 학습됩니다.")
        if args.require_safety_adapter_for_all_targets:
            raise ValueError(msg.replace("→ 해당 layer 는 subspace 제약 없이 일반 LoRA 로 학습됩니다.",
                                         "(--require_safety_adapter_for_all_targets)"))
        logger.warning(msg)

    # ── 레이어별 compact SVD → Q_S ──────────────────────────────────
    rows = []
    n_dense_checked = 0
    for key in sorted(pairs.keys()):
        layer_idx, layer_type = key
        slot = pairs[key]
        if "A" not in slot or "B" not in slot:
            logger.warning(f"layer {layer_idx} {layer_type}: lora_A/lora_B 짝이 불완전 → skip")
            continue

        A = slot["A"].to(args.device)          # (r, n)
        B = slot["B"].to(args.device)          # (m, r)
        svd_dtype = {"float32": torch.float32, "float64": torch.float64}[args.svd_dtype]
        P, S, Q, r_nominal = compact_svd_from_lora(B, A, scaling_s, rank_tol=args.svd_rank_tol,
                                                   dtype=svd_dtype)
        r_eff = int(S.numel())

        if r_eff == 0:
            logger.warning(f"layer {layer_idx} {layer_type}: safety update 가 수치적으로 0 "
                           f"(r_eff=0) → 제약 미적용")
            k, mode = 0, "empty"
        else:
            k, mode = select_protected_rank(S, args.adapter_subspace_top_k,
                                            args.adapter_subspace_energy)

        Q_S = Q[:, :k].contiguous()
        recon = relative_reconstruction_error(B, A, scaling_s, P, S, Q)
        ortho = orthogonality_error(Q_S)
        energies = cumulative_energy(S)

        row = {
            "layer": layer_idx, "layer_type": layer_type,
            "in_features": int(A.shape[1]), "out_features": int(B.shape[0]),
            "r_nominal": r_nominal, "r_effective": r_eff,
            "protected_k": k, "selection_mode": mode,
            "singular_values": [float(x) for x in S.tolist()],
            "cumulative_energy": [float(x) for x in energies],
            "energy_captured_by_k": float(energies[k - 1]) if k > 0 else 0.0,
            "reconstruction_error": recon,
            "orthogonality_error": ortho,
            "delta_s_fro_norm": lowrank_fro_norm(B, A, scaling_s),
        }

        # dense 교차검증 (옵션, 처음 N 개만)
        if n_dense_checked < args.verify_dense_layers:
            dense = scaling_s * (B.to(torch.float32) @ A.to(torch.float32))
            approx = (P * S) @ Q.transpose(0, 1)
            dense_err = float(torch.linalg.matrix_norm(dense - approx) /
                              (torch.linalg.matrix_norm(dense) + 1e-12))
            row["reconstruction_error_dense_check"] = dense_err
            logger.info(f"  [dense check] layer {layer_idx} {layer_type}: "
                        f"lowrank={recon:.3e} dense={dense_err:.3e}")
            del dense, approx
            n_dense_checked += 1

        save_subspace(args.out_dir, key, {
            "Q_S": Q_S.detach().to(torch.float32).cpu(),
            "singular_values": S.detach().cpu(),
            "r_nominal": r_nominal, "r_effective": r_eff, "protected_k": k,
            "selection_mode": mode, "scaling_s": scaling_s,
            "reconstruction_error": recon, "orthogonality_error": ortho,
            "in_features": int(A.shape[1]), "out_features": int(B.shape[0]),
        })
        rows.append(row)

        logger.info(f"layer {layer_idx:2d} {layer_type:9s} | r={r_nominal} r_eff={r_eff} "
                    f"k={k} ({mode}) energy={row['energy_captured_by_k']:.4f} "
                    f"recon={recon:.2e} ortho={ortho:.2e}")

    if not rows:
        raise RuntimeError("생성된 subspace artifact 가 없습니다.")

    max_recon = max(r["reconstruction_error"] for r in rows)
    max_ortho = max(r["orthogonality_error"] for r in rows)
    ks = [r["protected_k"] for r in rows]
    report = {
        "num_modules": len(rows),
        "protected_k_min": min(ks), "protected_k_max": max(ks),
        "protected_k_mean": sum(ks) / len(ks),
        "max_reconstruction_error": max_recon,
        "max_orthogonality_error": max_ortho,
        "layers": rows,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    metadata = {
        "safety_adapter_path": args.safety_adapter_path,
        "target_modules": target_projs, "target_layers": args.target_layers,
        "r_s": r_s, "lora_alpha_s": alpha_s, "scaling_s": scaling_s,
        "use_rslora": use_rslora,
        "adapter_subspace_top_k": args.adapter_subspace_top_k,
        "adapter_subspace_energy": args.adapter_subspace_energy,
        "svd_rank_tol": args.svd_rank_tol, "svd_dtype": args.svd_dtype,
        "selection_mode": rows[0]["selection_mode"],
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"✓ {len(rows)} modules → {args.out_dir}")
    logger.info(f"  protected_k: min={min(ks)} max={max(ks)} mean={sum(ks)/len(ks):.2f}")
    logger.info(f"  max reconstruction error : {max_recon:.3e}")
    logger.info(f"  max orthogonality error  : {max_ortho:.3e}")
    if max_recon > 1e-4:
        logger.warning("reconstruction error 가 큽니다 — svd_rank_tol 을 낮춰보세요.")
    if max_ortho > 1e-4:
        logger.warning("orthogonality error 가 큽니다 — Q_S 가 직교하지 않습니다.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
