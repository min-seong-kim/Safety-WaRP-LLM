#!/usr/bin/env python
"""Stage W3 — column score + WSR basis → 보호 subspace U_S.

명세 `adapter_aware_column_wsr_lora_implementation.md` §8, §14.

Stage W2 가 만든 column score 는 작으므로(모듈당 n floats), 이 단계는 **수 초**에 끝난다.
따라서 여러 (importance_mode × selection rule) 조합을 마음껏 만들어 두고, 비싼 downstream
학습은 그중 고른 것만 돌리면 된다 — `adapter_subspace_lora` 의 Stage 1 과 같은 설계.

선택 규칙 우선순위 (§8 말미): top_k → keep_ratio → score_energy.
셋 다 없으면 오류. 조용한 기본값을 두지 않는다.

`--random_control` 은 §17.6 통제군: 같은 U 에서 **동일한 k** 를 무작위로 고른다.
개선이 safety-aware 선택 덕인지 단순 capacity 감소 덕인지 분리한다.
"""

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.adapter_wsr_column import (  # noqa: E402
    VALID_IMPORTANCE_MODES, load_scores, save_subspace, score_summary,
    select_directions, select_random_directions, spearman,
    subspace_orthogonality_error,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_adapter_wsr_subspace")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--column_scores_dir", required=True, help="Stage W2 산출물")
    ap.add_argument("--wsr_basis_dir", required=True, help="Phase 1 basis (U_l)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer_type", default="attn_q,attn_k,attn_v,ffn_up,ffn_down")

    ap.add_argument("--importance_mode", default="adapter_taylor",
                    choices=list(VALID_IMPORTANCE_MODES))
    ap.add_argument("--direction_top_k", type=int, default=None)
    ap.add_argument("--direction_keep_ratio", type=float, default=None)
    ap.add_argument("--direction_score_energy", type=float, default=None)

    ap.add_argument("--random_control", action="store_true",
                    help="§17.6: 동일 k 를 무작위 선택 (safety-aware 선택의 통제군)")
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--match_k_from", default=None,
                    help="random_control 의 k 를 이 subspace dir 의 k 와 층별로 맞춘다")

    ap.add_argument("--basis_dtype", default="float32", choices=["float32", "float64"])
    return ap.parse_args()


def load_basis(basis_dir, layer_types):
    out = {}
    for lt in layer_types:
        d = os.path.join(basis_dir, lt)
        if not os.path.isdir(d):
            logger.warning(f"basis layer_type dir 없음: {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith("layer_") and fname.endswith("_svd.pt"):
                li = int(fname.split("_")[1])
                sd = torch.load(os.path.join(d, fname), map_location="cpu")
                out[(li, lt)] = {"U": sd["U"], "S": sd.get("S")}
    return out


def main():
    args = parse_args()
    layer_types = [x.strip() for x in args.layer_type.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    n_rules = sum(x is not None for x in
                  (args.direction_top_k, args.direction_keep_ratio, args.direction_score_energy))
    if n_rules == 0:
        raise ValueError("선택 규칙이 없습니다: --direction_top_k / --direction_keep_ratio / "
                         "--direction_score_energy 중 하나를 지정하세요.")
    if n_rules > 1:
        logger.warning(f"선택 규칙이 {n_rules}개 지정됨 → 우선순위 "
                       f"top_k > keep_ratio > score_energy 로 해석합니다.")

    scores_all = load_scores(args.column_scores_dir, layer_types)
    basis = load_basis(args.wsr_basis_dir, layer_types)
    logger.info(f"scores={len(scores_all)} modules | basis={len(basis)} modules")

    match_k = {}
    if args.match_k_from:
        from models.adapter_wsr_column import load_subspaces
        for k_, p_ in load_subspaces(args.match_k_from, layer_types).items():
            match_k[k_] = int(p_["k"])
        logger.info(f"random_control k 를 {args.match_k_from} 에 맞춤 ({len(match_k)} modules)")

    basis_dtype = {"float32": torch.float32, "float64": torch.float64}[args.basis_dtype]
    gen = torch.Generator().manual_seed(args.random_seed)

    rows = []
    for key in sorted(set(scores_all) & set(basis)):
        li, lt = key
        payload = scores_all[key]
        if args.importance_mode not in payload["scores"]:
            logger.warning(f"layer {li} {lt}: mode={args.importance_mode} score 없음 → skip")
            continue
        c = payload["scores"][args.importance_mode].to(torch.float32)
        U = basis[key]["U"].to(basis_dtype)
        n = int(U.shape[1])
        if int(c.numel()) != n:
            raise ValueError(f"layer {li} {lt}: score 길이 {c.numel()} != basis n {n}. "
                             f"score 와 basis 가 다른 실행에서 나온 것 아닌지 확인하세요.")

        if args.random_control:
            k = match_k.get(key)
            if k is None:
                idx_ref, k, sel_mode, sel_val = select_directions(
                    c, args.direction_top_k, args.direction_keep_ratio,
                    args.direction_score_energy)
            else:
                sel_mode, sel_val = "match_k", float(k)
            idx = select_random_directions(n, k, gen)
        else:
            idx, k, sel_mode, sel_val = select_directions(
                c, args.direction_top_k, args.direction_keep_ratio,
                args.direction_score_energy)

        U_S = U[:, idx].contiguous()
        ortho = subspace_orthogonality_error(U_S)
        summ = score_summary(c, idx)

        # 선택이 단순히 top principal activation direction 으로 퇴화했는지 진단 (§15.6)
        act_rank_corr = None
        S = basis[key].get("S")
        if S is not None and S.numel() == n:
            # activation singular value 는 내림차순 → 순위가 높을수록 주성분
            act_rank_corr = spearman(c, S.to(torch.float32))

        save_subspace(args.out_dir, key, {
            "module_key": key,
            "U_S": U_S.to(torch.float32),
            "selected_indices": idx.cpu(),
            "column_scores": c.cpu(),
            "k": k, "n": n,
            "importance_mode": args.importance_mode,
            "column_aggregation": payload.get("column_aggregation"),
            "selection_mode": sel_mode, "selection_value": sel_val,
            "random_control": bool(args.random_control),
            "random_seed": args.random_seed if args.random_control else None,
            "subspace_orthogonality_error": ortho,
            "orthogonality_error": ortho,     # verify() 호환 키
            "safety_adapter_scaling": payload.get("safety_adapter_scaling"),
        })

        row = {"layer": li, "layer_type": lt, "k": k, "n": n,
               "selection_mode": sel_mode, "selection_value": sel_val,
               "subspace_orthogonality_error": ortho,
               "activation_rank_spearman": act_rank_corr}
        row.update(summ)
        rows.append(row)

    if not rows:
        raise RuntimeError("subspace 를 하나도 만들지 못했습니다. "
                           "column_scores_dir / wsr_basis_dir / layer_type 을 확인하세요.")

    ks = [r["k"] for r in rows]

    def _mean(f):
        v = [r[f] for r in rows if r.get(f) is not None]
        return sum(v) / len(v) if v else None

    report = {
        "num_modules": len(rows),
        "importance_mode": args.importance_mode,
        "random_control": bool(args.random_control),
        "random_seed": args.random_seed if args.random_control else None,
        "selection": {"top_k": args.direction_top_k,
                      "keep_ratio": args.direction_keep_ratio,
                      "score_energy": args.direction_score_energy,
                      "match_k_from": args.match_k_from},
        "k_min": min(ks), "k_max": max(ks), "k_mean": sum(ks) / len(ks),
        "max_subspace_orthogonality_error": max(r["subspace_orthogonality_error"] for r in rows),
        "mean_topk_score_mass": _mean("topk_score_mass"),
        "mean_activation_rank_spearman": _mean("activation_rank_spearman"),
        "column_scores_dir": args.column_scores_dir,
        "wsr_basis_dir": args.wsr_basis_dir,
        "layers": rows,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"✓ {len(rows)} modules → {args.out_dir}")
    logger.info(f"  mode={args.importance_mode} random={args.random_control}")
    logger.info(f"  k: min={report['k_min']} mean={report['k_mean']:.1f} max={report['k_max']}")
    logger.info(f"  top-k score mass (mean) : {report['mean_topk_score_mass']}")
    logger.info(f"  activation-rank Spearman: {report['mean_activation_rank_spearman']}")
    logger.info(f"  max ‖U_SᵀU_S−I‖_F       : {report['max_subspace_orthogonality_error']:.3e}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
