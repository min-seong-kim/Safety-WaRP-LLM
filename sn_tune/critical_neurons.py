"""
critical_neurons.py — Critical(robust) safety neuron = Safety \\ Utility.

RSN-Tune 의 "R" 이 여기서 나온다.

    N_safe        safety 코퍼스(circuit_breakers)로 검출한 뉴런
    N_foundation  일반 코퍼스(wikipedia)로 검출한 뉴런
    N_robust      = N_safe - (N_safe ∩ N_foundation)

SN-Tune 은 `N_safe` 를, RSN-Tune 은 `N_robust` 를 쓴다. downstream 능력에도 쓰이는
뉴런을 빼두면 safety 를 건드리면서 태스크 성능을 덜 깎는다는 것이 논문의 주장이다.

**원공간·WaRP 공간 양쪽에 그대로 쓴다.** 두 검출기가 같은 5줄 포맷을 쓰기 때문이다.
다만 같은 공간에서 뽑은 safety/utility 파일끼리만 빼야 의미가 있다 — 원공간 safety 에서
WaRP 공간 utility 를 빼면 인덱스의 의미가 달라 숫자만 줄어들 뿐 아무 뜻이 없다.
`--space` 로 표시해두면 메타 파일에 남는다.

사용
    python -m sn_tune.critical_neurons \\
        --safety_file  ./sn_tune/output_neurons/safety_....txt \\
        --utility_file ./sn_tune/output_neurons/utility_....txt \\
        --output_file  ./sn_tune/output_neurons/critical_....txt \\
        --model_name   kmseong/llama2_7b-chat-Safety-FT-lr5e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Set

from .neuron_file import (
    MODULE_ORDER,
    assert_nonempty,
    count_neurons,
    load_neuron_file,
    save_neuron_file,
    summarize,
)

logger = logging.getLogger(__name__)


def compute_critical(
    safety: Dict[str, Dict[int, Set[int]]],
    utility: Dict[str, Dict[int, Set[int]]],
) -> Dict[str, Dict[int, Set[int]]]:
    """Critical = Safety - (Safety ∩ Utility), 레이어·모듈별로."""
    critical: Dict[str, Dict[int, Set[int]]] = {}
    for module in MODULE_ORDER:
        safety_mod = safety.get(module, {})
        utility_mod = utility.get(module, {})
        critical[module] = {
            layer: idx - utility_mod.get(layer, set())
            for layer, idx in safety_mod.items()
        }
    return critical


def overlap_stats(
    safety: Dict[str, Dict[int, Set[int]]],
    utility: Dict[str, Dict[int, Set[int]]],
    critical: Dict[str, Dict[int, Set[int]]],
) -> Dict[str, Dict[str, int]]:
    """모듈별 safety / utility / overlap / critical 개수."""
    stats: Dict[str, Dict[str, int]] = {}
    for module in MODULE_ORDER:
        s_mod, u_mod, c_mod = safety.get(module, {}), utility.get(module, {}), critical.get(module, {})
        layers = set(s_mod) | set(u_mod)
        stats[module] = {
            "safety": sum(len(v) for v in s_mod.values()),
            "utility": sum(len(v) for v in u_mod.values()),
            "overlap": sum(len(s_mod.get(l, set()) & u_mod.get(l, set())) for l in layers),
            "critical": sum(len(v) for v in c_mod.values()),
        }
    return stats


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety_file", required=True)
    ap.add_argument("--utility_file", required=True)
    ap.add_argument("--output_file", default=None,
                    help="기본값: safety 파일과 같은 디렉토리의 critical_safety_neuron_<ts>.txt")
    ap.add_argument("--model_name", default=None,
                    help="주면 결과 뉴런/파라미터 비율까지 찍는다")
    ap.add_argument("--space", choices=["original", "warp"], default="original",
                    help="두 입력이 어느 좌표계에서 나왔는지 (메타 기록용)")
    args = ap.parse_args(argv)

    safety = load_neuron_file(args.safety_file)
    utility = load_neuron_file(args.utility_file)

    # 검출이 조용히 빈 파일을 뱉는 사고가 잦다. 여기서 바로 끊는다.
    assert_nonempty(args.safety_file, safety)
    assert_nonempty(args.utility_file, utility)

    critical = compute_critical(safety, utility)

    out_file = args.output_file
    if out_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(os.path.dirname(os.path.abspath(args.safety_file)),
                                f"critical_safety_neuron_{ts}.txt")
    save_neuron_file(out_file, critical)

    stats = overlap_stats(safety, utility, critical)
    c_all, s_all, u_all = count_neurons(critical), count_neurons(safety), count_neurons(utility)

    logger.info("모듈별 (safety / utility / overlap / critical)")
    for module in MODULE_ORDER:
        s = stats[module]
        logger.info("  %-9s %7d / %7d / %7d / %7d", module,
                    s["safety"], s["utility"], s["overlap"], s["critical"])
    logger.info("  %-9s %7d / %7d / %7s / %7d", "TOTAL",
                s_all["total"], u_all["total"], "-", c_all["total"])

    if s_all["total"]:
        kept = 100.0 * c_all["total"] / s_all["total"]
        logger.info("safety 중 critical 로 남은 비율: %.1f%%", kept)
        if c_all["total"] == 0:
            logger.error("critical 이 0개다 — utility 가 safety 를 전부 덮었다. "
                         "utility 검출의 top-k 를 낮춰라.")
            return 1

    meta = {
        "safety_file": os.path.abspath(args.safety_file),
        "utility_file": os.path.abspath(args.utility_file),
        "output_file": os.path.abspath(out_file),
        "space": args.space,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {"safety": s_all, "utility": u_all, "critical": c_all},
        "per_module": stats,
    }
    if args.model_name:
        meta["summary"] = summarize(out_file, args.model_name)
        logger.info("critical 파라미터 비율: %.4f%%", meta["summary"]["param_pct"])

    with open(out_file + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("저장: %s", out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
