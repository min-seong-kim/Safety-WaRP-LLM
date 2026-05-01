"""
Phase 1 로그에서 Top-10% rank variance ratio를 파싱하여 모듈별 그래프를 생성

Usage:
    python plot_top10pct_variance.py --log_file ./logs/phase1_XXXXXX.log
    python plot_top10pct_variance.py --log_file ./logs/phase1_XXXXXX.log --output ./figures/top10pct.png
    python plot_top10pct_variance.py --log_file log1.log log2.log --labels "prompt+resp" "prompt-only"


python plot_top10pct_variance.py --log_file ./logs/phase1_20260429_202424.log --output ./figures/variance.png

"""

import re
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MODULE_ORDER = ["attn_q", "attn_k", "attn_v", "ffn_down", "ffn_up"]
MODULE_COLORS = {
    "attn_q":  "#4C72B0",
    "attn_k":  "#55A868",
    "attn_v":  "#C44E52",
    "ffn_down":"#DD8452",
    "ffn_up":  "#8172B2",
}
MODULE_LABELS = {
    "attn_q":  "Attn Q",
    "attn_k":  "Attn K",
    "attn_v":  "Attn V",
    "ffn_down":"FFN Down",
    "ffn_up":  "FFN Up",
}

# 로그 파싱 패턴
# "Layer 3 (ffn_down):"  → layer_idx, module
LAYER_RE  = re.compile(r"Layer\s+(\d+)\s+\((\w+)\):")
# "Top-10% rank variance ratio: 67.97% (top 409 / 4096 dims)"
RATIO_RE  = re.compile(r"Top-10%\s+rank\s+variance\s+ratio:\s+([\d.]+)%")


def parse_log(log_path: str) -> dict[str, dict[int, float]]:
    """
    Returns:
        data[module][layer_idx] = top10pct_variance_ratio (float)
    """
    data = defaultdict(dict)

    current_layer = None
    current_module = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = LAYER_RE.search(line)
            if m:
                current_layer = int(m.group(1))
                current_module = m.group(2)
                continue

            m = RATIO_RE.search(line)
            if m and current_layer is not None and current_module is not None:
                ratio = float(m.group(1))
                data[current_module][current_layer] = ratio
                current_layer = None
                current_module = None

    return dict(data)


def plot_single(data: dict, title: str, output_path: str):
    """단일 로그 파일 → 5-panel 서브플롯"""
    modules = [m for m in MODULE_ORDER if m in data]
    if not modules:
        print("No data to plot.")
        return

    # 레이어 수
    all_layers = sorted({
        layer
        for m in modules
        for layer in data[m]
    })
    n_layers = max(all_layers) + 1

    fig, axes = plt.subplots(1, len(modules), figsize=(4 * len(modules), 4), sharey=True)
    if len(modules) == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax, module in zip(axes, modules):
        layer_data = data[module]
        xs = sorted(layer_data.keys())
        ys = [layer_data[x] for x in xs]

        color = MODULE_COLORS.get(module, "gray")
        ax.plot(ys, xs, color=color, linewidth=1.8, marker="o", markersize=3)
        ax.fill_betweenx(xs, ys, alpha=0.15, color=color)

        ax.set_title(MODULE_LABELS.get(module, module), fontsize=11)
        ax.set_xlabel("Top-10% rank\nvariance ratio (%)", fontsize=9)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.axvline(x=90, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="90%")
        ax.invert_yaxis()

    axes[0].set_ylabel("Layer index", fontsize=10)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(4))

    # 범례 (90% 기준선)
    axes[-1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_overlay(data_list: list[dict], labels: list[str], output_path: str):
    """여러 로그 파일 비교 overlay → 모듈별 서브플롯"""
    all_modules = []
    for module in MODULE_ORDER:
        if any(module in d for d in data_list):
            all_modules.append(module)

    n_mods = len(all_modules)
    if n_mods == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, n_mods, figsize=(4 * n_mods, 4), sharey=True)
    if n_mods == 1:
        axes = [axes]

    fig.suptitle("Top-10% rank variance ratio comparison", fontsize=13, fontweight="bold", y=1.02)

    linestyles = ["-", "--", "-.", ":"]
    cmap = plt.get_cmap("tab10")

    for ax, module in zip(axes, all_modules):
        for i, (data, label) in enumerate(zip(data_list, labels)):
            if module not in data:
                continue
            layer_data = data[module]
            xs = sorted(layer_data.keys())
            ys = [layer_data[x] for x in xs]
            ax.plot(ys, xs,
                    color=cmap(i),
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label=label)

        ax.set_title(MODULE_LABELS.get(module, module), fontsize=11)
        ax.set_xlabel("Top-10% rank\nvariance ratio (%)", fontsize=9)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.axvline(x=90, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.invert_yaxis()

    axes[0].set_ylabel("Layer index", fontsize=10)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(4))
    axes[0].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(data: dict, label: str):
    print(f"\n{'='*60}")
    print(f"Summary: {label}")
    print(f"{'='*60}")
    for module in MODULE_ORDER:
        if module not in data:
            continue
        vals = [v for v in data[module].values()]
        print(f"  {MODULE_LABELS[module]:12s}: "
              f"mean={np.mean(vals):.1f}%  "
              f"min={np.min(vals):.1f}%  "
              f"max={np.max(vals):.1f}%  "
              f"layers={len(vals)}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 로그에서 Top-10% rank variance ratio 그래프 생성"
    )
    parser.add_argument(
        "--log_file", nargs="+", required=True,
        help="Phase 1 로그 파일 경로 (복수 지정 시 overlay 비교)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="각 로그 파일의 레이블 (overlay 비교 시 사용)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="출력 이미지 경로 (기본: 로그 파일명 기반 자동 생성)"
    )
    args = parser.parse_args()

    # 레이블 기본값
    labels = args.labels if args.labels else [Path(f).stem for f in args.log_file]
    if len(labels) < len(args.log_file):
        labels += [Path(f).stem for f in args.log_file[len(labels):]]

    # 파싱
    data_list = []
    for log_file in args.log_file:
        print(f"Parsing: {log_file}")
        d = parse_log(log_file)
        data_list.append(d)
        print_summary(d, Path(log_file).stem)

    # 출력 경로
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.log_file[0]).stem
        output_path = str(Path(args.log_file[0]).parent / f"{stem}_top10pct_variance.png")

    # 플롯
    if len(data_list) == 1:
        plot_single(data_list[0], title=f"Top-10% rank variance ratio\n({labels[0]})", output_path=output_path)
    else:
        plot_overlay(data_list, labels, output_path=output_path)


if __name__ == "__main__":
    main()
