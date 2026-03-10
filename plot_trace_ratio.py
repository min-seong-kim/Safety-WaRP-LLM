"""
Plot Gram matrix trace ratio: Attention (q,k,v,o) vs MLP (gate,up,down)
per layer, from a phase1 log file.

python plot_trace_ratio.py logs/phase1_20260303_165945.log -o figures/trace_ratio.png
"""

import re
import sys
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Parse log
# --------------------------------------------------------------------------- #
LAYER_HEADER  = re.compile(r"Layer\s+(\d+)\s+\((\w+)\):")
TRACE_LINE    = re.compile(r"Gram matrix trace:\s*([\d.]+)")

ATTN_TYPES = {"attn_q", "attn_k", "attn_v", "attn_o"}
MLP_TYPES  = {"ffn_gate", "ffn_up", "ffn_down"}


def parse_log(log_path: str) -> dict[int, dict[str, float]]:
    """
    Returns {layer_idx: {layer_type: trace_value, ...}, ...}
    """
    data: dict[int, dict[str, float]] = defaultdict(dict)

    current_layer: int | None = None
    current_type:  str | None = None

    with open(log_path, "r") as f:
        for line in f:
            # Match "Layer N (type):"
            m = LAYER_HEADER.search(line)
            if m:
                current_layer = int(m.group(1))
                current_type  = m.group(2)
                continue

            # Match "Gram matrix trace: VALUE"
            if current_layer is not None and current_type is not None:
                m = TRACE_LINE.search(line)
                if m:
                    trace = float(m.group(1))
                    data[current_layer][current_type] = trace
                    current_layer = None
                    current_type  = None

    return data


# --------------------------------------------------------------------------- #
# 2. Aggregate per layer
# --------------------------------------------------------------------------- #
def aggregate(data: dict[int, dict[str, float]]):
    layers = sorted(data.keys())
    attn_traces = []
    mlp_traces  = []

    for l in layers:
        layer_data = data[l]
        attn_sum = sum(layer_data.get(t, 0.0) for t in ATTN_TYPES)
        mlp_sum  = sum(layer_data.get(t, 0.0) for t in MLP_TYPES)
        attn_traces.append(attn_sum)
        mlp_traces.append(mlp_sum)

    return layers, np.array(attn_traces), np.array(mlp_traces)


# --------------------------------------------------------------------------- #
# 3. Plot
# --------------------------------------------------------------------------- #
def plot(layers, attn_traces, mlp_traces, output_path: str):
    total = attn_traces + mlp_traces
    # Avoid division by zero
    safe_total = np.where(total == 0, 1.0, total)

    attn_ratio = attn_traces / safe_total
    mlp_ratio  = mlp_traces  / safe_total

    n = len(layers)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n * 0.55), 10))

    # ---- (a) Stacked proportion bar chart --------------------------------- #
    ax = axes[0]
    ax.bar(x, attn_ratio, label="Attention (q+k+v+o)", color="#4C72B0")
    ax.bar(x, mlp_ratio,  bottom=attn_ratio,
           label="MLP (gate+up+down)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Gram Matrix Trace Ratio: Attention vs MLP (stacked)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotation: attn ratio on each bar
    for i in range(n):
        if attn_ratio[i] > 0.05:
            ax.text(x[i], attn_ratio[i] / 2, f"{attn_ratio[i]:.2f}",
                    ha="center", va="center", fontsize=6, color="white")
        if mlp_ratio[i] > 0.05:
            ax.text(x[i], attn_ratio[i] + mlp_ratio[i] / 2,
                    f"{mlp_ratio[i]:.2f}",
                    ha="center", va="center", fontsize=6, color="white")

    # ---- (b) Absolute trace (log scale) ----------------------------------- #
    ax2 = axes[1]
    width = 0.35
    ax2.bar(x - width / 2, attn_traces, width,
            label="Attention (q+k+v+o)", color="#4C72B0")
    ax2.bar(x + width / 2, mlp_traces,  width,
            label="MLP (gate+up+down)", color="#DD8452")

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Trace (log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Gram Matrix Trace: Attention vs MLP (absolute, log scale)")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[✓] Saved plot to: {output_path}")

    # Print summary table
    print(f"\n{'Layer':>6} | {'Attn trace':>18} | {'MLP trace':>18} | "
          f"{'Attn %':>8} | {'MLP %':>8}")
    print("-" * 70)
    for i, l in enumerate(layers):
        print(f"{l:>6} | {attn_traces[i]:>18.2f} | {mlp_traces[i]:>18.2f} | "
              f"{attn_ratio[i]*100:>7.2f}% | {mlp_ratio[i]*100:>7.2f}%")

    total_attn = attn_traces.sum()
    total_mlp  = mlp_traces.sum()
    grand      = total_attn + total_mlp
    print("-" * 70)
    print(f"{'TOTAL':>6} | {total_attn:>18.2f} | {total_mlp:>18.2f} | "
          f"{total_attn/grand*100:>7.2f}% | {total_mlp/grand*100:>7.2f}%")


def plot_total_summary(attn_traces, mlp_traces, output_path: str):
    """전체 레이어(0~N)를 합산하여 Attention vs MLP 비율을 단일 막대 그래프로 표시."""
    import os

    total_attn = float(attn_traces.sum())
    total_mlp  = float(mlp_traces.sum())
    grand      = total_attn + total_mlp

    attn_ratio = total_attn / grand
    mlp_ratio  = total_mlp  / grand

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ---- (a) Stacked single bar ------------------------------------------- #
    ax = axes[0]
    ax.bar([0], [attn_ratio], color="#4C72B0", label=f"Attention  {attn_ratio*100:.1f}%")
    ax.bar([0], [mlp_ratio],  bottom=[attn_ratio],
           color="#DD8452", label=f"MLP  {mlp_ratio*100:.1f}%")
    ax.text(0, attn_ratio / 2, f"{attn_ratio*100:.1f}%",
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    ax.text(0, attn_ratio + mlp_ratio / 2, f"{mlp_ratio*100:.1f}%",
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")

    ax.set_xticks([0])
    ax.set_xticklabels(["All layers"])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Total Trace Ratio\nAttention vs MLP")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- (b) Grouped absolute bar ----------------------------------------- #
    ax2 = axes[1]
    bars = ax2.bar(["Attention\n(q+k+v+o)", "MLP\n(gate+up+down)"],
                   [total_attn, total_mlp],
                   color=["#4C72B0", "#DD8452"], width=0.4)
    for bar, val in zip(bars, [total_attn, total_mlp]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.02,
                 f"{val:.3e}",
                 ha="center", va="bottom", fontsize=11)

    ax2.set_ylabel("Total Trace")
    ax2.set_yscale("log")
    ax2.set_title("Total Trace (absolute, log scale)\nAll layers summed")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("All-Layer Summary: Attention vs MLP", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Saved total summary plot to: {output_path}")
    print(f"    Attention total: {total_attn:.3e}  ({attn_ratio*100:.2f}%)")
    print(f"    MLP total:       {total_mlp:.3e}  ({mlp_ratio*100:.2f}%)")


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Plot Gram matrix trace ratio (Attention vs MLP) from phase1 log")
    parser.add_argument(
        "log_path",
        nargs="?",
        default="./logs/phase1_20260303_165945.log",
        help="Path to phase1 log file"
    )
    parser.add_argument(
        "-o", "--output",
        default="./figures/trace_ratio.png",
        help="Output PNG path (per-layer)"
    )
    parser.add_argument(
        "--output-total",
        default="./figures/trace_ratio_total.png",
        help="Output PNG path (all-layer summary)"
    )
    args = parser.parse_args()

    print(f"[*] Parsing: {args.log_path}")
    data = parse_log(args.log_path)
    print(f"[*] Found {len(data)} layers")

    layers, attn_traces, mlp_traces = aggregate(data)
    plot(layers, attn_traces, mlp_traces, args.output)
    plot_total_summary(attn_traces, mlp_traces, args.output_total)


if __name__ == "__main__":
    main()
