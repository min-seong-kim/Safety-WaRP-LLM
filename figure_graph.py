import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODEL_DISPLAY_NAMES = {
    "Llama 2 7B Chat": "Llama-2-7B-Chat",
    "Llama 2 13B Chat": "Llama-2-13B-Chat",
    "Llama 3.2 3B Instruct": "Llama-3.2-3B-Instruct",
    "Llama 3.1 8B Instruct": "Llama-3.1-8B-Instruct",
    "Qwen 2.5 7B Instruct": "Qwen-2.5-7B-Instruct",
    "Gemma 2 9B IT": "Gemma-2-9B-IT",
}

METHOD_STYLES = {
    "Before Downstream Tuning": dict(marker="X", facecolor="#7A7A7A", edgecolor="#333333", size=150),
    "Full Params FT": dict(marker="o", facecolor="#4E79A7", edgecolor="#333333", size=140),
    "SafeInstr": dict(marker="s", facecolor="#F28E2B", edgecolor="#333333", size=140),
    "Resta": dict(marker="P", facecolor="#76B7B2", edgecolor="#333333", size=140),
    "SafeDelta": dict(marker="D", facecolor="#B07AA1", edgecolor="#333333", size=140),
    "SN-Tune": dict(marker="v", facecolor="#E15759", edgecolor="#333333", size=140),
    "RSN-Tune": dict(marker="^", facecolor="#FF8C7A", edgecolor="#333333", size=140),
    "WSR-Tune (Ours)": dict(marker="*", facecolor="#D4A017", edgecolor="#333333", size=300),
}

LEGEND_ROWS = [
    ["Before Downstream Tuning", "Full Params FT", "SafeInstr", "Resta", "SafeDelta", "SN-Tune", "RSN-Tune", "WSR-Tune (Ours)"],
]


def clean_latex_cell(cell):
    cell = cell.strip()
    previous = None
    while previous != cell:
        previous = cell
        cell = re.sub(r"\\(?:textbf|uline|emph)\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"\$([^$]*)\$", r"\1", cell)
    cell = cell.replace(r"\\", "")
    cell = cell.replace(r"\%", "%")
    cell = cell.replace("{", "").replace("}", "")
    cell = re.sub(r"\\[A-Za-z]+(?:\([^)]*\))?(?:\[[^\]]*\])?", "", cell)
    return " ".join(cell.split())


def parse_float(cell):
    match = re.search(r"-?\d+(?:\.\d+)?", clean_latex_cell(cell))
    if not match:
        raise ValueError(f"Could not parse numeric value from cell: {cell!r}")
    return float(match.group(0))


def iter_latex_rows(tabular_text):
    row_parts = []
    for line in tabular_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "&" not in line and re.match(r"\\(?:begin|end|toprule|midrule|bottomrule|cmidrule)", line):
            continue
        row_parts.append(line)
        if line.endswith(r"\\"):
            row = " ".join(row_parts)
            row_parts = []
            yield re.sub(r"\\\\\s*$", "", row).strip()


def parse_tabular(tabular_text):
    model_names = re.findall(
        r"\\multicolumn\{6\}\{c\|?\}\{\\textbf\{([^{}]+)\}\}",
        tabular_text,
    )
    if len(model_names) != 2:
        raise ValueError(f"Expected two model names in a tabular, found {model_names}")

    downstream_metrics = []
    records = {
        model_names[0]: {"metric": None, "points": []},
        model_names[1]: {"metric": None, "points": []},
    }

    for row in iter_latex_rows(tabular_text):
        cells = [clean_latex_cell(cell) for cell in row.split("&")]
        if len(cells) != 13:
            continue

        if cells[0] == "Method":
            downstream_metrics = [cells[6], cells[12]]
            records[model_names[0]]["metric"] = downstream_metrics[0]
            records[model_names[1]]["metric"] = downstream_metrics[1]
            continue

        if not downstream_metrics:
            continue

        method = cells[0]
        if method.startswith("\\") or method in {"toprule", "midrule", "bottomrule"}:
            continue

        for group_index, model_name in enumerate(model_names):
            start = 1 + group_index * 6
            avg_asr = parse_float(cells[start + 4])
            downstream = parse_float(cells[start + 5])
            records[model_name]["points"].append(
                {
                    "method": method,
                    "downstream": downstream,
                    "avg_asr": avg_asr,
                    "defense_success": 100.0 - avg_asr,
                }
            )

    return records


def parse_table_file(tex_path):
    text = Path(tex_path).read_text(encoding="utf-8")
    tabulars = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, flags=re.DOTALL)
    if not tabulars:
        raise ValueError(f"No tabular environments found in {tex_path}")

    all_records = {}
    for tabular in tabulars:
        all_records.update(parse_tabular(tabular))
    return all_records


def slugify(text):
    text = MODEL_DISPLAY_NAMES.get(text, text)
    text = re.sub(r"[^A-Za-z0-9.]+", "_", text).strip("_")
    return text.lower()


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Calibri", "Carlito", "Arial", "DejaVu Sans"],
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_legend_handles(methods):
    handles = []
    for method in methods:
        style = METHOD_STYLES.get(
            method,
            dict(marker="o", facecolor="#D9D9D9", edgecolor="#555555", size=48),
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                label=method,
                linestyle="None",
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["edgecolor"],
                markeredgewidth=0.65,
                markersize=5.8 if method != "WSR-Tune (Ours)" else 8.0,
            )
        )
    return handles


def get_model_order(records):
    return list(records.keys())


def get_legend_rows(records):
    present_methods = {
        point["method"]
        for model_record in records.values()
        for point in model_record["points"]
    }
    rows = [
        [method for method in row if method in present_methods]
        for row in LEGEND_ROWS
    ]
    known_methods = {method for row in LEGEND_ROWS for method in row}
    extras = sorted(present_methods - known_methods)
    if extras:
        rows.append(extras)
    return [row for row in rows if row]


def draw_model_axis(ax, model_name, model_record, show_ylabel=False):
    points = model_record["points"]
    if not points:
        raise ValueError(f"No data rows found for {model_name}")

    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    metric = model_record["metric"]
    x_values = [point["downstream"] for point in points]
    y_values = [point["defense_success"] for point in points]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max(1.0, (x_max - x_min) * 0.12)
    y_pad = max(1.0, (y_max - y_min) * 0.12)

    for point in points:
        method = point["method"]
        style = METHOD_STYLES.get(
            method,
            dict(marker="o", facecolor="#D9D9D9", edgecolor="#555555", size=62),
        )
        ax.scatter(
            point["downstream"],
            point["defense_success"],
            s=style["size"],
            marker=style["marker"],
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=0.65,
            alpha=0.92,
            zorder=3,
        )
        if method == "WSR-Tune (Ours)":
            ax.annotate(
                "(Ours)",
                (point["downstream"], point["defense_success"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=4,
            )

    ax.set_title(display_name, pad=5)
    ax.set_xlabel(f"{metric} Accuracy (%) ↑")
    if show_ylabel:
        ax.set_ylabel("Defense Success Rate (%) ↑")
    ax.grid(True, linestyle=":", linewidth=0.55, color="#D6CEC0", alpha=0.9)
    ax.set_axisbelow(True)

    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), min(101.0, y_max + y_pad))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=2.5, pad=1.8)


def add_shared_legend(fig, records):
    legend_rows = get_legend_rows(records)
    y_positions = [0.075, 0.035, -0.005]

    for index, row in enumerate(legend_rows):
        legend = fig.legend(
            handles=make_legend_handles(row),
            loc="lower center",
            bbox_to_anchor=(0.5, y_positions[index]),
            ncol=min(4, len(row)),
            frameon=False,
            fontsize=8,
            handletextpad=0.35,
            columnspacing=1.1,
            borderaxespad=0.0,
        )
        if index < len(legend_rows) - 1:
            fig.add_artist(legend)


def plot_legend(records, output_dir):
    legend_rows = get_legend_rows(records)
    fig_height = 0.28 + 0.24 * len(legend_rows)
    fig = plt.figure(figsize=(5.4, fig_height))

    if len(legend_rows) == 1:
        y_positions = [0.40]
    elif len(legend_rows) == 2:
        y_positions = [0.58, 0.16]
    else:
        y_positions = [0.70, 0.36, 0.02]

    for index, row in enumerate(legend_rows):
        legend = fig.legend(
            handles=make_legend_handles(row),
            loc="lower center",
            bbox_to_anchor=(0.5, y_positions[index]),
            ncol=min(4, len(row)),
            frameon=False,
            fontsize=20,
            handletextpad=0.35,
            columnspacing=1.1,
            borderaxespad=0.0,
        )
        if index < len(legend_rows) - 1:
            fig.add_artist(legend)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "method_legend"
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", pad_inches=0.01, facecolor="white")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)
    return output_base


def plot_combined(records, output_dir):
    model_order = get_model_order(records)
    fig_width = 2.7 * len(model_order)
    fig, axes = plt.subplots(1, len(model_order), figsize=(fig_width, 2.65))
    if len(model_order) == 1:
        axes = [axes]

    for index, model_name in enumerate(model_order):
        draw_model_axis(axes[index], model_name, records[model_name], show_ylabel=index == 0)

    add_shared_legend(fig, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "combined_tradeoff"
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.1)
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_base


def plot_model(model_name, model_record, output_dir):
    fig, ax = plt.subplots(figsize=(4.9, 3.7))
    draw_model_axis(ax, model_name, model_record, show_ylabel=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / f"{slugify(model_name)}_tradeoff"
    fig.tight_layout()
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_base


def main():
    parser = argparse.ArgumentParser(
        description="Plot independent downstream-vs-defense trade-off figures from a LaTeX table."
    )
    parser.add_argument("tex_file", help="Path to the LaTeX table file, e.g., file.tex")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated PNG/PDF figures. Defaults to the current directory.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one combined row figure in addition to individual column figures.",
    )
    args = parser.parse_args()

    set_plot_style()
    records = parse_table_file(args.tex_file)
    output_dir = Path(args.output_dir)

    for model_name in get_model_order(records):
        output_base = plot_model(model_name, records[model_name], output_dir)
        print(f"Wrote {output_base}.png, {output_base}.pdf, and {output_base}.svg")

    output_base = plot_legend(records, output_dir)
    print(f"Wrote {output_base}.png, {output_base}.pdf, and {output_base}.svg")

    if args.combined:
        output_base = plot_combined(records, output_dir)
        print(f"Wrote {output_base}.png, {output_base}.pdf, and {output_base}.svg")


if __name__ == "__main__":
    main()
