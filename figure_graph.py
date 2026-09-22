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
    "SEAL": dict(marker="<", facecolor="#59A14F", edgecolor="#333333", size=140),
    "AsFT": dict(marker=">", facecolor="#9C755F", edgecolor="#333333", size=140),
    "Lisa": dict(marker="p", facecolor="#7E4EA9", edgecolor="#333333", size=140),
    "SafeDelta": dict(marker="D", facecolor="#B07AA1", edgecolor="#333333", size=140),
    "SN-Tune": dict(marker="v", facecolor="#E15759", edgecolor="#333333", size=140),
    "RSN-Tune": dict(marker="^", facecolor="#FF8C7A", edgecolor="#333333", size=140),
    "WSR-Tune (Ours)": dict(marker="*", facecolor="#D4A017", edgecolor="#333333", size=300),
}

# 범례를 몇 줄로 펼칠지와 마커 확대 배율. ncol 은 기법 수에서 역산한다(기법이 늘어도 줄 수 유지).
LEGEND_TARGET_ROWS = 2
LEGEND_MARKER_SCALE = 2.0

# tem_csv.txt 같은 TSV 표의 기법 표기를 METHOD_STYLES 키로 옮긴다.
METHOD_ALIASES = {
    "SSFT": "Before Downstream Tuning",
    "SSFT+GSM": "Full Params FT",
    "SSFT+DT": "Full Params FT",
    "SN": "SN-Tune",
    "RSN": "RSN-Tune",
    "WSR-Tune": "WSR-Tune (Ours)",
}

# TSV 블록의 헤더 행은 이 다섯 칼럼으로 알아본다. 바로 뒤 칼럼이 downstream 과제 이름.
TASK_TSV_HEADER = ["Direct", "AutoDAN", "PAIR", "PAP", "AVG"]

LEGEND_ROWS = [
    [
        "Before Downstream Tuning",
        "Full Params FT",
        "SafeInstr",
        "Resta",
        "SEAL",
        "AsFT",
        "Lisa",
        "SafeDelta",
        "SN-Tune",
        "RSN-Tune",
        "WSR-Tune (Ours)",
    ],
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


def strip_latex_comment(line):
    """줄 끝 LaTeX 주석(%)을 제거한다. 이스케이프된 \\% 는 보존."""
    out = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            out.append(line[index : index + 2])
            index += 2
            continue
        if char == "%":
            break
        out.append(char)
        index += 1
    return "".join(out)


def iter_latex_rows(tabular_text):
    row_parts = []
    for line in tabular_text.splitlines():
        line = strip_latex_comment(line).strip()
        if not line:
            continue
        if "&" not in line and re.match(r"\\(?:begin|end|toprule|midrule|bottomrule|cmidrule)", line):
            continue
        row_parts.append(line)
        if line.endswith(r"\\"):
            row = " ".join(row_parts)
            row_parts = []
            yield re.sub(r"\\\\\s*$", "", row).strip()


# 모델당 칼럼 수별 레이아웃. 그림에 실제로 쓰는 값은 평균 ASR 과 downstream 정확도 둘뿐이라,
# 각 모델 블록 시작점으로부터의 오프셋만 알면 된다.
#   6 = attack-wise 표 (논문 Table 8): Direct AutoDAN PAIR PAP AVG Accuracy
#   5 = delta 요약표 (논문 Table 2):   JB-AVG dS Accuracy dD dOverall
TABLE_LAYOUTS = {
    6: {"avg_offset": 4, "downstream_offset": 5},
    5: {"avg_offset": 0, "downstream_offset": 2},
}

# 열 머리글이 이 값이면 "GSM8K" 같은 과제 이름이 아니므로 축 라벨에 중복해 넣지 않는다.
GENERIC_METRIC_NAMES = {"", "accuracy", "acc", "downstream"}


def detect_layout(tabular_text):
    """모델 헤더의 \\multicolumn 폭으로 표 레이아웃을 판별한다.

    6 을 먼저 시도해야 한다. attack-wise 표에는 Safety 소계용 \\multicolumn{5} 가
    두 개 들어 있어서, 5 를 먼저 보면 그것을 모델 이름으로 오인한다.
    """
    for group_width in (6, 5):
        model_names = re.findall(
            r"\\multicolumn\{%d\}\{c\|?\}\{\\textbf\{([^{}]+)\}\}" % group_width,
            tabular_text,
        )
        if len(model_names) == 2:
            return group_width, model_names
    raise ValueError(
        "Expected two model names declared as \\multicolumn{6} or \\multicolumn{5} "
        "in a tabular, found none"
    )


def parse_tabular(tabular_text):
    group_width, model_names = detect_layout(tabular_text)
    layout = TABLE_LAYOUTS[group_width]
    expected_cells = 1 + 2 * group_width

    header_seen = False
    records = {
        model_names[0]: {"metric": None, "points": []},
        model_names[1]: {"metric": None, "points": []},
    }

    for row in iter_latex_rows(tabular_text):
        cells = [clean_latex_cell(cell) for cell in row.split("&")]
        if len(cells) != expected_cells:
            continue

        if cells[0] == "Method":
            for group_index, model_name in enumerate(model_names):
                start = 1 + group_index * group_width
                metric = cells[start + layout["downstream_offset"]]
                if metric.lower() in GENERIC_METRIC_NAMES:
                    metric = None
                records[model_name]["metric"] = metric
            header_seen = True
            continue

        if not header_seen:
            continue

        method = cells[0]
        if method.startswith("\\") or method in {"toprule", "midrule", "bottomrule"}:
            continue

        for group_index, model_name in enumerate(model_names):
            start = 1 + group_index * group_width
            avg_asr = parse_float(cells[start + layout["avg_offset"]])
            downstream = parse_float(cells[start + layout["downstream_offset"]])
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


def parse_task_tsv(tsv_path, model_label):
    """과제별 블록이 이어진 TSV 를 읽는다. 블록 하나가 그림 패널 하나가 된다.

    값이 0~1 비율로 적혀 있으므로 100 을 곱해 퍼센트로 맞춘다.
    """
    records = {}
    current = None

    for raw_line in Path(tsv_path).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in raw_line.split("\t")]

        if len(cells) >= 9 and cells[3:8] == TASK_TSV_HEADER:
            task = cells[8]
            panel_name = f"{model_label} ({task})" if model_label else task
            records[panel_name] = {"metric": task, "points": []}
            current = records[panel_name]
            continue

        if current is None or len(cells) < 9:
            continue

        try:
            avg_asr = float(cells[7]) * 100.0
            downstream = float(cells[8]) * 100.0
        except ValueError:
            # 블록 사이의 빈 줄 / 소제목 줄. 다음 헤더가 나올 때까지 수집을 멈춘다.
            current = None
            continue

        method = METHOD_ALIASES.get(cells[2], cells[2])
        current["points"].append(
            {
                "method": method,
                "downstream": downstream,
                "avg_asr": avg_asr,
                "defense_success": 100.0 - avg_asr,
            }
        )

    if not records:
        raise ValueError(f"No task blocks found in {tsv_path}")
    return records


def load_records(input_path, model_label):
    """LaTeX 표와 과제별 TSV 를 내용으로 구분해 읽는다 (둘 다 .txt 로 쓰이므로 확장자는 못 믿는다)."""
    text = Path(input_path).read_text(encoding="utf-8")
    if "\\begin{tabular}" in text:
        return parse_table_file(input_path)
    return parse_task_tsv(input_path, model_label)


def drop_methods(records, excluded):
    if not excluded:
        return records
    for model_record in records.values():
        model_record["points"] = [
            point for point in model_record["points"] if point["method"] not in excluded
        ]
    return records


def parse_axis_overrides(spec):
    """'25' -> 모든 패널에 적용. 'MedQA=25,ARC-C=30' -> 패널별 적용.

    키는 패널 제목이나 과제 이름의 일부면 된다 (대소문자 무시).
    """
    overrides = {}
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            overrides[key.strip().lower()] = float(value)
        else:
            overrides[None] = float(chunk)
    return overrides


def resolve_xmin(overrides, panel_name, metric):
    if not overrides:
        return None
    haystack = f"{panel_name} {metric or ''}".lower()
    for key, value in overrides.items():
        if key is not None and key in haystack:
            return value
    return overrides.get(None)


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


def legend_ncol(method_count, target_rows):
    """target_rows 줄에 들어가도록 열 수를 역산한다."""
    target_rows = max(1, target_rows)
    return max(1, -(-method_count // target_rows))


def make_legend_handles(methods, marker_scale=1.0):
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
                markersize=(5.8 if method != "WSR-Tune (Ours)" else 8.0) * marker_scale,
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


def draw_model_axis(ax, model_name, model_record, show_ylabel=False, xmin=None):
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
    ax.set_xlabel(f"{metric} Accuracy (%) ↑" if metric else "Downstream task Accuracy (%) ↑")
    if show_ylabel:
        ax.set_ylabel("Defense Success Rate (%) ↑")
    ax.grid(True, linestyle=":", linewidth=0.55, color="#D6CEC0", alpha=0.9)
    ax.set_axisbelow(True)

    left = max(0.0, x_min - x_pad) if xmin is None else xmin
    if xmin is not None and xmin > x_min:
        clipped = [p["method"] for p in points if p["downstream"] < xmin]
        if clipped:
            print(
                f"  warning: {model_name}: xmin={xmin} hides "
                + ", ".join(f"{m} ({d:.2f})" for m, d in
                            ((p["method"], p["downstream"]) for p in points if p["downstream"] < xmin))
            )
    ax.set_xlim(left, x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), min(101.0, y_max + y_pad))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=2.5, pad=1.8)


def add_shared_legend(
    fig,
    records,
    target_rows=LEGEND_TARGET_ROWS,
    marker_scale=1.0,
    fontsize=8,
    y_start=0.02,
    y_step=0.04,
):
    legend_rows = get_legend_rows(records)

    for index, row in enumerate(legend_rows):
        legend = fig.legend(
            handles=make_legend_handles(row, marker_scale=marker_scale),
            loc="lower center",
            bbox_to_anchor=(0.5, y_start - index * y_step),
            ncol=legend_ncol(len(row), target_rows),
            frameon=False,
            fontsize=fontsize,
            handletextpad=0.35,
            columnspacing=1.1,
            borderaxespad=0.0,
        )
        if index < len(legend_rows) - 1:
            fig.add_artist(legend)


def plot_legend(records, output_dir, target_rows=LEGEND_TARGET_ROWS, marker_scale=LEGEND_MARKER_SCALE):
    legend_rows = get_legend_rows(records)
    # 한 LEGEND_ROWS 묶음이 matplotlib 안에서 target_rows 줄로 접히므로 높이도 그만큼 잡는다.
    fig_height = 0.28 + 0.24 * len(legend_rows) * max(1, target_rows)
    fig = plt.figure(figsize=(5.4, fig_height))

    if len(legend_rows) == 1:
        y_positions = [0.40]
    elif len(legend_rows) == 2:
        y_positions = [0.58, 0.16]
    else:
        y_positions = [0.70, 0.36, 0.02]

    for index, row in enumerate(legend_rows):
        legend = fig.legend(
            handles=make_legend_handles(row, marker_scale=marker_scale),
            loc="lower center",
            bbox_to_anchor=(0.5, y_positions[index]),
            ncol=legend_ncol(len(row), target_rows),
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


def plot_combined(
    records,
    output_dir,
    target_rows=LEGEND_TARGET_ROWS,
    ylabel_every_panel=False,
    panel_width=4.3,
    panel_height=3.6,
    xmin_overrides=None,
):
    model_order = get_model_order(records)
    # 패널 폭은 단일 그림(4.9in)에 준해 잡는다. 좁게 잡으면 rcParams 글자 크기가 그대로라
    # 제목과 축 라벨이 이웃 패널을 침범한다.
    fig_width = panel_width * len(model_order)
    # 범례가 차지할 세로 비율. 줄 수가 늘면 그만큼 아래 여백을 더 준다.
    legend_fraction = min(0.30, 0.045 + 0.058 * max(1, target_rows))
    fig, axes = plt.subplots(
        1, len(model_order), figsize=(fig_width, panel_height / (1.0 - legend_fraction))
    )
    if len(model_order) == 1:
        axes = [axes]

    for index, model_name in enumerate(model_order):
        draw_model_axis(
            axes[index],
            model_name,
            records[model_name],
            show_ylabel=ylabel_every_panel or index == 0,
            xmin=resolve_xmin(xmin_overrides, model_name, records[model_name]["metric"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "combined_tradeoff"
    fig.tight_layout(rect=(0, legend_fraction, 1, 1), w_pad=1.4)
    add_shared_legend(
        fig,
        records,
        target_rows=target_rows,
        marker_scale=1.35,
        fontsize=11,
        y_start=0.015,
        y_step=0.055,
    )
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_base


def plot_model(model_name, model_record, output_dir, xmin=None):
    fig, ax = plt.subplots(figsize=(4.9, 3.7))
    draw_model_axis(ax, model_name, model_record, show_ylabel=True, xmin=xmin)

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
    parser.add_argument(
        "tex_file",
        help="Path to a LaTeX table file, or a task-block TSV such as tem_csv.txt",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated PNG/PDF figures. Defaults to the current directory.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Downstream metric name for the x-axis label, e.g. GSM8K. "
            "Needed for delta-style tables whose column header is just 'Accuracy'."
        ),
    )
    parser.add_argument(
        "--model-label",
        default="Llama-2-7B-Chat",
        help=(
            "Model name used in each panel title for TSV input, "
            "e.g. 'Llama-2-7B-Chat (GSM8K)'. Ignored for LaTeX tables."
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated method names to drop, e.g. 'Before Downstream Tuning'.",
    )
    parser.add_argument(
        "--xmin",
        default="",
        help=(
            "Left edge of the x-axis. A bare number applies to every panel; "
            "'MedQA=25,ARC-C=30' applies per panel. Points below it are hidden, "
            "and the run warns when that happens."
        ),
    )
    parser.add_argument(
        "--ylabel-every-panel",
        action="store_true",
        help="Show the y-axis label on every panel of the combined figure.",
    )
    parser.add_argument(
        "--legend-rows",
        type=int,
        default=LEGEND_TARGET_ROWS,
        help=f"Number of rows to wrap the legend into (default: {LEGEND_TARGET_ROWS}).",
    )
    parser.add_argument(
        "--legend-marker-scale",
        type=float,
        default=LEGEND_MARKER_SCALE,
        help=f"Marker size multiplier in the standalone legend (default: {LEGEND_MARKER_SCALE}).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one combined row figure in addition to individual column figures.",
    )
    args = parser.parse_args()

    set_plot_style()
    records = load_records(args.tex_file, args.model_label)
    excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
    records = drop_methods(records, excluded)
    if args.metric:
        for model_record in records.values():
            model_record["metric"] = args.metric
    output_dir = Path(args.output_dir)

    xmin_overrides = parse_axis_overrides(args.xmin)

    for model_name in get_model_order(records):
        output_base = plot_model(
            model_name,
            records[model_name],
            output_dir,
            xmin=resolve_xmin(xmin_overrides, model_name, records[model_name]["metric"]),
        )
        print(f"Wrote {output_base}.png, {output_base}.pdf, and {output_base}.svg")

    output_base = plot_legend(
        records,
        output_dir,
        target_rows=args.legend_rows,
        marker_scale=args.legend_marker_scale,
    )
    print(f"Wrote {output_base}.png, {output_base}.pdf, and {output_base}.svg")

    if args.combined:
        output_base = plot_combined(
            records,
            output_dir,
            target_rows=args.legend_rows,
            ylabel_every_panel=args.ylabel_every_panel,
            xmin_overrides=xmin_overrides,
        )
        print(f"Wrote {output_base}.png, {output_base}.pdf, and {output_base}.svg")


if __name__ == "__main__":
    main()
