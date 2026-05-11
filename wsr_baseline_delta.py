import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Data from table
# Safety Score = 100 - AVG ASR
# =========================
data = [
    {
        "model": "Llama-2-7B-Chat",
        "baseline": "SN-Tune",
        "avg_before": 27.18,
        "gsm_before": 38.99,
        "avg_after": 23.07,
        "gsm_after": 41.02,
    },
    {
        "model": "Llama-2-7B-Chat",
        "baseline": "SafeInstr",
        "avg_before": 13.55,
        "gsm_before": 38.44,
        "avg_after": 9.53,
        "gsm_after": 39.27,
    },
    {
        "model": "Llama-2-13B-Chat",
        "baseline": "SN-Tune",
        "avg_before": 28.97,
        "gsm_before": 48.22,
        "avg_after": 24.69,
        "gsm_after": 48.75,
    },
    {
        "model": "Llama-2-13B-Chat",
        "baseline": "SafeInstr",
        "avg_before": 5.82,
        "gsm_before": 46.55,
        "avg_after": 2.93,
        "gsm_after": 48.60,
    },
]

# Convert ASR AVG to safety score
for d in data:
    d["safety_before"] = 100.0 - d["avg_before"]
    d["safety_after"] = 100.0 - d["avg_after"]

# =========================
# Style
# =========================
model_styles = {
    "Llama-2-7B-Chat": {
        "marker": "o",
        "color": "tab:blue",
        "label": "Llama-2-7B-Chat",
    },
    "Llama-2-13B-Chat": {
        "marker": "s",
        "color": "tab:orange",
        "label": "Llama-2-13B-Chat",
    },
}

baseline_offsets = {
    ("Llama-2-7B-Chat", "SN-Tune"): (1.5, -0.35),
    ("Llama-2-7B-Chat", "SafeInstr"): (1.65, -0.35),
    ("Llama-2-13B-Chat", "SN-Tune"): (1.5, -0.35),
    ("Llama-2-13B-Chat", "SafeInstr"): (1.65, -0.35),
}

# =========================
# Main figure (without legend)
# =========================
fig, ax = plt.subplots(figsize=(8.2, 5.8))

for d in data:
    model = d["model"]
    baseline = d["baseline"]
    style = model_styles[model]

    x0, y0 = d["gsm_before"], d["safety_before"]
    x1, y1 = d["gsm_after"], d["safety_after"]

    # Before point
    ax.scatter(
        x0, y0,
        marker=style["marker"],
        s=95,
        color=style["color"],
        alpha=0.75,
        zorder=3,
    )

    # After point (+WSR)
    ax.scatter(
        x1, y1,
        marker="*",
        s=230,
        color=style["color"],
        edgecolors=style["color"],
        linewidths=1.0,
        zorder=4,
    )

    # Arrow
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            linestyle="--",
            linewidth=1.8,
            color=style["color"],
            shrinkA=5,
            shrinkB=5,
            mutation_scale=25,
        ),
        zorder=2,
    )

    # Baseline label
    dx, dy = baseline_offsets[(model, baseline)]
    ax.text(
        x0 + dx + 0.4, y0 + dy + 0.3,
        baseline,
        fontsize=14,
        ha="right",
        va="top",
        color="black",
    )

    # +WSR label
    ax.text(
        x1 + 0.2, y1 + 0.3,
        "+WSR",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=style["color"],
    )

# Reduce empty space
all_x = [d["gsm_before"] for d in data] + [d["gsm_after"] for d in data]
all_y = [d["safety_before"] for d in data] + [d["safety_after"] for d in data]

x_margin = 2.2
y_margin = 2.2
ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

# Best direction annotation
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()


# x축과 y축의 눈금 숫자(70, 75, 38, 40 등) 크기 조절
ax.tick_params(axis='x', labelsize=15) # x축 숫자 크기
ax.tick_params(axis='y', labelsize=15) # y축 숫자 크기


ax.set_title("Effect of Applying WSR-Tune to Existing Baselines", fontsize=16, pad=12)
ax.set_xlabel("GSM8K Accuracy (%) ↑", fontsize=16, fontweight="bold")
ax.set_ylabel("Defence Success Rate ↑", fontsize=16, fontweight="bold")
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)

plt.tight_layout()
# 메인 그래프 저장 시
plt.savefig("wsr_baseline_improvement.svg", format="svg", bbox_inches="tight")

# 범례 저장 시
plt.savefig("wsr_baseline_improvement_legend.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()

# =========================
# Separate legend figure
# =========================
legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='tab:blue', markeredgecolor='tab:blue',
           markersize=9, label='Llama-2-7B-Chat'),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor='tab:orange', markeredgecolor='tab:orange',
           markersize=9, label='Llama-2-13B-Chat'),
    Line2D([0], [0], marker='*', color='w',
           markerfacecolor='gray', markeredgecolor='gray',
           markersize=14, label='After WSR-Tune'),
    # Line2D([0], [0], linestyle='--', color='gray',
    #        linewidth=1.8, label='WSR-Tune Improvement'),
]

fig_leg = plt.figure(figsize=(7.2, 1.0))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis("off")

legend = ax_leg.legend(
    handles=legend_handles,
    loc="center",
    ncol=1,
    frameon=True,
    fontsize=14,
)

plt.tight_layout()
plt.savefig("wsr_baseline_improvement_legend.svg", format="svg", bbox_inches="tight",pad_inches=0.01, transparent=True)
plt.savefig("wsr_baseline_improvement_legend.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)
plt.show()