#!/usr/bin/env python3
"""BioDesignBench Fig 5: Per-Category Analysis — simplified dot plot (120×80mm).

Panel A (left ~55%):  Per molecular subject (5 rows × 5 agents)
Panel B (right ~45%): De novo vs Redesign (2 rows × 5 agents)
Bottom: shared legend

Output: 120mm x 80mm, 300 dpi.  All text >= 7 pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── Nature-style rcParams ────────────────────────────────────────────────
_MIN_PT = 7
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": _MIN_PT,
    "axes.linewidth": 0.5,
    "axes.labelsize": _MIN_PT,
    "axes.titlesize": 8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.labelsize": _MIN_PT,
    "ytick.labelsize": _MIN_PT,
    "legend.fontsize": _MIN_PT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── 7 agents: Oracle + Human Expert + User mode LLMs + Hardcoded ────────
AGENTS = [
    {"name": "Oracle",         "cond": "Oracle",               "color": "#DDAA33", "marker": "D", "ms": 5.0},
    {"name": "Human Expert",   "cond": "Human Expert",         "color": "#AA3377", "marker": "^", "ms": 5.5},
    {"name": "DeepSeek V3",    "cond": "DeepSeek V3 user",     "color": "#4477AA", "marker": "o", "ms": 5.5},
    {"name": "GPT-5",          "cond": "GPT-5 user",           "color": "#EE6677", "marker": "o", "ms": 5.5},
    {"name": "Sonnet 4.5",     "cond": "Sonnet 4.5 user",      "color": "#228833", "marker": "o", "ms": 5.5},
    {"name": "Gemini 2.5 Pro", "cond": "Gemini 2.5 Pro user",  "color": "#66CCEE", "marker": "o", "ms": 5.5},
    {"name": "Hardcoded",      "cond": "Hardcoded Pipeline",    "color": "#EE7733", "marker": "s", "ms": 5.0},
]

SUBJECT_LABELS = {
    "scaffold": "Scaffold",
    "binder": "Binder",
    "enzyme": "Enzyme",
    "fluorescent_protein": "Fluor. Protein",
    "antibody": "Antibody",
}

APPROACH_LABELS = {
    "de_novo": "De Novo",
    "redesign": "Redesign",
}


def load_data():
    from scripts.analysis.load_results import load_all
    return load_all()


def _draw_dotplot(ax, group_col, group_order, group_labels, df, show_scores=False):
    """Draw a Cleveland dot plot for given grouping."""
    n_groups = len(group_order)
    y_pos = np.arange(n_groups)
    n_agents = len(AGENTS)
    dodge = np.linspace(-0.28, 0.28, n_agents)

    # Alternating bands
    for i in range(n_groups):
        if i % 2 == 0:
            ax.axhspan(i - 0.42, i + 0.42, color="#f5f5f5", zorder=0)

    for ai, agent in enumerate(AGENTS):
        sub = df[df["condition"] == agent["cond"]]
        grp_means = sub.groupby(group_col, observed=True)["total"].mean()

        for gi, grp in enumerate(group_order):
            if grp not in grp_means.index:
                continue
            val = grp_means[grp]
            y = y_pos[gi] + dodge[ai]

            # Connector line
            ax.plot([0, val], [y, y], color=agent["color"],
                    alpha=0.12, linewidth=0.5, zorder=1)
            # Dot
            ax.scatter(val, y, marker=agent["marker"],
                       s=agent["ms"] ** 2, zorder=3,
                       facecolors=agent["color"], edgecolors="white",
                       linewidths=0.4)

            # Score annotation (Panel B only)
            if show_scores:
                ax.text(val + 1.0, y, f"{val:.0f}",
                        fontsize=_MIN_PT - 1, color=agent["color"],
                        va="center", ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels, fontsize=_MIN_PT, rotation=30,
                       rotation_mode="anchor", ha="right", va="center")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", which="major", color="#e0e0e0", lw=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    df = load_data()

    # ── Figure + nested GridSpec ─────────────────────────────────────────
    fig_w = 120 / 25.4
    fig_h = 80 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(2, 1, height_ratios=[0.88, 0.12], hspace=0.20)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.40)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_leg = fig.add_subplot(gs[1, :])

    # ── Panel A: Per Molecular Subject ───────────────────────────────────
    # Order by task count descending (bottom → top in plot)
    # Use Oracle for counting (always complete), fall back to first available
    if "Oracle" in df["condition"].cat.categories:
        ref = df[df["condition"] == "Oracle"]
    else:
        ref = df[df["condition"] == "DeepSeek V3 user"]
    subj_counts = ref.groupby("molecular_subject", observed=True).size()\
        .sort_values(ascending=True)
    subj_order = subj_counts.index.tolist()
    subj_labels = [f"{SUBJECT_LABELS.get(s, s)} ({subj_counts[s]})"
                   for s in subj_order]

    _draw_dotplot(ax_a, "molecular_subject", subj_order, subj_labels, df)

    # ── Panel B: De Novo vs Redesign ─────────────────────────────────────
    appr_counts = ref.groupby("design_approach", observed=True).size()\
        .sort_values(ascending=True)
    appr_order = appr_counts.index.tolist()
    appr_labels = [f"{APPROACH_LABELS.get(a, a)} ({appr_counts[a]})"
                   for a in appr_order]

    _draw_dotplot(ax_b, "design_approach", appr_order, appr_labels, df,
                  show_scores=False)

    # ── Shared legend (bottom) ───────────────────────────────────────────
    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], marker=a["marker"], color="w",
               markerfacecolor=a["color"], markeredgecolor="white",
               markersize=a["ms"] - 0.5, label=a["name"])
        for a in AGENTS
    ]
    ax_leg.legend(
        handles=legend_handles, loc="center", ncol=4,
        fontsize=_MIN_PT, frameon=False,
        handletextpad=0.3, columnspacing=1.0,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.04, top=0.96)

    # Center legend on full figure width
    p = ax_leg.get_position()
    leg_w = 0.70
    ax_leg.set_position([0.5 - leg_w / 2, p.y0, leg_w, p.height])

    png_path = OUT / "fig5_category_dotplot.png"
    pdf_path = OUT / "fig5_category_dotplot.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Size: {fig_w:.2f}\" x {fig_h:.2f}\" ({fig_w*25.4:.0f}mm x {fig_h*25.4:.0f}mm)")


if __name__ == "__main__":
    main()
