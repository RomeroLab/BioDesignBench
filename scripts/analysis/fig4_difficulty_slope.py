#!/usr/bin/env python3
"""BioDesignBench Fig 4: Difficulty Analysis — 2×2 small multiples slope chart.

Each subplot: 1 model (User + BM) + Hardcoded reference.
Shared y-axis range for cross-model comparison.

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

# ── Models (same order & palette as Fig 2) ───────────────────────────────
MODELS = [
    {"name": "DeepSeek V3", "user": "DeepSeek V3 user", "bm": "DeepSeek V3 benchmark", "color": "#4477AA"},
    {"name": "GPT-5",       "user": "GPT-5 user",       "bm": "GPT-5 benchmark",       "color": "#EE6677"},
    {"name": "Sonnet 4.5",    "user": "Sonnet 4.5 user",  "bm": "Sonnet 4.5 benchmark",  "color": "#228833"},
    {"name": "Gemini 2.5 Pro","user": "Gemini 2.5 Pro user","bm": "Gemini 2.5 Pro benchmark","color": "#66CCEE"},
]

DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_SHORT = ["Easy", "Med", "Hard"]


def load_difficulty_data():
    """Load mean scores by difficulty level for each condition."""
    from scripts.analysis.load_results import load_all
    df = load_all()
    diff_counts = df[df["condition"] == df["condition"].cat.categories[0]]\
        .groupby("difficulty", observed=True).size().to_dict()
    means = df.groupby(["condition", "difficulty"], observed=True)["total"]\
        .mean().unstack(fill_value=0)
    return means, diff_counts


def main():
    means, diff_counts = load_difficulty_data()
    x = np.arange(len(DIFF_ORDER))

    # X-tick labels with counts
    xtick_labels = [f"{s}\n({diff_counts.get(d, 0)})" for d, s in zip(DIFF_ORDER, DIFF_SHORT)]

    # Baseline data
    y_hp = [means.loc["Hardcoded Pipeline", d] if d in means.columns else 0
            for d in DIFF_ORDER]
    y_he = [means.loc["Human Expert", d] if "Human Expert" in means.index and d in means.columns else 0
            for d in DIFF_ORDER]

    # ── Figure: 2×2 + bottom legend row ──────────────────────────────────
    fig_w = 120 / 25.4
    fig_h = 80 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.08],
                          hspace=0.65, wspace=0.25)

    axes = [
        fig.add_subplot(gs[0, 0]),  # DeepSeek
        fig.add_subplot(gs[0, 1]),  # GPT-5
        fig.add_subplot(gs[1, 0]),  # Sonnet
        fig.add_subplot(gs[1, 1]),  # Gemini
    ]

    # Shared y-range
    y_lo, y_hi = 18, 72

    for idx, (model, ax) in enumerate(zip(MODELS, axes)):
        color = model["color"]

        # User mode
        y_user = [means.loc[model["user"], d] if d in means.columns else 0
                  for d in DIFF_ORDER]
        ax.plot(x, y_user, color=color, linewidth=2.0, linestyle="-",
                marker="o", markersize=5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)

        # BM mode
        y_bm = [means.loc[model["bm"], d] if d in means.columns else 0
                for d in DIFF_ORDER]
        ax.plot(x, y_bm, color=color, linewidth=1.2, linestyle="--",
                marker="o", markersize=4, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=0.8, zorder=2)

        # Hardcoded reference
        ax.plot(x, y_hp, color="#EE7733", linewidth=0.8, linestyle=":",
                marker="s", markersize=3, markerfacecolor="#EE7733",
                markeredgecolor="white", markeredgewidth=0.3, zorder=1)

        # Human Expert reference
        ax.plot(x, y_he, color="#AA3377", linewidth=0.8, linestyle="-.",
                marker="^", markersize=3, markerfacecolor="#AA3377",
                markeredgecolor="white", markeredgewidth=0.3, zorder=1)

        # No end-of-line score annotations — trends speak for themselves

        # Subplot title
        ax.set_title(model["name"], fontsize=_MIN_PT + 1, fontweight="bold",
                     color=color, pad=4)

        # Axes
        ax.set_xlim(-0.15, x[-1] + 0.15)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=_MIN_PT)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(axis="y", which="major", color="#e0e0e0", lw=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Only left column gets y-label
        if idx % 2 == 0:
            ax.set_ylabel("Mean Score", fontsize=_MIN_PT)
        else:
            ax.tick_params(axis="y", labelleft=True)

    # ── Shared legend (bottom) ───────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.0, linestyle="-",
               marker="o", markersize=4, markerfacecolor="#444444",
               markeredgecolor="white", label="User mode"),
        Line2D([0], [0], color="#444444", linewidth=1.2, linestyle="--",
               marker="o", markersize=3.5, markerfacecolor="white",
               markeredgecolor="#444444", label="Benchmark mode"),
        Line2D([0], [0], color="#AA3377", linewidth=0.8, linestyle="-.",
               marker="^", markersize=3, markerfacecolor="#AA3377",
               markeredgecolor="white", label="Human Expert"),
        Line2D([0], [0], color="#EE7733", linewidth=0.8, linestyle=":",
               marker="s", markersize=3, markerfacecolor="#EE7733",
               markeredgecolor="white", label="Hardcoded Pipeline"),
    ]
    ax_leg.legend(
        handles=legend_handles, loc="center", ncol=4,
        fontsize=_MIN_PT, frameon=False,
        handlelength=2.0, columnspacing=1.5, handletextpad=0.5,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.04, top=0.92)

    png_path = OUT / "fig4_difficulty_slope.png"
    pdf_path = OUT / "fig4_difficulty_slope.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Size: {fig_w:.2f}\" x {fig_h:.2f}\" ({fig_w*25.4:.0f}mm x {fig_h*25.4:.0f}mm)")


if __name__ == "__main__":
    main()
