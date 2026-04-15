#!/usr/bin/env python3
"""SI Figure 5: AF2 Validation Metrics.

Left: 2x2 grid of box plots (pLDDT, pTM, ipTM, i_pAE) with shared axes.
Right: Scatter of Quality score vs pLDDT fraction with regression.

Layout: 2 columns — left=2×2 box plots (shared y/x), right=scatter.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

from scipy import stats


# ── Minimum data points required to include a metric in the box plot ──
_MIN_DATA_PER_METRIC = 5

# ── AF2 metric columns and display labels ─────────────────────────────
_AF2_METRICS = [
    ("pLDDT_frac", "pLDDT"),
    ("pTM_frac", "pTM"),
    ("ipTM_frac", "ipTM"),
    ("i_pAE_frac", "i_pAE"),
]


def _box_subplot(ax, df: pd.DataFrame, col: str, label: str,
                 show_ylabel: bool = False, show_xlabel: bool = False) -> bool:
    """Draw one box-plot subplot for a single AF2 metric."""
    sub = df[[col, "condition"]].dropna(subset=[col])
    if len(sub) < _MIN_DATA_PER_METRIC:
        ax.text(
            0.5, 0.5, f"No data\n({col})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=MIN_PT, color="grey",
        )
        ax.set_title(label, fontsize=MIN_PT, fontweight="bold")
        ax.set_xticks([])
        return False

    conditions = [c for c in CONDITION_ORDER if c in sub["condition"].unique()]
    if not conditions:
        ax.set_title(label, fontsize=MIN_PT, fontweight="bold")
        ax.set_xticks([])
        return False

    data_groups = [sub.loc[sub["condition"] == c, col].values for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conditions]

    bp = ax.boxplot(
        data_groups,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=1.5, alpha=0.4, markerfacecolor="#888888"),
        medianprops=dict(color="black", linewidth=0.6),
        whiskerprops=dict(linewidth=0.4),
        capprops=dict(linewidth=0.4),
        boxprops=dict(linewidth=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # X-ticks: model names on bottom row only
    ax.set_xticks(range(1, len(conditions) + 1))
    if show_xlabel:
        short = [CONDITION_SHORT.get(c, c) for c in conditions]
        ax.set_xticklabels(short, rotation=45, ha="right",
                           fontsize=max(MIN_PT - 3, 4))
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    ax.set_title(label, fontsize=MIN_PT, fontweight="bold", pad=2)

    if show_ylabel:
        ax.set_ylabel("Fraction", fontsize=MIN_PT - 1)
    else:
        ax.set_yticklabels([])

    ax.set_ylim(-0.05, 1.1)
    style_grid(ax, axis="y")
    return True


def panel_boxplots(fig, gs_left) -> None:
    """2x2 AF2 fraction box plots with shared axes."""
    df = load_all()

    # Create 2×2 subplots within the left gridspec
    axes = [
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1])],
    ]

    for idx, (col, label) in enumerate(_AF2_METRICS):
        row, col_idx = idx // 2, idx % 2
        ax = axes[row][col_idx]
        show_ylabel = (col_idx == 0)   # only left column
        show_xlabel = (row == 1)       # only bottom row
        _box_subplot(ax, df, col, label,
                     show_ylabel=show_ylabel, show_xlabel=show_xlabel)


def panel_scatter(ax) -> None:
    """Quality score vs pLDDT scatter with regression and outlier highlights."""
    df = load_all()

    metric_col = "pLDDT_frac"
    metric_label = "pLDDT"
    sub = df[["quality", metric_col, "condition"]].dropna(subset=["quality", metric_col])

    if len(sub) < _MIN_DATA_PER_METRIC:
        metric_col = "pTM_frac"
        metric_label = "pTM"
        sub = df[["quality", metric_col, "condition"]].dropna(subset=["quality", metric_col])

    if len(sub) < _MIN_DATA_PER_METRIC:
        ax.text(
            0.5, 0.5, "Insufficient AF2 data\nfor scatter plot",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=MIN_PT, color="grey",
        )
        return

    # Plot each condition
    for cond in CONDITION_ORDER:
        cond_sub = sub[sub["condition"] == cond]
        if cond_sub.empty:
            continue
        c = CONDITION_COLORS.get(cond, "#888888")
        mkw = cond_marker_kw(cond)
        ax.scatter(
            cond_sub[metric_col], cond_sub["quality"],
            s=8, alpha=0.5, label=cond, zorder=3, **mkw,
        )

    # Regression line
    x_vals = sub[metric_col].values
    y_vals = sub["quality"].values
    r_val, p_val = stats.pearsonr(x_vals, y_vals)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="#555555",
            linewidth=0.8, zorder=2)

    # Highlight outliers
    outlier_mask = (
        ((sub["quality"] > 25) & (sub[metric_col] < 0.5))
        | ((sub["quality"] < 10) & (sub[metric_col] > 0.8))
    )
    outliers = sub[outlier_mask]
    if len(outliers) > 0:
        ax.scatter(
            outliers[metric_col], outliers["quality"],
            s=20, facecolors="none", edgecolors="red", linewidths=0.8,
            zorder=5, label=f"Outliers (n={len(outliers)})",
        )

    # Pearson annotation
    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(
        0.05, 0.95,
        f"r = {r_val:.3f}\n{p_str}\nn = {len(sub)}",
        transform=ax.transAxes, fontsize=MIN_PT - 1, va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel(f"{metric_label} fraction", fontsize=MIN_PT)
    ax.set_ylabel("Quality score (0-35)", fontsize=MIN_PT)
    style_grid(ax)

    # Store handles for external legend
    ax._leg_handles, ax._leg_labels = ax.get_legend_handles_labels()


def main() -> None:
    """Create SI Figure 5 and save to results/analysis/."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Main layout: left=box plots, right=scatter + legend below
    gs_main = GridSpec(
        2, 2, figure=fig,
        width_ratios=[0.45, 0.55],
        height_ratios=[0.88, 0.12],
        hspace=0.55, wspace=0.35,
    )

    # Left: 2×2 box plot grid
    gs_left = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_main[0, 0],
        hspace=0.25, wspace=0.10,
    )

    panel_boxplots(fig, gs_left)

    # Right: scatter plot
    ax_scatter = fig.add_subplot(gs_main[0, 1])
    panel_scatter(ax_scatter)

    # Bottom: shared legend spanning both columns
    ax_leg = fig.add_subplot(gs_main[1, :])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=ax_scatter._leg_handles, labels=ax_scatter._leg_labels,
        loc="center", fontsize=max(MIN_PT - 2, 4), frameon=False,
        ncol=3, markerscale=1.5,
        handletextpad=0.3, columnspacing=1.0, labelspacing=0.6,
    )

    save_fig(fig, "si_fig5_af2_validation")


if __name__ == "__main__":
    main()
