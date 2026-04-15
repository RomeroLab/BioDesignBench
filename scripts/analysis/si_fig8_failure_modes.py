#!/usr/bin/env python3
"""SI Figure 8: Failure Mode Analysis.

(a) Zero-Score Component Heatmap — fraction of tasks where each component
    scores exactly zero, per condition.
(b) Error Type Stacked Bar — tasks classified into failure categories
    (no output, wrong approach, poor quality, tool failure, low diversity,
    partial success, success) as stacked horizontal bars per condition.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

import numpy as np
import matplotlib.gridspec as gridspec


# ── Error type classification ────────────────────────────────────────
_ERROR_TYPES = [
    "No output",
    "Approach=0",
    "Quality=0",
    "Tool failure",
    "Low diversity",
    "Success",
    "Partial",
]

# Qualitative color scheme (distinct, colorblind-friendly)
_ERROR_COLORS = {
    "No output":      "#d73027",   # dark red
    "Approach=0":     "#fc8d59",   # orange
    "Quality=0":      "#fee08b",   # yellow
    "Tool failure":   "#91bfdb",   # light blue
    "Low diversity":  "#4575b4",   # dark blue
    "Success":        "#1a9850",   # green
    "Partial":        "#d9d9d9",   # gray
}


def _classify_error(row: pd.Series) -> str:
    """Classify a single task result into one of the error types.

    Priority order (first match wins):
      1. No output: total = 0
      2. Approach=0: approach = 0 but total > 0
      3. Quality=0: quality = 0 but approach > 0
      4. Tool failure: failed_tools > 0
      5. Low diversity: diversity = 0 but quality > 0
      6. Success: total > 30
      7. Partial: everything else
    """
    total = row.get("total", 0) or 0
    approach = row.get("approach", 0) or 0
    quality = row.get("quality", 0) or 0
    diversity = row.get("diversity", 0) or 0
    failed = row.get("failed_tools", 0) or 0

    if total == 0:
        return "No output"
    if approach == 0:
        return "Approach=0"
    if quality == 0:
        return "Quality=0"
    if failed > 0:
        return "Tool failure"
    if diversity == 0 and quality > 0:
        return "Low diversity"
    if total > 30:
        return "Success"
    return "Partial"


def panel_a(ax) -> None:
    """Zero-Score Component Heatmap (conditions x components)."""
    df = load_all()

    n_conds = len(CONDITION_ORDER_NO_ORACLE)
    n_comps = len(COMPONENTS)
    matrix = np.zeros((n_conds, n_comps))

    for i, cond in enumerate(CONDITION_ORDER_NO_ORACLE):
        cond_df = df[df["condition"] == cond]
        n_tasks = len(cond_df)
        if n_tasks == 0:
            continue
        for j, comp in enumerate(COMPONENTS):
            n_zero = (cond_df[comp] == 0).sum()
            matrix[i, j] = 100.0 * n_zero / n_tasks

    # Custom colormap matching SI palette
    _HEAT_COLORS = ["#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    heat_cmap = mcolors.LinearSegmentedColormap.from_list("heat", _HEAT_COLORS, N=256)
    im = ax.imshow(matrix, cmap=heat_cmap, aspect="auto", interpolation="nearest",
                   vmin=0, vmax=max(matrix.max(), 1.0))

    # Labels
    row_labels = [CONDITION_SHORT[c] for c in CONDITION_ORDER_NO_ORACLE]
    col_labels = [COMP_LABELS[c] for c in COMPONENTS]
    ax.set_yticks(range(n_conds))
    ax.set_yticklabels(row_labels, fontsize=MIN_PT - 1)
    ax.set_xticks(range(n_comps))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=MIN_PT - 1)

    # Annotate cells with percentage
    for i in range(n_conds):
        for j in range(n_comps):
            val = matrix[i, j]
            normed = val / max(matrix.max(), 1e-9)
            rgba = heat_cmap(normed)
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=MIN_PT - 2, color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Zero-score fraction (%)", fontsize=MIN_PT - 1)
    cbar.ax.tick_params(labelsize=MIN_PT - 1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=True, left=True, right=False)



def panel_b(ax) -> None:
    """Error Type Stacked Horizontal Bar Chart."""
    df = load_all()

    # Classify each row
    df = df.copy()
    df["error_type"] = df.apply(_classify_error, axis=1)

    # Build fraction matrix: conditions x error_types
    n_conds = len(CONDITION_ORDER_NO_ORACLE)
    fractions = {etype: np.zeros(n_conds) for etype in _ERROR_TYPES}

    for i, cond in enumerate(CONDITION_ORDER_NO_ORACLE):
        cond_df = df[df["condition"] == cond]
        n_tasks = len(cond_df)
        if n_tasks == 0:
            continue
        counts = cond_df["error_type"].value_counts()
        for etype in _ERROR_TYPES:
            fractions[etype][i] = counts.get(etype, 0) / n_tasks

    # Plot stacked horizontal bars
    y_pos = np.arange(n_conds)
    left = np.zeros(n_conds)

    for etype in _ERROR_TYPES:
        vals = fractions[etype]
        ax.barh(y_pos, vals, left=left, height=0.7,
                color=_ERROR_COLORS[etype], edgecolor="white",
                linewidth=0.3, label=etype, zorder=3)
        left += vals

    # Labels
    row_labels = [CONDITION_SHORT[c] for c in CONDITION_ORDER_NO_ORACLE]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=MIN_PT - 1)
    ax.set_xlabel("Fraction of tasks", fontsize=MIN_PT)
    ax.set_xlim(0, 1.0)

    ax.invert_yaxis()  # top condition first

    # Legend drawn externally
    ax._leg_handles, ax._leg_labels = ax.get_legend_handles_labels()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)



def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.45, 0.45, 0.10],
        hspace=0.50,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    panel_a(ax_a)
    panel_b(ax_b)

    # Separate legend row
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=ax_b._leg_handles, labels=ax_b._leg_labels,
        loc="center", fontsize=max(MIN_PT - 2, 4), frameon=False,
        ncol=7, handlelength=1.0, handletextpad=0.3, columnspacing=0.5,
    )

    save_fig(fig, "si_fig8_failure_modes")


if __name__ == "__main__":
    main()
