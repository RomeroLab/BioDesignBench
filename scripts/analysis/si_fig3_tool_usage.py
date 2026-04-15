#!/usr/bin/env python3
"""SI Figure 3: Tool Usage Analysis.

(a) Tool Frequency Heatmap — mean tool calls per task for each of 17 tools
    across 9 conditions, with hierarchical clustering on both axes.
(b) Tool Diversity vs Total Score Scatter — Shannon entropy of tool usage
    distribution vs mean total score per condition, with Pearson correlation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def _discover_tools(df: pd.DataFrame) -> list[str]:
    """Discover all unique tool names from tool_sequence data."""
    from collections import Counter
    counts = Counter()
    for seq in df["tool_sequence"]:
        if isinstance(seq, list):
            counts.update(seq)
    # Sort by frequency (most used first)
    return [name for name, _ in counts.most_common()]


def _build_tool_freq_matrix(df: pd.DataFrame, tools: list[str]) -> pd.DataFrame:
    """Build (tools) x (conditions) matrix of mean tool calls per task."""
    records: list[dict[str, float]] = []
    for cond in CONDITION_ORDER_NO_ORACLE:
        cond_df = df[df["condition"] == cond]
        if cond_df.empty:
            records.append({t: 0.0 for t in tools})
            continue
        counts = {t: 0.0 for t in tools}
        n_tasks = len(cond_df)
        for _, row in cond_df.iterrows():
            seq = row["tool_sequence"]
            if not isinstance(seq, list):
                continue
            for tool_name in seq:
                if tool_name in counts:
                    counts[tool_name] += 1
        records.append({t: v / n_tasks for t, v in counts.items()})

    mat = pd.DataFrame(records, index=CONDITION_ORDER_NO_ORACLE).T  # tools x conditions
    return mat


def _shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy H = -sum(p_i * log2(p_i)) from raw counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def panel_a(ax) -> None:
    """Tool Frequency Heatmap with hierarchical clustering on both axes."""
    df = load_all()
    tools = _discover_tools(df)
    if not tools:
        ax.text(0.5, 0.5, "No tool usage data", transform=ax.transAxes,
                ha="center", va="center", fontsize=MIN_PT, color="grey")
        return

    # Exclude non-bio utility tools from heatmap
    _UTILITY_TOOLS = {"execute_python", "write_file", "read_file"}
    bio_tools = [t for t in tools if t not in _UTILITY_TOOLS]
    if not bio_tools:
        bio_tools = tools

    mat = _build_tool_freq_matrix(df, bio_tools)

    # Filter out tools with zero usage
    row_sums = mat.sum(axis=1)
    active_tools = row_sums[row_sums > 0].index.tolist()
    if not active_tools:
        active_tools = bio_tools
    mat_filtered = mat.loc[active_tools]

    data = mat_filtered.values

    # Hierarchical clustering on rows (tools) — only if >1 active tool
    if data.shape[0] > 1 and data.max() > 0:
        row_dist = pdist(data, metric="euclidean")
        if len(row_dist) > 0 and not np.all(row_dist == 0):
            row_link = linkage(row_dist, method="ward")
            row_order = leaves_list(row_link)
        else:
            row_order = np.arange(data.shape[0])
    else:
        row_order = np.arange(data.shape[0])

    # Keep CONDITION_ORDER (no clustering on columns)
    col_order = np.arange(data.shape[1])

    # Reorder
    reordered = data[np.ix_(row_order, col_order)]
    row_labels = [mat_filtered.index[i] for i in row_order]
    col_labels = [CONDITION_ORDER_NO_ORACLE[i] for i in col_order]  # full names

    # Custom colormap matching SI Fig 1/2
    _HEAT_COLORS = ["#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    heat_cmap = mcolors.LinearSegmentedColormap.from_list("heat", _HEAT_COLORS, N=256)
    im = ax.imshow(reordered, cmap=heat_cmap, aspect="auto", interpolation="nearest")

    # Use short condition names to save space
    short_col_labels = [CONDITION_SHORT.get(c, c) for c in col_labels]
    ax.set_xticks(range(len(short_col_labels)))
    ax.set_xticklabels(short_col_labels, rotation=55, ha="right", fontsize=max(MIN_PT - 2, 4))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=max(MIN_PT - 2, 5))

    # Annotate cells with value >= 0.5 (skip near-zero to reduce clutter)
    _ANNOT_PT = max(MIN_PT - 3, 4)
    for i in range(reordered.shape[0]):
        for j in range(reordered.shape[1]):
            val = reordered[i, j]
            if val >= 0.5:
                normed = val / max(reordered.max(), 1e-9)
                rgba = heat_cmap(normed)
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "white" if lum < 0.5 else "black"
                fmt = f"{val:.0f}" if val >= 1.0 else f"{val:.1f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        fontsize=_ANNOT_PT, color=text_color)

    # Vertical colorbar on the right of heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Mean calls / task", fontsize=MIN_PT, labelpad=2)
    cbar.ax.tick_params(labelsize=MIN_PT - 1, width=0.3, length=2)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=True, left=True, right=False)



def panel_b(ax) -> None:
    """Tool Diversity (Shannon entropy) vs Mean Total Score scatter."""
    df = load_all()
    all_tool_names = _discover_tools(df)

    entropies: list[float] = []
    mean_scores: list[float] = []
    conditions: list[str] = []

    for cond in CONDITION_ORDER_NO_ORACLE:
        cond_df = df[df["condition"] == cond]
        if cond_df.empty:
            continue

        # Aggregate all tool calls across tasks in this condition
        tool_counts = np.zeros(len(all_tool_names))
        for _, row in cond_df.iterrows():
            seq = row["tool_sequence"]
            if not isinstance(seq, list):
                continue
            for tool_name in seq:
                if tool_name in all_tool_names:
                    tool_counts[all_tool_names.index(tool_name)] += 1

        h = _shannon_entropy(tool_counts)
        mean_total = cond_df["total"].mean()

        entropies.append(h)
        mean_scores.append(mean_total)
        conditions.append(cond)

    x = np.array(entropies)
    y = np.array(mean_scores)

    # Scatter with condition-specific markers (legend drawn externally)
    for i, cond in enumerate(conditions):
        kw = cond_marker_kw(cond)
        ax.scatter(x[i], y[i], s=50, zorder=5, label=cond, **kw)

    # Store handles for external legend
    ax._leg_handles, ax._leg_labels = ax.get_legend_handles_labels()

    # Pearson correlation + regression line
    # Guard against constant input (e.g., all entropies identical)
    x_var = np.var(x)
    y_var = np.var(y)
    if len(x) >= 3 and x_var > 1e-12 and y_var > 1e-12:
        r_val, p_val = pearsonr(x, y)
        # Regression line (use try/except for numerical stability)
        try:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
            y_line = np.polyval(z, x_line)
            ax.plot(x_line, y_line, color="#999999", linestyle="--",
                    linewidth=0.8, zorder=2)
        except np.linalg.LinAlgError:
            pass  # skip regression line if SVD fails

        # Annotation
        sig = "*" if p_val < 0.05 else ""
        ax.text(0.05, 0.95, f"r = {r_val:.2f}{sig}",
                transform=ax.transAxes, fontsize=MIN_PT,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", alpha=0.8))

    ax.set_xlabel("Shannon entropy (tool diversity)", fontsize=MIN_PT)
    ax.set_ylabel("Mean total score", fontsize=MIN_PT)
    style_grid(ax)


def main() -> None:
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[0.88, 0.12],
                  width_ratios=[0.50, 0.50],
                  hspace=0.7, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_a(ax_a)
    panel_b(ax_b)

    # Bottom row: legend spanning both columns
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=ax_b._leg_handles, labels=ax_b._leg_labels,
                  loc="center right", fontsize=max(MIN_PT - 2, 4),
                  ncol=2, frameon=False,
                  handletextpad=0.6, columnspacing=2.5, labelspacing=0.8)

    save_fig(fig, "si_fig3_tool_usage")


if __name__ == "__main__":
    main()
