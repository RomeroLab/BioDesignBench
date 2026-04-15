#!/usr/bin/env python3
"""BDB-022 Fig 2: Main Leaderboard (stacked horizontal bar chart).

Generates a Nature-style stacked horizontal bar chart showing 9 agent
conditions sorted by total score, with 6 scoring components color-coded.
User-mode bars use solid fill; benchmark-mode bars use hatched fill;
the Hardcoded Pipeline bar uses a gray edge.

An inset table (9 rows x 7 columns) provides exact numeric breakdowns.

Output: results/analysis/fig2_leaderboard.png (300 dpi)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so `from scripts.analysis.load_results import load_all` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analysis.load_results import load_all

# ---------------------------------------------------------------------------
# Nature-style rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    }
)

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------
COMPONENTS: list[str] = [
    "approach",
    "orchestration",
    "quality",
    "feasibility",
    "novelty",
    "diversity",
]

COMPONENT_LABELS: dict[str, str] = {
    "approach": "Approach (20)",
    "orchestration": "Orchestration (15)",
    "quality": "Quality (35)",
    "feasibility": "Feasibility (15)",
    "novelty": "Novelty (5)",
    "diversity": "Diversity (10)",
}

COMPONENT_COLORS: dict[str, str] = {
    "approach": "#4477AA",
    "orchestration": "#66CCEE",
    "quality": "#228833",
    "feasibility": "#CCBB44",
    "novelty": "#EE6677",
    "diversity": "#AA3377",
}


def _compute_condition_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-condition mean scores for the 6 components and total.

    Returns:
        DataFrame indexed by condition with columns for each component
        and a 'total' column, sorted by total descending.
    """
    agg = df.groupby("condition", observed=False)[COMPONENTS + ["total"]].mean()
    agg = agg.sort_values("total", ascending=False)
    return agg


def _mode_for_condition(condition: str) -> str:
    """Determine the display mode from the condition name."""
    lower = condition.lower()
    if "benchmark" in lower:
        return "benchmark"
    if "hardcoded" in lower or "pipeline" in lower:
        return "hardcoded"
    return "user"


def generate_fig2(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Fig 2: Main Leaderboard stacked horizontal bar chart.

    Args:
        df: Full results DataFrame from load_all().
        output_path: Path to write the PNG file.
    """
    means = _compute_condition_means(df)

    # Reverse so highest total is at top of the horizontal bar chart
    means = means.iloc[::-1]

    n_conditions = len(means)
    conditions = means.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(n_conditions)
    bar_height = 0.65
    left = np.zeros(n_conditions)

    bars_by_component: dict[str, list] = {}

    for comp in COMPONENTS:
        widths = means[comp].values
        color = COMPONENT_COLORS[comp]

        bar_list = []
        for i, cond in enumerate(conditions):
            mode = _mode_for_condition(cond)

            hatch = "//" if mode == "benchmark" else None
            edgecolor = "#888888" if mode == "hardcoded" else "white"
            linewidth = 1.2 if mode == "hardcoded" else 0.5

            b = ax.barh(
                y_pos[i],
                widths[i],
                height=bar_height,
                left=left[i],
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                hatch=hatch,
                zorder=3,
            )
            bar_list.append(b)

        bars_by_component[comp] = bar_list
        left += widths

    # Total score labels at bar ends
    for i, cond in enumerate(conditions):
        total = means.loc[cond, "total"]
        ax.text(
            total + 0.8,
            y_pos[i],
            f"{total:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    # Axis configuration
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions, fontsize=10)
    ax.set_xlabel("Score (out of 100)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_title(
        "BioDesignBench Leaderboard",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # Grid lines (behind bars)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------------
    # Legend: component colors + mode indicators
    # -----------------------------------------------------------------------
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=COMPONENT_COLORS[c], edgecolor="white", label=COMPONENT_LABELS[c])
        for c in COMPONENTS
    ]
    # Mode indicators
    legend_handles.append(
        Patch(facecolor="#CCCCCC", edgecolor="white", label="User mode (solid)")
    )
    legend_handles.append(
        Patch(
            facecolor="#CCCCCC",
            edgecolor="white",
            hatch="//",
            label="Benchmark mode (hatched)",
        )
    )
    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor="#888888",
            linewidth=1.2,
            label="Hardcoded Pipeline",
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        ncol=1,
    )

    # -----------------------------------------------------------------------
    # Inset table: 9 agents x 7 columns (6 components + Total)
    # -----------------------------------------------------------------------
    # Re-sort for table display: highest total first
    table_data = means.iloc[::-1]  # Reverse back to descending for table

    col_labels = [COMPONENT_LABELS[c].split(" (")[0] for c in COMPONENTS] + ["Total"]

    cell_text = []
    for _, row in table_data.iterrows():
        vals = [f"{row[c]:.1f}" for c in COMPONENTS]
        vals.append(f"{row['total']:.1f}")
        cell_text.append(vals)

    row_labels = table_data.index.tolist()

    # Position the table below the chart
    table_ax = fig.add_axes([0.13, -0.02, 0.82, 0.22])  # [left, bottom, width, height]
    table_ax.axis("off")

    tbl = table_ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.3)

    # Style header row
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#F0F0F0")
        cell.set_text_props(fontweight="bold")

    # Style row labels
    for i in range(len(row_labels)):
        cell = tbl[i + 1, -1]  # Row label cells have col index -1
        cell.set_text_props(fontsize=7)

    # Color the Total column
    for i in range(len(row_labels)):
        cell = tbl[i + 1, len(col_labels) - 1]
        cell.set_text_props(fontweight="bold")

    # Color component column headers to match bar colors
    for j, comp in enumerate(COMPONENTS):
        cell = tbl[0, j]
        cell.set_facecolor(COMPONENT_COLORS[comp])
        cell.set_text_props(color="white", fontweight="bold")

    # Total column header
    tbl[0, len(col_labels) - 1].set_facecolor("#333333")
    tbl[0, len(col_labels) - 1].set_text_props(color="white", fontweight="bold")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Entry point: load data, generate figure."""
    df = load_all()
    output_path = PROJECT_ROOT / "results" / "analysis" / "fig2_leaderboard.png"
    generate_fig2(df, output_path)

    # Print summary
    means = _compute_condition_means(df)
    print("\nLeaderboard (sorted by total):")
    for cond, row in means.iterrows():
        print(f"  {cond:30s}  {row['total']:5.1f}")


if __name__ == "__main__":
    main()
