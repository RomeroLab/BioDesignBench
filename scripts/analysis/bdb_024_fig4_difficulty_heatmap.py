#!/usr/bin/env python3
"""Fig 4: Difficulty Heatmap (9 agents x 3 difficulty levels).

Generates an annotated seaborn heatmap showing mean total score for each
(condition, difficulty) combination. Rows are 9 agent conditions sorted by
overall mean total descending; columns are Easy / Medium / Hard.

Special annotations:
  - DeepSeek's difficulty-invariance (62-63 across all levels)
  - Sonnet's Easy -> Hard drop

Usage:
    python scripts/analysis/bdb_024_fig4_difficulty_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants ─────────────────────────────────────────────────────────────

DIFFICULTY_COLS = ["Easy", "Medium", "Hard"]
DIFFICULTY_RAW = ["easy", "medium", "hard"]

# Thresholds for detecting noteworthy patterns
INVARIANCE_THRESHOLD = 3.0  # max spread across difficulty for "invariant" label
DROP_THRESHOLD = 10.0  # Easy-Hard delta to flag as a significant drop

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"


# ── Data processing ──────────────────────────────────────────────────────


def build_pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a 9x3 pivot table of mean total scores by (condition, difficulty).

    Args:
        df: DataFrame with columns 'condition', 'total', 'difficulty'.

    Returns:
        Tuple of (pivot DataFrame with rows=conditions, cols=Easy/Medium/Hard,
                  row_order as list of condition names sorted by overall mean desc).
    """
    # Filter to known difficulty levels only
    df_filtered = df[df["difficulty"].isin(DIFFICULTY_RAW)].copy()

    # Map difficulty to display names
    diff_map = dict(zip(DIFFICULTY_RAW, DIFFICULTY_COLS))
    df_filtered["difficulty_display"] = df_filtered["difficulty"].map(diff_map)

    # Compute mean total per (condition, difficulty)
    pivot = df_filtered.pivot_table(
        index="condition",
        columns="difficulty_display",
        values="total",
        aggfunc="mean",
    )

    # Ensure column order
    pivot = pivot.reindex(columns=DIFFICULTY_COLS)

    # Sort rows by overall mean total descending
    overall_means = df_filtered.groupby("condition")["total"].mean()
    row_order = overall_means.sort_values(ascending=False).index.tolist()
    pivot = pivot.reindex(row_order)

    return pivot, row_order


def find_annotations(pivot: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect noteworthy patterns in the pivot table.

    Returns a list of annotation dicts, each with keys:
        - condition: str
        - kind: "invariance" | "drop"
        - detail: str description
    """
    annotations: list[dict[str, Any]] = []

    for condition in pivot.index:
        row = pivot.loc[condition]
        spread = row.max() - row.min()

        # Difficulty invariance: small spread across levels
        if spread < INVARIANCE_THRESHOLD:
            annotations.append(
                {
                    "condition": condition,
                    "kind": "invariance",
                    "detail": (
                        f"{condition}: {row.min():.0f}-{row.max():.0f} "
                        f"(spread={spread:.1f})"
                    ),
                }
            )

        # Significant Easy->Hard drop
        if "Easy" in row.index and "Hard" in row.index:
            drop = row["Easy"] - row["Hard"]
            if drop >= DROP_THRESHOLD:
                annotations.append(
                    {
                        "condition": condition,
                        "kind": "drop",
                        "detail": (
                            f"{condition}: Easy={row['Easy']:.0f} -> "
                            f"Hard={row['Hard']:.0f} (drop={drop:.0f})"
                        ),
                    }
                )

    return annotations


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_heatmap(
    pivot: pd.DataFrame,
    row_order: list[str],
    annotations: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render the annotated difficulty heatmap and save to disk.

    Args:
        pivot: 9x3 DataFrame (rows=conditions, cols=Easy/Medium/Hard).
        row_order: Condition names in display order (desc by total).
        annotations: List of annotation dicts from find_annotations().
        output_path: Path to save the PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 10))

    # Color map: reversed RdYlBu (dark blue=high, dark red=low)
    cmap = plt.cm.RdYlBu

    # Determine vmin/vmax from data (rounded to nice bounds)
    vmin = max(0, float(np.floor(pivot.min().min() / 5) * 5))
    vmax = min(100, float(np.ceil(pivot.max().max() / 5) * 5))
    # Ensure at least a 10-point range
    if vmax - vmin < 10:
        vmax = vmin + 10

    # Draw the heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Mean Score", "shrink": 0.7},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        ax=ax,
    )

    # Build lookup: condition -> row index in the heatmap
    condition_to_row = {cond: i for i, cond in enumerate(pivot.index)}

    # Annotate special patterns
    for ann in annotations:
        cond = ann["condition"]
        if cond not in condition_to_row:
            continue
        row_idx = condition_to_row[cond]

        if ann["kind"] == "invariance":
            # Draw a gold border rectangle around the entire row
            rect = mpatches.FancyBboxPatch(
                (0, row_idx),
                len(DIFFICULTY_COLS),
                1,
                boxstyle="round,pad=0.02",
                linewidth=2.5,
                edgecolor="#FFD700",
                facecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)
            # Add a star in the margin
            ax.annotate(
                "*",
                xy=(-0.15, row_idx + 0.5),
                xycoords="data",
                fontsize=18,
                fontweight="bold",
                color="#FFD700",
                ha="center",
                va="center",
                annotation_clip=False,
            )
        elif ann["kind"] == "drop":
            # Draw a red dashed border around the row
            rect = mpatches.FancyBboxPatch(
                (0, row_idx),
                len(DIFFICULTY_COLS),
                1,
                boxstyle="round,pad=0.02",
                linewidth=2.0,
                edgecolor="#E53935",
                facecolor="none",
                linestyle="--",
                zorder=5,
            )
            ax.add_patch(rect)
            # Add a down arrow in the margin
            ax.annotate(
                "v",
                xy=(len(DIFFICULTY_COLS) + 0.15, row_idx + 0.5),
                xycoords="data",
                fontsize=14,
                fontweight="bold",
                color="#E53935",
                ha="center",
                va="center",
                annotation_clip=False,
                fontfamily="monospace",
            )

    # Title and labels
    ax.set_title("Mean Score by Difficulty Level", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Rotate y-axis labels for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12, fontweight="bold")

    # Build legend for annotation markers
    legend_handles = []
    invariance_found = any(a["kind"] == "invariance" for a in annotations)
    drop_found = any(a["kind"] == "drop" for a in annotations)

    if invariance_found:
        legend_handles.append(
            mpatches.Patch(
                edgecolor="#FFD700",
                facecolor="none",
                linewidth=2,
                label="Difficulty-invariant (spread < 3 pts)",
            )
        )
    if drop_found:
        legend_handles.append(
            mpatches.Patch(
                edgecolor="#E53935",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label="Significant Easy->Hard drop (>= 10 pts)",
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
            fontsize=9,
            ncol=2,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    """Load results, build pivot, generate heatmap."""
    from scripts.analysis.load_results import load_all

    print("Loading results...")
    df = load_all()
    print(f"  Loaded {len(df)} rows ({df['task_id'].nunique()} tasks x "
          f"{df['condition'].nunique()} conditions)")

    # Summarize difficulty distribution
    for diff in DIFFICULTY_RAW:
        n = df[df["difficulty"] == diff]["task_id"].nunique()
        print(f"  {diff.capitalize()}: {n} tasks")

    print("\nBuilding pivot table...")
    pivot, row_order = build_pivot(df)
    print(pivot.to_string())

    print("\nDetecting annotations...")
    annotations = find_annotations(pivot)
    for ann in annotations:
        print(f"  [{ann['kind'].upper()}] {ann['detail']}")

    output_path = OUTPUT_DIR / "fig4_difficulty_heatmap.png"
    print(f"\nGenerating heatmap -> {output_path}")
    plot_heatmap(pivot, row_order, annotations, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
