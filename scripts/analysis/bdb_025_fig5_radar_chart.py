#!/usr/bin/env python3
"""BDB-025 Fig 5: Per-Category Radar/Spider Chart.

Generates a two-panel figure:
  Panel A -- Radar chart of 7 legacy categories for the top 4 agents.
  Panel B -- De novo vs Redesign comparison across 5 molecular subjects.

Outputs:
    results/analysis/fig5_radar_chart.png (300 dpi)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all

# ── Constants ────────────────────────────────────────────────────────────────

LEGACY_CATEGORIES = ["Binder", "SeqOpt", "CpxPrd", "ConfDv", "Backbone", "AbBody", "PPI"]
LEGACY_COUNTS = {
    "Binder": 13,
    "SeqOpt": 20,
    "CpxPrd": 16,
    "ConfDv": 13,
    "Backbone": 6,
    "AbBody": 4,
    "PPI": 4,
}

# Comprehensive task_type -> legacy category mapping.
# The load_results LEGACY_CATEGORY_MAP only covers old-style task types.
# Template-based task types (cfd_*, cpx_*, dnb_*, dnk_*, sqo_*) need
# explicit mapping to the 7 canonical categories.
_FULL_LEGACY_MAP: dict[str, str] = {
    # ── AbBody (4): antibody CDR optimisation tasks ──
    "antibody_optimization": "AbBody",
    # ── Binder (13): de novo binder design tasks ──
    "binder_design": "Binder",
    "peptide_design": "Binder",
    "dnb_ab": "Binder",       # de novo binder targeting antibody epitopes
    "dnb_enz": "Binder",      # de novo binder targeting enzymes
    # ── PPI (4): protein-protein interaction design ──
    "ppi_design": "PPI",
    # ── Backbone (6): de novo backbone/scaffold generation ──
    "scaffold_design": "Backbone",
    "dnk_str": "Backbone",    # de novo backbone (structure)
    # ── ConfDv (13): conformational diversity exploration ──
    "cfd_enz": "ConfDv",
    "cfd_flu": "ConfDv",
    "cfd_sig": "ConfDv",
    "cfd_str": "ConfDv",
    # ── CpxPrd (16): complex prediction tasks ──
    "cpx_enz": "CpxPrd",
    "cpx_sig": "CpxPrd",
    "cpx_str": "CpxPrd",
    # ── SeqOpt (20): sequence optimisation / redesign tasks ──
    "stability_optimization": "SeqOpt",
    "enzyme_design": "SeqOpt",
    "fluorescence_design": "SeqOpt",
    "sqo_ab": "SeqOpt",
    "sqo_enz": "SeqOpt",
    "sqo_flu": "SeqOpt",
    "sqo_sig": "SeqOpt",
    "sqo_str": "SeqOpt",
    "dnb_sig": "SeqOpt",     # de novo binder (signaling) -> treated as SeqOpt
}

TOP4_CONDITIONS = [
    "DeepSeek V3 user",
    "GPT-5 user",
    "Sonnet 4.5 user",
    "Hardcoded Pipeline",
]

AGENT_COLORS: dict[str, str] = {
    "DeepSeek V3 user": "#1f77b4",
    "GPT-5 user": "#ff7f0e",
    "Sonnet 4.5 user": "#2ca02c",
    "Hardcoded Pipeline": "#7f7f7f",
}

APPROACH_COLORS: dict[str, str] = {
    "de_novo": "#1f77b4",
    "redesign": "#ff7f0e",
}

SUBJECT_LABELS: dict[str, str] = {
    "antibody": "Antibody",
    "enzyme": "Enzyme",
    "binder": "Binder",
    "scaffold": "Scaffold",
    "fluorescent_protein": "Fluor. Protein",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _remap_legacy_category(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full task_type -> legacy_category mapping.

    The upstream ``load_all()`` only maps a subset of task types to the
    7 canonical legacy categories.  This function remaps the
    ``legacy_category`` column using the comprehensive ``_FULL_LEGACY_MAP``
    so that all 76 tasks have a valid category.

    Args:
        df: DataFrame from ``load_all()``.

    Returns:
        DataFrame with corrected ``legacy_category`` column.
    """
    df = df.copy()
    df["legacy_category"] = df["task_type"].map(_FULL_LEGACY_MAP).fillna(df["legacy_category"])
    return df


def _build_category_label(cat: str) -> str:
    """Build axis label with task count, e.g. 'Binder (13)'."""
    count = LEGACY_COUNTS.get(cat, "")
    return f"{cat} ({count})" if count else cat


def _close_polygon(values: list[float]) -> list[float]:
    """Append first value to close the radar polygon."""
    return values + values[:1]


def _radar_angles(n: int) -> list[float]:
    """Compute evenly spaced angles for *n* axes, closing the polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    return angles + angles[:1]


# ── Data computation ─────────────────────────────────────────────────────────


def compute_panel_a_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean total score per (condition, legacy_category) for the top 4 agents.

    Args:
        df: Full DataFrame from load_all(), already remapped.

    Returns:
        DataFrame pivoted with conditions as rows and legacy categories as columns.
    """
    subset = df[df["condition"].isin(TOP4_CONDITIONS)].copy()
    means = (
        subset.groupby(["condition", "legacy_category"], observed=True)["total"]
        .mean()
        .reset_index()
    )
    pivot = means.pivot_table(
        index="condition",
        columns="legacy_category",
        values="total",
        aggfunc="first",
        observed=True,
    )
    # Reorder columns and rows
    pivot = pivot.reindex(columns=LEGACY_CATEGORIES)
    pivot = pivot.reindex(TOP4_CONDITIONS)
    return pivot


def compute_panel_b_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean total score per (design_approach, molecular_subject) across all conditions.

    Args:
        df: Full DataFrame from load_all(), already remapped.

    Returns:
        DataFrame with approaches as rows and molecular subjects as columns.
    """
    approaches = ["de_novo", "redesign"]
    subjects = list(SUBJECT_LABELS.keys())

    subset = df[
        df["design_approach"].isin(approaches)
        & df["molecular_subject"].isin(subjects)
    ].copy()

    means = (
        subset.groupby(["design_approach", "molecular_subject"], observed=True)["total"]
        .mean()
        .reset_index()
    )
    pivot = means.pivot_table(
        index="design_approach",
        columns="molecular_subject",
        values="total",
        aggfunc="first",
        observed=True,
    )
    pivot = pivot.reindex(columns=subjects)
    pivot = pivot.reindex(approaches)
    return pivot


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw the 7-axis radar chart for top 4 agents.

    Args:
        ax: Polar matplotlib axes.
        data: Pivoted DataFrame (conditions x categories).
    """
    categories = list(data.columns)
    n = len(categories)
    angles = _radar_angles(n)
    labels = [_build_category_label(c) for c in categories]

    for condition in data.index:
        values = data.loc[condition].fillna(0).tolist()
        values_closed = _close_polygon(values)
        color = AGENT_COLORS.get(condition, "#888888")
        ax.plot(angles, values_closed, linewidth=1.8, label=condition, color=color)
        ax.fill(angles, values_closed, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    # Radial limits
    ax.set_ylim(0, 80)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=7, color="grey")

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.12),
        fontsize=8,
        framealpha=0.9,
    )
    ax.set_title("A.  Per-Category Profile (Top 4 Agents)", fontsize=11, pad=20)


def plot_panel_b(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw the 5-axis radar chart for de novo vs redesign.

    Args:
        ax: Polar matplotlib axes.
        data: Pivoted DataFrame (approaches x subjects).
    """
    subjects = list(data.columns)
    n = len(subjects)
    angles = _radar_angles(n)
    labels = [SUBJECT_LABELS.get(s, s) for s in subjects]

    for approach in data.index:
        values = data.loc[approach].fillna(0).tolist()
        values_closed = _close_polygon(values)
        color = APPROACH_COLORS.get(approach, "#888888")
        display_name = approach.replace("_", " ").title()
        ax.plot(angles, values_closed, linewidth=1.8, label=display_name, color=color)
        ax.fill(angles, values_closed, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    # Radial limits
    ax.set_ylim(0, 80)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=7, color="grey")

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.30, 1.12),
        fontsize=8,
        framealpha=0.9,
    )
    ax.set_title("B.  De Novo vs Redesign by Subject", fontsize=11, pad=20)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load data, compute means, and generate the two-panel radar figure."""
    print("Loading results...")
    df = load_all()
    print(
        f"  Loaded {len(df)} rows "
        f"({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)"
    )

    # Remap legacy_category to the 7 canonical names
    df = _remap_legacy_category(df)

    # Sanity check: verify all 7 categories are present
    mapped_cats = set(df["legacy_category"].unique())
    expected_cats = set(LEGACY_CATEGORIES)
    missing = expected_cats - mapped_cats
    if missing:
        print(f"  WARNING: missing legacy categories after remap: {missing}")
    unmapped = mapped_cats - expected_cats
    if unmapped:
        print(f"  WARNING: unmapped legacy categories: {unmapped}")

    # Show per-category task counts
    cat_counts = df.groupby("legacy_category")["task_id"].nunique()
    print("  Category task counts:")
    for cat in LEGACY_CATEGORIES:
        n = cat_counts.get(cat, 0)
        expected = LEGACY_COUNTS.get(cat, "?")
        marker = " *" if n != expected else ""
        print(f"    {cat:10s}: {n:3d}  (expected {expected}){marker}")

    # ── Compute data ─────────────────────────────────────────────────────────
    panel_a = compute_panel_a_data(df)
    panel_b = compute_panel_b_data(df)

    print("\n=== Panel A: Per-Category Means (Top 4 Agents) ===")
    print(panel_a.to_string(float_format="%.1f"))

    print("\n=== Panel B: De Novo vs Redesign by Subject ===")
    print(panel_b.to_string(float_format="%.1f"))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))

    ax_a = fig.add_subplot(121, polar=True)
    plot_panel_a(ax_a, panel_a)

    ax_b = fig.add_subplot(122, polar=True)
    plot_panel_b(ax_b, panel_b)

    fig.tight_layout(pad=3.0)

    out_dir = PROJECT_ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig5_radar_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
