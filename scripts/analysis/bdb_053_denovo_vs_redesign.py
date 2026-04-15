#!/usr/bin/env python3
"""BDB-053: De novo vs Redesign performance gap analysis.

Analyzes the performance gap between de_novo and redesign design approaches
across the 2x5 taxonomy (DesignApproach x MolecularSubject) using 76 tasks
evaluated under 9 conditions.

Outputs (saved to results/analysis/):
    - denovo_vs_redesign_table.csv   : Overall gap table (9 conditions)
    - per_subject_gap.csv            : Gap within each molecular_subject
    - methodology_bias_table.csv     : task_type distribution check
    - interaction_plot.png           : Agent x DesignApproach interaction
    - gap_summary.txt                : Summary findings
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Overall gap table
# ---------------------------------------------------------------------------

def overall_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each of the 9 conditions, compute mean total score for de_novo
    vs redesign tasks, plus the gap (de_novo - redesign).

    Returns a DataFrame saved as denovo_vs_redesign_table.csv.
    """
    grouped = (
        df.groupby(["condition", "design_approach"])["total"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    grouped = grouped.rename(columns={"de_novo": "de_novo_mean", "redesign": "redesign_mean"})
    grouped["gap"] = grouped["de_novo_mean"] - grouped["redesign_mean"]
    grouped = grouped.reset_index()
    grouped = grouped.sort_values("gap", ascending=False)
    return grouped


# ---------------------------------------------------------------------------
# 2. Per-molecular-subject gap
# ---------------------------------------------------------------------------

def per_subject_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Within each molecular_subject, compare de_novo vs redesign mean scores.

    Only includes cells that have tasks in both approaches.
    """
    rows = []
    for subject in sorted(df["molecular_subject"].unique()):
        sub_df = df[df["molecular_subject"] == subject]
        approaches_present = sub_df["design_approach"].unique()

        if "de_novo" not in approaches_present or "redesign" not in approaches_present:
            continue

        dn_mean = sub_df[sub_df["design_approach"] == "de_novo"]["total"].mean()
        rd_mean = sub_df[sub_df["design_approach"] == "redesign"]["total"].mean()
        dn_count = sub_df[sub_df["design_approach"] == "de_novo"]["task_id"].nunique()
        rd_count = sub_df[sub_df["design_approach"] == "redesign"]["task_id"].nunique()

        rows.append({
            "molecular_subject": subject,
            "de_novo_mean": round(dn_mean, 2),
            "redesign_mean": round(rd_mean, 2),
            "gap": round(dn_mean - rd_mean, 2),
            "de_novo_n_tasks": dn_count,
            "redesign_n_tasks": rd_count,
        })

    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Methodology bias check
# ---------------------------------------------------------------------------

def methodology_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Count task_type distribution within de_novo vs redesign.

    Check whether the task_type mix differs across design approaches,
    which could confound the gap analysis.
    """
    cross = pd.crosstab(
        df["design_approach"],
        df["task_type"],
        margins=True,
    )
    # Normalize rows to proportions (excluding the 'All' row for proportions)
    prop = cross.div(cross["All"], axis=0).drop(columns="All")
    prop = prop.reset_index()
    prop = prop.rename(columns={"design_approach": "approach"})

    # Also build a raw count table
    counts = cross.reset_index().rename(columns={"design_approach": "approach"})

    # Merge counts and proportions side by side
    result_rows = []
    for _, row in counts.iterrows():
        approach = row["approach"]
        prop_row = prop[prop["approach"] == approach].iloc[0] if approach in prop["approach"].values else None
        entry = {"approach": approach}
        for col in counts.columns:
            if col in ("approach", "All"):
                continue
            count_val = row[col]
            pct_val = prop_row[col] * 100 if prop_row is not None and col in prop_row.index else 0
            entry[f"{col}_count"] = int(count_val)
            entry[f"{col}_pct"] = round(pct_val, 1)
        entry["total"] = int(row["All"])
        result_rows.append(entry)

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# 4. Agent-specific strengths: interaction plot
# ---------------------------------------------------------------------------

def interaction_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Interaction plot: agent x design_approach.

    Lines for each condition, x-axis = de_novo / redesign, y-axis = mean score.
    """
    pivot = (
        df.groupby(["condition", "design_approach"])["total"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    conditions = pivot.index.tolist()
    x_positions = [0, 1]
    x_labels = ["de_novo", "redesign"]

    for condition in conditions:
        dn_val = pivot.loc[condition, "de_novo"] if "de_novo" in pivot.columns else np.nan
        rd_val = pivot.loc[condition, "redesign"] if "redesign" in pivot.columns else np.nan
        ax.plot(
            x_positions,
            [dn_val, rd_val],
            marker="o",
            markersize=8,
            linewidth=2,
            label=condition,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylabel("Mean Total Score", fontsize=13)
    ax.set_xlabel("Design Approach", fontsize=13)
    ax.set_title("Agent x Design Approach Interaction", fontsize=15, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        frameon=True,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.3, 1.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Gap summary text
# ---------------------------------------------------------------------------

def write_gap_summary(
    df: pd.DataFrame,
    gap_table: pd.DataFrame,
    subject_gap: pd.DataFrame,
    bias_table: pd.DataFrame,
    out_path: Path,
) -> str:
    """Write a plain-text summary of the de novo vs redesign gap analysis."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("BDB-053: De Novo vs Redesign Performance Gap Analysis")
    lines.append("=" * 70)
    lines.append("")

    # Task counts
    dn_tasks = df[df["design_approach"] == "de_novo"]["task_id"].nunique()
    rd_tasks = df[df["design_approach"] == "redesign"]["task_id"].nunique()
    lines.append(f"Task distribution: de_novo={dn_tasks}, redesign={rd_tasks}")

    # Molecular subject counts
    subject_counts = df.drop_duplicates("task_id").groupby("molecular_subject").size()
    lines.append(f"Molecular subjects: {dict(subject_counts)}")
    lines.append("")

    # Overall gap
    lines.append("--- 1. Overall Gap (de_novo - redesign) by Condition ---")
    for _, row in gap_table.iterrows():
        lines.append(
            f"  {row['condition']:35s}  "
            f"de_novo={row['de_novo_mean']:5.1f}  "
            f"redesign={row['redesign_mean']:5.1f}  "
            f"gap={row['gap']:+5.1f}"
        )

    avg_gap = gap_table["gap"].mean()
    lines.append(f"\n  Average gap across conditions: {avg_gap:+.1f}")
    if avg_gap > 0:
        lines.append("  --> De novo tasks score HIGHER on average.")
    elif avg_gap < 0:
        lines.append("  --> Redesign tasks score HIGHER on average.")
    else:
        lines.append("  --> No systematic gap.")
    lines.append("")

    # Per-subject gap
    lines.append("--- 2. Per-Molecular-Subject Gap ---")
    if len(subject_gap) == 0:
        lines.append("  No subjects have tasks in both approaches.")
    else:
        for _, row in subject_gap.iterrows():
            lines.append(
                f"  {row['molecular_subject']:25s}  "
                f"de_novo={row['de_novo_mean']:5.1f} (n={row['de_novo_n_tasks']})  "
                f"redesign={row['redesign_mean']:5.1f} (n={row['redesign_n_tasks']})  "
                f"gap={row['gap']:+5.1f}"
            )
    lines.append("")

    # Methodology bias
    lines.append("--- 3. Methodology Bias Check (task_type distribution) ---")
    lines.append("  Question: Does task_type distribution differ between approaches?")
    unique_types_dn = (
        df[df["design_approach"] == "de_novo"]["task_type"].value_counts().to_dict()
    )
    unique_types_rd = (
        df[df["design_approach"] == "redesign"]["task_type"].value_counts().to_dict()
    )
    lines.append(f"  de_novo task_types:  {unique_types_dn}")
    lines.append(f"  redesign task_types: {unique_types_rd}")

    # Check overlap
    dn_types = set(unique_types_dn.keys())
    rd_types = set(unique_types_rd.keys())
    shared = dn_types & rd_types
    dn_only = dn_types - rd_types
    rd_only = rd_types - dn_types
    lines.append(f"  Shared task_types:       {shared if shared else 'none'}")
    lines.append(f"  de_novo-only types:      {dn_only if dn_only else 'none'}")
    lines.append(f"  redesign-only types:     {rd_only if rd_only else 'none'}")
    if dn_only or rd_only:
        lines.append(
            "  WARNING: Task_type distributions differ -- gap may partly reflect "
            "methodology differences, not just approach difficulty."
        )
    else:
        lines.append("  Task_type distributions overlap fully -- minimal methodology bias.")
    lines.append("")

    # Interaction summary
    lines.append("--- 4. Agent-Specific Interaction ---")
    # Find which condition has the largest and smallest gap
    if len(gap_table) > 0:
        max_gap_row = gap_table.loc[gap_table["gap"].idxmax()]
        min_gap_row = gap_table.loc[gap_table["gap"].idxmin()]
        lines.append(
            f"  Largest gap:  {max_gap_row['condition']} "
            f"(gap={max_gap_row['gap']:+.1f})"
        )
        lines.append(
            f"  Smallest gap: {min_gap_row['condition']} "
            f"(gap={min_gap_row['gap']:+.1f})"
        )
        # Check if any condition reverses the direction
        positive = (gap_table["gap"] > 0).sum()
        negative = (gap_table["gap"] < 0).sum()
        zero = (gap_table["gap"] == 0).sum()
        lines.append(
            f"  Direction: {positive} conditions favor de_novo, "
            f"{negative} favor redesign, {zero} neutral"
        )
        if positive > 0 and negative > 0:
            lines.append(
                "  NOTE: Gap direction is inconsistent across conditions -- "
                "agent-approach interaction is present."
            )
    lines.append("")

    lines.append("--- 5. Key Takeaways ---")
    lines.append(f"  * Overall mean gap: {avg_gap:+.1f} points")
    if len(subject_gap) > 0:
        largest_subj = subject_gap.iloc[0]
        lines.append(
            f"  * Largest per-subject gap: {largest_subj['molecular_subject']} "
            f"({largest_subj['gap']:+.1f})"
        )
    lines.append(f"  * See interaction_plot.png for agent-specific patterns")
    lines.append("")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    out_path.write_text(summary + "\n")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    df = load_all()
    print(f"  Loaded {len(df)} rows "
          f"({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)")

    # Filter to known approaches
    df = df[df["design_approach"].isin(["de_novo", "redesign"])].copy()
    dn_tasks = df[df["design_approach"] == "de_novo"]["task_id"].nunique()
    rd_tasks = df[df["design_approach"] == "redesign"]["task_id"].nunique()
    print(f"  de_novo tasks: {dn_tasks}, redesign tasks: {rd_tasks}")

    # Subject distribution
    subject_dist = df.drop_duplicates("task_id").groupby("molecular_subject").size()
    print(f"  Molecular subject counts: {dict(subject_dist)}")
    print()

    # --- Analysis 1: Overall gap table ---
    print("1. Overall gap table...")
    gap_table = overall_gap_table(df)
    gap_table.to_csv(OUTPUT_DIR / "denovo_vs_redesign_table.csv", index=False, float_format="%.2f")
    print(gap_table.to_string(index=False))
    print()

    # --- Analysis 2: Per-subject gap ---
    print("2. Per-molecular-subject gap...")
    subj_gap = per_subject_gap(df)
    subj_gap.to_csv(OUTPUT_DIR / "per_subject_gap.csv", index=False, float_format="%.2f")
    print(subj_gap.to_string(index=False))
    print()

    # --- Analysis 3: Methodology bias ---
    print("3. Methodology bias check...")
    bias = methodology_bias(df)
    bias.to_csv(OUTPUT_DIR / "methodology_bias_table.csv", index=False)
    print(bias.to_string(index=False))
    print()

    # --- Analysis 4: Interaction plot ---
    print("4. Generating interaction plot...")
    plot_path = OUTPUT_DIR / "interaction_plot.png"
    interaction_plot(df, plot_path)
    print(f"  Saved to {plot_path}")
    print()

    # --- Analysis 5: Gap summary ---
    print("5. Writing gap summary...")
    summary_path = OUTPUT_DIR / "gap_summary.txt"
    summary = write_gap_summary(df, gap_table, subj_gap, bias, summary_path)
    print(summary)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
