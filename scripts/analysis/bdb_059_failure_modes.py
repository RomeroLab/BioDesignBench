#!/usr/bin/env python3
"""BDB-059: Failure Mode Categorization Across All Agents.

Categorizes and visualizes failure modes across the 9 benchmark conditions
using observable signals (zero-score components, tool call counts, error
patterns) to understand *why* agents fail, not just *how much*.

Analyses:
  1. Per-component zero-score frequency heatmap
  2. Error type classification and breakdown by condition
  3. Gemini-specific failure deep-dive
  4. Hardcoded Pipeline failure analysis
  5. Model-specific failure profiles (top 3 per condition)

Outputs (results/analysis/):
  - zero_score_heatmap.png
  - error_type_breakdown.csv
  - gemini_failure_analysis.txt
  - hardcoded_failure_analysis.txt
  - failure_profile_summary.csv

Usage:
    python -m scripts.analysis.bdb_059_failure_modes
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all  # noqa: E402

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]
N_TASKS = 76

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nature-style plot defaults
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ---------------------------------------------------------------------------
# 1. Per-component zero-score frequency heatmap
# ---------------------------------------------------------------------------

def compute_zero_score_pct(df: pd.DataFrame) -> pd.DataFrame:
    """For each condition x component, compute % of tasks with score == 0.

    Args:
        df: DataFrame with 'condition' column and COMPONENTS columns.

    Returns:
        DataFrame with conditions as rows, components as columns,
        values as zero-score percentages (0-100).
    """
    rows = []
    for condition in df["condition"].cat.categories:
        cond_df = df[df["condition"] == condition]
        n = len(cond_df)
        if n == 0:
            continue
        entry = {"condition": condition}
        for comp in COMPONENTS:
            zero_count = (cond_df[comp] == 0).sum()
            entry[comp] = round(zero_count / n * 100, 1)
        rows.append(entry)
    return pd.DataFrame(rows).set_index("condition")


def plot_zero_score_heatmap(zero_pct: pd.DataFrame, out_path: Path) -> None:
    """Plot annotated heatmap of zero-score percentages.

    Args:
        zero_pct: DataFrame from compute_zero_score_pct().
        out_path: Path to save the PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        zero_pct,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Zero-Score %"},
        annot_kws={"size": 9},
    )

    ax.set_title(
        "Zero-Score Frequency by Condition and Component",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Scoring Component", fontsize=11)
    ax.set_ylabel("Condition", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Error type classification
# ---------------------------------------------------------------------------

ERROR_TYPES = [
    "no_tool_call",
    "tool_failure",
    "no_output",
    "wrong_approach",
    "low_quality",
    "partial_success",
    "success",
]


def classify_error(row: pd.Series) -> str:
    """Classify a single task result into an error type.

    Classification is ordered from most severe to least severe. The first
    matching condition wins.

    Args:
        row: A single row from the DataFrame.

    Returns:
        One of the ERROR_TYPES strings.
    """
    if row["total"] >= 30:
        return "success"
    if row["num_tool_calls"] == 0:
        return "no_tool_call"
    if row["failed_tools"] > 0:
        return "tool_failure"
    if row["quality"] == 0 and row["feasibility"] == 0 and row["num_tool_calls"] > 0:
        return "no_output"
    if row["approach"] == 0 and row["orchestration"] == 0:
        return "wrong_approach"
    if row["quality"] < 5 and row["feasibility"] > 0:
        return "low_quality"
    if row["total"] > 0 and row["total"] < 30:
        return "partial_success"
    # Fallback: score is exactly 0 with no tool calls already handled
    return "no_output"


def compute_error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every row and count error types per condition.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with conditions as rows, error types as columns,
        values as counts.
    """
    df = df.copy()
    df["error_type"] = df.apply(classify_error, axis=1)

    pivot = pd.crosstab(df["condition"], df["error_type"])

    # Ensure all error types are present as columns
    for et in ERROR_TYPES:
        if et not in pivot.columns:
            pivot[et] = 0

    # Reorder columns
    pivot = pivot[ERROR_TYPES]
    return pivot


# ---------------------------------------------------------------------------
# 3. Gemini special analysis
# ---------------------------------------------------------------------------

def gemini_failure_analysis(df: pd.DataFrame, out_path: Path) -> str:
    """Deep-dive into Gemini 2.5 Pro failure patterns.

    Args:
        df: Full DataFrame from load_all().
        out_path: Path to save the text report.

    Returns:
        The report text.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("BDB-059: Gemini 2.5 Pro Failure Analysis")
    lines.append("=" * 70)
    lines.append("")

    gemini_df = df[df["llm"] == "Gemini 2.5 Pro"].copy()
    if gemini_df.empty:
        lines.append("No Gemini 2.5 Pro data found.")
        report = "\n".join(lines)
        out_path.write_text(report + "\n")
        return report

    # Overall stats
    for mode in ["benchmark", "user"]:
        mode_df = gemini_df[gemini_df["mode"] == mode]
        if mode_df.empty:
            continue
        cond_name = mode_df["condition"].iloc[0]
        lines.append(f"--- {cond_name} ---")
        lines.append(f"  N tasks: {len(mode_df)}")
        lines.append(f"  Mean total: {mode_df['total'].mean():.1f}")
        lines.append(f"  Median total: {mode_df['total'].median():.1f}")
        lines.append(f"  Std total: {mode_df['total'].std():.1f}")
        lines.append("")

        # Zero-score component counts
        lines.append("  Zero-score component counts:")
        for comp in COMPONENTS:
            zero_n = (mode_df[comp] == 0).sum()
            pct = zero_n / len(mode_df) * 100
            lines.append(f"    {comp:<15s}: {zero_n:3d} / {len(mode_df)} ({pct:.1f}%)")
        lines.append("")

        # Component means
        lines.append("  Component means:")
        for comp in COMPONENTS:
            lines.append(f"    {comp:<15s}: {mode_df[comp].mean():.2f}")
        lines.append("")

        # Tool usage stats
        lines.append("  Tool usage:")
        lines.append(f"    Mean tool calls: {mode_df['num_tool_calls'].mean():.1f}")
        lines.append(f"    Mean failed tools: {mode_df['failed_tools'].mean():.1f}")
        no_tools = (mode_df["num_tool_calls"] == 0).sum()
        lines.append(f"    Tasks with zero tool calls: {no_tools}")
        lines.append("")

        # Error type distribution
        error_labels = mode_df.apply(classify_error, axis=1)
        error_counts = error_labels.value_counts()
        lines.append("  Error type distribution:")
        for et in ERROR_TYPES:
            count = error_counts.get(et, 0)
            pct = count / len(mode_df) * 100
            lines.append(f"    {et:<20s}: {count:3d} ({pct:.1f}%)")
        lines.append("")

    # Successful tasks (total > 40) vs failed tasks
    lines.append("--- Successful vs Failed Tasks ---")
    success_df = gemini_df[gemini_df["total"] > 40]
    failed_df = gemini_df[gemini_df["total"] <= 40]

    lines.append(f"  Successful tasks (total > 40): {len(success_df)}")
    if not success_df.empty:
        lines.append("  Successful task IDs and scores:")
        for _, row in success_df.sort_values("total", ascending=False).iterrows():
            lines.append(
                f"    {row['task_id']:<25s}  {row['condition']:<30s}  total={row['total']:.1f}"
            )
        lines.append("")

        # What categories succeed?
        lines.append("  Successful task categories:")
        cat_counts = success_df["legacy_category"].value_counts()
        for cat, count in cat_counts.items():
            lines.append(f"    {cat}: {count}")
    else:
        lines.append("  No tasks scored above 40.")
    lines.append("")

    lines.append(f"  Failed tasks (total <= 40): {len(failed_df)}")
    if not failed_df.empty:
        lines.append("  Failed task categories:")
        cat_counts = failed_df["legacy_category"].value_counts()
        for cat, count in cat_counts.items():
            lines.append(f"    {cat}: {count}")
    lines.append("")

    # Compare component profiles
    lines.append("--- Component Profile Comparison ---")
    lines.append(f"  {'Component':<15s} {'Success mean':>14s} {'Failed mean':>14s} {'Delta':>8s}")
    lines.append("  " + "-" * 55)
    for comp in COMPONENTS:
        s_mean = success_df[comp].mean() if not success_df.empty else 0.0
        f_mean = failed_df[comp].mean() if not failed_df.empty else 0.0
        delta = s_mean - f_mean
        lines.append(f"  {comp:<15s} {s_mean:>14.2f} {f_mean:>14.2f} {delta:>+8.2f}")
    lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    return report


# ---------------------------------------------------------------------------
# 4. Hardcoded Pipeline failure analysis
# ---------------------------------------------------------------------------

def hardcoded_failure_analysis(df: pd.DataFrame, out_path: Path) -> str:
    """Analyze tasks where the Hardcoded Pipeline scores very low.

    Args:
        df: Full DataFrame from load_all().
        out_path: Path to save the text report.

    Returns:
        The report text.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("BDB-059: Hardcoded Pipeline Failure Analysis")
    lines.append("=" * 70)
    lines.append("")

    hp_df = df[df["condition"] == "Hardcoded Pipeline"].copy()
    if hp_df.empty:
        lines.append("No Hardcoded Pipeline data found.")
        report = "\n".join(lines)
        out_path.write_text(report + "\n")
        return report

    # Overall stats
    lines.append(f"Total tasks: {len(hp_df)}")
    lines.append(f"Mean total: {hp_df['total'].mean():.1f}")
    lines.append(f"Median total: {hp_df['total'].median():.1f}")
    lines.append("")

    # Low-scoring tasks (total < 20)
    low_df = hp_df[hp_df["total"] < 20].sort_values("total")
    lines.append(f"Tasks with total < 20: {len(low_df)} / {len(hp_df)}")
    lines.append("")

    if not low_df.empty:
        # Category distribution of low-scoring tasks
        lines.append("--- Legacy Category Distribution of Low-Scoring Tasks ---")
        cat_counts = low_df["legacy_category"].value_counts()
        for cat, count in cat_counts.items():
            pct = count / len(low_df) * 100
            lines.append(f"  {cat:<15s}: {count:3d} ({pct:.1f}%)")
        lines.append("")

        # Design approach distribution
        lines.append("--- Design Approach Distribution of Low-Scoring Tasks ---")
        approach_counts = low_df["design_approach"].value_counts()
        for approach, count in approach_counts.items():
            pct = count / len(low_df) * 100
            lines.append(f"  {approach:<15s}: {count:3d} ({pct:.1f}%)")
        lines.append("")

        # Molecular subject distribution
        lines.append("--- Molecular Subject Distribution of Low-Scoring Tasks ---")
        subject_counts = low_df["molecular_subject"].value_counts()
        for subject, count in subject_counts.items():
            pct = count / len(low_df) * 100
            lines.append(f"  {subject:<20s}: {count:3d} ({pct:.1f}%)")
        lines.append("")

        # Detailed list
        lines.append("--- Individual Low-Scoring Tasks ---")
        lines.append(
            f"  {'task_id':<25s} {'total':>6s} {'appr':>5s} {'orch':>5s} "
            f"{'qual':>5s} {'feas':>5s} {'novl':>5s} {'divr':>5s} {'legacy_cat':<15s}"
        )
        lines.append("  " + "-" * 95)
        for _, row in low_df.iterrows():
            lines.append(
                f"  {row['task_id']:<25s} {row['total']:>6.1f} {row['approach']:>5.1f} "
                f"{row['orchestration']:>5.1f} {row['quality']:>5.1f} "
                f"{row['feasibility']:>5.1f} {row['novelty']:>5.1f} "
                f"{row['diversity']:>5.1f} {row['legacy_category']:<15s}"
            )
        lines.append("")

        # Component breakdown for low-scoring tasks
        lines.append("--- Component Zero-Score Rates in Low-Scoring Tasks ---")
        for comp in COMPONENTS:
            zero_n = (low_df[comp] == 0).sum()
            pct = zero_n / len(low_df) * 100
            lines.append(f"  {comp:<15s}: {zero_n:3d} / {len(low_df)} ({pct:.1f}%)")
        lines.append("")

    # Compare low vs high scoring tasks
    high_df = hp_df[hp_df["total"] >= 20]
    lines.append("--- Low (<20) vs High (>=20) Comparison ---")
    lines.append(f"  Low-scoring:  n={len(low_df)}, mean={low_df['total'].mean():.1f}")
    lines.append(f"  High-scoring: n={len(high_df)}, mean={high_df['total'].mean():.1f}")
    lines.append("")

    if not low_df.empty and not high_df.empty:
        lines.append(f"  {'Component':<15s} {'Low mean':>10s} {'High mean':>10s} {'Delta':>8s}")
        lines.append("  " + "-" * 45)
        for comp in COMPONENTS:
            low_mean = low_df[comp].mean()
            high_mean = high_df[comp].mean()
            delta = high_mean - low_mean
            lines.append(
                f"  {comp:<15s} {low_mean:>10.2f} {high_mean:>10.2f} {delta:>+8.2f}"
            )
    lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    return report


# ---------------------------------------------------------------------------
# 5. Model-specific failure profiles
# ---------------------------------------------------------------------------

def compute_failure_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """For each condition, find the top 3 most common failure types.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with columns: condition, rank, error_type, count, pct.
    """
    df = df.copy()
    df["error_type"] = df.apply(classify_error, axis=1)

    rows = []
    for condition in df["condition"].cat.categories:
        cond_df = df[df["condition"] == condition]
        n = len(cond_df)
        if n == 0:
            continue

        counts = cond_df["error_type"].value_counts()
        for rank, (error_type, count) in enumerate(counts.head(3).items(), start=1):
            rows.append(
                {
                    "condition": condition,
                    "rank": rank,
                    "error_type": error_type,
                    "count": count,
                    "pct": round(count / n * 100, 1),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None) -> None:
    """Run all failure mode analyses and save outputs.

    Args:
        output_dir: Directory to save results. Defaults to results/analysis/.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_all()
    n_tasks = df["task_id"].nunique()
    n_conditions = df["condition"].nunique()
    print(f"Loaded {len(df)} rows ({n_tasks} tasks x {n_conditions} conditions)")
    print()

    # ── 1. Zero-score frequency heatmap ──────────────────────────────────
    print("=" * 60)
    print("1. ZERO-SCORE FREQUENCY HEATMAP")
    print("=" * 60)

    zero_pct = compute_zero_score_pct(df)
    print(zero_pct.to_string())
    print()

    heatmap_path = output_dir / "zero_score_heatmap.png"
    plot_zero_score_heatmap(zero_pct, heatmap_path)
    print(f"Saved: {heatmap_path}")
    print()

    # ── 2. Error type classification ─────────────────────────────────────
    print("=" * 60)
    print("2. ERROR TYPE BREAKDOWN")
    print("=" * 60)

    error_breakdown = compute_error_breakdown(df)
    print(error_breakdown.to_string())
    print()

    breakdown_path = output_dir / "error_type_breakdown.csv"
    error_breakdown.to_csv(breakdown_path)
    print(f"Saved: {breakdown_path}")
    print()

    # ── 3. Gemini failure analysis ───────────────────────────────────────
    print("=" * 60)
    print("3. GEMINI 2.5 PRO FAILURE ANALYSIS")
    print("=" * 60)

    gemini_path = output_dir / "gemini_failure_analysis.txt"
    gemini_report = gemini_failure_analysis(df, gemini_path)
    print(gemini_report)
    print(f"Saved: {gemini_path}")
    print()

    # ── 4. Hardcoded Pipeline failure analysis ───────────────────────────
    print("=" * 60)
    print("4. HARDCODED PIPELINE FAILURE ANALYSIS")
    print("=" * 60)

    hardcoded_path = output_dir / "hardcoded_failure_analysis.txt"
    hardcoded_report = hardcoded_failure_analysis(df, hardcoded_path)
    print(hardcoded_report)
    print(f"Saved: {hardcoded_path}")
    print()

    # ── 5. Failure profiles summary ──────────────────────────────────────
    print("=" * 60)
    print("5. MODEL-SPECIFIC FAILURE PROFILES")
    print("=" * 60)

    profiles = compute_failure_profiles(df)
    print(profiles.to_string(index=False))
    print()

    profiles_path = output_dir / "failure_profile_summary.csv"
    profiles.to_csv(profiles_path, index=False)
    print(f"Saved: {profiles_path}")
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Overall error distribution
    df_classified = df.copy()
    df_classified["error_type"] = df_classified.apply(classify_error, axis=1)
    overall_counts = df_classified["error_type"].value_counts()
    total_rows = len(df_classified)
    print("  Overall error type distribution:")
    for et in ERROR_TYPES:
        count = overall_counts.get(et, 0)
        pct = count / total_rows * 100
        print(f"    {et:<20s}: {count:4d} ({pct:.1f}%)")
    print()

    # Worst component (highest zero-score rate overall)
    overall_zero = {}
    for comp in COMPONENTS:
        overall_zero[comp] = (df[comp] == 0).sum() / len(df) * 100
    worst_comp = max(overall_zero, key=overall_zero.get)  # type: ignore[arg-type]
    print(
        f"  Highest overall zero-score rate: '{worst_comp}' "
        f"({overall_zero[worst_comp]:.1f}%)"
    )

    best_comp = min(overall_zero, key=overall_zero.get)  # type: ignore[arg-type]
    print(
        f"  Lowest overall zero-score rate:  '{best_comp}' "
        f"({overall_zero[best_comp]:.1f}%)"
    )
    print()

    print(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
