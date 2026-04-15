#!/usr/bin/env python3
"""BDB-056: AF2 Post-Evaluation Metrics Distributions.

Analyzes AlphaFold2 quality metrics across 9 benchmark conditions,
including quality score distributions, tier breakdowns, fraction-level
analysis of pLDDT/pTM/ipTM/i_pAE, and scoring validation (outlier
identification).

Outputs (saved to results/analysis/):
    - af2_violin_quality.png        : Quality score (0-35) violin plots per condition
    - af2_tier_breakdown.png        : Grouped bar chart of mean tier_a/b/c per condition
    - quality_vs_total_scatter.png  : Quality vs total score scatter with Pearson r
    - af2_fraction_violins.png      : pLDDT/pTM/ipTM/i_pAE fraction violin plots
    - scoring_validation_summary.txt: Outlier identification (high total/low quality, etc.)

Usage:
    python -m scripts.analysis.bdb_056_af2_metrics
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
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style defaults (Nature-style, consistent with other BDB scripts)
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

# Condition display order (short labels for x-axis)
CONDITION_SHORT = {
    "DeepSeek V3 benchmark": "DS-V3\nbm",
    "DeepSeek V3 user": "DS-V3\nus",
    "GPT-5 benchmark": "GPT-5\nbm",
    "GPT-5 user": "GPT-5\nus",
    "Sonnet 4.5 benchmark": "Son4.5\nbm",
    "Sonnet 4.5 user": "Son4.5\nus",
    "Gemini 2.5 Pro benchmark": "Gem2.5\nbm",
    "Gemini 2.5 Pro user": "Gem2.5\nus",
    "Hardcoded Pipeline": "Hard-\ncoded",
}

# Color palette: one color per LLM, lighter shade for benchmark, darker for user
CONDITION_COLORS = {
    "DeepSeek V3 benchmark": "#7bafd4",
    "DeepSeek V3 user": "#1f77b4",
    "GPT-5 benchmark": "#ffbf80",
    "GPT-5 user": "#ff7f0e",
    "Sonnet 4.5 benchmark": "#7fcc7f",
    "Sonnet 4.5 user": "#2ca02c",
    "Gemini 2.5 Pro benchmark": "#e88e8e",
    "Gemini 2.5 Pro user": "#d62728",
    "Hardcoded Pipeline": "#9467bd",
}


def _short_label(condition: str) -> str:
    """Return a short multiline label for x-axis ticks."""
    return CONDITION_SHORT.get(condition, condition)


# ---------------------------------------------------------------------------
# 1. Quality score violin plots (0-35)
# ---------------------------------------------------------------------------


def plot_quality_violins(df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side violin plots of quality scores for all 9 conditions.

    Args:
        df: DataFrame with 'condition' and 'quality' columns.
        out_path: Path to save the PNG figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    conditions = df["condition"].cat.categories.tolist()

    # Map condition to color via hue
    color_map = {c: CONDITION_COLORS.get(c, "#888888") for c in conditions}
    sns.violinplot(
        data=df,
        x="condition",
        y="quality",
        hue="condition",
        order=conditions,
        hue_order=conditions,
        palette=color_map,
        inner="box",
        linewidth=0.8,
        saturation=0.85,
        ax=ax,
        cut=0,
        legend=False,
    )

    # Overlay individual points
    sns.stripplot(
        data=df,
        x="condition",
        y="quality",
        order=conditions,
        color="black",
        alpha=0.15,
        size=2,
        jitter=0.15,
        ax=ax,
    )

    # Annotate medians
    for i, cond in enumerate(conditions):
        subset = df[df["condition"] == cond]["quality"]
        median_val = subset.median()
        ax.text(
            i,
            median_val + 0.8,
            f"{median_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([_short_label(c) for c in conditions], fontsize=8)
    ax.set_ylabel("Quality Score (0-35)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(
        "AF2 Quality Score Distribution by Condition",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(-1, 37)
    ax.axhline(y=35, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Quality tier breakdown (grouped bar chart)
# ---------------------------------------------------------------------------


def plot_tier_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of mean tier_a, tier_b, tier_c per condition.

    Tiers represent quality sub-scores that sum to the quality score:
        tier_a: structure quality (pLDDT, pTM)
        tier_b: interface quality (ipTM, i_pAE) or physics metrics
        tier_c: advanced metrics

    Args:
        df: DataFrame with 'condition', 'tier_a', 'tier_b', 'tier_c' columns.
        out_path: Path to save the PNG figure.
    """
    conditions = df["condition"].cat.categories.tolist()
    tier_means = df.groupby("condition", observed=True)[["tier_a", "tier_b", "tier_c"]].mean()
    tier_means = tier_means.loc[conditions]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_a = ax.bar(
        x - width,
        tier_means["tier_a"],
        width,
        label="Tier A (structure)",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_b = ax.bar(
        x,
        tier_means["tier_b"],
        width,
        label="Tier B (interface/physics)",
        color="#DD8452",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_c = ax.bar(
        x + width,
        tier_means["tier_c"],
        width,
        label="Tier C (advanced)",
        color="#55A868",
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate bar values
    for bars in [bars_a, bars_b, bars_c]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.15,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(c) for c in conditions], fontsize=8)
    ax.set_ylabel("Mean Tier Score", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(
        "Quality Tier Breakdown by Condition",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Quality vs Total score scatter
# ---------------------------------------------------------------------------


def plot_quality_vs_total(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot: x=quality, y=total, colored by condition, with Pearson r.

    Args:
        df: DataFrame with 'quality', 'total', 'condition' columns.
        out_path: Path to save the PNG figure.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    conditions = df["condition"].cat.categories.tolist()

    for cond in conditions:
        subset = df[df["condition"] == cond]
        color = CONDITION_COLORS.get(cond, "#888888")
        ax.scatter(
            subset["quality"],
            subset["total"],
            c=color,
            label=_short_label(cond).replace("\n", " "),
            alpha=0.55,
            s=25,
            edgecolors="white",
            linewidth=0.3,
        )

    # Overall Pearson r
    valid = df[["quality", "total"]].dropna()
    r_val, p_val = stats.pearsonr(valid["quality"], valid["total"])

    # Regression line
    slope, intercept = np.polyfit(valid["quality"], valid["total"], 1)
    x_line = np.linspace(valid["quality"].min(), valid["quality"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.0, alpha=0.6)

    # Annotation
    p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r_val:.3f}\n{p_str}\nn = {len(valid)}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax.set_xlabel("Quality Score (0-35)", fontsize=11)
    ax.set_ylabel("Total Score (0-100)", fontsize=11)
    ax.set_title(
        "Quality Score vs Total Score",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        fontsize=7,
        frameon=True,
        loc="lower right",
        ncol=2,
        markerscale=1.2,
    )
    ax.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Fraction-level violin plots (pLDDT, pTM, ipTM, i_pAE)
# ---------------------------------------------------------------------------


def plot_fraction_violins(df: pd.DataFrame, out_path: Path) -> None:
    """Violin plots for each AF2 fraction metric per condition.

    Creates a 2x2 subplot grid for pLDDT_frac, pTM_frac, ipTM_frac, i_pAE_frac.
    Only plots data where the fraction is not null.

    Args:
        df: DataFrame with condition and fraction columns.
        out_path: Path to save the PNG figure.
    """
    fraction_cols = [
        ("pLDDT_frac", "pLDDT Fraction"),
        ("pTM_frac", "pTM Fraction"),
        ("ipTM_frac", "ipTM Fraction"),
        ("i_pAE_frac", "i_pAE Fraction"),
    ]

    conditions = df["condition"].cat.categories.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    for idx, (col, title) in enumerate(fraction_cols):
        ax = axes[idx]

        # Filter to non-null values
        plot_df = df[df[col].notna()].copy()

        if len(plot_df) == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="grey",
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        n_total = len(plot_df)
        n_conditions = plot_df["condition"].nunique()

        color_map = {c: CONDITION_COLORS.get(c, "#888888") for c in conditions}
        sns.violinplot(
            data=plot_df,
            x="condition",
            y=col,
            hue="condition",
            order=conditions,
            hue_order=conditions,
            palette=color_map,
            inner="box",
            linewidth=0.7,
            saturation=0.85,
            ax=ax,
            cut=0,
            legend=False,
        )

        # Per-condition median annotation
        for i, cond in enumerate(conditions):
            subset = plot_df[plot_df["condition"] == cond][col]
            if len(subset) > 0:
                median_val = subset.median()
                ax.text(
                    i,
                    median_val + 0.02,
                    f"{median_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                    color="#333333",
                )

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([_short_label(c) for c in conditions], fontsize=7)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_title(
            f"{title} (n={n_total}, {n_conditions} conditions)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "AF2 Metric Fraction Distributions by Condition",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Outlier identification (scoring validation)
# ---------------------------------------------------------------------------


def identify_outliers(df: pd.DataFrame, out_path: Path) -> str:
    """Identify scoring outliers: high total / low quality and vice versa.

    Uses IQR-based approach on the residuals of quality ~ total relationship.

    Args:
        df: DataFrame with 'quality', 'total', 'task_id', 'condition' columns.
        out_path: Path to save the text summary.

    Returns:
        The summary text.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("BDB-056: Scoring Validation Summary -- Outlier Identification")
    lines.append("=" * 72)
    lines.append("")

    valid = df[["task_id", "condition", "quality", "total", "tier_a", "tier_b", "tier_c"]].dropna(
        subset=["quality", "total"]
    )

    # --- Section 1: Descriptive statistics ---
    lines.append("--- 1. Descriptive Statistics ---")
    lines.append(f"  Total records: {len(valid)}")
    lines.append(f"  Unique tasks:  {valid['task_id'].nunique()}")
    lines.append(f"  Conditions:    {valid['condition'].nunique()}")
    lines.append("")
    lines.append(f"  {'Metric':<15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Q1':>8s} {'Med':>8s} {'Q3':>8s} {'Max':>8s}")
    lines.append("  " + "-" * 75)
    for col in ["quality", "total"]:
        desc = valid[col].describe()
        lines.append(
            f"  {col:<15s} "
            f"{desc['mean']:>8.2f} "
            f"{desc['std']:>8.2f} "
            f"{desc['min']:>8.2f} "
            f"{desc['25%']:>8.2f} "
            f"{desc['50%']:>8.2f} "
            f"{desc['75%']:>8.2f} "
            f"{desc['max']:>8.2f}"
        )
    lines.append("")

    # --- Section 2: Correlation ---
    r_val, p_val = stats.pearsonr(valid["quality"], valid["total"])
    rho_val, rho_p = stats.spearmanr(valid["quality"], valid["total"])
    lines.append("--- 2. Quality-Total Correlation ---")
    lines.append(f"  Pearson r  = {r_val:.4f}  (p = {p_val:.2e})")
    lines.append(f"  Spearman rho = {rho_val:.4f}  (p = {rho_p:.2e})")
    lines.append("")

    # --- Section 3: Residual-based outliers ---
    lines.append("--- 3. Residual-Based Outlier Detection ---")
    lines.append("  Method: Fit quality ~ total OLS, flag residuals > 1.5*IQR")
    lines.append("")

    slope, intercept = np.polyfit(valid["total"], valid["quality"], 1)
    valid = valid.copy()
    valid["quality_predicted"] = slope * valid["total"] + intercept
    valid["residual"] = valid["quality"] - valid["quality_predicted"]

    q1_res = valid["residual"].quantile(0.25)
    q3_res = valid["residual"].quantile(0.75)
    iqr = q3_res - q1_res
    lower_bound = q1_res - 1.5 * iqr
    upper_bound = q3_res + 1.5 * iqr

    lines.append(f"  Regression: quality = {slope:.3f} * total + {intercept:.3f}")
    lines.append(f"  Residual IQR: Q1={q1_res:.2f}, Q3={q3_res:.2f}, IQR={iqr:.2f}")
    lines.append(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    lines.append("")

    # High quality / low total (positive residual outliers)
    high_q_low_t = valid[valid["residual"] > upper_bound].sort_values("residual", ascending=False)
    lines.append(f"  (a) HIGH QUALITY, LOW TOTAL (residual > {upper_bound:.2f}): {len(high_q_low_t)} cases")
    if len(high_q_low_t) > 0:
        lines.append(f"      {'task_id':<20s} {'condition':<30s} {'quality':>8s} {'total':>8s} {'resid':>8s}")
        lines.append("      " + "-" * 78)
        for _, row in high_q_low_t.head(20).iterrows():
            lines.append(
                f"      {row['task_id']:<20s} "
                f"{str(row['condition']):<30s} "
                f"{row['quality']:>8.1f} "
                f"{row['total']:>8.1f} "
                f"{row['residual']:>8.2f}"
            )
    lines.append("")

    # Low quality / high total (negative residual outliers)
    low_q_high_t = valid[valid["residual"] < lower_bound].sort_values("residual", ascending=True)
    lines.append(f"  (b) LOW QUALITY, HIGH TOTAL (residual < {lower_bound:.2f}): {len(low_q_high_t)} cases")
    if len(low_q_high_t) > 0:
        lines.append(f"      {'task_id':<20s} {'condition':<30s} {'quality':>8s} {'total':>8s} {'resid':>8s}")
        lines.append("      " + "-" * 78)
        for _, row in low_q_high_t.head(20).iterrows():
            lines.append(
                f"      {row['task_id']:<20s} "
                f"{str(row['condition']):<30s} "
                f"{row['quality']:>8.1f} "
                f"{row['total']:>8.1f} "
                f"{row['residual']:>8.2f}"
            )
    lines.append("")

    # --- Section 4: Absolute threshold outliers ---
    lines.append("--- 4. Absolute Threshold Outliers ---")

    # High total (>= 80) but low quality (< 15)
    ht_lq = valid[(valid["total"] >= 80) & (valid["quality"] < 15)]
    lines.append(f"  (a) Total >= 80 AND Quality < 15: {len(ht_lq)} cases")
    if len(ht_lq) > 0:
        for _, row in ht_lq.sort_values("quality").head(15).iterrows():
            lines.append(
                f"      {row['task_id']:<20s} "
                f"{str(row['condition']):<30s} "
                f"q={row['quality']:>5.1f}  t={row['total']:>5.1f}"
            )
    lines.append("")

    # Low total (< 30) but high quality (>= 25)
    lt_hq = valid[(valid["total"] < 30) & (valid["quality"] >= 25)]
    lines.append(f"  (b) Total < 30 AND Quality >= 25: {len(lt_hq)} cases")
    if len(lt_hq) > 0:
        for _, row in lt_hq.sort_values("total").head(15).iterrows():
            lines.append(
                f"      {row['task_id']:<20s} "
                f"{str(row['condition']):<30s} "
                f"q={row['quality']:>5.1f}  t={row['total']:>5.1f}"
            )
    lines.append("")

    # --- Section 5: Zero-quality tasks ---
    lines.append("--- 5. Zero-Quality Tasks ---")
    zero_q = valid[valid["quality"] == 0]
    n_zero = len(zero_q)
    n_zero_tasks = zero_q["task_id"].nunique() if n_zero > 0 else 0
    lines.append(f"  Tasks with quality=0: {n_zero} cases ({n_zero_tasks} unique tasks)")
    if n_zero > 0:
        zero_by_cond = zero_q.groupby("condition", observed=True).size()
        for cond, count in zero_by_cond.items():
            lines.append(f"    {str(cond):<35s} {count} tasks")

        # What total scores do these zero-quality tasks have?
        lines.append(f"\n  Total score stats for zero-quality tasks:")
        lines.append(f"    Mean total:   {zero_q['total'].mean():.1f}")
        lines.append(f"    Median total: {zero_q['total'].median():.1f}")
        lines.append(f"    Max total:    {zero_q['total'].max():.1f}")
    lines.append("")

    # --- Section 6: Per-condition quality statistics ---
    lines.append("--- 6. Per-Condition Quality Statistics ---")
    lines.append(
        f"  {'Condition':<35s} {'Mean':>6s} {'Med':>6s} {'Std':>6s} "
        f"{'%Zero':>6s} {'%Max':>6s} {'n':>5s}"
    )
    lines.append("  " + "-" * 72)
    for cond in valid["condition"].cat.categories:
        subset = valid[valid["condition"] == cond]["quality"]
        if len(subset) == 0:
            continue
        pct_zero = (subset == 0).sum() / len(subset) * 100
        pct_max = (subset == 35).sum() / len(subset) * 100
        lines.append(
            f"  {str(cond):<35s} "
            f"{subset.mean():>6.1f} "
            f"{subset.median():>6.1f} "
            f"{subset.std():>6.1f} "
            f"{pct_zero:>5.1f}% "
            f"{pct_max:>5.1f}% "
            f"{len(subset):>5d}"
        )
    lines.append("")

    # --- Section 7: Fraction metric coverage ---
    lines.append("--- 7. AF2 Fraction Metric Coverage ---")
    fraction_cols = ["pLDDT_frac", "pTM_frac", "ipTM_frac", "i_pAE_frac"]
    for col in fraction_cols:
        n_present = df[col].notna().sum()
        n_total_rows = len(df)
        pct = n_present / n_total_rows * 100 if n_total_rows > 0 else 0
        if n_present > 0:
            mean_val = df[col].mean()
            median_val = df[col].median()
            lines.append(
                f"  {col:<14s}  present={n_present:>4d}/{n_total_rows} ({pct:>5.1f}%)  "
                f"mean={mean_val:.3f}  median={median_val:.3f}"
            )
        else:
            lines.append(f"  {col:<14s}  present=   0/{n_total_rows} (  0.0%)")
    lines.append("")

    lines.append("=" * 72)

    summary = "\n".join(lines)
    out_path.write_text(summary + "\n")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all AF2 metrics analyses and save outputs."""
    print("Loading data...")
    df = load_all()
    n_tasks = df["task_id"].nunique()
    n_conditions = df["condition"].nunique()
    print(f"  Loaded {len(df)} rows ({n_tasks} tasks x {n_conditions} conditions)")
    print()

    # Quick overview
    print("Quality score overview:")
    print(f"  Mean:   {df['quality'].mean():.2f}")
    print(f"  Median: {df['quality'].median():.2f}")
    print(f"  Std:    {df['quality'].std():.2f}")
    print(f"  Range:  [{df['quality'].min():.1f}, {df['quality'].max():.1f}]")
    print()

    # Fraction metric availability
    for col in ["pLDDT_frac", "pTM_frac", "ipTM_frac", "i_pAE_frac"]:
        n_avail = df[col].notna().sum()
        print(f"  {col}: {n_avail}/{len(df)} non-null ({100 * n_avail / len(df):.1f}%)")
    print()

    # --- 1. Quality violin plots ---
    print("1. Generating quality score violin plots...")
    violin_path = OUTPUT_DIR / "af2_violin_quality.png"
    plot_quality_violins(df, violin_path)
    print(f"   Saved: {violin_path}")
    print()

    # --- 2. Tier breakdown ---
    print("2. Generating tier breakdown bar chart...")
    tier_path = OUTPUT_DIR / "af2_tier_breakdown.png"
    plot_tier_breakdown(df, tier_path)
    print(f"   Saved: {tier_path}")

    # Print tier means
    tier_means = df.groupby("condition", observed=True)[["tier_a", "tier_b", "tier_c"]].mean()
    print("   Per-condition tier means:")
    print(f"   {'Condition':<35s} {'tier_a':>7s} {'tier_b':>7s} {'tier_c':>7s} {'sum':>7s}")
    print("   " + "-" * 60)
    for cond in df["condition"].cat.categories:
        if cond in tier_means.index:
            row = tier_means.loc[cond]
            print(
                f"   {str(cond):<35s} "
                f"{row['tier_a']:>7.2f} "
                f"{row['tier_b']:>7.2f} "
                f"{row['tier_c']:>7.2f} "
                f"{row['tier_a'] + row['tier_b'] + row['tier_c']:>7.2f}"
            )
    print()

    # --- 3. Quality vs Total scatter ---
    print("3. Generating quality vs total scatter plot...")
    scatter_path = OUTPUT_DIR / "quality_vs_total_scatter.png"
    plot_quality_vs_total(df, scatter_path)
    r_val, _ = stats.pearsonr(df["quality"], df["total"])
    print(f"   Pearson r = {r_val:.4f}")
    print(f"   Saved: {scatter_path}")
    print()

    # --- 4. Fraction violin plots ---
    print("4. Generating AF2 fraction violin plots...")
    frac_path = OUTPUT_DIR / "af2_fraction_violins.png"
    plot_fraction_violins(df, frac_path)
    print(f"   Saved: {frac_path}")
    print()

    # --- 5. Outlier identification ---
    print("5. Identifying scoring outliers...")
    summary_path = OUTPUT_DIR / "scoring_validation_summary.txt"
    summary_text = identify_outliers(df, summary_path)
    print(summary_text)
    print(f"\n   Saved: {summary_path}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
