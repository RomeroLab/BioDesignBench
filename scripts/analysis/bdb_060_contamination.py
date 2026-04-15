#!/usr/bin/env python3
"""BDB-060: Data Contamination Detection Analysis.

Analyzes contamination detection results from BioDesignBench's 5-layer
contamination defense system. A contamination_score >= 0.5 flags a task
as potentially contaminated.

Analyses:
  1. Contamination summary per condition (flagged count, ratio, mean score)
  2. Flagged vs clean task score comparison
  3. Contamination score distribution histogram
  4. Per-agent contamination profile (LLM x legacy_category cross-tab)
  5. Leaderboard before/after contamination zeroing (approximate)

Outputs (results/analysis/):
  - contamination_summary_table.csv
  - flagged_vs_clean_comparison.csv
  - contamination_histogram.png
  - contamination_analysis_summary.txt
  - leaderboard_before_after_zeroing.csv

Usage:
    python -m scripts.analysis.bdb_060_contamination
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

CONTAMINATION_THRESHOLD = 0.5

# ── Nature-style plot defaults ─────────────────────────────────────────────

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
# 1. Contamination summary per condition
# ---------------------------------------------------------------------------


def contamination_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition contamination summary.

    For each condition compute:
      - total tasks evaluated
      - number flagged (contamination_score >= threshold)
      - flagged ratio
      - mean contamination_score

    Args:
        df: Full results DataFrame with contamination_score column.

    Returns:
        Summary DataFrame with one row per condition.
    """
    rows: list[dict] = []
    for condition in df["condition"].cat.categories:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue
        n_total = len(cond_df)
        n_flagged = int((cond_df["contamination_score"] >= CONTAMINATION_THRESHOLD).sum())
        flagged_ratio = n_flagged / n_total if n_total > 0 else 0.0
        mean_score = float(cond_df["contamination_score"].mean())
        rows.append(
            {
                "condition": condition,
                "n_tasks": n_total,
                "n_flagged": n_flagged,
                "flagged_ratio": round(flagged_ratio, 4),
                "mean_contamination_score": round(mean_score, 4),
            }
        )

    summary_df = pd.DataFrame(rows)

    # Append an overall row
    n_total_all = len(df)
    n_flagged_all = int((df["contamination_score"] >= CONTAMINATION_THRESHOLD).sum())
    overall = {
        "condition": "OVERALL",
        "n_tasks": n_total_all,
        "n_flagged": n_flagged_all,
        "flagged_ratio": round(n_flagged_all / n_total_all, 4) if n_total_all > 0 else 0.0,
        "mean_contamination_score": round(float(df["contamination_score"].mean()), 4),
    }
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([overall])], ignore_index=True
    )

    return summary_df


# ---------------------------------------------------------------------------
# 2. Flagged vs Clean comparison
# ---------------------------------------------------------------------------


def flagged_vs_clean_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean total scores for flagged vs clean tasks.

    Note: the 'total' score in the data may already reflect zeroing for
    contaminated tasks. This comparison shows the current (post-penalty)
    scores, not the original pre-penalty scores.

    Args:
        df: Full results DataFrame.

    Returns:
        DataFrame with flagged/clean mean scores per condition.
    """
    flagged_mask = df["contamination_score"] >= CONTAMINATION_THRESHOLD
    df = df.copy()
    df["status"] = np.where(flagged_mask, "flagged", "clean")

    rows: list[dict] = []
    for condition in df["condition"].cat.categories:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue

        clean_df = cond_df[cond_df["status"] == "clean"]
        flagged_df = cond_df[cond_df["status"] == "flagged"]

        rows.append(
            {
                "condition": condition,
                "clean_n": len(clean_df),
                "clean_mean_total": round(float(clean_df["total"].mean()), 2)
                if len(clean_df) > 0
                else np.nan,
                "flagged_n": len(flagged_df),
                "flagged_mean_total": round(float(flagged_df["total"].mean()), 2)
                if len(flagged_df) > 0
                else np.nan,
                "score_delta": round(
                    float(clean_df["total"].mean()) - float(flagged_df["total"].mean()), 2
                )
                if len(clean_df) > 0 and len(flagged_df) > 0
                else np.nan,
            }
        )

    # Overall row
    clean_all = df[df["status"] == "clean"]
    flagged_all = df[df["status"] == "flagged"]
    overall = {
        "condition": "OVERALL",
        "clean_n": len(clean_all),
        "clean_mean_total": round(float(clean_all["total"].mean()), 2)
        if len(clean_all) > 0
        else np.nan,
        "flagged_n": len(flagged_all),
        "flagged_mean_total": round(float(flagged_all["total"].mean()), 2)
        if len(flagged_all) > 0
        else np.nan,
        "score_delta": round(
            float(clean_all["total"].mean()) - float(flagged_all["total"].mean()), 2
        )
        if len(clean_all) > 0 and len(flagged_all) > 0
        else np.nan,
    }

    result = pd.DataFrame(rows + [overall])
    return result


# ---------------------------------------------------------------------------
# 3. Contamination score distribution histogram
# ---------------------------------------------------------------------------


def plot_contamination_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Plot histogram of contamination_score colored by flagged status.

    Args:
        df: Full results DataFrame.
        out_path: Path for saved PNG.
    """
    scores = df["contamination_score"].values
    flagged_mask = scores >= CONTAMINATION_THRESHOLD

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine bin edges spanning [0, max(scores) or 1]
    score_max = max(float(scores.max()), 1.0) if len(scores) > 0 else 1.0
    bins = np.linspace(0, score_max, 41)

    clean_scores = scores[~flagged_mask]
    flagged_scores = scores[flagged_mask]

    # Plot stacked histogram: clean first, then flagged
    if len(clean_scores) > 0:
        ax.hist(
            clean_scores,
            bins=bins,
            color="#4C72B0",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=f"Clean (n={len(clean_scores)})",
            zorder=3,
        )
    if len(flagged_scores) > 0:
        ax.hist(
            flagged_scores,
            bins=bins,
            color="#C44E52",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=f"Flagged (n={len(flagged_scores)})",
            zorder=3,
        )

    # Threshold line
    ax.axvline(
        x=CONTAMINATION_THRESHOLD,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({CONTAMINATION_THRESHOLD})",
        zorder=5,
    )

    ax.set_xlabel("Contamination Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Contamination Score Distribution Across All Data Points",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Annotate summary stats
    mean_score = float(scores.mean()) if len(scores) > 0 else 0.0
    n_flagged = int(flagged_mask.sum())
    annotation = (
        f"N={len(scores)}, Mean={mean_score:.3f}\n"
        f"Flagged: {n_flagged} ({n_flagged / len(scores) * 100:.1f}%)"
        if len(scores) > 0
        else "No data"
    )
    ax.text(
        0.98,
        0.85,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Per-agent contamination profile
# ---------------------------------------------------------------------------


def per_agent_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-LLM flagged counts and LLM x legacy_category cross-tab.

    Args:
        df: Full results DataFrame.

    Returns:
        Tuple of (per_llm_summary, cross_tab).
        per_llm_summary: DataFrame with llm, n_tasks, n_flagged, flagged_ratio.
        cross_tab: DataFrame of flagged counts by llm (rows) x legacy_category (cols).
    """
    flagged_mask = df["contamination_score"] >= CONTAMINATION_THRESHOLD
    df = df.copy()
    df["is_flagged"] = flagged_mask.astype(int)

    # Per-LLM summary
    llm_summary_rows: list[dict] = []
    for llm in sorted(df["llm"].unique()):
        llm_df = df[df["llm"] == llm]
        n_total = len(llm_df)
        n_flagged = int(llm_df["is_flagged"].sum())
        llm_summary_rows.append(
            {
                "llm": llm,
                "n_tasks": n_total,
                "n_flagged": n_flagged,
                "flagged_ratio": round(n_flagged / n_total, 4) if n_total > 0 else 0.0,
                "mean_contamination_score": round(float(llm_df["contamination_score"].mean()), 4),
            }
        )
    per_llm = pd.DataFrame(llm_summary_rows).sort_values("n_flagged", ascending=False)

    # Cross-tab: llm x legacy_category flagged counts
    cross_tab = pd.crosstab(
        df[df["is_flagged"] == 1]["llm"],
        df[df["is_flagged"] == 1]["legacy_category"],
        margins=True,
    )
    # If no flagged tasks at all, create an empty cross-tab with proper structure
    if cross_tab.empty:
        cross_tab = pd.DataFrame(
            {"(no flagged tasks)": []}, index=pd.Index([], name="llm")
        )

    return per_llm, cross_tab


# ---------------------------------------------------------------------------
# 5. Leaderboard before/after zeroing
# ---------------------------------------------------------------------------


def leaderboard_before_after(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate leaderboard comparison with and without contamination penalty.

    The 'total' column reflects scores after contamination zeroing.
    We approximate pre-penalty scores by noting that flagged tasks currently
    have their scores (likely zeroed or penalized). Since we do not have the
    original pre-penalty scores stored separately, we show:
      - current_mean: mean total with penalty applied (as-is)
      - without_flagged_mean: mean total excluding flagged tasks entirely
      - rank_current, rank_without_flagged: rankings under each scenario

    This is approximate because the true pre-penalty score is unknown.

    Args:
        df: Full results DataFrame.

    Returns:
        DataFrame with condition-level leaderboard comparison.
    """
    flagged_mask = df["contamination_score"] >= CONTAMINATION_THRESHOLD

    rows: list[dict] = []
    for condition in df["condition"].cat.categories:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue

        current_mean = float(cond_df["total"].mean())
        clean_df = cond_df[~(cond_df["contamination_score"] >= CONTAMINATION_THRESHOLD)]
        without_flagged_mean = float(clean_df["total"].mean()) if len(clean_df) > 0 else np.nan
        n_flagged = int((cond_df["contamination_score"] >= CONTAMINATION_THRESHOLD).sum())

        rows.append(
            {
                "condition": condition,
                "n_tasks": len(cond_df),
                "n_flagged": n_flagged,
                "current_mean_total": round(current_mean, 2),
                "without_flagged_mean_total": round(without_flagged_mean, 2)
                if not np.isnan(without_flagged_mean)
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)

    # Compute ranks (higher score = better = rank 1)
    result["rank_current"] = (
        result["current_mean_total"].rank(ascending=False, method="min").astype(int)
    )
    if result["without_flagged_mean_total"].notna().any():
        result["rank_without_flagged"] = (
            result["without_flagged_mean_total"]
            .rank(ascending=False, method="min")
            .astype("Int64")
        )
    else:
        result["rank_without_flagged"] = result["rank_current"]

    result["rank_change"] = result["rank_current"] - result["rank_without_flagged"]

    result = result.sort_values("rank_current").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Summary text writer
# ---------------------------------------------------------------------------


def write_analysis_summary(
    df: pd.DataFrame,
    summary_table: pd.DataFrame,
    comparison: pd.DataFrame,
    per_llm: pd.DataFrame,
    cross_tab: pd.DataFrame,
    leaderboard: pd.DataFrame,
    out_path: Path,
) -> str:
    """Write comprehensive contamination analysis summary text.

    Args:
        df: Full results DataFrame.
        summary_table: Output of contamination_summary().
        comparison: Output of flagged_vs_clean_comparison().
        per_llm: Per-LLM summary from per_agent_profile().
        cross_tab: LLM x legacy_category cross-tab from per_agent_profile().
        leaderboard: Output of leaderboard_before_after().
        out_path: Path to save .txt file.

    Returns:
        The summary text as a string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("BDB-060: Data Contamination Detection Analysis")
    lines.append("=" * 70)
    lines.append("")

    n_total = len(df)
    n_flagged = int((df["contamination_score"] >= CONTAMINATION_THRESHOLD).sum())
    n_conditions = df["condition"].nunique()
    n_tasks = df["task_id"].nunique()
    lines.append(f"Dataset: {n_total} data points ({n_tasks} tasks x {n_conditions} conditions)")
    lines.append(f"Contamination threshold: {CONTAMINATION_THRESHOLD}")
    lines.append(f"Total flagged: {n_flagged} / {n_total} ({n_flagged / n_total * 100:.1f}%)")
    lines.append(f"Mean contamination score: {df['contamination_score'].mean():.4f}")
    lines.append("")

    # No contamination case
    if n_flagged == 0:
        lines.append("*** NO TASKS WERE FLAGGED FOR CONTAMINATION ***")
        lines.append("")
        lines.append(
            "All contamination_score values are below the threshold of "
            f"{CONTAMINATION_THRESHOLD}. This indicates that the 5-layer "
            "contamination defense system did not detect any data leakage "
            "concerns in the current evaluation results."
        )
        lines.append("")
        lines.append("Score statistics:")
        lines.append(f"  min:    {df['contamination_score'].min():.4f}")
        lines.append(f"  max:    {df['contamination_score'].max():.4f}")
        lines.append(f"  median: {df['contamination_score'].median():.4f}")
        lines.append(f"  std:    {df['contamination_score'].std():.4f}")
        lines.append("")

    # Section 1: Per-condition summary
    lines.append("--- 1. Per-Condition Contamination Summary ---")
    lines.append("")
    lines.append(
        f"  {'Condition':<35s} {'Tasks':>5s} {'Flagged':>7s} "
        f"{'Ratio':>7s} {'Mean Score':>10s}"
    )
    lines.append("  " + "-" * 66)
    for _, row in summary_table.iterrows():
        lines.append(
            f"  {str(row['condition']):<35s} {row['n_tasks']:>5d} "
            f"{row['n_flagged']:>7d} {row['flagged_ratio']:>7.2%} "
            f"{row['mean_contamination_score']:>10.4f}"
        )
    lines.append("")

    # Section 2: Flagged vs Clean
    lines.append("--- 2. Flagged vs Clean Task Score Comparison ---")
    lines.append("")
    lines.append(
        "  NOTE: 'total' scores may already reflect zeroing for flagged tasks."
    )
    lines.append(
        "  The 'flagged_mean_total' shows current (post-penalty) scores."
    )
    lines.append("")
    lines.append(
        f"  {'Condition':<35s} {'Clean n':>7s} {'Clean mean':>10s} "
        f"{'Flag n':>6s} {'Flag mean':>9s} {'Delta':>7s}"
    )
    lines.append("  " + "-" * 76)
    for _, row in comparison.iterrows():
        delta_str = f"{row['score_delta']:>7.1f}" if pd.notna(row["score_delta"]) else "    N/A"
        flag_mean = (
            f"{row['flagged_mean_total']:>9.1f}"
            if pd.notna(row["flagged_mean_total"])
            else "      N/A"
        )
        lines.append(
            f"  {str(row['condition']):<35s} {row['clean_n']:>7d} "
            f"{row['clean_mean_total']:>10.1f} {row['flagged_n']:>6d} "
            f"{flag_mean} {delta_str}"
        )
    lines.append("")

    # Section 3: Per-agent profile
    lines.append("--- 3. Per-Agent (LLM) Contamination Profile ---")
    lines.append("")
    lines.append(
        f"  {'LLM':<25s} {'Tasks':>5s} {'Flagged':>7s} "
        f"{'Ratio':>7s} {'Mean Score':>10s}"
    )
    lines.append("  " + "-" * 56)
    for _, row in per_llm.iterrows():
        lines.append(
            f"  {row['llm']:<25s} {row['n_tasks']:>5d} "
            f"{row['n_flagged']:>7d} {row['flagged_ratio']:>7.2%} "
            f"{row['mean_contamination_score']:>10.4f}"
        )
    lines.append("")

    # Cross-tab
    lines.append("  LLM x Legacy Category Flagged Counts:")
    lines.append("")
    if n_flagged > 0:
        cross_str = cross_tab.to_string()
        for line in cross_str.split("\n"):
            lines.append(f"  {line}")
    else:
        lines.append("  (No flagged tasks -- cross-tab is empty)")
    lines.append("")

    # Concentration check
    if n_flagged > 0:
        lines.append("  Concentration analysis:")
        # Which LLM is most flagged?
        top_llm = per_llm.iloc[0]
        lines.append(
            f"    Most flagged LLM: {top_llm['llm']} "
            f"({top_llm['n_flagged']} flagged, {top_llm['flagged_ratio']:.1%})"
        )
        # Which category is most flagged?
        flagged_df = df[df["contamination_score"] >= CONTAMINATION_THRESHOLD]
        cat_counts = flagged_df["legacy_category"].value_counts()
        if len(cat_counts) > 0:
            top_cat = cat_counts.index[0]
            lines.append(
                f"    Most flagged category: {top_cat} ({cat_counts.iloc[0]} flags)"
            )
        lines.append("")

    # Section 4: Leaderboard impact
    lines.append("--- 4. Leaderboard Impact (Before/After Zeroing) ---")
    lines.append("")
    lines.append(
        "  NOTE: Approximation only. 'without_flagged' excludes flagged tasks"
    )
    lines.append(
        "  entirely rather than restoring original scores (which are unavailable)."
    )
    lines.append("")
    lines.append(
        f"  {'Condition':<35s} {'Curr Mean':>9s} {'W/o Flag':>8s} "
        f"{'Rank':>4s} {'Rank*':>5s} {'Chg':>4s}"
    )
    lines.append("  " + "-" * 67)
    for _, row in leaderboard.iterrows():
        wf_mean = (
            f"{row['without_flagged_mean_total']:>8.1f}"
            if pd.notna(row["without_flagged_mean_total"])
            else "     N/A"
        )
        rank_wf = (
            f"{row['rank_without_flagged']:>5d}"
            if pd.notna(row["rank_without_flagged"])
            else "  N/A"
        )
        rank_chg = (
            f"{row['rank_change']:>+4d}"
            if pd.notna(row["rank_change"])
            else " N/A"
        )
        lines.append(
            f"  {str(row['condition']):<35s} {row['current_mean_total']:>9.1f} "
            f"{wf_mean} {row['rank_current']:>4d} {rank_wf} {rank_chg}"
        )
    lines.append("")

    # Key takeaways
    lines.append("--- 5. Key Takeaways ---")
    lines.append("")
    if n_flagged == 0:
        lines.append(
            "  * No contamination detected across all 9 conditions."
        )
        lines.append(
            "  * The 5-layer contamination defense system cleared all tasks."
        )
        lines.append(
            "  * Leaderboard rankings are unaffected by contamination penalties."
        )
    else:
        pct_flagged = n_flagged / n_total * 100
        lines.append(f"  * {n_flagged} / {n_total} data points flagged ({pct_flagged:.1f}%)")

        # Check if contamination is uniform or concentrated
        per_llm_sorted = per_llm.sort_values("n_flagged", ascending=False)
        top_agent = per_llm_sorted.iloc[0]
        bottom_agent = per_llm_sorted.iloc[-1]
        if top_agent["n_flagged"] > 2 * max(bottom_agent["n_flagged"], 1):
            lines.append(
                f"  * Contamination is CONCENTRATED: {top_agent['llm']} has "
                f"{top_agent['n_flagged']} flags vs {bottom_agent['llm']} with "
                f"{bottom_agent['n_flagged']}"
            )
        else:
            lines.append("  * Contamination is spread relatively evenly across agents.")

        # Rank changes
        any_rank_change = (leaderboard["rank_change"].abs() > 0).any()
        if any_rank_change:
            lines.append(
                "  * Contamination zeroing DOES affect leaderboard rankings."
            )
        else:
            lines.append(
                "  * Contamination zeroing does NOT change leaderboard rankings."
            )
    lines.append("")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    out_path.write_text(summary + "\n")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all contamination analyses and save outputs."""
    print("Loading data...")
    df = load_all()
    n_tasks = df["task_id"].nunique()
    n_conditions = df["condition"].nunique()
    print(f"  Loaded {len(df)} rows ({n_tasks} tasks x {n_conditions} conditions)")

    n_flagged = int((df["contamination_score"] >= CONTAMINATION_THRESHOLD).sum())
    print(f"  Contamination threshold: {CONTAMINATION_THRESHOLD}")
    print(f"  Flagged data points: {n_flagged} / {len(df)}")
    print(f"  Mean contamination score: {df['contamination_score'].mean():.4f}")
    print()

    # ── 1. Contamination summary table ────────────────────────────────────
    print("1. Contamination summary per condition...")
    summary_table = contamination_summary(df)
    summary_path = OUTPUT_DIR / "contamination_summary_table.csv"
    summary_table.to_csv(summary_path, index=False, float_format="%.4f")
    print(summary_table.to_string(index=False))
    print(f"  Saved: {summary_path}")
    print()

    # ── 2. Flagged vs Clean comparison ────────────────────────────────────
    print("2. Flagged vs Clean comparison...")
    comparison = flagged_vs_clean_comparison(df)
    comparison_path = OUTPUT_DIR / "flagged_vs_clean_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.2f")
    print(comparison.to_string(index=False))
    print(f"  Saved: {comparison_path}")
    print()

    # ── 3. Contamination histogram ────────────────────────────────────────
    print("3. Generating contamination histogram...")
    hist_path = OUTPUT_DIR / "contamination_histogram.png"
    plot_contamination_histogram(df, hist_path)
    print(f"  Saved: {hist_path}")
    print()

    # ── 4. Per-agent contamination profile ────────────────────────────────
    print("4. Per-agent contamination profile...")
    per_llm, cross_tab = per_agent_profile(df)
    print("  Per-LLM summary:")
    print(per_llm.to_string(index=False))
    print()
    if n_flagged > 0:
        print("  LLM x Legacy Category cross-tab (flagged counts):")
        print(cross_tab.to_string())
    else:
        print("  (No flagged tasks -- cross-tab is empty)")
    print()

    # ── 5. Leaderboard before/after zeroing ───────────────────────────────
    print("5. Leaderboard before/after zeroing...")
    leaderboard = leaderboard_before_after(df)
    leaderboard_path = OUTPUT_DIR / "leaderboard_before_after_zeroing.csv"
    leaderboard.to_csv(leaderboard_path, index=False, float_format="%.2f")
    print(leaderboard.to_string(index=False))
    print(f"  Saved: {leaderboard_path}")
    print()

    # ── Write comprehensive summary text ──────────────────────────────────
    print("6. Writing analysis summary...")
    summary_txt_path = OUTPUT_DIR / "contamination_analysis_summary.txt"
    summary_text = write_analysis_summary(
        df, summary_table, comparison, per_llm, cross_tab, leaderboard, summary_txt_path
    )
    print(summary_text)
    print(f"  Saved: {summary_txt_path}")
    print()

    print(f"All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
