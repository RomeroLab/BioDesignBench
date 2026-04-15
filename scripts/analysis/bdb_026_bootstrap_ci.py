#!/usr/bin/env python3
"""BDB-026: Bootstrap CI, Paired Permutation Tests, Effect Size, and Rank Stability.

Performs rigorous non-parametric statistical comparisons across 9 benchmark
conditions (76 tasks each), producing:
  1. Bootstrap 95% CIs for each condition mean
  2. Paired permutation p-value matrix (36 pairwise comparisons)
  3. Cohen's d effect size matrix
  4. Bonferroni and Holm-Bonferroni multiple comparison corrections
  5. Rank stability analysis across bootstrap resamples

Outputs saved to results/analysis/:
  - bootstrap_ci_table.csv
  - pvalue_matrix.csv
  - effect_size_matrix.csv
  - rank_stability_plot.png
  - statistical_summary.txt
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

# Leaderboard order for consistent display
LEADERBOARD_ORDER = [
    "DeepSeek V3 user",
    "DeepSeek V3 benchmark",
    "GPT-5 user",
    "Sonnet 4.5 user",
    "Sonnet 4.5 benchmark",
    "GPT-5 benchmark",
    "Hardcoded Pipeline",
    "Gemini 2.5 Pro user",
    "Gemini 2.5 Pro benchmark",
]


# ── Core statistical functions ────────────────────────────────────────────────


def significance_stars(p: float) -> str:
    """Map p-value to significance stars.

    Args:
        p: p-value (may be NaN).

    Returns:
        '***' if p <= 0.001, '**' if p <= 0.01, '*' if p <= 0.05, else ''.
    """
    if np.isnan(p):
        return ""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def compute_bootstrap_ci(
    score_matrix: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap 95% percentile confidence intervals for each condition.

    Args:
        score_matrix: DataFrame with tasks as rows, conditions as columns.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: condition, mean, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)
    n_tasks = len(score_matrix)
    rows = []

    for cond in score_matrix.columns:
        scores = score_matrix[cond].values
        observed_mean = float(np.mean(scores))

        # Bootstrap resample means
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n_tasks, size=n_tasks)
            boot_means[b] = np.mean(scores[idx])

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        rows.append({
            "condition": cond,
            "mean": observed_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return pd.DataFrame(rows)


def compute_pvalue_matrix(
    score_matrix: pd.DataFrame,
    n_perm: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired permutation test for all pairwise condition comparisons.

    For each pair (A, B), computes per-task differences d_i = A_i - B_i,
    then permutes the signs 10,000 times to build a null distribution of
    the mean difference. Two-sided p-value.

    Args:
        score_matrix: DataFrame with tasks as rows, conditions as columns.
        n_perm: Number of permutations.
        seed: Random seed for reproducibility.

    Returns:
        9x9 DataFrame of p-values (diagonal = NaN).
    """
    conditions = list(score_matrix.columns)
    n_cond = len(conditions)
    n_tasks = len(score_matrix)
    rng = np.random.RandomState(seed)

    pvals = pd.DataFrame(
        np.full((n_cond, n_cond), np.nan),
        index=conditions,
        columns=conditions,
    )

    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if i == j:
                continue
            if j < i:
                # Already computed; symmetric
                pvals.iloc[i, j] = pvals.iloc[j, i]
                continue

            diff = score_matrix[c1].values - score_matrix[c2].values
            observed_stat = abs(np.mean(diff))

            # Generate random sign flips
            signs = rng.choice([-1, 1], size=(n_perm, n_tasks))
            perm_means = np.abs(np.mean(signs * diff[np.newaxis, :], axis=1))

            # Two-sided p-value: fraction of permutations >= observed
            p = float(np.mean(perm_means >= observed_stat))
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p

    return pvals


def compute_effect_size_matrix(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's d for all pairwise comparisons.

    Cohen's d = (mean_A - mean_B) / pooled_sd, where
    pooled_sd = sqrt((var_A + var_B) / 2).

    Args:
        score_matrix: DataFrame with tasks as rows, conditions as columns.

    Returns:
        9x9 DataFrame of Cohen's d values (diagonal = 0, antisymmetric).
    """
    conditions = list(score_matrix.columns)
    n_cond = len(conditions)

    es = pd.DataFrame(
        np.zeros((n_cond, n_cond)),
        index=conditions,
        columns=conditions,
    )

    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if i == j:
                continue
            if j < i:
                es.iloc[i, j] = -es.iloc[j, i]
                continue

            scores_a = score_matrix[c1].values
            scores_b = score_matrix[c2].values
            mean_diff = np.mean(scores_a) - np.mean(scores_b)
            pooled_sd = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)

            if pooled_sd < 1e-12:
                d = 0.0
            else:
                d = mean_diff / pooled_sd

            es.iloc[i, j] = d
            es.iloc[j, i] = -d

    return es


def apply_corrections(
    raw_pvals: list[float],
    n_comparisons: int,
) -> dict[str, list[float]]:
    """Apply Bonferroni and Holm-Bonferroni corrections.

    Args:
        raw_pvals: List of raw p-values.
        n_comparisons: Total number of comparisons (for Bonferroni denominator).

    Returns:
        Dict with 'bonferroni' and 'holm' keys, each a list of corrected p-values
        in the same order as raw_pvals.
    """
    n = len(raw_pvals)

    # Bonferroni: p_adj = min(p * n_comparisons, 1.0)
    bonferroni = [min(p * n_comparisons, 1.0) for p in raw_pvals]

    # Holm-Bonferroni: sort by raw p, adjust sequentially
    indexed = sorted(enumerate(raw_pvals), key=lambda x: x[1])
    holm = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n_comparisons - rank)
        adjusted = min(adjusted, 1.0)
        # Enforce monotonicity: adjusted p cannot decrease
        cummax = max(cummax, adjusted)
        holm[orig_idx] = cummax

    return {"bonferroni": bonferroni, "holm": holm}


def compute_rank_distribution(
    score_matrix: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute rank distribution of conditions across bootstrap resamples.

    In each bootstrap, resample tasks (rows) with replacement, compute
    condition means, then rank them (1 = best).

    Args:
        score_matrix: DataFrame with tasks as rows, conditions as columns.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of shape (n_boot, n_conditions) with integer ranks.
    """
    rng = np.random.RandomState(seed)
    n_tasks = len(score_matrix)
    conditions = list(score_matrix.columns)
    values = score_matrix.values  # (n_tasks, n_cond)

    rank_data = np.empty((n_boot, len(conditions)), dtype=int)

    for b in range(n_boot):
        idx = rng.randint(0, n_tasks, size=n_tasks)
        boot_means = np.mean(values[idx, :], axis=0)
        # Rank: highest mean = rank 1
        # argsort of negated means gives rank order
        order = np.argsort(-boot_means)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(conditions) + 1)
        rank_data[b, :] = ranks

    return pd.DataFrame(rank_data, columns=conditions)


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_rank_stability(
    rank_df: pd.DataFrame,
    output_path: Path,
    leaderboard_order: list[str] | None = None,
) -> None:
    """Create rank stability violin/box plot.

    Args:
        rank_df: DataFrame from compute_rank_distribution.
        output_path: Path to save the figure.
        leaderboard_order: Optional ordering of conditions for the y-axis.
    """
    if leaderboard_order is None:
        leaderboard_order = list(rank_df.columns)

    # Filter to conditions actually present in rank_df
    conditions = [c for c in leaderboard_order if c in rank_df.columns]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for horizontal box plots
    box_data = [rank_df[cond].values for cond in conditions]

    bp = ax.boxplot(
        box_data,
        vert=False,
        tick_labels=conditions,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Color gradient: rank 1 region = green, rank 9 region = red
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(conditions)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Rank (1 = best)", fontsize=12)
    ax.set_title("Rank Stability Across 10,000 Bootstrap Resamples", fontsize=14)
    ax.set_xlim(0.5, len(conditions) + 0.5)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.invert_yaxis()  # Best (rank 1) at top
    ax.grid(axis="x", alpha=0.3)

    # Add median rank annotation
    for i, cond in enumerate(conditions):
        median_rank = np.median(rank_df[cond].values)
        rank1_pct = (rank_df[cond] == 1).mean() * 100
        ax.annotate(
            f"med={median_rank:.0f}, R1={rank1_pct:.0f}%",
            xy=(len(conditions) + 0.3, i + 1),
            fontsize=8,
            va="center",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def main() -> None:
    """Run full statistical analysis pipeline."""
    from scripts.analysis.load_results import load_score_matrix

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading score matrix...")
    score_matrix = load_score_matrix()
    n_tasks = len(score_matrix)
    n_cond = len(score_matrix.columns)
    print(f"  {n_tasks} tasks x {n_cond} conditions")

    # Reorder columns to leaderboard order
    available = [c for c in LEADERBOARD_ORDER if c in score_matrix.columns]
    score_matrix = score_matrix[available]

    # ── 1. Bootstrap 95% CI ───────────────────────────────────────────────
    print("\n[1/5] Computing bootstrap 95% CIs (10,000 resamples)...")
    ci_table = compute_bootstrap_ci(score_matrix, n_boot=10_000, seed=42)
    ci_table.to_csv(OUTPUT_DIR / "bootstrap_ci_table.csv", index=False, float_format="%.2f")
    print("  Saved bootstrap_ci_table.csv")

    for _, row in ci_table.iterrows():
        print(f"  {row['condition']:30s}  {row['mean']:5.1f}  [{row['ci_lower']:5.1f}, {row['ci_upper']:5.1f}]")

    # ── 2. Paired permutation test ────────────────────────────────────────
    print("\n[2/5] Running paired permutation tests (10,000 permutations, 36 pairs)...")
    pval_matrix = compute_pvalue_matrix(score_matrix, n_perm=10_000, seed=42)

    # Format with significance stars (use object dtype to avoid FutureWarning)
    pval_display = pd.DataFrame(
        index=pval_matrix.index,
        columns=pval_matrix.columns,
        dtype=object,
    )
    for c1 in pval_display.index:
        for c2 in pval_display.columns:
            val = pval_matrix.loc[c1, c2]
            if np.isnan(val):
                pval_display.loc[c1, c2] = "---"
            else:
                stars = significance_stars(val)
                pval_display.loc[c1, c2] = f"{val:.4f}{stars}"

    pval_display.to_csv(OUTPUT_DIR / "pvalue_matrix.csv")
    print("  Saved pvalue_matrix.csv")

    # ── 3. Effect size (Cohen's d) ────────────────────────────────────────
    print("\n[3/5] Computing Cohen's d effect sizes...")
    es_matrix = compute_effect_size_matrix(score_matrix)
    es_matrix.to_csv(OUTPUT_DIR / "effect_size_matrix.csv", float_format="%.3f")
    print("  Saved effect_size_matrix.csv")

    # ── 4. Multiple comparison correction ─────────────────────────────────
    print("\n[4/5] Applying multiple comparison corrections...")
    n_comparisons = n_cond * (n_cond - 1) // 2  # 36
    conditions = list(score_matrix.columns)

    pair_results = []
    raw_pvals_list = []
    pair_labels = []

    for c1, c2 in combinations(conditions, 2):
        p = pval_matrix.loc[c1, c2]
        d = es_matrix.loc[c1, c2]
        raw_pvals_list.append(p)
        pair_labels.append((c1, c2))
        pair_results.append({
            "condition_1": c1,
            "condition_2": c2,
            "raw_pvalue": p,
            "cohens_d": d,
        })

    corrections = apply_corrections(raw_pvals_list, n_comparisons=n_comparisons)

    for i, pr in enumerate(pair_results):
        pr["bonferroni_p"] = corrections["bonferroni"][i]
        pr["holm_p"] = corrections["holm"][i]
        pr["bonferroni_sig"] = significance_stars(corrections["bonferroni"][i])
        pr["holm_sig"] = significance_stars(corrections["holm"][i])
        pr["raw_sig"] = significance_stars(pr["raw_pvalue"])

    corrections_df = pd.DataFrame(pair_results)
    corrections_df = corrections_df.sort_values("raw_pvalue").reset_index(drop=True)

    # ── 5. Rank stability ─────────────────────────────────────────────────
    print("\n[5/5] Computing rank stability (10,000 bootstraps)...")
    rank_df = compute_rank_distribution(score_matrix, n_boot=10_000, seed=42)

    plot_path = OUTPUT_DIR / "rank_stability_plot.png"
    plot_rank_stability(rank_df, plot_path, leaderboard_order=available)
    print(f"  Saved {plot_path.name}")

    # Rank 1 percentages
    print("\n  Rank 1 percentages:")
    for cond in available:
        r1_pct = (rank_df[cond] == 1).mean() * 100
        print(f"    {cond:30s}  {r1_pct:5.1f}%")

    # Print the required statement
    ds_user_rank1_pct = (rank_df["DeepSeek V3 user"] == 1).mean() * 100
    print(f"\nDeepSeek V3 user is rank 1 in {ds_user_rank1_pct:.1f}% of bootstraps")

    # ── Statistical summary ───────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "statistical_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BioDesignBench Statistical Analysis Summary (BDB-026)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tasks:              {n_tasks}\n")
        f.write(f"Conditions:         {n_cond}\n")
        f.write(f"Bootstrap samples:  10,000\n")
        f.write(f"Permutation tests:  10,000\n")
        f.write(f"Random seed:        42\n")
        f.write(f"Pairwise tests:     {n_comparisons}\n\n")

        f.write("1. Bootstrap 95% Confidence Intervals\n")
        f.write("-" * 40 + "\n")
        for _, row in ci_table.iterrows():
            f.write(
                f"  {row['condition']:30s}  "
                f"mean={row['mean']:5.1f}  "
                f"95% CI=[{row['ci_lower']:5.1f}, {row['ci_upper']:5.1f}]\n"
            )

        f.write("\n2. Significant Pairwise Differences (raw p < 0.05)\n")
        f.write("-" * 40 + "\n")
        sig_pairs = corrections_df[corrections_df["raw_pvalue"] <= 0.05]
        for _, row in sig_pairs.iterrows():
            f.write(
                f"  {row['condition_1']:30s} vs {row['condition_2']:30s}  "
                f"p={row['raw_pvalue']:.4f}{row['raw_sig']:4s}  "
                f"d={row['cohens_d']:+.3f}  "
                f"Bonferroni={row['bonferroni_p']:.4f}{row['bonferroni_sig']:4s}  "
                f"Holm={row['holm_p']:.4f}{row['holm_sig']:4s}\n"
            )

        nonsig_count = len(corrections_df[corrections_df["raw_pvalue"] > 0.05])
        f.write(f"\n  Non-significant pairs (p > 0.05): {nonsig_count}\n")

        f.write(f"\n3. Summary of Corrections ({n_comparisons} comparisons)\n")
        f.write("-" * 40 + "\n")
        for threshold_name, col in [("Raw", "raw_pvalue"), ("Bonferroni", "bonferroni_p"), ("Holm", "holm_p")]:
            n_sig = (corrections_df[col] <= 0.05).sum()
            f.write(f"  {threshold_name:15s}  p<=0.05: {n_sig:2d}/{n_comparisons}\n")

        f.write("\n4. Rank Stability\n")
        f.write("-" * 40 + "\n")
        for cond in available:
            median_rank = np.median(rank_df[cond].values)
            r1_pct = (rank_df[cond] == 1).mean() * 100
            r1_or_2_pct = (rank_df[cond] <= 2).mean() * 100
            f.write(
                f"  {cond:30s}  median_rank={median_rank:.0f}  "
                f"R1={r1_pct:5.1f}%  R1-2={r1_or_2_pct:5.1f}%\n"
            )

        f.write(f"\nDeepSeek V3 user is rank 1 in {ds_user_rank1_pct:.1f}% of bootstraps\n")

        f.write("\n5. Effect Size Interpretation Guide\n")
        f.write("-" * 40 + "\n")
        f.write("  |d| < 0.2  : negligible\n")
        f.write("  |d| = 0.2  : small\n")
        f.write("  |d| = 0.5  : medium\n")
        f.write("  |d| = 0.8  : large\n")
        f.write("  |d| > 1.2  : very large\n")

    print(f"\n  Saved {summary_path.name}")
    print("\nDone. All outputs saved to results/analysis/")


if __name__ == "__main__":
    main()
