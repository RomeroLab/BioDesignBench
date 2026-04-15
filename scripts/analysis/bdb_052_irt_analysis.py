#!/usr/bin/env python3
"""BDB-052: Task Discrimination Analysis using Item Response Theory.

Performs classical test theory analysis and 2-Parameter Logistic (2PL) IRT
modelling on the 76-task x 9-condition BioDesignBench score matrix.

Analyses:
  1. Classical test theory (mean, std, discrimination index)
  2. IRT 2PL fitting via MLE (scipy.optimize)
  3. Calibration check: assigned difficulty vs IRT beta
  4. Per-category discrimination summary

Outputs (results/analysis/):
  - irt_parameters.csv
  - calibration_scatter.png
  - discrimination_histogram.png
  - irt_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all, load_score_matrix  # noqa: E402

np.random.seed(42)

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Difficulty string -> numeric
DIFFICULTY_NUM = {"easy": 1, "medium": 2, "hard": 3}


# ============================================================================
# 1. Classical Test Theory
# ============================================================================

def classical_test_theory(score_mat: pd.DataFrame) -> pd.DataFrame:
    """Compute per-task classical statistics across the 9 conditions.

    Discrimination index = std / max_possible_std, where max_possible_std = 50
    (the standard deviation of a Bernoulli(0.5) scaled to [0, 100]).
    """
    stats = pd.DataFrame(
        {
            "mean_score": score_mat.mean(axis=1),
            "std_score": score_mat.std(axis=1, ddof=0),
            "min_score": score_mat.min(axis=1),
            "max_score": score_mat.max(axis=1),
        }
    )
    max_possible_std = 50.0
    stats["discrimination_index"] = stats["std_score"] / max_possible_std
    return stats


# ============================================================================
# 2. IRT 2-Parameter Logistic (2PL) -- direct MLE fitting
# ============================================================================

def _logistic_2pl(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """P(correct | theta) = 1 / (1 + exp(-alpha * (theta - beta)))."""
    z = alpha * (theta - beta)
    # Clip for numerical stability
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _neg_log_likelihood_2pl(
    params: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
) -> float:
    """Negative log-likelihood for a single item under 2PL.

    Parameters
    ----------
    params : (alpha, beta)
    theta  : array of examinee abilities (J,)
    y      : observed proportions in [0, 1] for each examinee (J,)
    """
    alpha, beta = params
    # Enforce alpha > 0 via soft penalty
    if alpha <= 0:
        return 1e12
    p = _logistic_2pl(theta, alpha, beta)
    # Use quasi-likelihood treating y as continuous [0,1]
    eps = 1e-10
    p = np.clip(p, eps, 1.0 - eps)
    # Cross-entropy loss (continuous Bernoulli approximation)
    nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return nll


def fit_irt_2pl(score_mat: pd.DataFrame) -> pd.DataFrame:
    """Fit 2PL IRT model to the score matrix.

    Steps:
      1. Normalise scores to [0, 1].
      2. Estimate examinee abilities theta_j from mean scores.
      3. For each task, fit (alpha_i, beta_i) via MLE using scipy.minimize.

    Returns DataFrame indexed by task_id with columns alpha, beta.
    """
    # Normalise to [0, 1]
    y_mat = score_mat.values / 100.0  # shape (n_tasks, n_conditions)

    # Estimate abilities: z-score of condition means across tasks
    condition_means = y_mat.mean(axis=0)  # (n_conditions,)
    theta = (condition_means - condition_means.mean()) / (condition_means.std() + 1e-10)

    n_tasks = y_mat.shape[0]
    results = []

    for i in range(n_tasks):
        y_i = y_mat[i, :]

        # Initial guesses
        alpha_init = 1.0
        beta_init = 0.0

        res = minimize(
            _neg_log_likelihood_2pl,
            x0=np.array([alpha_init, beta_init]),
            args=(theta, y_i),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )
        alpha_hat, beta_hat = res.x
        # Ensure alpha > 0 (discrimination must be positive)
        alpha_hat = max(alpha_hat, 0.01)
        results.append({"alpha": alpha_hat, "beta": beta_hat})

    irt_df = pd.DataFrame(results, index=score_mat.index)
    irt_df.index.name = "task_id"
    return irt_df


# ============================================================================
# 3. Calibration check
# ============================================================================

def calibration_analysis(
    irt_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    """Compare assigned difficulty with IRT beta.

    Returns:
        rho: Spearman correlation coefficient
        pval: p-value
        miscalibrated: DataFrame of miscalibrated tasks
    """
    merged = irt_df.join(meta_df[["difficulty", "legacy_category"]], how="left")
    merged["difficulty_num"] = merged["difficulty"].str.lower().map(DIFFICULTY_NUM)
    valid = merged.dropna(subset=["difficulty_num"])

    if len(valid) < 3:
        return np.nan, np.nan, pd.DataFrame()

    rho, pval = spearmanr(valid["difficulty_num"], valid["beta"])

    # Identify miscalibrated tasks
    # Define thresholds: split beta into tertiles for comparison
    beta_thirds = valid["beta"].quantile([1 / 3, 2 / 3]).values
    valid = valid.copy()
    valid["beta_category"] = pd.cut(
        valid["beta"],
        bins=[-np.inf, beta_thirds[0], beta_thirds[1], np.inf],
        labels=["low_beta", "mid_beta", "high_beta"],
    )

    miscalibrated = valid[
        ((valid["difficulty"].str.lower() == "easy") & (valid["beta_category"] == "high_beta"))
        | ((valid["difficulty"].str.lower() == "hard") & (valid["beta_category"] == "low_beta"))
    ].copy()
    miscalibrated = miscalibrated[
        ["difficulty", "difficulty_num", "beta", "beta_category", "legacy_category"]
    ]

    return rho, pval, miscalibrated


# ============================================================================
# 4. Per-category discrimination
# ============================================================================

def per_category_discrimination(
    irt_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean alpha and beta per legacy_category."""
    merged = irt_df.join(meta_df[["legacy_category"]], how="left")
    summary = (
        merged.groupby("legacy_category")
        .agg(
            n_tasks=("alpha", "size"),
            mean_alpha=("alpha", "mean"),
            std_alpha=("alpha", "std"),
            mean_beta=("beta", "mean"),
            std_beta=("beta", "std"),
        )
        .sort_values("mean_alpha", ascending=False)
    )
    return summary


# ============================================================================
# Visualisation
# ============================================================================

def plot_calibration_scatter(
    irt_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    rho: float,
    pval: float,
    output_path: Path,
) -> None:
    """Scatter: assigned difficulty (jittered) vs IRT beta."""
    merged = irt_df.join(meta_df[["difficulty", "legacy_category"]], how="left")
    merged["difficulty_num"] = merged["difficulty"].str.lower().map(DIFFICULTY_NUM)
    valid = merged.dropna(subset=["difficulty_num"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Jitter x for visibility
    jitter = np.random.normal(0, 0.08, size=len(valid))
    x = valid["difficulty_num"].values + jitter

    categories = valid["legacy_category"].unique()
    cmap = plt.cm.tab10
    cat_colors = {cat: cmap(i / max(len(categories) - 1, 1)) for i, cat in enumerate(sorted(categories))}

    for cat in sorted(categories):
        mask = valid["legacy_category"] == cat
        ax.scatter(
            x[mask],
            valid.loc[mask, "beta"],
            c=[cat_colors[cat]],
            label=cat,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
            s=50,
        )

    ax.set_xlabel("Assigned Difficulty (1=Easy, 2=Medium, 3=Hard)", fontsize=12)
    ax.set_ylabel("IRT Difficulty Parameter (beta)", fontsize=12)
    ax.set_title(
        f"Calibration: Assigned Difficulty vs IRT Beta\n"
        f"Spearman rho = {rho:.3f}, p = {pval:.4f}",
        fontsize=13,
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.legend(
        title="Category",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_discrimination_histogram(
    irt_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Histogram of IRT discrimination (alpha) values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        irt_df["alpha"],
        bins=20,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    median_alpha = irt_df["alpha"].median()
    ax.axvline(
        median_alpha,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {median_alpha:.2f}",
    )
    ax.set_xlabel("IRT Discrimination (alpha)", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Distribution of Task Discrimination Parameters", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# Summary text
# ============================================================================

def write_summary(
    ctt_stats: pd.DataFrame,
    irt_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    rho: float,
    pval: float,
    miscalibrated: pd.DataFrame,
    cat_disc: pd.DataFrame,
    output_path: Path,
) -> str:
    """Write and return the summary text."""
    merged = irt_df.join(ctt_stats[["mean_score", "std_score", "discrimination_index"]], how="left")
    merged = merged.join(meta_df[["difficulty", "legacy_category"]], how="left")

    # Top / bottom discriminative (by classical discrimination index)
    top10 = merged.nlargest(10, "discrimination_index")
    bot10 = merged.nsmallest(10, "discrimination_index")

    lines = []
    lines.append("=" * 72)
    lines.append("BDB-052: Task Discrimination Analysis -- IRT Summary")
    lines.append("=" * 72)
    lines.append("")

    # -- Classical --
    lines.append("1. CLASSICAL TEST THEORY")
    lines.append("-" * 40)
    lines.append(f"   Tasks: {len(merged)}")
    lines.append(f"   Mean score (grand): {merged['mean_score'].mean():.1f}")
    lines.append(f"   Mean discrimination index: {merged['discrimination_index'].mean():.3f}")
    lines.append(f"   Median discrimination index: {merged['discrimination_index'].median():.3f}")
    lines.append("")

    lines.append("   Top 10 Most Discriminative Tasks (classical):")
    for tid, row in top10.iterrows():
        lines.append(
            f"     {tid:25s}  DI={row['discrimination_index']:.3f}  "
            f"mean={row['mean_score']:.1f}  std={row['std_score']:.1f}  "
            f"cat={row['legacy_category']}"
        )
    lines.append("")

    lines.append("   Bottom 10 Least Discriminative Tasks (classical):")
    for tid, row in bot10.iterrows():
        lines.append(
            f"     {tid:25s}  DI={row['discrimination_index']:.3f}  "
            f"mean={row['mean_score']:.1f}  std={row['std_score']:.1f}  "
            f"cat={row['legacy_category']}"
        )
    lines.append("")

    # -- IRT 2PL --
    lines.append("2. IRT 2-PARAMETER LOGISTIC MODEL")
    lines.append("-" * 40)
    lines.append(f"   Mean alpha (discrimination): {irt_df['alpha'].mean():.3f}")
    lines.append(f"   Median alpha: {irt_df['alpha'].median():.3f}")
    lines.append(f"   Std alpha: {irt_df['alpha'].std():.3f}")
    lines.append(f"   Range alpha: [{irt_df['alpha'].min():.3f}, {irt_df['alpha'].max():.3f}]")
    lines.append("")
    lines.append(f"   Mean beta (difficulty): {irt_df['beta'].mean():.3f}")
    lines.append(f"   Median beta: {irt_df['beta'].median():.3f}")
    lines.append(f"   Std beta: {irt_df['beta'].std():.3f}")
    lines.append(f"   Range beta: [{irt_df['beta'].min():.3f}, {irt_df['beta'].max():.3f}]")
    lines.append("")

    top10_irt = merged.nlargest(10, "alpha")
    lines.append("   Top 10 Most Discriminative Tasks (IRT alpha):")
    for tid, row in top10_irt.iterrows():
        lines.append(
            f"     {tid:25s}  alpha={row['alpha']:.3f}  beta={row['beta']:.3f}  "
            f"mean={row['mean_score']:.1f}  cat={row['legacy_category']}"
        )
    lines.append("")

    bot10_irt = merged.nsmallest(10, "alpha")
    lines.append("   Bottom 10 Least Discriminative Tasks (IRT alpha):")
    for tid, row in bot10_irt.iterrows():
        lines.append(
            f"     {tid:25s}  alpha={row['alpha']:.3f}  beta={row['beta']:.3f}  "
            f"mean={row['mean_score']:.1f}  cat={row['legacy_category']}"
        )
    lines.append("")

    # -- Calibration --
    lines.append("3. CALIBRATION CHECK")
    lines.append("-" * 40)
    lines.append(f"   Spearman rho (assigned difficulty vs IRT beta): {rho:.3f}")
    lines.append(f"   p-value: {pval:.4f}")
    sig_str = "SIGNIFICANT" if (not np.isnan(pval) and pval < 0.05) else "NOT significant"
    lines.append(f"   Correlation is {sig_str} at alpha=0.05")
    lines.append("")

    if len(miscalibrated) > 0:
        lines.append(f"   Miscalibrated tasks ({len(miscalibrated)}):")
        for tid, row in miscalibrated.iterrows():
            lines.append(
                f"     {tid:25s}  assigned={row['difficulty']:8s}  "
                f"beta={row['beta']:.3f} ({row['beta_category']})  "
                f"cat={row['legacy_category']}"
            )
    else:
        lines.append("   No clearly miscalibrated tasks detected.")
    lines.append("")

    # -- Per-category --
    lines.append("4. PER-CATEGORY DISCRIMINATION")
    lines.append("-" * 40)
    for cat, row in cat_disc.iterrows():
        lines.append(
            f"   {cat:10s}  n={int(row['n_tasks']):2d}  "
            f"mean_alpha={row['mean_alpha']:.3f} +/- {row['std_alpha']:.3f}  "
            f"mean_beta={row['mean_beta']:.3f} +/- {row['std_beta']:.3f}"
        )
    lines.append("")
    lines.append("=" * 72)

    summary_text = "\n".join(lines)
    output_path.write_text(summary_text)
    print(f"  Saved: {output_path}")
    return summary_text


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("BDB-052: Task Discrimination Analysis (IRT)")
    print("=" * 50)

    # -- Load data --
    print("\nLoading data...")
    score_mat = load_score_matrix()
    df_all = load_all()
    print(f"  Score matrix: {score_mat.shape[0]} tasks x {score_mat.shape[1]} conditions")

    # Build per-task metadata (take first row per task)
    meta_df = (
        df_all.groupby("task_id")
        .first()[["difficulty", "legacy_category"]]
    )

    # Align to score matrix index
    meta_df = meta_df.reindex(score_mat.index)

    # -- 1. Classical test theory --
    print("\n1. Classical test theory...")
    ctt_stats = classical_test_theory(score_mat)
    top10_classical = ctt_stats.nlargest(10, "discrimination_index")
    bot10_classical = ctt_stats.nsmallest(10, "discrimination_index")

    print("   Top 10 most discriminative:")
    for tid, row in top10_classical.iterrows():
        print(f"     {tid:25s}  DI={row['discrimination_index']:.3f}  mean={row['mean_score']:.1f}")
    print("   Bottom 10 least discriminative:")
    for tid, row in bot10_classical.iterrows():
        print(f"     {tid:25s}  DI={row['discrimination_index']:.3f}  mean={row['mean_score']:.1f}")

    # -- 2. IRT 2PL --
    print("\n2. Fitting IRT 2PL model...")
    irt_df = fit_irt_2pl(score_mat)
    print(f"   Alpha: mean={irt_df['alpha'].mean():.3f}, median={irt_df['alpha'].median():.3f}")
    print(f"   Beta:  mean={irt_df['beta'].mean():.3f}, median={irt_df['beta'].median():.3f}")

    # -- 3. Calibration --
    print("\n3. Calibration check...")
    rho, pval, miscalibrated = calibration_analysis(irt_df, meta_df)
    print(f"   Spearman rho = {rho:.3f}, p = {pval:.4f}")
    if len(miscalibrated) > 0:
        print(f"   {len(miscalibrated)} miscalibrated tasks found")
    else:
        print("   No miscalibrated tasks")

    # -- 4. Per-category --
    print("\n4. Per-category discrimination...")
    cat_disc = per_category_discrimination(irt_df, meta_df)
    for cat, row in cat_disc.iterrows():
        print(
            f"   {cat:10s}  mean_alpha={row['mean_alpha']:.3f}  mean_beta={row['mean_beta']:.3f}"
        )

    # -- Save outputs --
    print("\nSaving outputs...")

    # CSV: merge IRT params with classical stats and metadata
    out_df = irt_df.copy()
    out_df = out_df.join(ctt_stats[["mean_score", "std_score"]], how="left")
    out_df = out_df.join(meta_df[["difficulty", "legacy_category"]], how="left")
    out_df["assigned_difficulty"] = out_df["difficulty"]
    out_df = out_df[
        ["alpha", "beta", "assigned_difficulty", "legacy_category", "mean_score", "std_score"]
    ]
    csv_path = OUTPUT_DIR / "irt_parameters.csv"
    out_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Plots
    plot_calibration_scatter(
        irt_df, meta_df, rho, pval, OUTPUT_DIR / "calibration_scatter.png"
    )
    plot_discrimination_histogram(irt_df, OUTPUT_DIR / "discrimination_histogram.png")

    # Summary text
    summary = write_summary(
        ctt_stats, irt_df, meta_df, rho, pval, miscalibrated, cat_disc,
        OUTPUT_DIR / "irt_summary.txt",
    )

    print("\n" + summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
