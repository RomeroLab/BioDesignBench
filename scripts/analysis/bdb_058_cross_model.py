#!/usr/bin/env python3
"""BDB-058: Cross-model agreement and task-level score consistency.

Analyzes how 9 evaluation conditions (4 LLMs x 2 modes + Hardcoded Pipeline)
agree on task-level rankings. Identifies consensus tasks (easy/hard for all)
and controversial tasks (large disagreement across conditions).

Analyses:
  1. Spearman rank correlation matrix (9x9 clustermap)
  2. Consensus tasks (lowest cross-condition std)
  3. Controversial tasks (highest cross-condition std)
  4. Split-half reliability (1000 random splits)
  5. Agent clustering dendrogram (hierarchical, 76-dimensional vectors)

Outputs (results/analysis/):
  - cross_model_correlation_heatmap.png
  - consensus_tasks.csv
  - controversial_tasks.csv
  - reliability_coefficient.txt
  - agent_dendrogram.png

Usage:
    python -m scripts.analysis.bdb_058_cross_model
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all, load_score_matrix

# ── Nature-style plot defaults ───────────────────────────────────────────────

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

# ── Analysis functions ───────────────────────────────────────────────────────


def compute_spearman_matrix(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlation between all conditions.

    Args:
        score_matrix: DataFrame with tasks as rows, conditions as columns,
            and total scores as values.

    Returns:
        Square DataFrame (n_conditions x n_conditions) of Spearman rho values.
    """
    conditions = score_matrix.columns.tolist()
    n = len(conditions)
    corr = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x = score_matrix.iloc[:, i].values
            y = score_matrix.iloc[:, j].values
            # Drop rows where either is NaN
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 3:
                corr[i, j] = corr[j, i] = np.nan
            else:
                rho, _ = spearmanr(x[valid], y[valid])
                corr[i, j] = rho
                corr[j, i] = rho

    return pd.DataFrame(corr, index=conditions, columns=conditions)


def _compute_task_stats(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-task summary statistics across conditions.

    Args:
        score_matrix: Tasks (rows) x conditions (columns).

    Returns:
        DataFrame with columns: task_id, mean, std, min, max, range,
        sorted by task_id (unsorted -- caller decides sort order).
    """
    stats = pd.DataFrame(
        {
            "task_id": score_matrix.index,
            "mean": score_matrix.mean(axis=1).values,
            "std": score_matrix.std(axis=1, ddof=1).values,
            "min": score_matrix.min(axis=1).values,
            "max": score_matrix.max(axis=1).values,
        }
    )
    stats["range"] = stats["max"] - stats["min"]
    # Handle single-column case: std will be NaN, fill with 0
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def find_consensus_tasks(
    score_matrix: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find tasks with the highest cross-condition agreement (lowest std).

    Args:
        score_matrix: Tasks (rows) x conditions (columns).
        top_n: Number of top consensus tasks to return. If larger than
            the number of tasks, returns all tasks.

    Returns:
        DataFrame with columns: task_id, mean, std, min, max, range.
        Sorted by std ascending (most consensus first).
    """
    stats = _compute_task_stats(score_matrix)
    stats = stats.sort_values("std", ascending=True).reset_index(drop=True)
    return stats.head(min(top_n, len(stats)))


def find_controversial_tasks(
    score_matrix: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find tasks with the lowest cross-condition agreement (highest std).

    For each controversial task, identifies the condition pair with the
    largest score disagreement.

    Args:
        score_matrix: Tasks (rows) x conditions (columns).
        top_n: Number of top controversial tasks to return.

    Returns:
        DataFrame with columns: task_id, mean, std, min, max, range,
        max_disagree_pair, max_disagree_diff.
        Sorted by std descending (most controversial first).
    """
    stats = _compute_task_stats(score_matrix)
    stats = stats.sort_values("std", ascending=False).reset_index(drop=True)
    stats = stats.head(min(top_n, len(stats)))

    # For each task, find the condition pair with largest disagreement
    conditions = score_matrix.columns.tolist()
    disagree_pairs = []
    disagree_diffs = []

    for _, row in stats.iterrows():
        task_id = row["task_id"]
        scores = score_matrix.loc[task_id]
        max_diff = 0.0
        max_pair = ""

        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                diff = abs(scores.iloc[i] - scores.iloc[j])
                if diff > max_diff:
                    max_diff = diff
                    max_pair = f"{conditions[i]} vs {conditions[j]}"

        disagree_pairs.append(max_pair)
        disagree_diffs.append(max_diff)

    stats["max_disagree_pair"] = disagree_pairs
    stats["max_disagree_diff"] = disagree_diffs
    return stats


def compute_split_half_reliability(
    score_matrix: pd.DataFrame,
    n_splits: int = 1000,
) -> tuple[float, float]:
    """Estimate split-half reliability via random condition splits.

    Splits the 9 conditions into two halves (4 vs 5), computes mean
    task scores per half, then the Spearman correlation between halves.
    Repeats n_splits times and returns the mean and std of the
    reliability coefficients.

    Args:
        score_matrix: Tasks (rows) x conditions (columns).
        n_splits: Number of random splits.

    Returns:
        Tuple of (mean_reliability, std_reliability).
    """
    rng = np.random.default_rng(42)
    n_conds = score_matrix.shape[1]
    reliabilities = []

    for _ in range(n_splits):
        indices = rng.permutation(n_conds)
        half1_idx = indices[: n_conds // 2]
        half2_idx = indices[n_conds // 2 :]

        mean1 = score_matrix.iloc[:, half1_idx].mean(axis=1).values
        mean2 = score_matrix.iloc[:, half2_idx].mean(axis=1).values

        # Drop NaN pairs
        valid = ~(np.isnan(mean1) | np.isnan(mean2))
        if valid.sum() < 3:
            continue

        rho, _ = spearmanr(mean1[valid], mean2[valid])
        if not np.isnan(rho):
            reliabilities.append(rho)

    if len(reliabilities) == 0:
        return 0.0, 0.0

    return float(np.mean(reliabilities)), float(np.std(reliabilities, ddof=1))


# ── Plotting functions ───────────────────────────────────────────────────────


def plot_correlation_heatmap(
    score_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot hierarchically clustered Spearman correlation heatmap.

    Args:
        score_matrix: Tasks (rows) x conditions (columns).
        output_path: Path to save the PNG file.
    """
    corr = compute_spearman_matrix(score_matrix)

    # Use seaborn clustermap for hierarchical clustering
    g = sns.clustermap(
        corr,
        method="average",
        metric="euclidean",
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        figsize=(10, 10),
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.6, "label": "Spearman rho"},
        dendrogram_ratio=(0.12, 0.12),
    )

    g.ax_heatmap.set_title(
        "Cross-Model Spearman Rank Correlation",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8
    )

    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_agent_dendrogram(
    score_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot hierarchical clustering dendrogram on condition score vectors.

    Each condition is represented by its 76-dimensional task score vector.
    Uses Ward's method for linkage.

    Args:
        score_matrix: Tasks (rows) x conditions (columns).
        output_path: Path to save the PNG file.
    """
    # Transpose: conditions as rows, tasks as features
    data = score_matrix.T.fillna(0).values
    labels = score_matrix.columns.tolist()

    Z = linkage(data, method="ward", metric="euclidean")

    fig, ax = plt.subplots(figsize=(10, 6))

    dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=35,
        leaf_font_size=9,
        color_threshold=0,
        above_threshold_color="#4C72B0",
    )

    ax.set_title(
        "Agent Clustering Dendrogram (Ward's Method)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Distance", fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(output_dir: Path | None = None) -> None:
    """Run all cross-model agreement analyses and save outputs.

    Args:
        output_dir: Directory to save results. Defaults to results/analysis/.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Load data
    score_matrix = load_score_matrix()
    n_tasks, n_conditions = score_matrix.shape
    print(f"Loaded score matrix: {n_tasks} tasks x {n_conditions} conditions")
    print()

    # ── 1. Spearman rank correlation matrix ──────────────────────────────
    print("=" * 60)
    print("1. SPEARMAN RANK CORRELATION MATRIX")
    print("=" * 60)

    corr = compute_spearman_matrix(score_matrix)
    print(corr.round(3).to_string())
    print()

    # Off-diagonal stats
    mask = ~np.eye(n_conditions, dtype=bool)
    off_diag = corr.values[mask]
    print(f"  Off-diagonal rho: mean={off_diag.mean():.3f}, "
          f"min={off_diag.min():.3f}, max={off_diag.max():.3f}")

    heatmap_path = output_dir / "cross_model_correlation_heatmap.png"
    plot_correlation_heatmap(score_matrix, heatmap_path)
    print(f"  Saved: {heatmap_path}")
    print()

    # ── 2. Consensus tasks ───────────────────────────────────────────────
    print("=" * 60)
    print("2. CONSENSUS TASKS (top 10 lowest std)")
    print("=" * 60)

    consensus = find_consensus_tasks(score_matrix, top_n=10)
    print(consensus.to_string(index=False, float_format="%.2f"))
    print()

    consensus_path = output_dir / "consensus_tasks.csv"
    consensus.to_csv(consensus_path, index=False, float_format="%.2f")
    print(f"  Saved: {consensus_path}")
    print()

    # ── 3. Controversial tasks ───────────────────────────────────────────
    print("=" * 60)
    print("3. CONTROVERSIAL TASKS (top 10 highest std)")
    print("=" * 60)

    controversial = find_controversial_tasks(score_matrix, top_n=10)
    print(controversial.to_string(index=False, float_format="%.2f"))
    print()

    controversial_path = output_dir / "controversial_tasks.csv"
    controversial.to_csv(controversial_path, index=False, float_format="%.2f")
    print(f"  Saved: {controversial_path}")
    print()

    # ── 4. Split-half reliability ────────────────────────────────────────
    print("=" * 60)
    print("4. SPLIT-HALF RELIABILITY (1000 splits)")
    print("=" * 60)

    mean_r, std_r = compute_split_half_reliability(score_matrix, n_splits=1000)
    reliability_text = (
        f"Split-half reliability (1000 random splits of 9 conditions into 4 vs 5):\n"
        f"  Mean Spearman rho: {mean_r:.4f}\n"
        f"  Std:               {std_r:.4f}\n"
        f"  95% CI:            [{mean_r - 1.96 * std_r:.4f}, {mean_r + 1.96 * std_r:.4f}]\n"
    )
    print(reliability_text)

    reliability_path = output_dir / "reliability_coefficient.txt"
    reliability_path.write_text(reliability_text)
    print(f"  Saved: {reliability_path}")
    print()

    # ── 5. Agent clustering dendrogram ───────────────────────────────────
    print("=" * 60)
    print("5. AGENT CLUSTERING DENDROGRAM")
    print("=" * 60)

    dendrogram_path = output_dir / "agent_dendrogram.png"
    plot_agent_dendrogram(score_matrix, dendrogram_path)
    print(f"  Saved: {dendrogram_path}")
    print()

    # Interpret clustering: do same-LLM pairs cluster together?
    print("  Clustering analysis:")
    data_T = score_matrix.T.fillna(0).values
    labels = score_matrix.columns.tolist()
    Z = linkage(data_T, method="ward", metric="euclidean")

    # Extract cluster assignments at various cut levels
    from scipy.cluster.hierarchy import fcluster

    # Cut into 2..5 clusters and check groupings
    for n_clust in [2, 3, 4, 5]:
        clusters = fcluster(Z, t=n_clust, criterion="maxclust")
        cluster_map = {}
        for label, clust in zip(labels, clusters):
            cluster_map.setdefault(clust, []).append(label)
        print(f"  k={n_clust}: {dict(cluster_map)}")

    print()

    # Check whether same-LLM BM/US pairs are nearest neighbors
    print("  Same-LLM BM/US pair clustering check:")
    llm_pairs = [
        ("DeepSeek V3 benchmark", "DeepSeek V3 user"),
        ("GPT-5 benchmark", "GPT-5 user"),
        ("Sonnet 4.5 benchmark", "Sonnet 4.5 user"),
        ("Gemini 2.5 Pro benchmark", "Gemini 2.5 Pro user"),
    ]
    for bm, us in llm_pairs:
        if bm in labels and us in labels:
            rho = corr.loc[bm, us]
            print(f"    {bm} <-> {us}: rho={rho:.3f}")
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY FINDINGS")
    print("=" * 60)

    # Most correlated pair (off-diagonal)
    corr_vals = corr.copy()
    np.fill_diagonal(corr_vals.values, np.nan)
    max_idx = corr_vals.stack().idxmax()
    max_rho = corr_vals.loc[max_idx[0], max_idx[1]]
    print(f"  - Most correlated pair: {max_idx[0]} & {max_idx[1]} (rho={max_rho:.3f})")

    # Least correlated pair
    min_idx = corr_vals.stack().idxmin()
    min_rho = corr_vals.loc[min_idx[0], min_idx[1]]
    print(f"  - Least correlated pair: {min_idx[0]} & {min_idx[1]} (rho={min_rho:.3f})")

    # Reliability
    print(f"  - Split-half reliability: {mean_r:.3f} +/- {std_r:.3f}")

    # Easiest consensus task
    top_consensus = consensus.iloc[0]
    print(
        f"  - Most consensus task: {top_consensus['task_id']} "
        f"(mean={top_consensus['mean']:.1f}, std={top_consensus['std']:.1f})"
    )

    # Most controversial task
    top_controversial = controversial.iloc[0]
    print(
        f"  - Most controversial task: {top_controversial['task_id']} "
        f"(mean={top_controversial['mean']:.1f}, std={top_controversial['std']:.1f})"
    )
    print()


if __name__ == "__main__":
    main()
