#!/usr/bin/env python3
"""BDB-050: Component Variance Decomposition.

Decomposes benchmark score variance across the 6 scoring components
(approach, orchestration, quality, feasibility, novelty, diversity)
to understand which dimensions drive total score differences.

Analyses:
  1. Pearson/Spearman correlation matrix (6 components + total)
  2. PCA on 6 components with scree plot
  3. ANOVA-style R^2 decomposition per component
  4. Per-agent (condition) variance decomposition

Outputs (results/analysis/):
  - variance_decomposition_table.csv
  - component_correlation_heatmap.png
  - pca_explained_variance.png
  - per_agent_variance.csv

Usage:
    python -m scripts.analysis.bdb_050_variance_decomposition
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all

# ── Constants ────────────────────────────────────────────────────────────────

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]
MAX_POINTS = {
    "approach": 20,
    "orchestration": 15,
    "quality": 35,
    "feasibility": 15,
    "novelty": 5,
    "diversity": 10,
}
SCORE_COLS = COMPONENTS + ["total"]

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


def compute_correlations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices for score columns.

    Args:
        df: DataFrame with columns in SCORE_COLS.

    Returns:
        Tuple of (pearson_corr, spearman_corr), each a 7x7 DataFrame.
    """
    score_data = df[SCORE_COLS]
    pearson = score_data.corr(method="pearson")
    spearman = score_data.corr(method="spearman")
    return pearson, spearman


def run_pca(df: pd.DataFrame) -> dict[str, Any]:
    """Run PCA on the 6 scoring components.

    Args:
        df: DataFrame with columns in COMPONENTS.

    Returns:
        Dict with keys: explained_variance_ratio, loadings, pca_object.
    """
    X = df[COMPONENTS].values.astype(float)
    # Standardize before PCA
    X_centered = X - X.mean(axis=0)
    std = X.std(axis=0)
    # Guard against zero std (constant columns)
    std[std == 0] = 1.0
    X_scaled = X_centered / std

    pca = PCA(n_components=len(COMPONENTS))
    pca.fit(X_scaled)

    loadings = pd.DataFrame(
        pca.components_,
        columns=COMPONENTS,
        index=[f"PC{i + 1}" for i in range(len(COMPONENTS))],
    )

    return {
        "explained_variance_ratio": list(pca.explained_variance_ratio_),
        "loadings": loadings,
        "pca_object": pca,
    }


def compute_r_squared(df: pd.DataFrame) -> dict[str, float]:
    """Compute R^2 of each component predicting total score independently.

    For each component, fits a univariate linear regression:
        total ~ component
    and returns the R^2 value.

    Args:
        df: DataFrame with columns in COMPONENTS and 'total'.

    Returns:
        Dict mapping component name to R^2 value.
    """
    y = df["total"].values.reshape(-1, 1)
    r2_dict: dict[str, float] = {}

    for comp in COMPONENTS:
        X = df[comp].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        r2 = float(model.score(X, y))
        r2_dict[comp] = r2

    return r2_dict


def compute_per_agent_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute std of each component per condition across tasks.

    Args:
        df: DataFrame with 'condition' column and COMPONENTS.

    Returns:
        DataFrame with conditions as rows, components as columns,
        values are standard deviations.
    """
    result = df.groupby("condition", observed=True)[COMPONENTS].std(ddof=1)
    # Fill NaN (e.g., single-task conditions) with 0
    result = result.fillna(0.0)
    return result


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the main variance decomposition summary table.

    Combines correlation with total, R^2, and normalized variance
    contribution into a single table.

    Args:
        df: DataFrame with SCORE_COLS columns.

    Returns:
        DataFrame with columns: component, max_points, pearson_r,
        spearman_rho, r_squared, variance_contribution_pct.
    """
    pearson, spearman = compute_correlations(df)
    r2_dict = compute_r_squared(df)

    # Normalize R^2 to sum to 100%
    r2_total = sum(r2_dict.values())
    if r2_total == 0:
        r2_total = 1.0  # Avoid division by zero

    rows = []
    for comp in COMPONENTS:
        rows.append(
            {
                "component": comp,
                "max_points": MAX_POINTS[comp],
                "pearson_r": round(float(pearson.loc[comp, "total"]), 4),
                "spearman_rho": round(float(spearman.loc[comp, "total"]), 4),
                "r_squared": round(r2_dict[comp], 4),
                "variance_contribution_pct": round(
                    r2_dict[comp] / r2_total * 100, 2
                ),
            }
        )

    return pd.DataFrame(rows)


# ── Plotting functions ───────────────────────────────────────────────────────


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot annotated correlation heatmap for 6 components + total.

    Args:
        df: DataFrame with SCORE_COLS columns.
        output_path: Path to save the PNG file.
    """
    pearson, _ = compute_correlations(df)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Use a diverging colormap centered at 0
    mask = np.zeros_like(pearson.values, dtype=bool)
    sns.heatmap(
        pearson,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"size": 7},
    )

    ax.set_title("Component Correlation Matrix", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pca_scree(
    pca_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot PCA scree plot with explained variance ratios.

    Args:
        pca_result: Output from run_pca().
        output_path: Path to save the PNG file.
    """
    ratios = pca_result["explained_variance_ratio"]
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Bar chart for individual variance
    bars = ax.bar(
        x,
        [r * 100 for r in ratios],
        color="#4C72B0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Individual",
        zorder=3,
    )

    # Line for cumulative
    ax.plot(
        x,
        [c * 100 for c in cumulative],
        "o-",
        color="#C44E52",
        markersize=5,
        linewidth=1.5,
        label="Cumulative",
        zorder=4,
    )

    # Annotate bar values
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{ratio * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xlabel("Principal Component", fontsize=9)
    ax.set_ylabel("Explained Variance (%)", fontsize=9)
    ax.set_title("PCA Explained Variance (Scree Plot)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(output_dir: Path | None = None) -> None:
    """Run all variance decomposition analyses and save outputs.

    Args:
        output_dir: Directory to save results. Defaults to results/analysis/.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_all()
    n_tasks = df["task_id"].nunique()
    n_conditions = df["condition"].nunique()
    print(f"Loaded {len(df)} rows ({n_tasks} tasks x {n_conditions} conditions)")
    print()

    # ── 1. Correlation analysis ──────────────────────────────────────────
    pearson, spearman = compute_correlations(df)
    print("=" * 60)
    print("1. COMPONENT-TOTAL CORRELATIONS")
    print("=" * 60)
    print(f"  {'Component':<15s} {'Pearson r':>10s} {'Spearman rho':>13s}")
    print("  " + "-" * 40)
    for comp in COMPONENTS:
        print(
            f"  {comp:<15s} {pearson.loc[comp, 'total']:>10.3f} "
            f"{spearman.loc[comp, 'total']:>13.3f}"
        )
    print()

    # ── 2. PCA ───────────────────────────────────────────────────────────
    pca_result = run_pca(df)
    print("=" * 60)
    print("2. PCA EXPLAINED VARIANCE")
    print("=" * 60)
    cumulative = 0.0
    for i, ratio in enumerate(pca_result["explained_variance_ratio"]):
        cumulative += ratio
        print(f"  PC{i + 1}: {ratio * 100:5.1f}%  (cumulative: {cumulative * 100:5.1f}%)")
    print()

    # Top loadings for PC1
    loadings = pca_result["loadings"]
    pc1_loadings = loadings.loc["PC1"].abs().sort_values(ascending=False)
    print("  PC1 top loadings (absolute):")
    for comp, val in pc1_loadings.items():
        sign = "+" if loadings.loc["PC1", comp] > 0 else "-"
        print(f"    {comp:<15s} {sign}{val:.3f}")
    print()

    # ── 3. R^2 decomposition ─────────────────────────────────────────────
    r2_dict = compute_r_squared(df)
    print("=" * 60)
    print("3. ANOVA-STYLE R^2 DECOMPOSITION")
    print("=" * 60)
    print(f"  {'Component':<15s} {'Max pts':>8s} {'R^2':>8s} {'Var. contrib.':>14s}")
    print("  " + "-" * 48)
    r2_total = sum(r2_dict.values())
    for comp in sorted(r2_dict, key=r2_dict.get, reverse=True):  # type: ignore[arg-type]
        pct = r2_dict[comp] / r2_total * 100 if r2_total > 0 else 0
        print(
            f"  {comp:<15s} {MAX_POINTS[comp]:>8d} {r2_dict[comp]:>8.3f} {pct:>13.1f}%"
        )
    print()

    # ── 4. Per-agent variance ────────────────────────────────────────────
    agent_var = compute_per_agent_variance(df)
    print("=" * 60)
    print("4. PER-AGENT COMPONENT STD (across tasks)")
    print("=" * 60)
    print(f"  {'Condition':<30s}", end="")
    for comp in COMPONENTS:
        print(f" {comp[:5]:>6s}", end="")
    print()
    print("  " + "-" * 68)
    for cond in agent_var.index:
        print(f"  {str(cond):<30s}", end="")
        for comp in COMPONENTS:
            print(f" {agent_var.loc[cond, comp]:>6.2f}", end="")
        print()
    print()

    # ── Build and save summary table ─────────────────────────────────────
    summary = build_summary_table(df)
    summary_path = output_dir / "variance_decomposition_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # ── Save per-agent variance CSV ──────────────────────────────────────
    agent_var_path = output_dir / "per_agent_variance.csv"
    agent_var.to_csv(agent_var_path)
    print(f"Saved: {agent_var_path}")

    # ── Plot correlation heatmap ─────────────────────────────────────────
    heatmap_path = output_dir / "component_correlation_heatmap.png"
    plot_correlation_heatmap(df, heatmap_path)
    print(f"Saved: {heatmap_path}")

    # ── Plot PCA scree ───────────────────────────────────────────────────
    scree_path = output_dir / "pca_explained_variance.png"
    plot_pca_scree(pca_result, scree_path)
    print(f"Saved: {scree_path}")

    # ── Summary findings ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY FINDINGS")
    print("=" * 60)

    # Highest R^2 component
    top_comp = max(r2_dict, key=r2_dict.get)  # type: ignore[arg-type]
    top_r2 = r2_dict[top_comp]
    top_pct = top_r2 / r2_total * 100 if r2_total > 0 else 0
    print(
        f"  - '{top_comp}' (max {MAX_POINTS[top_comp]} pts) explains "
        f"{top_pct:.1f}% of total score variance (R^2={top_r2:.3f})"
    )

    # Lowest R^2 component
    bot_comp = min(r2_dict, key=r2_dict.get)  # type: ignore[arg-type]
    bot_r2 = r2_dict[bot_comp]
    bot_pct = bot_r2 / r2_total * 100 if r2_total > 0 else 0
    print(
        f"  - '{bot_comp}' (max {MAX_POINTS[bot_comp]} pts) explains "
        f"{bot_pct:.1f}% of total score variance (R^2={bot_r2:.3f})"
    )

    # PCA dimensionality
    cum_90 = 0
    for i, r in enumerate(pca_result["explained_variance_ratio"]):
        cum_90 += r
        if cum_90 >= 0.9:
            print(f"  - {i + 1} principal components explain >=90% of variance")
            break

    # Highest inter-component correlation (off-diagonal)
    pearson_no_total = pearson.loc[COMPONENTS, COMPONENTS]
    mask = np.triu(np.ones_like(pearson_no_total, dtype=bool), k=1)
    upper_vals = pearson_no_total.where(mask)
    max_corr_idx = upper_vals.stack().abs().idxmax()
    max_corr_val = pearson_no_total.loc[max_corr_idx[0], max_corr_idx[1]]
    print(
        f"  - Highest inter-component correlation: "
        f"{max_corr_idx[0]}-{max_corr_idx[1]} (r={max_corr_val:.3f})"
    )

    # Most variable component across agents
    mean_std = agent_var.mean()
    most_variable = mean_std.idxmax()
    print(
        f"  - Most variable component across tasks: "
        f"'{most_variable}' (mean std={mean_std[most_variable]:.2f})"
    )
    print()


if __name__ == "__main__":
    main()
