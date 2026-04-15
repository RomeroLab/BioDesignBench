#!/usr/bin/env python3
"""BDB-054: MCP tool usage frequency analysis across 9 conditions.

Analyzes how agents use the 17 MCP protein-design tools, including:
1. Tool frequency heatmap with hierarchical clustering
2. Tool diversity (Shannon entropy) vs performance correlation
3. Per-tool predictiveness of task scores
4. Benchmark vs User mode tool usage delta per LLM

Outputs (saved to results/analysis/):
    - tool_frequency_heatmap.png      : Clustered heatmap of per-tool mean calls
    - tool_diversity_correlation.png  : Entropy vs mean total score scatter
    - tool_predictiveness_table.csv   : Top tools correlated with score
    - mode_tool_usage_delta.csv       : BM vs US tool count differences per LLM
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
# Constants
# ---------------------------------------------------------------------------
MCP_TOOLS = [
    "design_binder",
    "analyze_interface",
    "validate_design",
    "optimize_sequence",
    "suggest_hotspots",
    "get_design_status",
    "predict_complex",
    "predict_structure",
    "score_stability",
    "energy_minimize",
    "generate_backbone",
    "rosetta_score",
    "rosetta_relax",
    "rosetta_interface_score",
    "rosetta_design",
    "predict_structure_boltz",
    "predict_affinity_boltz",
]

# Canonical name lookup: lowercase/variant -> standard name
_CANONICAL: dict[str, str] = {}
for t in MCP_TOOLS:
    _CANONICAL[t] = t
    _CANONICAL[t.lower()] = t
    _CANONICAL[t.replace("_", "-")] = t
    _CANONICAL[t.replace("_", "")] = t


def _normalize_tool_name(raw: str) -> str | None:
    """Normalize a raw tool name to a canonical MCP tool name.

    Returns None if the tool is not an MCP tool (e.g., execute_python,
    read_file, write_file, etc.).
    """
    clean = raw.strip().lower()
    if clean in _CANONICAL:
        return _CANONICAL[clean]
    # Try replacing hyphens with underscores
    alt = clean.replace("-", "_")
    if alt in _CANONICAL:
        return _CANONICAL[alt]
    return None


def _explode_tool_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Explode tool_sequence lists into one row per tool call.

    Adds a 'tool' column with the normalized MCP tool name.
    Rows with non-MCP tools are excluded.

    Args:
        df: Full DataFrame from load_all() with tool_sequence column.

    Returns:
        Exploded DataFrame with columns: condition, task_id, tool, total, mode, llm.
    """
    keep_cols = ["condition", "task_id", "tool_sequence", "total", "mode", "llm"]
    exploded = df[keep_cols].explode("tool_sequence").copy()
    exploded = exploded.rename(columns={"tool_sequence": "raw_tool"})
    exploded = exploded.dropna(subset=["raw_tool"])
    exploded["tool"] = exploded["raw_tool"].apply(_normalize_tool_name)
    # Keep only recognized MCP tools
    exploded = exploded.dropna(subset=["tool"])
    return exploded[["condition", "task_id", "tool", "total", "mode", "llm"]]


def _count_per_tool_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """Count MCP tool calls per tool per (condition, task_id).

    Returns a DataFrame with columns: condition, task_id, tool, call_count, total.
    """
    exploded = _explode_tool_calls(df)
    counts = (
        exploded.groupby(["condition", "task_id", "tool", "total"], observed=False)
        .size()
        .reset_index(name="call_count")
    )
    return counts


# ---------------------------------------------------------------------------
# 1. Tool frequency heatmap
# ---------------------------------------------------------------------------

def compute_tool_frequency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean calls per task for each MCP tool across conditions.

    Returns a DataFrame with conditions as rows and MCP tools as columns,
    values representing mean calls per task.
    """
    exploded = _explode_tool_calls(df)
    n_tasks_per_condition = df.groupby("condition", observed=False)["task_id"].nunique()

    # Total calls per tool per condition
    tool_counts = (
        exploded.groupby(["condition", "tool"], observed=False)
        .size()
        .reset_index(name="total_calls")
    )

    # Normalize by number of tasks
    tool_counts["n_tasks"] = tool_counts["condition"].map(n_tasks_per_condition)
    tool_counts["mean_calls_per_task"] = tool_counts["total_calls"] / tool_counts["n_tasks"]

    # Pivot to matrix
    matrix = tool_counts.pivot_table(
        index="condition",
        columns="tool",
        values="mean_calls_per_task",
        fill_value=0.0,
    )

    # Ensure all MCP tools appear as columns (even if zero usage)
    for tool in MCP_TOOLS:
        if tool not in matrix.columns:
            matrix[tool] = 0.0
    matrix = matrix[MCP_TOOLS]

    return matrix


def plot_tool_frequency_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    """Plot a seaborn clustermap of tool usage frequency.

    Hierarchical clustering is applied to both rows (conditions) and
    columns (tools).

    Args:
        matrix: Conditions x Tools DataFrame of mean calls per task.
        out_path: Path to save the PNG.
    """
    # Drop columns (tools) that are all zeros to avoid clustering issues
    nonzero_cols = matrix.columns[matrix.sum(axis=0) > 0]
    plot_matrix = matrix[nonzero_cols]

    if plot_matrix.empty or plot_matrix.shape[1] < 2:
        # Fallback: plain heatmap if not enough data to cluster
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title("MCP Tool Usage Frequency (mean calls per task)", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # Determine clustering feasibility
    cluster_rows = plot_matrix.shape[0] >= 2
    cluster_cols = plot_matrix.shape[1] >= 2

    g = sns.clustermap(
        plot_matrix,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        figsize=(14, 10),
        linewidths=0.5,
        linecolor="white",
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        dendrogram_ratio=(0.1, 0.12),
        cbar_kws={"label": "Mean calls per task"},
        annot_kws={"fontsize": 8},
    )

    g.ax_heatmap.set_xlabel("MCP Tool", fontsize=12)
    g.ax_heatmap.set_ylabel("Condition", fontsize=12)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9
    )

    g.fig.suptitle(
        "MCP Tool Usage Frequency (mean calls per task)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    g.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


# ---------------------------------------------------------------------------
# 2. Tool diversity (Shannon entropy) vs performance
# ---------------------------------------------------------------------------

def _shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy from an array of counts.

    Args:
        counts: Array of non-negative integers (tool call counts).

    Returns:
        Shannon entropy in bits. Returns 0.0 if total is zero.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Filter out zero-probability entries to avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_tool_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Shannon entropy of MCP tool usage distribution per condition.

    Also computes mean total score per condition for correlation.

    Returns:
        DataFrame with columns: condition, tool_entropy, mean_total.
    """
    exploded = _explode_tool_calls(df)

    rows = []
    for condition in df["condition"].cat.categories:
        cond_calls = exploded[exploded["condition"] == condition]
        cond_df = df[df["condition"] == condition]

        # Count calls per tool
        tool_counts = np.array([
            len(cond_calls[cond_calls["tool"] == t]) for t in MCP_TOOLS
        ])
        entropy = _shannon_entropy(tool_counts)
        mean_total = cond_df["total"].mean()

        rows.append({
            "condition": condition,
            "tool_entropy": entropy,
            "mean_total": mean_total,
        })

    return pd.DataFrame(rows)


def plot_tool_diversity_correlation(
    diversity_df: pd.DataFrame, out_path: Path
) -> tuple[float, float]:
    """Scatter plot: tool entropy vs mean total score with Pearson r.

    Args:
        diversity_df: DataFrame with tool_entropy and mean_total columns.
        out_path: Path to save the PNG.

    Returns:
        Tuple of (pearson_r, p_value).
    """
    x = diversity_df["tool_entropy"].values
    y = diversity_df["mean_total"].values

    # Pearson correlation (requires at least 3 data points)
    if len(x) >= 3:
        r, p = stats.pearsonr(x, y)
    else:
        r, p = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x, y, s=100, c="#4477AA", edgecolors="white", linewidth=1.5, zorder=5)

    # Label each point with condition name
    for _, row in diversity_df.iterrows():
        ax.annotate(
            row["condition"],
            (row["tool_entropy"], row["mean_total"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
            alpha=0.8,
        )

    # Regression line if significant
    if len(x) >= 3 and np.isfinite(r):
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
        y_line = np.polyval(z, x_line)
        ax.plot(x_line, y_line, "--", color="#EE6677", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Tool Diversity (Shannon Entropy, bits)", fontsize=12)
    ax.set_ylabel("Mean Total Score", fontsize=12)

    r_str = f"r = {r:.3f}" if np.isfinite(r) else "r = N/A"
    p_str = f"p = {p:.3f}" if np.isfinite(p) else "p = N/A"
    ax.set_title(
        f"Tool Diversity vs Performance ({r_str}, {p_str})",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return r, p


# ---------------------------------------------------------------------------
# 3. Predictive tools: per-tool correlation with task score
# ---------------------------------------------------------------------------

def compute_tool_predictiveness(df: pd.DataFrame) -> pd.DataFrame:
    """For each MCP tool, compute Pearson r between per-task usage count and score.

    Aggregates across all conditions to maximize statistical power.

    Returns:
        DataFrame with columns: tool, pearson_r, p_value, mean_calls, pct_tasks_used,
        sorted by absolute pearson_r descending.
    """
    exploded = _explode_tool_calls(df)

    results = []
    for tool in MCP_TOOLS:
        # For each (condition, task_id), count calls to this tool
        tool_calls = exploded[exploded["tool"] == tool]
        tool_counts = (
            tool_calls.groupby(["condition", "task_id"], observed=False)
            .size()
            .reset_index(name="call_count")
        )

        # Merge with all (condition, task_id) pairs to get zeros
        all_pairs = df[["condition", "task_id", "total"]].copy()
        merged = all_pairs.merge(
            tool_counts, on=["condition", "task_id"], how="left"
        )
        merged["call_count"] = merged["call_count"].fillna(0)

        x = merged["call_count"].values
        y = merged["total"].values

        # Only compute correlation if there is variance in both
        if np.std(x) > 0 and np.std(y) > 0 and len(x) >= 3:
            r, p = stats.pearsonr(x, y)
        else:
            r, p = 0.0, 1.0

        mean_calls = x.mean()
        pct_used = (x > 0).mean() * 100

        results.append({
            "tool": tool,
            "pearson_r": round(r, 4),
            "p_value": round(p, 6),
            "mean_calls": round(mean_calls, 3),
            "pct_tasks_used": round(pct_used, 1),
        })

    result_df = pd.DataFrame(results)
    result_df["abs_r"] = result_df["pearson_r"].abs()
    result_df = result_df.sort_values("abs_r", ascending=False).drop(columns="abs_r")
    return result_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Benchmark vs User mode tool usage delta
# ---------------------------------------------------------------------------

def compute_mode_tool_delta(df: pd.DataFrame) -> pd.DataFrame:
    """For each LLM, compare per-tool mean calls in BM vs US mode.

    Returns:
        DataFrame with columns: llm, tool, bm_mean, us_mean, delta (us - bm), pct_change.
    """
    # Filter to paired modes only
    paired = df[df["mode"].isin(["benchmark", "user"])].copy()
    if paired.empty:
        return pd.DataFrame(
            columns=["llm", "tool", "bm_mean", "us_mean", "delta", "pct_change"]
        )

    exploded = _explode_tool_calls(paired)

    llms = sorted(paired["llm"].unique())
    rows = []

    for llm in llms:
        for tool in MCP_TOOLS:
            means: dict[str, float] = {}
            for mode_key in ["benchmark", "user"]:
                cond_df = paired[(paired["llm"] == llm) & (paired["mode"] == mode_key)]
                n_tasks = cond_df["task_id"].nunique()
                if n_tasks == 0:
                    means[mode_key] = 0.0
                    continue

                tool_calls = exploded[
                    (exploded["llm"] == llm)
                    & (exploded["mode"] == mode_key)
                    & (exploded["tool"] == tool)
                ]
                means[mode_key] = len(tool_calls) / n_tasks

            bm_mean = means["benchmark"]
            us_mean = means["user"]
            delta = us_mean - bm_mean
            pct = (delta / bm_mean * 100) if bm_mean > 0 else (
                np.inf if us_mean > 0 else 0.0
            )
            rows.append({
                "llm": llm,
                "tool": tool,
                "bm_mean": round(bm_mean, 3),
                "us_mean": round(us_mean, 3),
                "delta": round(delta, 3),
                "pct_change": round(pct, 1) if np.isfinite(pct) else None,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all tool usage analyses and save outputs."""
    print("Loading results...")
    df = load_all()
    print(
        f"  Loaded {len(df)} rows "
        f"({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)"
    )

    # Quick overview of tool_sequence column
    total_tool_calls = df["tool_sequence"].apply(len).sum()
    has_tools = (df["tool_sequence"].apply(len) > 0).sum()
    print(f"  Total tool calls in dataset: {total_tool_calls}")
    print(f"  Rows with at least 1 tool call: {has_tools}/{len(df)}")

    # Count MCP vs non-MCP
    exploded_all = df[["tool_sequence"]].explode("tool_sequence").dropna()
    if len(exploded_all) > 0:
        mcp_mask = exploded_all["tool_sequence"].apply(
            lambda x: _normalize_tool_name(x) is not None
        )
        n_mcp = mcp_mask.sum()
        n_other = (~mcp_mask).sum()
        print(f"  MCP tool calls: {n_mcp}, non-MCP tool calls: {n_other}")
    print()

    # ── 1. Tool frequency heatmap ──────────────────────────────────────────
    print("1. Computing tool frequency matrix...")
    freq_matrix = compute_tool_frequency_matrix(df)
    heatmap_path = OUTPUT_DIR / "tool_frequency_heatmap.png"
    plot_tool_frequency_heatmap(freq_matrix, heatmap_path)
    print(f"  Saved heatmap to {heatmap_path}")
    print("  Top 5 most-used tools (mean across conditions):")
    tool_means = freq_matrix.mean(axis=0).sort_values(ascending=False)
    for tool, val in tool_means.head(5).items():
        print(f"    {tool:30s} {val:.3f} calls/task")
    print()

    # ── 2. Tool diversity vs performance ───────────────────────────────────
    print("2. Computing tool diversity (Shannon entropy)...")
    diversity_df = compute_tool_diversity(df)
    diversity_path = OUTPUT_DIR / "tool_diversity_correlation.png"
    r, p = plot_tool_diversity_correlation(diversity_df, diversity_path)
    print(f"  Saved scatter to {diversity_path}")
    print(f"  Pearson r = {r:.3f}, p = {p:.3f}")
    print("  Per-condition entropy:")
    for _, row in diversity_df.sort_values("tool_entropy", ascending=False).iterrows():
        print(
            f"    {row['condition']:35s}  "
            f"entropy={row['tool_entropy']:.3f}  "
            f"score={row['mean_total']:.1f}"
        )
    print()

    # ── 3. Tool predictiveness ─────────────────────────────────────────────
    print("3. Computing per-tool score predictiveness...")
    pred_df = compute_tool_predictiveness(df)
    pred_path = OUTPUT_DIR / "tool_predictiveness_table.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved table to {pred_path}")
    print("  Top 5 most predictive tools (by |Pearson r|):")
    for _, row in pred_df.head(5).iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(
            f"    {row['tool']:30s}  r={row['pearson_r']:+.4f}{sig}  "
            f"p={row['p_value']:.4f}  "
            f"used_in={row['pct_tasks_used']:.0f}% tasks"
        )
    print()

    # ── 4. BM vs US mode tool delta ────────────────────────────────────────
    print("4. Computing BM vs US tool usage delta...")
    delta_df = compute_mode_tool_delta(df)
    delta_path = OUTPUT_DIR / "mode_tool_usage_delta.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"  Saved table to {delta_path}")

    if not delta_df.empty:
        # Summarize: biggest deltas per LLM
        print("  Largest absolute deltas per LLM:")
        for llm in sorted(delta_df["llm"].unique()):
            llm_df = delta_df[delta_df["llm"] == llm].copy()
            llm_df["abs_delta"] = llm_df["delta"].abs()
            top = llm_df.sort_values("abs_delta", ascending=False).head(3)
            print(f"    {llm}:")
            for _, row in top.iterrows():
                print(
                    f"      {row['tool']:30s}  "
                    f"BM={row['bm_mean']:.3f}  US={row['us_mean']:.3f}  "
                    f"delta={row['delta']:+.3f}"
                )
    print()

    # ── Summary ────────────────────────────────────────────────────────────
    print("=== Summary ===")
    print(f"  Most-used MCP tool overall: {tool_means.idxmax()} ({tool_means.max():.3f} calls/task)")
    least_used = tool_means[tool_means > 0]
    if not least_used.empty:
        print(
            f"  Least-used MCP tool (non-zero): {least_used.idxmin()} "
            f"({least_used.min():.3f} calls/task)"
        )
    unused = tool_means[tool_means == 0].index.tolist()
    if unused:
        print(f"  Never-used MCP tools: {unused}")
    print(f"  Tool diversity-performance correlation: r={r:.3f}, p={p:.3f}")
    if pred_df.iloc[0]["p_value"] < 0.05:
        print(
            f"  Most predictive tool: {pred_df.iloc[0]['tool']} "
            f"(r={pred_df.iloc[0]['pearson_r']:+.4f})"
        )
    else:
        print("  No individual tool significantly predicts task score (p < 0.05)")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
