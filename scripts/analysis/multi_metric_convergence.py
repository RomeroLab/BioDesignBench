#!/usr/bin/env python3
"""Multi-Metric Quality Convergence Analysis for BioDesignBench.

Analyzes whether pipeline-successful tasks converge in quality across agents
using three independent metric families: AF2, ESM-2, and Rosetta.

Outputs:
    figures/fig_convergence_boxplots.pdf       — 3-panel condition × metric boxplot
    figures/fig_convergence_scatter_matrix.pdf  — Pairwise metric correlations
    figures/fig_convergence_approach_quality.pdf — Approach score vs composite quality
    results/analysis/convergence_summary.csv    — Per-condition stats for completed tasks
    results/analysis/convergence_report.md      — Markdown narrative report
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"

# Condition display order and colors
AGENT_CONDITIONS = [
    "Oracle",
    "Hardcoded Pipeline",
    "GPT-5 user",
    "GPT-5 benchmark",
    "Sonnet 4.5 user",
    "Sonnet 4.5 benchmark",
    "DeepSeek V3 user",
    "DeepSeek V3 benchmark",
    "Gemini 2.5 Pro user",
    "Gemini 2.5 Pro benchmark",
    "Human Expert",
]

# Short labels for plots
SHORT_LABELS = {
    "Oracle": "Oracle",
    "Hardcoded Pipeline": "Hardcoded",
    "GPT-5 user": "GPT5-U",
    "GPT-5 benchmark": "GPT5-B",
    "Sonnet 4.5 user": "Son4.5-U",
    "Sonnet 4.5 benchmark": "Son4.5-B",
    "DeepSeek V3 user": "DSV3-U",
    "DeepSeek V3 benchmark": "DSV3-B",
    "Gemini 2.5 Pro user": "Gem-U",
    "Gemini 2.5 Pro benchmark": "Gem-B",
    "Human Expert": "Human",
}

COLORS = {
    "Oracle": "#2d2d2d",
    "Hardcoded Pipeline": "#7f7f7f",
    "GPT-5 user": "#1f77b4",
    "GPT-5 benchmark": "#aec7e8",
    "Sonnet 4.5 user": "#ff7f0e",
    "Sonnet 4.5 benchmark": "#ffbb78",
    "DeepSeek V3 user": "#2ca02c",
    "DeepSeek V3 benchmark": "#98df8a",
    "Gemini 2.5 Pro user": "#d62728",
    "Gemini 2.5 Pro benchmark": "#ff9896",
    "Human Expert": "#9467bd",
}


# ═════════════════════════════════════════════════════════════════════════
# Data loading & merging
# ═════════════════════════════════════════════════════════════════════════


def load_merged() -> pd.DataFrame:
    """Merge rubric scores (load_all) with additional metrics (CSV)."""
    # Rubric scores
    scores = load_all()

    # Additional metrics (task-level aggregated)
    metrics = pd.read_csv(ANALYSIS_DIR / "additional_metrics.csv")

    # Merge on (task_id, condition)
    merged = scores.merge(metrics, on=["task_id", "condition"], how="left")
    return merged


def define_completion(df: pd.DataFrame) -> pd.DataFrame:
    """Define pipeline completion.

    A task is 'completed' if the agent produced at least one valid design
    with a non-NaN Boltz pLDDT (meaning structure prediction ran successfully).
    This aligns with the paper's definition: the pipeline ran to completion
    and produced evaluable output.
    """
    df = df.copy()
    df["completed"] = (
        df["boltz_plddt"].notna()
        & (df["boltz_plddt"] > 0)
        & (df["quality"] > 0)  # got a non-zero quality score
    )
    return df


# ═════════════════════════════════════════════════════════════════════════
# Analysis 1: Conditional quality comparison (completed tasks only)
# ═════════════════════════════════════════════════════════════════════════


def compute_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-condition mean ± std for each metric family."""
    completed = df[df["completed"]].copy()

    rows = []
    for cond in AGENT_CONDITIONS:
        c = completed[completed["condition"] == cond]
        n = len(c)
        if n == 0:
            continue
        rows.append({
            "condition": cond,
            "n_completed": n,
            "n_total": len(df[df["condition"] == cond]),
            "completion_rate": n / len(df[df["condition"] == cond]),
            # AF2 metrics (from Boltz predictions)
            "boltz_plddt_mean": c["boltz_plddt"].mean(),
            "boltz_plddt_std": c["boltz_plddt"].std(),
            "boltz_ptm_mean": c["boltz_ptm"].mean(),
            "boltz_ptm_std": c["boltz_ptm"].std(),
            # ESM-2
            "esm2_ppl_mean": c["esm2_ppl"].mean(),
            "esm2_ppl_std": c["esm2_ppl"].std(),
            # Rosetta
            "rosetta_per_res_mean": c["rosetta_per_res"].mean(),
            "rosetta_per_res_std": c["rosetta_per_res"].std(),
            # Rubric scores
            "approach_mean": c["approach"].mean(),
            "quality_mean": c["quality"].mean(),
            "total_mean": c["total"].mean(),
        })
    return pd.DataFrame(rows)


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis test across conditions for each metric."""
    completed = df[df["completed"]].copy()

    # Only include LLM agent conditions (exclude Oracle, Human Expert, Hardcoded)
    agent_conds = [c for c in AGENT_CONDITIONS
                   if c not in ("Oracle", "Human Expert", "Hardcoded Pipeline")]

    results = {}
    for metric in ["boltz_plddt", "boltz_ptm", "esm2_ppl", "rosetta_per_res"]:
        groups = []
        for cond in agent_conds:
            vals = completed.loc[completed["condition"] == cond, metric].dropna()
            if len(vals) >= 3:
                groups.append(vals.values)

        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            results[metric] = {"H": stat, "p": p, "n_groups": len(groups)}

            # Effect size: eta-squared approximation
            n_total = sum(len(g) for g in groups)
            eta_sq = (stat - len(groups) + 1) / (n_total - len(groups))
            results[metric]["eta_sq"] = max(0, eta_sq)
        else:
            results[metric] = {"H": None, "p": None, "n_groups": len(groups)}

    return results


# ═════════════════════════════════════════════════════════════════════════
# Analysis 2: Metric correlations
# ═════════════════════════════════════════════════════════════════════════


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlations between metric families."""
    completed = df[df["completed"]].copy()

    metrics = {
        "Boltz pLDDT": "boltz_plddt",
        "Boltz pTM": "boltz_ptm",
        "ESM-2 PPL": "esm2_ppl",
        "Rosetta/res": "rosetta_per_res",
    }

    rows = []
    names = list(metrics.keys())
    for i, (n1, c1) in enumerate(metrics.items()):
        for j, (n2, c2) in enumerate(metrics.items()):
            if j <= i:
                continue
            valid = completed[[c1, c2]].dropna()
            if len(valid) < 10:
                continue
            rho, p = stats.spearmanr(valid[c1], valid[c2])
            rows.append({
                "metric_1": n1, "metric_2": n2,
                "spearman_rho": rho, "p_value": p, "n": len(valid),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# Analysis 3: Approach score vs multi-metric quality
# ═════════════════════════════════════════════════════════════════════════


def compute_composite_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Create a composite multi-metric quality score via rank-averaging."""
    completed = df[df["completed"]].copy()

    # Rank each metric (handle direction: higher is better for pLDDT/pTM,
    # lower is better for PPL/Rosetta)
    completed["rank_plddt"] = completed["boltz_plddt"].rank(pct=True)
    completed["rank_ptm"] = completed["boltz_ptm"].rank(pct=True)
    completed["rank_ppl"] = (1 - completed["esm2_ppl"].rank(pct=True))  # invert
    completed["rank_rosetta"] = (1 - completed["rosetta_per_res"].rank(pct=True))  # invert

    completed["composite_quality"] = (
        completed[["rank_plddt", "rank_ptm", "rank_ppl", "rank_rosetta"]].mean(axis=1)
    )
    return completed


# ═════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════


def plot_convergence_boxplots(df: pd.DataFrame):
    """3-panel boxplot: AF2 / ESM-2 / Rosetta for completed tasks by condition."""
    completed = df[df["completed"]].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("boltz_plddt", "Boltz pLDDT", "higher = better"),
        ("esm2_ppl", "ESM-2 Pseudo-Perplexity", "lower = more protein-like"),
        ("rosetta_per_res", "Rosetta Energy / Residue", "lower = more stable"),
    ]

    for ax, (col, title, subtitle) in zip(axes, metrics):
        data = []
        labels = []
        colors = []
        for cond in AGENT_CONDITIONS:
            vals = completed.loc[completed["condition"] == cond, col].dropna()
            if len(vals) == 0:
                continue
            data.append(vals.values)
            labels.append(SHORT_LABELS.get(cond, cond))
            colors.append(COLORS.get(cond, "#999"))

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-Metric Quality Distribution (Pipeline-Completed Tasks Only)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_convergence_boxplots.pdf")
    fig.savefig(FIGURES_DIR / "fig_convergence_boxplots.png")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'fig_convergence_boxplots.pdf'}")


def plot_scatter_matrix(df: pd.DataFrame):
    """Pairwise scatter matrix of 4 metrics, colored by condition type."""
    completed = df[df["completed"]].copy()

    metrics = ["boltz_plddt", "boltz_ptm", "esm2_ppl", "rosetta_per_res"]
    labels = ["Boltz pLDDT", "Boltz pTM", "ESM-2 PPL", "Rosetta/res"]
    n = len(metrics)

    fig, axes = plt.subplots(n, n, figsize=(14, 14))

    # Assign colors
    completed["color"] = completed["condition"].map(COLORS)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                # Histogram on diagonal
                vals = completed[metrics[i]].dropna()
                ax.hist(vals, bins=30, color="#4a86c8", alpha=0.7, edgecolor="white")
                ax.set_ylabel("Count" if j == 0 else "")
            elif i > j:
                # Scatter below diagonal
                valid = completed[[metrics[j], metrics[i], "color"]].dropna()
                ax.scatter(valid[metrics[j]], valid[metrics[i]],
                          c=valid["color"], alpha=0.4, s=12, edgecolors="none")
                # Add correlation
                rho, p = stats.spearmanr(valid[metrics[j]], valid[metrics[i]])
                ax.text(0.05, 0.95, f"ρ={rho:.2f}\np={p:.1e}",
                       transform=ax.transAxes, fontsize=8, va="top",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.axis("off")

            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            if j == 0 and i != j:
                ax.set_ylabel(labels[i], fontsize=9)

    fig.suptitle("Pairwise Metric Correlations (Completed Tasks)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_convergence_scatter_matrix.pdf")
    fig.savefig(FIGURES_DIR / "fig_convergence_scatter_matrix.png")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'fig_convergence_scatter_matrix.pdf'}")


def plot_approach_vs_quality(df: pd.DataFrame):
    """Scatter: Approach score vs composite multi-metric quality."""
    completed = compute_composite_quality(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Approach vs composite
    for cond in AGENT_CONDITIONS:
        c = completed[completed["condition"] == cond]
        if len(c) == 0:
            continue
        axes[0].scatter(c["approach"], c["composite_quality"],
                       c=COLORS.get(cond, "#999"), label=SHORT_LABELS.get(cond, cond),
                       alpha=0.6, s=25, edgecolors="none")

    rho, p = stats.spearmanr(
        completed["approach"].dropna(),
        completed.loc[completed["approach"].notna(), "composite_quality"],
    )
    axes[0].set_xlabel("Approach Score (0–20)")
    axes[0].set_ylabel("Composite Multi-Metric Quality")
    axes[0].set_title(f"Approach vs Composite Quality\n(Spearman ρ={rho:.3f}, p={p:.1e})")
    axes[0].grid(alpha=0.3)

    # Panel 2: Approach vs Boltz pLDDT
    rho2, p2 = stats.spearmanr(
        completed["approach"].dropna(),
        completed.loc[completed["approach"].notna(), "boltz_plddt"],
    )
    for cond in AGENT_CONDITIONS:
        c = completed[completed["condition"] == cond]
        if len(c) == 0:
            continue
        axes[1].scatter(c["approach"], c["boltz_plddt"],
                       c=COLORS.get(cond, "#999"), alpha=0.6, s=25, edgecolors="none")
    axes[1].set_xlabel("Approach Score (0–20)")
    axes[1].set_ylabel("Boltz pLDDT")
    axes[1].set_title(f"Approach vs Boltz pLDDT\n(Spearman ρ={rho2:.3f}, p={p2:.1e})")
    axes[1].grid(alpha=0.3)

    # Panel 3: Approach vs ESM-2 PPL (inverted - lower is better)
    rho3, p3 = stats.spearmanr(
        completed["approach"].dropna(),
        completed.loc[completed["approach"].notna(), "esm2_ppl"],
    )
    for cond in AGENT_CONDITIONS:
        c = completed[completed["condition"] == cond]
        if len(c) == 0:
            continue
        axes[2].scatter(c["approach"], c["esm2_ppl"],
                       c=COLORS.get(cond, "#999"), alpha=0.6, s=25, edgecolors="none")
    axes[2].set_xlabel("Approach Score (0–20)")
    axes[2].set_ylabel("ESM-2 PPL (lower = better)")
    axes[2].set_title(f"Approach vs ESM-2 PPL\n(Spearman ρ={rho3:.3f}, p={p3:.1e})")
    axes[2].grid(alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
              fontsize=8, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Approach Score vs Multi-Metric Quality (Completed Tasks)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_convergence_approach_quality.pdf")
    fig.savefig(FIGURES_DIR / "fig_convergence_approach_quality.png")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'fig_convergence_approach_quality.pdf'}")


# ═════════════════════════════════════════════════════════════════════════
# Report generation
# ═════════════════════════════════════════════════════════════════════════


def generate_report(condition_stats: pd.DataFrame,
                    stat_tests: dict,
                    correlations: pd.DataFrame,
                    df: pd.DataFrame) -> str:
    """Generate markdown report."""
    completed = df[df["completed"]]
    composite = compute_composite_quality(df)

    lines = [
        "# Multi-Metric Quality Convergence Analysis",
        "",
        "## Overview",
        "",
        "This analysis tests whether pipeline-successful tasks converge in quality",
        "across agents when measured with three independent metric families:",
        "- **AF2 (Boltz)**: pLDDT, pTM — structural confidence",
        "- **ESM-2**: pseudo-perplexity — sequence naturalness",
        "- **Rosetta**: energy/residue — physics-based stability",
        "",
        f"**Total tasks**: {len(df)} ({len(df[df['completed']])} completed, "
        f"{len(df[~df['completed']])} incomplete)",
        "",
        "---",
        "",
        "## 1. Pipeline Completion Rates",
        "",
        "| Condition | Completed | Total | Rate |",
        "|-----------|-----------|-------|------|",
    ]

    for _, r in condition_stats.iterrows():
        lines.append(
            f"| {r['condition']} | {r['n_completed']} | {r['n_total']} | "
            f"{r['completion_rate']:.1%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Conditional Quality Comparison (Completed Tasks Only)",
        "",
        "| Condition | N | pLDDT | pTM | ESM-2 PPL | Rosetta/res |",
        "|-----------|---|-------|-----|-----------|-------------|",
    ]

    for _, r in condition_stats.iterrows():
        lines.append(
            f"| {r['condition']} | {r['n_completed']} | "
            f"{r['boltz_plddt_mean']:.3f}±{r['boltz_plddt_std']:.3f} | "
            f"{r['boltz_ptm_mean']:.3f}±{r['boltz_ptm_std']:.3f} | "
            f"{r['esm2_ppl_mean']:.1f}±{r['esm2_ppl_std']:.1f} | "
            f"{r['rosetta_per_res_mean']:.2f}±{r['rosetta_per_res_std']:.2f} |"
        )

    lines += [
        "",
        "### Statistical Tests (Kruskal-Wallis, LLM agents only)",
        "",
        "| Metric | H-statistic | p-value | η² | Interpretation |",
        "|--------|-------------|---------|-----|----------------|",
    ]

    for metric, res in stat_tests.items():
        if res["p"] is not None:
            interp = (
                "No significant difference" if res["p"] > 0.05
                else "Small effect" if res["eta_sq"] < 0.06
                else "Medium effect" if res["eta_sq"] < 0.14
                else "Large effect"
            )
            sig = "ns" if res["p"] > 0.05 else ("*" if res["p"] > 0.01 else
                  ("**" if res["p"] > 0.001 else "***"))
            lines.append(
                f"| {metric} | {res['H']:.1f} | {res['p']:.2e} {sig} | "
                f"{res['eta_sq']:.3f} | {interp} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 3. Metric Correlations",
        "",
        "| Metric 1 | Metric 2 | Spearman ρ | p-value | N |",
        "|----------|----------|------------|---------|---|",
    ]

    for _, r in correlations.iterrows():
        lines.append(
            f"| {r['metric_1']} | {r['metric_2']} | {r['spearman_rho']:.3f} | "
            f"{r['p_value']:.2e} | {r['n']} |"
        )

    # Approach vs composite quality
    rho, p = stats.spearmanr(
        composite["approach"].dropna(),
        composite.loc[composite["approach"].notna(), "composite_quality"],
    )

    lines += [
        "",
        "---",
        "",
        "## 4. Approach Score vs Multi-Metric Quality",
        "",
        f"**Spearman ρ = {rho:.3f}** (p = {p:.2e})",
        "",
        "Interpretation: " + (
            "Strong positive correlation — following validated pipelines yields "
            "better multi-metric quality." if rho > 0.3 and p < 0.05
            else "Moderate correlation — approach matters but doesn't fully "
            "determine quality." if rho > 0.15 and p < 0.05
            else "Weak or no correlation — approach score and multi-metric quality "
            "are largely independent."
        ),
        "",
        "---",
        "",
        "## 5. Key Findings",
        "",
    ]

    # Auto-generate findings
    # Check convergence: compare std across agents vs within agents
    agent_conds = [c for c in AGENT_CONDITIONS
                   if c not in ("Oracle", "Human Expert", "Hardcoded Pipeline")]
    agent_stats = condition_stats[condition_stats["condition"].isin(agent_conds)]

    plddt_range = agent_stats["boltz_plddt_mean"].max() - agent_stats["boltz_plddt_mean"].min()
    ppl_range = agent_stats["esm2_ppl_mean"].max() - agent_stats["esm2_ppl_mean"].min()
    ros_range = agent_stats["rosetta_per_res_mean"].max() - agent_stats["rosetta_per_res_mean"].min()

    lines.append(f"1. **Cross-agent quality range (completed tasks)**:")
    lines.append(f"   - Boltz pLDDT: {plddt_range:.3f} spread across agents")
    lines.append(f"   - ESM-2 PPL: {ppl_range:.1f} spread")
    lines.append(f"   - Rosetta/res: {ros_range:.2f} spread")
    lines.append("")

    # Convergence verdict
    converges_plddt = plddt_range < 0.1
    converges_ppl = ppl_range < 5.0
    converges_ros = ros_range < 2.0

    if converges_plddt and converges_ppl and converges_ros:
        lines.append("2. **CONVERGENCE CONFIRMED**: All three metric families show small ")
        lines.append("   cross-agent spread among completed tasks → tools determine quality.")
    elif converges_plddt and not (converges_ppl or converges_ros):
        lines.append("2. **PARTIAL CONVERGENCE**: AF2 metrics converge but ESM-2/Rosetta reveal ")
        lines.append("   quality differences invisible to AF2 → multi-metric evaluation needed.")
    else:
        lines.append("2. **DIVERGENCE DETECTED**: Significant quality differences persist even ")
        lines.append("   among completed tasks → agent capability matters beyond pipeline completion.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/analysis/multi_metric_convergence.py`*")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def main():
    print("Loading and merging data...")
    df = load_merged()
    df = define_completion(df)
    print(f"  Total: {len(df)} rows, {df['completed'].sum()} completed")

    # Analysis 1: Condition stats
    print("\n1. Computing per-condition statistics...")
    cond_stats = compute_condition_stats(df)
    cond_stats.to_csv(ANALYSIS_DIR / "convergence_summary.csv", index=False)
    print(f"   Saved: convergence_summary.csv")

    # Statistical tests
    print("\n2. Running statistical tests...")
    stat_tests = run_statistical_tests(df)
    for metric, res in stat_tests.items():
        if res["p"] is not None:
            sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 else "*" if res["p"] < 0.05 else "ns"
            print(f"   {metric}: H={res['H']:.1f}, p={res['p']:.2e} {sig}, η²={res['eta_sq']:.3f}")

    # Analysis 2: Correlations
    print("\n3. Computing metric correlations...")
    corrs = compute_correlations(df)
    for _, r in corrs.iterrows():
        print(f"   {r['metric_1']} vs {r['metric_2']}: ρ={r['spearman_rho']:.3f} (p={r['p_value']:.2e})")

    # Plots
    print("\n4. Generating figures...")
    plot_convergence_boxplots(df)
    plot_scatter_matrix(df)
    plot_approach_vs_quality(df)

    # Report
    print("\n5. Generating report...")
    report = generate_report(cond_stats, stat_tests, corrs, df)
    report_path = ANALYSIS_DIR / "convergence_report.md"
    report_path.write_text(report)
    print(f"   Saved: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
