#!/usr/bin/env python3
"""MVP Step Coverage vs Multi-Metric Quality Correlation Analysis.

Tests whether MVP pipeline step coverage predicts multi-metric quality
better than the current reference-based Approach score.

Outputs:
    figures/fig_mvp_quality_scatter.pdf       — 2×2 scatter: MVP vs each metric
    figures/fig_mvp_predictor_comparison.pdf   — Bar: Approach ρ vs MVP ρ
    figures/fig_mvp_backbone_rosetta.pdf       — Boxplot: backbone gen vs Rosetta
    results/analysis/mvp_quality_correlation.csv
    results/analysis/mvp_quality_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from biodesignbench.taxonomy import get_category
from scripts.analysis.load_results import load_all
from scripts.analysis.reviewer_defense import (
    MINIMUM_VIABLE_STEPS,
    AGENT_CONDITIONS,
    SHORT_NAMES,
    COLORS,
    _compute_mvp_coverage,
)

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

ALL_CONDITIONS = AGENT_CONDITIONS + ["Oracle", "Human Expert"]

SHORT_LABELS = {
    **SHORT_NAMES,
    "Oracle": "Oracle",
    "Human Expert": "Human",
}


# ═════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════


def load_merged() -> pd.DataFrame:
    """Merge rubric scores with MVP coverage and additional metrics."""
    scores = load_all()

    # MVP coverage
    scores["mvp_coverage"] = scores.apply(_compute_mvp_coverage, axis=1)

    # Per-step coverage (for stage-level analysis)
    scores["has_backbone_gen"] = scores.apply(
        lambda r: _has_stage(r, "backbone_generation"), axis=1
    )
    scores["has_seq_design"] = scores.apply(
        lambda r: _has_stage(r, "sequence_design"), axis=1
    )
    scores["has_struct_pred"] = scores.apply(
        lambda r: _has_stage(r, "structure_prediction"), axis=1
    )

    # Additional metrics
    metrics = pd.read_csv(ANALYSIS_DIR / "additional_metrics.csv")
    merged = scores.merge(metrics, on=["task_id", "condition"], how="left")

    # Completion flag
    merged["completed"] = (
        merged["boltz_plddt"].notna()
        & (merged["boltz_plddt"] > 0)
        & (merged["quality"] > 0)
    )

    # Composite quality (rank-average, higher = better)
    completed = merged[merged["completed"]].copy()
    completed["rank_plddt"] = completed["boltz_plddt"].rank(pct=True)
    completed["rank_ptm"] = completed["boltz_ptm"].rank(pct=True)
    completed["rank_ppl"] = 1 - completed["esm2_ppl"].rank(pct=True)
    completed["rank_rosetta"] = 1 - completed["rosetta_per_res"].rank(pct=True)
    completed["composite_quality"] = completed[
        ["rank_plddt", "rank_ptm", "rank_ppl", "rank_rosetta"]
    ].mean(axis=1)

    # Merge composite back
    merged = merged.merge(
        completed[["task_id", "condition", "composite_quality"]],
        on=["task_id", "condition"],
        how="left",
    )

    return merged


def _has_stage(row: pd.Series, stage_name: str) -> bool:
    """Check if a specific MVP stage was covered."""
    task_id = row["task_id"]
    cat = get_category(task_id)
    if cat is None:
        return False

    key = (cat.approach.value, cat.subject.value)
    mvp = MINIMUM_VIABLE_STEPS.get(key)
    if mvp is None or stage_name not in mvp:
        return False

    tools_used = row.get("tool_sequence", [])
    if not tools_used:
        return False

    return bool(set(tools_used) & set(mvp[stage_name]))


# ═════════════════════════════════════════════════════════════════════════
# Analysis 1: MVP coverage vs Quality correlations
# ═════════════════════════════════════════════════════════════════════════


def compute_correlations(df: pd.DataFrame, exclude_gemini: bool = False) -> dict:
    """Compute Spearman ρ for MVP coverage vs each quality metric."""
    completed = df[df["completed"]].copy()
    if exclude_gemini:
        completed = completed[~completed["condition"].str.contains("Gemini")]

    # Filter to agent conditions only (with valid MVP coverage)
    completed = completed[completed["mvp_coverage"].notna()]

    results = {}
    targets = {
        "Boltz pLDDT": "boltz_plddt",
        "Boltz pTM": "boltz_ptm",
        "ESM-2 PPL": "esm2_ppl",
        "Rosetta/res": "rosetta_per_res",
        "Composite Quality": "composite_quality",
    }

    for name, col in targets.items():
        valid = completed[[col, "mvp_coverage"]].dropna()
        if len(valid) < 10:
            results[name] = {"rho": np.nan, "p": np.nan, "n": len(valid)}
            continue
        rho, p = sp_stats.spearmanr(valid["mvp_coverage"], valid[col])
        results[name] = {"rho": rho, "p": p, "n": len(valid)}

    # Also compute Approach score vs same targets for comparison
    for name, col in targets.items():
        valid = completed[[col, "approach"]].dropna()
        if len(valid) < 10:
            results[f"Approach→{name}"] = {"rho": np.nan, "p": np.nan, "n": len(valid)}
            continue
        rho, p = sp_stats.spearmanr(valid["approach"], valid[col])
        results[f"Approach→{name}"] = {"rho": rho, "p": p, "n": len(valid)}

    return results


# ═════════════════════════════════════════════════════════════════════════
# Analysis 2: Per-stage quality impact
# ═════════════════════════════════════════════════════════════════════════


def compute_stage_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Compare quality metrics between tasks with/without each stage."""
    completed = df[df["completed"]].copy()
    # Only de_novo tasks for backbone analysis (redesign doesn't have backbone stage)
    de_novo = completed[completed["design_approach"] == "de_novo"]

    rows = []
    for stage, col in [
        ("backbone_generation", "has_backbone_gen"),
        ("sequence_design", "has_seq_design"),
        ("structure_prediction", "has_struct_pred"),
    ]:
        for metric, metric_col in [
            ("Boltz pLDDT", "boltz_plddt"),
            ("ESM-2 PPL", "esm2_ppl"),
            ("Rosetta/res", "rosetta_per_res"),
        ]:
            # Use de_novo for backbone, all for others
            data = de_novo if stage == "backbone_generation" else completed

            with_stage = data[data[col] == True][metric_col].dropna()
            without_stage = data[data[col] == False][metric_col].dropna()

            if len(with_stage) >= 5 and len(without_stage) >= 5:
                u_stat, u_p = sp_stats.mannwhitneyu(
                    with_stage, without_stage, alternative="two-sided"
                )
                # Effect size (rank-biserial correlation)
                n1, n2 = len(with_stage), len(without_stage)
                r_rb = 1 - 2 * u_stat / (n1 * n2)
            else:
                u_stat, u_p, r_rb = np.nan, np.nan, np.nan

            rows.append({
                "stage": stage,
                "metric": metric,
                "with_mean": with_stage.mean() if len(with_stage) > 0 else np.nan,
                "with_std": with_stage.std() if len(with_stage) > 0 else np.nan,
                "with_n": len(with_stage),
                "without_mean": without_stage.mean() if len(without_stage) > 0 else np.nan,
                "without_std": without_stage.std() if len(without_stage) > 0 else np.nan,
                "without_n": len(without_stage),
                "mann_whitney_p": u_p,
                "rank_biserial_r": r_rb,
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════


def plot_mvp_quality_scatter(df: pd.DataFrame):
    """2×2 scatter: MVP coverage vs each quality metric."""
    completed = df[df["completed"] & df["mvp_coverage"].notna()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        ("boltz_plddt", "Boltz pLDDT", True),
        ("boltz_ptm", "Boltz pTM", True),
        ("esm2_ppl", "ESM-2 PPL", False),       # lower is better
        ("rosetta_per_res", "Rosetta/res", False),  # lower is better
    ]

    for ax, (col, label, higher_better) in zip(axes.flat, panels):
        for cond in ALL_CONDITIONS:
            c = completed[completed["condition"] == cond]
            if len(c) == 0:
                continue
            short = SHORT_LABELS.get(cond, cond).replace("\n", " ")
            color = COLORS.get(cond, "#999")
            # Add jitter to MVP coverage for visibility
            jitter = np.random.normal(0, 0.015, len(c))
            ax.scatter(
                c["mvp_coverage"] + jitter, c[col],
                c=color, alpha=0.5, s=20, label=short, edgecolors="none",
            )

        # Correlation
        valid = completed[["mvp_coverage", col]].dropna()
        if len(valid) >= 10:
            rho, p = sp_stats.spearmanr(valid["mvp_coverage"], valid[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(
                0.05, 0.95 if higher_better else 0.05,
                f"ρ={rho:.3f} ({sig})\nn={len(valid)}",
                transform=ax.transAxes, fontsize=9, va="top" if higher_better else "bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        ax.set_xlabel("MVP Step Coverage")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
              fontsize=7, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("MVP Pipeline Step Coverage vs Multi-Metric Quality",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_mvp_quality_scatter.pdf")
    fig.savefig(FIGURES_DIR / "fig_mvp_quality_scatter.png")
    plt.close(fig)
    print(f"Saved: fig_mvp_quality_scatter.pdf")


def plot_predictor_comparison(corrs_all: dict, corrs_no_gem: dict):
    """Bar chart comparing Approach ρ vs MVP ρ for each quality metric."""
    metrics = ["Boltz pLDDT", "Boltz pTM", "ESM-2 PPL", "Rosetta/res", "Composite Quality"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (corrs, title_suffix) in zip(axes, [
        (corrs_all, "All Agents"),
        (corrs_no_gem, "Excluding Gemini"),
    ]):
        x = np.arange(len(metrics))
        width = 0.35

        # MVP coverage rhos
        mvp_rhos = [abs(corrs[m]["rho"]) if not np.isnan(corrs[m]["rho"]) else 0
                    for m in metrics]
        # Approach rhos
        approach_rhos = [abs(corrs[f"Approach→{m}"]["rho"])
                        if not np.isnan(corrs[f"Approach→{m}"]["rho"]) else 0
                        for m in metrics]

        bars1 = ax.bar(x - width/2, approach_rhos, width, label="Approach Score",
                      color="#ff7f0e", alpha=0.8)
        bars2 = ax.bar(x + width/2, mvp_rhos, width, label="MVP Coverage",
                      color="#1f77b4", alpha=0.8)

        # Add significance stars
        for i, m in enumerate(metrics):
            for offset, key in [(-width/2, f"Approach→{m}"), (width/2, m)]:
                p = corrs[key]["p"]
                if p is not None and not np.isnan(p):
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    if sig:
                        val = abs(corrs[key]["rho"])
                        ax.text(i + offset, val + 0.01, sig, ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in metrics], fontsize=8)
        ax.set_ylabel("|Spearman ρ|")
        ax.set_title(f"Predictor Comparison ({title_suffix})")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 0.55)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Which Predicts Quality Better: Approach Score or MVP Coverage?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_mvp_predictor_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig_mvp_predictor_comparison.png")
    plt.close(fig)
    print(f"Saved: fig_mvp_predictor_comparison.pdf")


def plot_backbone_rosetta(df: pd.DataFrame):
    """Boxplot: backbone generation YES/NO vs Rosetta score (de_novo only)."""
    completed = df[df["completed"]].copy()
    de_novo = completed[completed["design_approach"] == "de_novo"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, label) in zip(axes, [
        ("rosetta_per_res", "Rosetta Energy / Residue"),
        ("boltz_plddt", "Boltz pLDDT"),
        ("esm2_ppl", "ESM-2 PPL"),
    ]):
        with_bb = de_novo[de_novo["has_backbone_gen"] == True][col].dropna()
        without_bb = de_novo[de_novo["has_backbone_gen"] == False][col].dropna()

        bp = ax.boxplot(
            [with_bb.values, without_bb.values],
            tick_labels=["With Backbone\nGeneration", "Without Backbone\nGeneration"],
            patch_artist=True, showfliers=False, widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#2ca02c")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#d62728")
        bp["boxes"][1].set_alpha(0.6)

        # Mann-Whitney test
        if len(with_bb) >= 5 and len(without_bb) >= 5:
            _, p = sp_stats.mannwhitneyu(with_bb, without_bb, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(
                0.5, 0.95, f"p={p:.2e} {sig}\nn={len(with_bb)} vs {len(without_bb)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
            )

        ax.set_ylabel(label)
        ax.set_title(f"{label}\n(De Novo Tasks Only)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Effect of Backbone Generation on Design Quality",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_mvp_backbone_rosetta.pdf")
    fig.savefig(FIGURES_DIR / "fig_mvp_backbone_rosetta.png")
    plt.close(fig)
    print(f"Saved: fig_mvp_backbone_rosetta.pdf")


# ═════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════


def generate_report(
    corrs_all: dict,
    corrs_no_gem: dict,
    stage_impact: pd.DataFrame,
    df: pd.DataFrame,
) -> str:
    """Generate markdown report."""
    completed = df[df["completed"]]

    lines = [
        "# MVP Step Coverage vs Multi-Metric Quality",
        "",
        "## Hypothesis",
        "",
        '"MVP step coverage (did the agent execute the essential pipeline stages?)"',
        "is a better predictor of design quality than the current Approach score",
        "(which measures reference-paper matching).",
        "",
        "---",
        "",
        "## 1. MVP Coverage → Quality Correlations",
        "",
        "### All Agents",
        "",
        "| Metric | MVP→Quality ρ | Approach→Quality ρ | MVP better? | N |",
        "|--------|---------------|---------------------|-------------|---|",
    ]

    metrics = ["Boltz pLDDT", "Boltz pTM", "ESM-2 PPL", "Rosetta/res", "Composite Quality"]
    for m in metrics:
        mvp = corrs_all[m]
        app = corrs_all[f"Approach→{m}"]
        mvp_better = abs(mvp["rho"]) > abs(app["rho"]) if not np.isnan(mvp["rho"]) else False
        sig_mvp = _sig(mvp["p"])
        sig_app = _sig(app["p"])
        lines.append(
            f"| {m} | {mvp['rho']:.3f} {sig_mvp} | {app['rho']:.3f} {sig_app} | "
            f"{'**YES**' if mvp_better else 'no'} | {mvp['n']} |"
        )

    lines += [
        "",
        "### Excluding Gemini 2.5 Pro (sensitivity check)",
        "",
        "| Metric | MVP→Quality ρ | Approach→Quality ρ | MVP better? | N |",
        "|--------|---------------|---------------------|-------------|---|",
    ]

    for m in metrics:
        mvp = corrs_no_gem[m]
        app = corrs_no_gem[f"Approach→{m}"]
        mvp_better = abs(mvp["rho"]) > abs(app["rho"]) if not np.isnan(mvp["rho"]) else False
        sig_mvp = _sig(mvp["p"])
        sig_app = _sig(app["p"])
        lines.append(
            f"| {m} | {mvp['rho']:.3f} {sig_mvp} | {app['rho']:.3f} {sig_app} | "
            f"{'**YES**' if mvp_better else 'no'} | {mvp['n']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Pipeline Stage Impact on Quality",
        "",
        "| Stage | Metric | With (mean±std, n) | Without (mean±std, n) | p-value | r |",
        "|-------|--------|--------------------|-----------------------|---------|---|",
    ]

    for _, r in stage_impact.iterrows():
        p_str = f"{r['mann_whitney_p']:.2e}" if not np.isnan(r["mann_whitney_p"]) else "N/A"
        sig = _sig(r["mann_whitney_p"]) if not np.isnan(r["mann_whitney_p"]) else ""
        lines.append(
            f"| {r['stage']} | {r['metric']} | "
            f"{r['with_mean']:.3f}±{r['with_std']:.3f} (n={r['with_n']}) | "
            f"{r['without_mean']:.3f}±{r['without_std']:.3f} (n={r['without_n']}) | "
            f"{p_str} {sig} | {r['rank_biserial_r']:.3f} |"
        )

    # Key findings
    lines += [
        "",
        "---",
        "",
        "## 3. Key Findings",
        "",
    ]

    # Check which predictor wins
    mvp_wins = sum(
        1 for m in metrics
        if not np.isnan(corrs_all[m]["rho"])
        and abs(corrs_all[m]["rho"]) > abs(corrs_all[f"Approach→{m}"]["rho"])
    )
    total = sum(1 for m in metrics if not np.isnan(corrs_all[m]["rho"]))

    if mvp_wins > total / 2:
        lines.append(
            f"1. **MVP coverage is a better quality predictor** in {mvp_wins}/{total} metrics. "
            "Redefining Approach scoring around MVP coverage would improve "
            "alignment with actual design quality."
        )
    elif mvp_wins == total / 2:
        lines.append(
            f"1. **Mixed results**: MVP coverage and Approach score are comparable "
            f"predictors ({mvp_wins}/{total} metrics favor MVP)."
        )
    else:
        lines.append(
            f"1. **Approach score remains the better predictor** in {total - mvp_wins}/{total} metrics. "
            "Reference-paper matching captures quality aspects that MVP coverage misses."
        )

    # Backbone analysis
    bb_ros = stage_impact[
        (stage_impact["stage"] == "backbone_generation") &
        (stage_impact["metric"] == "Rosetta/res")
    ]
    if len(bb_ros) > 0:
        row = bb_ros.iloc[0]
        if not np.isnan(row["mann_whitney_p"]) and row["mann_whitney_p"] < 0.05:
            lines.append(
                f"\n2. **Backbone generation significantly affects Rosetta energy** "
                f"(p={row['mann_whitney_p']:.2e}): with backbone = {row['with_mean']:.2f}, "
                f"without = {row['without_mean']:.2f} per residue."
            )
        else:
            lines.append(
                "\n2. **Backbone generation does not significantly affect Rosetta energy** "
                f"(p={row['mann_whitney_p']:.2e})."
            )

    # Gemini sensitivity
    composite_all = corrs_all.get("Composite Quality", {})
    composite_no_gem = corrs_no_gem.get("Composite Quality", {})
    if not np.isnan(composite_all.get("rho", np.nan)) and not np.isnan(composite_no_gem.get("rho", np.nan)):
        delta = abs(composite_no_gem["rho"]) - abs(composite_all["rho"])
        if abs(delta) > 0.05:
            lines.append(
                f"\n3. **Gemini exclusion changes results**: MVP→Composite ρ shifts from "
                f"{composite_all['rho']:.3f} to {composite_no_gem['rho']:.3f} (Δ={delta:+.3f})."
            )
        else:
            lines.append(
                f"\n3. **Results robust to Gemini exclusion**: MVP→Composite ρ = "
                f"{composite_all['rho']:.3f} → {composite_no_gem['rho']:.3f} (Δ={delta:+.3f})."
            )

    lines += [
        "",
        "---",
        "",
        "*Generated by `scripts/analysis/mvp_quality_correlation.py`*",
    ]
    return "\n".join(lines)


def _sig(p) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def main():
    print("Loading and merging data...")
    df = load_merged()
    completed = df[df["completed"]]
    print(f"  Total: {len(df)} rows, {len(completed)} completed")
    print(f"  MVP coverage available: {df['mvp_coverage'].notna().sum()}")

    # Analysis 1: Correlations
    print("\n1. MVP Coverage → Quality correlations...")
    corrs_all = compute_correlations(df, exclude_gemini=False)
    corrs_no_gem = compute_correlations(df, exclude_gemini=True)

    print("\n  All agents:")
    for name in ["Boltz pLDDT", "Boltz pTM", "ESM-2 PPL", "Rosetta/res", "Composite Quality"]:
        mvp = corrs_all[name]
        app = corrs_all[f"Approach→{name}"]
        winner = "MVP" if abs(mvp["rho"]) > abs(app["rho"]) else "Approach"
        print(f"    {name}: MVP ρ={mvp['rho']:.3f}, Approach ρ={app['rho']:.3f} → {winner}")

    print("\n  Excluding Gemini:")
    for name in ["Boltz pLDDT", "Boltz pTM", "ESM-2 PPL", "Rosetta/res", "Composite Quality"]:
        mvp = corrs_no_gem[name]
        app = corrs_no_gem[f"Approach→{name}"]
        winner = "MVP" if abs(mvp["rho"]) > abs(app["rho"]) else "Approach"
        print(f"    {name}: MVP ρ={mvp['rho']:.3f}, Approach ρ={app['rho']:.3f} → {winner}")

    # Analysis 2: Stage impact
    print("\n2. Per-stage quality impact...")
    stage_impact = compute_stage_impact(df)
    for _, r in stage_impact.iterrows():
        p_str = f"p={r['mann_whitney_p']:.2e}" if not np.isnan(r["mann_whitney_p"]) else "N/A"
        print(f"    {r['stage']} → {r['metric']}: {p_str}")

    # Save CSV
    corr_rows = []
    for label, corrs in [("all_agents", corrs_all), ("excl_gemini", corrs_no_gem)]:
        for name in ["Boltz pLDDT", "Boltz pTM", "ESM-2 PPL", "Rosetta/res", "Composite Quality"]:
            corr_rows.append({
                "subset": label,
                "metric": name,
                "mvp_rho": corrs[name]["rho"],
                "mvp_p": corrs[name]["p"],
                "approach_rho": corrs[f"Approach→{name}"]["rho"],
                "approach_p": corrs[f"Approach→{name}"]["p"],
                "n": corrs[name]["n"],
            })
    pd.DataFrame(corr_rows).to_csv(
        ANALYSIS_DIR / "mvp_quality_correlation.csv", index=False
    )
    stage_impact.to_csv(
        ANALYSIS_DIR / "mvp_stage_impact.csv", index=False
    )
    print("\n  Saved CSVs")

    # Plots
    print("\n3. Generating figures...")
    plot_mvp_quality_scatter(df)
    plot_predictor_comparison(corrs_all, corrs_no_gem)
    plot_backbone_rosetta(df)

    # Report
    print("\n4. Generating report...")
    report = generate_report(corrs_all, corrs_no_gem, stage_impact, df)
    report_path = ANALYSIS_DIR / "mvp_quality_report.md"
    report_path.write_text(report)
    print(f"   Saved: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
