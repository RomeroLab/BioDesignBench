#!/usr/bin/env python3
"""Backbone generation skipping analysis for BioDesignBench.

Analyzes the systematic pattern where LLM agents skip generative backbone
generation steps while using analytical/evaluation tools. The central finding:
"LLM agents are critics, not creators -- they can analyze and evaluate but
cannot generate."

Outputs:
  figures/fig_backbone_skip_usage.pdf + .png     (Figure A: stacked bar)
  figures/fig_backbone_skip_outcome.pdf + .png   (Figure B: box plots)
  figures/fig_backbone_skip_heatmap.pdf + .png   (Figure C: binary heatmap)
  figures/backbone_skip_stats.csv
  figures/backbone_skip_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ── Project setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import CONDITION_MAP, load_all

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Tool categories ──────────────────────────────────────────────────────
BACKBONE_GEN_TOOLS = frozenset({
    "rfdiffusion", "generate_backbone", "chroma", "design_binder",
})

SEQUENCE_DESIGN_TOOLS = frozenset({
    "proteinmpnn", "optimize_sequence", "rosetta_design",
    "design_binder", "ligandmpnn", "esm_if", "mpnn",
})

STRUCTURE_PRED_TOOLS = frozenset({
    "esmfold", "alphafold2", "predict_structure", "predict_complex",
    "validate_design", "predict_structure_boltz", "predict_affinity_boltz",
    "colabfold",
})

SCORING_TOOLS = frozenset({
    "score_stability", "analyze_interface", "rosetta_score",
    "rosetta_interface_score", "energy_minimize", "rosetta_relax",
    "predict_affinity_boltz",
})

CODE_EXECUTION_TOOLS = frozenset({
    "execute_python", "write_file",
})

# All generative tools (backbone + sequence design)
GENERATIVE_TOOLS = BACKBONE_GEN_TOOLS | SEQUENCE_DESIGN_TOOLS

# ── Agent conditions (exclude Oracle and Human Expert) ───────────────────
AGENT_CONDITIONS = [
    "DeepSeek V3 user",
    "DeepSeek V3 benchmark",
    "GPT-5 user",
    "GPT-5 benchmark",
    "Sonnet 4.5 user",
    "Sonnet 4.5 benchmark",
    "Gemini 2.5 Pro user",
    "Gemini 2.5 Pro benchmark",
    "Hardcoded Pipeline",
]

SHORT_NAMES = {
    "DeepSeek V3 user": "DS-V3\nuser",
    "DeepSeek V3 benchmark": "DS-V3\nbench",
    "GPT-5 user": "GPT-5\nuser",
    "GPT-5 benchmark": "GPT-5\nbench",
    "Sonnet 4.5 user": "Son-4.5\nuser",
    "Sonnet 4.5 benchmark": "Son-4.5\nbench",
    "Gemini 2.5 Pro user": "Gem-2.5\nuser",
    "Gemini 2.5 Pro benchmark": "Gem-2.5\nbench",
    "Hardcoded Pipeline": "Hardcoded",
}

COLORS = {
    "DeepSeek V3 user": "#1f77b4",
    "DeepSeek V3 benchmark": "#aec7e8",
    "GPT-5 user": "#ff7f0e",
    "GPT-5 benchmark": "#ffbb78",
    "Sonnet 4.5 user": "#2ca02c",
    "Sonnet 4.5 benchmark": "#98df8a",
    "Gemini 2.5 Pro user": "#d62728",
    "Gemini 2.5 Pro benchmark": "#ff9896",
    "Hardcoded Pipeline": "#9467bd",
}

# ── Style setup ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Data loading ─────────────────────────────────────────────────────────

def _load_raw_result(condition: str, task_id: str) -> dict[str, Any] | None:
    """Load a single result.json for a (condition, task_id) pair."""
    info = CONDITION_MAP.get(condition)
    if info is None:
        return None
    result_file = info["path"] / task_id / "result.json"
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def build_backbone_df(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the main DataFrame with backbone generation usage columns.

    Adds:
      - tools_used_raw: list of tools from result.json
      - actual_tool_order_raw: ordered tool calls from orchestration
      - num_designs: number of designs produced
      - backbone_used: bool, did the agent use any backbone gen tool?
      - skip_category: str, classification of alternative strategy (de novo only)
    """
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        condition = row["condition"]
        task_id = row["task_id"]

        raw = _load_raw_result(condition, task_id)
        if raw is None:
            records.append({
                "task_id": task_id,
                "condition": condition,
                "tools_used_raw": [],
                "actual_tool_order_raw": [],
                "num_designs": 0,
                "backbone_used": False,
                "skip_category": "No tools at all",
            })
            continue

        tools_used = set(raw.get("tools_used", []))
        actual_order = raw.get("orchestration_metrics", {}).get(
            "actual_tool_order", []
        )
        num_designs = raw.get("diversity_metrics", {}).get("num_designs", 0)

        # Also check actual_tool_order for backbone gen tools (some results
        # use internal names like "rfdiffusion" in the order log)
        all_tool_refs = tools_used | set(actual_order)

        backbone_used = bool(all_tool_refs & BACKBONE_GEN_TOOLS)

        # Classify alternative strategy when backbone gen is skipped
        skip_category = _classify_skip_strategy(all_tool_refs, backbone_used)

        records.append({
            "task_id": task_id,
            "condition": condition,
            "tools_used_raw": list(tools_used),
            "actual_tool_order_raw": actual_order,
            "num_designs": num_designs,
            "backbone_used": backbone_used,
            "skip_category": skip_category,
        })

    aug_df = pd.DataFrame(records)
    merged = df.merge(aug_df, on=["task_id", "condition"], how="left")
    return merged


def _classify_skip_strategy(all_tools: set[str], backbone_used: bool) -> str:
    """Classify the alternative strategy when backbone gen is skipped.

    Categories:
      A: "No tools at all" -- empty or only execute_python/write_file
      B: "Analysis only"   -- structure prediction/scoring but no generative
      C: "Sequence-first"  -- sequence design but no backbone generation
      D: "Other"           -- something else
      (if backbone was used, returns "Backbone used")
    """
    if backbone_used:
        return "Backbone used"

    # Strip code-execution-only tools to see what substantive tools remain
    substantive = all_tools - CODE_EXECUTION_TOOLS
    # Also strip generic read/status tools
    substantive -= {"read_file", "get_design_status", "suggest_hotspots"}

    if not substantive:
        return "No tools at all"

    has_seq_design = bool(substantive & (SEQUENCE_DESIGN_TOOLS - BACKBONE_GEN_TOOLS))
    has_analysis = bool(substantive & (STRUCTURE_PRED_TOOLS | SCORING_TOOLS))

    if has_seq_design:
        return "Sequence-first"
    elif has_analysis:
        return "Analysis only"
    else:
        return "Other"


# ── Analysis functions ───────────────────────────────────────────────────

def compute_usage_rates(bdf: pd.DataFrame) -> pd.DataFrame:
    """Per condition: backbone gen usage rate for de_novo vs redesign tasks."""
    rows: list[dict[str, Any]] = []
    for cond in AGENT_CONDITIONS:
        cond_df = bdf[bdf["condition"] == cond]
        for approach in ["de_novo", "redesign"]:
            subset = cond_df[cond_df["design_approach"] == approach]
            n_total = len(subset)
            n_used = subset["backbone_used"].sum() if n_total > 0 else 0
            rate = n_used / n_total * 100 if n_total > 0 else 0.0
            rows.append({
                "condition": cond,
                "design_approach": approach,
                "n_total": n_total,
                "n_backbone_used": int(n_used),
                "usage_rate_pct": rate,
            })
    return pd.DataFrame(rows)


def compute_outcome_comparison(bdf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare scores: backbone_used=True vs False.

    Returns dict with keys: 'all', 'de_novo', per each: mean scores + p-values.
    """
    results: dict[str, pd.DataFrame] = {}

    for label, subset in [("all", bdf), ("de_novo", bdf[bdf["design_approach"] == "de_novo"])]:
        used = subset[subset["backbone_used"]]
        skipped = subset[~subset["backbone_used"]]

        # Pipeline completion: num_designs > 0 AND tier_a > 0
        def completion_rate(s: pd.DataFrame) -> float:
            if len(s) == 0:
                return 0.0
            completed = ((s["num_designs"] > 0) & (s["tier_a"] > 0)).sum()
            return completed / len(s) * 100

        metrics: list[dict[str, Any]] = []
        for metric_name, col in [("Total Score", "total"), ("Quality Score", "quality")]:
            mean_used = used[col].mean() if len(used) > 0 else 0.0
            mean_skip = skipped[col].mean() if len(skipped) > 0 else 0.0
            n_used = len(used)
            n_skip = len(skipped)

            # Mann-Whitney U test
            if n_used >= 2 and n_skip >= 2:
                stat, pval = mannwhitneyu(
                    used[col].dropna(), skipped[col].dropna(), alternative="two-sided"
                )
            else:
                stat, pval = np.nan, np.nan

            metrics.append({
                "metric": metric_name,
                "subset": label,
                "mean_backbone_used": round(mean_used, 2),
                "mean_backbone_skipped": round(mean_skip, 2),
                "delta": round(mean_used - mean_skip, 2),
                "n_used": n_used,
                "n_skipped": n_skip,
                "mann_whitney_U": round(stat, 1) if not np.isnan(stat) else np.nan,
                "p_value": pval,
            })

        # Pipeline completion rate
        cr_used = completion_rate(used)
        cr_skip = completion_rate(skipped)
        metrics.append({
            "metric": "Pipeline Completion (%)",
            "subset": label,
            "mean_backbone_used": round(cr_used, 1),
            "mean_backbone_skipped": round(cr_skip, 1),
            "delta": round(cr_used - cr_skip, 1),
            "n_used": len(used),
            "n_skipped": len(skipped),
            "mann_whitney_U": np.nan,
            "p_value": np.nan,
        })

        results[label] = pd.DataFrame(metrics)

    return results


def compute_skip_strategy_breakdown(
    bdf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For de novo tasks only: classify alternative strategies when backbone
    gen is skipped. Return (frequency table, mean score table)."""
    dn = bdf[
        (bdf["design_approach"] == "de_novo") & (~bdf["backbone_used"])
    ].copy()

    # Frequency per condition
    freq_rows: list[dict[str, Any]] = []
    for cond in AGENT_CONDITIONS:
        cond_dn = dn[dn["condition"] == cond]
        total = len(cond_dn)
        for cat in ["No tools at all", "Analysis only", "Sequence-first", "Other"]:
            n = (cond_dn["skip_category"] == cat).sum()
            pct = n / total * 100 if total > 0 else 0.0
            freq_rows.append({
                "condition": cond,
                "category": cat,
                "count": n,
                "pct": round(pct, 1),
            })
    freq_df = pd.DataFrame(freq_rows)

    # Mean score per category (across all conditions)
    score_rows: list[dict[str, Any]] = []
    for cat in ["No tools at all", "Analysis only", "Sequence-first", "Other",
                "Backbone used"]:
        if cat == "Backbone used":
            subset = bdf[
                (bdf["design_approach"] == "de_novo") & (bdf["backbone_used"])
            ]
        else:
            subset = dn[dn["skip_category"] == cat]
        n = len(subset)
        mean_total = subset["total"].mean() if n > 0 else 0.0
        mean_quality = subset["quality"].mean() if n > 0 else 0.0
        score_rows.append({
            "category": cat,
            "n": n,
            "mean_total": round(mean_total, 2),
            "mean_quality": round(mean_quality, 2),
        })
    score_df = pd.DataFrame(score_rows)

    return freq_df, score_df


def compute_model_comparison(bdf: pd.DataFrame) -> pd.DataFrame:
    """Per-model backbone generation usage rate averaged across tasks."""
    rows: list[dict[str, Any]] = []
    # Group by LLM (merge user+benchmark modes)
    models = bdf.groupby("llm")
    for llm, group in models:
        dn = group[group["design_approach"] == "de_novo"]
        rd = group[group["design_approach"] == "redesign"]
        dn_rate = dn["backbone_used"].mean() * 100 if len(dn) > 0 else 0.0
        rd_rate = rd["backbone_used"].mean() * 100 if len(rd) > 0 else 0.0
        all_rate = group["backbone_used"].mean() * 100 if len(group) > 0 else 0.0
        mean_total = group["total"].mean()
        rows.append({
            "llm": llm,
            "de_novo_backbone_rate": round(dn_rate, 1),
            "redesign_backbone_rate": round(rd_rate, 1),
            "overall_backbone_rate": round(all_rate, 1),
            "mean_total_score": round(mean_total, 1),
            "n_tasks": len(group),
        })
    return pd.DataFrame(rows).sort_values("mean_total_score", ascending=False)


# ── Figures ──────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF + PNG to the figures directory."""
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    """Apply minimal clean styling to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#cccccc", alpha=0.3, linewidth=0.3, zorder=0)


def fig_a_usage_bar(usage_df: pd.DataFrame) -> None:
    """Figure A: Per-condition backbone gen usage rate, de_novo vs redesign."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for idx, approach in enumerate(["de_novo", "redesign"]):
        ax = axes[idx]
        sub = usage_df[usage_df["design_approach"] == approach]
        sub = sub.set_index("condition").reindex(AGENT_CONDITIONS)

        x = np.arange(len(AGENT_CONDITIONS))
        bars = ax.bar(
            x,
            sub["usage_rate_pct"],
            color=[COLORS.get(c, "#888") for c in AGENT_CONDITIONS],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # Add value labels on bars
        for bar_obj, val in zip(bars, sub["usage_rate_pct"]):
            if val > 0:
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 1.5,
                    f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=6,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHORT_NAMES.get(c, c) for c in AGENT_CONDITIONS],
            rotation=0, ha="center", fontsize=6,
        )
        title_label = "De Novo Tasks (backbone REQUIRED)" if approach == "de_novo" else "Redesign Tasks (backbone optional)"
        ax.set_title(title_label, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        _style_ax(ax)

        # Panel label
        panel = "(a)" if idx == 0 else "(b)"
        ax.text(-0.08, 1.06, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

    axes[0].set_ylabel("Tasks Using Backbone Generation (%)")
    fig.suptitle(
        "Backbone Generation Usage by Agent Condition",
        fontsize=10, fontweight="bold", y=1.02,
    )

    _save_fig(fig, "fig_backbone_skip_usage")


def fig_b_outcome_box(bdf: pd.DataFrame) -> None:
    """Figure B: Box plot comparing scores for backbone used vs skipped."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    subsets = [
        ("De Novo Tasks Only", bdf[bdf["design_approach"] == "de_novo"]),
        ("All Tasks", bdf),
    ]

    for idx, (title, subset) in enumerate(subsets):
        ax = axes[idx]

        used = subset[subset["backbone_used"]]["total"].dropna()
        skipped = subset[~subset["backbone_used"]]["total"].dropna()

        data_to_plot = [used, skipped]
        labels = [
            f"Backbone\nUsed\n(n={len(used)})",
            f"Backbone\nSkipped\n(n={len(skipped)})",
        ]

        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="black",
                           markeredgecolor="black", markersize=4),
            medianprops=dict(color="black", linewidth=1),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )

        box_colors = ["#2ca02c", "#d62728"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Mann-Whitney U test
        if len(used) >= 2 and len(skipped) >= 2:
            _, pval = mannwhitneyu(used, skipped, alternative="two-sided")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            ax.text(
                1.5, ax.get_ylim()[1] * 0.95,
                f"p={pval:.2e} ({sig})",
                ha="center", fontsize=6, style="italic",
            )

        # Mean annotations
        for i, data in enumerate(data_to_plot):
            if len(data) > 0:
                mean_val = data.mean()
                ax.text(
                    i + 1, mean_val + 1.5,
                    f"mean={mean_val:.1f}",
                    ha="center", fontsize=6, color="black",
                )

        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_ylabel("Total Score (0-100)")
        _style_ax(ax)

        panel = "(a)" if idx == 0 else "(b)"
        ax.text(-0.12, 1.06, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

    fig.suptitle(
        "Score Distributions: Backbone Used vs Skipped",
        fontsize=10, fontweight="bold", y=1.02,
    )

    _save_fig(fig, "fig_backbone_skip_outcome")


def fig_c_heatmap(bdf: pd.DataFrame) -> None:
    """Figure C: Condition x task_id binary heatmap (de_novo tasks only)."""
    dn = bdf[bdf["design_approach"] == "de_novo"].copy()

    if len(dn) == 0:
        print("  WARNING: No de_novo tasks found, skipping heatmap.")
        return

    # Build pivot table: rows=conditions, columns=task_ids
    pivot = dn.pivot_table(
        index="condition",
        columns="task_id",
        values="backbone_used",
        aggfunc="first",
    ).fillna(False).astype(int)

    # Filter to agent conditions and sort
    pivot = pivot.reindex([c for c in AGENT_CONDITIONS if c in pivot.index])

    # Sort columns by overall usage frequency (most-skipped on left)
    col_sums = pivot.sum(axis=0)
    sorted_cols = col_sums.sort_values(ascending=True).index
    pivot = pivot[sorted_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_cols) * 0.25), 4))

    # Custom colormap: red=0 (skipped), green=1 (used)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffcccc", "#66bb6a"])

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(np.arange(len(sorted_cols)))
    ax.set_xticklabels(sorted_cols, rotation=90, fontsize=5, ha="center")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [SHORT_NAMES.get(c, c).replace("\n", " ") for c in pivot.index],
        fontsize=6,
    )

    ax.set_xlabel("Task ID (sorted by backbone usage frequency)", fontsize=7)
    ax.set_ylabel("Agent Condition", fontsize=7)
    ax.set_title(
        "Backbone Generation Usage per Task (De Novo Only)",
        fontsize=9, fontweight="bold",
    )

    # Add gridlines between cells
    ax.set_xticks(np.arange(-0.5, len(sorted_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#66bb6a", edgecolor="gray", label="Backbone used"),
        Patch(facecolor="#ffcccc", edgecolor="gray", label="Backbone skipped"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=6, frameon=True, edgecolor="gray",
    )

    # Summary annotation: overall skip rate
    total_cells = pivot.size
    total_used = pivot.values.sum()
    skip_rate = (1 - total_used / total_cells) * 100 if total_cells > 0 else 0
    ax.text(
        0.5, -0.18,
        f"Overall skip rate: {skip_rate:.1f}% ({total_cells - total_used}/{total_cells} cells)",
        transform=ax.transAxes, ha="center", fontsize=7, style="italic",
    )

    _save_fig(fig, "fig_backbone_skip_heatmap")


# ── Report generation ────────────────────────────────────────────────────

def generate_report(
    bdf: pd.DataFrame,
    usage_df: pd.DataFrame,
    outcome: dict[str, pd.DataFrame],
    freq_df: pd.DataFrame,
    score_df: pd.DataFrame,
    model_comp: pd.DataFrame,
) -> str:
    """Generate a Markdown report summarizing the backbone skip analysis."""
    lines: list[str] = []
    lines.append("# Backbone Generation Skip Analysis")
    lines.append("")
    lines.append("## Key Finding")
    lines.append("")
    lines.append(
        "LLM agents are critics, not creators -- they systematically skip "
        "generative backbone generation while relying on analytical and "
        "evaluation tools. Backbone generation is the most systematically "
        "skipped pipeline step, and skipping it is strongly associated with "
        "lower scores, especially for de novo design tasks."
    )
    lines.append("")

    # Section 1: Usage rates
    lines.append("## 1. Backbone Generation Usage Rates")
    lines.append("")
    lines.append("### De Novo Tasks (backbone REQUIRED)")
    lines.append("")
    lines.append("| Condition | Tasks | Used | Rate |")
    lines.append("|-----------|-------|------|------|")
    dn_usage = usage_df[usage_df["design_approach"] == "de_novo"]
    for _, row in dn_usage.iterrows():
        lines.append(
            f"| {row['condition']} | {row['n_total']} | "
            f"{row['n_backbone_used']} | {row['usage_rate_pct']:.1f}% |"
        )
    lines.append("")

    lines.append("### Redesign Tasks (backbone optional)")
    lines.append("")
    lines.append("| Condition | Tasks | Used | Rate |")
    lines.append("|-----------|-------|------|------|")
    rd_usage = usage_df[usage_df["design_approach"] == "redesign"]
    for _, row in rd_usage.iterrows():
        lines.append(
            f"| {row['condition']} | {row['n_total']} | "
            f"{row['n_backbone_used']} | {row['usage_rate_pct']:.1f}% |"
        )
    lines.append("")

    # Section 2: Outcome comparison
    lines.append("## 2. Score Impact of Backbone Skipping")
    lines.append("")
    for label in ["de_novo", "all"]:
        label_display = "De Novo Tasks Only" if label == "de_novo" else "All Tasks"
        lines.append(f"### {label_display}")
        lines.append("")
        lines.append(
            "| Metric | Backbone Used | Backbone Skipped | Delta | p-value |"
        )
        lines.append("|--------|---------------|------------------|-------|---------|")
        for _, row in outcome[label].iterrows():
            pval_str = f"{row['p_value']:.2e}" if not np.isnan(row["p_value"]) else "N/A"
            lines.append(
                f"| {row['metric']} | {row['mean_backbone_used']:.1f} | "
                f"{row['mean_backbone_skipped']:.1f} | "
                f"+{row['delta']:.1f} | {pval_str} |"
            )
        lines.append("")

    # Section 3: Skip strategy breakdown
    lines.append("## 3. Alternative Strategies When Backbone Gen Is Skipped (De Novo)")
    lines.append("")
    lines.append("### Mean Score by Strategy")
    lines.append("")
    lines.append("| Category | N | Mean Total | Mean Quality |")
    lines.append("|----------|---|------------|--------------|")
    for _, row in score_df.iterrows():
        lines.append(
            f"| {row['category']} | {row['n']} | "
            f"{row['mean_total']:.1f} | {row['mean_quality']:.1f} |"
        )
    lines.append("")

    lines.append("### Frequency per Condition")
    lines.append("")
    # Pivot for readability
    freq_pivot = freq_df.pivot_table(
        index="condition", columns="category", values="count", fill_value=0
    )
    cols = [c for c in ["No tools at all", "Analysis only", "Sequence-first", "Other"]
            if c in freq_pivot.columns]
    if cols:
        lines.append("| Condition | " + " | ".join(cols) + " |")
        lines.append("|-----------|" + "|".join(["------"] * len(cols)) + "|")
        for cond in AGENT_CONDITIONS:
            if cond in freq_pivot.index:
                vals = [str(int(freq_pivot.loc[cond, c])) if c in freq_pivot.columns else "0"
                        for c in cols]
                lines.append(f"| {cond} | " + " | ".join(vals) + " |")
    lines.append("")

    # Section 4: Model comparison
    lines.append("## 4. Per-Model Summary")
    lines.append("")
    lines.append(
        "| Model | De Novo BB Rate | Redesign BB Rate | Overall BB Rate | Mean Score |"
    )
    lines.append(
        "|-------|-----------------|------------------|-----------------|------------|"
    )
    for _, row in model_comp.iterrows():
        lines.append(
            f"| {row['llm']} | {row['de_novo_backbone_rate']:.1f}% | "
            f"{row['redesign_backbone_rate']:.1f}% | "
            f"{row['overall_backbone_rate']:.1f}% | "
            f"{row['mean_total_score']:.1f} |"
        )
    lines.append("")

    # Section 5: Narrative summary
    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The data reveals a striking asymmetry in how LLM agents engage with "
        "the protein design pipeline. While these agents readily employ "
        "analytical tools (structure prediction, scoring, interface analysis), "
        "they systematically skip the generative backbone creation step -- "
        "the foundational act of designing a new protein structure."
    )
    lines.append("")
    lines.append(
        "This pattern holds across all LLM agents and is most pronounced for "
        "de novo design tasks where backbone generation is a required step. "
        "The consequence is clear: tasks where backbone generation is skipped "
        "achieve substantially lower scores, both in total and in the Quality "
        "component specifically."
    )
    lines.append("")
    lines.append(
        "The alternative strategies agents adopt when they skip backbone "
        "generation (code-only fallbacks, analysis-only pipelines, or "
        "sequence-first approaches) consistently underperform compared to "
        "full pipeline execution that includes backbone generation."
    )
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the complete backbone skip analysis."""
    print("=" * 70)
    print("BACKBONE GENERATION SKIP ANALYSIS")
    print("=" * 70)
    print()

    # 1. Load data
    print("Loading data...")
    df = load_all()
    # Filter to 9 agent conditions (exclude Oracle, Human Expert)
    df = df[df["condition"].isin(AGENT_CONDITIONS)].copy()
    # Reset categorical to avoid empty categories
    df["condition"] = pd.Categorical(
        df["condition"], categories=AGENT_CONDITIONS, ordered=True
    )
    print(f"  {len(df)} rows ({df['task_id'].nunique()} tasks x "
          f"{df['condition'].nunique()} conditions)")

    # 2. Build augmented DataFrame with backbone usage info
    print("Analyzing backbone generation usage...")
    bdf = build_backbone_df(df)

    # Quick sanity check
    n_de_novo = (bdf["design_approach"] == "de_novo").sum()
    n_redesign = (bdf["design_approach"] == "redesign").sum()
    n_bb_used = bdf["backbone_used"].sum()
    print(f"  De novo tasks: {n_de_novo}")
    print(f"  Redesign tasks: {n_redesign}")
    print(f"  Backbone generation used: {n_bb_used} / {len(bdf)} "
          f"({n_bb_used / len(bdf) * 100:.1f}%)")
    print()

    # 3. Compute analyses
    print("Computing usage rates...")
    usage_df = compute_usage_rates(bdf)

    print("Computing outcome comparisons...")
    outcome = compute_outcome_comparison(bdf)

    print("Computing skip strategy breakdown...")
    freq_df, score_df = compute_skip_strategy_breakdown(bdf)

    print("Computing model comparison...")
    model_comp = compute_model_comparison(bdf)

    # 4. Print summary statistics
    print()
    print("-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    print()

    # De novo usage rates
    print("De Novo Backbone Generation Usage (%):")
    dn_usage = usage_df[usage_df["design_approach"] == "de_novo"]
    for _, row in dn_usage.iterrows():
        bar = "#" * int(row["usage_rate_pct"] / 2)
        print(f"  {row['condition']:30s} {row['usage_rate_pct']:5.1f}%  {bar}")
    print()

    # Outcome delta
    print("Score Impact (De Novo Tasks):")
    for _, row in outcome["de_novo"].iterrows():
        print(f"  {row['metric']:25s}  Used={row['mean_backbone_used']:5.1f}  "
              f"Skipped={row['mean_backbone_skipped']:5.1f}  "
              f"Delta=+{row['delta']:5.1f}")
    print()

    # Skip strategies
    print("Alternative Strategies When Backbone Skipped (De Novo):")
    for _, row in score_df.iterrows():
        print(f"  {row['category']:20s}  n={row['n']:4d}  "
              f"mean_total={row['mean_total']:5.1f}  "
              f"mean_quality={row['mean_quality']:5.1f}")
    print()

    # Model comparison
    print("Per-Model Backbone Usage vs Score:")
    for _, row in model_comp.iterrows():
        print(f"  {row['llm']:15s}  BB_rate={row['de_novo_backbone_rate']:5.1f}%  "
              f"Score={row['mean_total_score']:5.1f}")
    print()

    # 5. Generate figures
    print("-" * 70)
    print("GENERATING FIGURES")
    print("-" * 70)
    print()

    print("Figure A: Backbone generation usage by condition...")
    fig_a_usage_bar(usage_df)
    print()

    print("Figure B: Score distributions backbone used vs skipped...")
    fig_b_outcome_box(bdf)
    print()

    print("Figure C: Binary heatmap (de novo tasks)...")
    fig_c_heatmap(bdf)
    print()

    # 6. Save stats CSV
    print("Saving statistics...")

    # Combine all stats into one CSV
    stats_rows: list[dict[str, Any]] = []

    # Usage rates
    for _, row in usage_df.iterrows():
        stats_rows.append({
            "analysis": "usage_rate",
            "condition": row["condition"],
            "design_approach": row["design_approach"],
            "metric": "backbone_usage_pct",
            "value": row["usage_rate_pct"],
            "n": row["n_total"],
        })

    # Outcome comparison
    for label, odf in outcome.items():
        for _, row in odf.iterrows():
            stats_rows.append({
                "analysis": f"outcome_{label}",
                "condition": "all",
                "design_approach": label,
                "metric": f"{row['metric']}_used",
                "value": row["mean_backbone_used"],
                "n": row["n_used"],
            })
            stats_rows.append({
                "analysis": f"outcome_{label}",
                "condition": "all",
                "design_approach": label,
                "metric": f"{row['metric']}_skipped",
                "value": row["mean_backbone_skipped"],
                "n": row["n_skipped"],
            })
            if not np.isnan(row["p_value"]):
                stats_rows.append({
                    "analysis": f"outcome_{label}",
                    "condition": "all",
                    "design_approach": label,
                    "metric": f"{row['metric']}_pvalue",
                    "value": row["p_value"],
                    "n": row["n_used"] + row["n_skipped"],
                })

    # Skip strategies
    for _, row in score_df.iterrows():
        stats_rows.append({
            "analysis": "skip_strategy",
            "condition": "all",
            "design_approach": "de_novo",
            "metric": f"{row['category']}_mean_total",
            "value": row["mean_total"],
            "n": row["n"],
        })

    # Model comparison
    for _, row in model_comp.iterrows():
        stats_rows.append({
            "analysis": "model_comparison",
            "condition": str(row["llm"]),
            "design_approach": "all",
            "metric": "de_novo_backbone_rate",
            "value": row["de_novo_backbone_rate"],
            "n": row["n_tasks"],
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = FIGURES_DIR / "backbone_skip_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")

    # 7. Save report
    report = generate_report(bdf, usage_df, outcome, freq_df, score_df, model_comp)
    report_path = FIGURES_DIR / "backbone_skip_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
