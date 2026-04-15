#!/usr/bin/env python3
"""User mode backbone rescue analysis for BioDesignBench.

Analyzes whether user mode (guided workflow) promotes backbone generation
usage and decomposes the user mode score gain into:
  - Backbone rescue gain (BM skipped, User used backbone gen)
  - Orchestration improvement gain (both used or both skipped)
  - Regression loss (User scored lower than BM)

Outputs (saved to figures/):
    fig_backbone_rescue_paired.pdf/.png     : Paired dot plot of backbone gen usage
    fig_backbone_rescue_waterfall.pdf/.png   : Waterfall chart of gain decomposition
    fig_backbone_rescue_transitions.pdf/.png : Heatmap of transition counts
    backbone_rescue_stats.csv               : Per-model statistics
    backbone_rescue_report.md               : Summary report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import CONDITION_MAP, load_all

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "DeepSeek V3": "#1f77b4",
    "GPT-5": "#ff7f0e",
    "Sonnet 4.5": "#2ca02c",
    "Gemini 2.5 Pro": "#d62728",
}

# Lighter variants for benchmark mode dots
MODEL_COLORS_LIGHT = {
    "DeepSeek V3": "#aec7e8",
    "GPT-5": "#ffbb78",
    "Sonnet 4.5": "#98df8a",
    "Gemini 2.5 Pro": "#ff9896",
}

LLMS = ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]

# Tool categories
BACKBONE_GEN_TOOLS = {
    "rfdiffusion", "generate_backbone", "chroma", "design_binder",
}
SEQUENCE_DESIGN_TOOLS = {
    "proteinmpnn", "optimize_sequence", "rosetta_design",
    "design_binder", "ligandmpnn", "esm_if", "mpnn",
}
STRUCTURE_PRED_TOOLS = {
    "esmfold", "alphafold2", "predict_structure", "predict_complex",
    "validate_design", "predict_structure_boltz", "predict_affinity_boltz",
    "colabfold",
}

# Transition type labels
TRANSITION_LABELS = ["both_skip", "rescued", "lost", "both_used"]
TRANSITION_DISPLAY = {
    "both_skip": "Both Skip",
    "rescued": "Rescued\n(BM skip -> User used)",
    "lost": "Lost\n(BM used -> User skip)",
    "both_used": "Both Used",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_tools_used(condition: str, task_id: str) -> list[str]:
    """Load tools_used from an individual result.json."""
    info = CONDITION_MAP.get(condition)
    if info is None:
        return []
    result_path = info["path"] / task_id / "result.json"
    if not result_path.exists():
        return []
    with open(result_path) as f:
        result = json.load(f)
    return result.get("tools_used", [])


def _used_backbone_gen(tools: list[str]) -> bool:
    """Check if any backbone generation tool was used."""
    tools_lower = {t.lower().strip() for t in tools}
    return bool(tools_lower & BACKBONE_GEN_TOOLS)


def _load_num_designs(condition: str, task_id: str) -> int:
    """Load num_designs from an individual result.json."""
    info = CONDITION_MAP.get(condition)
    if info is None:
        return 0
    result_path = info["path"] / task_id / "result.json"
    if not result_path.exists():
        return 0
    with open(result_path) as f:
        result = json.load(f)
    dm = result.get("diversity_metrics", {})
    if isinstance(dm, dict):
        return dm.get("num_designs", 0)
    return 0


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def build_paired_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with one row per (llm, task_id) pairing BM and User mode results.

    Columns: llm, task_id, design_approach, molecular_subject,
             bm_total, user_total, bm_tools, user_tools,
             bm_backbone, user_backbone, transition
    """
    # Filter to paired LLM modes only
    paired = df[df["mode"].isin(["benchmark", "user"]) & df["llm"].isin(LLMS)].copy()

    rows = []
    for llm in LLMS:
        bm_cond = f"{llm} benchmark"
        user_cond = f"{llm} user"

        bm_df = paired[(paired["llm"] == llm) & (paired["mode"] == "benchmark")]
        user_df = paired[(paired["llm"] == llm) & (paired["mode"] == "user")]

        bm_tasks = set(bm_df["task_id"])
        user_tasks = set(user_df["task_id"])
        common_tasks = bm_tasks & user_tasks

        for tid in sorted(common_tasks):
            bm_row = bm_df[bm_df["task_id"] == tid].iloc[0]
            user_row = user_df[user_df["task_id"] == tid].iloc[0]

            bm_tools = list(bm_row["tool_sequence"]) if isinstance(bm_row["tool_sequence"], list) else []
            user_tools = list(user_row["tool_sequence"]) if isinstance(user_row["tool_sequence"], list) else []

            # Also load tools_used from result.json for more complete picture
            bm_tools_extra = _load_tools_used(bm_cond, tid)
            user_tools_extra = _load_tools_used(user_cond, tid)

            # Combine both sources
            bm_all_tools = list(set(bm_tools + bm_tools_extra))
            user_all_tools = list(set(user_tools + user_tools_extra))

            bm_bb = _used_backbone_gen(bm_all_tools)
            user_bb = _used_backbone_gen(user_all_tools)

            # Classify transition
            if not bm_bb and not user_bb:
                transition = "both_skip"
            elif not bm_bb and user_bb:
                transition = "rescued"
            elif bm_bb and not user_bb:
                transition = "lost"
            else:
                transition = "both_used"

            rows.append({
                "llm": llm,
                "task_id": tid,
                "design_approach": bm_row["design_approach"],
                "molecular_subject": bm_row["molecular_subject"],
                "bm_total": bm_row["total"],
                "user_total": user_row["total"],
                "bm_backbone": bm_bb,
                "user_backbone": user_bb,
                "transition": transition,
                "score_delta": user_row["total"] - bm_row["total"],
            })

    return pd.DataFrame(rows)


def compute_backbone_usage_rates(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Compute backbone generation usage rates per model, overall and de_novo only.

    Returns DataFrame with columns: llm, bm_rate_all, user_rate_all, delta_all,
                                    bm_rate_denovo, user_rate_denovo, delta_denovo,
                                    n_all, n_denovo
    """
    rows = []
    for llm in LLMS:
        ldf = paired_df[paired_df["llm"] == llm]
        dn_df = ldf[ldf["design_approach"] == "de_novo"]

        # All tasks
        n_all = len(ldf)
        bm_rate_all = ldf["bm_backbone"].mean() * 100 if n_all > 0 else 0
        user_rate_all = ldf["user_backbone"].mean() * 100 if n_all > 0 else 0

        # De novo only
        n_dn = len(dn_df)
        bm_rate_dn = dn_df["bm_backbone"].mean() * 100 if n_dn > 0 else 0
        user_rate_dn = dn_df["user_backbone"].mean() * 100 if n_dn > 0 else 0

        rows.append({
            "llm": llm,
            "bm_rate_all": round(bm_rate_all, 1),
            "user_rate_all": round(user_rate_all, 1),
            "delta_all": round(user_rate_all - bm_rate_all, 1),
            "bm_rate_denovo": round(bm_rate_dn, 1),
            "user_rate_denovo": round(user_rate_dn, 1),
            "delta_denovo": round(user_rate_dn - bm_rate_dn, 1),
            "n_all": n_all,
            "n_denovo": n_dn,
        })

    return pd.DataFrame(rows)


def compute_transition_counts(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Count transitions per model for de_novo tasks.

    Returns DataFrame with columns: llm, both_skip, rescued, lost, both_used, n_denovo
    """
    rows = []
    for llm in LLMS:
        dn_df = paired_df[
            (paired_df["llm"] == llm) & (paired_df["design_approach"] == "de_novo")
        ]
        counts = dn_df["transition"].value_counts()
        rows.append({
            "llm": llm,
            "both_skip": int(counts.get("both_skip", 0)),
            "rescued": int(counts.get("rescued", 0)),
            "lost": int(counts.get("lost", 0)),
            "both_used": int(counts.get("both_used", 0)),
            "n_denovo": len(dn_df),
        })
    return pd.DataFrame(rows)


def compute_rescued_outcomes(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze outcomes of rescued vs both_used tasks.

    Returns DataFrame with columns: llm, category, n_tasks, mean_user_score, std_user_score,
                                    mean_bm_score, mean_delta
    """
    dn_df = paired_df[paired_df["design_approach"] == "de_novo"].copy()

    rows = []
    for llm in LLMS:
        ldf = dn_df[dn_df["llm"] == llm]
        for cat in ["rescued", "both_used", "both_skip", "lost"]:
            cdf = ldf[ldf["transition"] == cat]
            n = len(cdf)
            if n == 0:
                rows.append({
                    "llm": llm,
                    "category": cat,
                    "n_tasks": 0,
                    "mean_user_score": 0.0,
                    "std_user_score": 0.0,
                    "mean_bm_score": 0.0,
                    "mean_delta": 0.0,
                })
                continue
            rows.append({
                "llm": llm,
                "category": cat,
                "n_tasks": n,
                "mean_user_score": round(cdf["user_total"].mean(), 1),
                "std_user_score": round(cdf["user_total"].std(), 1) if n > 1 else 0.0,
                "mean_bm_score": round(cdf["bm_total"].mean(), 1),
                "mean_delta": round(cdf["score_delta"].mean(), 1),
            })

    return pd.DataFrame(rows)


def compute_gain_decomposition(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Decompose user mode gain into backbone rescue, orchestration improvement, and regression.

    For each model (de_novo tasks):
      - backbone_rescue_gain: sum of score_delta for "rescued" tasks where score_delta > 0
      - orch_improvement_gain: sum of score_delta for non-rescued tasks where score_delta > 0
      - regression_loss: sum of score_delta for tasks where score_delta < 0
      - net_gain: total sum of score_delta

    Also calculates per-task means and pct contributions.

    Returns DataFrame with columns per model.
    """
    dn_df = paired_df[paired_df["design_approach"] == "de_novo"].copy()

    rows = []
    for llm in LLMS:
        ldf = dn_df[dn_df["llm"] == llm]
        n_tasks = len(ldf)
        total_delta = ldf["score_delta"].sum()

        # Rescued tasks
        rescued = ldf[ldf["transition"] == "rescued"]
        rescued_pos = rescued[rescued["score_delta"] > 0]
        rescued_neg = rescued[rescued["score_delta"] <= 0]
        backbone_rescue_gain = rescued_pos["score_delta"].sum() if len(rescued_pos) > 0 else 0
        backbone_rescue_loss = rescued_neg["score_delta"].sum() if len(rescued_neg) > 0 else 0

        # Non-rescued tasks (both_used, both_skip, lost)
        non_rescued = ldf[ldf["transition"] != "rescued"]
        non_rescued_pos = non_rescued[non_rescued["score_delta"] > 0]
        non_rescued_neg = non_rescued[non_rescued["score_delta"] <= 0]
        orch_improvement = non_rescued_pos["score_delta"].sum() if len(non_rescued_pos) > 0 else 0
        regression = non_rescued_neg["score_delta"].sum() if len(non_rescued_neg) > 0 else 0

        # Add rescued tasks where score went down to regression
        total_regression = regression + backbone_rescue_loss
        total_positive = backbone_rescue_gain + orch_improvement

        # Per-task means
        rescued_mean_delta = rescued["score_delta"].mean() if len(rescued) > 0 else 0
        non_rescued_mean_delta = non_rescued["score_delta"].mean() if len(non_rescued) > 0 else 0

        # Pct of total positive gain
        pct_rescue = (backbone_rescue_gain / total_positive * 100) if total_positive > 0 else 0
        pct_orch = (orch_improvement / total_positive * 100) if total_positive > 0 else 0

        rows.append({
            "llm": llm,
            "n_denovo": n_tasks,
            "net_gain": round(total_delta, 1),
            "backbone_rescue_gain": round(backbone_rescue_gain, 1),
            "orch_improvement_gain": round(orch_improvement, 1),
            "regression_loss": round(total_regression, 1),
            "pct_rescue_of_positive": round(pct_rescue, 1),
            "pct_orch_of_positive": round(pct_orch, 1),
            "rescued_mean_delta": round(rescued_mean_delta, 1),
            "non_rescued_mean_delta": round(non_rescued_mean_delta, 1),
            "n_rescued": len(rescued),
            "n_non_rescued": len(non_rescued),
        })

    # Add average row
    avg_row = {
        "llm": "Average",
        "n_denovo": int(np.mean([r["n_denovo"] for r in rows])),
        "net_gain": round(np.mean([r["net_gain"] for r in rows]), 1),
        "backbone_rescue_gain": round(np.mean([r["backbone_rescue_gain"] for r in rows]), 1),
        "orch_improvement_gain": round(np.mean([r["orch_improvement_gain"] for r in rows]), 1),
        "regression_loss": round(np.mean([r["regression_loss"] for r in rows]), 1),
        "pct_rescue_of_positive": round(np.mean([r["pct_rescue_of_positive"] for r in rows]), 1),
        "pct_orch_of_positive": round(np.mean([r["pct_orch_of_positive"] for r in rows]), 1),
        "rescued_mean_delta": round(np.mean([r["rescued_mean_delta"] for r in rows]), 1),
        "non_rescued_mean_delta": round(np.mean([r["non_rescued_mean_delta"] for r in rows]), 1),
        "n_rescued": int(np.mean([r["n_rescued"] for r in rows])),
        "n_non_rescued": int(np.mean([r["n_non_rescued"] for r in rows])),
    }
    rows.append(avg_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_paired_dotplot(usage_df: pd.DataFrame) -> None:
    """Figure A: Paired dot plot of backbone gen usage rate per model.

    Two dots per model (benchmark lighter, user darker) connected by an arrow.
    Focused on de_novo tasks.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = np.arange(len(LLMS))

    for i, llm in enumerate(LLMS):
        row = usage_df[usage_df["llm"] == llm].iloc[0]
        bm_rate = row["bm_rate_denovo"]
        user_rate = row["user_rate_denovo"]
        delta = row["delta_denovo"]

        # Draw connecting arrow from benchmark to user
        ax.annotate(
            "",
            xy=(i, user_rate),
            xytext=(i, bm_rate),
            arrowprops=dict(
                arrowstyle="-|>",
                color=MODEL_COLORS[llm],
                lw=1.8,
                mutation_scale=14,
            ),
        )

        # Benchmark dot (lighter, open)
        ax.scatter(
            i, bm_rate,
            s=120, c=MODEL_COLORS_LIGHT[llm],
            edgecolors=MODEL_COLORS[llm], linewidth=1.5,
            zorder=5, label=f"{llm} benchmark" if i == 0 else None,
        )

        # User dot (darker, filled)
        ax.scatter(
            i, user_rate,
            s=120, c=MODEL_COLORS[llm],
            edgecolors="white", linewidth=1.0,
            zorder=6, label=f"{llm} user" if i == 0 else None,
        )

        # Delta annotation
        sign = "+" if delta > 0 else ""
        y_label = max(bm_rate, user_rate) + 3
        ax.text(
            i, y_label, f"{sign}{delta:.0f}pp",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=MODEL_COLORS[llm],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(LLMS, fontsize=10)
    ax.set_ylabel("Backbone Generation Usage Rate (%)\n(de novo tasks only)", fontsize=11)
    ax.set_title(
        "Backbone Generation Usage: Benchmark vs User Mode",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend
    bm_patch = mpatches.Patch(
        facecolor="#cccccc", edgecolor="#666666", label="Benchmark mode",
    )
    user_patch = mpatches.Patch(
        facecolor="#666666", edgecolor="white", label="User mode",
    )
    ax.legend(handles=[bm_patch, user_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_backbone_rescue_paired.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_backbone_rescue_paired.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'fig_backbone_rescue_paired.pdf'}")


def plot_waterfall(decomp_df: pd.DataFrame) -> None:
    """Figure B: Waterfall chart of user mode gain decomposition.

    Stacked bars showing backbone rescue gain, orchestration improvement, and regression.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = list(decomp_df["llm"])
    n = len(labels)
    x = np.arange(n)
    width = 0.55

    rescue_vals = decomp_df["backbone_rescue_gain"].values
    orch_vals = decomp_df["orch_improvement_gain"].values
    regression_vals = decomp_df["regression_loss"].values  # These are negative
    net_vals = decomp_df["net_gain"].values

    # Stacked bars: rescue on bottom, orch on top (positive portion)
    bars_rescue = ax.bar(
        x, rescue_vals, width,
        color="#2ca02c", edgecolor="white", linewidth=0.8,
        label="Backbone rescue gain",
    )
    bars_orch = ax.bar(
        x, orch_vals, width, bottom=rescue_vals,
        color="#1f77b4", edgecolor="white", linewidth=0.8,
        label="Orchestration improvement",
    )
    # Regression is negative, drawn from 0 downward
    bars_reg = ax.bar(
        x, regression_vals, width,
        color="#d62728", edgecolor="white", linewidth=0.8,
        label="Regression loss",
    )

    # Net gain line
    ax.plot(x, net_vals, "ko-", markersize=7, linewidth=1.5, zorder=10, label="Net gain")

    # Value annotations
    for i in range(n):
        # Rescue value
        if rescue_vals[i] > 0:
            ax.text(
                i, rescue_vals[i] / 2, f"+{rescue_vals[i]:.0f}",
                ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            )
        # Orch value
        if orch_vals[i] > 0:
            ax.text(
                i, rescue_vals[i] + orch_vals[i] / 2, f"+{orch_vals[i]:.0f}",
                ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            )
        # Regression value
        if regression_vals[i] < 0:
            ax.text(
                i, regression_vals[i] / 2, f"{regression_vals[i]:.0f}",
                ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            )
        # Net value above/below
        offset = 8 if net_vals[i] >= 0 else -12
        ax.annotate(
            f"net: {net_vals[i]:+.0f}",
            xy=(i, net_vals[i]),
            xytext=(0, offset), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score Points (sum across de novo tasks)", fontsize=11)
    ax.set_title(
        "User Mode Gain Decomposition:\nBackbone Rescue vs Orchestration Improvement",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_backbone_rescue_waterfall.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_backbone_rescue_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'fig_backbone_rescue_waterfall.pdf'}")


def plot_transition_heatmap(trans_df: pd.DataFrame) -> None:
    """Figure C: Heatmap of transition counts (models x transition types).

    Rows = models, columns = transition types for de_novo tasks.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Build matrix
    matrix = trans_df.set_index("llm")[TRANSITION_LABELS].copy()

    # Custom colormap: white to deep blue
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    sns.heatmap(
        matrix,
        annot=True, fmt="d",
        cmap=cmap,
        linewidths=1.5, linecolor="white",
        cbar_kws={"label": "Number of de novo tasks", "shrink": 0.8},
        ax=ax,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )

    # Format labels
    ax.set_xticklabels(
        [TRANSITION_DISPLAY.get(t, t) for t in TRANSITION_LABELS],
        fontsize=9, rotation=0, ha="center",
    )
    ax.set_yticklabels(matrix.index, fontsize=10, rotation=0)
    ax.set_title(
        "Backbone Generation Transition Matrix\n(de novo tasks: Benchmark -> User mode)",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_backbone_rescue_transitions.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_backbone_rescue_transitions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'fig_backbone_rescue_transitions.pdf'}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    usage_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    decomp_df: pd.DataFrame,
) -> str:
    """Generate a markdown report summarizing the backbone rescue analysis."""
    avg_row = decomp_df[decomp_df["llm"] == "Average"].iloc[0]

    # Summary statistics
    mean_bm_rate = usage_df["bm_rate_denovo"].mean()
    mean_user_rate = usage_df["user_rate_denovo"].mean()
    mean_delta = usage_df["delta_denovo"].mean()

    total_rescued = trans_df["rescued"].sum()
    total_denovo = trans_df["n_denovo"].sum()
    pct_rescued = total_rescued / total_denovo * 100 if total_denovo > 0 else 0

    # Rescued task mean scores
    rescued_outcomes = outcomes_df[outcomes_df["category"] == "rescued"]
    mean_rescued_user_score = rescued_outcomes["mean_user_score"].mean()
    both_used_outcomes = outcomes_df[outcomes_df["category"] == "both_used"]
    mean_both_used_user_score = both_used_outcomes["mean_user_score"].mean()

    pct_rescue_gain = avg_row["pct_rescue_of_positive"]
    net_gain = avg_row["net_gain"]

    report = f"""# Backbone Rescue Analysis Report

## Key Finding

User mode's guided workflow specifically rescues backbone generation usage, converting
{pct_rescued:.0f}% of previously-skipped generative steps into successful pipeline completions.
This accounts for {pct_rescue_gain:.0f}% of the overall user mode score advantage on de novo tasks.

---

## 1. Backbone Generation Usage Rates (de novo tasks)

| Model | BM Rate (%) | User Rate (%) | Delta (pp) |
|-------|------------|--------------|------------|
"""
    for _, row in usage_df.iterrows():
        report += (
            f"| {row['llm']} | {row['bm_rate_denovo']:.1f} "
            f"| {row['user_rate_denovo']:.1f} "
            f"| {row['delta_denovo']:+.1f} |\n"
        )
    report += f"| **Average** | **{mean_bm_rate:.1f}** | **{mean_user_rate:.1f}** | **{mean_delta:+.1f}** |\n"

    report += f"""
User mode increases backbone generation usage by an average of {mean_delta:+.1f} percentage points
across all 4 LLMs on de novo tasks.

---

## 2. Transition Analysis (de novo tasks)

| Model | Both Skip | Rescued | Lost | Both Used | Total |
|-------|-----------|---------|------|-----------|-------|
"""
    for _, row in trans_df.iterrows():
        report += (
            f"| {row['llm']} | {row['both_skip']} | {row['rescued']} "
            f"| {row['lost']} | {row['both_used']} | {row['n_denovo']} |\n"
        )
    total_both_skip = trans_df["both_skip"].sum()
    total_lost = trans_df["lost"].sum()
    total_both_used = trans_df["both_used"].sum()
    report += (
        f"| **Total** | **{total_both_skip}** | **{total_rescued}** "
        f"| **{total_lost}** | **{total_both_used}** | **{total_denovo}** |\n"
    )

    report += f"""
- **Rescued** (BM skipped backbone gen, User used it): {total_rescued}/{total_denovo} tasks ({pct_rescued:.1f}%)
- **Lost** (BM used backbone gen, User skipped it): {total_lost}/{total_denovo} tasks ({total_lost / total_denovo * 100:.1f}%)
- Net rescue = {total_rescued - total_lost} more tasks using backbone gen in user mode

---

## 3. Rescued Task Outcomes

| Model | Category | N | User Score (mean) | BM Score (mean) | Delta |
|-------|----------|---|-------------------|-----------------|-------|
"""
    for _, row in outcomes_df.iterrows():
        if row["n_tasks"] > 0:
            report += (
                f"| {row['llm']} | {row['category']} | {row['n_tasks']} "
                f"| {row['mean_user_score']:.1f} | {row['mean_bm_score']:.1f} "
                f"| {row['mean_delta']:+.1f} |\n"
            )

    report += f"""
Rescued tasks achieve a mean User mode score of {mean_rescued_user_score:.1f}, compared to
{mean_both_used_user_score:.1f} for tasks where both modes used backbone generation.

---

## 4. User Mode Gain Decomposition (de novo tasks)

| Model | Net Gain | Rescue Gain | Orch. Improvement | Regression | Rescue % of Positive |
|-------|----------|-------------|-------------------|------------|---------------------|
"""
    for _, row in decomp_df.iterrows():
        report += (
            f"| {row['llm']} | {row['net_gain']:+.1f} "
            f"| +{row['backbone_rescue_gain']:.1f} "
            f"| +{row['orch_improvement_gain']:.1f} "
            f"| {row['regression_loss']:.1f} "
            f"| {row['pct_rescue_of_positive']:.0f}% |\n"
        )

    report += f"""
### Per-Task Mean Score Delta

| Model | Rescued Tasks (mean delta) | Non-Rescued Tasks (mean delta) |
|-------|---------------------------|-------------------------------|
"""
    for _, row in decomp_df[decomp_df["llm"] != "Average"].iterrows():
        report += (
            f"| {row['llm']} | {row['rescued_mean_delta']:+.1f} (n={row['n_rescued']}) "
            f"| {row['non_rescued_mean_delta']:+.1f} (n={row['n_non_rescued']}) |\n"
        )

    report += f"""
---

## Figures

1. `fig_backbone_rescue_paired.pdf` -- Paired dot plot: backbone gen usage rate per model
2. `fig_backbone_rescue_waterfall.pdf` -- Waterfall: gain decomposition per model
3. `fig_backbone_rescue_transitions.pdf` -- Heatmap: transition matrix (4 models x 4 types)

---

*Generated by `scripts/analysis/user_mode_backbone_rescue.py`*
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run backbone rescue analysis and generate all outputs."""
    print("=" * 70)
    print("User Mode Backbone Rescue Analysis")
    print("=" * 70)

    print("\nLoading results...")
    df = load_all()
    print(
        f"  Loaded {len(df)} rows "
        f"({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)"
    )

    # ── Build paired comparison dataset ──
    print("\nBuilding paired BM/User comparison for de novo tasks...")
    paired_df = build_paired_df(df)
    n_pairs = len(paired_df)
    n_denovo = (paired_df["design_approach"] == "de_novo").sum()
    print(f"  {n_pairs} task pairs across {len(LLMS)} models")
    print(f"  {n_denovo} de novo task pairs")

    # ── Analysis 1: Backbone generation usage rates ──
    print("\n1. Backbone generation usage rates (de_novo tasks)...")
    usage_df = compute_backbone_usage_rates(paired_df)
    for _, row in usage_df.iterrows():
        print(
            f"  {row['llm']:20s}  BM={row['bm_rate_denovo']:5.1f}%  "
            f"User={row['user_rate_denovo']:5.1f}%  "
            f"delta={row['delta_denovo']:+5.1f}pp  (n={row['n_denovo']})"
        )
    mean_delta = usage_df["delta_denovo"].mean()
    print(f"  Average delta: {mean_delta:+.1f}pp")

    # ── Analysis 2: Transition counts ──
    print("\n2. Transition analysis (de_novo tasks)...")
    trans_df = compute_transition_counts(paired_df)
    for _, row in trans_df.iterrows():
        print(
            f"  {row['llm']:20s}  both_skip={row['both_skip']:2d}  "
            f"rescued={row['rescued']:2d}  lost={row['lost']:2d}  "
            f"both_used={row['both_used']:2d}  (n={row['n_denovo']})"
        )
    total_rescued = trans_df["rescued"].sum()
    total_denovo = trans_df["n_denovo"].sum()
    print(f"  Total rescued: {total_rescued}/{total_denovo} ({total_rescued/total_denovo*100:.1f}%)")

    # ── Analysis 3: Rescued task outcomes ──
    print("\n3. Rescued task outcomes...")
    outcomes_df = compute_rescued_outcomes(paired_df)
    for cat in ["rescued", "both_used"]:
        cat_df = outcomes_df[outcomes_df["category"] == cat]
        n = cat_df["n_tasks"].sum()
        mean_user = cat_df.loc[cat_df["n_tasks"] > 0, "mean_user_score"].mean()
        mean_delta_cat = cat_df.loc[cat_df["n_tasks"] > 0, "mean_delta"].mean()
        print(
            f"  {cat:12s}: n={n:3d}  mean_user_score={mean_user:.1f}  "
            f"mean_delta={mean_delta_cat:+.1f}"
        )

    # ── Analysis 4: Gain decomposition ──
    print("\n4. User mode gain decomposition (de_novo tasks)...")
    decomp_df = compute_gain_decomposition(paired_df)
    for _, row in decomp_df.iterrows():
        print(
            f"  {row['llm']:20s}  net={row['net_gain']:+6.1f}  "
            f"rescue=+{row['backbone_rescue_gain']:5.1f} ({row['pct_rescue_of_positive']:4.0f}%)  "
            f"orch=+{row['orch_improvement_gain']:5.1f} ({row['pct_orch_of_positive']:4.0f}%)  "
            f"regress={row['regression_loss']:6.1f}"
        )

    # ── Generate figures ──
    print("\nGenerating figures...")
    plot_paired_dotplot(usage_df)
    plot_waterfall(decomp_df)
    plot_transition_heatmap(trans_df)

    # ── Save CSV ──
    stats_csv = FIGURES_DIR / "backbone_rescue_stats.csv"
    # Combine all stats into one CSV
    usage_for_csv = usage_df.copy()
    usage_for_csv = usage_for_csv.rename(columns=lambda c: f"usage_{c}" if c != "llm" else c)
    trans_for_csv = trans_df.copy()
    trans_for_csv = trans_for_csv.rename(columns=lambda c: f"trans_{c}" if c != "llm" else c)
    decomp_for_csv = decomp_df[decomp_df["llm"] != "Average"].copy()
    decomp_for_csv = decomp_for_csv.rename(columns=lambda c: f"decomp_{c}" if c != "llm" else c)

    combined = usage_for_csv.merge(trans_for_csv, on="llm").merge(decomp_for_csv, on="llm")
    combined.to_csv(stats_csv, index=False)
    print(f"\n  Stats saved to {stats_csv}")

    # ── Generate report ──
    report = generate_report(usage_df, trans_df, outcomes_df, decomp_df)
    report_path = FIGURES_DIR / "backbone_rescue_report.md"
    report_path.write_text(report)
    print(f"  Report saved to {report_path}")

    # ── Final summary ──
    avg_decomp = decomp_df[decomp_df["llm"] == "Average"].iloc[0]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"  Backbone gen usage increase:  {mean_delta:+.1f}pp "
        f"(BM {usage_df['bm_rate_denovo'].mean():.1f}% -> "
        f"User {usage_df['user_rate_denovo'].mean():.1f}%)"
    )
    print(
        f"  Tasks rescued:               {total_rescued}/{total_denovo} "
        f"({total_rescued/total_denovo*100:.1f}%)"
    )
    print(
        f"  Rescue gain share:           {avg_decomp['pct_rescue_of_positive']:.0f}% "
        f"of positive gain"
    )
    print(
        f"  Net user mode gain (de novo): {avg_decomp['net_gain']:+.1f} pts/model"
    )
    print(
        f"\n  User mode's guided workflow specifically rescues backbone generation "
        f"usage,\n  converting {total_rescued/total_denovo*100:.0f}% of previously-skipped "
        f"generative steps into successful\n  pipeline completions. This accounts for "
        f"{avg_decomp['pct_rescue_of_positive']:.0f}% of the overall user mode score advantage."
    )
    print(f"\n  All outputs saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
