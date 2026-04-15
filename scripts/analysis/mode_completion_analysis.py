#!/usr/bin/env python3
"""Mode completion analysis: user vs benchmark pipeline completion rates.

Analyzes how user-mode guidance (rich system prompt, 17 tools) vs benchmark-mode
(minimal prompt, 14 tools) affects pipeline completion rates across 4 LLMs.

Produces:
    - figures/fig_mode_completion.pdf/.png    (grouped bar chart)
    - figures/fig_mode_transitions.pdf/.png   (transition heatmap)
    - figures/mode_completion_stats.csv        (per-condition rates)
    - figures/mode_completion_report.md        (summary report)

Usage:
    python -m scripts.analysis.mode_completion_analysis
    python scripts/analysis/mode_completion_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all, CONDITION_MAP, EXCLUDED_TASKS

# ── Output directory ─────────────────────────────────────────────────────────
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
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

MODEL_COLORS: dict[str, str] = {
    "DeepSeek V3": "#1f77b4",
    "GPT-5": "#ff7f0e",
    "Sonnet 4.5": "#2ca02c",
    "Gemini 2.5 Pro": "#d62728",
}

PAIRED_LLMS: list[str] = ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]

# ── Tool categories for strict completion ────────────────────────────────────
BACKBONE_TOOLS: set[str] = {
    "rfdiffusion", "generate_backbone", "chroma", "design_binder",
}

SEQ_DESIGN_TOOLS: set[str] = {
    "proteinmpnn", "optimize_sequence", "rosetta_design",
    "design_binder", "ligandmpnn", "esm_if", "mpnn",
}

STRUCT_PRED_TOOLS: set[str] = {
    "esmfold", "alphafold2", "predict_structure", "predict_complex",
    "validate_design", "predict_structure_boltz", "predict_affinity_boltz",
    "colabfold",
}

SCORING_TOOLS: set[str] = {
    "score_stability", "analyze_interface", "rosetta_score",
    "rosetta_interface_score", "energy_minimize", "rosetta_relax",
}


def _setup_style() -> None:
    """Apply publication-quality rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load_raw_result(condition: str, task_id: str) -> dict[str, Any] | None:
    """Load a single raw result.json for a (condition, task_id) pair.

    Args:
        condition: Condition name from CONDITION_MAP.
        task_id: Task identifier.

    Returns:
        Parsed JSON dict or None if file does not exist.
    """
    info = CONDITION_MAP.get(condition)
    if info is None:
        return None
    result_path = info["path"] / task_id / "result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def _get_tools_used(result: dict[str, Any]) -> list[str]:
    """Extract the tools_used list from a result dict.

    Falls back to orchestration_metrics.actual_tool_order if tools_used is empty.

    Args:
        result: Parsed result.json dict.

    Returns:
        List of tool name strings.
    """
    tools = result.get("tools_used", [])
    if not tools:
        tools = result.get("orchestration_metrics", {}).get("actual_tool_order", [])
    return tools


def _has_stage(tools: list[str], stage_tools: set[str]) -> bool:
    """Check if any tool from a pipeline stage is present in the used tools.

    Args:
        tools: List of tool names used.
        stage_tools: Set of tool names belonging to a pipeline stage.

    Returns:
        True if at least one tool from the stage was used.
    """
    tools_lower = {t.lower() for t in tools}
    return bool(tools_lower & {s.lower() for s in stage_tools})


def _is_strict_complete(tools: list[str], is_redesign: bool) -> bool:
    """Check strict pipeline completion: all required stages present.

    For de novo tasks: backbone gen + seq design + struct pred + scoring (4 stages).
    For redesign tasks: seq design + struct pred + scoring (3 stages, backbone optional).

    Args:
        tools: List of tool names used.
        is_redesign: Whether the task is a redesign task.

    Returns:
        True if all required stages are present.
    """
    has_seq = _has_stage(tools, SEQ_DESIGN_TOOLS)
    has_struct = _has_stage(tools, STRUCT_PRED_TOOLS)
    has_scoring = _has_stage(tools, SCORING_TOOLS)

    if is_redesign:
        return has_seq and has_struct and has_scoring
    else:
        has_backbone = _has_stage(tools, BACKBONE_TOOLS)
        return has_backbone and has_seq and has_struct and has_scoring


def compute_completion_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lenient and strict completion rates per condition.

    Lenient: num_designs > 0 (produced at least one design).
    Strict: all required pipeline stages present in tools_used.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with columns: condition, mode, llm, n_tasks, n_lenient,
        rate_lenient, n_strict, rate_strict.
    """
    # Filter to the 8 paired conditions + Hardcoded
    target_conditions = [c for c in df["condition"].unique()
                         if c in COLORS]

    rows: list[dict[str, Any]] = []

    for condition in target_conditions:
        cond_df = df[df["condition"] == condition]
        info = CONDITION_MAP[condition]
        n_tasks = len(cond_df)
        n_lenient = 0
        n_strict = 0

        for _, row in cond_df.iterrows():
            task_id = row["task_id"]
            is_redesign = row["design_approach"] == "redesign"

            # Load raw result for tools_used and num_designs
            result = _load_raw_result(condition, task_id)
            if result is None:
                continue

            # Lenient: num_designs > 0
            num_designs = result.get("diversity_metrics", {}).get("num_designs", 0)
            if num_designs is not None and num_designs > 0:
                n_lenient += 1

            # Strict: all pipeline stages present
            tools = _get_tools_used(result)
            if _is_strict_complete(tools, is_redesign):
                n_strict += 1

        rate_lenient = (n_lenient / n_tasks * 100) if n_tasks > 0 else 0.0
        rate_strict = (n_strict / n_tasks * 100) if n_tasks > 0 else 0.0

        rows.append({
            "condition": condition,
            "mode": info["mode"],
            "llm": info["llm"],
            "n_tasks": n_tasks,
            "n_lenient": n_lenient,
            "rate_lenient": rate_lenient,
            "n_strict": n_strict,
            "rate_strict": rate_strict,
        })

    return pd.DataFrame(rows)


def compute_mode_deltas(completion_df: pd.DataFrame) -> pd.DataFrame:
    """Compute user vs benchmark delta in completion rates per model.

    Args:
        completion_df: DataFrame from compute_completion_rates().

    Returns:
        DataFrame with columns: llm, bm_lenient, us_lenient, delta_lenient,
        bm_strict, us_strict, delta_strict.
    """
    rows: list[dict[str, Any]] = []

    for llm in PAIRED_LLMS:
        bm = completion_df[(completion_df["llm"] == llm) & (completion_df["mode"] == "benchmark")]
        us = completion_df[(completion_df["llm"] == llm) & (completion_df["mode"] == "user")]

        if bm.empty or us.empty:
            continue

        bm_len = bm.iloc[0]["rate_lenient"]
        us_len = us.iloc[0]["rate_lenient"]
        bm_str = bm.iloc[0]["rate_strict"]
        us_str = us.iloc[0]["rate_strict"]

        rows.append({
            "llm": llm,
            "bm_lenient": bm_len,
            "us_lenient": us_len,
            "delta_lenient": us_len - bm_len,
            "bm_strict": bm_str,
            "us_strict": us_str,
            "delta_strict": us_str - bm_str,
        })

    return pd.DataFrame(rows)


def compute_transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-task transition matrix: BM success/fail vs User success/fail.

    Uses lenient completion (num_designs > 0) as the success criterion.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        Tuple of (transition_counts DataFrame, rescued_tasks DataFrame).
        transition_counts: columns = llm, both_fail, rescued, regressed, both_succeed.
        rescued_tasks: columns = llm, task_id, quality_user, quality_bm, design_approach.
    """
    transition_rows: list[dict[str, Any]] = []
    rescued_rows: list[dict[str, Any]] = []
    both_succeed_rows: list[dict[str, Any]] = []

    for llm in PAIRED_LLMS:
        bm_cond = f"{llm} benchmark"
        us_cond = f"{llm} user"

        bm_df = df[df["condition"] == bm_cond].set_index("task_id")
        us_df = df[df["condition"] == us_cond].set_index("task_id")

        common_tasks = sorted(set(bm_df.index) & set(us_df.index))

        counts = {"both_fail": 0, "rescued": 0, "regressed": 0, "both_succeed": 0}

        for tid in common_tasks:
            # Load raw results for num_designs
            bm_result = _load_raw_result(bm_cond, tid)
            us_result = _load_raw_result(us_cond, tid)

            bm_ok = False
            us_ok = False

            if bm_result is not None:
                bm_nd = bm_result.get("diversity_metrics", {}).get("num_designs", 0)
                bm_ok = bm_nd is not None and bm_nd > 0

            if us_result is not None:
                us_nd = us_result.get("diversity_metrics", {}).get("num_designs", 0)
                us_ok = us_nd is not None and us_nd > 0

            # Quality scores from DataFrame
            quality_bm = bm_df.loc[tid, "quality"] if tid in bm_df.index else 0.0
            quality_us = us_df.loc[tid, "quality"] if tid in us_df.index else 0.0
            design_approach = bm_df.loc[tid, "design_approach"] if tid in bm_df.index else "unknown"

            if not bm_ok and not us_ok:
                counts["both_fail"] += 1
            elif not bm_ok and us_ok:
                counts["rescued"] += 1
                rescued_rows.append({
                    "llm": llm,
                    "task_id": tid,
                    "quality_user": quality_us,
                    "quality_bm": quality_bm,
                    "design_approach": design_approach,
                })
            elif bm_ok and not us_ok:
                counts["regressed"] += 1
            else:
                counts["both_succeed"] += 1
                both_succeed_rows.append({
                    "llm": llm,
                    "task_id": tid,
                    "quality_user": quality_us,
                    "quality_bm": quality_bm,
                    "design_approach": design_approach,
                })

        transition_rows.append({"llm": llm, **counts, "n_total": len(common_tasks)})

    trans_df = pd.DataFrame(transition_rows)
    rescued_df = pd.DataFrame(rescued_rows) if rescued_rows else pd.DataFrame(
        columns=["llm", "task_id", "quality_user", "quality_bm", "design_approach"]
    )
    # Store both_succeed for quality comparison
    rescued_df.attrs["both_succeed_rows"] = both_succeed_rows

    return trans_df, rescued_df


# ── Figure 1: Grouped bar chart ─────────────────────────────────────────────

def plot_completion_bars(completion_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart: model x mode completion rates.

    Shows lenient and strict rates side by side for each condition.

    Args:
        completion_df: DataFrame from compute_completion_rates().

    Returns:
        matplotlib Figure.
    """
    # Order: paired models grouped, then hardcoded
    display_order: list[str] = []
    for llm in PAIRED_LLMS:
        display_order.append(f"{llm} user")
        display_order.append(f"{llm} benchmark")
    if "Hardcoded Pipeline" in completion_df["condition"].values:
        display_order.append("Hardcoded Pipeline")

    # Filter to only conditions present in data
    display_order = [c for c in display_order if c in completion_df["condition"].values]
    plot_df = completion_df.set_index("condition").loc[display_order].reset_index()

    n = len(plot_df)
    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Lenient bars
    bars_len = ax.bar(
        x - bar_w / 2, plot_df["rate_lenient"], bar_w,
        label="Lenient (num_designs > 0)",
        color=[COLORS.get(c, "#888888") for c in plot_df["condition"]],
        edgecolor="white", linewidth=0.5, alpha=0.85,
    )

    # Strict bars (hatched)
    bars_str = ax.bar(
        x + bar_w / 2, plot_df["rate_strict"], bar_w,
        label="Strict (all pipeline stages)",
        color=[COLORS.get(c, "#888888") for c in plot_df["condition"]],
        edgecolor="black", linewidth=0.5, alpha=0.5,
        hatch="///",
    )

    # Value labels on bars
    for bar_group in [bars_len, bars_str]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 1.0,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=6.5,
                )

    # Separator lines between model groups
    for i in range(1, len(PAIRED_LLMS)):
        sep_x = i * 2 - 0.5
        ax.axvline(sep_x, color="#cccccc", linestyle=":", linewidth=0.5, alpha=0.7)

    # Add a separator before Hardcoded if present
    if "Hardcoded Pipeline" in display_order:
        sep_x = len(PAIRED_LLMS) * 2 - 0.5
        ax.axvline(sep_x, color="#cccccc", linestyle=":", linewidth=0.5, alpha=0.7)

    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Pipeline Completion: User Mode vs Benchmark Mode")
    ax.set_xticks(x)

    # Shorter labels for x-axis
    short_labels: list[str] = []
    for c in plot_df["condition"]:
        if "user" in c:
            short_labels.append(c.replace(" user", "\n(user)"))
        elif "benchmark" in c:
            short_labels.append(c.replace(" benchmark", "\n(bm)"))
        else:
            short_labels.append(c)
    ax.set_xticklabels(short_labels, fontsize=7, ha="center")

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ── Figure 2: Transition heatmap ────────────────────────────────────────────

def plot_transition_heatmap(trans_df: pd.DataFrame) -> plt.Figure:
    """Create transition matrix heatmap: 4 LLMs x 4 transition categories.

    Args:
        trans_df: DataFrame from compute_transitions() with columns
                  llm, both_fail, rescued, regressed, both_succeed.

    Returns:
        matplotlib Figure.
    """
    categories = ["both_fail", "rescued", "regressed", "both_succeed"]
    cat_labels = ["BM fail\nUser fail", "BM fail\nUser success\n(rescued)",
                  "BM success\nUser fail\n(regressed)", "BM success\nUser success"]

    # Build matrix: rows=LLMs, cols=transition categories
    matrix = trans_df.set_index("llm")[categories].loc[PAIRED_LLMS].values
    n_totals = trans_df.set_index("llm").loc[PAIRED_LLMS, "n_total"].values

    # Convert to percentages
    pct_matrix = (matrix / n_totals[:, None] * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Custom colormap: white -> light blue -> dark blue
    im = ax.imshow(pct_matrix, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(pct_matrix.max(), 100))

    # Annotate cells with count and percentage
    for i in range(len(PAIRED_LLMS)):
        for j in range(len(categories)):
            count = int(matrix[i, j])
            pct = pct_matrix[i, j]
            text_color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    color=text_color, fontweight="bold" if pct > 30 else "normal")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.set_yticks(range(len(PAIRED_LLMS)))
    ax.set_yticklabels(PAIRED_LLMS, fontsize=9)
    ax.set_title("Task-Level Transitions: Benchmark Mode to User Mode", fontsize=11)

    # Color bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% of tasks", fontsize=8)

    # Border styling
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    return fig


# ── Figure 3: Quality box plot ──────────────────────────────────────────────

def plot_quality_boxplot(
    rescued_df: pd.DataFrame,
    both_succeed_rows: list[dict[str, Any]],
) -> plt.Figure | None:
    """Box plot comparing quality scores: rescued vs both-succeed tasks.

    Args:
        rescued_df: DataFrame of rescued tasks with quality_user column.
        both_succeed_rows: List of dicts for both-succeed tasks.

    Returns:
        matplotlib Figure, or None if insufficient data.
    """
    both_df = pd.DataFrame(both_succeed_rows) if both_succeed_rows else pd.DataFrame(
        columns=["llm", "task_id", "quality_user", "quality_bm", "design_approach"]
    )

    if rescued_df.empty and both_df.empty:
        print("  WARNING: No rescued or both-succeed tasks found. Skipping quality boxplot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Panel (a): Quality distribution by transition type (all LLMs pooled)
    ax = axes[0]
    data_groups: list[list[float]] = []
    labels: list[str] = []
    box_colors: list[str] = []

    if not rescued_df.empty:
        data_groups.append(rescued_df["quality_user"].tolist())
        labels.append(f"Rescued\n(n={len(rescued_df)})")
        box_colors.append("#e74c3c")

    if not both_df.empty:
        data_groups.append(both_df["quality_user"].tolist())
        labels.append(f"Both succeed\n(n={len(both_df)})")
        box_colors.append("#2ecc71")

    if data_groups:
        bp = ax.boxplot(
            data_groups, patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Quality Score (User Mode)")
    ax.set_title("(a) Quality by Transition Type", fontsize=10)
    ax.grid(axis="y", alpha=0.2, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): Per-LLM rescued quality
    ax = axes[1]
    if not rescued_df.empty:
        llm_data: list[list[float]] = []
        llm_labels: list[str] = []
        llm_colors_list: list[str] = []
        for llm in PAIRED_LLMS:
            subset = rescued_df[rescued_df["llm"] == llm]["quality_user"]
            if not subset.empty:
                llm_data.append(subset.tolist())
                llm_labels.append(f"{llm}\n(n={len(subset)})")
                llm_colors_list.append(MODEL_COLORS.get(llm, "#888888"))

        if llm_data:
            bp = ax.boxplot(
                llm_data, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
            )
            for patch, color in zip(bp["boxes"], llm_colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticklabels(llm_labels, fontsize=7)
    else:
        ax.text(0.5, 0.5, "No rescued tasks", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888888")

    ax.set_title("(b) Rescued Task Quality by Model", fontsize=10)
    ax.grid(axis="y", alpha=0.2, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(
    completion_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    rescued_df: pd.DataFrame,
    both_succeed_rows: list[dict[str, Any]],
) -> str:
    """Generate a Markdown summary report.

    Args:
        completion_df: Per-condition completion rates.
        delta_df: User-benchmark deltas per model.
        trans_df: Transition counts per model.
        rescued_df: Rescued task details.
        both_succeed_rows: Both-succeed task details.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []
    lines.append("# Mode Completion Analysis Report")
    lines.append("")
    lines.append("## 1. Per-Condition Completion Rates")
    lines.append("")
    lines.append("| Condition | N | Lenient (%) | Strict (%) |")
    lines.append("|-----------|---|-------------|------------|")
    for _, row in completion_df.iterrows():
        lines.append(
            f"| {row['condition']} | {row['n_tasks']} | "
            f"{row['rate_lenient']:.1f} ({row['n_lenient']}) | "
            f"{row['rate_strict']:.1f} ({row['n_strict']}) |"
        )

    lines.append("")
    lines.append("## 2. User vs Benchmark Delta")
    lines.append("")
    lines.append("| Model | BM Lenient | User Lenient | Delta Lenient | BM Strict | User Strict | Delta Strict |")
    lines.append("|-------|------------|--------------|---------------|-----------|-------------|--------------|")
    for _, row in delta_df.iterrows():
        lines.append(
            f"| {row['llm']} | {row['bm_lenient']:.1f}% | {row['us_lenient']:.1f}% | "
            f"{row['delta_lenient']:+.1f}pp | {row['bm_strict']:.1f}% | "
            f"{row['us_strict']:.1f}% | {row['delta_strict']:+.1f}pp |"
        )

    # Average delta
    if not delta_df.empty:
        avg_delta_len = delta_df["delta_lenient"].mean()
        avg_delta_str = delta_df["delta_strict"].mean()
        lines.append(f"| **Average** | | | **{avg_delta_len:+.1f}pp** | | | **{avg_delta_str:+.1f}pp** |")

    lines.append("")
    lines.append("## 3. Transition Analysis")
    lines.append("")
    lines.append("| Model | Both Fail | Rescued | Regressed | Both Succeed | N |")
    lines.append("|-------|-----------|---------|-----------|--------------|---|")
    for _, row in trans_df.iterrows():
        n = row["n_total"]
        lines.append(
            f"| {row['llm']} | {row['both_fail']} ({row['both_fail']/n*100:.0f}%) | "
            f"{row['rescued']} ({row['rescued']/n*100:.0f}%) | "
            f"{row['regressed']} ({row['regressed']/n*100:.0f}%) | "
            f"{row['both_succeed']} ({row['both_succeed']/n*100:.0f}%) | {n} |"
        )

    # Rescued task quality
    lines.append("")
    lines.append("## 4. Rescued Task Quality")
    lines.append("")
    if not rescued_df.empty:
        median_q = rescued_df["quality_user"].median()
        mean_q = rescued_df["quality_user"].mean()
        lines.append(f"- Total rescued tasks: {len(rescued_df)}")
        lines.append(f"- Mean quality (user mode): {mean_q:.1f}")
        lines.append(f"- Median quality (user mode): {median_q:.1f}")
        lines.append("")
        both_df = pd.DataFrame(both_succeed_rows)
        if not both_df.empty:
            both_mean = both_df["quality_user"].mean()
            both_median = both_df["quality_user"].median()
            lines.append(f"- Both-succeed mean quality (user): {both_mean:.1f}")
            lines.append(f"- Both-succeed median quality (user): {both_median:.1f}")
    else:
        lines.append("No rescued tasks found.")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by mode_completion_analysis.py*")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full mode completion analysis pipeline."""
    _setup_style()

    print("Loading results...")
    df = load_all()
    print(f"  Loaded {len(df)} rows "
          f"({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)")

    # ── 1. Completion rates ──────────────────────────────────────────────────
    print("\n=== Computing Completion Rates ===")
    completion_df = compute_completion_rates(df)
    completion_df.to_csv(FIGURES_DIR / "mode_completion_stats.csv", index=False)
    print(completion_df[["condition", "n_tasks", "rate_lenient", "rate_strict"]].to_string(
        index=False, float_format="%.1f"
    ))

    # ── 2. Mode deltas ───────────────────────────────────────────────────────
    print("\n=== User vs Benchmark Delta ===")
    delta_df = compute_mode_deltas(completion_df)
    print(delta_df.to_string(index=False, float_format="%.1f"))

    # ── 3. Transitions ───────────────────────────────────────────────────────
    print("\n=== Transition Analysis ===")
    trans_df, rescued_df = compute_transitions(df)
    both_succeed_rows: list[dict[str, Any]] = rescued_df.attrs.get("both_succeed_rows", [])
    print(trans_df.to_string(index=False))

    if not rescued_df.empty:
        print(f"\n  Rescued tasks: {len(rescued_df)}")
        print(f"  Mean rescued quality (user): {rescued_df['quality_user'].mean():.1f}")
    else:
        print("\n  No rescued tasks found.")

    # ── 4. Figures ───────────────────────────────────────────────────────────
    print("\n=== Generating Figures ===")

    # Figure 1: Grouped bar chart
    fig1 = plot_completion_bars(completion_df)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "fig_mode_completion.pdf", dpi=300, facecolor="white")
    fig1.savefig(FIGURES_DIR / "fig_mode_completion.png", dpi=300, facecolor="white")
    plt.close(fig1)
    print(f"  Saved: {FIGURES_DIR / 'fig_mode_completion.pdf'}")
    print(f"  Saved: {FIGURES_DIR / 'fig_mode_completion.png'}")

    # Figure 2: Transition heatmap
    fig2 = plot_transition_heatmap(trans_df)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "fig_mode_transitions.pdf", dpi=300, facecolor="white")
    fig2.savefig(FIGURES_DIR / "fig_mode_transitions.png", dpi=300, facecolor="white")
    plt.close(fig2)
    print(f"  Saved: {FIGURES_DIR / 'fig_mode_transitions.pdf'}")
    print(f"  Saved: {FIGURES_DIR / 'fig_mode_transitions.png'}")

    # Figure 3: Quality boxplot
    fig3 = plot_quality_boxplot(rescued_df, both_succeed_rows)
    if fig3 is not None:
        fig3.tight_layout()
        fig3.savefig(FIGURES_DIR / "fig_mode_quality_rescued.pdf", dpi=300, facecolor="white")
        fig3.savefig(FIGURES_DIR / "fig_mode_quality_rescued.png", dpi=300, facecolor="white")
        plt.close(fig3)
        print(f"  Saved: {FIGURES_DIR / 'fig_mode_quality_rescued.pdf'}")
        print(f"  Saved: {FIGURES_DIR / 'fig_mode_quality_rescued.png'}")

    # ── 5. Report ────────────────────────────────────────────────────────────
    report = generate_report(completion_df, delta_df, trans_df, rescued_df, both_succeed_rows)
    report_path = FIGURES_DIR / "mode_completion_report.md"
    report_path.write_text(report)
    print(f"\n  Report saved: {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    if not delta_df.empty:
        avg_len = delta_df["delta_lenient"].mean()
        avg_str = delta_df["delta_strict"].mean()
        print(f"  Average lenient delta (user - bm): {avg_len:+.1f} pp")
        print(f"  Average strict delta (user - bm):  {avg_str:+.1f} pp")

    if not trans_df.empty:
        total_rescued = trans_df["rescued"].sum()
        total_regressed = trans_df["regressed"].sum()
        total_tasks = trans_df["n_total"].sum()
        print(f"  Total rescued tasks: {total_rescued}/{total_tasks} "
              f"({total_rescued/total_tasks*100:.1f}%)")
        print(f"  Total regressed tasks: {total_regressed}/{total_tasks} "
              f"({total_regressed/total_tasks*100:.1f}%)")

    print("\nDone. All outputs saved to figures/")


if __name__ == "__main__":
    main()
