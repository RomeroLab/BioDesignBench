#!/usr/bin/env python3
"""BDB-023 Fig 3: Benchmark vs User Mode Paired Comparison.

3-panel figure comparing benchmark (BM) and user (US) mode scores for 4 LLMs:
  Panel A: Paired dot plot with arrows (BM -> US) per LLM
  Panel B: Per-component uplift heatmap (LLMs x 6 components)
  Panel C: Per-task delta distribution (histogram/KDE overlay)

Usage:
    python -m scripts.analysis.bdb_023_fig3_mode_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all

# ── Constants ──────────────────────────────────────────────────────────────

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]

# Only LLMs with paired BM/US data (excludes Hardcoded Pipeline)
PAIRED_LLMS = ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]

# Okabe-Ito palette from docs/color_palette.md
LLM_COLORS: dict[str, str] = {
    "DeepSeek V3": "#CC79A7",     # Reddish Purple
    "GPT-5": "#56B4E9",           # Sky Blue
    "Sonnet 4.5": "#E69F00",      # Orange
    "Gemini 2.5 Pro": "#009E73",  # Bluish Green
}

# Markers for BM vs US
MARKER_BM = "o"  # circle
MARKER_US = "^"  # triangle

OUT_DIR = PROJECT_ROOT / "results" / "analysis"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "font.family": "sans-serif",
})


# ── Data helpers ───────────────────────────────────────────────────────────


def _build_paired_df(df: pd.DataFrame) -> pd.DataFrame:
    """Merge benchmark and user rows into paired rows per (task_id, llm).

    Args:
        df: DataFrame from load_all() with columns including
            task_id, mode, llm, total, and the 6 component scores.

    Returns:
        DataFrame with columns: task_id, llm, total_bm, total_us, delta,
        and {comp}_bm, {comp}_us, {comp}_delta for each component.
    """
    bm = df[df["mode"] == "benchmark"].copy()
    us = df[df["mode"] == "user"].copy()

    # Rename score columns for merge
    bm_cols = {"total": "total_bm"}
    us_cols = {"total": "total_us"}
    for comp in COMPONENTS:
        bm_cols[comp] = f"{comp}_bm"
        us_cols[comp] = f"{comp}_us"

    bm = bm.rename(columns=bm_cols)
    us = us.rename(columns=us_cols)

    # Select merge keys + renamed columns
    bm_keep = ["task_id", "llm", "total_bm"] + [f"{c}_bm" for c in COMPONENTS]
    us_keep = ["task_id", "llm", "total_us"] + [f"{c}_us" for c in COMPONENTS]

    paired = pd.merge(
        bm[bm_keep],
        us[us_keep],
        on=["task_id", "llm"],
        how="inner",
    )

    # Compute deltas
    paired["delta"] = paired["total_us"] - paired["total_bm"]
    for comp in COMPONENTS:
        paired[f"{comp}_delta"] = paired[f"{comp}_us"] - paired[f"{comp}_bm"]

    return paired.reset_index(drop=True)


def _llm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-LLM summary stats (benchmark mean, user mean, delta).

    Args:
        df: DataFrame from load_all().

    Returns:
        DataFrame indexed by llm with columns: bm_mean, us_mean, delta.
    """
    bm = (
        df[df["mode"] == "benchmark"]
        .groupby("llm")["total"]
        .mean()
        .rename("bm_mean")
    )
    us = (
        df[df["mode"] == "user"]
        .groupby("llm")["total"]
        .mean()
        .rename("us_mean")
    )
    summary = pd.concat([bm, us], axis=1).dropna()
    summary["delta"] = summary["us_mean"] - summary["bm_mean"]
    return summary.reset_index()


# ── Panel drawing functions ────────────────────────────────────────────────


def _draw_panel_a(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Panel A: Paired dot plot with arrows from BM to US per LLM.

    Args:
        ax: Matplotlib axes to draw on.
        summary: DataFrame with llm, bm_mean, us_mean, delta columns.
    """
    # Order LLMs by delta descending
    ordered = summary.sort_values("delta", ascending=True).reset_index(drop=True)

    for i, row in ordered.iterrows():
        llm = row["llm"]
        color = LLM_COLORS.get(llm, "#333333")
        y = i

        # BM dot
        ax.scatter(row["bm_mean"], y, marker=MARKER_BM, s=100, color=color,
                   edgecolors="black", linewidths=0.5, zorder=3)
        # US dot
        ax.scatter(row["us_mean"], y, marker=MARKER_US, s=100, color=color,
                   edgecolors="black", linewidths=0.5, zorder=3)
        # Arrow from BM to US
        ax.annotate(
            "",
            xy=(row["us_mean"], y),
            xytext=(row["bm_mean"], y),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=2,
                shrinkA=6,
                shrinkB=6,
            ),
        )
        # Delta annotation
        x_text = max(row["bm_mean"], row["us_mean"]) + 1.5
        sign = "+" if row["delta"] >= 0 else ""
        ax.text(
            x_text, y, f"{sign}{row['delta']:.1f}",
            va="center", ha="left", fontsize=10, fontweight="bold",
            color=color,
        )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["llm"], fontsize=10)
    ax.set_xlabel("Mean Score (0-100)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("A. Benchmark vs User Mode Scores", fontsize=12, fontweight="bold")

    # Legend for markers
    ax.scatter([], [], marker=MARKER_BM, s=60, color="gray", edgecolors="black",
               linewidths=0.5, label="Benchmark")
    ax.scatter([], [], marker=MARKER_US, s=60, color="gray", edgecolors="black",
               linewidths=0.5, label="User")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def _draw_panel_b(ax: plt.Axes, paired: pd.DataFrame) -> None:
    """Panel B: Per-component uplift heatmap (LLMs x 6 components).

    Args:
        ax: Matplotlib axes to draw on.
        paired: DataFrame from _build_paired_df with component deltas.
    """
    # Compute mean delta per (llm, component)
    llm_order = [llm for llm in PAIRED_LLMS if llm in paired["llm"].unique()]

    matrix = np.zeros((len(llm_order), len(COMPONENTS)))
    for i, llm in enumerate(llm_order):
        llm_data = paired[paired["llm"] == llm]
        for j, comp in enumerate(COMPONENTS):
            matrix[i, j] = llm_data[f"{comp}_delta"].mean()

    # Determine symmetric color range
    abs_max = max(np.abs(matrix).max(), 1.0)

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[c.title() for c in COMPONENTS],
        yticklabels=llm_order,
        cmap="RdBu_r",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "User - Benchmark", "shrink": 0.8},
    )
    ax.set_title("B. Per-Component Uplift", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)


def _draw_panel_c(ax: plt.Axes, paired: pd.DataFrame) -> None:
    """Panel C: Stage-level plan vs execution rate (operationalization gap).

    Shows plan rate vs execution rate for 4 pipeline stages using
    DeepSeek V3 and Sonnet 4.5 traces (most reliable), benchmark mode only.
    The key finding: sequence design is planned 30% but executed 0%.

    Args:
        ax: Matplotlib axes to draw on.
        paired: DataFrame from _build_paired_df (unused, kept for API compat).
    """
    # Load reasoning trace data
    traces_path = PROJECT_ROOT / "figures" / "reasoning_trace_summary.csv"
    if not traces_path.exists():
        ax.text(0.5, 0.5, "reasoning_trace_summary.csv\nnot found",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title("C. Plan vs Execution by Stage", fontsize=12, fontweight="bold")
        return

    traces = pd.read_csv(traces_path)
    reliable = traces[traces["model"].isin(["DeepSeek V3", "Sonnet 4.5"])]

    steps = ["backbone", "sequence", "structure", "scoring"]
    step_labels = ["Backbone\nGeneration", "Sequence\nDesign", "Structure\nPrediction", "Scoring"]

    # Benchmark mode data (shows the clearest gap)
    bm = reliable[reliable["mode"] == "benchmark"]
    plan_rates = [bm[f"plan_{s}"].mean() for s in steps]
    exec_rates = [bm[f"exec_{s}"].mean() for s in steps]

    # Also compute user mode for comparison text
    us = reliable[reliable["mode"] == "user"]
    us_plan = [us[f"plan_{s}"].mean() for s in steps]
    us_exec = [us[f"exec_{s}"].mean() for s in steps]

    x = np.arange(len(steps))
    bar_width = 0.32
    plan_color = "#56B4E9"   # Sky Blue
    exec_color = "#E69F00"   # Orange
    gap_color = "#d62728"    # Red

    ax.bar(x - bar_width / 2, plan_rates, bar_width,
           color=plan_color, alpha=0.85, label="Planned (reasoning)",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + bar_width / 2, exec_rates, bar_width,
           color=exec_color, alpha=0.85, label="Executed (tool calls)",
           edgecolor="white", linewidth=0.5)

    # Annotate gaps above bar pairs
    for i, (p, e) in enumerate(zip(plan_rates, exec_rates)):
        gap = p - e
        if gap > 0.02:
            y_ann = max(p, e) + 0.03
            is_large = gap > 0.1
            ax.text(
                i, y_ann,
                f"Δ {gap:.0%}",
                fontsize=8, fontweight="bold" if is_large else "normal",
                color=gap_color if is_large else "#999999",
                ha="center", va="bottom",
            )

    # Callout for the dramatic sequence design gap
    ax.annotate(
        "Seq. design:\nplanned 30%,\nexecuted 0%",
        xy=(1 - bar_width / 2, plan_rates[1]),
        xytext=(2.2, 0.58),
        fontsize=8, fontweight="bold", color=gap_color,
        arrowprops=dict(arrowstyle="->", color=gap_color, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                  edgecolor=gap_color, alpha=0.95),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("C. Plan vs Execution by Stage (Unguided)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Subtitle: data source
    ax.text(
        0.98, 0.02,
        "DeepSeek V3 + Sonnet 4.5\n(benchmark mode, n=152)",
        transform=ax.transAxes, fontsize=7, color="#888888",
        ha="right", va="bottom",
        fontstyle="italic",
    )


# ── Main figure generator ─────────────────────────────────────────────────


def generate_fig3(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Generate Fig 3: Benchmark vs User Mode Paired Comparison.

    Args:
        df: DataFrame from load_all() containing all conditions.
        save_path: Output file path. Defaults to results/analysis/fig3_mode_comparison.png.
    """
    if save_path is None:
        save_path = OUT_DIR / "fig3_mode_comparison.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to paired LLMs only (exclude hardcoded pipeline)
    df_paired_llms = df[df["llm"].isin(PAIRED_LLMS)].copy()

    # Build paired data
    paired = _build_paired_df(df_paired_llms)
    summary = _llm_summary(df_paired_llms)
    summary = summary[summary["llm"].isin(PAIRED_LLMS)]

    # Create figure with gridspec layout:
    #   Top row: Panel A (full width)
    #   Bottom row: Panel B (left), Panel C (right)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, :])   # Panel A spans full top row
    ax_b = fig.add_subplot(gs[1, 0])   # Panel B bottom-left
    ax_c = fig.add_subplot(gs[1, 1])   # Panel C bottom-right

    _draw_panel_a(ax_a, summary)
    _draw_panel_b(ax_b, paired)
    _draw_panel_c(ax_c, paired)

    fig.suptitle(
        "Fig 3: Benchmark vs User Mode Paired Comparison",
        fontsize=14, fontweight="bold", y=0.98,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Load data and generate figure."""
    print("Loading results...")
    df = load_all()

    # Print summary table
    summary = _llm_summary(df)
    summary = summary[summary["llm"].isin(PAIRED_LLMS)]
    print("\nPaired LLM Summary:")
    print(f"  {'LLM':<20s} {'Benchmark':>10s} {'User':>10s} {'Delta':>10s}")
    print("  " + "-" * 52)
    for _, row in summary.sort_values("delta", ascending=False).iterrows():
        print(
            f"  {row['llm']:<20s} {row['bm_mean']:>10.1f} {row['us_mean']:>10.1f} "
            f"{row['delta']:>+10.1f}"
        )

    print("\nGenerating Fig 3...")
    generate_fig3(df)
    print("Done.")


if __name__ == "__main__":
    main()
