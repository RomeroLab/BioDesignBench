#!/usr/bin/env python3
"""Figure 3 Panel C replacement options.

Option A: Reasoning Trace Excerpts — text panel showing actual agent behavior
Option B: Stage-Level Plan-Execution Gap Decomposition — grouped bar chart

Usage:
    python -m scripts.analysis.fig3_panelC_options
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "figures"

# Okabe-Ito palette
COLORS = {
    "deepseek": "#CC79A7",
    "gpt": "#56B4E9",
    "sonnet": "#E69F00",
    "gemini": "#009E73",
    "oracle": "#D55E00",
    "hardcoded": "#0072B2",
    "expert": "#666666",
}

# Case colors
CASE_A_COLOR = "#2ca02c"  # green — full knowledge
CASE_B_COLOR = "#d62728"  # red — tool gap (knew but couldn't execute)

# Stage colors (from tool categories in color_palette.md)
PLAN_COLOR = "#56B4E9"    # Sky Blue
EXEC_COLOR = "#E69F00"    # Orange
GAP_COLOR = "#d62728"     # Red accent for gap annotation

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "font.family": "sans-serif",
})


# ── Option A: Reasoning Trace Excerpts ─────────────────────────────────────


def draw_option_a(save_path: Path | None = None) -> None:
    """Panel C Option A: Reasoning trace excerpts showing operationalization gap.

    Three text boxes showing actual agent reasoning → action pairs:
    - Case B (tool gap): Agent mentions the step but doesn't execute it
    - Case A (full knowledge): Agent mentions and executes the step
    """
    if save_path is None:
        save_path = OUT_DIR / "figure3_panelC_optionA.png"

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(
        5.0, 11.7, "C. Operationalization Gap: Reasoning vs Execution",
        fontsize=11, fontweight="bold", ha="center", va="top",
    )

    # Three trace examples stacked vertically
    examples = [
        {
            "case": "B",
            "label": "Tool Gap",
            "color": CASE_B_COLOR,
            "bg": "#fff0f0",
            "agent": "Sonnet 4.5",
            "task": "cpx_str_007 (user)",
            "plan_line": (
                'Plan:  "...backbone generation and sequence optimization"'
            ),
            "action_line": (
                "Exec:  predict_structure → score_stability"
            ),
            "verdict": "BB gen + Seq design skipped",
            "verdict_symbol": "✗",
        },
        {
            "case": "B",
            "label": "Tool Gap",
            "color": CASE_B_COLOR,
            "bg": "#fff0f0",
            "agent": "DeepSeek V3",
            "task": "scaffold_002 (benchmark)",
            "plan_line": (
                'Plan:  "...RFdiffusion backbones, then ProteinMPNN sequences"'
            ),
            "action_line": (
                "Exec:  generate_backbone → execute_python(heuristic)"
            ),
            "verdict": "Seq design via LLM fallback",
            "verdict_symbol": "✗",
        },
        {
            "case": "A",
            "label": "Full Pipeline",
            "color": CASE_A_COLOR,
            "bg": "#f0fff0",
            "agent": "DeepSeek V3",
            "task": "cpx_sig_008 (user)",
            "plan_line": (
                'Plan:  "...RFdiffusion → ProteinMPNN → AF2 validation"'
            ),
            "action_line": (
                "Exec:  generate_backbone → optimize_sequence → predict_complex"
            ),
            "verdict": "Full pipeline executed",
            "verdict_symbol": "✓",
        },
    ]

    y_positions = [9.0, 6.0, 3.0]  # top of each box
    box_height = 2.6
    box_width = 9.4
    x_left = 0.3

    for example, y_top in zip(examples, y_positions):
        # Background box
        rect = mpatches.FancyBboxPatch(
            (x_left, y_top - box_height),
            box_width, box_height,
            boxstyle="round,pad=0.15",
            facecolor=example["bg"],
            edgecolor=example["color"],
            linewidth=1.8,
            alpha=0.9,
        )
        ax.add_patch(rect)

        # Case badge
        badge_x = x_left + 0.25
        badge_y = y_top - 0.35
        badge = mpatches.FancyBboxPatch(
            (badge_x, badge_y - 0.35),
            1.8, 0.45,
            boxstyle="round,pad=0.06",
            facecolor=example["color"],
            edgecolor="none",
            alpha=0.9,
        )
        ax.add_patch(badge)
        ax.text(
            badge_x + 0.9, badge_y - 0.12,
            f"Case {example['case']}: {example['label']}",
            fontsize=8, fontweight="bold", color="white",
            ha="center", va="center",
        )

        # Agent + task info (right of badge)
        ax.text(
            badge_x + 2.15, badge_y - 0.12,
            f"{example['agent']}  ·  {example['task']}",
            fontsize=8.5, color="#444444", va="center",
            fontstyle="italic",
        )

        # Plan text line
        plan_y = y_top - 1.2
        ax.text(
            x_left + 0.5, plan_y,
            example["plan_line"],
            fontsize=8.5, color="#333333", va="center",
            fontfamily="monospace",
        )

        # Action line
        action_y = y_top - 1.85
        ax.text(
            x_left + 0.5, action_y,
            example["action_line"],
            fontsize=8.5, color="#333333", va="center",
            fontfamily="monospace",
        )

        # Verdict badge (right side, bottom)
        verdict_x = x_left + box_width - 0.3
        verdict_y = y_top - box_height + 0.4
        ax.text(
            verdict_x, verdict_y,
            f"{example['verdict_symbol']} {example['verdict']}",
            fontsize=9, fontweight="bold",
            color=example["color"],
            ha="right", va="center",
        )

    # Caption
    ax.text(
        5.0, 0.15,
        "Agents articulate correct pipelines in reasoning but fail to operationalize generative steps.",
        fontsize=8.5, ha="center", va="bottom", color="#777777",
        fontstyle="italic",
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Option A: {save_path}")


# ── Option B: Stage-Level Gap Decomposition ────────────────────────────────


def draw_option_b(save_path: Path | None = None) -> None:
    """Panel C Option B: Stage-level plan vs execution rate grouped bar chart.

    Shows plan rate vs execution rate for 4 pipeline stages
    (backbone generation, sequence design, structure prediction, scoring),
    using only DeepSeek V3 and Sonnet 4.5 traces (most reliable).
    """
    if save_path is None:
        save_path = OUT_DIR / "figure3_panelC_optionB.png"

    traces_path = PROJECT_ROOT / "figures" / "reasoning_trace_summary.csv"
    traces = pd.read_csv(traces_path)

    # Filter to reliable models only
    reliable = traces[traces["model"].isin(["DeepSeek V3", "Sonnet 4.5"])]

    steps = ["backbone", "sequence", "structure", "scoring"]
    step_labels = [
        "Backbone\nGeneration",
        "Sequence\nDesign",
        "Structure\nPrediction",
        "Scoring /\nValidation",
    ]

    # Compute per-mode rates
    modes = ["benchmark", "user"]
    mode_labels = ["Unguided (Benchmark)", "Guided (User)"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
    fig.suptitle(
        "C. Plan vs Execution Rate by Pipeline Stage",
        fontsize=11, fontweight="bold", y=1.01,
    )

    bar_width = 0.32
    x = np.arange(len(steps))

    for ax, mode, mode_label in zip(axes, modes, mode_labels):
        sub = reliable[reliable["mode"] == mode]
        plan_rates = [sub[f"plan_{s}"].mean() for s in steps]
        exec_rates = [sub[f"exec_{s}"].mean() for s in steps]

        ax.bar(
            x - bar_width / 2, plan_rates, bar_width,
            color=PLAN_COLOR, alpha=0.85, label="Planned (in reasoning)",
            edgecolor="white", linewidth=0.5,
        )
        ax.bar(
            x + bar_width / 2, exec_rates, bar_width,
            color=EXEC_COLOR, alpha=0.85, label="Executed (tool calls)",
            edgecolor="white", linewidth=0.5,
        )

        # Annotate gaps above bar pairs
        for i, (p, e) in enumerate(zip(plan_rates, exec_rates)):
            gap = p - e
            if gap > 0.02:
                y_ann = max(p, e) + 0.04
                is_large = gap > 0.1
                ax.text(
                    i, y_ann,
                    f"Δ {gap:.0%}",
                    fontsize=7.5, fontweight="bold" if is_large else "normal",
                    color=GAP_COLOR if is_large else "#999999",
                    ha="center", va="bottom",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, fontsize=8)
        ax.set_title(mode_label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Rate", fontsize=10)
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # Highlight the dramatic sequence design gap in benchmark mode
    axes[0].annotate(
        "Sequence design:\nplanned 30%, executed 0%",
        xy=(1 - bar_width / 2, 0.30),
        xytext=(2.0, 0.60),
        fontsize=8,
        fontweight="bold",
        color=GAP_COLOR,
        arrowprops=dict(arrowstyle="->", color=GAP_COLOR, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                  edgecolor=GAP_COLOR, alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Option B: {save_path}")


# ── Option B variant: Single panel with both modes overlaid ────────────────


def draw_option_b_combined(save_path: Path | None = None) -> None:
    """Option B variant: Single panel with benchmark + user overlay.

    More compact: 4 stage groups, each with 4 bars
    (BM-plan, BM-exec, US-plan, US-exec).
    """
    if save_path is None:
        save_path = OUT_DIR / "figure3_panelC_optionB_combined.png"

    traces_path = PROJECT_ROOT / "figures" / "reasoning_trace_summary.csv"
    traces = pd.read_csv(traces_path)
    reliable = traces[traces["model"].isin(["DeepSeek V3", "Sonnet 4.5"])]

    steps = ["backbone", "sequence", "structure", "scoring"]
    step_labels = [
        "Backbone\nGeneration",
        "Sequence\nDesign",
        "Structure\nPrediction",
        "Scoring",
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    bm = reliable[reliable["mode"] == "benchmark"]
    us = reliable[reliable["mode"] == "user"]

    bm_plan = [bm[f"plan_{s}"].mean() for s in steps]
    bm_exec = [bm[f"exec_{s}"].mean() for s in steps]
    us_plan = [us[f"plan_{s}"].mean() for s in steps]
    us_exec = [us[f"exec_{s}"].mean() for s in steps]

    x = np.arange(len(steps))
    w = 0.18

    # Benchmark bars (hatched)
    ax.bar(x - 1.5 * w, bm_plan, w, color=PLAN_COLOR, alpha=0.5,
           hatch="//", edgecolor=PLAN_COLOR, linewidth=0.5, label="BM Plan")
    ax.bar(x - 0.5 * w, bm_exec, w, color=EXEC_COLOR, alpha=0.5,
           hatch="//", edgecolor=EXEC_COLOR, linewidth=0.5, label="BM Exec")
    # User bars (solid)
    ax.bar(x + 0.5 * w, us_plan, w, color=PLAN_COLOR, alpha=0.85,
           edgecolor="white", linewidth=0.5, label="US Plan")
    ax.bar(x + 1.5 * w, us_exec, w, color=EXEC_COLOR, alpha=0.85,
           edgecolor="white", linewidth=0.5, label="US Exec")

    # Gap annotations for benchmark sequence design
    gap_bm_seq = bm_plan[1] - bm_exec[1]
    if gap_bm_seq > 0.01:
        ax.annotate(
            f"Gap: {gap_bm_seq:.0%}\n(plan ≠ exec)",
            xy=(1 - 0.5 * w, bm_exec[1] + 0.02),
            xytext=(1.5, 0.55),
            fontsize=7.5, fontweight="bold", color=GAP_COLOR,
            arrowprops=dict(arrowstyle="->", color=GAP_COLOR, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                      edgecolor=GAP_COLOR, alpha=0.9),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "C. Plan vs Execution Rate by Pipeline Stage",
        fontsize=11, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Option B combined: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate all Panel C options."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_option_a()
    draw_option_b()
    draw_option_b_combined()
    print("\nAll Panel C options generated in figures/")


if __name__ == "__main__":
    main()
