#!/usr/bin/env python3
"""BioDesignBench Fig 1: Overview Schematic for NMI paper.

Three-panel left-to-right flow:
  1. Task Design (76 tasks, 2x5 taxonomy)
  2. MCP Framework (17 tools, BM vs US modes)
  3. Evaluation Pipeline (6-component rubric)
  + Bottom: 9 conditions comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── Color palette (Nature-style) ─────────────────────────────────────────
C_BLUE = "#4477AA"
C_CYAN = "#66CCEE"
C_GREEN = "#228833"
C_YELLOW = "#CCBB44"
C_RED = "#EE6677"
C_PURPLE = "#AA3377"
C_GRAY = "#BBBBBB"
C_DARK = "#333333"
C_LIGHT_BG = "#F5F5F5"
C_WHITE = "#FFFFFF"

# Panel background colors
C_PANEL1 = "#EBF0F7"  # light blue
C_PANEL2 = "#F0EBF7"  # light purple
C_PANEL3 = "#EBF7F0"  # light green


def draw_rounded_box(ax, x, y, w, h, label, color, fontsize=9, text_color=C_DARK,
                     alpha=1.0, linewidth=1.0, edgecolor=C_DARK):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center", fontsize=fontsize,
        color=text_color, fontweight="bold", zorder=3,
        transform=ax.transData,
    )
    return box


def draw_panel_bg(ax, x, y, w, h, color, title, title_fontsize=11):
    """Draw panel background with title."""
    bg = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=C_DARK,
        linewidth=1.5, alpha=0.5, zorder=0,
    )
    ax.add_patch(bg)
    ax.text(
        x + w / 2, y + h - 0.02, title,
        ha="center", va="top", fontsize=title_fontsize,
        fontweight="bold", color=C_DARK, zorder=1,
    )


def draw_arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=2):
    """Draw a simple arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=6,head_length=4",
        color=color, linewidth=lw, zorder=5,
        mutation_scale=10,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Panel 1: Task Design (left) ──────────────────────────────────────
    p1_x, p1_y, p1_w, p1_h = 0.02, 0.15, 0.28, 0.80
    draw_panel_bg(ax, p1_x, p1_y, p1_w, p1_h, C_PANEL1, "Task Design")

    # Title info
    ax.text(p1_x + p1_w / 2, p1_y + p1_h - 0.07, "76 Protein Design Tasks",
            ha="center", va="center", fontsize=10, color=C_DARK)

    # 2x5 taxonomy matrix
    subjects = ["Ab", "Enz", "Bnd", "Scf", "FP"]
    approaches = ["De novo", "Redesign"]
    counts = [
        [4, 2, 19, 21, 1],   # de_novo
        [5, 10, 0, 4, 10],   # redesign
    ]

    mx, my = p1_x + 0.04, p1_y + 0.22
    cw, ch = 0.038, 0.10
    gap = 0.005

    # Column headers
    for j, subj in enumerate(subjects):
        ax.text(mx + 0.04 + j * (cw + gap) + cw / 2, my + 2 * (ch + gap) + 0.02,
                subj, ha="center", va="center", fontsize=7, fontweight="bold", color=C_DARK)

    # Row headers + cells
    for i, approach in enumerate(approaches):
        row_y = my + (1 - i) * (ch + gap)
        ax.text(mx - 0.005, row_y + ch / 2, approach,
                ha="right", va="center", fontsize=7, fontweight="bold", color=C_DARK)
        for j in range(5):
            cell_x = mx + 0.04 + j * (cw + gap)
            count = counts[i][j]
            if count == 0:
                color = C_GRAY
                label = "—"
            else:
                intensity = min(count / 25, 1.0)
                color = plt.cm.Blues(0.2 + 0.6 * intensity)
                label = str(count)
            draw_rounded_box(ax, cell_x, row_y, cw, ch, label,
                             color=color, fontsize=8, linewidth=0.5, edgecolor="#999999")

    # Totals row
    ax.text(mx - 0.005, my - 0.03, "Total",
            ha="right", va="center", fontsize=7, fontweight="bold", color=C_DARK)
    totals = [9, 12, 19, 25, 11]
    for j, total in enumerate(totals):
        ax.text(mx + 0.04 + j * (cw + gap) + cw / 2, my - 0.03,
                str(total), ha="center", va="center", fontsize=7, fontweight="bold", color=C_BLUE)

    # Representative icons (simplified protein shapes)
    icons_y = p1_y + 0.06
    icon_labels = ["Antibody", "Enzyme", "Binder", "Scaffold", "FP"]
    icon_colors = [C_RED, C_GREEN, C_BLUE, C_PURPLE, C_YELLOW]
    for i, (label, color) in enumerate(zip(icon_labels, icon_colors)):
        ix = mx + 0.02 + i * 0.05
        circle = plt.Circle((ix, icons_y), 0.015, color=color, alpha=0.6, zorder=2)
        ax.add_patch(circle)
        ax.text(ix, icons_y - 0.03, label, ha="center", va="center",
                fontsize=5, color=C_DARK)

    # ── Panel 2: MCP Framework (center) ──────────────────────────────────
    p2_x, p2_y, p2_w, p2_h = 0.34, 0.15, 0.30, 0.80
    draw_panel_bg(ax, p2_x, p2_y, p2_w, p2_h, C_PANEL2, "MCP Tool Framework")

    # LLM Agent box
    agent_x = p2_x + 0.04
    agent_y = p2_y + p2_h - 0.20
    draw_rounded_box(ax, agent_x, agent_y, 0.22, 0.08,
                     "LLM Agent\n(Claude/GPT-5/DeepSeek/Gemini)",
                     C_CYAN, fontsize=7, linewidth=1.5)

    # Arrow down to tools
    draw_arrow(ax, agent_x + 0.11, agent_y, agent_x + 0.11, agent_y - 0.04)

    # Tool grid (17 tools in Docker)
    tool_y = p2_y + 0.30
    ax.text(p2_x + p2_w / 2, tool_y + 0.22, "17 MCP Tools (Docker)",
            ha="center", va="center", fontsize=8, fontweight="bold", color=C_DARK)

    tools = [
        "RFdiffusion", "ProteinMPNN", "ESMFold",
        "AlphaFold2", "Boltz", "PyRosetta",
        "Foldseek", "Interface\nAnalysis", "Design\nBinder",
    ]
    tool_colors = [C_BLUE, C_GREEN, C_CYAN,
                   C_RED, C_PURPLE, C_YELLOW,
                   C_GRAY, C_GREEN, C_BLUE]

    tw, th = 0.065, 0.045
    for i, (tool, tc) in enumerate(zip(tools, tool_colors)):
        row, col = divmod(i, 3)
        tx = p2_x + 0.035 + col * (tw + 0.005)
        ty = tool_y + 0.10 - row * (th + 0.008)
        draw_rounded_box(ax, tx, ty, tw, th, tool, tc,
                         fontsize=5.5, linewidth=0.5, edgecolor="#777777", alpha=0.8)

    # Mode comparison boxes
    mode_y = p2_y + 0.05
    draw_rounded_box(ax, p2_x + 0.02, mode_y, 0.12, 0.12,
                     "Benchmark\nMode\n(16 tools)\n(no hints)",
                     "#FFE0E0", fontsize=6, linewidth=1.0, edgecolor=C_RED)
    draw_rounded_box(ax, p2_x + 0.16, mode_y, 0.12, 0.12,
                     "User\nMode\n(17 tools)\n(+workflows)",
                     "#E0FFE0", fontsize=6, linewidth=1.0, edgecolor=C_GREEN)

    ax.text(p2_x + 0.15, mode_y + 0.06, "vs", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_DARK)

    # ── Panel 3: Evaluation Pipeline (right) ─────────────────────────────
    p3_x, p3_y, p3_w, p3_h = 0.68, 0.15, 0.30, 0.80
    draw_panel_bg(ax, p3_x, p3_y, p3_w, p3_h, C_PANEL3, "Evaluation Pipeline")

    # 6-component rubric as stacked bar
    components = [
        ("Quality", 35, C_GREEN),
        ("Approach", 20, C_BLUE),
        ("Orchestration", 15, C_CYAN),
        ("Feasibility", 15, C_YELLOW),
        ("Diversity", 10, C_PURPLE),
        ("Novelty", 5, C_RED),
    ]

    bar_x = p3_x + 0.05
    bar_w = 0.08
    bar_bottom = p3_y + 0.30
    bar_total_h = 0.40
    cumulative = 0
    for name, pts, color in components:
        h = (pts / 100) * bar_total_h
        rect = mpatches.FancyBboxPatch(
            (bar_x, bar_bottom + cumulative), bar_w, h,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor=C_WHITE, linewidth=1,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(bar_x + bar_w + 0.01, bar_bottom + cumulative + h / 2,
                f"{name} ({pts})", ha="left", va="center",
                fontsize=7, color=C_DARK, zorder=3)
        cumulative += h

    ax.text(bar_x + bar_w / 2, bar_bottom + cumulative + 0.02,
            "100-Point Rubric", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=C_DARK)

    # AF2 verification box
    af2_y = p3_y + 0.13
    draw_rounded_box(ax, p3_x + 0.04, af2_y, 0.22, 0.10,
                     "AF2 Independent\nVerification\n(pLDDT, ipTM, i_pAE)",
                     "#FFFDE0", fontsize=7, linewidth=1.0, edgecolor=C_YELLOW)

    # Quality scoring detail
    qs_y = p3_y + p3_h - 0.22
    ax.text(p3_x + 0.18, qs_y, "3-Tier Quality:", ha="left", va="center",
            fontsize=7, fontweight="bold", color=C_DARK)
    tiers = ["A: Structure (pLDDT, pTM)", "B: Interface (ipTM, i_pAE)", "C: Physics (BSA, ddG)"]
    for i, tier in enumerate(tiers):
        ax.text(p3_x + 0.18, qs_y - 0.035 * (i + 1), tier,
                ha="left", va="center", fontsize=6, color=C_DARK)

    # ── Flow arrows between panels ───────────────────────────────────────
    draw_arrow(ax, p1_x + p1_w, 0.55, p2_x, 0.55, color=C_DARK, lw=2.5)
    draw_arrow(ax, p2_x + p2_w, 0.55, p3_x, 0.55, color=C_DARK, lw=2.5)

    # ── Bottom: 9 Conditions ─────────────────────────────────────────────
    bot_y = 0.02
    ax.text(0.50, bot_y + 0.08, "9 Evaluation Conditions",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_DARK)

    conditions = [
        ("DeepSeek V3", "BM/US"),
        ("GPT-5", "BM/US"),
        ("Sonnet 4.5", "BM/US"),
        ("Gemini 2.5", "BM/US"),
        ("Hardcoded", "Pipeline"),
    ]
    cond_colors = [C_BLUE, C_RED, C_GREEN, C_YELLOW, C_GRAY]
    cw_bot = 0.15
    start_x = 0.10
    for i, ((name, sub), color) in enumerate(zip(conditions, cond_colors)):
        cx = start_x + i * (cw_bot + 0.02)
        draw_rounded_box(ax, cx, bot_y, cw_bot, 0.05,
                         f"{name}\n{sub}", color,
                         fontsize=6, alpha=0.7, linewidth=0.8, edgecolor="#666666")

    # ── Title ────────────────────────────────────────────────────────────
    ax.text(0.50, 0.98, "BioDesignBench: Benchmarking AI Agents for Protein Design",
            ha="center", va="top", fontsize=13, fontweight="bold", color=C_DARK,
            style="italic")

    plt.tight_layout()
    outpath = OUT / "fig1_overview_schematic.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=C_WHITE)
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
