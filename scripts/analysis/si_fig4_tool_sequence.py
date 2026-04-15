#!/usr/bin/env python3
"""SI Figure 4: Tool Usage Pattern Analysis.

Left:  Avg tool calls per task (volume)
Right: Avg unique tools per task (repertoire breadth)

Reveals how agents differ in tool call volume vs diversity of tool usage.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

import numpy as np


def main() -> None:
    df = load_all()

    avg_lens: list[float] = []
    avg_unique: list[float] = []
    face_colors: list[str] = []
    edge_colors: list[str] = []

    for cond in CONDITION_ORDER_NO_ORACLE:
        cond_df = df[df["condition"] == cond]
        seqs = [s for s in cond_df["tool_sequence"]
                if isinstance(s, list) and len(s) > 0]

        if not seqs:
            avg_lens.append(0.0)
            avg_unique.append(0.0)
        else:
            avg_lens.append(np.mean([len(s) for s in seqs]))
            avg_unique.append(np.mean([len(set(s)) for s in seqs]))

        c = CONDITION_COLORS.get(cond, "#888888")
        if is_bm(cond) or "Hardcoded" in cond:
            face_colors.append("white")
            edge_colors.append(c)
        else:
            face_colors.append(c)
            edge_colors.append(c)

    y_pos = np.arange(len(CONDITION_ORDER_NO_ORACLE))
    bar_h = 0.38

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [0.55, 0.45], "wspace": 0.15},
    )

    # ── Left panel: Avg tool calls per task ──────────────────────────
    ax_left.barh(y_pos, avg_lens, height=bar_h * 1.5,
                 color=face_colors, edgecolor=edge_colors,
                 linewidth=0.8, zorder=3)

    for i, val in enumerate(avg_lens):
        if val > 0:
            ax_left.text(val + 0.3, y_pos[i], f"{val:.1f}",
                         va="center", ha="left",
                         fontsize=MIN_PT - 1, color="#333333")

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(CONDITION_ORDER_NO_ORACLE, fontsize=MIN_PT - 2,
                            rotation=30, rotation_mode="anchor",
                            ha="right", va="center")
    ax_left.set_xlabel("Avg tool calls / task", fontsize=MIN_PT)
    ax_left.invert_yaxis()
    style_grid(ax_left, axis="x")

    # ── Right panel: Avg unique tools per task ────────────────────────
    ax_right.barh(y_pos, avg_unique, height=bar_h * 1.5,
                  color=face_colors, edgecolor=edge_colors,
                  linewidth=0.8, zorder=3)

    for i, val in enumerate(avg_unique):
        if val > 0:
            ax_right.text(val + 0.1, y_pos[i], f"{val:.1f}",
                          va="center", ha="left",
                          fontsize=MIN_PT - 1, color="#555555")

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([], fontsize=MIN_PT - 2)
    ax_right.set_xlabel("Avg unique tools / task", fontsize=MIN_PT)
    ax_right.invert_yaxis()
    style_grid(ax_right, axis="x")

    save_fig(fig, "si_fig4_tool_sequence")


if __name__ == "__main__":
    main()
