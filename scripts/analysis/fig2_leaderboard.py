#!/usr/bin/env python3
"""BioDesignBench Fig 2: Main Leaderboard — bar chart + side heatmap.

2-panel layout:
  Left  (~48%): Total score horizontal bars (11 conditions)
  Right (~52%): Component score heatmap (blue-to-red), horizontal colorbar below

Includes: Oracle, Human Expert, 4 LLMs (user/BM pairs), Hardcoded Pipeline.
Output: 120mm x 100mm, 300 dpi.  All text >= 7 pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── Nature-style rcParams — minimum 7 pt everywhere ─────────────────────
_MIN_PT = 7
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": _MIN_PT,
    "axes.linewidth": 0.5,
    "axes.labelsize": _MIN_PT,
    "axes.titlesize": 8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.labelsize": _MIN_PT,
    "ytick.labelsize": _MIN_PT,
    "legend.fontsize": _MIN_PT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Shared blue-to-red colormap (heatmap) ────────────────────────────────
_CMAP_COLORS = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
SCORE_CMAP = mcolors.LinearSegmentedColormap.from_list("score_br", _CMAP_COLORS, N=256)


def load_data():
    """Load per-condition component means from results."""
    from scripts.analysis.load_results import load_all
    df = load_all()
    components = ["approach", "orchestration", "quality",
                  "feasibility", "novelty", "diversity"]
    means = df.groupby("condition", observed=True)[components + ["total"]].mean()
    return means, components


# ── Model definitions with per-model colors ──────────────────────────────
MODELS = [
    {"name": "DeepSeek V3", "user": "DeepSeek V3 user",    "bm": "DeepSeek V3 benchmark", "color": "#4477AA"},
    {"name": "GPT-5",       "user": "GPT-5 user",          "bm": "GPT-5 benchmark",       "color": "#EE6677"},
    {"name": "Sonnet 4.5",  "user": "Sonnet 4.5 user",     "bm": "Sonnet 4.5 benchmark",  "color": "#228833"},
    {"name": "Gemini 2.5 Pro","user": "Gemini 2.5 Pro user","bm": "Gemini 2.5 Pro benchmark","color": "#66CCEE"},
]

# Baselines/references (non-LLM entries) — Paul Tol palette, no overlap with models
BASELINE_COLORS = {
    "Oracle":             "#DDAA33",
    "Human Expert":       "#AA3377",
    "Hardcoded Pipeline": "#EE7733",
}

COMP_META = {
    "approach":      ("Appr", 20),
    "orchestration": ("Orch", 15),
    "quality":       ("Qual", 35),
    "feasibility":   ("Feas", 15),
    "novelty":       ("Nov",   5),
    "diversity":     ("Div",  10),
}


def main():
    means, components = load_data()

    # Sort models by user score descending
    MODELS.sort(key=lambda m: means.loc[m["user"], "total"], reverse=True)

    # Build row order (bottom-up for barh; reversed so top of figure = highest)
    # Layout: Oracle > Human Expert > Hardcoded > [LLM pairs sorted by user]
    rows = []  # (label, condition_key, row_type, color)

    # LLM pairs at bottom (reversed for barh: lowest score first)
    for model in reversed(MODELS):
        rows.append((f"{model['name']} BM",   model["bm"],   "bm",   model["color"]))
        rows.append((f"{model['name']} User",  model["user"], "user", model["color"]))

    # Baselines at top (comparison targets)
    rows.append(("Hardcoded Pipeline", "Hardcoded Pipeline", "baseline_top", BASELINE_COLORS["Hardcoded Pipeline"]))
    rows.append(("Human Expert", "Human Expert", "baseline_top", BASELINE_COLORS["Human Expert"]))
    rows.append(("Oracle", "Oracle", "baseline_top", BASELINE_COLORS["Oracle"]))

    n_rows = len(rows)
    y_pos = np.arange(n_rows)

    # ── Figure + GridSpec ────────────────────────────────────────────────
    fig_w = 120 / 25.4   # 4.724 in
    fig_h = 80 / 25.4    # 3.15 in  (120×80 mm, 3:2 ratio)
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(3, 1, height_ratios=[0.84, 0.08, 0.08], hspace=0.20)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[0.46, 0.54], wspace=0.02)
    ax_bar = fig.add_subplot(gs_top[0, 0])
    ax_hm  = fig.add_subplot(gs_top[0, 1])
    ax_leg = fig.add_subplot(gs[1])
    ax_cb  = fig.add_subplot(gs[2])

    # ── Bar chart ──────────────────────────────────────────────────────
    bar_h = 0.7
    for i, (label, cond, rtype, mcolor) in enumerate(rows):
        total = means.loc[cond, "total"]
        if rtype == "bm":
            alpha = 0.40
        else:
            alpha = 1.0

        # Hatch for baselines
        hatch = "//" if rtype == "baseline_top" else None

        ax_bar.barh(
            y_pos[i], total, height=bar_h,
            color=mcolor, alpha=alpha, hatch=hatch,
            edgecolor="white", linewidth=0.3, zorder=2,
        )
        # Score annotation
        fw = "bold" if rtype in ("user", "baseline_top") else "normal"
        ax_bar.text(
            total + 0.8, y_pos[i], f"{total:.1f}",
            va="center", ha="left", fontsize=_MIN_PT,
            color="#222222", fontweight=fw,
        )

    # Y-axis labels
    ylabels = [r[0] for r in rows]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ylabels, fontsize=_MIN_PT, rotation=25,
                           rotation_mode="anchor", ha="right", va="center")
    for i, tl in enumerate(ax_bar.get_yticklabels()):
        tl.set_color(rows[i][3])  # color
        rtype = rows[i][2]
        if rtype in ("user", "baseline_top"):
            tl.set_fontweight("bold")
        elif rtype == "bm":
            tl.set_fontstyle("italic")
            tl.set_alpha(0.6)

    # Group separators
    n_llm_rows = len(MODELS) * 2
    # Between each LLM model pair
    for j in range(1, len(MODELS)):
        ax_bar.axhline(j * 2 - 0.5, color="#cccccc", lw=0.4, zorder=1)
    # Between LLMs and baselines (thicker)
    ax_bar.axhline(n_llm_rows - 0.5, color="#aaaaaa", lw=0.6, zorder=1)

    # Axes styling
    ax_bar.set_xlim(0, 100)
    ax_bar.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_bar.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_bar.grid(axis="x", which="major", color="#e0e0e0", lw=0.3, zorder=0)
    ax_bar.set_ylim(-0.5, n_rows - 0.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="y", length=0)

    # ── Component heatmap (blue-to-red colormap) ───────────────────────
    n_comp = len(components)
    data = np.zeros((n_rows, n_comp))
    for i, (_, cond, _, _) in enumerate(rows):
        for j, comp in enumerate(components):
            data[i, j] = means.loc[cond, comp]

    # Normalize each column to [0, 1] by its max possible points
    max_pts = np.array([COMP_META[c][1] for c in components], dtype=float)
    normed = data / max_pts[np.newaxis, :]

    # Draw cells
    for i in range(n_rows):
        for j in range(n_comp):
            frac = np.clip(normed[i, j], 0, 1)
            cell_color = SCORE_CMAP(frac)
            rect = plt.Rectangle(
                (j, i - 0.5), 1, 1,
                facecolor=cell_color, edgecolor="white", linewidth=0.5,
            )
            ax_hm.add_patch(rect)
            lum = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            txt_col = "white" if lum < 0.55 else "#222222"
            ax_hm.text(
                j + 0.5, i, f"{data[i, j]:.1f}",
                ha="center", va="center", fontsize=_MIN_PT,
                color=txt_col,
            )

    ax_hm.set_xlim(0, n_comp)
    ax_hm.set_ylim(-0.5, n_rows - 0.5)

    # Column headers
    col_labels = [f"{COMP_META[c][0]} ({COMP_META[c][1]})" for c in components]
    ax_hm.set_xticks([j + 0.5 for j in range(n_comp)])
    ax_hm.set_xticklabels(
        col_labels, fontsize=6, ha="left", rotation=45, rotation_mode="anchor",
    )
    ax_hm.tick_params(axis="x", length=0, pad=1)
    ax_hm.xaxis.set_ticks_position("top")

    ax_hm.set_yticks([])
    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    # Group separators (match bar chart)
    n_llm_rows = len(MODELS) * 2
    for j in range(1, len(MODELS)):
        ax_hm.axhline(j * 2 - 0.5, color="#cccccc", lw=0.4)
    ax_hm.axhline(n_llm_rows - 0.5, color="#aaaaaa", lw=0.6)

    # ── Legend (separate row) ────────────────────────────────────────────
    from matplotlib.patches import Patch
    ax_leg.axis("off")
    legend_handles = [
        Patch(facecolor="#444444", edgecolor="white", alpha=1.0, label="User mode"),
        Patch(facecolor="#444444", edgecolor="white", alpha=0.40, label="Benchmark mode"),
        Patch(facecolor="#444444", edgecolor="white", alpha=0.7,
              hatch="//////", label="Baselines"),
    ]
    ax_leg.legend(
        handles=legend_handles, loc="center", ncol=3,
        fontsize=_MIN_PT, frameon=False,
        handlelength=1.8, columnspacing=1.5, handletextpad=0.8,
    )

    # ── Colorbar (horizontal, bottom row) ────────────────────────────────
    sm = mcm.ScalarMappable(cmap=SCORE_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.ax.tick_params(labelsize=_MIN_PT, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("% of max", fontsize=_MIN_PT, labelpad=2)

    # ── Save ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.05, top=0.84)

    # Center legend & colorbar on full figure width (after subplots_adjust)
    for ax_bottom in [ax_leg, ax_cb]:
        p = ax_bottom.get_position()
        center_w = 0.56
        ax_bottom.set_position([0.5 - center_w / 2, p.y0, center_w, p.height])

    png_path = OUT / "fig2_leaderboard.png"
    pdf_path = OUT / "fig2_leaderboard.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Size: {fig_w:.2f}\" x {fig_h:.2f}\" ({fig_w*25.4:.0f}mm x {fig_h*25.4:.0f}mm)")


if __name__ == "__main__":
    main()
