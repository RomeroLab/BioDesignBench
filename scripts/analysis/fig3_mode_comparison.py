#!/usr/bin/env python3
"""BioDesignBench Fig 3: Benchmark vs User Mode — 2-panel (120×80mm).

Left  (~44%): Per-component uplift heatmap (4 LLMs × 6 components + Total)
Right (~56%): Per-task delta distribution (violin + strip)
Bottom:       Horizontal colorbar for heatmap

Output: 120mm x 80mm, 300 dpi.  All text >= 7 pt.
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
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── Nature-style rcParams ────────────────────────────────────────────────
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

# ── Model definitions (same order & palette as Fig 2) ────────────────────
# Order: by User-mode total score descending (same as Fig 2)
MODELS = [
    {"name": "DeepSeek V3", "llm": "DeepSeek V3", "color": "#4477AA"},
    {"name": "GPT-5",       "llm": "GPT-5",       "color": "#EE6677"},
    {"name": "Sonnet 4.5",    "llm": "Sonnet 4.5",  "color": "#228833"},
    {"name": "Gemini 2.5 Pro","llm": "Gemini 2.5 Pro","color": "#66CCEE"},
]

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]
COMP_LABELS = ["Appr (20)", "Orch (15)", "Qual (35)", "Feas (15)", "Nov (5)", "Div (10)"]


def load_paired_deltas():
    """Load per-task deltas (User - Benchmark) for each LLM and component."""
    from scripts.analysis.load_results import load_all
    df = load_all()

    user_df = df[df["mode"] == "user"].set_index(["llm", "task_id"])
    bm_df = df[df["mode"] == "benchmark"].set_index(["llm", "task_id"])

    # Get user-mode total means for sorting (match Fig 2 order)
    user_means = df[df["mode"] == "user"].groupby("llm")["total"].mean()

    rows = []
    for model in MODELS:
        llm = model["llm"]
        for task_id in user_df.loc[llm].index:
            if (llm, task_id) not in bm_df.index:
                continue
            u = user_df.loc[(llm, task_id)]
            b = bm_df.loc[(llm, task_id)]
            row = {"llm": llm, "name": model["name"], "task_id": task_id}
            for comp in COMPONENTS:
                row[f"d_{comp}"] = u[comp] - b[comp]
            row["d_total"] = u["total"] - b["total"]
            rows.append(row)

    return pd.DataFrame(rows), user_means


def main():
    delta_df, user_means = load_paired_deltas()

    # Model order: by user-mode total score descending (matches Fig 2)
    model_colors = {m["name"]: m["color"] for m in MODELS}
    model_llm = {m["name"]: m["llm"] for m in MODELS}
    model_names = [m["name"] for m in MODELS]
    model_order = sorted(
        model_names,
        key=lambda n: user_means.get(model_llm[n], 0),
        reverse=True,
    )

    # For heatmap: reverse so highest score is at top (y=0=bottom, y=N=top)
    hm_order = list(reversed(model_order))

    # ── Figure + nested GridSpec ─────────────────────────────────────────
    fig_w = 120 / 25.4
    fig_h = 80 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(2, 1, height_ratios=[0.91, 0.09], hspace=0.22)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[0.52, 0.48], wspace=0.38)
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[0.52, 0.48], wspace=0.38)

    ax_hm = fig.add_subplot(gs_top[0, 0])
    ax_vp = fig.add_subplot(gs_top[0, 1])
    ax_cb = fig.add_subplot(gs_bot[0, 0])
    # gs_bot[0,1] left empty

    # ── Heatmap: Per-component uplift ────────────────────────────────────
    n_models = len(hm_order)
    n_cols = len(COMPONENTS) + 1  # +1 for Total

    # Build data matrix (row 0 = top model)
    hm_data = np.zeros((n_models, n_cols))
    for i, name in enumerate(hm_order):
        sub = delta_df[delta_df["name"] == name]
        for j, comp in enumerate(COMPONENTS):
            hm_data[i, j] = sub[f"d_{comp}"].mean()
        hm_data[i, -1] = sub["d_total"].mean()

    # Same blue-to-red colormap as Fig 2, centered at 0
    _CMAP_COLORS = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    cmap = mcolors.LinearSegmentedColormap.from_list("score_br", _CMAP_COLORS, N=256)
    vmax = np.max(np.abs(hm_data)) * 1.1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Draw cells
    for i in range(n_models):
        for j in range(n_cols):
            is_total = (j == n_cols - 1)
            val = hm_data[i, j]
            cell_color = cmap(norm(val))
            rect = plt.Rectangle(
                (j, i - 0.5), 1, 1,
                facecolor=cell_color, edgecolor="white",
                linewidth=0.8 if is_total else 0.5,
            )
            ax_hm.add_patch(rect)
            lum = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            txt_col = "white" if lum < 0.5 else "#222222"
            sign = "+" if val > 0 else ""
            ax_hm.text(
                j + 0.5, i, f"{sign}{val:.1f}",
                ha="center", va="center", fontsize=_MIN_PT,
                fontweight="bold" if is_total else "normal",
                color=txt_col,
            )

    # Total column separator
    ax_hm.axvline(n_cols - 1, color="#888888", lw=0.8, zorder=3)

    ax_hm.set_xlim(0, n_cols)
    ax_hm.set_ylim(-0.5, n_models - 0.5)

    # Y-axis: model names — tilted 25° with model color (matches Fig 2)
    ax_hm.set_yticks(range(n_models))
    ax_hm.set_yticklabels(
        hm_order, fontsize=_MIN_PT, rotation=45,
        rotation_mode="anchor", ha="right", va="center",
    )
    for i, tl in enumerate(ax_hm.get_yticklabels()):
        tl.set_color(model_colors[hm_order[i]])
        tl.set_fontweight("bold")
    ax_hm.tick_params(axis="y", length=0)

    # X-axis: component names — rotated 45°, flush (matches Fig 2)
    col_labels = COMP_LABELS + ["Total"]
    ax_hm.set_xticks([j + 0.5 for j in range(n_cols)])
    ax_hm.set_xticklabels(
        col_labels, fontsize=_MIN_PT, ha="left",
        rotation=45, rotation_mode="anchor",
    )
    ax_hm.tick_params(axis="x", length=0, pad=1)
    ax_hm.xaxis.set_ticks_position("top")

    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    # ── Colorbar (horizontal, below heatmap) ─────────────────────────────
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Δ score (User − BM)", fontsize=_MIN_PT, labelpad=2)
    cb.ax.tick_params(labelsize=_MIN_PT, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)

    # ── Violin + strip: Per-task delta distribution ──────────────────────
    positions = np.arange(n_models)
    vp_width = 0.6

    # First pass: collect data + compute annotation positions
    all_deltas = {}
    for i, name in enumerate(model_order):
        task_deltas = delta_df[delta_df["name"] == name]["d_total"].values
        all_deltas[name] = task_deltas

        color = model_colors[name]

        # Violin
        parts = ax_vp.violinplot(
            task_deltas, positions=[i], vert=True,
            widths=vp_width, showextrema=False, showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.5)

        # Box (IQR)
        q1, med, q3 = np.percentile(task_deltas, [25, 50, 75])
        ax_vp.vlines(i, q1, q3, color=color, lw=2.5, zorder=3)
        ax_vp.scatter([i], [med], color=color, s=18, zorder=4,
                      edgecolors="white", linewidths=0.5)

        # Strip (jittered points)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(task_deltas))
        ax_vp.scatter(
            i + jitter, task_deltas,
            s=4, alpha=0.25, color=color, zorder=2, linewidths=0,
        )

    # Zero line
    ax_vp.axhline(0, color="#888888", lw=0.6, ls="--", zorder=1)

    # Set ylim, then add annotations at consistent y
    ymin, ymax = ax_vp.get_ylim()
    ax_vp.set_ylim(ymin, ymax + 8)
    annot_y = ymax + 1
    for i, name in enumerate(model_order):
        color = model_colors[name]
        td = all_deltas[name]
        n_pos = np.sum(td > 0)
        n_neg = np.sum(td < 0)
        ax_vp.text(
            i, annot_y, f"{n_pos}+ / {n_neg}−",
            ha="left", va="bottom", fontsize=_MIN_PT, color=color,
            fontweight="bold", rotation=45, rotation_mode="anchor",
        )

    # X-axis: model names — tilted 25° with color (matches Fig 2)
    ax_vp.set_xticks(positions)
    ax_vp.set_xticklabels(
        model_order, fontsize=_MIN_PT, rotation=45,
        rotation_mode="anchor", ha="right",
    )
    for i, tl in enumerate(ax_vp.get_xticklabels()):
        tl.set_color(model_colors[model_order[i]])
        tl.set_fontweight("bold")

    ax_vp.set_ylabel("Score Δ (User − BM)", fontsize=_MIN_PT)
    ax_vp.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_vp.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_vp.spines["top"].set_visible(False)
    ax_vp.spines["right"].set_visible(False)
    ax_vp.tick_params(axis="x", length=0)

    # ── Save ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.84)

    png_path = OUT / "fig3_mode_comparison.png"
    pdf_path = OUT / "fig3_mode_comparison.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Size: {fig_w:.2f}\" x {fig_h:.2f}\" ({fig_w*25.4:.0f}mm x {fig_h*25.4:.0f}mm)")


if __name__ == "__main__":
    main()
