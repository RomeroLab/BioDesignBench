#!/usr/bin/env python3
"""SI Figure 7: Cross-Model Agreement Analysis.

(a) 9x9 Spearman rank correlation heatmap between conditions,
    reordered by hierarchical clustering (Ward linkage on 1-rho distance).
(b) Agent dendrogram from the same clustering, colored by LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def _compute_spearman_matrix(score_mat: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise Spearman rho for all condition pairs.

    Returns:
        (rho_matrix, condition_names)
    """
    conds = [c for c in CONDITION_ORDER_NO_ORACLE if c in score_mat.columns]
    n = len(conds)
    rho_mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            # Align on shared tasks (drop rows with NaN in either)
            shared = score_mat[[conds[i], conds[j]]].dropna()
            if len(shared) < 3:
                rho_mat[i, j] = rho_mat[j, i] = 0.0
                continue
            rho, _ = spearmanr(shared[conds[i]], shared[conds[j]])
            rho_mat[i, j] = rho
            rho_mat[j, i] = rho

    return rho_mat, conds


def _cluster_order(rho_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical clustering on 1 - rho distance. Returns (linkage_Z, leaf_order)."""
    dist = 1 - rho_mat
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # ensure symmetry
    # Clip to avoid negative distances from floating-point
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)
    return Z, order


def panel_a(ax, rho_mat: np.ndarray, conds: list[str], order: np.ndarray) -> None:
    """9x9 Spearman rank correlation heatmap, hierarchically clustered."""
    n = len(conds)
    reordered = rho_mat[np.ix_(order, order)]
    labels = [CONDITION_SHORT.get(conds[i], conds[i]) for i in order]

    # Diverging colormap: blue (neg) → #e0f3f8 (center) → warm (pos), matching SI palette
    _DIV_COLORS = ["#2166ac", "#4393c3", "#92c5de", "#e0f3f8",
                   "#fee090", "#fc8d59", "#d73027"]
    corr_cmap = mcolors.LinearSegmentedColormap.from_list("corr_div", _DIV_COLORS, N=256)
    im = ax.imshow(reordered, cmap=corr_cmap, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=max(MIN_PT - 2, 4))
    ax.set_yticklabels(labels, fontsize=max(MIN_PT - 2, 4))

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = reordered[i, j]
            rgba = corr_cmap((val + 1) / 2)
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(MIN_PT - 3, 4), color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=True, left=True, right=False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.8)
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman $\\rho$", fontsize=MIN_PT - 1, labelpad=2)
    cbar.ax.tick_params(labelsize=max(MIN_PT - 2, 4), width=0.3, length=2)

    # Store im reference for external use
    ax._im = im
    ax._corr_cmap = corr_cmap



def panel_b(ax, Z: np.ndarray, conds: list[str]) -> None:
    """Dendrogram colored by LLM, horizontal orientation."""
    short_labels = [CONDITION_SHORT.get(c, c) for c in conds]

    # Map each leaf to its LLM color
    llm_for_cond = {}
    for c in conds:
        for llm_name, color in LLM_COLORS.items():
            if c.startswith(llm_name) or llm_name in c:
                llm_for_cond[c] = color
                break
        else:
            llm_for_cond[c] = "#888888"

    # Plot dendrogram (horizontal) with labels
    dendro = dendrogram(
        Z,
        labels=short_labels,
        orientation="right",
        ax=ax,
        leaf_font_size=max(MIN_PT - 2, 4),
        above_threshold_color="#aaaaaa",
    )

    # Get leaf order from dendrogram
    leaf_order = dendro["leaves"]

    # Color the y-tick labels by LLM
    for tick_label, leaf_idx in zip(ax.get_yticklabels(), leaf_order):
        tick_label.set_color(llm_for_cond[conds[leaf_idx]])
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(max(MIN_PT - 2, 4))

    ax.set_xlabel("Ward distance ($1 - \\rho$)", fontsize=MIN_PT)

    # Color branches by LLM where possible
    # Dendrogram icoord/dcoord: each link has color
    # Re-color links connecting two leaves of the same LLM
    x_coords = dendro["dcoord"]
    y_coords = dendro["icoord"]
    colors = dendro["color_list"]

    # Build leaf position -> LLM color map
    leaf_positions = {5 + 10 * i: llm_for_cond[conds[leaf_order[i]]] for i in range(len(leaf_order))}

    def _get_branch_color(yc: list[float]) -> str | None:
        """If both endpoints are leaves of same LLM, return that color."""
        leaf_colors = set()
        for y in [yc[0], yc[3]]:
            if y in leaf_positions:
                leaf_colors.add(leaf_positions[y])
        if len(leaf_colors) == 1:
            return leaf_colors.pop()
        return None

    for i, (xc, yc) in enumerate(zip(x_coords, y_coords)):
        bc = _get_branch_color(yc)
        if bc:
            # Redraw this link with LLM color
            ax.plot([xc[0], xc[1]], [yc[0], yc[1]], color=bc, linewidth=1.0)
            ax.plot([xc[1], xc[2]], [yc[1], yc[2]], color=bc, linewidth=1.0)
            ax.plot([xc[2], xc[3]], [yc[2], yc[3]], color=bc, linewidth=1.0)

    style_grid(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(left=False)




def main() -> None:
    score_mat = load_score_matrix()
    rho_mat, conds = _compute_spearman_matrix(score_mat)
    Z, order = _cluster_order(rho_mat)

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [0.45, 0.55], "wspace": 0.55},
    )

    panel_a(axes[0], rho_mat, conds, order)
    panel_b(axes[1], Z, conds)

    save_fig(fig, "si_fig7_cross_model_agreement")


if __name__ == "__main__":
    main()
