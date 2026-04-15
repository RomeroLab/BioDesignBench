#!/usr/bin/env python3
"""SI Figure 1: Component Variance Analysis.

(a) 6x6 Pearson correlation heatmap of scoring components, reordered by
    hierarchical clustering (Ward linkage).
(b) PCA scree plot showing explained variance per principal component,
    with cumulative overlay.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _luminance(hex_color: str) -> float:
    """Compute relative luminance (0-1) for a hex color string."""
    rgb = mcolors.hex2color(hex_color)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def panel_a(ax) -> None:
    """6x6 component correlation matrix, hierarchically clustered."""
    df = load_all()
    df = df[df["condition"] != "Oracle"]  # Oracle distorts variance analysis
    comp_df = df[COMPONENTS].dropna()

    # Pearson correlation
    corr = comp_df.corr(method="pearson")

    # Hierarchical clustering to reorder axes
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # ensure symmetry
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    labels = [COMP_LABELS[COMPONENTS[i]] for i in order]
    reordered = corr.values[np.ix_(order, order)]

    # Diverging colormap — pre-compute RGBA for Illustrator PDF compatibility
    # (avoids indexed-color-image issues in Adobe Illustrator)
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    rgba_data = cmap(norm(reordered))  # (n, n, 4) RGBA array
    im = ax.imshow(rgba_data, aspect="equal", rasterized=True)

    # Invisible ScalarMappable for colorbar (since imshow got raw RGBA)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=MIN_PT)
    ax.set_yticklabels(labels, fontsize=MIN_PT)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = reordered[i, j]
            lum = 0.2126 * rgba_data[i, j, 0] + 0.7152 * rgba_data[i, j, 1] + 0.0722 * rgba_data[i, j, 2]
            text_color = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=MIN_PT - 1, color=text_color)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(top=False, bottom=True, left=True, right=False)

    # Colorbar (horizontal, below heatmap)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.65)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Pearson r", fontsize=MIN_PT)
    cbar.ax.tick_params(labelsize=MIN_PT - 1, width=0.3, length=2)



def panel_b(ax) -> None:
    """PCA scree plot with cumulative explained variance overlay."""
    df = load_all()
    df = df[df["condition"] != "Oracle"]
    comp_df = df[COMPONENTS].dropna()

    # Standardize before PCA
    X = StandardScaler().fit_transform(comp_df.values)
    pca = PCA(n_components=6, random_state=42)
    pca.fit(X)

    var_ratio = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(var_ratio)
    pcs = np.arange(1, 7)

    # Sequential blue gradient for bars
    blues = plt.cm.Blues(np.linspace(0.4, 0.9, 6))

    bars = ax.bar(pcs, var_ratio, color=blues, edgecolor="white", linewidth=0.3,
                  zorder=3, width=0.65)

    # Cumulative line
    ax.plot(pcs, cumulative, color="#333333", marker="D", markersize=3,
            linewidth=1.0, zorder=4)

    # Annotate PC1
    ax.annotate(
        f"PC1: {var_ratio[0]:.1f}%",
        xy=(1, var_ratio[0]),
        xytext=(2.0, var_ratio[0] + 5),
        fontsize=MIN_PT, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5),
        ha="center",
    )

    # Y-axis label on cumulative
    ax.set_ylabel("Variance explained (%)", fontsize=MIN_PT)
    ax.set_xlabel("Principal component", fontsize=MIN_PT)
    ax.set_xticks(pcs)
    ax.set_xticklabels([f"PC{i}" for i in pcs], fontsize=MIN_PT)
    ax.set_ylim(0, 105)

    style_grid(ax, axis="y")


def main() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [0.48, 0.52], "wspace": 0.55},
    )

    panel_a(axes[0])
    panel_b(axes[1])

    save_fig(fig, "si_fig1_component_variance")


if __name__ == "__main__":
    main()
