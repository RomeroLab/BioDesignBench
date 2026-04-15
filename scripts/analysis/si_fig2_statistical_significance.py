#!/usr/bin/env python3
"""SI Figure 2: Statistical Significance Analysis.

(a) Forest plot -- mean total score with 95% bootstrap CI per condition.
(b) Pairwise significance heatmap (paired permutation test, Bonferroni).
(c) Rank stability -- bootstrap distribution of ranks across conditions.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

import numpy as np
from matplotlib.gridspec import GridSpec


def _bootstrap_ci(
    scores: np.ndarray, n_boot: int = 10_000, ci: float = 0.95, rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) via bootstrap resampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(scores)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = scores[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(scores.mean()), float(lo), float(hi)


def _paired_permutation_pvalue(
    x: np.ndarray, y: np.ndarray, n_perm: int = 10_000, rng: np.random.Generator | None = None,
) -> float:
    """Two-sided paired permutation test p-value."""
    if rng is None:
        rng = np.random.default_rng(42)
    diff = x - y
    obs = np.abs(diff.mean())
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        if np.abs((diff * signs).mean()) >= obs:
            count += 1
    return count / n_perm


def panel_a(ax) -> None:
    """Forest plot: mean total score + 95% bootstrap CI."""
    score_mat = load_score_matrix()
    rng = np.random.default_rng(42)

    # Compute stats per condition
    stats = []
    for cond in CONDITION_ORDER:
        if cond not in score_mat.columns:
            continue
        vals = score_mat[cond].dropna().values
        mean, lo, hi = _bootstrap_ci(vals, rng=rng)
        stats.append((cond, mean, lo, hi))

    # Sort by mean (descending) -- highest at top
    stats.sort(key=lambda x: x[1], reverse=True)

    y_positions = np.arange(len(stats))
    for i, (cond, mean, lo, hi) in enumerate(stats):
        mkw = cond_marker_kw(cond)
        color = CONDITION_COLORS.get(cond, "#888888")
        ax.errorbar(
            mean, i, xerr=[[mean - lo], [hi - mean]],
            fmt="none", ecolor=color, elinewidth=0.8, capsize=2, capthick=0.6,
            zorder=3,
        )
        ax.scatter(mean, i, s=25, zorder=4, **mkw)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [cond for cond, _, _, _ in stats],
        fontsize=MIN_PT - 1,
        rotation=30, rotation_mode="anchor", ha="right", va="center",
    )
    ax.set_xlabel("Mean total score", fontsize=MIN_PT)
    ax.invert_yaxis()
    style_grid(ax, axis="x")


def panel_b(ax) -> None:
    """Pairwise significance heatmap with permutation tests."""
    score_mat = load_score_matrix()
    rng = np.random.default_rng(42)

    conds = [c for c in CONDITION_ORDER if c in score_mat.columns]
    n = len(conds)
    n_comparisons = n * (n - 1) // 2

    # Compute pairwise p-values (upper triangle)
    pvals = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Align on shared tasks
            shared = score_mat[[conds[i], conds[j]]].dropna()
            x = shared[conds[i]].values
            y = shared[conds[j]].values
            raw_p = _paired_permutation_pvalue(x, y, rng=rng)
            pvals[i, j] = min(raw_p * n_comparisons, 1.0)  # Bonferroni
            pvals[j, i] = pvals[i, j]

    # -log10(p) for display, capped
    neg_log_p = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            p = max(pvals[i, j], 1e-10)  # avoid log(0)
            neg_log_p[i, j] = -np.log10(p)

    # Plot upper triangle as heatmap
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    display = np.where(mask, neg_log_p, np.nan)

    # Custom colormap matching SI Fig 1 style
    _SIG_COLORS = ["#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    sig_cmap = mcolors.LinearSegmentedColormap.from_list("sig", _SIG_COLORS, N=256)
    im = ax.imshow(display, cmap=sig_cmap, vmin=0, vmax=4, aspect="equal")

    # Diagonal gray
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="#dddddd",
                                   edgecolor="white", linewidth=0.3))

    # Annotate all cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", fontsize=MIN_PT - 2, color="#888888")
            elif j > i:
                # Upper triangle: significance stars
                p = pvals[i, j]
                if p < 0.001:
                    label = "***"
                elif p < 0.01:
                    label = "**"
                elif p < 0.05:
                    label = "*"
                else:
                    label = "ns"
                val = neg_log_p[i, j]
                rgba = sig_cmap(min(val / 4, 1.0)) if not np.isnan(val) else (1, 1, 1, 1)
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                tc = "white" if lum < 0.5 else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=MIN_PT - 2, color=tc)
            else:
                # Lower triangle: just significance stars (no color)
                p = pvals[i, j]
                if p < 0.001:
                    label = "***"
                elif p < 0.01:
                    label = "**"
                elif p < 0.05:
                    label = "*"
                else:
                    label = "ns"
                ax.text(j, i, label, ha="center", va="center", fontsize=MIN_PT - 2,
                        color="#555555")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(conds, rotation=55, ha="right",
                       fontsize=MIN_PT - 2)
    ax.set_yticklabels(conds, fontsize=MIN_PT - 2,
                       rotation=30, rotation_mode="anchor",
                       ha="right", va="center")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=True, left=True, right=False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=1.2)
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("$-\\log_{10}(p)$", fontsize=MIN_PT)
    cbar.ax.tick_params(labelsize=MIN_PT - 1, width=0.3, length=2)



def panel_c(ax) -> None:
    """Rank stability: bootstrap distribution of condition ranks."""
    score_mat = load_score_matrix()
    rng = np.random.default_rng(42)

    conds = [c for c in CONDITION_ORDER if c in score_mat.columns]
    n_conds = len(conds)
    task_ids = score_mat.index.tolist()
    n_tasks = len(task_ids)
    n_boot = 10_000

    # Matrix of scores: tasks x conditions
    mat = score_mat[conds].values  # (n_tasks, n_conds)

    # Bootstrap rank counts: rank_counts[cond_idx, rank] = count
    rank_counts = np.zeros((n_conds, n_conds), dtype=int)

    for _ in range(n_boot):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        boot_means = np.nanmean(mat[idx, :], axis=0)
        # Rank: 1=best (highest mean)
        ranks = n_conds - np.argsort(np.argsort(boot_means))  # rank from 1
        for ci in range(n_conds):
            rank_counts[ci, ranks[ci] - 1] += 1

    # Frequency (%)
    rank_freq = rank_counts / n_boot * 100

    # Sort conditions by median rank (best first)
    median_ranks = np.array([
        np.average(np.arange(1, n_conds + 1), weights=rank_freq[ci]) for ci in range(n_conds)
    ])
    sorted_idx = np.argsort(median_ranks)

    y_positions = np.arange(n_conds)
    bar_height = 0.7

    # Stacked horizontal bars
    rank_colors = plt.cm.Blues(np.linspace(0.2, 0.9, n_conds))

    for i, ci in enumerate(sorted_idx):
        left = 0.0
        for r in range(n_conds):
            width = rank_freq[ci, r]
            ax.barh(i, width, left=left, height=bar_height,
                    color=rank_colors[r], edgecolor="white", linewidth=0.2)
            if width > 8:  # Only label if wide enough
                ax.text(left + width / 2, i, f"{r+1}", ha="center", va="center",
                        fontsize=MIN_PT - 2, color="white" if r >= n_conds // 2 else "black")
            left += width

    ax.set_yticks(y_positions)
    ax.set_yticklabels([conds[ci] for ci in sorted_idx], fontsize=MIN_PT,
                       rotation=30, rotation_mode="anchor", ha="right", va="center")
    ax.set_xlabel("Frequency (%)", fontsize=MIN_PT, labelpad=2)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    style_grid(ax, axis="x")

    # Return rank_colors and n_conds for external legend
    ax._rank_colors = rank_colors
    ax._n_conds = n_conds


def main() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [0.44, 0.56], "wspace": 0.55},
    )

    panel_a(axes[0])
    panel_b(axes[1])

    save_fig(fig, "si_fig2_statistical_significance")


if __name__ == "__main__":
    main()
