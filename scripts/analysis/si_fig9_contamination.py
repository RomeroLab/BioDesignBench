#!/usr/bin/env python3
"""SI Figure 9: Contamination Analysis.

(a) Contamination score distribution -- overlapping semi-transparent
    histograms (or KDE curves) for each of the 9 conditions.  A vertical
    dashed red line marks the 0.5 threshold.

(b) Leaderboard before vs after zeroing -- paired horizontal bar chart
    showing the original mean total score and the mean total score after
    zeroing tasks with contamination_score > 0.5.  Sorted by original
    score descending, with delta annotations.

Layout: 1 row, 2 columns (width_ratios=[0.50, 0.50], wspace=0.35).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

CONTAMINATION_THRESHOLD = 0.5


def panel_a(ax) -> None:
    """Contamination score distribution per condition (overlapping KDE / histogram)."""
    df = load_all()

    # Ensure contamination_score exists and has valid values
    if "contamination_score" not in df.columns:
        ax.text(0.5, 0.5, "No contamination_score column",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=MIN_PT, color="grey")
        return

    scores_all = df["contamination_score"].dropna()
    if len(scores_all) == 0:
        ax.text(0.5, 0.5, "No contamination data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=MIN_PT, color="grey")
        return

    bins = np.linspace(0, 1, 21)

    # Determine whether to use histogram (enough spread) or annotate low-variance
    unique_vals = scores_all.nunique()
    all_zero = (scores_all == 0).all()

    for cond in CONDITION_ORDER_NO_ORACLE:
        cond_scores = df.loc[df["condition"] == cond, "contamination_score"].dropna()
        if len(cond_scores) == 0:
            continue
        c = CONDITION_COLORS.get(cond, "#888888")
        short = CONDITION_SHORT.get(cond, cond)

        if unique_vals <= 3:
            # Very little spread: use a stem-like visualization
            # Count values in bins for cleaner display
            counts, _ = np.histogram(cond_scores.values, bins=bins)
            ax.step(
                bins[:-1], counts, where="mid",
                color=c, alpha=0.7, linewidth=0.9, label=short,
            )
        else:
            ax.hist(
                cond_scores.values, bins=bins,
                color=c, alpha=0.25, edgecolor=c, linewidth=0.5,
                label=short, histtype="stepfilled",
            )

    # Threshold line
    ax.axvline(
        x=CONTAMINATION_THRESHOLD, color="red", linestyle="--",
        linewidth=1.0, alpha=0.8, zorder=5,
    )
    ax.text(
        CONTAMINATION_THRESHOLD + 0.02, ax.get_ylim()[1] * 0.92,
        "threshold", fontsize=max(MIN_PT - 1, 5), color="red",
        va="top", ha="left",
    )

    n_above = int((scores_all >= CONTAMINATION_THRESHOLD).sum())
    if n_above == 0:
        ax.text(
            0.97, 0.97,
            "No tasks exceeded\nthreshold",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=MIN_PT, fontstyle="italic", color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

    ax.set_xlabel("Contamination score", fontsize=MIN_PT)
    ax.set_ylabel("Count", fontsize=MIN_PT)
    ax.legend(
        fontsize=max(MIN_PT - 2, 5), frameon=True, loc="upper right",
        ncol=2, handletextpad=0.3, columnspacing=0.5,
    )
    style_grid(ax, axis="y")


def panel_b(ax) -> None:
    """Paired horizontal bar chart: original vs zeroed mean total score."""
    df = load_all()

    if "contamination_score" not in df.columns:
        ax.text(0.5, 0.5, "No contamination_score column",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=MIN_PT, color="grey")
        return

    # Compute original and zeroed means per condition
    records: list[dict] = []
    for cond in CONDITION_ORDER_NO_ORACLE:
        cond_df = df[df["condition"] == cond]
        if cond_df.empty:
            continue
        original_mean = float(cond_df["total"].mean())

        # Zeroed: set total to 0 for tasks with contamination_score > 0.5
        zeroed_totals = cond_df["total"].copy()
        contaminated_mask = cond_df["contamination_score"] >= CONTAMINATION_THRESHOLD
        zeroed_totals.loc[contaminated_mask] = 0.0
        zeroed_mean = float(zeroed_totals.mean())

        n_flagged = int(contaminated_mask.sum())
        records.append({
            "condition": cond,
            "original": original_mean,
            "zeroed": zeroed_mean,
            "delta": zeroed_mean - original_mean,
            "n_flagged": n_flagged,
        })

    if not records:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=MIN_PT, color="grey")
        return

    # Sort by original score descending
    records.sort(key=lambda r: r["original"], reverse=True)

    conditions = [r["condition"] for r in records]
    originals = [r["original"] for r in records]
    zerodeds = [r["zeroed"] for r in records]
    deltas = [r["delta"] for r in records]
    n_flagged_list = [r["n_flagged"] for r in records]

    y_pos = np.arange(len(conditions))
    bar_height = 0.35

    # Original bars (solid)
    colors_orig = [CONDITION_COLORS.get(c, "#888888") for c in conditions]
    ax.barh(
        y_pos - bar_height / 2, originals, height=bar_height,
        color=colors_orig, edgecolor="white", linewidth=0.3,
        label="Original", alpha=0.9, zorder=3,
    )

    # Zeroed bars (hatched / lighter)
    colors_zeroed = [CONDITION_COLORS.get(c, "#888888") for c in conditions]
    ax.barh(
        y_pos + bar_height / 2, zerodeds, height=bar_height,
        color=colors_zeroed, edgecolor="white", linewidth=0.3,
        label="After zeroing", alpha=0.45, hatch="//", zorder=3,
    )

    # Delta annotations
    max_score = max(max(originals), max(zerodeds)) if originals else 1
    for i, (orig, zeroed, delta, nf) in enumerate(
        zip(originals, zerodeds, deltas, n_flagged_list)
    ):
        if abs(delta) > 0.05:
            ax.text(
                max(orig, zeroed) + max_score * 0.02,
                y_pos[i],
                f"{delta:+.1f} ({nf})",
                va="center", ha="left",
                fontsize=max(MIN_PT - 2, 5), color="#333333",
            )
        elif nf == 0:
            ax.text(
                max(orig, zeroed) + max_score * 0.02,
                y_pos[i],
                "0",
                va="center", ha="left",
                fontsize=max(MIN_PT - 2, 5), color="#999999",
            )

    # Y-axis labels
    short_labels = [CONDITION_SHORT.get(c, c) for c in conditions]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=MIN_PT)
    ax.invert_yaxis()  # top = highest score

    ax.set_xlabel("Mean total score", fontsize=MIN_PT)
    ax.legend(
        fontsize=max(MIN_PT - 1, 5), frameon=True, loc="lower right",
        handletextpad=0.3,
    )

    style_grid(ax, axis="x")


def main() -> None:
    """Create SI Figure 9 and save to results/analysis/."""
    fig_w = FIG_W * 1.5   # 180 mm (7.09 in)
    fig_h = FIG_H * 1.5   # 120 mm (4.72 in) — 3:2 ratio

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [0.50, 0.50], "wspace": 0.35},
    )

    panel_a(axes[0])
    panel_b(axes[1])

    save_fig(fig, "si_fig9_contamination")


if __name__ == "__main__":
    main()
