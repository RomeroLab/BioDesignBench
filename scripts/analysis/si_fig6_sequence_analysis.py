#!/usr/bin/env python3
"""SI Figure 6: Sequence Analysis.

(a) Sequence length distribution — box plot per condition.
(b) Amino acid composition deviation heatmap — deviation from natural
    UniProt frequencies for 20 standard AAs across 9 conditions.

Sequences read from output/designed_sequences.fasta in each task directory.

Layout: 1 row, 2 columns + legend row below.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.si_common import *

from collections import defaultdict


# ── Standard amino acids & natural UniProt frequencies (%) ─────────────
STANDARD_AAS = list("ARNDCEQGHILKMFPSTWYV")

NATURAL_AA_FREQ: dict[str, float] = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.37,
    "E": 6.75, "Q": 3.93, "G": 7.08, "H": 2.27, "I": 5.96,
    "L": 9.66, "K": 5.84, "M": 2.42, "F": 3.86, "P": 4.70,
    "S": 6.56, "T": 5.34, "W": 1.08, "Y": 2.92, "V": 6.87,
}


def _parse_fasta(fasta_path: Path) -> list[str]:
    """Parse a FASTA file and return list of sequences."""
    sequences: list[str] = []
    current: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            elif line:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _extract_sequences_from_fasta() -> dict[str, list[str]]:
    """Extract designed sequences from FASTA files in output directories."""
    from scripts.analysis.load_results import CONDITION_MAP, EXCLUDED_TASKS

    cond_seqs: dict[str, list[str]] = defaultdict(list)
    for condition, info in CONDITION_MAP.items():
        agent_dir = info["path"]
        if not agent_dir.exists():
            continue
        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            # Check excluded
            tid = task_dir.name
            if tid in EXCLUDED_TASKS:
                continue
            fasta = task_dir / "output" / "designed_sequences.fasta"
            if not fasta.exists():
                continue
            seqs = _parse_fasta(fasta)
            for seq in seqs:
                seq_upper = seq.upper()
                # Basic sanity: mostly standard AA letters
                aa_frac = sum(1 for c in seq_upper if c in STANDARD_AAS) / max(len(seq_upper), 1)
                if aa_frac > 0.8 and len(seq_upper) >= 10:
                    cond_seqs[condition].append(seq_upper)
    return dict(cond_seqs)


def _compute_aa_freq(sequences: list[str]) -> dict[str, float]:
    """Compute amino acid frequency (%) over a collection of sequences."""
    counts: dict[str, int] = {aa: 0 for aa in STANDARD_AAS}
    total = 0
    for seq in sequences:
        for ch in seq:
            if ch in counts:
                counts[ch] += 1
                total += 1
    if total == 0:
        return {aa: 0.0 for aa in STANDARD_AAS}
    return {aa: (counts[aa] / total) * 100.0 for aa in STANDARD_AAS}


# ── Panel functions ────────────────────────────────────────────────────

def panel_a(ax, cond_seqs: dict[str, list[str]]) -> None:
    """Sequence length distribution: box plots per condition."""
    conditions = [c for c in CONDITION_ORDER if c in cond_seqs and len(cond_seqs[c]) > 0]
    if not conditions:
        ax.text(0.5, 0.5, "No designed sequences found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=MIN_PT, color="grey")
        return

    data_groups = [[len(s) for s in cond_seqs[c]] for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conditions]

    bp = ax.boxplot(
        data_groups,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=1.5, alpha=0.4, markerfacecolor="#666666"),
        medianprops=dict(color="black", linewidth=0.6),
        whiskerprops=dict(linewidth=0.4),
        capprops=dict(linewidth=0.4),
        boxprops=dict(linewidth=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    short_labels = [CONDITION_SHORT.get(c, c) for c in conditions]
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(short_labels, rotation=45, ha="right",
                       fontsize=max(MIN_PT - 2, 4))
    ax.set_ylabel("Sequence length\n(residues)", fontsize=MIN_PT - 1)

    # Annotate median values
    for i, data in enumerate(data_groups):
        median = np.median(data)
        ax.text(i + 1, ax.get_ylim()[1] * 0.02 + median, f"{int(median)}",
                ha="center", va="bottom", fontsize=max(MIN_PT - 3, 4),
                color="#333333")

    style_grid(ax, axis="y")


def panel_b(ax, cond_seqs: dict[str, list[str]]) -> None:
    """Amino acid composition deviation heatmap."""
    conditions = [c for c in CONDITION_ORDER if c in cond_seqs and len(cond_seqs[c]) > 0]
    if not conditions:
        ax.text(0.5, 0.5, "No designed sequences found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=MIN_PT, color="grey")
        return

    # Compute deviation matrix: rows = AAs, columns = conditions
    n_aa = len(STANDARD_AAS)
    n_cond = len(conditions)
    dev_matrix = np.zeros((n_aa, n_cond))

    for j, cond in enumerate(conditions):
        freq = _compute_aa_freq(cond_seqs[cond])
        for i, aa in enumerate(STANDARD_AAS):
            dev_matrix[i, j] = freq[aa] - NATURAL_AA_FREQ[aa]

    # Symmetric color limits
    abs_max = max(np.abs(dev_matrix).max(), 0.5)

    # Diverging colormap
    im = ax.imshow(
        dev_matrix, cmap=plt.cm.RdBu_r, aspect="auto",
        vmin=-abs_max, vmax=abs_max,
    )

    # Annotate cells where |deviation| > 1.5%
    for i in range(n_aa):
        for j in range(n_cond):
            val = dev_matrix[i, j]
            if abs(val) > 1.5:
                rgba = plt.cm.RdBu_r((val + abs_max) / (2 * abs_max))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "white" if lum < 0.45 else "black"
                ax.text(
                    j, i, f"{val:+.1f}",
                    ha="center", va="center",
                    fontsize=max(MIN_PT - 3, 3), color=text_color,
                )

    # Axis labels
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels(
        [CONDITION_SHORT.get(c, c) for c in conditions],
        rotation=45, ha="right", fontsize=max(MIN_PT - 2, 4),
    )
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(STANDARD_AAS, fontsize=max(MIN_PT - 2, 4))

    # Vertical colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Deviation from\nnatural (%)", fontsize=MIN_PT - 1, labelpad=2)
    cbar.ax.tick_params(labelsize=max(MIN_PT - 2, 4), width=0.3, length=2)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=True, left=True, right=False)


def main() -> None:
    """Create SI Figure 6 and save to results/analysis/."""
    from matplotlib.gridspec import GridSpec

    # Load sequences once (shared between panels)
    cond_seqs = _extract_sequences_from_fasta()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[0.90, 0.10],
        width_ratios=[0.40, 0.60],
        hspace=0.55, wspace=0.40,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_a(ax_a, cond_seqs)
    panel_b(ax_b, cond_seqs)

    # Bottom: shared legend
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")

    # Build legend from condition colors
    from matplotlib.patches import Patch
    handles = []
    for cond in CONDITION_ORDER:
        c = CONDITION_COLORS.get(cond, "#888888")
        if is_bm(cond) or "Hardcoded" in cond:
            handles.append(Patch(facecolor="white", edgecolor=c, linewidth=0.8, label=cond))
        else:
            handles.append(Patch(facecolor=c, edgecolor=c, alpha=0.75, label=cond))
    ax_leg.legend(
        handles=handles, loc="center",
        fontsize=max(MIN_PT - 2, 4), frameon=False,
        ncol=3, handletextpad=0.3, columnspacing=1.0, labelspacing=0.5,
    )

    save_fig(fig, "si_fig6_sequence_analysis")


if __name__ == "__main__":
    main()
