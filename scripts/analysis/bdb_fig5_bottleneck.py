#!/usr/bin/env python3
"""BioDesignBench Figure 5: Bottleneck Decomposition by Domain.

Restructured 2-panel figure:
  Panel A (60%): Full knowledge-gap decomposition (A/B/C/D) per model × domain
  Panel B (40%): Compact domain score distribution boxplot

Usage:
    python -m scripts.analysis.bdb_fig5_bottleneck
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Nature-style rcParams ───────────────────────────────────────────────────
_MIN_PT = 7
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": _MIN_PT,
    "axes.linewidth": 0.5,
    "axes.labelsize": _MIN_PT + 1,
    "axes.titlesize": 9,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.labelsize": _MIN_PT,
    "ytick.labelsize": _MIN_PT,
    "legend.fontsize": _MIN_PT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Colors ──────────────────────────────────────────────────────────────────

# Case colors (knowledge-gap decomposition)
CASE_COLORS = {
    "A": "#2ca02c",   # Green — Full Knowledge (plan + exec)
    "B": "#E69F00",   # Orange — Tool Gap (knows, can't execute)
    "C": "#CC79A7",   # Reddish Purple — Science Gap (doesn't know)
    "D": "#56B4E9",   # Sky Blue — Serendipity (executes w/o planning)
}

# Model colors (Okabe-Ito)
MODEL_COLORS = {
    "DeepSeek V3": "#CC79A7",
    "GPT-5": "#56B4E9",
    "Sonnet 4.5": "#E69F00",
    "Gemini 2.5 Pro": "#009E73",
}

# 3 MCP-capable LLMs in guided mode (Gemini excluded: 0% exec due to MCP limitation)
MODELS = ["DeepSeek V3", "GPT-5", "Sonnet 4.5"]
MODEL_CONDITIONS = {m: f"{m} user" for m in MODELS}
MODEL_SHORT = {"DeepSeek V3": "DS", "GPT-5": "G5", "Sonnet 4.5": "So"}

SUBJECTS = ["antibody", "enzyme", "binder", "scaffold", "fluorescent_protein"]
SUBJECT_LABELS = {
    "antibody": "Antibody",
    "enzyme": "Enzyme",
    "binder": "Binder",
    "scaffold": "Scaffold",
    "fluorescent_protein": "Fluor.\nProtein",
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_domain_gap() -> pd.DataFrame:
    """Load domain-level Case A/B/C/D fractions."""
    path = PROJECT_ROOT / "results" / "analysis" / "plan_exec_gap_v2_by_domain.csv"
    return pd.read_csv(path)


def load_scores() -> pd.DataFrame:
    """Load all scores via load_results."""
    from scripts.analysis.load_results import load_all
    return load_all()


# ── Panel A: Bottleneck Decomposition ───────────────────────────────────────

def _draw_panel_a(ax: plt.Axes, dom: pd.DataFrame) -> None:
    """Panel A: Tool Gap vs Science Gap — grouped bars per domain.

    For each of 5 molecular subjects: 3 model sub-groups.
    Each sub-group has 2 side-by-side bars (Tool Gap orange, Science Gap pink).
    This layout reveals domain-specific bottleneck patterns.
    """
    n_subjects = len(SUBJECTS)
    n_models = len(MODELS)

    # Layout: each subject occupies 1.0 unit on x-axis.
    # Within each subject: n_models sub-groups, each with 2 bars.
    sub_w = 0.12  # individual bar width
    pair_w = sub_w * 2 + 0.01  # width of one (B, C) pair
    group_w = pair_w * n_models + 0.04 * (n_models - 1)

    x_base = np.arange(n_subjects) * 1.2  # wider spacing between subjects

    for mi, model in enumerate(MODELS):
        cond = MODEL_CONDITIONS[model]
        color = MODEL_COLORS[model]
        pair_offset = (mi - (n_models - 1) / 2) * (pair_w + 0.04)

        b_vals, c_vals = [], []
        for si, subject in enumerate(SUBJECTS):
            row = dom[(dom["condition"] == cond) & (dom["subject"] == subject)]
            if len(row) == 0:
                b_vals.append(0)
                c_vals.append(0)
                continue
            r = row.iloc[0]
            b_vals.append(r["frac_case_b"])
            c_vals.append(r["frac_case_c"])

        x_b = x_base + pair_offset - sub_w / 2 - 0.005
        x_c = x_base + pair_offset + sub_w / 2 + 0.005

        # Tool Gap bars (B) — orange with model-colored edge
        ax.bar(x_b, b_vals, sub_w,
               color=CASE_COLORS["B"], edgecolor=color,
               linewidth=0.8, alpha=0.85,
               label=f"{model} — Tool" if mi == 0 else "_nolegend_")

        # Science Gap bars (C) — pink with model-colored edge
        ax.bar(x_c, c_vals, sub_w,
               color=CASE_COLORS["C"], edgecolor=color,
               linewidth=0.8, alpha=0.85,
               label=f"{model} — Science" if mi == 0 else "_nolegend_")

    # Model initials below each subject group
    for si in range(n_subjects):
        for mi, model in enumerate(MODELS):
            pair_offset = (mi - (n_models - 1) / 2) * (pair_w + 0.04)
            xpos = x_base[si] + pair_offset
            ax.text(xpos, -0.045, MODEL_SHORT[model],
                    ha="center", va="top", fontsize=5.5,
                    color=MODEL_COLORS[model], fontweight="bold")

    # Domain-level summary: B/(B+C) ratio
    for si, subject in enumerate(SUBJECTS):
        sub = dom[(dom["condition"].isin(MODEL_CONDITIONS.values())) &
                  (dom["subject"] == subject)]
        mean_b = sub["frac_case_b"].mean()
        mean_c = sub["frac_case_c"].mean()
        total = mean_b + mean_c
        if total > 0.01:
            ratio = mean_b / total
            # Place above the tallest bar in this group
            all_vals = []
            for mi, model in enumerate(MODELS):
                cond = MODEL_CONDITIONS[model]
                r = dom[(dom["condition"] == cond) & (dom["subject"] == subject)]
                if len(r) > 0:
                    all_vals.append(r.iloc[0]["frac_case_b"])
                    all_vals.append(r.iloc[0]["frac_case_c"])
            y_top = max(all_vals) + 0.04 if all_vals else 0.5
            is_tool_heavy = ratio > 0.25
            ax.text(
                x_base[si], y_top,
                f"B/(B+C)={ratio:.0%}",
                ha="center", va="bottom", fontsize=6,
                color=CASE_COLORS["B"] if is_tool_heavy else CASE_COLORS["C"],
                fontweight="bold",
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels([SUBJECT_LABELS[s] for s in SUBJECTS],
                       fontsize=_MIN_PT + 0.5)
    ax.set_ylabel("Fraction of Pipeline Steps", fontsize=_MIN_PT + 1)
    ax.set_ylim(-0.07, 1.0)
    ax.set_title("A. Bottleneck: Tool Gap vs Science Gap by Domain",
                 fontsize=9, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)

    # Compact legend: gap types + model edges
    gap_handles = [
        mpatches.Patch(facecolor=CASE_COLORS["B"], edgecolor="#888888",
                       linewidth=0.5, label="Tool Gap (B)"),
        mpatches.Patch(facecolor=CASE_COLORS["C"], edgecolor="#888888",
                       linewidth=0.5, label="Science Gap (C)"),
    ]
    model_handles = [
        mpatches.Patch(facecolor="white", edgecolor=MODEL_COLORS[m],
                       linewidth=1.5, label=m)
        for m in MODELS
    ]
    ax.legend(
        handles=gap_handles + model_handles,
        loc="upper right", fontsize=6, framealpha=0.92,
        ncol=1, handlelength=1.2, handletextpad=0.3,
        borderpad=0.4,
    )


# ── Panel B: Compact Score Distribution ─────────────────────────────────────

def _draw_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel B: Score distribution by domain — compact boxplot with medians."""
    # Filter to LLM user conditions only (match Panel A)
    conditions = list(MODEL_CONDITIONS.values())
    sub = df[df["condition"].isin(conditions)].copy()
    sub = sub[sub["molecular_subject"].isin(SUBJECTS)]

    # Order subjects to match Panel A
    sub["molecular_subject"] = pd.Categorical(
        sub["molecular_subject"], categories=SUBJECTS, ordered=True
    )

    # Collect data for boxplot
    data = []
    labels = []
    medians = []
    for s in SUBJECTS:
        vals = sub[sub["molecular_subject"] == s]["total"].dropna().values
        data.append(vals)
        labels.append(SUBJECT_LABELS[s])
        medians.append(np.median(vals) if len(vals) > 0 else 0)

    # Subject colors (soft tones)
    subject_colors = ["#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#0C2C84"]

    bp = ax.boxplot(
        data,
        positions=range(len(SUBJECTS)),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2, markerfacecolor="#cccccc",
                        markeredgecolor="none", alpha=0.5),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6, color="#666666"),
        capprops=dict(linewidth=0.6, color="#666666"),
        boxprops=dict(linewidth=0.5),
    )

    for patch, color in zip(bp["boxes"], subject_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate medians
    for i, med in enumerate(medians):
        ax.text(
            i, med + 2.5,
            f"{med:.0f}",
            ha="center", va="bottom",
            fontsize=_MIN_PT, fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(range(len(SUBJECTS)))
    ax.set_xticklabels(labels, fontsize=_MIN_PT)
    ax.set_ylabel("Total Score (/100)", fontsize=_MIN_PT + 1)
    ax.set_ylim(0, 105)
    ax.set_title("B. Score Distribution by Domain",
                 fontsize=9, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)


# ── SI Figure: Quality Heatmap (moved from old Figure 5) ───────────────────

def generate_si_quality_heatmap(df: pd.DataFrame,
                                save_path: Path | None = None) -> None:
    """Quality-by-Domain heatmap — moved to SI from the old Figure 5."""
    import seaborn as sns

    if save_path is None:
        save_path = OUT_DIR / "si_figure_quality_by_domain.png"

    conditions = [
        "Oracle", "Human Expert", "Hardcoded Pipeline",
        "GPT-5 user", "DeepSeek V3 user", "Sonnet 4.5 user",
    ]
    cond_labels = [
        "Oracle", "Human\nExpert", "Hardcoded",
        "GPT-5 G", "DeepSeek\nV3 G", "Sonnet\n4.5 G",
    ]

    sub = df[df["condition"].isin(conditions) & df["molecular_subject"].isin(SUBJECTS)]
    pivot = sub.pivot_table(
        index="molecular_subject", columns="condition",
        values="quality", aggfunc="mean",
    )
    pivot = pivot.reindex(index=SUBJECTS, columns=conditions)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".1f",
        cmap="YlOrRd", linewidths=0.5, linecolor="white",
        xticklabels=cond_labels,
        yticklabels=[SUBJECT_LABELS[s] for s in SUBJECTS],
        cbar_kws={"label": "Quality Score (/35)", "shrink": 0.8},
    )
    ax.set_title("Quality Score by Domain × Condition", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    for ext in [".pdf", ".svg"]:
        fig.savefig(save_path.with_suffix(ext), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved SI heatmap: {save_path}")


# ── Main figure generator ───────────────────────────────────────────────────

def generate_fig5(dom: pd.DataFrame, scores: pd.DataFrame,
                  save_path: Path | None = None) -> None:
    """Generate Figure 5: Bottleneck Decomposition by Domain.

    Args:
        dom: Domain-level Case A/B/C/D fractions from plan_exec_gap_v2_by_domain.csv.
        scores: Full score DataFrame from load_all().
        save_path: Output path. Defaults to figures/figure5.png.
    """
    if save_path is None:
        save_path = OUT_DIR / "figure5.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 2-panel layout: A (60%), B (40%)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(180 / 25.4, 80 / 25.4),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35},
    )

    _draw_panel_a(ax_a, dom)
    _draw_panel_b(ax_b, scores)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    for ext in [".pdf", ".svg"]:
        fig.savefig(save_path.with_suffix(ext), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path} (.png/.pdf/.svg)")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load data, generate Figure 5 + SI heatmap, print paper numbers."""
    print("Loading data...")
    dom = load_domain_gap()
    scores = load_scores()

    print("\nGenerating Figure 5...")
    generate_fig5(dom, scores)

    print("\nGenerating SI quality heatmap...")
    generate_si_quality_heatmap(scores)

    # Print paper numbers
    print("\n" + "=" * 70)
    print("PAPER NUMBERS — Figure 5 (Bottleneck Decomposition)")
    print("=" * 70)

    guided = dom[dom["condition"].isin(MODEL_CONDITIONS.values())]
    guided = guided[guided["subject"] != "unknown"]

    print("\n### Panel A: Case Distribution by Domain (Guided Mode, 4 LLMs avg)")
    print(f"\n{'Domain':22s} {'Case A':>8s} {'Case B':>8s} {'Case C':>8s} {'Case D':>8s} {'B/(B+C)':>8s}")
    print("-" * 66)
    for subject in SUBJECTS:
        sub = guided[guided["subject"] == subject]
        a = sub["frac_case_a"].mean()
        b = sub["frac_case_b"].mean()
        c = sub["frac_case_c"].mean()
        d = sub["frac_case_d"].mean()
        ratio = b / (b + c) if (b + c) > 0 else 0
        print(f"{SUBJECT_LABELS[subject].replace(chr(10), ' '):22s} "
              f"{a:>7.1%} {b:>7.1%} {c:>7.1%} {d:>7.1%} {ratio:>7.0%}")

    print("\n### Panel A: Per-Model Breakdown")
    for model in MODELS:
        cond = MODEL_CONDITIONS[model]
        sub = guided[guided["condition"] == cond]
        print(f"\n  {model}:")
        for _, row in sub.iterrows():
            s = row["subject"]
            b, c = row["frac_case_b"], row["frac_case_c"]
            label = "TOOL" if (b + c > 0 and b / (b + c) > 0.5) else "SCI"
            print(f"    {s:22s}: A={row['frac_case_a']:.1%}  "
                  f"B={b:.1%}  C={c:.1%}  → {label}")

    print("\n### Panel B: Domain Score Medians (Guided LLMs)")
    llm_conds = list(MODEL_CONDITIONS.values())
    llm_scores = scores[scores["condition"].isin(llm_conds)]
    for subject in SUBJECTS:
        sub = llm_scores[llm_scores["molecular_subject"] == subject]
        med = sub["total"].median()
        mean = sub["total"].mean()
        print(f"  {SUBJECT_LABELS[subject].replace(chr(10), ' '):22s}: "
              f"median={med:.1f}, mean={mean:.1f}, n={len(sub)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
