#!/usr/bin/env python3
"""BioDesignBench Fig 6: Tool Usage Analysis — 2×2 (180×160mm).

(a) Tool Frequency Heatmap — mean bio-tool calls per task across conditions
(b) Tool Diversity vs Performance — Shannon entropy vs total score scatter
(c) Pipeline Step Completion — per-step adherence to canonical pipeline
(d) Benchmark → User Mode Shift — arrow plot of tool volume vs diversity change

Output: 180mm × 160mm, 300 dpi.  All text >= 7 pt.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
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

# ── Unified Paul Tol palette ─────────────────────────────────────────────
CONDITION_COLORS = {
    "Human Expert":             "#AA3377",
    "Hardcoded Pipeline":       "#EE7733",
    "DeepSeek V3 user":         "#4477AA",
    "DeepSeek V3 benchmark":    "#4477AA",
    "GPT-5 user":               "#EE6677",
    "GPT-5 benchmark":          "#EE6677",
    "Sonnet 4.5 user":          "#228833",
    "Sonnet 4.5 benchmark":     "#228833",
    "Gemini 2.5 Pro user":      "#66CCEE",
    "Gemini 2.5 Pro benchmark": "#66CCEE",
}

COND_ORDER = [
    "Human Expert", "Hardcoded Pipeline",
    "DeepSeek V3 user", "DeepSeek V3 benchmark",
    "GPT-5 user", "GPT-5 benchmark",
    "Sonnet 4.5 user", "Sonnet 4.5 benchmark",
    "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark",
]

COND_SHORT = {
    "Human Expert": "Expert", "Hardcoded Pipeline": "Hardcoded",
    "DeepSeek V3 user": "DS V3 U", "DeepSeek V3 benchmark": "DS V3 B",
    "GPT-5 user": "GPT-5 U", "GPT-5 benchmark": "GPT-5 B",
    "Sonnet 4.5 user": "Son 4.5 U", "Sonnet 4.5 benchmark": "Son 4.5 B",
    "Gemini 2.5 Pro user": "Gem 2.5 U", "Gemini 2.5 Pro benchmark": "Gem 2.5 B",
}

# Full names for panels with enough horizontal space (e.g. panel C y-labels)
COND_FULL = {
    "Human Expert": "Human Expert", "Hardcoded Pipeline": "Hardcoded",
    "DeepSeek V3 user": "DeepSeek V3 User", "DeepSeek V3 benchmark": "DeepSeek V3 BM",
    "GPT-5 user": "GPT-5 User", "GPT-5 benchmark": "GPT-5 BM",
    "Sonnet 4.5 user": "Sonnet 4.5 User", "Sonnet 4.5 benchmark": "Sonnet 4.5 BM",
    "Gemini 2.5 Pro user": "Gemini 2.5 Pro User", "Gemini 2.5 Pro benchmark": "Gemini 2.5 Pro BM",
}

UTILITY_TOOLS = {"execute_python", "write_file", "read_file"}

# ── Pipeline step mapping (MCP tool name → canonical step) ───────────────
PIPELINE_STEP_MAP = {
    "generate_backbone": "backbone",
    "design_binder": "seq_design", "optimize_sequence": "seq_design",
    "rosetta_design": "seq_design",
    "predict_structure": "struct_pred", "predict_complex": "struct_pred",
    "predict_structure_boltz": "struct_pred", "predict_affinity_boltz": "struct_pred",
    "score_stability": "scoring", "analyze_interface": "scoring",
    "rosetta_score": "scoring", "rosetta_interface_score": "scoring",
    "validate_design": "scoring",
}
PIPELINE_STEPS = ["backbone", "seq_design", "struct_pred", "scoring"]
PIPELINE_LABELS = ["Backbone\ngeneration", "Sequence\ndesign",
                   "Structure\nprediction", "Scoring &\nvalidation"]

MODELS = [
    {"name": "DeepSeek V3",   "user": "DeepSeek V3 user",
     "bm": "DeepSeek V3 benchmark",    "color": "#4477AA"},
    {"name": "GPT-5",         "user": "GPT-5 user",
     "bm": "GPT-5 benchmark",          "color": "#EE6677"},
    {"name": "Sonnet 4.5",    "user": "Sonnet 4.5 user",
     "bm": "Sonnet 4.5 benchmark",     "color": "#228833"},
    {"name": "Gemini 2.5 Pro","user": "Gemini 2.5 Pro user",
     "bm": "Gemini 2.5 Pro benchmark", "color": "#66CCEE"},
]


# ── Data loading (cached) ────────────────────────────────────────────────
_CACHE: dict = {}


def _get_data() -> pd.DataFrame:
    """Load DataFrame with scores + MCP bio-tool sequences from raw results."""
    if "df" in _CACHE:
        return _CACHE["df"]

    from scripts.analysis.load_results import load_all, CONDITION_MAP, EXCLUDED_TASKS

    df = load_all()

    # Load MCP tool sequences from raw tool_call_log
    mcp_seqs: dict[tuple[str, str], list[str]] = {}
    for condition, info in CONDITION_MAP.items():
        if condition == "Oracle":
            continue
        agent_dir = info["path"]
        if not agent_dir.exists():
            continue
        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            rf = task_dir / "result.json"
            if not rf.exists():
                continue
            with open(rf) as f:
                result = json.load(f)
            tid = result.get("task_id", "")
            if tid in EXCLUDED_TASKS:
                continue
            tool_log = result.get("raw_output", {}).get("tool_call_log", [])
            seq = [tc.get("tool", "") for tc in tool_log if tc.get("tool")]
            mcp_seqs[(condition, tid)] = seq

    df["mcp_seq"] = df.apply(
        lambda r: mcp_seqs.get((r["condition"], r["task_id"]), []), axis=1,
    )
    df["bio_seq"] = df["mcp_seq"].apply(
        lambda s: [t for t in s if t not in UTILITY_TOOLS],
    )

    _CACHE["df"] = df
    return df


# ── Helpers ──────────────────────────────────────────────────────────────

def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═════════════════════════════════════════════════════════════════════════
# Panel A — Tool Frequency Heatmap
# ═════════════════════════════════════════════════════════════════════════

def panel_a(ax) -> None:
    """Mean bio-tool calls per task — (tools × conditions) heatmap."""
    df = _get_data()

    # Discover bio tools by total frequency
    counts: Counter = Counter()
    for seq in df["bio_seq"]:
        if isinstance(seq, list):
            counts.update(seq)
    sorted_tools = [t for t, _ in counts.most_common()]
    sorted_tools = sorted_tools[:8]  # Top 8 most-used bio tools for compactness
    if not sorted_tools:
        ax.text(0.5, 0.5, "No bio-tool data", transform=ax.transAxes,
                ha="center", va="center", fontsize=_MIN_PT, color="grey")
        return

    # Build matrix: tools (rows) × conditions (cols)
    records = []
    for cond in COND_ORDER:
        cond_df = df[df["condition"] == cond]
        n = max(len(cond_df), 1)
        row = {t: 0.0 for t in sorted_tools}
        for bio_seq in cond_df["bio_seq"]:
            if isinstance(bio_seq, list):
                for t in bio_seq:
                    if t in row:
                        row[t] += 1
        records.append({t: v / n for t, v in row.items()})

    mat = pd.DataFrame(records, index=COND_ORDER).T
    # Drop tools with zero total
    mat = mat[mat.sum(axis=1) > 0]
    data = mat.values
    vmax = max(data.max(), 1.0)

    cmap = plt.cm.OrRd
    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=0, vmax=vmax)

    # Labels
    full_cols = [COND_FULL[c] for c in COND_ORDER]
    ax.set_xticks(range(len(full_cols)))
    ax.set_xticklabels(full_cols, rotation=30, ha="right",
                       fontsize=max(_MIN_PT - 2, 5))
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=max(_MIN_PT - 2, 5))

    # Annotate cells >= 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v >= 0.5:
                rgba = cmap(v / vmax)
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                tc = "white" if lum < 0.5 else "black"
                fmt = f"{v:.1f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        fontsize=5, color=tc)

    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Mean calls / task", fontsize=_MIN_PT, labelpad=2)
    cbar.ax.tick_params(labelsize=_MIN_PT - 1, width=0.3, length=2)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(top=False, bottom=True, left=True, right=False)


# ═════════════════════════════════════════════════════════════════════════
# Panel B — Tool Diversity vs Performance
# ═════════════════════════════════════════════════════════════════════════

def panel_b(ax) -> None:
    """Shannon entropy of bio-tool distribution vs mean total score."""
    from scipy.stats import pearsonr

    df = _get_data()
    # Collect all unique bio tool names
    all_bio = sorted({t for seq in df["bio_seq"] if isinstance(seq, list)
                      for t in seq})

    xs, ys, cs = [], [], []
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        tool_counts = np.zeros(len(all_bio))
        for bio_seq in sub["bio_seq"]:
            if isinstance(bio_seq, list):
                for t in bio_seq:
                    if t in all_bio:
                        tool_counts[all_bio.index(t)] += 1
        xs.append(_shannon_entropy(tool_counts))
        ys.append(sub["total"].mean())
        cs.append(cond)

    x, y = np.array(xs), np.array(ys)

    # Scatter with per-condition markers
    for i, cond in enumerate(cs):
        c = CONDITION_COLORS.get(cond, "#888")
        if cond == "Human Expert":
            kw = dict(marker="^", facecolors=c, edgecolors="white", linewidths=0.4)
        elif "Hardcoded" in cond:
            kw = dict(marker="s", facecolors=c, edgecolors="white", linewidths=0.4)
        elif "user" in cond.lower():
            kw = dict(marker="o", facecolors=c, edgecolors="white", linewidths=0.4)
        else:
            kw = dict(marker="o", facecolors="white", edgecolors=c, linewidths=0.8)
        ax.scatter(x[i], y[i], s=55, zorder=5, **kw)

    # Pearson r + regression line
    if len(x) >= 3 and np.var(x) > 1e-12 and np.var(y) > 1e-12:
        r, p = pearsonr(x, y)
        try:
            z = np.polyfit(x, y, 1)
            xl = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
            ax.plot(xl, np.polyval(z, xl), color="#999", ls="--", lw=0.8, zorder=2)
        except np.linalg.LinAlgError:
            pass
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(0.05, 0.95, f"r = {r:.2f}{sig}",
                transform=ax.transAxes, fontsize=_MIN_PT, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.8))

    ax.set_xlabel("Tool diversity (Shannon entropy)", fontsize=_MIN_PT)
    ax.set_ylabel("Mean total score", fontsize=_MIN_PT)
    _style(ax)


# ═════════════════════════════════════════════════════════════════════════
# Panel C — Pipeline Step Completion
# ═════════════════════════════════════════════════════════════════════════

def panel_c(ax) -> None:
    """Fraction of tasks completing each canonical pipeline step."""
    df = _get_data()

    n_conds = len(COND_ORDER)
    n_steps = len(PIPELINE_STEPS)
    completion = np.zeros((n_conds, n_steps))

    for ci, cond in enumerate(COND_ORDER):
        sub = df[df["condition"] == cond]
        n = len(sub)
        if n == 0:
            continue
        for bio_seq in sub["bio_seq"]:
            if not isinstance(bio_seq, list):
                continue
            steps_hit = {PIPELINE_STEP_MAP.get(t) for t in bio_seq} - {None}
            for si, step in enumerate(PIPELINE_STEPS):
                if step in steps_hit:
                    completion[ci, si] += 1
        completion[ci] /= n

    # Grouped horizontal bars
    y_pos = np.arange(n_conds)
    bar_h = 0.19
    # Sequential blue palette: light → dark conveys pipeline progression
    step_colors = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]

    for si in range(n_steps):
        offset = (si - 1.5) * bar_h
        ax.barh(y_pos + offset, completion[:, si], height=bar_h * 0.88,
                color=step_colors[si], edgecolor="white", linewidth=0.3,
                label=PIPELINE_LABELS[si].replace("\n", " "), zorder=3)

    full = [COND_FULL[c] for c in COND_ORDER]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(full, fontsize=_MIN_PT - 1)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Fraction of tasks", fontsize=_MIN_PT)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.invert_yaxis()

    ax.legend(fontsize=max(_MIN_PT - 2, 5), frameon=True, loc="lower right",
              ncol=1, handletextpad=0.3, handlelength=1.2,
              borderpad=0.4, labelspacing=0.3)
    _style(ax)


# ═════════════════════════════════════════════════════════════════════════
# Panel D — Benchmark → User Mode Shift
# ═════════════════════════════════════════════════════════════════════════

def panel_d(ax) -> None:
    """Arrow plot: BM → User shift in bio-tool volume and diversity."""
    df = _get_data()

    def _bio_stats(sub):
        totals, uniques = [], []
        for bio_seq in sub["bio_seq"]:
            if not isinstance(bio_seq, list):
                totals.append(0); uniques.append(0); continue
            totals.append(len(bio_seq))
            uniques.append(len(set(bio_seq)))
        return np.mean(totals), np.mean(uniques)

    # Baselines as reference points (no text labels — shared legend handles ID)
    for cond, mk, ms in [("Human Expert", "^", 70), ("Hardcoded Pipeline", "s", 55)]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        tc, ut = _bio_stats(sub)
        c = CONDITION_COLORS[cond]
        ax.scatter(ut, tc, s=ms, marker=mk, facecolors=c,
                   edgecolors="white", linewidths=0.5, zorder=6)

    # LLM arrows: BM (open) → User (filled), no per-model labels
    for model in MODELS:
        bm_df = df[df["condition"] == model["bm"]]
        us_df = df[df["condition"] == model["user"]]
        if bm_df.empty or us_df.empty:
            continue

        tc_bm, ut_bm = _bio_stats(bm_df)
        tc_us, ut_us = _bio_stats(us_df)
        c = model["color"]

        # BM point (open circle)
        ax.scatter(ut_bm, tc_bm, s=40, marker="o", facecolors="white",
                   edgecolors=c, linewidths=0.8, zorder=5)
        # User point (filled circle)
        ax.scatter(ut_us, tc_us, s=55, marker="o", facecolors=c,
                   edgecolors="white", linewidths=0.4, zorder=6)

        # Arrow BM → User
        dx, dy = ut_us - ut_bm, tc_us - tc_bm
        if abs(dx) > 0.05 or abs(dy) > 0.05:
            ax.annotate("", xy=(ut_us, tc_us), xytext=(ut_bm, tc_bm),
                        arrowprops=dict(arrowstyle="-|>", color=c,
                                        lw=1.5, shrinkA=4, shrinkB=4),
                        zorder=4)

    ax.set_xlabel("Avg unique bio tools / task", fontsize=_MIN_PT)
    ax.set_ylabel("Avg bio tool calls / task", fontsize=_MIN_PT)
    _style(ax)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    fig_w = 180 / 25.4   # 7.09 in
    fig_h = 120 / 25.4   # 4.72 in  (180×120 mm, 3:2 ratio)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # 3 rows: top panels, bottom panels, shared legend at bottom
    gs = fig.add_gridspec(3, 2,
                          width_ratios=[0.55, 0.45],
                          height_ratios=[0.46, 0.46, 0.08],
                          hspace=0.45, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)

    # ── Shared legend (full-width bottom row) ─────────────────────────
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    shared_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#AA3377",
               markeredgecolor="white", markersize=5.5, label="Expert"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#EE7733",
               markeredgecolor="white", markersize=5, label="Hardcoded"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4477AA",
               markeredgecolor="white", markersize=5, label="DeepSeek V3"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#EE6677",
               markeredgecolor="white", markersize=5, label="GPT-5"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#228833",
               markeredgecolor="white", markersize=5, label="Sonnet 4.5"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#66CCEE",
               markeredgecolor="white", markersize=5, label="Gemini 2.5 Pro"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#444", markersize=5, label="(open = BM)"),
        Line2D([0], [0], color="#888", lw=1.5, marker=">", markersize=4,
               markerfacecolor="#888", label="BM → User"),
    ]
    ax_leg.legend(handles=shared_handles, loc="center", ncol=4,
                  fontsize=_MIN_PT, frameon=False,
                  handletextpad=0.3, columnspacing=1.0)

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.04, top=0.96)

    # Center legend on full figure width
    p = ax_leg.get_position()
    leg_w = 0.80
    ax_leg.set_position([0.5 - leg_w / 2, p.y0, leg_w, p.height])

    png = OUT / "fig6_tool_analysis.png"
    pdf = OUT / "fig6_tool_analysis.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close()
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Size: {fig_w:.2f}\" x {fig_h:.2f}\" ({fig_w*25.4:.0f}mm x {fig_h*25.4:.0f}mm)")


if __name__ == "__main__":
    main()
