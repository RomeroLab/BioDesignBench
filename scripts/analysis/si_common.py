#!/usr/bin/env python3
"""Shared style & utilities for BioDesignBench Supplementary Figures.

All SI figures import from here to ensure consistent:
  - Figure size (120×80mm, 300 dpi)
  - Font (Arial ≥ 7pt)
  - Color palette (LLM-specific + heatmap scales)
  - Panel labels, save helpers, condition metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────
MIN_PT = 7
FIG_W_MM, FIG_H_MM = 120, 80
FIG_W = FIG_W_MM / 25.4   # ≈ 4.72 in
FIG_H = FIG_H_MM / 25.4   # ≈ 3.15 in


def setup_style():
    """Apply Nature-style rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": MIN_PT,
        "axes.linewidth": 0.5,
        "axes.labelsize": MIN_PT,
        "axes.titlesize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.labelsize": MIN_PT,
        "ytick.labelsize": MIN_PT,
        "legend.fontsize": MIN_PT,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


setup_style()

# ── Color palette (Paul Tol bright, 7 fully distinct colors) ──────────
# Baselines: gold, purple, gray — distinct from all model colors
# Models:    blue, coral, green, cyan — no overlap with baselines
LLM_COLORS = {
    "Oracle":         "#DDAA33",  # gold/amber
    "Human Expert":   "#AA3377",  # purple/magenta
    "Hardcoded":      "#EE7733",  # orange
    "DeepSeek V3":    "#4477AA",  # blue
    "GPT-5":          "#EE6677",  # coral/red
    "Sonnet 4.5":     "#228833",  # green
    "Gemini 2.5 Pro": "#66CCEE",  # cyan
}

CONDITION_COLORS = {
    "Oracle":                   "#DDAA33",
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

# Canonical order: baselines first (comparison targets), then LLMs by score
CONDITION_ORDER = [
    "Oracle",
    "Human Expert",
    "Hardcoded Pipeline",
    "DeepSeek V3 user", "DeepSeek V3 benchmark",
    "GPT-5 user", "GPT-5 benchmark",
    "Sonnet 4.5 user", "Sonnet 4.5 benchmark",
    "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark",
]

# Without Oracle — for analytical figures where Oracle (87.3) compresses the scale
CONDITION_ORDER_NO_ORACLE = [c for c in CONDITION_ORDER if c != "Oracle"]

CONDITION_SHORT = {
    "Oracle":                   "Oracle",
    "Human Expert":             "Expert",
    "DeepSeek V3 user":         "DS V3 U",
    "DeepSeek V3 benchmark":    "DS V3 B",
    "GPT-5 user":               "GPT-5 U",
    "GPT-5 benchmark":          "GPT-5 B",
    "Sonnet 4.5 user":          "Son 4.5 U",
    "Sonnet 4.5 benchmark":     "Son 4.5 B",
    "Gemini 2.5 Pro user":      "Gem 2.5 U",
    "Gemini 2.5 Pro benchmark": "Gem 2.5 B",
    "Hardcoded Pipeline":       "Hardcoded",
}

# ── Components ────────────────────────────────────────────────────────
COMPONENTS = ["approach", "orchestration", "quality",
              "feasibility", "novelty", "diversity"]
COMP_LABELS = {
    "approach": "Approach", "orchestration": "Orchestration",
    "quality": "Quality", "feasibility": "Feasibility",
    "novelty": "Novelty", "diversity": "Diversity",
}
MAX_POINTS = {
    "approach": 20, "orchestration": 15, "quality": 35,
    "feasibility": 15, "novelty": 5, "diversity": 10,
}

# ── 17 MCP Tools ──────────────────────────────────────────────────────
ALL_TOOLS = [
    "design_binder", "analyze_interface", "validate_design",
    "optimize_sequence", "suggest_hotspots", "get_design_status",
    "predict_complex", "predict_structure", "score_stability",
    "energy_minimize", "generate_backbone", "rosetta_score",
    "rosetta_relax", "rosetta_interface_score", "rosetta_design",
    "predict_structure_boltz", "predict_affinity_boltz",
]

TOOL_SHORT = {
    "design_binder": "design_binder",
    "analyze_interface": "analyze_iface",
    "validate_design": "validate",
    "optimize_sequence": "opt_seq",
    "suggest_hotspots": "hotspots",
    "get_design_status": "status",
    "predict_complex": "pred_complex",
    "predict_structure": "pred_struct",
    "score_stability": "stability",
    "energy_minimize": "minimize",
    "generate_backbone": "gen_backbone",
    "rosetta_score": "ros_score",
    "rosetta_relax": "ros_relax",
    "rosetta_interface_score": "ros_iface",
    "rosetta_design": "ros_design",
    "predict_structure_boltz": "boltz_struct",
    "predict_affinity_boltz": "boltz_aff",
}

# ── Canonical pipeline steps (for adherence checking) ─────────────────
CANONICAL_PIPELINE = [
    "generate_backbone",  # backbone generation
    "design_binder",      # OR optimize_sequence → sequence design
    "predict_structure",  # OR predict_complex → structure prediction
    "score_stability",    # OR analyze_interface → scoring
]

PIPELINE_STEP_MAP = {
    "generate_backbone": "backbone",
    "design_binder": "seq_design",
    "optimize_sequence": "seq_design",
    "rosetta_design": "seq_design",
    "predict_structure": "struct_pred",
    "predict_complex": "struct_pred",
    "predict_structure_boltz": "struct_pred",
    "predict_affinity_boltz": "struct_pred",
    "score_stability": "scoring",
    "analyze_interface": "scoring",
    "rosetta_score": "scoring",
    "rosetta_interface_score": "scoring",
    "validate_design": "scoring",
}


# ── Helper functions ──────────────────────────────────────────────────

def panel_label(ax, label, x=-0.15, y=1.06):
    """Add bold (a)/(b)/(c) panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")


def save_fig(fig, name: str, tight: bool = True):
    """Save figure as PNG (300 dpi) + PDF (vector). Returns (png_path, pdf_path).

    Uses tight_layout to fit content within figure bounds, then saves at the
    figure's own size (no bbox_inches='tight') so the aspect ratio is preserved.
    """
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    svg = OUT / f"{name}.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {svg}")
    return png, pdf


def is_user(cond: str) -> bool:
    return "user" in cond.lower()


def is_bm(cond: str) -> bool:
    return "benchmark" in cond.lower()


def is_baseline(cond: str) -> bool:
    return cond in ("Oracle", "Human Expert", "Hardcoded Pipeline")


def cond_marker_kw(cond: str) -> dict:
    """Marker kwargs: filled for user/baselines, open for benchmark."""
    c = CONDITION_COLORS.get(cond, "#888888")
    if cond == "Oracle":
        return dict(marker="D", facecolors=c, edgecolors="white", linewidths=0.4)
    elif cond == "Human Expert":
        return dict(marker="^", facecolors=c, edgecolors="white", linewidths=0.4)
    elif "Hardcoded" in cond:
        return dict(marker="s", facecolors=c, edgecolors=c, linewidths=0.4)
    elif is_user(cond):
        return dict(marker="o", facecolors=c, edgecolors="white", linewidths=0.4)
    else:
        return dict(marker="o", facecolors="white", edgecolors=c, linewidths=0.8)


def cond_line_kw(cond: str) -> dict:
    """Line kwargs: solid user/baselines, dashed bm, dotted hardcoded."""
    c = CONDITION_COLORS.get(cond, "#888888")
    if cond == "Oracle":
        return dict(color=c, linestyle="-", linewidth=1.5)
    elif cond == "Human Expert":
        return dict(color=c, linestyle="-.", linewidth=1.2)
    elif "Hardcoded" in cond:
        return dict(color=c, linestyle=":", linewidth=0.8)
    elif is_user(cond):
        return dict(color=c, linestyle="-", linewidth=1.2)
    else:
        return dict(color=c, linestyle="--", linewidth=0.8)


def style_grid(ax, axis="both"):
    """Apply light gray gridlines (alpha=0.2)."""
    ax.grid(axis=axis, color="#cccccc", alpha=0.2, linewidth=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_raw_results() -> list[dict]:
    """Load all raw result.json files for detailed analysis (sequences, tool logs)."""
    from scripts.analysis.load_results import CONDITION_MAP, EXCLUDED_TASKS
    rows = []
    for condition, info in CONDITION_MAP.items():
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
            result["_condition"] = condition
            result["_mode"] = info["mode"]
            result["_llm"] = info["llm"]
            rows.append(result)
    return rows


# Re-export data loaders
from scripts.analysis.load_results import (
    load_all, load_score_matrix, load_component_matrix,
    CONDITION_MAP, EXCLUDED_TASKS,
)
