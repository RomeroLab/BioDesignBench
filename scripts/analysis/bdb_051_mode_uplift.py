#!/usr/bin/env python3
"""BDB-051: Benchmark -> User mode uplift per scoring component.

Analyzes how switching from benchmark mode (minimal system prompt, 14 tools)
to user mode (rich system prompt, 17 tools) affects each scoring component
across 4 LLMs with paired data.

Outputs (saved to results/analysis/):
    - mode_uplift_per_component.csv
    - mode_uplift_per_category.csv
    - negative_uplift_tasks.csv
    - mode_uplift_histogram.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]
SCORE_COLS = COMPONENTS + ["total"]
LLM_COLORS = {
    "DeepSeek V3": "#1f77b4",
    "GPT-5": "#ff7f0e",
    "Sonnet 4.5": "#2ca02c",
    "Gemini 2.5 Pro": "#d62728",
}
LEGACY_CATEGORIES = ["Binder", "SeqOpt", "CpxPrd", "ConfDv", "Backbone", "AbBody", "PPI"]


def _build_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Merge benchmark and user rows on (task_id, llm) to produce paired rows.

    Args:
        df: Full DataFrame from load_all() with mode, llm, task_id columns.

    Returns:
        DataFrame with bm_* and us_* prefixed score columns for each pair.
    """
    bm = df[df["mode"] == "benchmark"].copy()
    us = df[df["mode"] == "user"].copy()

    bm_cols = {col: f"bm_{col}" for col in SCORE_COLS}
    us_cols = {col: f"us_{col}" for col in SCORE_COLS}

    bm = bm.rename(columns=bm_cols)
    us = us.rename(columns=us_cols)

    merge_keys = ["task_id", "llm"]
    keep_from_bm = merge_keys + list(bm_cols.values()) + ["legacy_category", "difficulty"]
    keep_from_us = merge_keys + list(us_cols.values())

    paired = pd.merge(
        bm[keep_from_bm],
        us[keep_from_us],
        on=merge_keys,
        how="inner",
    )
    return paired


def compute_per_component_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean delta (user - benchmark) per component per LLM.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with columns: llm, approach_delta, ..., total_delta.
    """
    paired = _build_paired(df)

    for col in SCORE_COLS:
        paired[f"{col}_delta"] = paired[f"us_{col}"] - paired[f"bm_{col}"]

    delta_cols = [f"{col}_delta" for col in SCORE_COLS]
    result = paired.groupby("llm")[delta_cols].mean().reset_index()
    result.columns = ["llm"] + delta_cols
    return result


def compute_per_category_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean BM->US total delta per legacy category across all LLMs.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with columns: legacy_category, total_delta, plus per-component deltas.
    """
    paired = _build_paired(df)

    for col in SCORE_COLS:
        paired[f"{col}_delta"] = paired[f"us_{col}"] - paired[f"bm_{col}"]

    delta_cols = [f"{col}_delta" for col in SCORE_COLS]
    result = paired.groupby("legacy_category")[delta_cols].mean().reset_index()
    return result


def find_negative_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Find tasks where user mode total score < benchmark mode total score.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with columns: task_id, llm, bm_score, us_score, delta, legacy_category.
    """
    paired = _build_paired(df)
    paired["delta"] = paired["us_total"] - paired["bm_total"]

    neg = paired[paired["delta"] < 0].copy()
    neg = neg.rename(columns={"bm_total": "bm_score", "us_total": "us_score"})
    neg = neg[["task_id", "llm", "bm_score", "us_score", "delta", "legacy_category"]]
    neg = neg.sort_values(["delta", "task_id", "llm"]).reset_index(drop=True)
    return neg


def plot_uplift_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-task BM->US delta histogram with 4 LLMs overlaid.

    Args:
        df: Full DataFrame from load_all().
        out_path: Path to save the PNG figure.
    """
    paired = _build_paired(df)
    paired["delta"] = paired["us_total"] - paired["bm_total"]

    fig, ax = plt.subplots(figsize=(10, 6))

    llms = sorted(paired["llm"].unique())
    all_deltas = paired["delta"]
    bins = np.linspace(all_deltas.min() - 1, all_deltas.max() + 1, 30)

    for llm in llms:
        subset = paired[paired["llm"] == llm]["delta"]
        color = LLM_COLORS.get(llm, "#888888")
        ax.hist(subset, bins=bins, alpha=0.5, label=llm, color=color, edgecolor="white")

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("BM -> US Total Score Delta", fontsize=12)
    ax.set_ylabel("Task Count", fontsize=12)
    ax.set_title("Benchmark -> User Mode Uplift Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run all analyses and save outputs."""
    from scripts.analysis.load_results import load_all

    print("Loading results...")
    df = load_all()
    print(f"  Loaded {len(df)} rows ({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)")

    # Filter to paired modes only
    paired_llms = df[df["mode"].isin(["benchmark", "user"])]["llm"].unique()
    print(f"  LLMs with paired BM/US data: {sorted(paired_llms)}")

    out_dir = PROJECT_ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Per-component uplift ──────────────────────────────────────────────
    print("\n=== Per-Component Uplift (User - Benchmark) ===")
    comp_uplift = compute_per_component_uplift(df)
    comp_uplift.to_csv(out_dir / "mode_uplift_per_component.csv", index=False)
    print(comp_uplift.to_string(index=False, float_format="%.2f"))

    # Key finding: which component benefits most?
    delta_cols = [c for c in comp_uplift.columns if c.endswith("_delta")]
    mean_deltas = comp_uplift[delta_cols].mean()
    best_comp = mean_deltas.idxmax().replace("_delta", "")
    worst_comp = mean_deltas.idxmin().replace("_delta", "")
    print(f"\n  Largest average uplift:  {best_comp} (+{mean_deltas.max():.2f})")
    print(f"  Smallest average uplift: {worst_comp} ({mean_deltas.min():+.2f})")

    # ── 2. Per-category uplift ───────────────────────────────────────────────
    print("\n=== Per-Category Uplift (User - Benchmark) ===")
    cat_uplift = compute_per_category_uplift(df)
    cat_uplift.to_csv(out_dir / "mode_uplift_per_category.csv", index=False)
    print(cat_uplift.to_string(index=False, float_format="%.2f"))

    best_cat = cat_uplift.loc[cat_uplift["total_delta"].idxmax(), "legacy_category"]
    worst_cat = cat_uplift.loc[cat_uplift["total_delta"].idxmin(), "legacy_category"]
    print(f"\n  Most improved category:  {best_cat} (+{cat_uplift['total_delta'].max():.2f})")
    print(f"  Least improved category: {worst_cat} ({cat_uplift['total_delta'].min():+.2f})")

    # ── 3. Negative uplift tasks ─────────────────────────────────────────────
    print("\n=== Negative Uplift Tasks (User < Benchmark) ===")
    neg = find_negative_uplift(df)
    neg.to_csv(out_dir / "negative_uplift_tasks.csv", index=False)
    print(f"  Total negative uplift cases: {len(neg)}")
    if len(neg) > 0:
        print(f"  Unique tasks affected: {neg['task_id'].nunique()}")
        print(f"  By LLM:")
        for llm, count in neg["llm"].value_counts().items():
            print(f"    {llm}: {count} tasks")
        print(f"  By category:")
        for cat, count in neg["legacy_category"].value_counts().items():
            print(f"    {cat}: {count} tasks")
        print(f"\n  Worst cases (top 10):")
        print(neg.head(10).to_string(index=False, float_format="%.1f"))

    # ── 4. Histogram ─────────────────────────────────────────────────────────
    hist_path = out_dir / "mode_uplift_histogram.png"
    plot_uplift_histogram(df, hist_path)
    print(f"\n  Histogram saved to {hist_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    paired = _build_paired(df)
    paired["delta"] = paired["us_total"] - paired["bm_total"]
    n_positive = (paired["delta"] > 0).sum()
    n_zero = (paired["delta"] == 0).sum()
    n_negative = (paired["delta"] < 0).sum()
    n_total = len(paired)
    print(f"  Positive uplift: {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)")
    print(f"  Zero uplift:     {n_zero}/{n_total} ({100*n_zero/n_total:.1f}%)")
    print(f"  Negative uplift: {n_negative}/{n_total} ({100*n_negative/n_total:.1f}%)")
    print(f"  Mean total delta: {paired['delta'].mean():+.2f}")
    print(f"  Median total delta: {paired['delta'].median():+.2f}")

    print("\nDone. Outputs saved to results/analysis/")


if __name__ == "__main__":
    main()
