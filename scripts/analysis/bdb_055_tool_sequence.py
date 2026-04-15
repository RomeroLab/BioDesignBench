#!/usr/bin/env python3
"""BDB-055: Tool call chain patterns and pipeline adherence analysis.

Analyzes how agents sequence their tool calls, whether they follow
canonical protein design pipelines, and how tool usage patterns
correlate with scores.

Analyses:
  1. Pipeline adherence rate per condition
  2. High vs low score comparison (top/bottom quartile tool usage)
  3. Failed tool call analysis and correlation with score
  4. BM vs US chain length comparison per LLM

Outputs (saved to results/analysis/):
    - pipeline_adherence_table.csv
    - high_vs_low_comparison.csv
    - failed_tool_analysis.csv
    - tool_sequence_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

# ---------------------------------------------------------------------------
# Pipeline stage mapping
# ---------------------------------------------------------------------------

# Map individual tool names to canonical pipeline stages.
# A tool may belong to multiple stages (e.g., rosetta_design does both
# backbone generation and sequence design).
TOOL_TO_STAGE: dict[str, list[str]] = {
    "generate_backbone": ["backbone_generation"],
    "rosetta_design": ["backbone_generation", "sequence_design"],
    "optimize_sequence": ["sequence_design"],
    "design_binder": ["sequence_design"],
    "predict_structure": ["structure_prediction"],
    "predict_structure_boltz": ["structure_prediction"],
    "predict_complex": ["structure_prediction"],
    "score_stability": ["scoring"],
    "analyze_interface": ["scoring"],
    "rosetta_score": ["scoring"],
    "rosetta_interface_score": ["scoring"],
    "predict_affinity_boltz": ["scoring"],
}

# Canonical stage order (index = expected position in pipeline).
CANONICAL_ORDER = [
    "backbone_generation",
    "sequence_design",
    "structure_prediction",
    "scoring",
]


# ---------------------------------------------------------------------------
# 1. Pipeline adherence
# ---------------------------------------------------------------------------


def _stages_from_tools(tool_sequence: list[str]) -> list[str]:
    """Convert a list of tool names into an ordered list of pipeline stages.

    Each tool call is mapped to its stage(s).  Consecutive duplicates are
    collapsed so that we get a clean stage progression.

    Args:
        tool_sequence: List of tool names in call order.

    Returns:
        Deduplicated stage list preserving first-occurrence order.
    """
    stages: list[str] = []
    for tool in tool_sequence:
        mapped = TOOL_TO_STAGE.get(tool, [])
        for stage in mapped:
            if not stages or stages[-1] != stage:
                stages.append(stage)
    return stages


def _is_pipeline_adherent(stages: list[str]) -> bool:
    """Check whether stages appear in canonical order.

    Stages may be skipped (not every task requires all four) but the
    relative order of those that *do* appear must match the canonical
    sequence.

    Args:
        stages: Ordered list of pipeline stages for a single task.

    Returns:
        True if the observed order is consistent with the canonical order.
    """
    if not stages:
        return False

    # Build index list based on canonical order
    indices: list[int] = []
    for stage in stages:
        if stage in CANONICAL_ORDER:
            indices.append(CANONICAL_ORDER.index(stage))

    if not indices:
        return False

    # Check monotonically non-decreasing (allows repeated stages)
    return all(a <= b for a, b in zip(indices, indices[1:]))


def compute_pipeline_adherence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pipeline adherence metrics per condition.

    Args:
        df: Full DataFrame from load_all() with tool_sequence column.

    Returns:
        DataFrame with columns: condition, adherence_rate, mean_chain_length,
        mean_unique_tools.
    """
    rows: list[dict] = []

    for condition, grp in df.groupby("condition", observed=True):
        stages_list = grp["tool_sequence"].apply(_stages_from_tools)
        adherent = stages_list.apply(_is_pipeline_adherent)

        adherence_rate = adherent.mean() if len(grp) > 0 else 0.0
        mean_chain = grp["num_tool_calls"].mean()
        mean_unique = grp["tool_sequence"].apply(lambda x: len(set(x))).mean()

        rows.append(
            {
                "condition": str(condition),
                "adherence_rate": round(adherence_rate, 4),
                "mean_chain_length": round(mean_chain, 2),
                "mean_unique_tools": round(mean_unique, 2),
            }
        )

    result = pd.DataFrame(rows).sort_values("adherence_rate", ascending=False)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. High vs low score comparison
# ---------------------------------------------------------------------------


def compute_high_vs_low(df: pd.DataFrame) -> pd.DataFrame:
    """Compare tool usage between top and bottom quartile tasks per condition.

    For each condition the tasks are split into top quartile (>=75th
    percentile of total score) and bottom quartile (<=25th percentile).
    Then mean chain length, unique tools, and iteration rate are compared.

    Args:
        df: Full DataFrame from load_all().

    Returns:
        DataFrame with per-condition high/low comparison.
    """
    rows: list[dict] = []

    for condition, grp in df.groupby("condition", observed=True):
        q75 = grp["total"].quantile(0.75)
        q25 = grp["total"].quantile(0.25)

        high = grp[grp["total"] >= q75]
        low = grp[grp["total"] <= q25]

        if len(high) == 0 or len(low) == 0:
            continue

        def _has_iteration(seq: list[str]) -> bool:
            """Check if any tool name appears more than once."""
            return len(seq) != len(set(seq))

        rows.append(
            {
                "condition": str(condition),
                "high_n": len(high),
                "low_n": len(low),
                "high_mean_chain_length": round(high["num_tool_calls"].mean(), 2),
                "low_mean_chain_length": round(low["num_tool_calls"].mean(), 2),
                "high_mean_unique_tools": round(
                    high["tool_sequence"].apply(lambda x: len(set(x))).mean(), 2
                ),
                "low_mean_unique_tools": round(
                    low["tool_sequence"].apply(lambda x: len(set(x))).mean(), 2
                ),
                "high_iteration_rate": round(
                    high["tool_sequence"].apply(_has_iteration).mean(), 4
                ),
                "low_iteration_rate": round(
                    low["tool_sequence"].apply(_has_iteration).mean(), 4
                ),
                "high_mean_score": round(high["total"].mean(), 2),
                "low_mean_score": round(low["total"].mean(), 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Failed tool call analysis
# ---------------------------------------------------------------------------


def compute_failed_tool_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze failed tool calls per condition and their correlation with score.

    Args:
        df: Full DataFrame from load_all() with failed_tools column.

    Returns:
        DataFrame with per-condition failed tool metrics.
    """
    rows: list[dict] = []

    for condition, grp in df.groupby("condition", observed=True):
        mean_failed = grp["failed_tools"].mean()
        total_calls = grp["num_tool_calls"].sum()
        total_failed = grp["failed_tools"].sum()
        failed_ratio = total_failed / total_calls if total_calls > 0 else 0.0

        # Correlation between failed_tools and total score
        if grp["failed_tools"].std() > 0 and grp["total"].std() > 0:
            r, p = stats.pearsonr(grp["failed_tools"], grp["total"])
        else:
            r, p = 0.0, 1.0

        rows.append(
            {
                "condition": str(condition),
                "mean_failed_tools": round(mean_failed, 3),
                "failed_tool_ratio": round(failed_ratio, 4),
                "total_tool_calls": int(total_calls),
                "total_failed_calls": int(total_failed),
                "pearson_r_failed_vs_score": round(r, 4),
                "p_value": round(p, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_failed_tools", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. BM vs US chain comparison
# ---------------------------------------------------------------------------


def compute_bm_vs_us_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean chain length between benchmark and user mode per LLM.

    Args:
        df: Full DataFrame from load_all() with mode and llm columns.

    Returns:
        DataFrame with per-LLM chain length comparison.
    """
    paired = df[df["mode"].isin(["benchmark", "user"])].copy()

    if len(paired) == 0:
        return pd.DataFrame()

    rows: list[dict] = []

    for llm, grp in paired.groupby("llm"):
        bm = grp[grp["mode"] == "benchmark"]
        us = grp[grp["mode"] == "user"]

        if len(bm) == 0 or len(us) == 0:
            continue

        bm_chain = bm["num_tool_calls"].mean()
        us_chain = us["num_tool_calls"].mean()
        bm_unique = bm["tool_sequence"].apply(lambda x: len(set(x))).mean()
        us_unique = us["tool_sequence"].apply(lambda x: len(set(x))).mean()
        bm_score = bm["total"].mean()
        us_score = us["total"].mean()

        rows.append(
            {
                "llm": str(llm),
                "bm_mean_chain_length": round(bm_chain, 2),
                "us_mean_chain_length": round(us_chain, 2),
                "chain_delta": round(us_chain - bm_chain, 2),
                "bm_mean_unique_tools": round(bm_unique, 2),
                "us_mean_unique_tools": round(us_unique, 2),
                "unique_delta": round(us_unique - bm_unique, 2),
                "bm_mean_score": round(bm_score, 2),
                "us_mean_score": round(us_score, 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def write_summary(
    df: pd.DataFrame,
    adherence: pd.DataFrame,
    high_low: pd.DataFrame,
    failed: pd.DataFrame,
    bm_us: pd.DataFrame,
    out_path: Path,
) -> str:
    """Write a plain-text summary of all tool sequence analyses.

    Args:
        df: Full DataFrame.
        adherence: Pipeline adherence table.
        high_low: High vs low comparison table.
        failed: Failed tool analysis table.
        bm_us: BM vs US chain comparison table.
        out_path: Path to write the summary text file.

    Returns:
        The summary text as a string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("BDB-055: Tool Call Chain Patterns & Pipeline Adherence")
    lines.append("=" * 70)
    lines.append("")

    # Dataset overview
    n_tasks = df["task_id"].nunique()
    n_conditions = df["condition"].nunique()
    n_rows = len(df)
    lines.append(f"Dataset: {n_rows} rows ({n_tasks} tasks x {n_conditions} conditions)")
    lines.append(f"Total tool calls across all rows: {df['num_tool_calls'].sum()}")
    lines.append(f"Mean chain length: {df['num_tool_calls'].mean():.2f}")
    lines.append(f"Mean unique tools per task: {df['tool_sequence'].apply(lambda x: len(set(x))).mean():.2f}")
    lines.append("")

    # 1. Pipeline adherence
    lines.append("-" * 70)
    lines.append("1. PIPELINE ADHERENCE")
    lines.append("-" * 70)
    lines.append("")
    lines.append("Canonical pipeline: backbone_generation -> sequence_design")
    lines.append("                    -> structure_prediction -> scoring")
    lines.append("")
    lines.append(
        f"  {'Condition':<35s} {'Adherence':>10s} {'Chain len':>10s} {'Uniq tools':>11s}"
    )
    lines.append("  " + "-" * 68)
    for _, row in adherence.iterrows():
        lines.append(
            f"  {row['condition']:<35s} "
            f"{row['adherence_rate']:>9.1%} "
            f"{row['mean_chain_length']:>10.2f} "
            f"{row['mean_unique_tools']:>11.2f}"
        )

    best = adherence.iloc[0]
    worst = adherence.iloc[-1]
    lines.append("")
    lines.append(f"  Best adherence:  {best['condition']} ({best['adherence_rate']:.1%})")
    lines.append(f"  Worst adherence: {worst['condition']} ({worst['adherence_rate']:.1%})")
    lines.append("")

    # 2. High vs low score comparison
    lines.append("-" * 70)
    lines.append("2. HIGH vs LOW SCORE TOOL USAGE")
    lines.append("-" * 70)
    lines.append("")
    lines.append("  Comparing top quartile (>=P75) vs bottom quartile (<=P25) per condition.")
    lines.append("")

    if len(high_low) > 0:
        lines.append(
            f"  {'Condition':<35s} "
            f"{'High chain':>11s} {'Low chain':>10s} "
            f"{'High uniq':>10s} {'Low uniq':>9s} "
            f"{'High iter':>10s} {'Low iter':>9s}"
        )
        lines.append("  " + "-" * 96)
        for _, row in high_low.iterrows():
            lines.append(
                f"  {row['condition']:<35s} "
                f"{row['high_mean_chain_length']:>11.2f} {row['low_mean_chain_length']:>10.2f} "
                f"{row['high_mean_unique_tools']:>10.2f} {row['low_mean_unique_tools']:>9.2f} "
                f"{row['high_iteration_rate']:>9.1%} {row['low_iteration_rate']:>9.1%}"
            )

        # Aggregate insight
        mean_high_chain = high_low["high_mean_chain_length"].mean()
        mean_low_chain = high_low["low_mean_chain_length"].mean()
        mean_high_iter = high_low["high_iteration_rate"].mean()
        mean_low_iter = high_low["low_iteration_rate"].mean()
        lines.append("")
        lines.append(f"  Avg chain length -- High: {mean_high_chain:.2f}, Low: {mean_low_chain:.2f}")
        lines.append(f"  Avg iteration rate -- High: {mean_high_iter:.1%}, Low: {mean_low_iter:.1%}")
        if mean_high_chain > mean_low_chain:
            lines.append("  --> Higher-scoring tasks tend to use LONGER tool chains.")
        elif mean_high_chain < mean_low_chain:
            lines.append("  --> Higher-scoring tasks tend to use SHORTER tool chains.")
        else:
            lines.append("  --> No systematic chain length difference.")
    else:
        lines.append("  No data available for high/low comparison.")
    lines.append("")

    # 3. Failed tool analysis
    lines.append("-" * 70)
    lines.append("3. FAILED TOOL CALL ANALYSIS")
    lines.append("-" * 70)
    lines.append("")
    lines.append(
        f"  {'Condition':<35s} "
        f"{'Mean failed':>12s} {'Fail ratio':>11s} "
        f"{'r(fail,score)':>14s} {'p-value':>8s}"
    )
    lines.append("  " + "-" * 82)
    for _, row in failed.iterrows():
        sig = "*" if row["p_value"] < 0.05 else " "
        lines.append(
            f"  {row['condition']:<35s} "
            f"{row['mean_failed_tools']:>12.3f} "
            f"{row['failed_tool_ratio']:>10.2%} "
            f"{row['pearson_r_failed_vs_score']:>14.3f} "
            f"{row['p_value']:>7.4f}{sig}"
        )

    # Overall correlation
    if df["failed_tools"].std() > 0 and df["total"].std() > 0:
        r_all, p_all = stats.pearsonr(df["failed_tools"], df["total"])
        lines.append("")
        sig_str = "significant" if p_all < 0.05 else "not significant"
        lines.append(
            f"  Overall correlation (all rows): r={r_all:.3f}, p={p_all:.4f} ({sig_str})"
        )
        if r_all < 0:
            lines.append("  --> More failed tools are associated with LOWER scores.")
        elif r_all > 0:
            lines.append("  --> More failed tools are associated with HIGHER scores.")
    lines.append("")

    # 4. BM vs US chain comparison
    lines.append("-" * 70)
    lines.append("4. BENCHMARK vs USER MODE CHAIN LENGTH")
    lines.append("-" * 70)
    lines.append("")

    if len(bm_us) > 0:
        lines.append(
            f"  {'LLM':<20s} "
            f"{'BM chain':>9s} {'US chain':>9s} {'Delta':>7s} "
            f"{'BM uniq':>8s} {'US uniq':>8s} "
            f"{'BM score':>9s} {'US score':>9s}"
        )
        lines.append("  " + "-" * 81)
        for _, row in bm_us.iterrows():
            sign = "+" if row["chain_delta"] >= 0 else ""
            lines.append(
                f"  {row['llm']:<20s} "
                f"{row['bm_mean_chain_length']:>9.2f} "
                f"{row['us_mean_chain_length']:>9.2f} "
                f"{sign}{row['chain_delta']:>6.2f} "
                f"{row['bm_mean_unique_tools']:>8.2f} "
                f"{row['us_mean_unique_tools']:>8.2f} "
                f"{row['bm_mean_score']:>9.2f} "
                f"{row['us_mean_score']:>9.2f}"
            )

        mean_delta = bm_us["chain_delta"].mean()
        lines.append("")
        if mean_delta > 0:
            lines.append(
                f"  User mode leads to LONGER chains on average "
                f"(+{mean_delta:.2f} calls per task)."
            )
        elif mean_delta < 0:
            lines.append(
                f"  User mode leads to SHORTER chains on average "
                f"({mean_delta:.2f} calls per task)."
            )
        else:
            lines.append("  No systematic chain length difference between modes.")
    else:
        lines.append("  No paired BM/US data available.")
    lines.append("")

    # Key takeaways
    lines.append("=" * 70)
    lines.append("KEY TAKEAWAYS")
    lines.append("=" * 70)
    lines.append("")
    mean_adherence = adherence["adherence_rate"].mean()
    lines.append(
        f"  1. Mean pipeline adherence across conditions: {mean_adherence:.1%}"
    )
    if len(high_low) > 0:
        chain_diff = (
            high_low["high_mean_chain_length"].mean()
            - high_low["low_mean_chain_length"].mean()
        )
        lines.append(
            f"  2. Top-quartile tasks use {abs(chain_diff):.1f} "
            f"{'more' if chain_diff > 0 else 'fewer'} tool calls than bottom-quartile"
        )
    total_failed = df["failed_tools"].sum()
    total_calls = df["num_tool_calls"].sum()
    lines.append(
        f"  3. Overall tool failure rate: "
        f"{total_failed}/{total_calls} "
        f"({total_failed / total_calls:.1%} of calls)" if total_calls > 0 else
        "  3. No tool calls recorded."
    )
    if len(bm_us) > 0:
        lines.append(
            f"  4. BM->US mean chain delta: {bm_us['chain_delta'].mean():+.2f} calls/task"
        )
    lines.append("")

    summary = "\n".join(lines)
    out_path.write_text(summary + "\n")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all tool sequence analyses and save outputs."""
    print("Loading data...")
    df = load_all()
    print(
        f"  Loaded {len(df)} rows "
        f"({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)"
    )
    print(f"  Total tool calls: {df['num_tool_calls'].sum()}")
    print(f"  Mean chain length: {df['num_tool_calls'].mean():.2f}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Pipeline adherence ────────────────────────────────────────────
    print("=" * 60)
    print("1. PIPELINE ADHERENCE")
    print("=" * 60)
    adherence = compute_pipeline_adherence(df)
    adherence.to_csv(OUTPUT_DIR / "pipeline_adherence_table.csv", index=False)
    print(adherence.to_string(index=False))
    print()

    # ── 2. High vs low score comparison ──────────────────────────────────
    print("=" * 60)
    print("2. HIGH vs LOW SCORE COMPARISON")
    print("=" * 60)
    high_low = compute_high_vs_low(df)
    high_low.to_csv(OUTPUT_DIR / "high_vs_low_comparison.csv", index=False)
    print(high_low.to_string(index=False))
    print()

    # ── 3. Failed tool analysis ──────────────────────────────────────────
    print("=" * 60)
    print("3. FAILED TOOL CALL ANALYSIS")
    print("=" * 60)
    failed = compute_failed_tool_analysis(df)
    failed.to_csv(OUTPUT_DIR / "failed_tool_analysis.csv", index=False)
    print(failed.to_string(index=False))
    print()

    # ── 4. BM vs US chain comparison ─────────────────────────────────────
    print("=" * 60)
    print("4. BM vs US CHAIN COMPARISON")
    print("=" * 60)
    bm_us = compute_bm_vs_us_chains(df)
    if len(bm_us) > 0:
        print(bm_us.to_string(index=False))
    else:
        print("  No paired BM/US data available.")
    print()

    # ── Summary text ─────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "tool_sequence_summary.txt"
    summary = write_summary(df, adherence, high_low, failed, bm_us, summary_path)
    print(summary)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
