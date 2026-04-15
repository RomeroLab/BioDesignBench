#!/usr/bin/env python3
"""PROMPT 13: Benchmark vs User Mode Depth Comparison.

Does guidance rescue depth, or only binary coverage?
Compares guided (user) vs unguided (benchmark) mode using depth metrics from
PROMPT_11 (plan_exec_depth_v3) and PROMPT_12 (candidate_screening_v1).

Outputs:
  results/analysis/mode_depth_comparison_v1.csv             — per model × metric
  results/analysis/mode_depth_comparison_v1_by_task.csv      — per task × model × metric
  results/analysis/mode_depth_comparison_v1_by_category.csv  — per category × model × metric
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PAIRS = [
    ("DeepSeek V3",     "DeepSeek V3 user",     "DeepSeek V3 benchmark"),
    ("GPT-5",           "GPT-5 user",           "GPT-5 benchmark"),
    ("Sonnet 4.5",      "Sonnet 4.5 user",      "Sonnet 4.5 benchmark"),
    ("Gemini 2.5 Pro",  "Gemini 2.5 Pro user",  "Gemini 2.5 Pro benchmark"),
]

ALL_STAGES = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]

# Depth metrics to extract per task from plan_exec_depth_v3
DEPTH_METRICS_STAGE = {
    "backbone_gen_depth": "backbone_generation",
    "seqdesign_depth": "sequence_design",
    "structpred_depth": "structure_prediction",
    "scoring_depth": "scoring_validation",
}

# Binary coverage metrics (binary_exec per stage)
COVERAGE_METRICS_STAGE = {
    "backbone_gen_coverage": "backbone_generation",
    "seqdesign_coverage": "sequence_design",
    "structpred_coverage": "structure_prediction",
    "scoring_coverage": "scoring_validation",
}

# Candidate-level metrics from PROMPT_12
CANDIDATE_METRICS = [
    "n_candidates",
    "mean_eval_per_candidate",
    "mean_eval_diversity_per_candidate",
    "frac_multi_metric",
    "total_eval_calls",
    "funnel_depth",
]


# ---------------------------------------------------------------------------
# Data loading & pivoting
# ---------------------------------------------------------------------------

def load_depth_per_task(depth_csv: Path) -> pd.DataFrame:
    """Pivot plan_exec_depth_v3 from stage-level to task-level wide format.

    Returns DataFrame with columns:
        task_id, condition, approach, subject,
        total_exec_depth, backbone_gen_depth, seqdesign_depth, structpred_depth, scoring_depth,
        backbone_gen_coverage, seqdesign_coverage, structpred_coverage, scoring_coverage,
        eval_multiplier, mmvs, plan_specificity_mean
    """
    df = pd.read_csv(depth_csv)

    # Aggregate per (task_id, condition)
    rows = []
    for (tid, cond), grp in df.groupby(["task_id", "condition"]):
        approach = grp["approach"].iloc[0]
        subject = grp["subject"].iloc[0]

        row = {
            "task_id": tid,
            "condition": cond,
            "approach": approach,
            "subject": subject,
            "total_exec_depth": grp["exec_depth"].sum(),
            "eval_multiplier": grp["eval_multiplier"].iloc[0],
            "mmvs": grp["mmvs"].iloc[0],
            "plan_specificity_mean": grp["plan_specificity"].mean(),
        }

        for metric_name, stage in DEPTH_METRICS_STAGE.items():
            stage_row = grp[grp["stage"] == stage]
            row[metric_name] = stage_row["exec_depth"].sum() if len(stage_row) > 0 else 0

        for metric_name, stage in COVERAGE_METRICS_STAGE.items():
            stage_row = grp[grp["stage"] == stage]
            row[metric_name] = stage_row["binary_exec"].max() if len(stage_row) > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


def load_candidate_per_task(cand_csv: Path) -> pd.DataFrame:
    """Load candidate_screening_v1_task_summary.csv."""
    return pd.read_csv(cand_csv)


# ---------------------------------------------------------------------------
# Guidance Rescue Index
# ---------------------------------------------------------------------------

def rescue_index(expert_mean: float, guided_mean: float, unguided_mean: float) -> float:
    """Compute Guidance Rescue Index.

    0 = guidance doesn't reduce gap at all
    1 = guidance fully closes gap to expert
    <0 = guidance widens the gap
    """
    gap_unguided = expert_mean - unguided_mean
    if abs(gap_unguided) < 1e-9:
        return np.nan  # already at expert level
    gap_guided = expert_mean - guided_mean
    return 1 - (gap_guided / gap_unguided)


# ---------------------------------------------------------------------------
# Paired Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def _safe_wilcoxon(x, y):
    """Wilcoxon signed-rank with safe handling."""
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) < 5:
        return np.nan, np.nan, np.nan
    try:
        stat, p = wilcoxon(nonzero)
        # Effect size r = Z / sqrt(N)
        n = len(nonzero)
        z = abs((stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24))
        r = z / np.sqrt(n)
        return stat, p, r
    except ValueError:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def compute_model_metric_comparison(
    depth_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare guided vs unguided for each model × metric.

    Returns:
        summary_df: per model × metric aggregated stats
        task_df: per task × model × metric paired values
    """
    # Merge depth and candidate data
    merged = depth_df.copy()
    cand_cols = ["task_id", "condition"] + CANDIDATE_METRICS
    cand_subset = cand_df[cand_cols].copy()
    merged = merged.merge(cand_subset, on=["task_id", "condition"], how="left")

    # All metrics to compare
    all_metrics = [
        "total_exec_depth",
        "backbone_gen_depth",
        "seqdesign_depth",
        "structpred_depth",
        "scoring_depth",
        "eval_multiplier",
        "mmvs",
        "plan_specificity_mean",
        "backbone_gen_coverage",
        "seqdesign_coverage",
        "structpred_coverage",
        "scoring_coverage",
    ] + CANDIDATE_METRICS

    # Expert reference
    expert = merged[merged["condition"] == "Human Expert"]

    summary_rows = []
    task_rows = []

    for model, user_cond, bench_cond in MODEL_PAIRS:
        guided = merged[merged["condition"] == user_cond].set_index("task_id")
        unguided = merged[merged["condition"] == bench_cond].set_index("task_id")

        # Align on shared tasks
        shared_tasks = sorted(guided.index.intersection(unguided.index))
        if len(shared_tasks) == 0:
            continue

        g = guided.loc[shared_tasks]
        u = unguided.loc[shared_tasks]

        for metric in all_metrics:
            if metric not in g.columns or metric not in u.columns:
                continue

            g_vals = g[metric].fillna(0).values.astype(float)
            u_vals = u[metric].fillna(0).values.astype(float)

            mean_g = np.mean(g_vals)
            mean_u = np.mean(u_vals)
            delta = mean_g - mean_u

            # Paired Wilcoxon
            stat, p, r = _safe_wilcoxon(g_vals, u_vals)

            # Expert reference for rescue index
            expert_vals = expert[metric].fillna(0).values.astype(float) if metric in expert.columns else np.array([0.0])
            expert_mean = np.mean(expert_vals)
            ri = rescue_index(expert_mean, mean_g, mean_u)

            summary_rows.append({
                "model": model,
                "metric": metric,
                "mean_guided": round(mean_g, 4),
                "mean_unguided": round(mean_u, 4),
                "delta": round(delta, 4),
                "pct_change": round(delta / max(abs(mean_u), 1e-6) * 100, 1),
                "wilcoxon_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                "p_value": p,
                "effect_r": round(r, 3) if not np.isnan(r) else np.nan,
                "expert_mean": round(expert_mean, 4),
                "rescue_index": round(ri, 3) if not np.isnan(ri) else np.nan,
            })

            # Per-task detail
            for i, tid in enumerate(shared_tasks):
                task_rows.append({
                    "task_id": tid,
                    "model": model,
                    "metric": metric,
                    "value_guided": round(g_vals[i], 4),
                    "value_unguided": round(u_vals[i], 4),
                    "delta": round(g_vals[i] - u_vals[i], 4),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(task_rows)


def compute_category_breakdown(
    depth_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mode delta by approach (de_novo vs redesign) and molecular subject."""
    merged = depth_df.copy()
    cand_cols = ["task_id", "condition"] + CANDIDATE_METRICS
    cand_subset = cand_df[cand_cols].copy()
    merged = merged.merge(cand_subset, on=["task_id", "condition"], how="left")

    key_metrics = [
        "total_exec_depth", "scoring_depth", "eval_multiplier", "mmvs",
        "n_candidates", "mean_eval_per_candidate", "total_eval_calls",
    ]

    rows = []
    for model, user_cond, bench_cond in MODEL_PAIRS:
        guided = merged[merged["condition"] == user_cond]
        unguided = merged[merged["condition"] == bench_cond]

        for category_col, category_name in [("approach", "approach"), ("subject", "subject")]:
            categories = sorted(set(guided[category_col].dropna().unique()) |
                                set(unguided[category_col].dropna().unique()))

            for cat in categories:
                g = guided[guided[category_col] == cat]
                u = unguided[unguided[category_col] == cat]
                if len(g) == 0 or len(u) == 0:
                    continue

                for metric in key_metrics:
                    if metric not in g.columns:
                        continue
                    g_vals = g[metric].fillna(0).values.astype(float)
                    u_vals = u[metric].fillna(0).values.astype(float)
                    mean_g = np.mean(g_vals)
                    mean_u = np.mean(u_vals)

                    # Mann-Whitney for unequal sizes
                    try:
                        _, p = mannwhitneyu(g_vals, u_vals, alternative="two-sided")
                    except ValueError:
                        p = np.nan

                    rows.append({
                        "model": model,
                        "category_type": category_name,
                        "category": cat,
                        "metric": metric,
                        "mean_guided": round(mean_g, 4),
                        "mean_unguided": round(mean_u, 4),
                        "delta": round(mean_g - mean_u, 4),
                        "p_value": p,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage-Depth correlation (Step 4)
# ---------------------------------------------------------------------------

def compute_coverage_depth_correlation(task_detail_df: pd.DataFrame) -> pd.DataFrame:
    """Correlate ΔCoverage with ΔDepth per model."""
    rows = []
    for model in task_detail_df["model"].unique():
        m = task_detail_df[task_detail_df["model"] == model]

        # ΔCoverage: sum of binary coverage deltas across stages
        cov_metrics = ["backbone_gen_coverage", "seqdesign_coverage",
                       "structpred_coverage", "scoring_coverage"]
        depth_metric = "total_exec_depth"

        # Pivot to get per-task ΔCoverage and ΔDepth
        cov_deltas = []
        for cm in cov_metrics:
            sub = m[m["metric"] == cm]
            if len(sub) > 0:
                cov_deltas.append(sub.set_index("task_id")["delta"])

        depth_sub = m[m["metric"] == depth_metric]
        if len(depth_sub) == 0 or len(cov_deltas) == 0:
            continue

        # Sum coverage deltas
        cov_sum = pd.concat(cov_deltas, axis=1).sum(axis=1)
        dep = depth_sub.set_index("task_id")["delta"]

        shared = cov_sum.index.intersection(dep.index)
        if len(shared) < 10:
            continue

        rho, p = spearmanr(cov_sum.loc[shared].values, dep.loc[shared].values)
        rows.append({
            "model": model,
            "metric_pair": "ΔCoverage_sum vs ΔDepth_total",
            "rho": round(rho, 3),
            "p_value": p,
            "n": len(shared),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_tables(
    summary_df: pd.DataFrame,
    category_df: pd.DataFrame,
    corr_df: pd.DataFrame,
):
    """Print formatted tables."""

    # ── Table 1: Guidance Rescue — Coverage vs Depth ──
    print("\n" + "=" * 130)
    print("Table 1: Guidance Rescue Index — Coverage vs Depth")
    print("  (RI=0: no rescue, RI=1: fully rescued to Expert level, RI<0: guidance hurts)")
    print("=" * 130)

    # Select key metrics for display
    display_metrics = [
        ("backbone_gen_coverage", "BB Coverage"),
        ("scoring_coverage", "Scoring Cov"),
        ("total_exec_depth", "Total Depth"),
        ("scoring_depth", "Scoring Depth"),
        ("eval_multiplier", "Eval Mult"),
        ("mmvs", "MMVS"),
        ("mean_eval_per_candidate", "Eval/Cand"),
        ("n_candidates", "N_Cands"),
        ("total_eval_calls", "Total Eval"),
    ]

    # Header
    header = f"{'Model':<16s}"
    for _, label in display_metrics:
        header += f" | {label:>13s}"
    print(header)

    sub_header = f"{'':16s}"
    for _, _ in display_metrics:
        sub_header += f" | {'RI (Δ)':>13s}"
    print(sub_header)
    print("-" * 130)

    for model in ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]:
        m = summary_df[summary_df["model"] == model]
        line = f"{model:<16s}"
        for metric_key, _ in display_metrics:
            row = m[m["metric"] == metric_key]
            if len(row) > 0:
                r = row.iloc[0]
                ri_val = r["rescue_index"]
                delta_val = r["delta"]
                if np.isnan(ri_val):
                    cell = f"— ({delta_val:+.2f})"
                else:
                    cell = f"{ri_val:+.2f} ({delta_val:+.2f})"
                line += f" | {cell:>13s}"
            else:
                line += f" | {'—':>13s}"
        print(line)

    # Expert reference row
    print("-" * 130)
    line = f"{'Expert ref':16s}"
    for metric_key, _ in display_metrics:
        rows = summary_df[summary_df["metric"] == metric_key]
        if len(rows) > 0:
            exp = rows.iloc[0]["expert_mean"]
            line += f" | {exp:>13.2f}"
        else:
            line += f" | {'—':>13s}"
    print(line)

    # ── Table 2: Guided vs Unguided means ──
    print("\n" + "=" * 130)
    print("Table 2: Guided vs Unguided Means (key metrics)")
    print("=" * 130)

    key_metrics = [
        ("total_exec_depth", "Total Depth"),
        ("scoring_depth", "Scoring Dep"),
        ("eval_multiplier", "Eval Mult"),
        ("mmvs", "MMVS"),
        ("mean_eval_per_candidate", "Eval/Cand"),
        ("total_eval_calls", "Total Eval"),
    ]

    header = f"{'Model':<16s}"
    for _, label in key_metrics:
        header += f" | {'G':>5s} {'U':>5s} {'Δ':>6s} {'p':>8s}"
    print(header)
    print("-" * 130)

    for model in ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]:
        m = summary_df[summary_df["model"] == model]
        line = f"{model:<16s}"
        for metric_key, _ in key_metrics:
            row = m[m["metric"] == metric_key]
            if len(row) > 0:
                r = row.iloc[0]
                g = r["mean_guided"]
                u = r["mean_unguided"]
                d = r["delta"]
                p = r["p_value"]
                p_str = f"{p:.1e}" if not np.isnan(p) else "—"
                line += f" | {g:>5.2f} {u:>5.2f} {d:>+6.2f} {p_str:>8s}"
            else:
                line += f" | {'—':>26s}"
        print(line)

    # ── Table 3: De novo vs Redesign mode effect ──
    print("\n" + "=" * 110)
    print("Table 3: Mode Effect by Task Type (De novo vs Redesign)")
    print("=" * 110)

    dn_rd_metrics = ["total_exec_depth", "scoring_depth", "eval_multiplier",
                     "mean_eval_per_candidate", "total_eval_calls"]

    header = f"{'Model':<16s} | {'Approach':<10s}"
    for m_key in dn_rd_metrics:
        short = m_key.replace("total_exec_depth", "TotDep").replace(
            "scoring_depth", "ScDep").replace("eval_multiplier", "EM").replace(
            "mean_eval_per_candidate", "Eval/C").replace("total_eval_calls", "TotEval")
        header += f" | {short:>8s}"
    print(header)
    print("-" * 110)

    for model in ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]:
        cat_m = category_df[(category_df["model"] == model) &
                            (category_df["category_type"] == "approach")]
        for approach in ["de_novo", "redesign"]:
            a = cat_m[cat_m["category"] == approach]
            line = f"{model:<16s} | {approach:<10s}"
            for m_key in dn_rd_metrics:
                row = a[a["metric"] == m_key]
                if len(row) > 0:
                    d = row.iloc[0]["delta"]
                    line += f" | {d:>+8.2f}"
                else:
                    line += f" | {'—':>8s}"
            print(line)

    # ── Table 4: Coverage-Depth Correlation ──
    if len(corr_df) > 0:
        print("\n" + "=" * 70)
        print("Table 4: ΔCoverage vs ΔDepth Correlation (per model)")
        print("=" * 70)
        print(f"{'Model':<16s} | {'ρ':>6s} | {'p-value':>10s} | {'n':>4s} | Interpretation")
        print("-" * 70)
        for _, r in corr_df.iterrows():
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
            interp = "coverage↑ ↔ depth↑" if r["rho"] > 0.3 else "weak/no link"
            print(f"{r['model']:<16s} | {r['rho']:>6.3f} | {r['p_value']:>10.2e} | {r['n']:>4d} | {interp} {sig}")

    # ── Summary: Which scenario? ──
    print("\n" + "=" * 80)
    print("Summary: Does guidance rescue depth?")
    print("=" * 80)

    # Compute average rescue index for coverage vs depth
    for model in ["DeepSeek V3", "GPT-5", "Sonnet 4.5", "Gemini 2.5 Pro"]:
        m = summary_df[summary_df["model"] == model]
        cov_ri = m[m["metric"].isin(["backbone_gen_coverage", "seqdesign_coverage",
                                     "structpred_coverage", "scoring_coverage"])]["rescue_index"]
        dep_ri = m[m["metric"].isin(["total_exec_depth", "scoring_depth",
                                     "eval_multiplier", "mmvs"])]["rescue_index"]
        cand_ri = m[m["metric"].isin(["mean_eval_per_candidate", "total_eval_calls",
                                      "n_candidates"])]["rescue_index"]

        mean_cov = cov_ri.mean()
        mean_dep = dep_ri.mean()
        mean_cand = cand_ri.mean()

        # Determine scenario
        if mean_dep > 0.3 and mean_cov > 0.3:
            scenario = "Scenario 1: Both coverage AND depth rescued"
        elif mean_cov > 0.3 and mean_dep <= 0.3:
            scenario = "Scenario 2: Coverage rescued, depth NOT rescued"
        elif mean_cov <= 0.3 and mean_dep > 0.3:
            scenario = "Scenario 3: Depth rescued, coverage NOT rescued (unexpected)"
        else:
            scenario = "Neither coverage nor depth meaningfully rescued"

        print(f"  {model:<16s}  Cov RI={mean_cov:+.3f}  Dep RI={mean_dep:+.3f}  "
              f"Cand RI={mean_cand:+.3f}  → {scenario}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    analysis_dir = ROOT / "results" / "analysis"

    print("[1/5] Loading source data...")
    depth_df = load_depth_per_task(analysis_dir / "plan_exec_depth_v3.csv")
    cand_df = load_candidate_per_task(analysis_dir / "candidate_screening_v1_task_summary.csv")
    print(f"  Depth: {len(depth_df)} task-level rows")
    print(f"  Candidate: {len(cand_df)} task-level rows")

    print("[2/5] Computing model × metric comparisons...")
    summary_df, task_detail_df = compute_model_metric_comparison(depth_df, cand_df)
    print(f"  {len(summary_df)} summary rows, {len(task_detail_df)} task-detail rows")

    print("[3/5] Computing category breakdowns...")
    category_df = compute_category_breakdown(depth_df, cand_df)
    print(f"  {len(category_df)} category rows")

    print("[4/5] Computing coverage-depth correlations...")
    corr_df = compute_coverage_depth_correlation(task_detail_df)
    print(f"  {len(corr_df)} correlation rows")

    # Save CSVs
    out_dir = analysis_dir
    summary_df.to_csv(out_dir / "mode_depth_comparison_v1.csv", index=False)
    task_detail_df.to_csv(out_dir / "mode_depth_comparison_v1_by_task.csv", index=False)
    category_df.to_csv(out_dir / "mode_depth_comparison_v1_by_category.csv", index=False)

    print(f"\n[5/5] Saved CSVs to {out_dir}/")
    print(f"  mode_depth_comparison_v1.csv             ({len(summary_df)} rows)")
    print(f"  mode_depth_comparison_v1_by_task.csv      ({len(task_detail_df)} rows)")
    print(f"  mode_depth_comparison_v1_by_category.csv  ({len(category_df)} rows)")

    print_tables(summary_df, category_df, corr_df)


if __name__ == "__main__":
    main()
