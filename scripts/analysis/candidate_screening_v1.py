#!/usr/bin/env python3
"""PROMPT 12: Candidate-Level Screening Analysis — "n=1 Problem" 정량화.

Segments each (task × condition) tool call sequence into distinct design candidates,
measures per-candidate evaluation depth, detects funnel patterns, and decomposes
the Expert-LLM scoring depth gap into N_candidates × Eval_per_candidate.

Outputs:
  results/analysis/candidate_screening_v1.csv                  — per candidate
  results/analysis/candidate_screening_v1_task_summary.csv     — per task × condition
  results/analysis/candidate_screening_v1_condition_summary.csv — per condition
  results/analysis/candidate_screening_v1_stats.csv            — statistical tests
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.load_results import CONDITION_MAP, EXCLUDED_TASKS, load_all

# ---------------------------------------------------------------------------
# Tool classifications (consistent with plan_exec_depth_v3 + section26)
# ---------------------------------------------------------------------------

GENERATIVE_TOOLS = frozenset({
    "generate_backbone",   # RFdiffusion
    "design_binder",       # RFdiffusion + ProteinMPNN composite
    "design_sequence",     # ProteinMPNN (alias)
    "optimize_sequence",   # ProteinMPNN inverse folding
    "rosetta_design",      # Rosetta fixed-backbone design
})

EVALUATIVE_TOOLS = frozenset({
    "predict_structure",
    "predict_structure_boltz",
    "predict_complex",
    "predict_affinity_boltz",
    "score_stability",
    "rosetta_score",
    "rosetta_relax",
    "analyze_interface",
    "rosetta_interface_score",
    "energy_minimize",
    "validate_design",
    "suggest_hotspots",
})

# Subcategories for eval diversity
EVAL_TOOL_CATEGORIES: dict[str, list[str]] = {
    "structure": ["predict_structure", "predict_structure_boltz", "predict_complex"],
    "energy": ["rosetta_score", "rosetta_relax", "energy_minimize"],
    "stability": ["score_stability"],
    "interface": ["analyze_interface", "rosetta_interface_score"],
    "affinity": ["predict_affinity_boltz"],
    "composite": ["validate_design"],
}

_EVAL_CAT_LOOKUP: dict[str, str] = {}
for _cat, _tools in EVAL_TOOL_CATEGORIES.items():
    for _t in _tools:
        _EVAL_CAT_LOOKUP[_t] = _cat

# Structure prediction tools (for funnel detection)
STRUCT_PRED_TOOLS = frozenset({
    "predict_structure", "predict_structure_boltz",
    "predict_complex", "predict_affinity_boltz",
})

# Scoring tools (for funnel detection)
SCORING_TOOLS = frozenset({
    "score_stability", "rosetta_score", "rosetta_relax",
    "analyze_interface", "rosetta_interface_score",
    "energy_minimize", "validate_design", "suggest_hotspots",
})

SCRIPTED_BASELINES = {"Oracle", "Human Expert", "Hardcoded Pipeline"}

LLM_USER_CONDITIONS = [
    "DeepSeek V3 user",
    "GPT-5 user",
    "Sonnet 4.5 user",
    "Gemini 2.5 Pro user",
]


# ---------------------------------------------------------------------------
# Step 1: Candidate segmentation
# ---------------------------------------------------------------------------

def segment_candidates(tool_call_log: list[dict]) -> list[dict]:
    """Segment tool call sequence into distinct design candidates.

    A new candidate boundary is detected when a generative tool is called
    AFTER evaluative tools have already been called in the current candidate.
    This captures the generate→evaluate→re-generate cycle.

    Returns list of dicts, each with:
      gen_tools: list of generative tool call dicts
      eval_tools: list of evaluative tool call dicts
      all_tools: list of all tool call dicts in this candidate
    """
    candidates: list[dict] = []
    current: dict = {"gen_tools": [], "eval_tools": [], "all_tools": []}

    for tc in tool_call_log:
        tool_name = tc.get("tool", "")

        if tool_name in GENERATIVE_TOOLS:
            # If we already have eval tools → new candidate boundary
            if current["eval_tools"]:
                candidates.append(current)
                current = {"gen_tools": [], "eval_tools": [], "all_tools": []}
            current["gen_tools"].append(tc)
            current["all_tools"].append(tc)
        elif tool_name in EVALUATIVE_TOOLS:
            current["eval_tools"].append(tc)
            current["all_tools"].append(tc)
        else:
            # Non-protein-design tools (execute_python, read_file, etc.) — skip
            current["all_tools"].append(tc)

    # Append last candidate if it has any protein design tools
    if current["gen_tools"] or current["eval_tools"]:
        candidates.append(current)

    return candidates


# ---------------------------------------------------------------------------
# Step 2: Per-candidate metrics
# ---------------------------------------------------------------------------

def compute_candidate_metrics(candidate: dict) -> dict:
    """Compute metrics for a single candidate."""
    eval_tools = [tc.get("tool", "") for tc in candidate["eval_tools"]]
    eval_categories = set(_EVAL_CAT_LOOKUP.get(t, "") for t in eval_tools) - {""}

    return {
        "n_gen_steps": len(candidate["gen_tools"]),
        "n_eval_steps": len(candidate["eval_tools"]),
        "n_unique_eval_tools": len(set(eval_tools)),
        "n_eval_categories": len(eval_categories),
        "eval_categories": ",".join(sorted(eval_categories)) if eval_categories else "",
        "has_structure_pred": any(tc.get("tool") in STRUCT_PRED_TOOLS for tc in candidate["eval_tools"]),
        "has_energy_score": any(tc.get("tool") in SCORING_TOOLS for tc in candidate["eval_tools"]),
        "has_interface_analysis": any(
            tc.get("tool") in {"analyze_interface", "rosetta_interface_score"}
            for tc in candidate["eval_tools"]
        ),
    }


# ---------------------------------------------------------------------------
# Step 4: Funnel detection
# ---------------------------------------------------------------------------

def detect_funnel_pattern(tool_call_log: list[dict], candidates: list[dict]) -> dict:
    """Detect design funnel pattern and measure its depth."""
    n_candidates = len(candidates)

    # Count tool types across entire sequence
    gen_count = sum(1 for tc in tool_call_log if tc.get("tool") in GENERATIVE_TOOLS)
    struct_pred_count = sum(1 for tc in tool_call_log if tc.get("tool") in STRUCT_PRED_TOOLS)
    scoring_count = sum(1 for tc in tool_call_log if tc.get("tool") in SCORING_TOOLS)

    # Funnel: multiple generations AND at least as many predictions AND scorings
    funnel_detected = (
        n_candidates >= 3
        and struct_pred_count >= n_candidates
        and scoring_count >= n_candidates
    )

    # Funnel depth: how many screening stages
    funnel_depth = sum([
        n_candidates >= 2,                       # multiple generation
        struct_pred_count >= 2,                   # multiple prediction
        scoring_count >= 2,                       # multiple scoring
        any(tc.get("tool") == "analyze_interface" for tc in tool_call_log),  # interface
    ])

    return {
        "funnel_detected": funnel_detected,
        "funnel_depth": funnel_depth,
        "initial_gen_count": gen_count,
        "struct_pred_count": struct_pred_count,
        "scoring_count": scoring_count,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_result_json(condition: str, task_id: str) -> dict | None:
    info = CONDITION_MAP.get(condition)
    if info is None:
        return None
    result_file = info["path"] / task_id / "result.json"
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def _get_tool_call_log(result: dict) -> list[dict]:
    return result.get("raw_output", {}).get("tool_call_log", [])


def _get_approach(task_id: str) -> str | None:
    try:
        from biodesignbench.taxonomy import get_category
        cat = get_category(task_id)
        if cat is not None:
            return cat.approach.value
    except ImportError:
        pass
    dn_prefixes = ("binder_", "scaffold_", "ppi_", "peptide_",
                   "dnb_", "dnk_", "cfd_", "cpx_")
    rd_prefixes = ("enzyme_", "stability_", "fluorescence_",
                   "antibody_", "sqo_")
    if task_id.startswith(dn_prefixes):
        return "de_novo"
    elif task_id.startswith(rd_prefixes):
        return "redesign"
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-candidate and per-task summary DataFrames."""
    cand_rows = []
    task_rows = []

    for _, row in df.iterrows():
        condition = row["condition"]
        task_id = row["task_id"]
        approach = _get_approach(task_id)
        if approach is None:
            continue

        result = _load_result_json(condition, task_id)
        if result is None:
            continue

        tcl = _get_tool_call_log(result)
        if not tcl:
            # No tool calls — still record a task row with zeros
            task_rows.append({
                "task_id": task_id,
                "condition": condition,
                "approach": approach,
                "n_candidates": 0,
                "n_eval_only_sessions": 0,
                "mean_eval_per_candidate": 0.0,
                "mean_eval_diversity_per_candidate": 0.0,
                "max_eval_per_candidate": 0,
                "screening_ratio": 0.0,
                "multi_metric_candidates": 0,
                "frac_multi_metric": 0.0,
                "funnel_detected": False,
                "funnel_depth": 0,
                "initial_gen_count": 0,
                "struct_pred_count": 0,
                "scoring_count": 0,
                "total_eval_calls": 0,
                "total_score": row.get("total", 0),
            })
            continue

        candidates = segment_candidates(tcl)
        if not candidates:
            task_rows.append({
                "task_id": task_id,
                "condition": condition,
                "approach": approach,
                "n_candidates": 0,
                "n_eval_only_sessions": 0,
                "mean_eval_per_candidate": 0.0,
                "mean_eval_diversity_per_candidate": 0.0,
                "max_eval_per_candidate": 0,
                "screening_ratio": 0.0,
                "multi_metric_candidates": 0,
                "frac_multi_metric": 0.0,
                "funnel_detected": False,
                "funnel_depth": 0,
                "initial_gen_count": 0,
                "struct_pred_count": 0,
                "scoring_count": 0,
                "total_eval_calls": 0,
                "total_score": row.get("total", 0),
            })
            continue

        # Per-candidate metrics
        cand_metrics_list = []
        for ci, cand in enumerate(candidates):
            cm = compute_candidate_metrics(cand)
            cand_rows.append({
                "task_id": task_id,
                "condition": condition,
                "approach": approach,
                "candidate_idx": ci,
                **cm,
            })
            cand_metrics_list.append(cm)

        # Task-level aggregation
        n_cand = len(candidates)
        n_eval_only = sum(1 for c in candidates if not c["gen_tools"])
        eval_counts = [cm["n_eval_steps"] for cm in cand_metrics_list]
        eval_div = [cm["n_unique_eval_tools"] for cm in cand_metrics_list]
        n_with_eval = sum(1 for e in eval_counts if e > 0)
        multi_metric = sum(1 for cm in cand_metrics_list if cm["n_eval_categories"] >= 3)

        funnel = detect_funnel_pattern(tcl, candidates)

        total_eval = sum(1 for tc in tcl if tc.get("tool") in EVALUATIVE_TOOLS)

        task_rows.append({
            "task_id": task_id,
            "condition": condition,
            "approach": approach,
            "n_candidates": n_cand,
            "n_eval_only_sessions": n_eval_only,
            "mean_eval_per_candidate": np.mean(eval_counts) if eval_counts else 0.0,
            "mean_eval_diversity_per_candidate": np.mean(eval_div) if eval_div else 0.0,
            "max_eval_per_candidate": max(eval_counts) if eval_counts else 0,
            "screening_ratio": n_with_eval / n_cand if n_cand > 0 else 0.0,
            "multi_metric_candidates": multi_metric,
            "frac_multi_metric": multi_metric / n_cand if n_cand > 0 else 0.0,
            "funnel_detected": funnel["funnel_detected"],
            "funnel_depth": funnel["funnel_depth"],
            "initial_gen_count": funnel["initial_gen_count"],
            "struct_pred_count": funnel["struct_pred_count"],
            "scoring_count": funnel["scoring_count"],
            "total_eval_calls": total_eval,
            "total_score": row.get("total", 0),
        })

    return pd.DataFrame(cand_rows), pd.DataFrame(task_rows)


def compute_condition_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate task-level data by condition."""
    rows = []
    for condition, grp in task_df.groupby("condition"):
        n_tasks = len(grp)
        mean_n_cand = grp["n_candidates"].mean()
        median_n_cand = grp["n_candidates"].median()
        mean_eval_per_cand = grp["mean_eval_per_candidate"].mean()
        mean_eval_div = grp["mean_eval_diversity_per_candidate"].mean()
        frac_funnel = grp["funnel_detected"].mean()
        total_eval = grp["total_eval_calls"].mean()

        # Multi-metric candidate rate (across all tasks)
        total_multi = grp["multi_metric_candidates"].sum()
        total_cand = grp["n_candidates"].sum()
        frac_multi = total_multi / total_cand if total_cand > 0 else 0.0

        # Decomposition: Total_Eval ≈ N_cand × Eval/cand
        # Use means for the decomposition
        decomp_cand = mean_n_cand
        decomp_eval = mean_eval_per_cand
        product = decomp_cand * decomp_eval

        rows.append({
            "condition": condition,
            "n_tasks": n_tasks,
            "mean_n_candidates": round(mean_n_cand, 2),
            "median_n_candidates": round(median_n_cand, 1),
            "mean_eval_per_candidate": round(mean_eval_per_cand, 2),
            "mean_eval_diversity_per_cand": round(mean_eval_div, 2),
            "frac_funnel_detected": round(frac_funnel, 3),
            "frac_multi_metric_candidates": round(frac_multi, 3),
            "total_eval_depth": round(total_eval, 2),
            "decomp_n_cand": round(decomp_cand, 2),
            "decomp_eval_per_cand": round(decomp_eval, 2),
            "decomp_product": round(product, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _safe_mannwhitney(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    if np.std(x) == 0 and np.std(y) == 0 and np.mean(x) == np.mean(y):
        return np.nan, 1.0, 0.0
    try:
        U, p = mannwhitneyu(x, y, alternative="two-sided")
        r = 1 - (2 * U) / (len(x) * len(y))  # rank-biserial
        return U, p, r
    except ValueError:
        return np.nan, np.nan, np.nan


def _cohens_d(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(x) - 1) * np.std(x, ddof=1) ** 2 + (len(y) - 1) * np.std(y, ddof=1) ** 2)
        / (len(x) + len(y) - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def _bonferroni(ps: list[float]) -> list[float]:
    n = len(ps)
    return [min(p * n, 1.0) if not np.isnan(p) else np.nan for p in ps]


def compute_stats(task_df: pd.DataFrame) -> pd.DataFrame:
    """Run all statistical tests."""
    stat_rows = []
    expert = task_df[task_df["condition"] == "Human Expert"]

    # ── 6a: N_candidates ──
    raw_ps = []
    for cond in LLM_USER_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        U, p, r = _safe_mannwhitney(expert["n_candidates"].values, llm["n_candidates"].values)
        d = _cohens_d(expert["n_candidates"].values, llm["n_candidates"].values)
        raw_ps.append(p)
        stat_rows.append({
            "test_name": "n_candidates",
            "comparison": f"Expert vs {cond}",
            "statistic": round(U, 2) if not np.isnan(U) else np.nan,
            "p_value": p,
            "effect_size_r": round(r, 3) if not np.isnan(r) else np.nan,
            "cohens_d": round(d, 3) if not np.isnan(d) else np.nan,
        })
    corrected = _bonferroni(raw_ps)
    for i in range(len(corrected)):
        stat_rows[-(len(corrected) - i)]["p_corrected"] = corrected[i]

    # ── 6b: Eval-per-candidate ──
    raw_ps = []
    start_idx = len(stat_rows)
    for cond in LLM_USER_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        U, p, r = _safe_mannwhitney(
            expert["mean_eval_per_candidate"].values,
            llm["mean_eval_per_candidate"].values,
        )
        d = _cohens_d(
            expert["mean_eval_per_candidate"].values,
            llm["mean_eval_per_candidate"].values,
        )
        raw_ps.append(p)
        stat_rows.append({
            "test_name": "eval_per_candidate",
            "comparison": f"Expert vs {cond}",
            "statistic": round(U, 2) if not np.isnan(U) else np.nan,
            "p_value": p,
            "effect_size_r": round(r, 3) if not np.isnan(r) else np.nan,
            "cohens_d": round(d, 3) if not np.isnan(d) else np.nan,
        })
    corrected = _bonferroni(raw_ps)
    for i, idx in enumerate(range(start_idx, len(stat_rows))):
        stat_rows[idx]["p_corrected"] = corrected[i]

    # ── 6c: Multi-metric fraction (Fisher exact) ──
    expert_multi = expert["multi_metric_candidates"].sum()
    expert_single = expert["n_candidates"].sum() - expert_multi

    for cond in LLM_USER_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        llm_multi = llm["multi_metric_candidates"].sum()
        llm_single = llm["n_candidates"].sum() - llm_multi

        if (expert_multi + expert_single) > 0 and (llm_multi + llm_single) > 0:
            table = [[expert_multi, expert_single], [llm_multi, llm_single]]
            odds, p = fisher_exact(table)
        else:
            odds, p = np.nan, np.nan

        stat_rows.append({
            "test_name": "multi_metric_fisher",
            "comparison": f"Expert vs {cond}",
            "statistic": round(odds, 3) if not np.isnan(odds) else np.nan,
            "p_value": p,
            "p_corrected": np.nan,
            "effect_size_r": np.nan,
            "cohens_d": np.nan,
        })

    # ── 6d: Funnel detection rate (Fisher exact) ──
    expert_funnel = expert["funnel_detected"].sum()
    expert_no_funnel = len(expert) - expert_funnel

    for cond in LLM_USER_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        llm_funnel = llm["funnel_detected"].sum()
        llm_no_funnel = len(llm) - llm_funnel

        if (expert_funnel + expert_no_funnel) > 0 and (llm_funnel + llm_no_funnel) > 0:
            table = [[int(expert_funnel), int(expert_no_funnel)],
                     [int(llm_funnel), int(llm_no_funnel)]]
            odds, p = fisher_exact(table)
        else:
            odds, p = np.nan, np.nan

        stat_rows.append({
            "test_name": "funnel_fisher",
            "comparison": f"Expert vs {cond}",
            "statistic": round(odds, 3) if not np.isnan(odds) else np.nan,
            "p_value": p,
            "p_corrected": np.nan,
            "effect_size_r": np.nan,
            "cohens_d": np.nan,
        })

    # ── 6e: Log-linear decomposition (relative contribution) ──
    # log(total_eval + 1) = β1 · log(n_cand + 1) + β2 · log(eval_per_cand + 1)
    valid = task_df[task_df["n_candidates"] > 0].copy()
    if len(valid) >= 20:
        from sklearn.linear_model import LinearRegression

        log_total = np.log1p(valid["total_eval_calls"].values).reshape(-1, 1)
        log_n_cand = np.log1p(valid["n_candidates"].values)
        log_eval_pc = np.log1p(valid["mean_eval_per_candidate"].values)
        X = np.column_stack([log_n_cand, log_eval_pc])

        reg = LinearRegression().fit(X, log_total.ravel())
        beta_cand, beta_eval = reg.coef_
        r2 = reg.score(X, log_total.ravel())

        stat_rows.append({
            "test_name": "log_linear_decomposition",
            "comparison": "log(total) ~ log(n_cand) + log(eval/cand)",
            "statistic": round(r2, 3),
            "p_value": np.nan,
            "p_corrected": np.nan,
            "effect_size_r": np.nan,
            "cohens_d": np.nan,
            "beta_n_cand": round(beta_cand, 3),
            "beta_eval_per_cand": round(beta_eval, 3),
        })

    # ── 6f: De novo vs Redesign gap ──
    for approach in ["de_novo", "redesign"]:
        sub_expert = expert[expert["approach"] == approach]
        for cond in LLM_USER_CONDITIONS:
            sub_llm = task_df[(task_df["condition"] == cond) & (task_df["approach"] == approach)]
            U, p, r = _safe_mannwhitney(
                sub_expert["n_candidates"].values,
                sub_llm["n_candidates"].values,
            )
            stat_rows.append({
                "test_name": f"n_candidates_{approach}",
                "comparison": f"Expert vs {cond}",
                "statistic": round(U, 2) if not np.isnan(U) else np.nan,
                "p_value": p,
                "p_corrected": np.nan,
                "effect_size_r": round(r, 3) if not np.isnan(r) else np.nan,
                "cohens_d": np.nan,
            })

    # ── 6g: Spearman — n_candidates vs total_score ──
    for metric_name, col in [
        ("n_candidates", "n_candidates"),
        ("mean_eval_per_candidate", "mean_eval_per_candidate"),
        ("total_eval_calls", "total_eval_calls"),
        ("funnel_depth", "funnel_depth"),
    ]:
        vals = task_df[[col, "total_score"]].dropna()
        if len(vals) >= 10:
            rho, p = spearmanr(vals[col], vals["total_score"])
            stat_rows.append({
                "test_name": "spearman_vs_score",
                "comparison": f"{metric_name} vs total_score",
                "statistic": round(rho, 3),
                "p_value": p,
                "p_corrected": np.nan,
                "effect_size_r": round(rho, 3),
                "cohens_d": np.nan,
            })

    return pd.DataFrame(stat_rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_tables(
    task_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    stats_df: pd.DataFrame,
):
    """Print formatted tables."""

    # ── Table 1: Candidate-Level Screening Summary ──
    print("\n" + "=" * 115)
    print("Table 1: Candidate-Level Screening Summary")
    print("=" * 115)
    print(
        f"{'Condition':<28s} | {'Mean':>6s} | {'Mean Eval/':>10s} | {'Mean Eval':>9s} | "
        f"{'Funnel':>7s} | {'Multi-met':>9s} | {'Screen':>7s}"
    )
    print(
        f"{'':28s} | {'Cands':>6s} | {'Candidate':>10s} | {'Div/C':>9s} | "
        f"{'Rate':>7s} | {'Cand Rate':>9s} | {'Ratio':>7s}"
    )
    print("-" * 115)

    for _, r in cond_df.sort_values("mean_n_candidates", ascending=False).iterrows():
        print(
            f"{r['condition']:<28s} | "
            f"{r['mean_n_candidates']:>6.2f} | "
            f"{r['mean_eval_per_candidate']:>10.2f} | "
            f"{r['mean_eval_diversity_per_cand']:>9.2f} | "
            f"{r['frac_funnel_detected']*100:>6.1f}% | "
            f"{r['frac_multi_metric_candidates']*100:>8.1f}% | "
            f"{r.get('total_eval_depth', 0):>7.2f}"
        )

    # ── Table 2: Depth Decomposition ──
    print("\n" + "=" * 100)
    print("Table 2: Depth Decomposition — Total Eval ≈ N_cand × Eval/cand")
    print("=" * 100)
    print(
        f"{'Condition':<28s} | {'Total Eval':>10s} | {'= N_cand':>8s} | "
        f"{'× Eval/C':>8s} | {'Product':>8s} | {'Primary Gap Source':<20s}"
    )
    print("-" * 100)

    # Get expert reference
    expert_row = cond_df[cond_df["condition"] == "Human Expert"]
    if len(expert_row) > 0:
        exp = expert_row.iloc[0]
        exp_n = exp["mean_n_candidates"]
        exp_e = exp["mean_eval_per_candidate"]
    else:
        exp_n, exp_e = 1.0, 1.0

    for _, r in cond_df.sort_values("total_eval_depth", ascending=False).iterrows():
        cond = r["condition"]
        # Determine primary gap source
        if cond == "Human Expert":
            gap_source = "—"
        else:
            n_ratio = r["mean_n_candidates"] / max(exp_n, 0.01)
            e_ratio = r["mean_eval_per_candidate"] / max(exp_e, 0.01)
            if n_ratio < 0.5 and e_ratio < 0.5:
                gap_source = "both"
            elif n_ratio < e_ratio:
                gap_source = "candidate count"
            else:
                gap_source = "eval depth"

        print(
            f"{cond:<28s} | "
            f"{r['total_eval_depth']:>10.2f} | "
            f"{r['decomp_n_cand']:>8.2f} | "
            f"{r['decomp_eval_per_cand']:>8.2f} | "
            f"{r['decomp_product']:>8.2f} | "
            f"{gap_source:<20s}"
        )

    # ── Table 3: De novo vs Redesign ──
    print("\n" + "=" * 105)
    print("Table 3: De novo vs Redesign — Candidate Counts & Eval Depth")
    print("=" * 105)
    print(
        f"{'Condition':<28s} | {'DN N_cand':>9s} | {'RD N_cand':>9s} | "
        f"{'DN Eval/C':>9s} | {'RD Eval/C':>9s} | {'DN Total':>8s} | {'RD Total':>8s}"
    )
    print("-" * 105)

    conditions_ordered = (
        ["Human Expert"] +
        sorted(LLM_USER_CONDITIONS) +
        ["Hardcoded Pipeline"]
    )

    for cond in conditions_ordered:
        grp = task_df[task_df["condition"] == cond]
        if len(grp) == 0:
            continue
        dn = grp[grp["approach"] == "de_novo"]
        rd = grp[grp["approach"] == "redesign"]

        dn_n = dn["n_candidates"].mean() if len(dn) > 0 else 0
        rd_n = rd["n_candidates"].mean() if len(rd) > 0 else 0
        dn_e = dn["mean_eval_per_candidate"].mean() if len(dn) > 0 else 0
        rd_e = rd["mean_eval_per_candidate"].mean() if len(rd) > 0 else 0
        dn_t = dn["total_eval_calls"].mean() if len(dn) > 0 else 0
        rd_t = rd["total_eval_calls"].mean() if len(rd) > 0 else 0

        print(
            f"{cond:<28s} | "
            f"{dn_n:>9.2f} | {rd_n:>9.2f} | "
            f"{dn_e:>9.2f} | {rd_e:>9.2f} | "
            f"{dn_t:>8.2f} | {rd_t:>8.2f}"
        )

    # ── Table 4: Statistical Tests ──
    print("\n" + "=" * 110)
    print("Table 4: Statistical Tests")
    print("=" * 110)
    print(
        f"{'Test':<28s} | {'Comparison':<30s} | {'Stat':>8s} | "
        f"{'p-value':>10s} | {'p-corr':>10s} | {'d':>6s}"
    )
    print("-" * 110)

    for _, r in stats_df.iterrows():
        stat = f"{r['statistic']:.2f}" if not np.isnan(r.get("statistic", np.nan)) else "—"
        p = f"{r['p_value']:.2e}" if not np.isnan(r.get("p_value", np.nan)) else "—"
        pc = f"{r['p_corrected']:.2e}" if not np.isnan(r.get("p_corrected", np.nan)) else "—"
        d = f"{r['cohens_d']:.2f}" if not np.isnan(r.get("cohens_d", np.nan)) else "—"
        print(
            f"{r['test_name']:<28s} | "
            f"{r['comparison']:<30s} | "
            f"{stat:>8s} | {p:>10s} | {pc:>10s} | {d:>6s}"
        )

    # ── Log-linear decomposition result ──
    decomp = stats_df[stats_df["test_name"] == "log_linear_decomposition"]
    if len(decomp) > 0:
        row = decomp.iloc[0]
        print(f"\n  Log-linear decomposition: R²={row['statistic']:.3f}")
        print(f"    β(N_candidates) = {row.get('beta_n_cand', '?')}")
        print(f"    β(Eval/candidate) = {row.get('beta_eval_per_cand', '?')}")
        beta_n = row.get("beta_n_cand", 0)
        beta_e = row.get("beta_eval_per_cand", 0)
        total_beta = abs(beta_n) + abs(beta_e)
        if total_beta > 0:
            print(f"    Relative contribution: N_cand={abs(beta_n)/total_beta*100:.1f}%, "
                  f"Eval/cand={abs(beta_e)/total_beta*100:.1f}%")

    # ── Spearman correlations ──
    spearman_rows = stats_df[stats_df["test_name"] == "spearman_vs_score"]
    if len(spearman_rows) > 0:
        print("\n  Spearman correlations with total_score:")
        for _, r in spearman_rows.iterrows():
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
            print(f"    {r['comparison']:<40s} rho={r['statistic']:.3f} p={r['p_value']:.2e} {sig}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/5] Loading results...")
    df = load_all()
    print(f"  {len(df)} rows ({df['task_id'].nunique()} tasks × {df['condition'].nunique()} conditions)")

    print("[2/5] Segmenting candidates & computing metrics...")
    cand_df, task_df = compute_all(df)
    print(f"  {len(cand_df)} candidate-level rows, {len(task_df)} task-level rows")

    print("[3/5] Computing condition summaries...")
    cond_df = compute_condition_summary(task_df)

    print("[4/5] Running statistical tests...")
    stats_df = compute_stats(task_df)
    print(f"  {len(stats_df)} tests computed")

    # Save CSVs
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_df.to_csv(out_dir / "candidate_screening_v1.csv", index=False)
    task_df.to_csv(out_dir / "candidate_screening_v1_task_summary.csv", index=False)
    cond_df.to_csv(out_dir / "candidate_screening_v1_condition_summary.csv", index=False)
    stats_df.to_csv(out_dir / "candidate_screening_v1_stats.csv", index=False)

    print(f"\n[5/5] Saved CSVs to {out_dir}/")
    print(f"  candidate_screening_v1.csv                  ({len(cand_df)} rows)")
    print(f"  candidate_screening_v1_task_summary.csv     ({len(task_df)} rows)")
    print(f"  candidate_screening_v1_condition_summary.csv ({len(cond_df)} rows)")
    print(f"  candidate_screening_v1_stats.csv            ({len(stats_df)} rows)")

    print_tables(task_df, cond_df, stats_df)


if __name__ == "__main__":
    main()
