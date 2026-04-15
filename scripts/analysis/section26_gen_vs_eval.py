#!/usr/bin/env python3
"""Section 2.6 — "LLM agents favour evaluation over generation"

Six quantitative analyses supporting the claim:

1. Generative vs Evaluative tool call ratio per condition
2. Statistical tests (Mann-Whitney U, Kruskal-Wallis, effect sizes)
3. Generate→Evaluate→Re-generate iteration cycle count
4. Backbone generation usage split by de novo vs redesign
5. Tool call diversity (Shannon entropy over functional categories)
6. First-generative-tool latency (position in call sequence)

Outputs:
  results/analysis/section26_gen_vs_eval.json   — full structured results
  stdout                                        — summary tables
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.load_results import CONDITION_MAP, load_all

# ───────────────────────────────────────────────────────────────────
# Tool classification (MCP-level names)
# ───────────────────────────────────────────────────────────────────

GENERATIVE_TOOLS = frozenset({
    "generate_backbone",   # RFdiffusion
    "design_binder",       # RFdiffusion + ProteinMPNN composite
    "design_sequence",     # ProteinMPNN
    "optimize_sequence",   # ProteinMPNN inverse folding
    "rosetta_design",      # Rosetta fixed-backbone design
})

EVALUATIVE_TOOLS = frozenset({
    "predict_structure",          # AlphaFold2/ESMFold monomer
    "predict_structure_boltz",    # Boltz structure prediction
    "predict_complex",            # AlphaFold2-Multimer
    "predict_affinity_boltz",     # Boltz affinity prediction
    "score_stability",            # Thermodynamic stability (ESM2)
    "rosetta_score",              # Rosetta energy
    "rosetta_relax",              # Rosetta relaxation
    "analyze_interface",          # Interface analysis
    "rosetta_interface_score",    # Rosetta interface energy
    "energy_minimize",            # Energy minimization (OpenMM)
    "validate_design",            # Multi-metric validation
    "suggest_hotspots",           # Hotspot identification
})

# Backbone generation specifically
BACKBONE_GEN_TOOLS = frozenset({
    "generate_backbone",
    "design_binder",   # composite but primary = backbone gen
})

# Functional categories for Shannon entropy (5 buckets)
FUNC_CATEGORIES = {
    "backbone_gen": {"generate_backbone", "design_binder"},
    "seq_design":   {"design_sequence", "optimize_sequence", "rosetta_design"},
    "struct_pred":  {"predict_structure", "predict_structure_boltz",
                     "predict_complex", "predict_affinity_boltz", "validate_design"},
    "scoring":      {"score_stability", "rosetta_score", "rosetta_relax",
                     "analyze_interface", "rosetta_interface_score",
                     "energy_minimize", "suggest_hotspots"},
    # "other" is anything not in the above
}

# Conditions
LLM_CONDITIONS = [
    "DeepSeek V3 benchmark", "DeepSeek V3 user",
    "GPT-5 benchmark", "GPT-5 user",
    "Sonnet 4.5 benchmark", "Sonnet 4.5 user",
    "Gemini 2.5 Pro benchmark", "Gemini 2.5 Pro user",
]

ALL_CONDITIONS = ["Human Expert"] + LLM_CONDITIONS

LLM_USER_CONDITIONS = [c for c in LLM_CONDITIONS if "user" in c]

# ───────────────────────────────────────────────────────────────────
# Data helpers
# ───────────────────────────────────────────────────────────────────

def _load_result(condition: str, task_id: str) -> dict | None:
    info = CONDITION_MAP.get(condition)
    if info is None:
        return None
    rf = info["path"] / task_id / "result.json"
    if not rf.exists():
        return None
    with open(rf) as f:
        return json.load(f)


def _tool_seq(result: dict) -> list[str]:
    """Ordered tool names from tool_call_log (MCP level)."""
    tcl = result.get("raw_output", {}).get("tool_call_log", [])
    return [tc.get("tool", "") for tc in tcl]


def _bio_tools(seq: list[str]) -> list[str]:
    """Filter to bio tools only (drop execute_python, web_search, etc.)."""
    bio = GENERATIVE_TOOLS | EVALUATIVE_TOOLS
    return [t for t in seq if t in bio]


def _categorize(tool: str) -> str:
    """Map tool → functional category."""
    for cat, tools in FUNC_CATEGORIES.items():
        if tool in tools:
            return cat
    return "other"


# ───────────────────────────────────────────────────────────────────
# Analysis 1: Generative vs Evaluative ratio per condition
# ───────────────────────────────────────────────────────────────────

def analysis1_gen_eval_ratio(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-task and per-condition gen/eval counts and ratios."""
    rows = []
    for _, row in df.iterrows():
        cond = row["condition"]
        tid = row["task_id"]
        result = _load_result(cond, tid)
        if result is None:
            continue
        seq = _tool_seq(result)
        bio = _bio_tools(seq)
        n_gen = sum(1 for t in bio if t in GENERATIVE_TOOLS)
        n_eval = sum(1 for t in bio if t in EVALUATIVE_TOOLS)
        n_bio = n_gen + n_eval
        rows.append({
            "task_id": tid,
            "condition": cond,
            "design_approach": row.get("design_approach", "unknown"),
            "n_gen": n_gen,
            "n_eval": n_eval,
            "n_bio": n_bio,
            "gen_ratio": n_gen / n_bio if n_bio > 0 else np.nan,
        })
    task_df = pd.DataFrame(rows)

    # Condition summary
    cond_rows = []
    for cond, grp in task_df.groupby("condition"):
        total_gen = grp["n_gen"].sum()
        total_eval = grp["n_eval"].sum()
        total_bio = total_gen + total_eval
        cond_rows.append({
            "condition": cond,
            "n_tasks": len(grp),
            "mean_gen": round(grp["n_gen"].mean(), 2),
            "mean_eval": round(grp["n_eval"].mean(), 2),
            "total_gen": int(total_gen),
            "total_eval": int(total_eval),
            "gen_ratio": round(total_gen / total_bio, 4) if total_bio > 0 else 0.0,
            "mean_task_gen_ratio": round(grp["gen_ratio"].mean(), 4),
        })
    cond_df = pd.DataFrame(cond_rows)
    return task_df, cond_df


# ───────────────────────────────────────────────────────────────────
# Analysis 2: Statistical tests
# ───────────────────────────────────────────────────────────────────

def _rank_biserial(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation (effect size for Mann-Whitney U).

    Positive = group 1 > group 2.  Uses scipy's U1 convention where
    large U1 means group 1 tends to rank higher.
    """
    return (2 * U) / (n1 * n2) - 1


def analysis2_stats(task_df: pd.DataFrame) -> dict:
    """Statistical tests on gen_ratio and absolute eval counts."""
    results = {}

    expert = task_df[task_df["condition"] == "Human Expert"]
    expert_ratios = expert.set_index("task_id")["gen_ratio"].dropna()
    expert_eval = expert.set_index("task_id")["n_eval"].dropna()
    expert_gen = expert.set_index("task_id")["n_gen"].dropna()

    # --- A) gen_ratio: Expert vs each LLM (two-sided) ---
    expert_vs_llm_ratio = []
    for cond in LLM_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        llm_ratios = llm.set_index("task_id")["gen_ratio"].dropna()
        common = expert_ratios.index.intersection(llm_ratios.index)
        if len(common) < 5:
            continue
        e = expert_ratios.loc[common].values
        l = llm_ratios.loc[common].values
        stat_u, p_u = stats.mannwhitneyu(e, l, alternative="two-sided")
        rb = _rank_biserial(stat_u, len(e), len(l))
        # Paired Wilcoxon
        diffs = e - l
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 5:
            stat_w, p_w = stats.wilcoxon(nonzero, alternative="two-sided")
        else:
            stat_w, p_w = np.nan, np.nan
        expert_vs_llm_ratio.append({
            "condition": cond,
            "n_paired": int(len(common)),
            "expert_mean": round(float(e.mean()), 4),
            "llm_mean": round(float(l.mean()), 4),
            "direction": "expert < LLM" if e.mean() < l.mean() else "expert > LLM",
            "mann_whitney_U": round(float(stat_u), 2),
            "mann_whitney_p": float(f"{p_u:.2e}") if p_u < 0.001 else round(float(p_u), 4),
            "rank_biserial_r": round(float(rb), 4),
            "wilcoxon_W": round(float(stat_w), 2) if not np.isnan(stat_w) else None,
            "wilcoxon_p": float(f"{p_w:.2e}") if (not np.isnan(p_w) and p_w < 0.001) else (round(float(p_w), 4) if not np.isnan(p_w) else None),
        })
    results["gen_ratio_expert_vs_llm"] = expert_vs_llm_ratio

    # --- B) Absolute eval calls: Expert vs each LLM (one-sided: expert > LLM) ---
    expert_vs_llm_eval = []
    for cond in LLM_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        llm_eval = llm.set_index("task_id")["n_eval"].dropna()
        common = expert_eval.index.intersection(llm_eval.index)
        if len(common) < 5:
            continue
        e = expert_eval.loc[common].values
        l = llm_eval.loc[common].values
        stat_u, p_u = stats.mannwhitneyu(e, l, alternative="greater")
        rb = _rank_biserial(stat_u, len(e), len(l))
        expert_vs_llm_eval.append({
            "condition": cond,
            "n_paired": int(len(common)),
            "expert_mean_eval": round(float(e.mean()), 2),
            "llm_mean_eval": round(float(l.mean()), 2),
            "ratio_expert_to_llm": round(float(e.mean() / l.mean()), 2) if l.mean() > 0 else None,
            "mann_whitney_U": round(float(stat_u), 2),
            "p_value": float(f"{p_u:.2e}") if p_u < 0.001 else round(float(p_u), 4),
            "rank_biserial_r": round(float(rb), 4),
        })
    results["eval_count_expert_vs_llm"] = expert_vs_llm_eval

    # --- C) Absolute gen calls: Expert vs each LLM (two-sided) ---
    expert_vs_llm_gen = []
    for cond in LLM_CONDITIONS:
        llm = task_df[task_df["condition"] == cond]
        llm_gen = llm.set_index("task_id")["n_gen"].dropna()
        common = expert_gen.index.intersection(llm_gen.index)
        if len(common) < 5:
            continue
        e = expert_gen.loc[common].values
        l = llm_gen.loc[common].values
        stat_u, p_u = stats.mannwhitneyu(e, l, alternative="two-sided")
        rb = _rank_biserial(stat_u, len(e), len(l))
        expert_vs_llm_gen.append({
            "condition": cond,
            "n_paired": int(len(common)),
            "expert_mean_gen": round(float(e.mean()), 2),
            "llm_mean_gen": round(float(l.mean()), 2),
            "mann_whitney_U": round(float(stat_u), 2),
            "p_value": float(f"{p_u:.2e}") if p_u < 0.001 else round(float(p_u), 4),
            "rank_biserial_r": round(float(rb), 4),
        })
    results["gen_count_expert_vs_llm"] = expert_vs_llm_gen

    # --- Kruskal-Wallis across 4 LLM user conditions ---
    user_groups = []
    user_labels = []
    for cond in LLM_USER_CONDITIONS:
        grp = task_df[task_df["condition"] == cond]["gen_ratio"].dropna()
        if len(grp) > 0:
            user_groups.append(grp.values)
            user_labels.append(cond)

    if len(user_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*user_groups)
        results["kruskal_wallis_user_mode"] = {
            "conditions": user_labels,
            "H_statistic": round(float(kw_stat), 4),
            "p_value": float(f"{kw_p:.2e}") if kw_p < 0.001 else round(float(kw_p), 4),
            "n_per_group": [len(g) for g in user_groups],
        }

    # --- Kruskal-Wallis across all 8 LLM conditions ---
    all_groups = []
    all_labels = []
    for cond in LLM_CONDITIONS:
        grp = task_df[task_df["condition"] == cond]["gen_ratio"].dropna()
        if len(grp) > 0:
            all_groups.append(grp.values)
            all_labels.append(cond)

    if len(all_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*all_groups)
        results["kruskal_wallis_all_llm"] = {
            "conditions": all_labels,
            "H_statistic": round(float(kw_stat), 4),
            "p_value": float(f"{kw_p:.2e}") if kw_p < 0.001 else round(float(kw_p), 4),
            "n_per_group": [len(g) for g in all_groups],
        }

    return results


# ───────────────────────────────────────────────────────────────────
# Analysis 3: Generate → Evaluate → Re-generate iteration cycles
# ───────────────────────────────────────────────────────────────────

def _count_iterations(bio_seq: list[str]) -> int:
    """Count G→E→G cycles in bio tool sequence.

    A cycle = Generative tool, then Evaluative tool, then Generative tool again.
    We count transitions G→E that are followed by E→G (possibly with intervening
    same-type tools).
    """
    if len(bio_seq) < 3:
        return 0

    # Encode as G/E sequence
    ge = []
    for t in bio_seq:
        if t in GENERATIVE_TOOLS:
            ge.append("G")
        elif t in EVALUATIVE_TOOLS:
            ge.append("E")

    # Collapse consecutive same letters
    collapsed = []
    for c in ge:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)

    # Count G→E→G patterns
    cycles = 0
    for i in range(len(collapsed) - 2):
        if collapsed[i] == "G" and collapsed[i+1] == "E" and collapsed[i+2] == "G":
            cycles += 1

    return cycles


def analysis3_iterations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-task iteration cycle counts."""
    rows = []
    for _, row in df.iterrows():
        cond = row["condition"]
        tid = row["task_id"]
        result = _load_result(cond, tid)
        if result is None:
            continue
        seq = _tool_seq(result)
        bio = _bio_tools(seq)
        iters = _count_iterations(bio)
        rows.append({
            "task_id": tid,
            "condition": cond,
            "design_approach": row.get("design_approach", "unknown"),
            "n_bio_calls": len(bio),
            "iteration_count": iters,
        })

    iter_df = pd.DataFrame(rows)

    # Condition summary
    cond_rows = []
    for cond, grp in iter_df.groupby("condition"):
        cond_rows.append({
            "condition": cond,
            "mean_iterations": round(grp["iteration_count"].mean(), 3),
            "median_iterations": round(grp["iteration_count"].median(), 1),
            "max_iterations": int(grp["iteration_count"].max()),
            "frac_with_any_cycle": round((grp["iteration_count"] > 0).mean(), 4),
        })
    cond_summary = pd.DataFrame(cond_rows)

    # Expert vs LLMs Mann-Whitney on iteration counts
    expert_iters = iter_df[iter_df["condition"] == "Human Expert"]["iteration_count"]
    llm_iters = iter_df[iter_df["condition"].isin(LLM_CONDITIONS)]["iteration_count"]
    if len(expert_iters) > 5 and len(llm_iters) > 5:
        stat, p = stats.mannwhitneyu(expert_iters, llm_iters, alternative="greater")
        rb = _rank_biserial(stat, len(expert_iters), len(llm_iters))
        cond_summary.attrs["expert_vs_llm_U"] = round(float(stat), 2)
        cond_summary.attrs["expert_vs_llm_p"] = round(float(p), 6)
        cond_summary.attrs["expert_vs_llm_rb"] = round(float(rb), 4)

    return iter_df, cond_summary


# ───────────────────────────────────────────────────────────────────
# Analysis 4: Backbone generation by de novo vs redesign
# ───────────────────────────────────────────────────────────────────

def analysis4_backbone_by_approach(df: pd.DataFrame) -> pd.DataFrame:
    """Backbone gen usage rate, split by design approach."""
    rows = []
    for _, row in df.iterrows():
        cond = row["condition"]
        tid = row["task_id"]
        approach = row.get("design_approach", "unknown")
        result = _load_result(cond, tid)
        if result is None:
            continue
        seq = _tool_seq(result)
        bio = _bio_tools(seq)
        backbone_calls = sum(1 for t in bio if t in BACKBONE_GEN_TOOLS)
        rows.append({
            "task_id": tid,
            "condition": cond,
            "design_approach": approach,
            "n_backbone_calls": backbone_calls,
            "has_backbone": backbone_calls > 0,
            "n_bio_calls": len(bio),
        })

    raw = pd.DataFrame(rows)

    # Condition × approach summary
    summary_rows = []
    for cond in ALL_CONDITIONS:
        for approach in ["de_novo", "redesign"]:
            subset = raw[(raw["condition"] == cond) & (raw["design_approach"] == approach)]
            if len(subset) == 0:
                continue
            summary_rows.append({
                "condition": cond,
                "design_approach": approach,
                "n_tasks": len(subset),
                "frac_using_backbone": round(subset["has_backbone"].mean(), 4),
                "mean_backbone_calls": round(subset["n_backbone_calls"].mean(), 2),
            })

    summary = pd.DataFrame(summary_rows)

    # Statistical test: within de_novo only, Expert vs each LLM
    dn_expert = raw[(raw["condition"] == "Human Expert") & (raw["design_approach"] == "de_novo")]
    test_rows = []
    for cond in LLM_CONDITIONS:
        dn_llm = raw[(raw["condition"] == cond) & (raw["design_approach"] == "de_novo")]
        if len(dn_expert) < 5 or len(dn_llm) < 5:
            continue
        e = dn_expert["has_backbone"].astype(int).values
        l = dn_llm["has_backbone"].astype(int).values
        stat, p = stats.mannwhitneyu(e, l, alternative="greater")
        rb = _rank_biserial(stat, len(e), len(l))
        test_rows.append({
            "condition": cond,
            "expert_backbone_rate": round(float(e.mean()), 4),
            "llm_backbone_rate": round(float(l.mean()), 4),
            "mann_whitney_U": round(float(stat), 2),
            "p_value": float(f"{p:.2e}") if p < 0.001 else round(float(p), 4),
            "rank_biserial_r": round(float(rb), 4),
        })

    summary.attrs["de_novo_tests"] = test_rows
    return summary


# ───────────────────────────────────────────────────────────────────
# Analysis 5: Shannon entropy of tool category distribution
# ───────────────────────────────────────────────────────────────────

def _shannon_entropy(counts: Counter, n_categories: int = 5) -> float:
    """Shannon entropy in bits from a counter."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * np.log2(p)
    return h


def analysis5_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-task Shannon entropy of functional category distribution."""
    rows = []
    for _, row in df.iterrows():
        cond = row["condition"]
        tid = row["task_id"]
        result = _load_result(cond, tid)
        if result is None:
            continue
        seq = _tool_seq(result)
        bio = _bio_tools(seq)

        # Count per category
        cat_counts = Counter()
        for t in bio:
            cat_counts[_categorize(t)] += 1

        h = _shannon_entropy(cat_counts)
        n_cats_used = sum(1 for v in cat_counts.values() if v > 0)

        rows.append({
            "task_id": tid,
            "condition": cond,
            "entropy": round(h, 4),
            "n_categories_used": n_cats_used,
            "n_bio_calls": len(bio),
            "cat_backbone_gen": cat_counts.get("backbone_gen", 0),
            "cat_seq_design": cat_counts.get("seq_design", 0),
            "cat_struct_pred": cat_counts.get("struct_pred", 0),
            "cat_scoring": cat_counts.get("scoring", 0),
            "cat_other": cat_counts.get("other", 0),
        })

    ent_df = pd.DataFrame(rows)

    # Condition summary
    cond_rows = []
    for cond, grp in ent_df.groupby("condition"):
        cond_rows.append({
            "condition": cond,
            "mean_entropy": round(grp["entropy"].mean(), 4),
            "median_entropy": round(grp["entropy"].median(), 4),
            "mean_n_categories": round(grp["n_categories_used"].mean(), 2),
        })
    cond_summary = pd.DataFrame(cond_rows)
    return cond_summary


# ───────────────────────────────────────────────────────────────────
# Analysis 6: First-generative-tool latency
# ───────────────────────────────────────────────────────────────────

def analysis6_latency(df: pd.DataFrame) -> pd.DataFrame:
    """Position of first generative tool in the bio call sequence."""
    rows = []
    for _, row in df.iterrows():
        cond = row["condition"]
        tid = row["task_id"]
        result = _load_result(cond, tid)
        if result is None:
            continue
        seq = _tool_seq(result)
        bio = _bio_tools(seq)

        # Position of first generative tool (1-indexed) in bio sequence
        first_gen_pos = None
        for i, t in enumerate(bio):
            if t in GENERATIVE_TOOLS:
                first_gen_pos = i + 1  # 1-indexed
                break

        # Also measure position in full sequence
        first_gen_full = None
        for i, t in enumerate(seq):
            if t in GENERATIVE_TOOLS:
                first_gen_full = i + 1
                break

        rows.append({
            "task_id": tid,
            "condition": cond,
            "first_gen_bio_pos": first_gen_pos,  # None if never called
            "first_gen_full_pos": first_gen_full,
            "n_bio_calls": len(bio),
            "n_total_calls": len(seq),
            "gen_ever_called": first_gen_pos is not None,
        })

    lat_df = pd.DataFrame(rows)

    # Condition summary
    cond_rows = []
    for cond, grp in lat_df.groupby("condition"):
        called = grp[grp["gen_ever_called"]]
        not_called_frac = 1.0 - grp["gen_ever_called"].mean()
        cond_rows.append({
            "condition": cond,
            "frac_gen_never_called": round(not_called_frac, 4),
            "mean_first_gen_bio_pos": round(called["first_gen_bio_pos"].mean(), 2) if len(called) > 0 else None,
            "median_first_gen_bio_pos": round(called["first_gen_bio_pos"].median(), 1) if len(called) > 0 else None,
            "mean_first_gen_full_pos": round(called["first_gen_full_pos"].mean(), 2) if len(called) > 0 else None,
        })
    cond_summary = pd.DataFrame(cond_rows)
    return cond_summary


# ───────────────────────────────────────────────────────────────────
# Console output
# ───────────────────────────────────────────────────────────────────

def print_all(a1_cond, a2_stats, a3_cond, a4_summary, a5_summary, a6_summary):
    sep = "=" * 90

    # ── Analysis 1 ──
    print(f"\n{sep}")
    print("Analysis 1: Generative vs Evaluative Tool Call Ratio")
    print(sep)
    print(f"{'Condition':<28s} {'N':>3s} {'MeanGen':>8s} {'MeanEval':>9s} "
          f"{'GenRatio':>9s} {'TaskAvg':>8s}")
    print("-" * 75)
    for _, r in a1_cond.sort_values("gen_ratio", ascending=False).iterrows():
        print(f"{r['condition']:<28s} {r['n_tasks']:>3.0f} "
              f"{r['mean_gen']:>8.2f} {r['mean_eval']:>9.2f} "
              f"{r['gen_ratio']:>9.4f} {r['mean_task_gen_ratio']:>8.4f}")

    # ── Analysis 2 ──
    print(f"\n{sep}")
    print("Analysis 2a: gen_ratio (Expert vs LLM, two-sided)")
    print(sep)
    tests = a2_stats.get("gen_ratio_expert_vs_llm", [])
    if tests:
        print(f"{'Condition':<28s} {'N':>3s} {'Exp':>6s} {'LLM':>6s} {'Dir':<14s} "
              f"{'U':>8s} {'p':>10s} {'r_rb':>7s}")
        print("-" * 85)
        for t in tests:
            print(f"{t['condition']:<28s} {t['n_paired']:>3d} "
                  f"{t['expert_mean']:>6.3f} {t['llm_mean']:>6.3f} {t['direction']:<14s} "
                  f"{t['mann_whitney_U']:>8.1f} {t['mann_whitney_p']:>10} "
                  f"{t['rank_biserial_r']:>7.3f}")

    print(f"\n{sep}")
    print("Analysis 2b: Absolute eval calls (Expert > LLM, one-sided)")
    print(sep)
    tests_eval = a2_stats.get("eval_count_expert_vs_llm", [])
    if tests_eval:
        print(f"{'Condition':<28s} {'N':>3s} {'Exp':>6s} {'LLM':>6s} {'×':>5s} "
              f"{'U':>8s} {'p':>10s} {'r_rb':>7s}")
        print("-" * 80)
        for t in tests_eval:
            ratio_str = f"{t['ratio_expert_to_llm']:>5.1f}" if t.get("ratio_expert_to_llm") else "  N/A"
            print(f"{t['condition']:<28s} {t['n_paired']:>3d} "
                  f"{t['expert_mean_eval']:>6.1f} {t['llm_mean_eval']:>6.1f} {ratio_str} "
                  f"{t['mann_whitney_U']:>8.1f} {t['p_value']:>10} "
                  f"{t['rank_biserial_r']:>7.3f}")

    print(f"\n{sep}")
    print("Analysis 2c: Absolute gen calls (Expert vs LLM, two-sided)")
    print(sep)
    tests_gen = a2_stats.get("gen_count_expert_vs_llm", [])
    if tests_gen:
        print(f"{'Condition':<28s} {'N':>3s} {'Exp':>6s} {'LLM':>6s} "
              f"{'U':>8s} {'p':>10s} {'r_rb':>7s}")
        print("-" * 75)
        for t in tests_gen:
            print(f"{t['condition']:<28s} {t['n_paired']:>3d} "
                  f"{t['expert_mean_gen']:>6.1f} {t['llm_mean_gen']:>6.1f} "
                  f"{t['mann_whitney_U']:>8.1f} {t['p_value']:>10} "
                  f"{t['rank_biserial_r']:>7.3f}")

    kw = a2_stats.get("kruskal_wallis_user_mode")
    if kw:
        print(f"\nKruskal-Wallis (4 LLM user conditions): H={kw['H_statistic']:.4f}, p={kw['p_value']}")
    kw_all = a2_stats.get("kruskal_wallis_all_llm")
    if kw_all:
        print(f"Kruskal-Wallis (all 8 LLM conditions):  H={kw_all['H_statistic']:.4f}, p={kw_all['p_value']}")

    # ── Analysis 3 ──
    print(f"\n{sep}")
    print("Analysis 3: Generate→Evaluate→Re-generate Iteration Cycles")
    print(sep)
    print(f"{'Condition':<28s} {'Mean':>6s} {'Med':>5s} {'Max':>4s} {'%Any':>6s}")
    print("-" * 55)
    for _, r in a3_cond.sort_values("mean_iterations", ascending=False).iterrows():
        print(f"{r['condition']:<28s} {r['mean_iterations']:>6.3f} "
              f"{r['median_iterations']:>5.1f} {r['max_iterations']:>4d} "
              f"{r['frac_with_any_cycle']*100:>5.1f}%")
    if hasattr(a3_cond, "attrs"):
        u = a3_cond.attrs.get("expert_vs_llm_U")
        p = a3_cond.attrs.get("expert_vs_llm_p")
        rb = a3_cond.attrs.get("expert_vs_llm_rb")
        if u is not None:
            print(f"\nExpert vs All LLMs: U={u}, p={p:.6f}, r_rb={rb}")

    # ── Analysis 4 ──
    print(f"\n{sep}")
    print("Analysis 4: Backbone Generation Usage (de novo vs redesign)")
    print(sep)
    print(f"{'Condition':<28s} {'Approach':<10s} {'N':>3s} {'%Using':>7s} {'MeanCalls':>10s}")
    print("-" * 65)
    for _, r in a4_summary.iterrows():
        print(f"{r['condition']:<28s} {r['design_approach']:<10s} "
              f"{r['n_tasks']:>3.0f} {r['frac_using_backbone']*100:>6.1f}% "
              f"{r['mean_backbone_calls']:>10.2f}")

    dn_tests = a4_summary.attrs.get("de_novo_tests", [])
    if dn_tests:
        print(f"\n  De novo only — Expert vs LLM backbone usage:")
        print(f"  {'Condition':<28s} {'Exp':>6s} {'LLM':>6s} {'U':>8s} {'p':>10s} {'r_rb':>7s}")
        for t in dn_tests:
            print(f"  {t['condition']:<28s} {t['expert_backbone_rate']:>6.3f} "
                  f"{t['llm_backbone_rate']:>6.3f} {t['mann_whitney_U']:>8.1f} "
                  f"{t['p_value']:>10} {t['rank_biserial_r']:>7.3f}")

    # ── Analysis 5 ──
    print(f"\n{sep}")
    print("Analysis 5: Tool Category Shannon Entropy")
    print(sep)
    print(f"{'Condition':<28s} {'MeanH':>7s} {'MedH':>7s} {'MeanCats':>9s}")
    print("-" * 55)
    for _, r in a5_summary.sort_values("mean_entropy", ascending=False).iterrows():
        print(f"{r['condition']:<28s} {r['mean_entropy']:>7.3f} "
              f"{r['median_entropy']:>7.3f} {r['mean_n_categories']:>9.2f}")

    # ── Analysis 6 ──
    print(f"\n{sep}")
    print("Analysis 6: First Generative Tool Latency")
    print(sep)
    print(f"{'Condition':<28s} {'%NeverGen':>10s} {'MeanPos':>8s} {'MedPos':>7s} {'FullPos':>8s}")
    print("-" * 65)
    for _, r in a6_summary.sort_values("frac_gen_never_called").iterrows():
        mp = f"{r['mean_first_gen_bio_pos']:>8.2f}" if r['mean_first_gen_bio_pos'] is not None else "     N/A"
        mdp = f"{r['median_first_gen_bio_pos']:>7.1f}" if r['median_first_gen_bio_pos'] is not None else "    N/A"
        fp = f"{r['mean_first_gen_full_pos']:>8.2f}" if r['mean_first_gen_full_pos'] is not None else "     N/A"
        print(f"{r['condition']:<28s} {r['frac_gen_never_called']*100:>9.1f}% "
              f"{mp} {mdp} {fp}")

    print()


# ───────────────────────────────────────────────────────────────────
# JSON serialization
# ───────────────────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """Convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if pd.isna(obj):
        return None
    return obj


class _Encoder(json.JSONEncoder):
    def default(self, o):
        v = _to_json_safe(o)
        if v is not o:
            return v
        return super().default(o)


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def main():
    print("[1/7] Loading results...")
    df = load_all()
    df = df[df["condition"].isin(ALL_CONDITIONS)].copy()
    print(f"  {len(df)} rows ({df['task_id'].nunique()} tasks × {df['condition'].nunique()} conditions)")

    print("[2/7] Analysis 1: Generative vs Evaluative ratio...")
    task_df, a1_cond = analysis1_gen_eval_ratio(df)

    print("[3/7] Analysis 2: Statistical tests...")
    a2 = analysis2_stats(task_df)

    print("[4/7] Analysis 3: Iteration cycles...")
    iter_df, a3_cond = analysis3_iterations(df)

    print("[5/7] Analysis 4: Backbone generation by approach...")
    a4 = analysis4_backbone_by_approach(df)
    a4_dn_tests = a4.attrs.get("de_novo_tests", [])

    print("[6/7] Analysis 5: Shannon entropy...")
    a5 = analysis5_entropy(df)

    print("[7/7] Analysis 6: First-generative latency...")
    a6 = analysis6_latency(df)

    # Print summary tables
    print_all(a1_cond, a2, a3_cond, a4, a5, a6)

    # Build JSON output
    output = {
        "analysis_1_gen_eval_ratio": {
            "per_condition": a1_cond.to_dict(orient="records"),
            "description": "Generative vs evaluative tool call counts and ratios per condition",
        },
        "analysis_2_statistical_tests": a2,
        "analysis_3_iteration_cycles": {
            "per_condition": a3_cond.to_dict(orient="records"),
            "expert_vs_llm": {
                "U": a3_cond.attrs.get("expert_vs_llm_U"),
                "p": a3_cond.attrs.get("expert_vs_llm_p"),
                "rank_biserial_r": a3_cond.attrs.get("expert_vs_llm_rb"),
            },
        },
        "analysis_4_backbone_by_approach": {
            "per_condition_approach": a4.to_dict(orient="records"),
            "de_novo_expert_vs_llm": a4_dn_tests,
        },
        "analysis_5_shannon_entropy": {
            "per_condition": a5.to_dict(orient="records"),
        },
        "analysis_6_first_gen_latency": {
            "per_condition": a6.to_dict(orient="records"),
        },
        "metadata": {
            "n_tasks": int(df["task_id"].nunique()),
            "n_conditions": int(df["condition"].nunique()),
            "conditions": ALL_CONDITIONS,
            "n_de_novo": int(df[df["condition"] == "Human Expert"]["design_approach"].value_counts().get("de_novo", 0)),
            "n_redesign": int(df[df["condition"] == "Human Expert"]["design_approach"].value_counts().get("redesign", 0)),
        },
    }

    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "section26_gen_vs_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=_Encoder)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
