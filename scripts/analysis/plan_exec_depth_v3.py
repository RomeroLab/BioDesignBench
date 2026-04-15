#!/usr/bin/env python3
"""Plan-Execution Depth v3: Continuous metrics beyond binary plan/exec.

Extends PROMPT_02's binary plan-execution gap with continuous depth metrics:
- Stage-level tool call counts (execution depth)
- Plan specificity (0-3 scale)
- Evaluation Multiplier (EM): eval calls per gen call
- Execution Depth Ratio (EDR): agent depth / expert depth
- Multi-Metric Validation Score (MMVS): diversity of evaluation tools
- Plan-Depth Gap (PDG): plan specificity - normalized depth
- Extended case classification: A+ / A / B / C / D

Outputs:
  results/analysis/plan_exec_depth_v3.csv              - per task x condition x stage
  results/analysis/plan_exec_depth_v3_summary.csv      - per condition aggregated
  results/analysis/plan_exec_depth_v3_stats.csv        - statistical test results
  results/analysis/plan_exec_depth_v3_guided_effect.csv - guided mode effect
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, spearmanr

from scripts.analysis.load_results import CONDITION_MAP, EXCLUDED_TASKS, load_all

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEPS = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]

REFERENCE_PIPELINE = {
    "de_novo": ALL_STEPS,
    "redesign": ["sequence_design", "structure_prediction", "scoring_validation"],
}

STAGE_TOOLS: dict[str, list[str]] = {
    "backbone_generation": ["generate_backbone", "design_binder"],
    "sequence_design": ["optimize_sequence", "rosetta_design"],
    "structure_prediction": [
        "predict_structure", "predict_structure_boltz",
        "predict_complex", "predict_affinity_boltz",
    ],
    "scoring_validation": [
        "score_stability", "rosetta_score", "rosetta_relax",
        "analyze_interface", "rosetta_interface_score",
        "energy_minimize", "validate_design", "suggest_hotspots",
    ],
}

# Flatten for quick lookup
_STAGE_LOOKUP: dict[str, str] = {}
for _stage, _tools in STAGE_TOOLS.items():
    for _t in _tools:
        _STAGE_LOOKUP[_t] = _stage

GENERATIVE_TOOLS = frozenset({
    "generate_backbone", "design_binder",
    "optimize_sequence", "rosetta_design",
})

EVALUATIVE_TOOLS = frozenset({
    "predict_structure", "predict_structure_boltz",
    "predict_complex", "predict_affinity_boltz",
    "score_stability", "rosetta_score", "rosetta_relax",
    "analyze_interface", "rosetta_interface_score",
    "energy_minimize", "validate_design", "suggest_hotspots",
})

PLAN_SPECIFICITY_INDICATORS: dict[str, dict[str, list[str]]] = {
    "backbone_generation": {
        "level_1_mention": ["backbone", "rfdiffusion", "generate"],
        "level_2_parameterized": [
            "contig", "hotspot", "num_designs", "diffusion_steps", "contiguous_weight",
        ],
        "level_3_iterated": [
            "multiple backbones", "generate several", "sample",
            "diverse candidates", "num_designs=", "batch",
        ],
    },
    "sequence_design": {
        "level_1_mention": ["proteinmpnn", "sequence design", "inverse folding"],
        "level_2_parameterized": ["sampling_temp", "num_seq_per_target", "temperature"],
        "level_3_iterated": [
            "multiple sequences", "sample several", "sequence diversity",
        ],
    },
    "structure_prediction": {
        "level_1_mention": ["alphafold", "structure prediction", "fold", "boltz"],
        "level_2_parameterized": ["num_models", "num_recycles", "multimer"],
        "level_3_iterated": [
            "predict all candidates", "screen", "compare structures",
        ],
    },
    "scoring_validation": {
        "level_1_mention": ["score", "rosetta", "energy", "validate", "plddt"],
        "level_2_parameterized": ["relax", "interface_score", "ddg", "sasa"],
        "level_3_iterated": [
            "rank candidates", "filter by", "threshold", "top-k",
            "compare scores", "select best",
        ],
    },
}

EVAL_TOOL_CATEGORIES: dict[str, list[str]] = {
    "structure": ["predict_structure", "predict_structure_boltz", "predict_complex"],
    "energy": ["rosetta_score", "rosetta_relax", "energy_minimize"],
    "stability": ["score_stability"],
    "interface": ["analyze_interface", "rosetta_interface_score"],
    "affinity": ["predict_affinity_boltz"],
    "composite": ["validate_design"],
}

# Flatten category lookup
_EVAL_CAT_LOOKUP: dict[str, str] = {}
for _cat, _tools in EVAL_TOOL_CATEGORIES.items():
    for _t in _tools:
        _EVAL_CAT_LOOKUP[_t] = _cat

SCRIPTED_BASELINES = {"Oracle", "Human Expert", "Hardcoded Pipeline"}

LLM_CONDITIONS = [
    "DeepSeek V3 user", "DeepSeek V3 benchmark",
    "GPT-5 user", "GPT-5 benchmark",
    "Sonnet 4.5 user", "Sonnet 4.5 benchmark",
    "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark",
]

LLM_PAIRS = [
    ("DeepSeek V3", "DeepSeek V3 user", "DeepSeek V3 benchmark"),
    ("GPT-5", "GPT-5 user", "GPT-5 benchmark"),
    ("Sonnet 4.5", "Sonnet 4.5 user", "Sonnet 4.5 benchmark"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark"),
]


# ---------------------------------------------------------------------------
# Core computation functions (unit-testable)
# ---------------------------------------------------------------------------

def count_stage_depth(tool_call_log: list[dict], stage: str) -> dict[str, int]:
    """Count tool calls belonging to a stage. Returns depth and unique tool count."""
    stage_tools = set(STAGE_TOOLS.get(stage, []))
    calls = [tc["tool"] for tc in tool_call_log if tc.get("tool") in stage_tools]
    return {
        "depth": len(calls),
        "unique_tools": len(set(calls)),
    }


def score_plan_specificity(reasoning: str, stage: str) -> int:
    """Score plan specificity for a stage: 0 (none), 1 (mention), 2 (params), 3 (iterated)."""
    indicators = PLAN_SPECIFICITY_INDICATORS.get(stage, {})
    text = reasoning.lower()

    level_3 = indicators.get("level_3_iterated", [])
    level_2 = indicators.get("level_2_parameterized", [])
    level_1 = indicators.get("level_1_mention", [])

    if any(kw in text for kw in level_3):
        return 3
    if any(kw in text for kw in level_2):
        return 2
    if any(kw in text for kw in level_1):
        return 1
    return 0


def compute_eval_multiplier(tool_call_log: list[dict]) -> float:
    """Evaluation Multiplier: eval calls / max(gen calls, 1)."""
    n_gen = sum(1 for tc in tool_call_log if tc.get("tool") in GENERATIVE_TOOLS)
    n_eval = sum(1 for tc in tool_call_log if tc.get("tool") in EVALUATIVE_TOOLS)
    return n_eval / max(n_gen, 1)


def compute_mmvs(tool_call_log: list[dict]) -> int:
    """Multi-Metric Validation Score: number of distinct eval tool categories used."""
    categories = set()
    for tc in tool_call_log:
        cat = _EVAL_CAT_LOOKUP.get(tc.get("tool", ""))
        if cat:
            categories.add(cat)
    return len(categories)


def normalize_depth(depth: int) -> float:
    """Normalize depth to 0-3 scale: 0→0, 1→1, 2-3→2, 4+→3 (log2-based)."""
    if depth <= 0:
        return 0.0
    if depth <= 1:
        return 1.0
    return min(1 + math.log2(depth), 3.0)


def classify_extended_case(
    plan_specificity: int, depth: int, median_expert_depth: float,
) -> str:
    """Extended case classification: A+ / A / B / C / D."""
    if plan_specificity == 0 and depth == 0:
        return "C"
    if plan_specificity == 0 and depth >= 1:
        return "D"
    if plan_specificity >= 1 and depth == 0:
        return "B"
    # plan >= 1 and depth >= 1
    if plan_specificity >= 2 and depth >= median_expert_depth:
        return "A+"
    return "A"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_result_json(condition: str, task_id: str) -> dict[str, Any] | None:
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


def _get_reasoning_trace(result: dict) -> str:
    return result.get("raw_output", {}).get("reasoning_trace", "")


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


def _get_subject(task_id: str) -> str | None:
    try:
        from biodesignbench.taxonomy import get_category
        cat = get_category(task_id)
        if cat is not None:
            return cat.subject.value
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def compute_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-task x condition x stage depth metrics."""
    rows = []

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
        reasoning = _get_reasoning_trace(result)
        is_baseline = condition in SCRIPTED_BASELINES
        ref_steps = REFERENCE_PIPELINE[approach]
        subject = _get_subject(task_id) or "unknown"

        # Task-level metrics
        em = compute_eval_multiplier(tcl)
        mmvs = compute_mmvs(tcl)
        n_gen = sum(1 for tc in tcl if tc.get("tool") in GENERATIVE_TOOLS)
        n_eval = sum(1 for tc in tcl if tc.get("tool") in EVALUATIVE_TOOLS)
        total_score = row.get("total", 0)

        for stage in ref_steps:
            sd = count_stage_depth(tcl, stage)
            depth = sd["depth"]
            unique_tools = sd["unique_tools"]

            # Binary exec (for backward compat with v2)
            binary_exec = 1 if depth > 0 else 0

            # Plan specificity
            if is_baseline:
                # Scripted baselines: assume level 3 (they are hand-crafted pipelines)
                plan_specificity = 3
                binary_plan = 1
            else:
                plan_specificity = score_plan_specificity(reasoning, stage)
                binary_plan = 1 if plan_specificity >= 1 else 0

            rows.append({
                "task_id": task_id,
                "condition": condition,
                "approach": approach,
                "subject": subject,
                "stage": stage,
                "binary_plan": binary_plan,
                "plan_specificity": plan_specificity,
                "binary_exec": binary_exec,
                "exec_depth": depth,
                "unique_tools_in_stage": unique_tools,
                "eval_multiplier": round(em, 3),
                "mmvs": mmvs,
                "n_gen_calls": n_gen,
                "n_eval_calls": n_eval,
                "total_score": total_score,
                # extended_case filled later (needs expert median)
                "extended_case": "",
            })

    return pd.DataFrame(rows)


def fill_extended_cases(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Fill extended_case using Human Expert median depth per stage."""
    df = detail_df.copy()

    # Compute expert median depth per stage
    expert = df[df["condition"] == "Human Expert"]
    expert_median = {}
    for stage in ALL_STEPS:
        stage_depths = expert.loc[
            (expert["stage"] == stage) & (expert["exec_depth"] > 0),
            "exec_depth",
        ]
        expert_median[stage] = stage_depths.median() if len(stage_depths) > 0 else 1.0

    for idx, row in df.iterrows():
        med = expert_median.get(row["stage"], 1.0)
        df.at[idx, "extended_case"] = classify_extended_case(
            row["plan_specificity"], row["exec_depth"], med,
        )

    return df


def compute_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per condition."""
    # Expert median depth per stage (for EDR)
    expert = detail_df[detail_df["condition"] == "Human Expert"]
    expert_mean_depth = {}
    for stage in ALL_STEPS:
        vals = expert.loc[
            (expert["stage"] == stage) & (expert["exec_depth"] > 0),
            "exec_depth",
        ]
        expert_mean_depth[stage] = vals.mean() if len(vals) > 0 else 1.0

    rows = []
    for condition, grp in detail_df.groupby("condition"):
        n_tasks = grp["task_id"].nunique()
        n_rows = len(grp)  # stage-level rows

        # Plan specificity (mean across all stage-rows)
        mean_plan_spec = grp["plan_specificity"].mean()

        # Exec depth (mean across all stage-rows)
        mean_exec_depth = grp["exec_depth"].mean()

        # EM and MMVS (task-level, deduplicate by taking first per task)
        task_grp = grp.drop_duplicates(subset=["task_id"])
        mean_em = task_grp["eval_multiplier"].mean()
        mean_mmvs = task_grp["mmvs"].mean()

        # Extended case fractions
        total_steps = len(grp)
        case_counts = grp["extended_case"].value_counts()
        frac_a_plus = case_counts.get("A+", 0) / total_steps
        frac_a = case_counts.get("A", 0) / total_steps
        frac_b = case_counts.get("B", 0) / total_steps
        frac_c = case_counts.get("C", 0) / total_steps
        frac_d = case_counts.get("D", 0) / total_steps

        # EDR per stage
        edr = {}
        for stage in ALL_STEPS:
            stage_rows = grp[(grp["stage"] == stage) & (grp["binary_exec"] == 1)]
            if len(stage_rows) > 0:
                agent_mean = stage_rows["exec_depth"].mean()
                edr[stage] = round(agent_mean / expert_mean_depth.get(stage, 1.0), 3)
            else:
                edr[stage] = 0.0

        rows.append({
            "condition": condition,
            "n_tasks": n_tasks,
            "mean_plan_specificity": round(mean_plan_spec, 3),
            "mean_exec_depth": round(mean_exec_depth, 3),
            "mean_eval_multiplier": round(mean_em, 3),
            "mean_mmvs": round(mean_mmvs, 3),
            "frac_case_A_plus": round(frac_a_plus, 4),
            "frac_case_A": round(frac_a, 4),
            "frac_case_B": round(frac_b, 4),
            "frac_case_C": round(frac_c, 4),
            "frac_case_D": round(frac_d, 4),
            "EDR_backbone": edr.get("backbone_generation", 0.0),
            "EDR_seqdesign": edr.get("sequence_design", 0.0),
            "EDR_structpred": edr.get("structure_prediction", 0.0),
            "EDR_scoring": edr.get("scoring_validation", 0.0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _safe_mannwhitney(x, y, alternative="two-sided"):
    """Mann-Whitney U with safe handling for small/identical samples."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    if np.std(x) == 0 and np.std(y) == 0 and np.mean(x) == np.mean(y):
        return np.nan, 1.0, 0.0
    try:
        U, p = mannwhitneyu(x, y, alternative=alternative)
        r = 1 - (2 * U) / (len(x) * len(y))  # rank-biserial effect size
        return U, p, r
    except ValueError:
        return np.nan, np.nan, np.nan


def _cohens_d(x, y):
    """Cohen's d effect size."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def _bonferroni(p_values: list[float]) -> list[float]:
    """Bonferroni correction."""
    n = len(p_values)
    return [min(p * n, 1.0) if not np.isnan(p) else np.nan for p in p_values]


def compute_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all statistical tests."""
    stat_rows = []

    # Get data organized by condition
    expert_rows = detail_df[detail_df["condition"] == "Human Expert"]
    llm_user_conditions = [c for c in LLM_CONDITIONS if c.endswith(" user")]

    # ── 5a: Execution Depth per stage ──
    raw_p_values = []
    for stage in ALL_STEPS:
        expert_depths = expert_rows.loc[expert_rows["stage"] == stage, "exec_depth"].values
        for cond in llm_user_conditions:
            llm_depths = detail_df.loc[
                (detail_df["condition"] == cond) & (detail_df["stage"] == stage),
                "exec_depth",
            ].values
            U, p, r = _safe_mannwhitney(expert_depths, llm_depths)
            raw_p_values.append(p)
            stat_rows.append({
                "test_name": "exec_depth_by_stage",
                "comparison": f"Expert vs {cond} [{stage}]",
                "statistic": round(U, 2) if not np.isnan(U) else np.nan,
                "p_value": p,
                "effect_size": round(r, 3) if not np.isnan(r) else np.nan,
                "interpretation": "",
            })

    # Apply Bonferroni correction
    corrected = _bonferroni(raw_p_values)
    for i, row in enumerate(stat_rows[-len(raw_p_values):]):
        row["p_value_corrected"] = corrected[i]
        if not np.isnan(corrected[i]):
            row["interpretation"] = (
                "significant" if corrected[i] < 0.05 else "not significant"
            )

    # ── 5b: Evaluation Multiplier ──
    expert_tasks = expert_rows.drop_duplicates(subset=["task_id"])
    expert_em = expert_tasks["eval_multiplier"].values

    raw_p_values = []
    for cond in llm_user_conditions:
        llm_tasks = detail_df[detail_df["condition"] == cond].drop_duplicates(subset=["task_id"])
        llm_em = llm_tasks["eval_multiplier"].values
        U, p, r = _safe_mannwhitney(expert_em, llm_em)
        d = _cohens_d(expert_em, llm_em)
        raw_p_values.append(p)
        stat_rows.append({
            "test_name": "eval_multiplier",
            "comparison": f"Expert vs {cond}",
            "statistic": round(U, 2) if not np.isnan(U) else np.nan,
            "p_value": p,
            "p_value_corrected": np.nan,
            "effect_size": round(d, 3) if not np.isnan(d) else np.nan,
            "interpretation": "",
        })
    corrected = _bonferroni(raw_p_values)
    for i, idx in enumerate(range(len(stat_rows) - len(raw_p_values), len(stat_rows))):
        stat_rows[idx]["p_value_corrected"] = corrected[i]
        if not np.isnan(corrected[i]):
            stat_rows[idx]["interpretation"] = (
                "significant" if corrected[i] < 0.05 else "not significant"
            )

    # ── 5c: MMVS ──
    expert_mmvs = expert_tasks["mmvs"].values

    raw_p_values = []
    for cond in llm_user_conditions:
        llm_tasks = detail_df[detail_df["condition"] == cond].drop_duplicates(subset=["task_id"])
        llm_mmvs = llm_tasks["mmvs"].values
        U, p, r = _safe_mannwhitney(expert_mmvs, llm_mmvs)
        raw_p_values.append(p)
        stat_rows.append({
            "test_name": "mmvs",
            "comparison": f"Expert vs {cond}",
            "statistic": round(U, 2) if not np.isnan(U) else np.nan,
            "p_value": p,
            "p_value_corrected": np.nan,
            "effect_size": round(r, 3) if not np.isnan(r) else np.nan,
            "interpretation": "",
        })
    corrected = _bonferroni(raw_p_values)
    for i, idx in enumerate(range(len(stat_rows) - len(raw_p_values), len(stat_rows))):
        stat_rows[idx]["p_value_corrected"] = corrected[i]
        if not np.isnan(corrected[i]):
            stat_rows[idx]["interpretation"] = (
                "significant" if corrected[i] < 0.05 else "not significant"
            )

    # ── 5d: Fisher exact — A+ vs A fraction ──
    expert_a_plus = len(expert_rows[expert_rows["extended_case"] == "A+"])
    expert_a = len(expert_rows[expert_rows["extended_case"] == "A"])

    for cond in llm_user_conditions:
        cond_rows = detail_df[detail_df["condition"] == cond]
        llm_a_plus = len(cond_rows[cond_rows["extended_case"] == "A+"])
        llm_a = len(cond_rows[cond_rows["extended_case"] == "A"])
        if (expert_a_plus + expert_a) > 0 and (llm_a_plus + llm_a) > 0:
            table = [[expert_a_plus, expert_a], [llm_a_plus, llm_a]]
            odds_ratio, p = fisher_exact(table)
        else:
            odds_ratio, p = np.nan, np.nan
        stat_rows.append({
            "test_name": "fisher_A_plus_vs_A",
            "comparison": f"Expert vs {cond}",
            "statistic": round(odds_ratio, 3) if not np.isnan(odds_ratio) else np.nan,
            "p_value": p,
            "p_value_corrected": np.nan,
            "effect_size": round(odds_ratio, 3) if not np.isnan(odds_ratio) else np.nan,
            "interpretation": f"p={'<0.05' if (not np.isnan(p) and p < 0.05) else '>=0.05'}",
        })

    # ── 5e: Kruskal-Wallis across LLMs (excluding Gemini) ──
    no_gemini = [c for c in llm_user_conditions if "Gemini" not in c]
    groups = []
    for cond in no_gemini:
        tasks = detail_df[detail_df["condition"] == cond].drop_duplicates(subset=["task_id"])
        groups.append(tasks["eval_multiplier"].values)
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        try:
            H, p = kruskal(*groups)
            stat_rows.append({
                "test_name": "kruskal_llm_no_gemini",
                "comparison": " vs ".join(no_gemini),
                "statistic": round(H, 3),
                "p_value": p,
                "p_value_corrected": np.nan,
                "effect_size": np.nan,
                "interpretation": "significant" if p < 0.05 else "not significant",
            })
        except ValueError:
            pass

    # ── 5f: Spearman correlation — depth vs score ──
    all_tasks = detail_df.drop_duplicates(subset=["task_id", "condition"])
    # Total depth per (task, condition)
    total_depth = detail_df.groupby(["task_id", "condition"])["exec_depth"].sum().reset_index()
    total_depth.columns = ["task_id", "condition", "total_depth"]
    merged = all_tasks.merge(total_depth, on=["task_id", "condition"], how="left")

    for metric_name, col in [("total_depth", "total_depth"),
                              ("eval_multiplier", "eval_multiplier"),
                              ("mmvs", "mmvs")]:
        vals = merged[[col, "total_score"]].dropna()
        if len(vals) >= 10:
            rho, p = spearmanr(vals[col], vals["total_score"])
            stat_rows.append({
                "test_name": "spearman_vs_score",
                "comparison": f"{metric_name} vs total_score",
                "statistic": round(rho, 3),
                "p_value": p,
                "p_value_corrected": np.nan,
                "effect_size": round(rho, 3),
                "interpretation": f"rho={rho:.3f}, {'sig' if p < 0.05 else 'ns'}",
            })

    # Per-stage depth vs score
    for stage in ALL_STEPS:
        stage_data = detail_df[detail_df["stage"] == stage][["task_id", "condition", "exec_depth", "total_score"]].dropna()
        if len(stage_data) >= 10:
            rho, p = spearmanr(stage_data["exec_depth"], stage_data["total_score"])
            stat_rows.append({
                "test_name": "spearman_stage_vs_score",
                "comparison": f"{stage} depth vs total_score",
                "statistic": round(rho, 3),
                "p_value": p,
                "p_value_corrected": np.nan,
                "effect_size": round(rho, 3),
                "interpretation": f"rho={rho:.3f}, {'sig' if p < 0.05 else 'ns'}",
            })

    return pd.DataFrame(stat_rows)


# ---------------------------------------------------------------------------
# Guided vs unguided comparison
# ---------------------------------------------------------------------------

def compute_guided_effect(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Compute guided (user) vs unguided (benchmark) depth deltas per model."""
    rows = []
    for model, guided_cond, unguided_cond in LLM_PAIRS:
        guided = detail_df[detail_df["condition"] == guided_cond]
        unguided = detail_df[detail_df["condition"] == unguided_cond]
        if len(guided) == 0 or len(unguided) == 0:
            continue

        # Total depth
        g_total = guided.groupby("task_id")["exec_depth"].sum().mean()
        u_total = unguided.groupby("task_id")["exec_depth"].sum().mean()

        # Per-stage depth
        stage_deltas = {}
        for stage in ALL_STEPS:
            g_stage = guided.loc[guided["stage"] == stage, "exec_depth"].mean()
            u_stage = unguided.loc[unguided["stage"] == stage, "exec_depth"].mean()
            stage_deltas[f"delta_depth_{stage}"] = round(g_stage - u_stage, 3)

        # EM and MMVS
        g_tasks = guided.drop_duplicates(subset=["task_id"])
        u_tasks = unguided.drop_duplicates(subset=["task_id"])
        delta_em = g_tasks["eval_multiplier"].mean() - u_tasks["eval_multiplier"].mean()
        delta_mmvs = g_tasks["mmvs"].mean() - u_tasks["mmvs"].mean()

        row = {
            "model": model,
            "delta_depth_total": round(g_total - u_total, 3),
            "delta_EM": round(delta_em, 3),
            "delta_MMVS": round(delta_mmvs, 3),
        }
        row.update(stage_deltas)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_tables(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    guided_df: pd.DataFrame,
):
    """Print formatted tables for paper."""

    # ── Table 1: Execution Depth by Stage ──
    print("\n" + "=" * 100)
    print("Table 1: Execution Depth by Stage (mean +/- std)")
    print("=" * 100)
    header = f"{'Condition':<28s}"
    for stage in ALL_STEPS:
        short = stage.replace("backbone_generation", "Backbone").replace(
            "sequence_design", "SeqDesign"
        ).replace("structure_prediction", "StructPred").replace(
            "scoring_validation", "Scoring"
        )
        header += f" | {short:>14s}"
    header += f" | {'Total':>8s}"
    print(header)
    print("-" * 100)

    # Sort by mean total depth descending
    cond_order = summary_df.sort_values("mean_exec_depth", ascending=False)["condition"].tolist()
    for condition in cond_order:
        grp = detail_df[detail_df["condition"] == condition]
        line = f"{condition:<28s}"
        total_depth = 0
        for stage in ALL_STEPS:
            vals = grp.loc[grp["stage"] == stage, "exec_depth"]
            m = vals.mean() if len(vals) > 0 else 0
            s = vals.std() if len(vals) > 1 else 0
            total_depth += m
            line += f" | {m:>5.2f} ({s:>4.2f})"
        line += f" | {total_depth:>8.2f}"
        print(line)

    # EDR row
    print("-" * 100)
    line = f"{'EDR vs Expert':<28s}"
    for stage in ALL_STEPS:
        col = f"EDR_{stage.replace('backbone_generation', 'backbone').replace('sequence_design', 'seqdesign').replace('structure_prediction', 'structpred').replace('scoring_validation', 'scoring')}"
        # Average EDR across LLM user conditions
        llm_user = summary_df[summary_df["condition"].str.endswith(" user")]
        if len(llm_user) > 0:
            val = llm_user[col].mean()
            line += f" | {val:>14.3f}"
        else:
            line += f" | {'N/A':>14s}"
    print(line)

    # ── Table 2: EM & MMVS ──
    print("\n" + "=" * 90)
    print("Table 2: Evaluation Multiplier & MMVS")
    print("=" * 90)
    print(f"{'Condition':<28s} | {'Mean EM':>10s} | {'Mean MMVS':>10s} | {'p(EM)':>10s} | {'p(MMVS)':>10s}")
    print("-" * 90)

    for _, r in summary_df.sort_values("mean_eval_multiplier", ascending=False).iterrows():
        cond = r["condition"]
        # Find p-values from stats
        em_stat = stats_df[(stats_df["test_name"] == "eval_multiplier") &
                           (stats_df["comparison"].str.contains(cond.replace(" ", " ")))]
        mmvs_stat = stats_df[(stats_df["test_name"] == "mmvs") &
                             (stats_df["comparison"].str.contains(cond.replace(" ", " ")))]
        em_p = em_stat["p_value_corrected"].values[0] if len(em_stat) > 0 else np.nan
        mmvs_p = mmvs_stat["p_value_corrected"].values[0] if len(mmvs_stat) > 0 else np.nan

        # Task-level stats
        grp = detail_df[detail_df["condition"] == cond].drop_duplicates(subset=["task_id"])
        em_std = grp["eval_multiplier"].std()
        mmvs_std = grp["mmvs"].std()

        em_str = f"{r['mean_eval_multiplier']:.2f}({em_std:.1f})"
        mmvs_str = f"{r['mean_mmvs']:.2f}({mmvs_std:.1f})"
        em_p_str = f"{em_p:.2e}" if not np.isnan(em_p) else "—"
        mmvs_p_str = f"{mmvs_p:.2e}" if not np.isnan(mmvs_p) else "—"

        print(f"{cond:<28s} | {em_str:>10s} | {mmvs_str:>10s} | {em_p_str:>10s} | {mmvs_p_str:>10s}")

    # ── Table 3: Extended Case Distribution ──
    print("\n" + "=" * 85)
    print("Table 3: Extended Case Distribution")
    print("=" * 85)
    print(f"{'Condition':<28s} | {'A+(Deep)':>8s} | {'A(Surf)':>8s} | {'B(Tool)':>8s} | {'C(Sci)':>7s} | {'D(Ser)':>7s}")
    print("-" * 85)

    for _, r in summary_df.sort_values("frac_case_A_plus", ascending=False).iterrows():
        print(f"{r['condition']:<28s} | "
              f"{r['frac_case_A_plus']*100:>7.1f}% | "
              f"{r['frac_case_A']*100:>7.1f}% | "
              f"{r['frac_case_B']*100:>7.1f}% | "
              f"{r['frac_case_C']*100:>6.1f}% | "
              f"{r['frac_case_D']*100:>6.1f}%")

    # ── Table 4: Depth vs Score Correlation ──
    print("\n" + "=" * 75)
    print("Table 4: Depth vs Score Correlation (Spearman)")
    print("=" * 75)
    corr_rows = stats_df[stats_df["test_name"].str.startswith("spearman")]
    print(f"{'Metric':<35s} | {'rho':>7s} | {'p-value':>12s} | {'Sig':>4s}")
    print("-" * 75)
    for _, r in corr_rows.iterrows():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{r['comparison']:<35s} | {r['statistic']:>7.3f} | {r['p_value']:>12.2e} | {sig:>4s}")

    # ── Guided Mode Effect ──
    if len(guided_df) > 0:
        print("\n" + "=" * 80)
        print("Guided Mode Effect on Depth")
        print("=" * 80)
        print(f"{'Model':<20s} | {'delta_depth':>11s} | {'delta_EM':>9s} | {'delta_MMVS':>10s}")
        print("-" * 60)
        for _, r in guided_df.iterrows():
            print(f"{r['model']:<20s} | {r['delta_depth_total']:>+11.3f} | "
                  f"{r['delta_EM']:>+9.3f} | {r['delta_MMVS']:>+10.3f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/6] Loading results...")
    df = load_all()
    print(f"  {len(df)} rows ({df['task_id'].nunique()} tasks x {df['condition'].nunique()} conditions)")

    print("[2/6] Computing per-task x stage depth metrics...")
    detail = compute_detail(df)
    print(f"  {len(detail)} stage-level rows")

    print("[3/6] Filling extended case classifications...")
    detail = fill_extended_cases(detail)

    print("[4/6] Computing summaries...")
    summary = compute_summary(detail)
    guided = compute_guided_effect(detail)

    print("[5/6] Running statistical tests...")
    stats = compute_stats(detail)
    print(f"  {len(stats)} tests computed")

    # Save CSVs
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail.to_csv(out_dir / "plan_exec_depth_v3.csv", index=False)
    summary.to_csv(out_dir / "plan_exec_depth_v3_summary.csv", index=False)
    stats.to_csv(out_dir / "plan_exec_depth_v3_stats.csv", index=False)
    guided.to_csv(out_dir / "plan_exec_depth_v3_guided_effect.csv", index=False)

    print(f"\n[6/6] Saved CSVs to {out_dir}/")
    print(f"  plan_exec_depth_v3.csv              ({len(detail)} rows)")
    print(f"  plan_exec_depth_v3_summary.csv      ({len(summary)} rows)")
    print(f"  plan_exec_depth_v3_stats.csv        ({len(stats)} rows)")
    print(f"  plan_exec_depth_v3_guided_effect.csv ({len(guided)} rows)")

    print_tables(detail, summary, stats, guided)


if __name__ == "__main__":
    main()
