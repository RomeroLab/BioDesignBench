#!/usr/bin/env python3
"""Unified data loader for BioDesignBench 9-condition analysis.

Loads 76 tasks × 9 conditions into a pandas DataFrame with:
- 6 component scores (approach, orchestration, quality, feasibility, novelty, diversity)
- Total score (partial_score)
- Task metadata (task_type, difficulty, design_approach, molecular_subject)
- Agent/condition metadata (agent_id, condition, mode, llm)
- Tool call logs
- AF2 metrics (pLDDT, pTM, ipTM, i_pAE)
- Contamination scores

Usage:
    from scripts.analysis.load_results import load_all, CONDITION_MAP
    df = load_all()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from biodesignbench.taxonomy import get_category, OLD_TO_NEW_MAPPING

# ── 11 conditions: (display_name, agent_dir_path, mode, llm) ─────────────
# Updated 2026-03-25: points to latest re-runs (March 20-21)
_RESULTS = PROJECT_ROOT / "results"

CONDITION_MAP: dict[str, dict] = {
    "Oracle": {
        "path": _RESULTS / "oracle" / "runs" / "run_20260308_oracle" / "agents" / "oracle",
        "mode": "oracle", "llm": "Oracle",
    },
    "Human Expert": {
        "path": _RESULTS / "human-expert-agent" / "runs" / "run_best_consolidated" / "agents" / "human-expert-agent",
        "mode": "expert", "llm": "Human Expert",
    },
    "DeepSeek V3 benchmark": {
        "path": _RESULTS / "deepseek-benchmark" / "runs" / "run_20260321_183634_188072" / "agents" / "deepseek-v3-tools-benchmark",
        "mode": "benchmark", "llm": "DeepSeek V3",
    },
    "DeepSeek V3 user": {
        "path": _RESULTS / "deepseek-user" / "runs" / "run_20260321_100937_943883" / "agents" / "deepseek-v3-tools-user",
        "mode": "user", "llm": "DeepSeek V3",
    },
    "GPT-5 benchmark": {
        "path": _RESULTS / "gpt5-benchmark" / "runs" / "run_20260320_180139_458921" / "agents" / "gpt5-tools-benchmark",
        "mode": "benchmark", "llm": "GPT-5",
    },
    "GPT-5 user": {
        "path": _RESULTS / "gpt5-user" / "runs" / "run_20260320_180139_376318" / "agents" / "gpt5-tools-user",
        "mode": "user", "llm": "GPT-5",
    },
    "Sonnet 4.5 benchmark": {
        "path": _RESULTS / "claude-benchmark" / "runs" / "run_20260321_035815_967069" / "agents" / "claude-code-benchmark",
        "mode": "benchmark", "llm": "Sonnet 4.5",
    },
    "Sonnet 4.5 user": {
        "path": _RESULTS / "claude-user" / "runs" / "run_20260321_085713_261891" / "agents" / "claude-code-user",
        "mode": "user", "llm": "Sonnet 4.5",
    },
    "Gemini 2.5 Pro benchmark": {
        "path": _RESULTS / "gemini-benchmark" / "runs" / "run_20260321_032650_681520" / "agents" / "gemini-tools-benchmark",
        "mode": "benchmark", "llm": "Gemini 2.5 Pro",
    },
    "Gemini 2.5 Pro user": {
        "path": _RESULTS / "gemini-user" / "runs" / "run_20260320_213636_157618" / "agents" / "gemini-tools-user",
        "mode": "user", "llm": "Gemini 2.5 Pro",
    },
    "Hardcoded Pipeline": {
        "path": _RESULTS / "hardcoded-pipeline" / "runs" / "run_best_consolidated" / "agents" / "hardcoded-pipeline",
        "mode": "hardcoded", "llm": "Hardcoded",
    },
}

# 76 common tasks (exclude 4 tasks not in hardcoded pipeline)
EXCLUDED_TASKS = {"binder_001", "binder_002", "dnb_ab_004", "peptide_003"}

# Legacy category mapping for 7 original categories
# Uses task_type field from JSON; new template tasks use prefix-based fallback.
LEGACY_CATEGORY_MAP = {
    "binder_design": "Binder",
    "sequence_optimization": "SeqOpt",
    "complex_prediction": "CpxPrd",
    "conformational_diversity": "ConfDv",
    "de_novo_backbone": "Backbone",
    "antibody_design": "AbBody",
    "antibody_optimization": "AbBody",
    "ppi_design": "PPI",
    "stability_optimization": "SeqOpt",
    "enzyme_design": "SeqOpt",
    "peptide_design": "Binder",
    "fluorescence_optimization": "SeqOpt",
    "fluorescence_design": "SeqOpt",
    "scaffold_design": "Backbone",
}


def _get_legacy_category(task_id: str, task_type: str) -> str:
    """Map task to one of 7 original categories using task_id prefix + task_type."""
    # First try task_type mapping
    if task_type in LEGACY_CATEGORY_MAP:
        return LEGACY_CATEGORY_MAP[task_type]
    # Fallback: use task_id prefix for new template tasks
    if task_id.startswith(("antibody_", "dnb_ab_")):
        return "AbBody"
    if task_id.startswith(("binder_", "peptide_", "dnb_enz_", "dnb_sig_")):
        return "Binder"
    if task_id.startswith("dnb_"):
        return "Binder"
    if task_id.startswith("ppi_"):
        return "PPI"
    if task_id.startswith(("scaffold_", "dnk_")):
        return "Backbone"
    if task_id.startswith("cpx_"):
        return "CpxPrd"
    if task_id.startswith("cfd_"):
        return "ConfDv"
    if task_id.startswith("sqo_"):
        return "SeqOpt"
    if task_id.startswith(("stability_", "enzyme_", "fluorescence_")):
        return "SeqOpt"
    return task_type  # fallback to raw task_type


def _load_task_metadata() -> dict[str, dict]:
    """Load task JSONs for metadata (difficulty, task_type, etc.)."""
    tasks_dir = PROJECT_ROOT / "tasks" / "tier2"
    meta = {}
    for f in tasks_dir.glob("*.json"):
        with open(f) as fh:
            t = json.load(fh)
        tid = t["task_id"]
        difficulty = (
            t.get("difficulty")
            or t.get("metadata", {}).get("difficulty", "unknown")
        )
        task_type = t.get("task_type", "unknown")
        legacy_cat = _get_legacy_category(tid, task_type)

        # Taxonomy
        cat = get_category(tid)
        approach = cat.approach.value if cat else "unknown"
        subject = cat.subject.value if cat else "unknown"

        meta[tid] = {
            "difficulty": difficulty,
            "task_type": task_type,
            "legacy_category": legacy_cat,
            "design_approach": approach,
            "molecular_subject": subject,
        }
    return meta


def _extract_row(result: dict, condition: str, info: dict, task_meta: dict) -> dict:
    """Extract a flat dict from a single result.json."""
    tid = result["task_id"]
    tm = task_meta.get(tid, {})

    # Component scores
    approach = result.get("approach_metrics", {}).get("score", 0)
    orchestration = result.get("orchestration_metrics", {}).get("score", 0)
    quality = result.get("quality_metrics", {}).get("score", 0)
    feasibility = result.get("feasibility_metrics", {}).get("score", 0)
    novelty = result.get("novelty_metrics", {}).get("score", 0)
    diversity = result.get("diversity_metrics", {}).get("score", 0)
    total = result.get("partial_score", 0)

    # Quality tier breakdown
    qm = result.get("quality_metrics", {})
    tier_a = qm.get("tier_a", 0.0)
    tier_b = qm.get("tier_b", 0.0)
    tier_c = qm.get("tier_c", 0.0)

    # AF2 metrics from quality breakdown
    breakdown = qm.get("breakdown", {})
    structure = breakdown.get("structure", {})
    interface = breakdown.get("interface", {})

    # Tool calls — use tools_used and orchestration actual_tool_order
    tools_used = result.get("tools_used", [])
    actual_tool_order = result.get("orchestration_metrics", {}).get("actual_tool_order", [])
    # Also check raw_output.tool_call_log for detailed calls
    raw_tool_log = result.get("raw_output", {}).get("tool_call_log", [])
    tool_names = actual_tool_order if actual_tool_order else tools_used
    num_tools = len(tools_used)
    unique_tools = len(set(tools_used))
    failed_tools = sum(1 for tc in raw_tool_log if not tc.get("success", True)) if raw_tool_log else 0

    # Contamination
    contamination = result.get("contamination_score", 0.0)

    # Independently verified
    verified = result.get("independently_verified", False)

    return {
        "task_id": tid,
        "condition": condition,
        "mode": info["mode"],
        "llm": info["llm"],
        "agent_id": result.get("agent_id", ""),
        # Scores
        "approach": approach,
        "orchestration": orchestration,
        "quality": quality,
        "feasibility": feasibility,
        "novelty": novelty,
        "diversity": diversity,
        "total": total if total else (approach + orchestration + quality + feasibility + novelty + diversity),
        # Quality tiers
        "tier_a": tier_a,
        "tier_b": tier_b,
        "tier_c": tier_c,
        # AF2 metrics (from scoring breakdown)
        "pLDDT_frac": structure.get("pLDDT", None),
        "pTM_frac": structure.get("pTM", None),
        "ipTM_frac": interface.get("ipTM", None),
        "i_pAE_frac": interface.get("i_pAE", None),
        # Tool usage
        "num_tool_calls": num_tools,
        "unique_tools": unique_tools,
        "failed_tools": failed_tools,
        "tool_sequence": tool_names,
        # Contamination
        "contamination_score": contamination,
        # Verification
        "independently_verified": verified,
        # Task metadata
        "difficulty": tm.get("difficulty", "unknown"),
        "task_type": tm.get("task_type", "unknown"),
        "legacy_category": tm.get("legacy_category", "unknown"),
        "design_approach": tm.get("design_approach", "unknown"),
        "molecular_subject": tm.get("molecular_subject", "unknown"),
    }


def load_all(include_excluded: bool = False) -> pd.DataFrame:
    """Load all 9 conditions × 76 tasks into a DataFrame.

    Args:
        include_excluded: If True, include the 4 excluded tasks too.

    Returns:
        DataFrame with one row per (task, condition).
    """
    task_meta = _load_task_metadata()
    rows = []

    for condition, info in CONDITION_MAP.items():
        agent_dir = info["path"]
        if not agent_dir.exists():
            print(f"WARNING: {condition} dir not found: {agent_dir}")
            continue

        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            result_file = task_dir / "result.json"
            if not result_file.exists():
                continue

            with open(result_file) as f:
                result = json.load(f)

            tid = result["task_id"]
            if not include_excluded and tid in EXCLUDED_TASKS:
                continue

            rows.append(_extract_row(result, condition, info, task_meta))

    df = pd.DataFrame(rows)

    # Sort by condition order and task_id
    condition_order = list(CONDITION_MAP.keys())
    df["condition"] = pd.Categorical(df["condition"], categories=condition_order, ordered=True)
    df = df.sort_values(["condition", "task_id"]).reset_index(drop=True)

    return df


def load_score_matrix() -> pd.DataFrame:
    """Load 76×9 score matrix (tasks as rows, conditions as columns)."""
    df = load_all()
    return df.pivot_table(
        index="task_id", columns="condition", values="total", aggfunc="first"
    )


def load_component_matrix() -> dict[str, pd.DataFrame]:
    """Load per-component score matrices (76×9 each)."""
    df = load_all()
    components = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]
    return {
        comp: df.pivot_table(
            index="task_id", columns="condition", values=comp, aggfunc="first"
        )
        for comp in components
    }


if __name__ == "__main__":
    df = load_all()
    print(f"Loaded {len(df)} rows ({df['task_id'].nunique()} tasks × {df['condition'].nunique()} conditions)")
    print(f"\nCondition means:")
    means = df.groupby("condition")["total"].mean().sort_values(ascending=False)
    for cond, mean in means.items():
        print(f"  {cond:30s} {mean:.1f}")
    print(f"\nComponent means:")
    for comp in ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]:
        print(f"  {comp:15s} {df[comp].mean():.1f}")
