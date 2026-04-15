#!/usr/bin/env python3
"""Plan-Execution Gap v2: Reference-pipeline-aligned scoring.

Aligns both plan and execution scores to the same per-task reference pipeline
(derived from DesignApproach in taxonomy).

Reference pipeline steps:
  de_novo:  [backbone_generation, sequence_design, structure_prediction, scoring_validation]
  redesign: [sequence_design, structure_prediction, scoring_validation]

Per-step classification:
  Case A (Full Knowledge):  plan=1 AND exec=1
  Case B (Tool Gap):        plan=1 AND exec=0
  Case C (Science Gap):     plan=0 AND exec=0
  Case D (Serendipity):     plan=0 AND exec=1

Outputs:
  results/analysis/plan_exec_gap_v2.csv            - per task×condition detail
  results/analysis/plan_exec_gap_v2_summary.csv     - per condition aggregated
  results/analysis/plan_exec_gap_v2_guided_effect.csv - per model guided delta
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Reference pipeline definitions
# ---------------------------------------------------------------------------

ALL_STEPS = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]

REFERENCE_PIPELINE = {
    "de_novo": ALL_STEPS,
    "redesign": ["sequence_design", "structure_prediction", "scoring_validation"],
}

# Map CSV column names to canonical step names
EXEC_COL = {
    "backbone_generation":  "backbone_gen",
    "sequence_design":      "seq_design",
    "structure_prediction": "struct_pred",
    "scoring_validation":   "scoring",
}

PLAN_COL = {
    "backbone_generation":  "plan_backbone",
    "sequence_design":      "plan_sequence",
    "structure_prediction": "plan_structure",
    "scoring_validation":   "plan_scoring",
}

# (model, mode) → condition
MODEL_MODE_TO_CONDITION = {
    ("GPT-5", "user"):            "GPT-5 user",
    ("GPT-5", "benchmark"):       "GPT-5 benchmark",
    ("Sonnet 4.5", "user"):       "Sonnet 4.5 user",
    ("Sonnet 4.5", "benchmark"):  "Sonnet 4.5 benchmark",
    ("DeepSeek V3", "user"):      "DeepSeek V3 user",
    ("DeepSeek V3", "benchmark"): "DeepSeek V3 benchmark",
    ("Gemini 2.5 Pro", "user"):       "Gemini 2.5 Pro user",
    ("Gemini 2.5 Pro", "benchmark"):  "Gemini 2.5 Pro benchmark",
    ("Hardcoded", "baseline"):    "Hardcoded Pipeline",
    ("Human Expert", "baseline"): "Human Expert",
    ("Oracle", "baseline"):       "Oracle",
}

# Scripted baselines: assume plan=1 for all reference steps
SCRIPTED_BASELINES = {"Oracle", "Human Expert", "Hardcoded Pipeline"}

# LLM pairs for guided effect
LLM_PAIRS = [
    ("DeepSeek V3", "DeepSeek V3 user", "DeepSeek V3 benchmark"),
    ("GPT-5", "GPT-5 user", "GPT-5 benchmark"),
    ("Sonnet 4.5", "Sonnet 4.5 user", "Sonnet 4.5 benchmark"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark"),
]


# ---------------------------------------------------------------------------
# Taxonomy lookup
# ---------------------------------------------------------------------------

def get_approach(task_id: str) -> str | None:
    """Return 'de_novo' or 'redesign' for a task, or None if unknown."""
    try:
        from biodesignbench.taxonomy import get_category
        cat = get_category(task_id)
        if cat is not None:
            return cat.approach.value
    except ImportError:
        pass

    # Fallback: prefix-based heuristic
    dn_prefixes = ("binder_", "scaffold_", "ppi_", "peptide_",
                   "dnb_", "dnk_", "cfd_", "cpx_")
    rd_prefixes = ("enzyme_", "stability_", "fluorescence_",
                   "antibody_", "sqo_")
    if task_id.startswith(dn_prefixes):
        return "de_novo"
    elif task_id.startswith(rd_prefixes):
        return "redesign"
    return None


def get_subject(task_id: str) -> str | None:
    """Return molecular subject for a task, or None if unknown."""
    try:
        from biodesignbench.taxonomy import get_category
        cat = get_category(task_id)
        if cat is not None:
            return cat.subject.value
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_exec_data() -> pd.DataFrame:
    """Load pipeline stage completion data (execution scores per step)."""
    path = ROOT / "results" / "rescoring" / "pipeline_stage_completion.csv"
    df = pd.read_csv(path)
    # Deduplicate: keep first occurrence per (task_id, condition)
    df = df.drop_duplicates(subset=["task_id", "condition"], keep="first")
    return df


def load_plan_data() -> pd.DataFrame:
    """Load reasoning trace analysis data (plan scores per step)."""
    path = ROOT / "figures" / "reasoning_trace_summary.csv"
    df = pd.read_csv(path)

    # Map (model, mode) → condition
    df["condition"] = df.apply(
        lambda r: MODEL_MODE_TO_CONDITION.get((r["model"], r["mode"])), axis=1
    )
    df = df.dropna(subset=["condition"])

    # Deduplicate
    df = df.drop_duplicates(subset=["task_id", "condition"], keep="first")
    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_v2(exec_df: pd.DataFrame, plan_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-task×condition v2 plan-exec gap metrics."""
    # Merge on (task_id, condition)
    merged = exec_df.merge(plan_df, on=["task_id", "condition"], how="inner",
                           suffixes=("_exec", "_plan"))

    rows = []
    for _, r in merged.iterrows():
        task_id = r["task_id"]
        condition = r["condition"]
        approach = get_approach(task_id)
        if approach is None:
            continue

        ref_steps = REFERENCE_PIPELINE[approach]
        n_ref = len(ref_steps)
        is_baseline = condition in SCRIPTED_BASELINES

        n_a = n_b = n_c = n_d = 0
        plan_sum = exec_sum = 0

        for step in ref_steps:
            exec_val = int(r.get(EXEC_COL[step], 0))

            if is_baseline:
                plan_val = 1  # Scripted baselines know the pipeline
            else:
                plan_val = int(r.get(PLAN_COL[step], 0))

            plan_sum += plan_val
            exec_sum += exec_val

            if plan_val == 1 and exec_val == 1:
                n_a += 1
            elif plan_val == 1 and exec_val == 0:
                n_b += 1
            elif plan_val == 0 and exec_val == 0:
                n_c += 1
            else:  # plan=0, exec=1
                n_d += 1

        plan_score = plan_sum / n_ref
        exec_score = exec_sum / n_ref
        gap = plan_score - exec_score

        subject = get_subject(task_id)

        rows.append({
            "task_id": task_id,
            "condition": condition,
            "approach": approach,
            "subject": subject or "unknown",
            "n_ref_steps": n_ref,
            "plan_score": round(plan_score, 4),
            "exec_score": round(exec_score, 4),
            "gap": round(gap, 4),
            "n_case_a": n_a,
            "n_case_b": n_b,
            "n_case_c": n_c,
            "n_case_d": n_d,
        })

    return pd.DataFrame(rows)


def compute_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per condition."""
    rows = []
    for condition, grp in detail_df.groupby("condition"):
        n = len(grp)
        total_steps = grp["n_ref_steps"].sum()

        # Case fractions: sum over all (task × step) pairs
        total_a = grp["n_case_a"].sum()
        total_b = grp["n_case_b"].sum()
        total_c = grp["n_case_c"].sum()
        total_d = grp["n_case_d"].sum()

        rows.append({
            "condition": condition,
            "n_tasks": n,
            "mean_plan_0to4": round(grp["plan_score"].mean() * 4, 3),
            "mean_exec_0to4": round(grp["exec_score"].mean() * 4, 3),
            "mean_gap": round(grp["gap"].mean(), 4),
            "frac_case_a": round(total_a / total_steps, 4) if total_steps > 0 else 0,
            "frac_case_b": round(total_b / total_steps, 4) if total_steps > 0 else 0,
            "frac_case_c": round(total_c / total_steps, 4) if total_steps > 0 else 0,
            "frac_case_d": round(total_d / total_steps, 4) if total_steps > 0 else 0,
        })

    return pd.DataFrame(rows)


def compute_guided_effect(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute guided (user) - unguided (benchmark) delta per model."""
    summary_idx = summary_df.set_index("condition")
    rows = []
    for model, guided, unguided in LLM_PAIRS:
        if guided not in summary_idx.index or unguided not in summary_idx.index:
            continue
        g = summary_idx.loc[guided]
        u = summary_idx.loc[unguided]
        delta_plan = round(g["mean_plan_0to4"] - u["mean_plan_0to4"], 3)
        delta_exec = round(g["mean_exec_0to4"] - u["mean_exec_0to4"], 3)
        rows.append({
            "model": model,
            "delta_plan": delta_plan,
            "delta_exec": delta_exec,
            "delta_plan_gt_delta_exec": delta_plan > delta_exec,
        })
    return pd.DataFrame(rows)


def compute_by_domain(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Case A/B/C/D by (condition, subject)."""
    rows = []
    for (condition, subject), grp in detail_df.groupby(["condition", "subject"]):
        total_steps = grp["n_ref_steps"].sum()
        if total_steps == 0:
            continue
        total_a = grp["n_case_a"].sum()
        total_b = grp["n_case_b"].sum()
        total_c = grp["n_case_c"].sum()
        total_d = grp["n_case_d"].sum()
        rows.append({
            "condition": condition,
            "subject": subject,
            "n_tasks": len(grp),
            "n_ref_steps_total": int(total_steps),
            "frac_case_a": round(total_a / total_steps, 4),
            "frac_case_b": round(total_b / total_steps, 4),
            "frac_case_c": round(total_c / total_steps, 4),
            "frac_case_d": round(total_d / total_steps, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(summary_df: pd.DataFrame, guided_df: pd.DataFrame,
                  domain_df: pd.DataFrame | None = None):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("Plan-Execution Gap v2: Reference Pipeline Aligned")
    print("=" * 80)

    print("\n── Condition Summary ────────────────────────────────────────")
    print(f"{'Condition':<28s} {'N':>3s} {'Plan':>5s} {'Exec':>5s} "
          f"{'Gap':>6s} {'A%':>5s} {'B%':>5s} {'C%':>5s} {'D%':>5s}")
    print("-" * 80)
    for _, r in summary_df.sort_values("mean_exec_0to4", ascending=False).iterrows():
        print(f"{r['condition']:<28s} {r['n_tasks']:>3.0f} "
              f"{r['mean_plan_0to4']:>5.2f} {r['mean_exec_0to4']:>5.2f} "
              f"{r['mean_gap']:>+6.3f} "
              f"{r['frac_case_a']*100:>5.1f} {r['frac_case_b']*100:>5.1f} "
              f"{r['frac_case_c']*100:>5.1f} {r['frac_case_d']*100:>5.1f}")

    print("\n── Guided Mode Effect ──────────────────────────────────────")
    print(f"{'Model':<20s} {'ΔPlan':>7s} {'ΔExec':>7s} {'Plan>Exec':>10s}")
    print("-" * 50)
    for _, r in guided_df.iterrows():
        print(f"{r['model']:<20s} {r['delta_plan']:>+7.3f} {r['delta_exec']:>+7.3f} "
              f"{'Yes' if r['delta_plan_gt_delta_exec'] else 'No':>10s}")

    if domain_df is not None and len(domain_df) > 0:
        print("\n── Domain Breakdown (user-mode LLMs only) ─────────────────")
        llm_user = [c for c in domain_df["condition"].unique()
                    if c.endswith(" user")]
        sub_order = ["antibody", "enzyme", "binder", "scaffold",
                     "fluorescent_protein"]
        print(f"{'Condition':<24s} {'Subject':<14s} {'N':>3s} "
              f"{'A%':>5s} {'B%':>5s} {'C%':>5s} {'D%':>5s}")
        print("-" * 75)
        for cond in sorted(llm_user):
            for subj in sub_order:
                row = domain_df[(domain_df["condition"] == cond) &
                                (domain_df["subject"] == subj)]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                print(f"{cond:<24s} {subj:<14s} {r['n_tasks']:>3.0f} "
                      f"{r['frac_case_a']*100:>5.1f} "
                      f"{r['frac_case_b']*100:>5.1f} "
                      f"{r['frac_case_c']*100:>5.1f} "
                      f"{r['frac_case_d']*100:>5.1f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/4] Loading execution data...")
    exec_df = load_exec_data()
    print(f"  {len(exec_df)} rows from pipeline_stage_completion.csv")

    print("[2/4] Loading plan data...")
    plan_df = load_plan_data()
    print(f"  {len(plan_df)} rows from reasoning_trace_summary.csv")

    print("[3/5] Computing v2 metrics...")
    detail = compute_v2(exec_df, plan_df)
    summary = compute_summary(detail)
    guided = compute_guided_effect(summary)
    by_domain = compute_by_domain(detail)

    print(f"  {len(detail)} task×condition pairs")

    # Save CSVs
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail.to_csv(out_dir / "plan_exec_gap_v2.csv", index=False)
    summary.to_csv(out_dir / "plan_exec_gap_v2_summary.csv", index=False)
    guided.to_csv(out_dir / "plan_exec_gap_v2_guided_effect.csv", index=False)
    by_domain.to_csv(out_dir / "plan_exec_gap_v2_by_domain.csv", index=False)

    print(f"\n[4/5] Saved CSVs to {out_dir}/")
    print(f"  plan_exec_gap_v2.csv              ({len(detail)} rows)")
    print(f"  plan_exec_gap_v2_summary.csv       ({len(summary)} rows)")
    print(f"  plan_exec_gap_v2_guided_effect.csv ({len(guided)} rows)")
    print(f"  plan_exec_gap_v2_by_domain.csv     ({len(by_domain)} rows)")

    print_summary(summary, guided, by_domain)


if __name__ == "__main__":
    main()
