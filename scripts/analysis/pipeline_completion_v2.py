#!/usr/bin/env python3
"""Pipeline Completion v2: Reference-pipeline-aligned.

Instead of measuring against a fixed 4-stage pipeline, each task's completion
is measured against its actual reference pipeline (de_novo=4, redesign=3 steps).

Outputs:
  results/analysis/pipeline_completion_v2.csv         - per task×condition
  results/analysis/pipeline_completion_v2_summary.csv  - per condition
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Reference pipeline
# ---------------------------------------------------------------------------

ALL_STEPS = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]

REFERENCE_PIPELINE = {
    "de_novo": ALL_STEPS,
    "redesign": ["sequence_design", "structure_prediction", "scoring_validation"],
}

EXEC_COL = {
    "backbone_generation":  "backbone_gen",
    "sequence_design":      "seq_design",
    "structure_prediction": "struct_pred",
    "scoring_validation":   "scoring",
}

SCRIPTED_BASELINES = {"Oracle", "Human Expert"}

# Condition mapping
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


def get_approach(task_id: str) -> str | None:
    try:
        from biodesignbench.taxonomy import get_category
        cat = get_category(task_id)
        if cat is not None:
            return cat.approach.value
    except ImportError:
        pass
    dn = ("binder_", "scaffold_", "ppi_", "peptide_", "dnb_", "dnk_", "cfd_", "cpx_")
    rd = ("enzyme_", "stability_", "fluorescence_", "antibody_", "sqo_")
    if task_id.startswith(dn):
        return "de_novo"
    elif task_id.startswith(rd):
        return "redesign"
    return None


def main():
    exec_df = pd.read_csv(ROOT / "results" / "rescoring" / "pipeline_stage_completion.csv")
    exec_df = exec_df.drop_duplicates(subset=["task_id", "condition"], keep="first")

    rows = []
    for _, r in exec_df.iterrows():
        task_id = r["task_id"]
        condition = r["condition"]
        approach = get_approach(task_id)
        if approach is None:
            continue

        ref_steps = REFERENCE_PIPELINE[approach]
        n_required = len(ref_steps)

        step_detail = {}
        n_completed = 0
        for step in ALL_STEPS:
            required = step in ref_steps
            done = int(r.get(EXEC_COL[step], 0))
            step_detail[f"{step}_req"] = int(required)
            step_detail[f"{step}_done"] = done
            if required and done:
                n_completed += 1

        # Oracle override: assume all reference steps complete
        if condition in SCRIPTED_BASELINES:
            n_completed = n_required
            for step in ref_steps:
                step_detail[f"{step}_done"] = 1

        completion_ratio = n_completed / n_required if n_required > 0 else 0

        rows.append({
            "task_id": task_id,
            "condition": condition,
            "approach": approach,
            "n_required": n_required,
            "n_completed": n_completed,
            "completion_ratio": round(completion_ratio, 4),
            **step_detail,
        })

    detail = pd.DataFrame(rows)

    # --- Summary per condition ---
    summary_rows = []
    for condition, grp in detail.groupby("condition"):
        n = len(grp)
        summary_rows.append({
            "condition": condition,
            "n_tasks": n,
            "full_completion_rate": round((grp["completion_ratio"] == 1.0).mean(), 4),
            "mean_completion_ratio": round(grp["completion_ratio"].mean(), 4),
            "frac_0pct": round((grp["completion_ratio"] == 0).mean(), 4),
            "frac_1_49pct": round(((grp["completion_ratio"] > 0) & (grp["completion_ratio"] < 0.5)).mean(), 4),
            "frac_50_99pct": round(((grp["completion_ratio"] >= 0.5) & (grp["completion_ratio"] < 1.0)).mean(), 4),
            "frac_100pct": round((grp["completion_ratio"] == 1.0).mean(), 4),
        })
    summary = pd.DataFrame(summary_rows)

    # --- Save ---
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "pipeline_completion_v2.csv", index=False)
    summary.to_csv(out / "pipeline_completion_v2_summary.csv", index=False)

    # --- Console ---
    print("=" * 70)
    print("Pipeline Completion v2: Reference Pipeline Aligned")
    print("=" * 70)
    print(f"\n{'Condition':<28s} {'N':>3s} {'Full%':>6s} {'Mean%':>6s}  "
          f"{'0%':>5s} {'1-49%':>6s} {'50-99%':>7s} {'100%':>5s}")
    print("-" * 70)
    for _, r in summary.sort_values("mean_completion_ratio", ascending=False).iterrows():
        print(f"{r['condition']:<28s} {r['n_tasks']:>3.0f} "
              f"{r['full_completion_rate']*100:>6.1f} {r['mean_completion_ratio']*100:>6.1f}  "
              f"{r['frac_0pct']*100:>5.1f} {r['frac_1_49pct']*100:>5.1f} "
              f"{r['frac_50_99pct']*100:>7.1f} {r['frac_100pct']*100:>5.1f}")

    # --- Compare old vs new ---
    old_pipe = pd.read_csv(ROOT / "results" / "rescoring" / "pipeline_stage_completion.csv")
    print("\n\nOld (4/4) vs New (ref-aligned) full completion rate:")
    print(f"{'Condition':<28s} {'Old 4/4':>8s} {'New ref':>8s} {'Δ':>6s}")
    print("-" * 55)
    for _, r in summary.sort_values("mean_completion_ratio", ascending=False).iterrows():
        cond = r["condition"]
        old_grp = old_pipe[old_pipe["condition"] == cond]
        old_rate = (old_grp["n_stages"] == 4).mean() if len(old_grp) > 0 else 0
        new_rate = r["full_completion_rate"]
        print(f"{cond:<28s} {old_rate*100:>7.1f}% {new_rate*100:>7.1f}% {(new_rate-old_rate)*100:>+5.1f}%")


if __name__ == "__main__":
    main()
