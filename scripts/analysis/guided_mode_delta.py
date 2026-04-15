#!/usr/bin/env python3
"""Guided Mode Delta: Exact component-level deltas with statistical tests.

Computes user-mode minus benchmark-mode delta per component per LLM,
with Wilcoxon signed-rank tests for statistical significance.

Outputs:
  results/analysis/guided_mode_delta_by_component.csv  - per (model, component)
  results/analysis/guided_mode_delta_summary.csv       - per model, wide format
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.analysis.load_results import load_all

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]
ALL_SCORES = COMPONENTS + ["total"]

LLM_PAIRS = [
    ("DeepSeek V3", "DeepSeek V3 user", "DeepSeek V3 benchmark"),
    ("GPT-5", "GPT-5 user", "GPT-5 benchmark"),
    ("Sonnet 4.5", "Sonnet 4.5 user", "Sonnet 4.5 benchmark"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark"),
]

# Bonferroni correction: 4 models × 7 score columns = 28 tests
N_TESTS = len(LLM_PAIRS) * len(ALL_SCORES)


def main():
    print("[1/3] Loading data...")
    df = load_all()

    # Build paired data: merge user and benchmark by (task_id, llm)
    bm = df[df["mode"] == "benchmark"][["task_id", "llm"] + ALL_SCORES].copy()
    us = df[df["mode"] == "user"][["task_id", "llm"] + ALL_SCORES].copy()

    bm = bm.rename(columns={c: f"bm_{c}" for c in ALL_SCORES})
    us = us.rename(columns={c: f"us_{c}" for c in ALL_SCORES})

    paired = pd.merge(bm, us, on=["task_id", "llm"], how="inner")
    print(f"  {len(paired)} paired (task, llm) observations")

    print("[2/3] Computing deltas and Wilcoxon tests...")
    detail_rows = []

    for model, guided_cond, unguided_cond in LLM_PAIRS:
        llm_data = paired[paired["llm"] == model]
        n = len(llm_data)

        for comp in ALL_SCORES:
            guided_vals = llm_data[f"us_{comp}"].values
            unguided_vals = llm_data[f"bm_{comp}"].values
            diffs = guided_vals - unguided_vals

            guided_mean = float(np.mean(guided_vals))
            unguided_mean = float(np.mean(unguided_vals))
            delta = float(np.mean(diffs))
            delta_median = float(np.median(diffs))

            # Wilcoxon signed-rank test (paired, non-parametric)
            # Need non-zero diffs for Wilcoxon
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 5:
                stat, p_val = wilcoxon(nonzero)
            else:
                stat, p_val = np.nan, 1.0

            # Bonferroni correction
            p_bonf = min(p_val * N_TESTS, 1.0) if not np.isnan(p_val) else 1.0

            detail_rows.append({
                "model": model,
                "component": comp,
                "n_tasks": n,
                "guided_mean": round(guided_mean, 2),
                "unguided_mean": round(unguided_mean, 2),
                "delta": round(delta, 2),
                "delta_median": round(delta_median, 2),
                "wilcoxon_stat": round(stat, 1) if not np.isnan(stat) else np.nan,
                "wilcoxon_p": round(p_val, 6) if not np.isnan(p_val) else np.nan,
                "bonferroni_p": round(p_bonf, 6),
                "sig_005": p_val < 0.05 if not np.isnan(p_val) else False,
                "sig_bonf": p_bonf < 0.05,
            })

    detail_df = pd.DataFrame(detail_rows)

    # Wide summary: one row per model, columns = component deltas + p-values
    summary_rows = []
    for model, _, _ in LLM_PAIRS:
        mdf = detail_df[detail_df["model"] == model]
        row = {"model": model, "n_tasks": int(mdf.iloc[0]["n_tasks"])}
        for _, r in mdf.iterrows():
            comp = r["component"]
            row[f"{comp}_delta"] = r["delta"]
            row[f"{comp}_p"] = r["wilcoxon_p"]
            row[f"{comp}_bonf_p"] = r["bonferroni_p"]
            row[f"{comp}_sig"] = r["sig_bonf"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Save
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_df.to_csv(out_dir / "guided_mode_delta_by_component.csv", index=False)
    summary_df.to_csv(out_dir / "guided_mode_delta_summary.csv", index=False)

    print(f"\n[3/3] Saved to {out_dir}/")
    print(f"  guided_mode_delta_by_component.csv ({len(detail_df)} rows)")
    print(f"  guided_mode_delta_summary.csv      ({len(summary_df)} rows)")

    # --- Console output ---
    print("\n" + "=" * 110)
    print("Guided Mode Delta: Component-Level (User - Benchmark)")
    print("=" * 110)

    # Table 1: Deltas
    print(f"\n{'Model':<16s} | {'App':>6s} | {'Orch':>6s} | {'Qual':>6s} | {'Feas':>6s} | "
          f"{'Nov':>6s} | {'Div':>6s} | {'TOTAL':>7s}")
    print("-" * 110)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<16s} | "
              f"{row['approach_delta']:>+6.2f} | "
              f"{row['orchestration_delta']:>+6.2f} | "
              f"{row['quality_delta']:>+6.2f} | "
              f"{row['feasibility_delta']:>+6.2f} | "
              f"{row['novelty_delta']:>+6.2f} | "
              f"{row['diversity_delta']:>+6.2f} | "
              f"{row['total_delta']:>+7.2f}")

    # Table 2: p-values (Bonferroni-corrected)
    print(f"\nBonferroni-corrected p-values (N={N_TESTS} tests):")
    print(f"{'Model':<16s} | {'App':>8s} | {'Orch':>8s} | {'Qual':>8s} | {'Feas':>8s} | "
          f"{'Nov':>8s} | {'Div':>8s} | {'TOTAL':>8s}")
    print("-" * 110)
    for _, row in summary_df.iterrows():
        parts = [f"{row['model']:<16s}"]
        for comp in ALL_SCORES:
            p = row[f"{comp}_bonf_p"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            parts.append(f"{p:>7.4f}{sig}")
        print(" | ".join(parts))

    # Key finding for paper
    print("\n── Key Finding: Quality Δ ──────────────────────────────────────")
    for _, row in detail_df[detail_df["component"] == "quality"].iterrows():
        sig = "SIG" if row["sig_bonf"] else "n.s."
        print(f"  {row['model']:<16s}  Δ = {row['delta']:>+5.2f}  "
              f"p = {row['wilcoxon_p']:.4f}  bonf = {row['bonferroni_p']:.4f}  [{sig}]")

    print()


if __name__ == "__main__":
    main()
