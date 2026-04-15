#!/usr/bin/env python3
"""Stochastic Tool Understanding Analysis (PROMPT 19).

Investigates whether LLM agents understand the stochastic vs. deterministic
nature of protein design tools.  Expert practitioners repeat stochastic tools
(RFdiffusion, ProteinMPNN) to explore the design space, while treating
deterministic tools (AlphaFold2, Rosetta scoring) as one-shot per input.

Analyses:
  1. Stochastic Repetition Index (SRI) per task/condition
  2. Redundant deterministic call detection
  3. Parameter specification patterns (num_designs, num_sequences, sampling_temp)
  4. Backbone generation depth → Diversity score correlation
  5. Adaptive strategy: de novo vs redesign SRI comparison
  6. Molecular subject–level SRI patterns
  7. Reasoning trace stochasticity awareness keywords

Outputs:
  results/analysis/stochastic_understanding.csv              - per task × condition
  results/analysis/stochastic_understanding_summary.csv      - per condition
  results/analysis/stochastic_understanding_stats.csv        - statistical tests
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from scripts.analysis.load_results import CONDITION_MAP, EXCLUDED_TASKS

# ── Tool classification ──────────────────────────────────────────────────────

STOCHASTIC_TOOLS = frozenset({
    "generate_backbone",    # RFdiffusion — random seed → different backbone
    "design_binder",        # RFdiffusion + ProteinMPNN composite — stochastic
    "design_sequence",      # ProteinMPNN — sampling_temp → stochastic output
    "optimize_sequence",    # ProteinMPNN multi-round — stochastic sampling
    "rosetta_design",       # Rosetta PackRotamers — Monte Carlo sampling
})

DETERMINISTIC_TOOLS = frozenset({
    "predict_structure",         # AlphaFold2 / ESMFold
    "predict_structure_boltz",   # Boltz
    "predict_complex",           # AlphaFold2-Multimer
    "predict_affinity_boltz",    # Boltz affinity
    "score_stability",           # ESM2 perplexity
    "rosetta_score",             # Rosetta energy
    "rosetta_relax",             # FastRelax (quasi-deterministic)
    "rosetta_interface_score",   # Deterministic
    "analyze_interface",         # Geometric analysis
    "energy_minimize",           # Gradient-based
    "validate_design",           # Deterministic check
    "suggest_hotspots",          # Computational alanine scanning
})

UTILITY_TOOLS = frozenset({"get_design_status", "read_file", "execute_python"})

ALL_DESIGN_TOOLS = STOCHASTIC_TOOLS | DETERMINISTIC_TOOLS

# Backbone generators specifically (for Analysis 4)
BACKBONE_GENERATORS = frozenset({"generate_backbone", "design_binder"})

# ── Stochasticity awareness keywords (Analysis 7) ───────────────────────────

AWARENESS_LEVEL_3 = [
    "sample multiple", "generate diverse", "different seeds",
    "stochastic", "random sampling", "explore design space",
    "multiple backbone", "generate several candidates",
    "run multiple times", "each run produces different",
    "design space exploration", "diverse candidates",
    "sample from", "diverse pool", "multiple rounds of generation",
    "random nature", "inherently stochastic",
]

AWARENESS_LEVEL_2 = [
    "num_designs", "multiple designs", "several sequences",
    "generate a few", "try different", "diverse set",
    "many candidates", "multiple candidates", "variety of",
    "different backbone", "sampling temperature",
    "different conformations", "explore",
]

# ── Parameter keys to detect ─────────────────────────────────────────────────

# For generate_backbone / design_binder
BACKBONE_PARAMS = {"num_designs"}
# For design_sequence
SEQDESIGN_PARAMS = {"num_sequences", "sampling_temp"}
# For optimize_sequence
OPTSEQ_PARAMS = {"temperature"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_task_metadata() -> dict[str, dict]:
    """Load task JSONs for approach/subject metadata."""
    from biodesignbench.taxonomy import get_category
    tasks_dir = ROOT / "tasks" / "tier2"
    meta = {}
    for f in tasks_dir.glob("*.json"):
        with open(f) as fh:
            t = json.load(fh)
        tid = t["task_id"]
        cat = get_category(tid)
        meta[tid] = {
            "design_approach": cat.approach.value if cat else "unknown",
            "molecular_subject": cat.subject.value if cat else "unknown",
        }
    return meta


def _load_all_results() -> list[dict]:
    """Load raw result.json for every task × condition."""
    rows = []
    for condition, info in CONDITION_MAP.items():
        agent_dir = info["path"]
        if not agent_dir.exists():
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
            if tid in EXCLUDED_TASKS:
                continue
            rows.append({
                "condition": condition,
                "mode": info["mode"],
                "llm": info["llm"],
                "result": result,
            })
    return rows


# ── Per-task analysis ─────────────────────────────────────────────────────────

def _analyze_tool_calls(tool_log: list[dict]) -> dict:
    """Compute stochastic/deterministic metrics from tool_call_log."""
    # Filter to design tools only
    design_calls = [tc for tc in tool_log if tc["tool"] in ALL_DESIGN_TOOLS]
    stochastic_calls = [tc for tc in design_calls if tc["tool"] in STOCHASTIC_TOOLS]
    deterministic_calls = [tc for tc in design_calls if tc["tool"] in DETERMINISTIC_TOOLS]

    n_stochastic = len(stochastic_calls)
    n_deterministic = len(deterministic_calls)

    stoch_types = set(tc["tool"] for tc in stochastic_calls)
    det_types = set(tc["tool"] for tc in deterministic_calls)
    n_stoch_types = len(stoch_types)
    n_det_types = len(det_types)

    # Repeat ratios
    stoch_repeat = n_stochastic / max(n_stoch_types, 1)
    det_repeat = n_deterministic / max(n_det_types, 1)

    # SRI: Stochastic Repetition Index
    sri = stoch_repeat / max(det_repeat, 0.5)

    # ── Redundant deterministic calls ──
    # Two calls to the same deterministic tool with the same arg key set
    # are "redundant" (we can't check actual values, but same params = likely same input)
    det_call_signatures = Counter()
    n_redundant_det = 0
    for tc in deterministic_calls:
        sig = (tc["tool"], tuple(sorted(tc.get("args_summary", {}).keys())))
        det_call_signatures[sig] += 1

    for sig, count in det_call_signatures.items():
        if count > 1:
            n_redundant_det += count - 1  # first call is not redundant

    # ── Useful stochastic repeats ──
    stoch_call_signatures = Counter()
    for tc in stochastic_calls:
        sig = (tc["tool"], tuple(sorted(tc.get("args_summary", {}).keys())))
        stoch_call_signatures[sig] += 1
    n_useful_stoch_repeats = sum(max(0, c - 1) for c in stoch_call_signatures.values())

    # ── Backbone generation depth ──
    n_backbone_calls = sum(1 for tc in tool_log if tc["tool"] in BACKBONE_GENERATORS)

    # ── Parameter specification ──
    num_designs_specified = False
    num_seq_specified = False
    sampling_temp_specified = False
    temperature_specified = False

    for tc in tool_log:
        args = tc.get("args_summary", {})
        if tc["tool"] in ("generate_backbone", "design_binder"):
            if "num_designs" in args:
                num_designs_specified = True
        if tc["tool"] == "design_sequence":
            if "num_sequences" in args:
                num_seq_specified = True
            if "sampling_temp" in args:
                sampling_temp_specified = True
        if tc["tool"] == "optimize_sequence":
            if "temperature" in args:
                temperature_specified = True

    return {
        "n_stochastic_calls": n_stochastic,
        "n_deterministic_calls": n_deterministic,
        "n_stochastic_types": n_stoch_types,
        "n_deterministic_types": n_det_types,
        "stochastic_repeat_ratio": round(stoch_repeat, 3),
        "deterministic_repeat_ratio": round(det_repeat, 3),
        "SRI": round(sri, 3),
        "n_redundant_deterministic_calls": n_redundant_det,
        "n_useful_stochastic_repeats": n_useful_stoch_repeats,
        "n_backbone_calls": n_backbone_calls,
        "num_designs_specified": num_designs_specified,
        "num_seq_specified": num_seq_specified,
        "sampling_temp_specified": sampling_temp_specified,
        "temperature_specified": temperature_specified,
    }


def _analyze_awareness(reasoning: str) -> int:
    """Score stochasticity awareness from reasoning trace (0-3)."""
    if not reasoning:
        return 0
    text = reasoning.lower()
    # Level 3: deep awareness
    if any(kw in text for kw in AWARENESS_LEVEL_3):
        return 3
    # Level 2: partial awareness
    if any(kw in text for kw in AWARENESS_LEVEL_2):
        return 2
    # Level 1: generic diversity mention
    if "diversity" in text or "diverse" in text:
        return 1
    return 0


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis():
    print("Loading data...")
    task_meta = _load_task_metadata()
    all_results = _load_all_results()
    print(f"  {len(all_results)} result entries loaded")

    # ── Build per-task rows ──
    rows = []
    for entry in all_results:
        result = entry["result"]
        tid = result["task_id"]
        condition = entry["condition"]
        llm = entry["llm"]
        mode = entry["mode"]

        meta = task_meta.get(tid, {})
        approach = meta.get("design_approach", "unknown")
        subject = meta.get("molecular_subject", "unknown")

        raw = result.get("raw_output", {})
        tool_log = raw.get("tool_call_log", [])
        reasoning = raw.get("reasoning_trace", "")

        # Scores
        diversity_score = result.get("diversity_metrics", {}).get("score", 0)
        total_score = result.get("partial_score", 0)
        n_designs = result.get("diversity_metrics", {}).get("num_designs", 0)
        if n_designs is None:
            n_designs = 0

        # Tool analysis
        tc_metrics = _analyze_tool_calls(tool_log)

        # Awareness
        awareness = _analyze_awareness(reasoning)

        row = {
            "task_id": tid,
            "condition": condition,
            "llm": llm,
            "mode": mode,
            "design_approach": approach,
            "molecular_subject": subject,
            # Tool counts
            **tc_metrics,
            # Scores
            "diversity_score": diversity_score,
            "total_score": total_score,
            "n_designs": n_designs,
            # Awareness
            "stochasticity_awareness_score": awareness,
            "reasoning_length": len(reasoning),
            # Has any tool calls?
            "has_tool_calls": len(tool_log) > 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Condition ordering ──
    cond_order = list(CONDITION_MAP.keys())
    df["condition"] = pd.Categorical(df["condition"], categories=cond_order, ordered=True)

    # ── Save per-task CSV ──
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "stochastic_understanding.csv", index=False)
    print(f"\nSaved: stochastic_understanding.csv ({len(df)} rows)")

    # ════════════════════════════════════════════════════════════════════════
    # Summary tables
    # ════════════════════════════════════════════════════════════════════════

    # Filter to conditions with tool calls for SRI analysis
    df_tools = df[df["has_tool_calls"]].copy()

    # ── Table 1: Stochastic Repetition Index ──
    print("\n" + "=" * 100)
    print("Table 1: Stochastic Repetition Index (SRI)")
    print("  SRI > 1 → stochastic tools repeated more than deterministic (understands stochasticity)")
    print("  SRI ≈ 1 → no differentiation between tool types")
    print("=" * 100)

    sri_summary = []
    expert_sri = df_tools[df_tools["condition"] == "Human Expert"]["SRI"]

    for cond in cond_order:
        sub = df_tools[df_tools["condition"] == cond]
        if len(sub) == 0:
            continue
        sri_vals = sub["SRI"]
        m, med, s = sri_vals.mean(), sri_vals.median(), sri_vals.std()

        # Mann-Whitney vs Expert
        p_val = np.nan
        cohens_d = np.nan
        if cond != "Human Expert" and len(expert_sri) > 0 and len(sri_vals) > 0:
            try:
                _, p_val = mannwhitneyu(expert_sri, sri_vals, alternative="two-sided")
            except ValueError:
                pass
            pooled_std = np.sqrt((expert_sri.std() ** 2 + sri_vals.std() ** 2) / 2)
            if pooled_std > 0:
                cohens_d = (expert_sri.mean() - sri_vals.mean()) / pooled_std

        sri_summary.append({
            "condition": cond,
            "mean_SRI": round(m, 2),
            "std_SRI": round(s, 2),
            "median_SRI": round(med, 2),
            "p_vs_expert": p_val,
            "cohens_d": round(cohens_d, 2) if not np.isnan(cohens_d) else np.nan,
            "n": len(sub),
        })

    sri_df = pd.DataFrame(sri_summary)

    print(f"{'Condition':30s} | {'Mean SRI (σ)':>14s} | {'Median':>8s} | {'p vs Expert':>12s} | {'Cohen d':>8s} | {'n':>4s}")
    print("-" * 100)
    for _, r in sri_df.iterrows():
        p_str = f"{r['p_vs_expert']:.2e}" if not np.isnan(r["p_vs_expert"]) else "—"
        d_str = f"{r['cohens_d']:.2f}" if not np.isnan(r["cohens_d"]) else "—"
        print(f"{r['condition']:30s} | {r['mean_SRI']:5.2f} ({r['std_SRI']:.1f})     | {r['median_SRI']:8.2f} | {p_str:>12s} | {d_str:>8s} | {r['n']:4d}")

    # ── Table 2: Stochastic vs Deterministic Call Counts ──
    print("\n" + "=" * 100)
    print("Table 2: Mean Stochastic vs Deterministic Tool Calls per Task")
    print("=" * 100)

    print(f"{'Condition':30s} | {'Stoch calls':>12s} | {'Det calls':>10s} | {'Stoch types':>12s} | {'Det types':>10s} | {'Useful rpts':>12s} | {'Redundant':>10s}")
    print("-" * 120)
    for cond in cond_order:
        sub = df_tools[df_tools["condition"] == cond]
        if len(sub) == 0:
            continue
        print(f"{cond:30s} | "
              f"{sub['n_stochastic_calls'].mean():12.1f} | "
              f"{sub['n_deterministic_calls'].mean():10.1f} | "
              f"{sub['n_stochastic_types'].mean():12.1f} | "
              f"{sub['n_deterministic_types'].mean():10.1f} | "
              f"{sub['n_useful_stochastic_repeats'].mean():12.1f} | "
              f"{sub['n_redundant_deterministic_calls'].mean():10.1f}")

    # ── Table 3: Parameter Usage ──
    print("\n" + "=" * 100)
    print("Table 3: Parameter Specification Rates")
    print("  % of tasks where the agent explicitly set the parameter")
    print("=" * 100)

    print(f"{'Condition':30s} | {'% num_designs':>14s} | {'% num_sequences':>16s} | {'% sampling_temp':>16s} | {'% temperature':>14s}")
    print("-" * 100)
    for cond in cond_order:
        sub = df_tools[df_tools["condition"] == cond]
        if len(sub) == 0:
            continue
        # Only count tasks where the relevant tool was called
        has_backbone = sub[sub["n_backbone_calls"] > 0]
        has_seqdesign = sub[sub["n_stochastic_calls"] > 0]  # any stochastic

        pct_nd = has_backbone["num_designs_specified"].mean() * 100 if len(has_backbone) > 0 else 0
        pct_ns = sub["num_seq_specified"].mean() * 100
        pct_st = sub["sampling_temp_specified"].mean() * 100
        pct_t = sub["temperature_specified"].mean() * 100

        print(f"{cond:30s} | {pct_nd:13.1f}% | {pct_ns:15.1f}% | {pct_st:15.1f}% | {pct_t:13.1f}%")

    # ── Table 4: Stochasticity Awareness in Reasoning ──
    print("\n" + "=" * 100)
    print("Table 4: Stochasticity Awareness in Reasoning Traces")
    print("  Level 3: deep awareness (explicit stochastic/sampling language)")
    print("  Level 2: partial awareness (mentions multiple designs/candidates)")
    print("  Level 1: surface (mentions diversity generically)")
    print("  Level 0: no awareness")
    print("=" * 100)

    print(f"{'Condition':30s} | {'Lvl 3 (deep)':>13s} | {'Lvl 2 (partial)':>16s} | {'Lvl 1 (surface)':>16s} | {'Lvl 0 (none)':>13s} | {'Mean SAS':>9s} | {'Mean trace len':>15s}")
    print("-" * 130)
    for cond in cond_order:
        sub = df[df["condition"] == cond]
        if len(sub) == 0:
            continue
        n = len(sub)
        lvl3 = (sub["stochasticity_awareness_score"] == 3).sum()
        lvl2 = (sub["stochasticity_awareness_score"] == 2).sum()
        lvl1 = (sub["stochasticity_awareness_score"] == 1).sum()
        lvl0 = (sub["stochasticity_awareness_score"] == 0).sum()
        mean_sas = sub["stochasticity_awareness_score"].mean()
        mean_len = sub["reasoning_length"].mean()

        print(f"{cond:30s} | {lvl3/n*100:12.1f}% | {lvl2/n*100:15.1f}% | {lvl1/n*100:15.1f}% | {lvl0/n*100:12.1f}% | {mean_sas:9.2f} | {mean_len:14.0f}")

    # ── Table 5: Adaptive Strategy (De Novo vs Redesign) ──
    print("\n" + "=" * 100)
    print("Table 5: Adaptive Strategy — De Novo vs Redesign SRI")
    print("  Adaptive Ratio > 1 → more stochastic repetition in de novo (understands context)")
    print("=" * 100)

    print(f"{'Condition':30s} | {'De Novo SRI':>12s} | {'Redesign SRI':>13s} | {'Adaptive Ratio':>15s} | {'p (paired)':>12s}")
    print("-" * 100)

    adaptive_rows = []
    for cond in cond_order:
        sub = df_tools[df_tools["condition"] == cond]
        if len(sub) == 0:
            continue
        dn = sub[sub["design_approach"] == "de_novo"]
        rd = sub[sub["design_approach"] == "redesign"]
        dn_sri = dn["SRI"].mean() if len(dn) > 0 else 0
        rd_sri = rd["SRI"].mean() if len(rd) > 0 else 0
        adapt = dn_sri / max(rd_sri, 0.1)

        # Paired comparison on tasks present in both? Not applicable since tasks have one approach.
        # Use Mann-Whitney instead
        p_adapt = np.nan
        if len(dn) > 2 and len(rd) > 2:
            try:
                _, p_adapt = mannwhitneyu(dn["SRI"], rd["SRI"], alternative="two-sided")
            except ValueError:
                pass

        print(f"{cond:30s} | {dn_sri:12.2f} | {rd_sri:13.2f} | {adapt:15.2f} | {f'{p_adapt:.2e}' if not np.isnan(p_adapt) else '—':>12s}")

        adaptive_rows.append({
            "condition": cond,
            "de_novo_SRI": round(dn_sri, 3),
            "redesign_SRI": round(rd_sri, 3),
            "adaptive_ratio": round(adapt, 3),
            "p_paired": p_adapt,
        })

    # ── Table 6: Backbone Generation → Diversity Score Correlation ──
    print("\n" + "=" * 100)
    print("Table 6: Backbone Generation Depth → Diversity Score Correlation")
    print("=" * 100)

    # Overall correlations
    valid = df_tools[(df_tools["n_backbone_calls"] > 0) | (df_tools["diversity_score"] > 0)]
    corr_sets = [
        ("All data (with tools)", df_tools),
        ("Excl. Oracle", df_tools[df_tools["condition"] != "Oracle"]),
        ("Excl. Gemini (already excluded)", df_tools),
        ("LLMs only", df_tools[df_tools["mode"].isin(["benchmark", "user"])]),
        ("Expert only", df_tools[df_tools["condition"] == "Human Expert"]),
    ]

    for label, subset in corr_sets:
        if len(subset) < 5:
            print(f"  {label:40s}: insufficient data (n={len(subset)})")
            continue
        rho_bb, p_bb = spearmanr(subset["n_backbone_calls"], subset["diversity_score"])
        rho_sc, p_sc = spearmanr(subset["n_stochastic_calls"], subset["diversity_score"])
        print(f"  {label:40s}: backbone→div ρ={rho_bb:+.3f} (p={p_bb:.2e}), stoch→div ρ={rho_sc:+.3f} (p={p_sc:.2e}), n={len(subset)}")

    # ── Table 7: Molecular Subject breakdown ──
    print("\n" + "=" * 100)
    print("Table 7: SRI by Molecular Subject (Expert vs LLM avg)")
    print("=" * 100)

    subjects = sorted(df_tools["molecular_subject"].unique())
    expert_data = df_tools[df_tools["condition"] == "Human Expert"]
    llm_data = df_tools[df_tools["mode"].isin(["benchmark", "user"])]

    print(f"{'Subject':20s} | {'Expert SRI':>11s} | {'LLM avg SRI':>12s} | {'Gap':>8s} | {'Expert n':>9s} | {'LLM n':>6s}")
    print("-" * 80)
    for subj in subjects:
        exp_sub = expert_data[expert_data["molecular_subject"] == subj]
        llm_sub = llm_data[llm_data["molecular_subject"] == subj]
        exp_sri = exp_sub["SRI"].mean() if len(exp_sub) > 0 else np.nan
        llm_sri = llm_sub["SRI"].mean() if len(llm_sub) > 0 else np.nan
        gap = (exp_sri - llm_sri) if not (np.isnan(exp_sri) or np.isnan(llm_sri)) else np.nan
        print(f"{subj:20s} | {exp_sri:11.2f} | {llm_sri:12.2f} | {gap:+8.2f} | {len(exp_sub):9d} | {len(llm_sub):6d}")

    # ════════════════════════════════════════════════════════════════════════
    # Summary CSV
    # ════════════════════════════════════════════════════════════════════════

    summary_rows = []
    for cond in cond_order:
        sub = df[df["condition"] == cond]
        sub_tools = df_tools[df_tools["condition"] == cond]
        if len(sub) == 0:
            continue

        dn = sub_tools[sub_tools["design_approach"] == "de_novo"] if len(sub_tools) > 0 else pd.DataFrame()
        rd = sub_tools[sub_tools["design_approach"] == "redesign"] if len(sub_tools) > 0 else pd.DataFrame()

        # Backbone tasks only for num_designs
        has_bb = sub_tools[sub_tools["n_backbone_calls"] > 0] if len(sub_tools) > 0 else pd.DataFrame()

        summary_rows.append({
            "condition": cond,
            "llm": sub["llm"].iloc[0],
            "mode": sub["mode"].iloc[0],
            "n_tasks": len(sub),
            "n_tasks_with_tools": len(sub_tools),
            "mean_SRI": round(sub_tools["SRI"].mean(), 3) if len(sub_tools) > 0 else np.nan,
            "median_SRI": round(sub_tools["SRI"].median(), 3) if len(sub_tools) > 0 else np.nan,
            "std_SRI": round(sub_tools["SRI"].std(), 3) if len(sub_tools) > 0 else np.nan,
            "mean_stochastic_calls": round(sub_tools["n_stochastic_calls"].mean(), 2) if len(sub_tools) > 0 else 0,
            "mean_deterministic_calls": round(sub_tools["n_deterministic_calls"].mean(), 2) if len(sub_tools) > 0 else 0,
            "mean_useful_stoch_repeats": round(sub_tools["n_useful_stochastic_repeats"].mean(), 2) if len(sub_tools) > 0 else 0,
            "mean_redundant_det_calls": round(sub_tools["n_redundant_deterministic_calls"].mean(), 2) if len(sub_tools) > 0 else 0,
            "frac_num_designs_specified": round(has_bb["num_designs_specified"].mean(), 3) if len(has_bb) > 0 else np.nan,
            "frac_num_seq_specified": round(sub_tools["num_seq_specified"].mean(), 3) if len(sub_tools) > 0 else 0,
            "frac_sampling_temp_specified": round(sub_tools["sampling_temp_specified"].mean(), 3) if len(sub_tools) > 0 else 0,
            "frac_temperature_specified": round(sub_tools["temperature_specified"].mean(), 3) if len(sub_tools) > 0 else 0,
            "mean_awareness_score": round(sub["stochasticity_awareness_score"].mean(), 2),
            "de_novo_SRI": round(dn["SRI"].mean(), 3) if len(dn) > 0 else np.nan,
            "redesign_SRI": round(rd["SRI"].mean(), 3) if len(rd) > 0 else np.nan,
            "adaptive_ratio": round((dn["SRI"].mean() / max(rd["SRI"].mean(), 0.1)), 3) if len(dn) > 0 and len(rd) > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "stochastic_understanding_summary.csv", index=False)
    print(f"\nSaved: stochastic_understanding_summary.csv ({len(summary_df)} rows)")

    # ════════════════════════════════════════════════════════════════════════
    # Statistical tests
    # ════════════════════════════════════════════════════════════════════════

    stats_rows = []

    expert_sri = df_tools[df_tools["condition"] == "Human Expert"]["SRI"]

    # Test 1: Expert vs each LLM condition (SRI)
    for cond in cond_order:
        if cond == "Human Expert":
            continue
        sub = df_tools[df_tools["condition"] == cond]
        if len(sub) == 0:
            continue
        try:
            stat, p = mannwhitneyu(expert_sri, sub["SRI"], alternative="two-sided")
            pooled_std = np.sqrt((expert_sri.std() ** 2 + sub["SRI"].std() ** 2) / 2)
            d = (expert_sri.mean() - sub["SRI"].mean()) / pooled_std if pooled_std > 0 else 0
        except (ValueError, ZeroDivisionError):
            stat, p, d = np.nan, np.nan, np.nan

        stats_rows.append({
            "test_name": "SRI: Expert vs LLM",
            "comparison": f"Human Expert vs {cond}",
            "statistic": round(stat, 2) if not np.isnan(stat) else np.nan,
            "p_value": p,
            "effect_size": round(d, 3) if not np.isnan(d) else np.nan,
            "interpretation": "Expert SRI higher" if d > 0 else "LLM SRI higher" if d < 0 else "equal",
        })

    # Test 2: Expert vs all LLMs pooled (SRI)
    all_llm_sri = df_tools[df_tools["mode"].isin(["benchmark", "user"])]["SRI"]
    if len(all_llm_sri) > 0 and len(expert_sri) > 0:
        try:
            stat, p = mannwhitneyu(expert_sri, all_llm_sri, alternative="two-sided")
            pooled_std = np.sqrt((expert_sri.std() ** 2 + all_llm_sri.std() ** 2) / 2)
            d = (expert_sri.mean() - all_llm_sri.mean()) / pooled_std if pooled_std > 0 else 0
        except (ValueError, ZeroDivisionError):
            stat, p, d = np.nan, np.nan, np.nan

        stats_rows.append({
            "test_name": "SRI: Expert vs ALL LLMs",
            "comparison": "Human Expert vs All LLMs pooled",
            "statistic": round(stat, 2) if not np.isnan(stat) else np.nan,
            "p_value": p,
            "effect_size": round(d, 3) if not np.isnan(d) else np.nan,
            "interpretation": "Expert SRI higher" if d > 0 else "LLM SRI higher" if d < 0 else "equal",
        })

    # Test 3: Hardcoded vs LLMs (SRI)
    hc_sri = df_tools[df_tools["condition"] == "Hardcoded Pipeline"]["SRI"]
    if len(hc_sri) > 0 and len(all_llm_sri) > 0:
        try:
            stat, p = mannwhitneyu(hc_sri, all_llm_sri, alternative="two-sided")
            pooled_std = np.sqrt((hc_sri.std() ** 2 + all_llm_sri.std() ** 2) / 2)
            d = (hc_sri.mean() - all_llm_sri.mean()) / pooled_std if pooled_std > 0 else 0
        except (ValueError, ZeroDivisionError):
            stat, p, d = np.nan, np.nan, np.nan

        stats_rows.append({
            "test_name": "SRI: Hardcoded vs ALL LLMs",
            "comparison": "Hardcoded Pipeline vs All LLMs pooled",
            "statistic": round(stat, 2) if not np.isnan(stat) else np.nan,
            "p_value": p,
            "effect_size": round(d, 3) if not np.isnan(d) else np.nan,
            "interpretation": "Hardcoded SRI higher" if d > 0 else "LLM SRI higher" if d < 0 else "equal",
        })

    # Test 4: De novo vs Redesign SRI (within each condition)
    for cond in cond_order:
        sub = df_tools[df_tools["condition"] == cond]
        dn = sub[sub["design_approach"] == "de_novo"]["SRI"]
        rd = sub[sub["design_approach"] == "redesign"]["SRI"]
        if len(dn) < 3 or len(rd) < 3:
            continue
        try:
            stat, p = mannwhitneyu(dn, rd, alternative="two-sided")
            pooled_std = np.sqrt((dn.std() ** 2 + rd.std() ** 2) / 2)
            d = (dn.mean() - rd.mean()) / pooled_std if pooled_std > 0 else 0
        except (ValueError, ZeroDivisionError):
            stat, p, d = np.nan, np.nan, np.nan

        stats_rows.append({
            "test_name": "Adaptive: De Novo vs Redesign SRI",
            "comparison": f"{cond}",
            "statistic": round(stat, 2) if not np.isnan(stat) else np.nan,
            "p_value": p,
            "effect_size": round(d, 3) if not np.isnan(d) else np.nan,
            "interpretation": "de novo SRI higher" if d > 0 else "redesign SRI higher" if d < 0 else "equal",
        })

    # Test 5: Correlation: stochastic_calls → diversity_score (all tools data)
    if len(df_tools) > 5:
        rho, p = spearmanr(df_tools["n_stochastic_calls"], df_tools["diversity_score"])
        stats_rows.append({
            "test_name": "Correlation: stoch_calls → diversity",
            "comparison": "All conditions with tools",
            "statistic": round(rho, 3),
            "p_value": p,
            "effect_size": round(rho, 3),
            "interpretation": f"Spearman rho={rho:.3f}",
        })

    # Test 5b: Correlation: backbone_calls → diversity_score
    if len(df_tools) > 5:
        rho, p = spearmanr(df_tools["n_backbone_calls"], df_tools["diversity_score"])
        stats_rows.append({
            "test_name": "Correlation: backbone_calls → diversity",
            "comparison": "All conditions with tools",
            "statistic": round(rho, 3),
            "p_value": p,
            "effect_size": round(rho, 3),
            "interpretation": f"Spearman rho={rho:.3f}",
        })

    # Test 5c: LLMs only
    llm_only = df_tools[df_tools["mode"].isin(["benchmark", "user"])]
    if len(llm_only) > 5:
        rho, p = spearmanr(llm_only["n_stochastic_calls"], llm_only["diversity_score"])
        stats_rows.append({
            "test_name": "Correlation: stoch_calls → diversity (LLMs)",
            "comparison": "LLMs only",
            "statistic": round(rho, 3),
            "p_value": p,
            "effect_size": round(rho, 3),
            "interpretation": f"Spearman rho={rho:.3f}",
        })

    # Test 6: Awareness score vs SRI correlation
    df_aware = df_tools[df_tools["reasoning_length"] > 50]  # need meaningful reasoning
    if len(df_aware) > 5:
        rho, p = spearmanr(df_aware["stochasticity_awareness_score"], df_aware["SRI"])
        stats_rows.append({
            "test_name": "Correlation: awareness → SRI",
            "comparison": "All with tools and reasoning",
            "statistic": round(rho, 3),
            "p_value": p,
            "effect_size": round(rho, 3),
            "interpretation": f"Spearman rho={rho:.3f}",
        })

    # Test 7: Awareness score → diversity_score
    if len(df_aware) > 5:
        rho, p = spearmanr(df_aware["stochasticity_awareness_score"], df_aware["diversity_score"])
        stats_rows.append({
            "test_name": "Correlation: awareness → diversity",
            "comparison": "All with tools and reasoning",
            "statistic": round(rho, 3),
            "p_value": p,
            "effect_size": round(rho, 3),
            "interpretation": f"Spearman rho={rho:.3f}",
        })

    # Test 8: Sensitivity check — rosetta_relax as stochastic
    print("\n" + "=" * 100)
    print("Sensitivity Check: rosetta_relax reclassified as stochastic")
    print("=" * 100)

    # Recompute SRI with rosetta_relax moved to stochastic
    STOCH_ALT = STOCHASTIC_TOOLS | {"rosetta_relax"}
    DET_ALT = DETERMINISTIC_TOOLS - {"rosetta_relax"}

    alt_expert_sris = []
    alt_llm_sris = []

    for _, row in df_tools.iterrows():
        result = None
        # Re-derive from tool call data already in the row
        # We need to re-count from raw data, but we only have aggregates
        # Use a proxy: if the task had rosetta_relax calls, they move from det to stoch
        # This is approximate; for exact we'd need raw tool logs
        pass  # We'll do this properly by re-traversing

    # For exact sensitivity, re-process from raw data
    alt_rows = []
    for entry in all_results:
        result = entry["result"]
        tid = result["task_id"]
        cond = entry["condition"]
        raw = result.get("raw_output", {})
        tool_log = raw.get("tool_call_log", [])
        if not tool_log:
            continue

        design_calls = [tc for tc in tool_log if tc["tool"] in (STOCH_ALT | DET_ALT)]
        stoch_calls = [tc for tc in design_calls if tc["tool"] in STOCH_ALT]
        det_calls = [tc for tc in design_calls if tc["tool"] in DET_ALT]

        n_st = len(stoch_calls)
        n_dt = len(det_calls)
        st_types = len(set(tc["tool"] for tc in stoch_calls))
        dt_types = len(set(tc["tool"] for tc in det_calls))

        sr = n_st / max(st_types, 1)
        dr = n_dt / max(dt_types, 1)
        sri_alt = sr / max(dr, 0.5)

        alt_rows.append({"condition": cond, "SRI_alt": sri_alt})

    alt_df = pd.DataFrame(alt_rows)
    alt_expert = alt_df[alt_df["condition"] == "Human Expert"]["SRI_alt"]
    alt_llm = alt_df[alt_df["condition"].isin([c for c, i in CONDITION_MAP.items() if i["mode"] in ("benchmark", "user")])]["SRI_alt"]

    if len(alt_expert) > 0 and len(alt_llm) > 0:
        try:
            stat, p = mannwhitneyu(alt_expert, alt_llm, alternative="two-sided")
            pooled_std = np.sqrt((alt_expert.std() ** 2 + alt_llm.std() ** 2) / 2)
            d = (alt_expert.mean() - alt_llm.mean()) / pooled_std if pooled_std > 0 else 0
        except (ValueError, ZeroDivisionError):
            stat, p, d = np.nan, np.nan, np.nan

        print(f"  Original SRI: Expert={expert_sri.mean():.2f}, LLMs={all_llm_sri.mean():.2f}")
        print(f"  Alt SRI (rosetta_relax=stoch): Expert={alt_expert.mean():.2f}, LLMs={alt_llm.mean():.2f}")
        print(f"  Mann-Whitney p={p:.2e}, Cohen's d={d:.2f}")
        print(f"  → {'Robust' if p < 0.05 else 'Not robust'}: result {'holds' if p < 0.05 else 'does NOT hold'} under reclassification")

        stats_rows.append({
            "test_name": "Sensitivity: rosetta_relax→stochastic",
            "comparison": "Expert vs All LLMs (alt classification)",
            "statistic": round(stat, 2) if not np.isnan(stat) else np.nan,
            "p_value": p,
            "effect_size": round(d, 3) if not np.isnan(d) else np.nan,
            "interpretation": f"Alt: Expert={alt_expert.mean():.2f} vs LLM={alt_llm.mean():.2f}",
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "stochastic_understanding_stats.csv", index=False)
    print(f"\nSaved: stochastic_understanding_stats.csv ({len(stats_df)} rows)")

    # ════════════════════════════════════════════════════════════════════════
    # Final verdict
    # ════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("VERDICT: Evidence Strength Assessment")
    print("=" * 100)

    exp_mean = expert_sri.mean() if len(expert_sri) > 0 else 0
    llm_mean = all_llm_sri.mean() if len(all_llm_sri) > 0 else 0

    # Check key metrics
    sri_gap = exp_mean - llm_mean
    expert_vs_llm_row = [r for r in stats_rows if r["test_name"] == "SRI: Expert vs ALL LLMs"]
    p_val = expert_vs_llm_row[0]["p_value"] if expert_vs_llm_row else 1.0
    d_val = expert_vs_llm_row[0]["effect_size"] if expert_vs_llm_row else 0.0

    # Awareness gap
    expert_awareness = df[df["condition"] == "Human Expert"]["stochasticity_awareness_score"].mean()
    llm_awareness = df[df["mode"].isin(["benchmark", "user"])]["stochasticity_awareness_score"].mean()

    print(f"  SRI gap (Expert - LLM):    {sri_gap:+.2f}  (Expert={exp_mean:.2f}, LLM={llm_mean:.2f})")
    print(f"  Statistical significance:  p = {p_val:.2e}")
    print(f"  Effect size (Cohen's d):   {d_val:.2f}")
    print(f"  Awareness gap:             Expert={expert_awareness:.2f}, LLM={llm_awareness:.2f}")

    if p_val < 0.001 and abs(d_val) > 0.5:
        print("\n  → SCENARIO A: Strong evidence. Recommend independent subsection.")
    elif p_val < 0.05 and abs(d_val) > 0.2:
        print("\n  → SCENARIO B: Moderate evidence. Recommend sub-finding within depth gap.")
    else:
        print("\n  → SCENARIO C: Weak evidence. Keep as context/discussion only.")

    return df, summary_df, stats_df


if __name__ == "__main__":
    run_analysis()
