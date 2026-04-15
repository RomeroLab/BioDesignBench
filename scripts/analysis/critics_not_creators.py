#!/usr/bin/env python3
"""Critics Not Creators: Generative vs Evaluative Tool Usage Analysis.

Quantifies the pattern where LLM agents preferentially use evaluative tools
(scoring, structure prediction) while avoiding generative tools (backbone
generation, sequence design) — behaving as critics, not creators.

Outputs:
  results/analysis/critics_not_creators.csv          - per-condition gen/eval summary
  results/analysis/critics_not_creators_patterns.csv - per-task pattern detection
  results/analysis/reasoning_trace_examples.md       - 3 curated examples for paper
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from scripts.analysis.load_results import CONDITION_MAP, load_all

# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------

GENERATIVE_TOOLS = frozenset({
    "generate_backbone",   # RFdiffusion
    "design_binder",       # RFdiffusion + ProteinMPNN composite
    "optimize_sequence",   # ProteinMPNN inverse folding
    "rosetta_design",      # Rosetta fixed-backbone design
})

EVALUATIVE_TOOLS = frozenset({
    "predict_structure",          # AlphaFold2 monomer
    "predict_structure_boltz",    # Boltz structure prediction
    "predict_complex",            # AlphaFold2-Multimer
    "predict_affinity_boltz",     # Boltz affinity prediction
    "score_stability",            # Thermodynamic stability
    "rosetta_score",              # Rosetta energy
    "rosetta_relax",              # Rosetta relaxation
    "analyze_interface",          # Interface analysis
    "rosetta_interface_score",    # Rosetta interface energy
    "energy_minimize",            # Energy minimization
    "validate_design",            # Multi-metric validation
    "suggest_hotspots",           # Hotspot identification
})

# Amino acid regex for LLM-as-generator detection (20+ uppercase AA letters)
_AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")
_AA_PATTERN = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY]{20,})\b")

# Patterns suggesting the LLM generated a sequence itself
_LLM_GEN_PHRASES = [
    r"here is (?:my|the|a) designed sequence",
    r"i(?:'ll| will) design the sequence",
    r"proposed sequence",
    r"my designed sequence",
    r"generated sequence",
    r"designed the following sequence",
]
_LLM_GEN_RE = re.compile("|".join(_LLM_GEN_PHRASES), re.IGNORECASE)

# Agent conditions (exclude scripted baselines for pattern analysis)
LLM_CONDITIONS = [
    "DeepSeek V3 user", "DeepSeek V3 benchmark",
    "GPT-5 user", "GPT-5 benchmark",
    "Sonnet 4.5 user", "Sonnet 4.5 benchmark",
    "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark",
]

ALL_CONDITIONS = LLM_CONDITIONS + ["Hardcoded Pipeline", "Human Expert", "Oracle"]

# LLM pairs for guided effect
LLM_PAIRS = [
    ("DeepSeek V3", "DeepSeek V3 user", "DeepSeek V3 benchmark"),
    ("GPT-5", "GPT-5 user", "GPT-5 benchmark"),
    ("Sonnet 4.5", "Sonnet 4.5 user", "Sonnet 4.5 benchmark"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark"),
]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_result_json(condition: str, task_id: str) -> dict[str, Any] | None:
    """Load result.json for a (condition, task_id) pair."""
    info = CONDITION_MAP.get(condition)
    if info is None:
        return None
    result_file = info["path"] / task_id / "result.json"
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def _get_tool_call_log(result: dict) -> list[dict]:
    """Extract ordered tool call log from result."""
    return result.get("raw_output", {}).get("tool_call_log", [])


def _get_reasoning_trace(result: dict) -> str:
    """Extract reasoning trace text from result."""
    return result.get("raw_output", {}).get("reasoning_trace", "")


def _get_tool_sequence(tcl: list[dict]) -> list[str]:
    """Extract ordered list of tool names from tool call log."""
    return [tc.get("tool", "") for tc in tcl]


# ---------------------------------------------------------------------------
# Per-task pattern detection
# ---------------------------------------------------------------------------

def detect_patterns(condition: str, task_id: str, result: dict) -> dict:
    """Detect critics-not-creators patterns for a single task."""
    tcl = _get_tool_call_log(result)
    tool_seq = _get_tool_sequence(tcl)
    reasoning = _get_reasoning_trace(result)
    tools_used = set(result.get("tools_used", []))

    # Count generative and evaluative calls
    gen_calls = [t for t in tool_seq if t in GENERATIVE_TOOLS]
    eval_calls = [t for t in tool_seq if t in EVALUATIVE_TOOLS]
    other_calls = [t for t in tool_seq if t not in GENERATIVE_TOOLS and t not in EVALUATIVE_TOOLS]

    n_gen = len(gen_calls)
    n_eval = len(eval_calls)
    n_other = len(other_calls)
    n_bio = n_gen + n_eval  # biological tools only

    gen_ratio = n_gen / n_bio if n_bio > 0 else 0.0

    # Pattern A: First bio tool call is evaluative
    bio_seq = [t for t in tool_seq if t in GENERATIVE_TOOLS or t in EVALUATIVE_TOOLS]
    eval_before_gen = False
    if bio_seq:
        eval_before_gen = bio_seq[0] in EVALUATIVE_TOOLS

    # Pattern B: Generative tools completely skipped
    gen_skipped = (n_gen == 0) and (n_eval > 0 or n_other > 0)

    # Pattern C: LLM-as-generator — agent generated amino acid sequence in reasoning
    llm_as_generator = False
    aa_matches = _AA_PATTERN.findall(reasoning)
    if aa_matches:
        # Filter out common short protein names that happen to match
        long_aa = [m for m in aa_matches if len(m) >= 30]
        if long_aa:
            llm_as_generator = True
    if _LLM_GEN_RE.search(reasoning):
        llm_as_generator = True

    # Pattern D: First bio tool evaluates the input (not an agent-generated design)
    eval_of_input = False
    if bio_seq and bio_seq[0] in EVALUATIVE_TOOLS:
        # If the first evaluative tool is called before any generative tool,
        # it's likely evaluating the input structure
        gen_seen = False
        for t in tool_seq:
            if t in GENERATIVE_TOOLS:
                gen_seen = True
                break
            if t in EVALUATIVE_TOOLS and not gen_seen:
                eval_of_input = True
                break

    return {
        "task_id": task_id,
        "condition": condition,
        "n_gen_calls": n_gen,
        "n_eval_calls": n_eval,
        "n_other_calls": n_other,
        "gen_ratio": round(gen_ratio, 4),
        "eval_before_gen": eval_before_gen,
        "gen_skipped": gen_skipped,
        "llm_as_generator": llm_as_generator,
        "eval_of_input": eval_of_input,
        "tool_sequence": "|".join(tool_seq) if tool_seq else "",
        "reasoning_len": len(reasoning),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_condition_summary(patterns_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per condition."""
    rows = []
    for condition, grp in patterns_df.groupby("condition"):
        n = len(grp)
        total_gen = grp["n_gen_calls"].sum()
        total_eval = grp["n_eval_calls"].sum()
        total_bio = total_gen + total_eval

        rows.append({
            "condition": condition,
            "n_tasks": n,
            "total_gen_calls": int(total_gen),
            "total_eval_calls": int(total_eval),
            "gen_ratio": round(total_gen / total_bio, 4) if total_bio > 0 else 0.0,
            "mean_gen_per_task": round(total_gen / n, 2),
            "mean_eval_per_task": round(total_eval / n, 2),
            "frac_eval_before_gen": round(grp["eval_before_gen"].mean(), 4),
            "frac_gen_skipped": round(grp["gen_skipped"].mean(), 4),
            "frac_llm_as_generator": round(grp["llm_as_generator"].mean(), 4),
            "frac_eval_of_input": round(grp["eval_of_input"].mean(), 4),
        })

    return pd.DataFrame(rows)


def compute_guided_effect(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute guided vs unguided delta for gen_ratio."""
    idx = summary_df.set_index("condition")
    rows = []
    for model, guided, unguided in LLM_PAIRS:
        if guided not in idx.index or unguided not in idx.index:
            continue
        g = idx.loc[guided]
        u = idx.loc[unguided]
        rows.append({
            "model": model,
            "gen_ratio_guided": g["gen_ratio"],
            "gen_ratio_unguided": u["gen_ratio"],
            "delta_gen_ratio": round(g["gen_ratio"] - u["gen_ratio"], 4),
            "frac_gen_skipped_guided": g["frac_gen_skipped"],
            "frac_gen_skipped_unguided": u["frac_gen_skipped"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reasoning trace example extraction
# ---------------------------------------------------------------------------

def find_examples(patterns_df: pd.DataFrame) -> list[dict]:
    """Find 3 curated examples for the paper."""
    examples = []

    # Example 1: Case B (Operationalization Gap) — Sonnet benchmark
    # Agent mentions RFdiffusion but doesn't call it
    sonnet_bm = patterns_df[
        (patterns_df["condition"] == "Sonnet 4.5 benchmark") &
        (patterns_df["gen_skipped"]) &
        (patterns_df["n_eval_calls"] > 0) &
        (patterns_df["reasoning_len"] > 100)
    ]
    if len(sonnet_bm) > 0:
        row = sonnet_bm.iloc[0]
        result = _load_result_json(row["condition"], row["task_id"])
        if result:
            reasoning = _get_reasoning_trace(result)
            tool_seq = _get_tool_sequence(_get_tool_call_log(result))
            examples.append({
                "type": "Case B — Operationalization Gap",
                "condition": row["condition"],
                "task_id": row["task_id"],
                "description": "Agent mentions pipeline steps but fails to execute generative tools",
                "reasoning_excerpt": reasoning[:500],
                "tool_sequence": tool_seq[:10],
                "gen_calls": int(row["n_gen_calls"]),
                "eval_calls": int(row["n_eval_calls"]),
            })

    # Example 2: LLM-as-generator — prefer benchmark conditions, fallback through all
    gpt_llm_gen = pd.DataFrame()
    for cond in ["GPT-5 benchmark", "GPT-5 user",
                 "Sonnet 4.5 benchmark", "DeepSeek V3 benchmark",
                 "Sonnet 4.5 user", "DeepSeek V3 user",
                 "Gemini 2.5 Pro benchmark", "Gemini 2.5 Pro user"]:
        candidates = patterns_df[
            (patterns_df["condition"] == cond) &
            (patterns_df["llm_as_generator"]) &
            (patterns_df["reasoning_len"] > 200)
        ]
        if len(candidates) > 0:
            # Prefer ones where gen_skipped=True (pure LLM generation)
            pure = candidates[candidates["gen_skipped"]]
            gpt_llm_gen = pure if len(pure) > 0 else candidates
            break
    if len(gpt_llm_gen) > 0:
        row = gpt_llm_gen.iloc[0]
        result = _load_result_json(row["condition"], row["task_id"])
        if result:
            reasoning = _get_reasoning_trace(result)
            # Find the AA sequence in the reasoning
            aa_matches = _AA_PATTERN.findall(reasoning)
            long_aa = [m for m in aa_matches if len(m) >= 30]
            tool_seq = _get_tool_sequence(_get_tool_call_log(result))
            examples.append({
                "type": "LLM-as-Generator",
                "condition": row["condition"],
                "task_id": row["task_id"],
                "description": "Agent generates amino acid sequence directly via LLM instead of using design tools",
                "reasoning_excerpt": reasoning[:500],
                "tool_sequence": tool_seq[:10],
                "generated_sequences": [s[:50] + "..." for s in long_aa[:3]] if long_aa else [],
                "gen_calls": int(row["n_gen_calls"]),
                "eval_calls": int(row["n_eval_calls"]),
            })

    # Example 3: Case A (Full Competence) — DeepSeek V3 user
    ds_full = patterns_df[
        (patterns_df["condition"] == "DeepSeek V3 user") &
        (~patterns_df["gen_skipped"]) &
        (patterns_df["n_gen_calls"] >= 2) &
        (patterns_df["n_eval_calls"] >= 2)
    ]
    if len(ds_full) > 0:
        row = ds_full.iloc[0]
        result = _load_result_json(row["condition"], row["task_id"])
        if result:
            tool_seq = _get_tool_sequence(_get_tool_call_log(result))
            examples.append({
                "type": "Case A — Full Competence (control)",
                "condition": row["condition"],
                "task_id": row["task_id"],
                "description": "Agent correctly executes both generative and evaluative pipeline steps",
                "tool_sequence": tool_seq[:15],
                "gen_calls": int(row["n_gen_calls"]),
                "eval_calls": int(row["n_eval_calls"]),
            })

    return examples


def write_examples_md(examples: list[dict], out_path: Path) -> None:
    """Write reasoning trace examples to markdown."""
    lines = ["# Reasoning Trace Examples for Section 2.3", ""]

    for i, ex in enumerate(examples, 1):
        lines.append(f"## Example {i}: {ex['type']}")
        lines.append(f"- **Condition**: {ex['condition']}")
        lines.append(f"- **Task**: {ex['task_id']}")
        lines.append(f"- **Description**: {ex['description']}")
        lines.append(f"- **Generative calls**: {ex['gen_calls']}")
        lines.append(f"- **Evaluative calls**: {ex['eval_calls']}")
        lines.append("")

        if "tool_sequence" in ex:
            lines.append("### Tool Sequence")
            lines.append("```")
            for j, t in enumerate(ex["tool_sequence"]):
                marker = "[GEN]" if t in GENERATIVE_TOOLS else "[EVAL]" if t in EVALUATIVE_TOOLS else "[OTHER]"
                lines.append(f"  {j+1}. {t} {marker}")
            lines.append("```")
            lines.append("")

        if "reasoning_excerpt" in ex:
            lines.append("### Reasoning Excerpt")
            lines.append("```")
            lines.append(ex["reasoning_excerpt"])
            lines.append("```")
            lines.append("")

        if ex.get("generated_sequences"):
            lines.append("### LLM-Generated Sequences Found")
            for seq in ex["generated_sequences"]:
                lines.append(f"- `{seq}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(summary_df: pd.DataFrame, guided_df: pd.DataFrame,
                  patterns_df: pd.DataFrame):
    """Print formatted summary tables."""
    print("\n" + "=" * 85)
    print("Critics Not Creators: Generative vs Evaluative Tool Usage")
    print("=" * 85)

    print("\n── Per-Condition Summary ────────────────────────────────────────")
    print(f"{'Condition':<28s} {'N':>3s} {'Gen':>4s} {'Eval':>5s} "
          f"{'Ratio':>6s} {'EvalFirst':>9s} {'GenSkip':>8s} {'LLMGen':>7s} {'EvalInp':>8s}")
    print("-" * 85)
    for _, r in summary_df.sort_values("gen_ratio", ascending=False).iterrows():
        print(f"{r['condition']:<28s} {r['n_tasks']:>3.0f} "
              f"{r['total_gen_calls']:>4.0f} {r['total_eval_calls']:>5.0f} "
              f"{r['gen_ratio']:>6.3f} "
              f"{r['frac_eval_before_gen']*100:>8.1f}% "
              f"{r['frac_gen_skipped']*100:>7.1f}% "
              f"{r['frac_llm_as_generator']*100:>6.1f}% "
              f"{r['frac_eval_of_input']*100:>7.1f}%")

    if len(guided_df) > 0:
        print("\n── Guided vs Unguided Gen Ratio ────────────────────────────────")
        print(f"{'Model':<20s} {'Guided':>8s} {'Unguided':>9s} {'Δ':>7s} "
              f"{'Skip_G':>7s} {'Skip_U':>7s}")
        print("-" * 60)
        for _, r in guided_df.iterrows():
            print(f"{r['model']:<20s} "
                  f"{r['gen_ratio_guided']:>8.3f} {r['gen_ratio_unguided']:>9.3f} "
                  f"{r['delta_gen_ratio']:>+7.3f} "
                  f"{r['frac_gen_skipped_guided']*100:>6.1f}% "
                  f"{r['frac_gen_skipped_unguided']*100:>6.1f}%")

    # Quick highlight: overall pattern prevalence for LLMs
    llm_only = patterns_df[patterns_df["condition"].isin(LLM_CONDITIONS)]
    n_llm = len(llm_only)
    if n_llm > 0:
        print(f"\n── Overall LLM Pattern Prevalence (N={n_llm}) ─────────────────")
        print(f"  Eval before gen:    {llm_only['eval_before_gen'].mean()*100:5.1f}%")
        print(f"  Gen skipped:        {llm_only['gen_skipped'].mean()*100:5.1f}%")
        print(f"  LLM-as-generator:   {llm_only['llm_as_generator'].mean()*100:5.1f}%")
        print(f"  Eval of input:      {llm_only['eval_of_input'].mean()*100:5.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/5] Loading results...")
    df = load_all()
    # Filter to conditions we care about
    df = df[df["condition"].isin(ALL_CONDITIONS)].copy()
    print(f"  {len(df)} rows")

    print("[2/5] Detecting patterns per task...")
    pattern_rows = []
    for _, row in df.iterrows():
        condition = row["condition"]
        task_id = row["task_id"]
        result = _load_result_json(condition, task_id)
        if result is None:
            continue
        patterns = detect_patterns(condition, task_id, result)
        patterns["design_approach"] = row.get("design_approach", "unknown")
        patterns["molecular_subject"] = row.get("molecular_subject", "unknown")
        pattern_rows.append(patterns)

    patterns_df = pd.DataFrame(pattern_rows)
    print(f"  {len(patterns_df)} task×condition pairs analyzed")

    print("[3/5] Computing summaries...")
    summary = compute_condition_summary(patterns_df)
    guided = compute_guided_effect(summary)

    print("[4/5] Finding reasoning trace examples...")
    examples = find_examples(patterns_df)
    print(f"  Found {len(examples)} examples")

    # Save
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / "critics_not_creators.csv", index=False)
    patterns_df.to_csv(out_dir / "critics_not_creators_patterns.csv", index=False)
    if len(guided) > 0:
        guided.to_csv(out_dir / "critics_not_creators_guided.csv", index=False)

    examples_path = out_dir / "reasoning_trace_examples.md"
    write_examples_md(examples, examples_path)

    print(f"\n[5/5] Saved to {out_dir}/")
    print(f"  critics_not_creators.csv          ({len(summary)} rows)")
    print(f"  critics_not_creators_patterns.csv  ({len(patterns_df)} rows)")
    if len(guided) > 0:
        print(f"  critics_not_creators_guided.csv    ({len(guided)} rows)")
    print(f"  reasoning_trace_examples.md        ({len(examples)} examples)")

    print_summary(summary, guided, patterns_df)


if __name__ == "__main__":
    main()
