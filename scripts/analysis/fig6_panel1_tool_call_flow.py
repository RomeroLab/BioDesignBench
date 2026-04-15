#!/usr/bin/env python3
"""
Figure 6 Panel 1: Tool Call Order Flow Analysis for Guided-Mode (User) Agents.

Extracts tool call order data from result.json files for 4 guided conditions,
maps tools to pipeline stages, and computes:
  - Distribution of first/last tool call stage
  - Mean stage transition matrix
  - Stage usage rate per position (1st through 10th call)

Output: results/analysis/tool_call_flow_by_model.csv
"""

import json
import glob
import os
import csv
from collections import Counter, defaultdict
from pathlib import Path

BASE = "/home/jk661/projects/BioDesignBench"
RESULTS = os.path.join(BASE, "results")
OUTPUT_DIR = os.path.join(RESULTS, "analysis")

# ── Agent configuration ──────────────────────────────────────────────────────
# Maps display name -> list of (glob pattern, agent_id_substring) search specs
# Ordered by priority: newest directories first
AGENT_CONFIG = {
    "GPT-5 user": [
        (f"{RESULTS}/full_run_user/runs/*/agents/gpt5-tools-user/*/result.json", "gpt5-tools-user"),
        (f"{RESULTS}/gpt5-user/runs/*/agents/gpt5-tools-user/*/result.json", "gpt5-tools-user"),
        (f"{RESULTS}/runs/*/agents/gpt5-tools-user/*/result.json", "gpt5-tools-user"),
    ],
    "DeepSeek V3 user": [
        (f"{RESULTS}/full_run_user/runs/*/agents/deepseek-v3-tools-user/*/result.json", "deepseek-v3-tools-user"),
        (f"{RESULTS}/deepseek-v3-user/runs/*/agents/deepseek-v3-tools-user/*/result.json", "deepseek-v3-tools-user"),
        (f"{RESULTS}/runs/*/agents/deepseek-v3-tools-user/*/result.json", "deepseek-v3-tools-user"),
    ],
    "Sonnet 4.5 user": [
        (f"{RESULTS}/full_run_user/runs/*/agents/sonnet-4.5-tools-user/*/result.json", "sonnet-4.5-tools-user"),
        (f"{RESULTS}/sonnet-4.5-user/runs/*/agents/sonnet-4.5-tools-user/*/result.json", "sonnet-4.5-tools-user"),
        (f"{RESULTS}/runs/*/agents/sonnet-4.5-tools-user/*/result.json", "sonnet-4.5-tools-user"),
    ],
    "Gemini 2.5 Pro user": [
        (f"{RESULTS}/full_run_user/runs/*/agents/gemini-2.5-pro-tools-user/*/result.json", "gemini-2.5-pro-tools-user"),
        (f"{RESULTS}/gemini3-pro-user/runs/*/agents/gemini3-pro-tools-user/*/result.json", "gemini3-pro-tools-user"),
        (f"{RESULTS}/runs/*/agents/gemini-2.5-pro-tools-user/*/result.json", "gemini-2.5-pro-tools-user"),
        (f"{RESULTS}/runs/*/agents/gemini3-pro-tools-user/*/result.json", "gemini3-pro-tools-user"),
    ],
}

# ── Tool-to-stage mapping ────────────────────────────────────────────────────
# Static assignments (pyrosetta handled dynamically)
TOOL_STAGE_STATIC = {
    "rfdiffusion": "backbone_generation",
    "proteinmpnn": "sequence_design",
    "esmfold": "structure_prediction",
    "alphafold2": "structure_prediction",
    "boltz": "structure_prediction",
    "esm2": "scoring",
    "suggesthotspots": "scoring",
    "analyzeinterface": "scoring",
    "rosetta": "scoring",
    "openmm": "scoring",
    "designantibodycdrs": "sequence_design",
    "antibodydesign": "sequence_design",
    "generatedesignbundle": "backbone_generation",
}

STRUCTURE_PREDICTION_TOOLS = {"esmfold", "alphafold2", "boltz"}

STAGES = ["backbone_generation", "sequence_design", "structure_prediction", "scoring", "other"]
MAX_POSITION = 10


def classify_pyrosetta(tool_order: list[str], idx: int) -> str:
    """Classify pyrosetta as sequence_design or scoring based on context.

    Heuristic: if pyrosetta appears before any structure_prediction tool
    in the sequence, classify as sequence_design; if after, classify as scoring.
    """
    # Find the index of the first structure_prediction tool
    first_struct_idx = None
    for i, t in enumerate(tool_order):
        if t in STRUCTURE_PREDICTION_TOOLS:
            first_struct_idx = i
            break

    if first_struct_idx is None:
        # No structure prediction tools in the sequence at all.
        # If rfdiffusion appeared before, likely scoring; otherwise sequence_design.
        has_rfdiffusion_before = any(
            tool_order[j] == "rfdiffusion" for j in range(idx)
        )
        if has_rfdiffusion_before:
            return "sequence_design"
        return "scoring"

    if idx < first_struct_idx:
        return "sequence_design"
    else:
        return "scoring"


def map_tool_to_stage(tool_name: str, tool_order: list[str], idx: int) -> str:
    """Map a tool name to its pipeline stage."""
    tool_lower = tool_name.lower()
    if tool_lower == "pyrosetta":
        return classify_pyrosetta(tool_order, idx)
    if tool_lower in TOOL_STAGE_STATIC:
        return TOOL_STAGE_STATIC[tool_lower]
    return "other"


def collect_results(agent_name: str, search_specs: list) -> dict:
    """Collect result.json data for an agent, deduplicating by task_id (newest first).

    Returns dict of task_id -> tool_order list.
    """
    seen_tasks = {}  # task_id -> tool_order

    for pattern, _ in search_specs:
        files = sorted(glob.glob(pattern), reverse=True)  # newest run dirs first
        for fpath in files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            task_id = data.get("task_id")
            if not task_id:
                continue

            tool_order = data.get("orchestration_metrics", {}).get("actual_tool_order", [])
            if not tool_order:
                continue

            # Keep first occurrence (newest run) per task_id
            if task_id not in seen_tasks:
                seen_tasks[task_id] = tool_order

    return seen_tasks


def compute_stage_sequences(task_orders: dict) -> list:
    """Convert tool orders to stage sequences.

    Returns list of stage sequences (one per task).
    """
    stage_sequences = []
    for task_id, tool_order in task_orders.items():
        stages = []
        for i, tool in enumerate(tool_order):
            stage = map_tool_to_stage(tool, tool_order, i)
            stages.append(stage)
        stage_sequences.append(stages)
    return stage_sequences


def compute_metrics(stage_sequences: list) -> dict:
    """Compute all metrics for a set of stage sequences."""
    n_tasks = len(stage_sequences)
    if n_tasks == 0:
        return {
            "n_tasks": 0,
            "first_stage_dist": {},
            "last_stage_dist": {},
            "transition_matrix": {},
            "position_usage": {},
        }

    # 1) Distribution of first tool call stage
    first_stage_counter = Counter()
    for seq in stage_sequences:
        if seq:
            first_stage_counter[seq[0]] += 1
    first_stage_dist = {s: first_stage_counter.get(s, 0) / n_tasks for s in STAGES}

    # 2) Distribution of last tool call stage
    last_stage_counter = Counter()
    for seq in stage_sequences:
        if seq:
            last_stage_counter[seq[-1]] += 1
    last_stage_dist = {s: last_stage_counter.get(s, 0) / n_tasks for s in STAGES}

    # 3) Mean stage transition matrix (from_stage -> to_stage counts, normalized per task)
    transition_counts = defaultdict(lambda: defaultdict(float))
    tasks_with_transitions = 0
    for seq in stage_sequences:
        if len(seq) < 2:
            continue
        tasks_with_transitions += 1
        for i in range(len(seq) - 1):
            transition_counts[seq[i]][seq[i + 1]] += 1

    # Normalize to mean per-task
    transition_matrix = {}
    for from_s in STAGES:
        transition_matrix[from_s] = {}
        for to_s in STAGES:
            raw = transition_counts[from_s][to_s]
            transition_matrix[from_s][to_s] = (
                raw / tasks_with_transitions if tasks_with_transitions > 0 else 0.0
            )

    # 4) Stage usage rate per position (1st through 10th call)
    position_usage = {}
    for pos in range(MAX_POSITION):
        pos_counter = Counter()
        tasks_at_pos = 0
        for seq in stage_sequences:
            if len(seq) > pos:
                pos_counter[seq[pos]] += 1
                tasks_at_pos += 1
        position_usage[pos + 1] = {
            s: pos_counter.get(s, 0) / tasks_at_pos if tasks_at_pos > 0 else 0.0
            for s in STAGES
        }
        position_usage[pos + 1]["_n_tasks"] = tasks_at_pos

    return {
        "n_tasks": n_tasks,
        "first_stage_dist": first_stage_dist,
        "last_stage_dist": last_stage_dist,
        "transition_matrix": transition_matrix,
        "position_usage": position_usage,
    }


def write_csv(all_metrics: dict, output_path: str):
    """Write comprehensive CSV with all metrics."""
    rows = []

    for model_name, metrics in all_metrics.items():
        n = metrics["n_tasks"]

        # Section 1: First stage distribution
        for stage in STAGES:
            rows.append({
                "model": model_name,
                "metric_type": "first_stage_distribution",
                "stage": stage,
                "position": "",
                "from_stage": "",
                "to_stage": "",
                "value": f"{metrics['first_stage_dist'].get(stage, 0):.4f}",
                "n_tasks": n,
            })

        # Section 2: Last stage distribution
        for stage in STAGES:
            rows.append({
                "model": model_name,
                "metric_type": "last_stage_distribution",
                "stage": stage,
                "position": "",
                "from_stage": "",
                "to_stage": "",
                "value": f"{metrics['last_stage_dist'].get(stage, 0):.4f}",
                "n_tasks": n,
            })

        # Section 3: Transition matrix
        for from_s in STAGES:
            for to_s in STAGES:
                val = metrics["transition_matrix"].get(from_s, {}).get(to_s, 0)
                if val > 0:
                    rows.append({
                        "model": model_name,
                        "metric_type": "transition_matrix",
                        "stage": "",
                        "position": "",
                        "from_stage": from_s,
                        "to_stage": to_s,
                        "value": f"{val:.4f}",
                        "n_tasks": n,
                    })

        # Section 4: Position usage
        for pos in range(1, MAX_POSITION + 1):
            for stage in STAGES:
                val = metrics["position_usage"].get(pos, {}).get(stage, 0)
                n_at_pos = metrics["position_usage"].get(pos, {}).get("_n_tasks", 0)
                rows.append({
                    "model": model_name,
                    "metric_type": "position_stage_usage",
                    "stage": stage,
                    "position": pos,
                    "from_stage": "",
                    "to_stage": "",
                    "value": f"{val:.4f}",
                    "n_tasks": n_at_pos,
                })

    fieldnames = ["model", "metric_type", "stage", "position", "from_stage", "to_stage", "value", "n_tasks"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(all_metrics: dict):
    """Print a human-readable summary to stdout."""
    print("=" * 80)
    print("FIGURE 6 PANEL 1: Tool Call Flow Analysis (Guided/User Mode)")
    print("=" * 80)

    for model_name, metrics in all_metrics.items():
        n = metrics["n_tasks"]
        print(f"\n{'─' * 70}")
        print(f"  {model_name}  ({n} tasks with tool calls)")
        print(f"{'─' * 70}")

        # First stage distribution
        print("\n  First Tool Call Stage Distribution:")
        for stage in STAGES:
            pct = metrics["first_stage_dist"].get(stage, 0) * 100
            if pct > 0:
                bar = "#" * int(pct / 2)
                print(f"    {stage:<24s} {pct:5.1f}%  {bar}")

        # Last stage distribution
        print("\n  Last Tool Call Stage Distribution:")
        for stage in STAGES:
            pct = metrics["last_stage_dist"].get(stage, 0) * 100
            if pct > 0:
                bar = "#" * int(pct / 2)
                print(f"    {stage:<24s} {pct:5.1f}%  {bar}")

        # Top transitions
        print("\n  Top Stage Transitions (mean per task):")
        transitions = []
        for from_s in STAGES:
            for to_s in STAGES:
                val = metrics["transition_matrix"].get(from_s, {}).get(to_s, 0)
                if val > 0.01:
                    transitions.append((from_s, to_s, val))
        transitions.sort(key=lambda x: -x[2])
        for from_s, to_s, val in transitions[:8]:
            print(f"    {from_s:<24s} -> {to_s:<24s} {val:.3f}")

        # Position usage (positions 1-5)
        print("\n  Stage Usage by Position (positions 1-5):")
        print(f"    {'Pos':<5s}", end="")
        for stage in STAGES:
            abbrev = stage[:8]
            print(f"{abbrev:>12s}", end="")
        print(f"{'n_tasks':>10s}")

        for pos in range(1, 6):
            pu = metrics["position_usage"].get(pos, {})
            n_at_pos = pu.get("_n_tasks", 0)
            print(f"    {pos:<5d}", end="")
            for stage in STAGES:
                pct = pu.get(stage, 0) * 100
                print(f"{pct:11.1f}%", end="")
            print(f"{n_at_pos:10d}")

    # Cross-model comparison table
    print(f"\n{'=' * 80}")
    print("CROSS-MODEL COMPARISON: First Stage Distribution (%)")
    print(f"{'=' * 80}")
    header = f"{'Model':<24s}"
    for stage in STAGES:
        header += f"{stage[:12]:>14s}"
    header += f"{'N':>8s}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in all_metrics.items():
        row = f"{model_name:<24s}"
        for stage in STAGES:
            pct = metrics["first_stage_dist"].get(stage, 0) * 100
            row += f"{pct:13.1f}%"
        row += f"{metrics['n_tasks']:8d}"
        print(row)

    print(f"\n{'=' * 80}")
    print("CROSS-MODEL COMPARISON: Last Stage Distribution (%)")
    print(f"{'=' * 80}")
    print(header)
    print("-" * len(header))
    for model_name, metrics in all_metrics.items():
        row = f"{model_name:<24s}"
        for stage in STAGES:
            pct = metrics["last_stage_dist"].get(stage, 0) * 100
            row += f"{pct:13.1f}%"
        row += f"{metrics['n_tasks']:8d}"
        print(row)

    # Mean tool chain length
    print(f"\n{'=' * 80}")
    print("MEAN TOOL CHAIN LENGTH")
    print(f"{'=' * 80}")
    for model_name in all_metrics:
        # We need to recompute from position usage _n_tasks
        pu = all_metrics[model_name]["position_usage"]
        n_total = all_metrics[model_name]["n_tasks"]
        if n_total == 0:
            print(f"  {model_name:<24s}  N/A")
            continue
        # Estimate mean length from n_tasks at each position
        total_calls = sum(pu.get(pos, {}).get("_n_tasks", 0) for pos in range(1, MAX_POSITION + 1))
        # tasks beyond position 10 still contribute
        mean_len = total_calls / n_total if n_total > 0 else 0
        print(f"  {model_name:<24s}  {mean_len:.1f} calls/task (positions 1-{MAX_POSITION})")


def main():
    all_metrics = {}

    for model_name, search_specs in AGENT_CONFIG.items():
        print(f"Collecting data for {model_name}...", flush=True)
        task_orders = collect_results(model_name, search_specs)
        print(f"  Found {len(task_orders)} tasks with tool call data")

        stage_sequences = compute_stage_sequences(task_orders)
        metrics = compute_metrics(stage_sequences)
        all_metrics[model_name] = metrics

        # Also store raw stage sequences for potential further analysis
        all_metrics[model_name]["_stage_sequences"] = stage_sequences

    # Compute mean chain length from raw sequences
    for model_name in all_metrics:
        seqs = all_metrics[model_name].pop("_stage_sequences", [])
        if seqs:
            lengths = [len(s) for s in seqs]
            all_metrics[model_name]["mean_chain_length"] = sum(lengths) / len(lengths)
            all_metrics[model_name]["median_chain_length"] = sorted(lengths)[len(lengths) // 2]
            all_metrics[model_name]["max_chain_length"] = max(lengths)
        else:
            all_metrics[model_name]["mean_chain_length"] = 0
            all_metrics[model_name]["median_chain_length"] = 0
            all_metrics[model_name]["max_chain_length"] = 0

    output_path = os.path.join(OUTPUT_DIR, "tool_call_flow_by_model.csv")
    write_csv(all_metrics, output_path)
    print(f"\nCSV written to: {output_path}")

    print_summary(all_metrics)

    # Print chain length stats
    print(f"\n{'=' * 80}")
    print("CHAIN LENGTH STATISTICS")
    print(f"{'=' * 80}")
    for model_name, metrics in all_metrics.items():
        print(
            f"  {model_name:<24s}  "
            f"mean={metrics['mean_chain_length']:.1f}  "
            f"median={metrics['median_chain_length']}  "
            f"max={metrics['max_chain_length']}"
        )


if __name__ == "__main__":
    main()
