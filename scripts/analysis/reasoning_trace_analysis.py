#!/usr/bin/env python3
"""
Reasoning Trace Analysis: Scientific Knowledge vs Tool Knowledge Decomposition.

Analyzes agent reasoning traces (text output before tool calls) to separate:
- Scientific knowledge: Does the agent know the right pipeline steps?
- Tool knowledge: Can the agent execute those steps with the available tools?

Outputs:
  figures/fig_plan_vs_execution.pdf  - 2D scatter (plan vs execution score)
  figures/fig_knowledge_gap_bar.pdf  - Stacked bar (Case A/B/C proportions)
  figures/fig_backbone_gen_gap.pdf   - Paired plot (backbone mention vs execution)
  figures/fig_gap_heatmap.pdf        - Heatmap (model × step plan-execution gap)
  figures/fig_guided_vs_unguided.pdf - Guided vs unguided comparison
  figures/reasoning_trace_summary.csv - Raw data
  figures/reasoning_trace_report.md   - Markdown report
"""

import json
import glob
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import csv

import numpy as np

# ---------------------------------------------------------------------------
# Pipeline step definitions (4 canonical stages)
# ---------------------------------------------------------------------------

PIPELINE_STEPS = {
    "backbone_generation": {
        "description": "De novo backbone/scaffold generation",
        "plan_keywords": [
            # Direct tool mentions
            r"\brfdiffusion\b",
            r"\brf\s*diffusion\b",
            # Conceptual mentions
            r"\bbackbone\s+generation\b",
            r"\bscaffold\s+generation\b",
            r"\bgenerate\s+(?:a\s+)?(?:new\s+)?(?:de\s+novo\s+)?(?:protein\s+)?backbone\b",
            r"\bde\s+novo\s+(?:protein\s+)?(?:structure|backbone|scaffold)\b",
            r"\bhallucinate\s+(?:a\s+)?(?:new\s+)?(?:backbone|structure)\b",
            r"\bnovel\s+(?:protein\s+)?(?:backbone|scaffold|structure)\b",
            r"\bgenerate\s+(?:a\s+)?(?:novel|new)\s+(?:protein\s+)?(?:structure|fold)\b",
            r"\bbackbone\s+(?:design|sampling|diffusion)\b",
            r"\bstructure\s+generation\b",
            r"\bdesign_binder\b",  # composite tool that includes backbone gen
        ],
        "execution_tools": [
            "generate_backbone", "design_binder", "rfdiffusion",
        ],
        "execution_functions": [
            "backbone_generation",
        ],
    },
    "sequence_design": {
        "description": "Inverse folding / sequence design",
        "plan_keywords": [
            r"\bproteinmpnn\b",
            r"\bprotein\s*mpnn\b",
            r"\binverse\s+folding\b",
            r"\bsequence\s+design\b",
            r"\bdesign\s+(?:a\s+)?(?:new\s+)?(?:amino\s+acid\s+)?sequence\b",
            r"\bsequence\s+optimization\b",
            r"\bdesign\s+sequences?\b",
            r"\boptimize\s+(?:the\s+)?sequence\b",
            r"\bsequence\s+recovery\b",
            r"\bfixed\s+backbone\s+design\b",
            r"\brosetta\s*design\b",
            r"\brosetta_design\b",
        ],
        "execution_tools": [
            "design_sequence", "design_binder", "optimize_sequence",
            "proteinmpnn", "rosetta_design",
        ],
        "execution_functions": [
            "sequence_design",
        ],
    },
    "structure_prediction": {
        "description": "Structure prediction / validation",
        "plan_keywords": [
            r"\balphafold\b",
            r"\baf2\b",
            r"\besmfold\b",
            r"\besm\s*fold\b",
            r"\bboltz\b",
            r"\bstructure\s+prediction\b",
            r"\bpredict\s+(?:the\s+)?(?:3d\s+)?structure\b",
            r"\bfold\s+(?:the\s+)?(?:designed\s+)?(?:sequence|protein)\b",
            r"\bplddt\b",
            r"\bconfidence\s+(?:score|metric|check)\b",
            r"\biptm\b",
            r"\bi_ptm\b",
            r"\bpae\b",
            r"\bvalidate\s+(?:the\s+)?(?:designed\s+)?structure\b",
            r"\bcomplex\s+prediction\b",
            r"\bpredict_structure\b",
            r"\bpredict_affinity\b",
        ],
        "execution_tools": [
            "predict_structure", "predict_structure_boltz",
            "predict_affinity_boltz", "validate_design",
            "esmfold", "alphafold2", "boltz",
        ],
        "execution_functions": [
            "structure_prediction", "complex_prediction",
        ],
    },
    "scoring_validation": {
        "description": "Energy scoring / stability validation",
        "plan_keywords": [
            r"\brosetta\b",
            r"\benergy\s+(?:score|function|minimization)\b",
            r"\bstability\b",
            r"\bscore\s+(?:the\s+)?(?:designed\s+)?(?:structure|protein|complex)\b",
            r"\bvalidat(?:e|ion)\b",
            r"\brelax\b",
            r"\bminimiz(?:e|ation)\b",
            r"\bref2015\b",
            r"\brosetta_relax\b",
            r"\brosetta_score\b",
            r"\bscore_stability\b",
            r"\bddg\b",
            r"\bbinding\s+energy\b",
            r"\binterface\s+(?:energy|analysis|score)\b",
            r"\banalyze_interface\b",
        ],
        "execution_tools": [
            "score_stability", "rosetta_relax", "rosetta_score",
            "rosetta_design", "analyze_interface", "validate_design",
        ],
        "execution_functions": [
            "stability_scoring", "energy_minimization",
            "physics_validation", "interface_analysis",
        ],
    },
}

# ---------------------------------------------------------------------------
# Agent / mode canonical mapping
# ---------------------------------------------------------------------------

AGENT_CANONICAL = {
    "gpt5-tools-user": ("GPT-5", "user"),
    "gpt5-tools-benchmark": ("GPT-5", "benchmark"),
    "gpt5-tools": ("GPT-5", "unknown"),
    "sonnet-4.5-tools-user": ("Sonnet 4.5", "user"),
    "sonnet-4.5-tools-benchmark": ("Sonnet 4.5", "benchmark"),
    "deepseek-v3-tools-user": ("DeepSeek V3", "user"),
    "deepseek-v3-tools-benchmark": ("DeepSeek V3", "benchmark"),
    "gemini-2.5-pro-tools-user": ("Gemini 2.5 Pro", "user"),
    "gemini-2.5-pro-tools-benchmark": ("Gemini 2.5 Pro", "benchmark"),
    "gemini-3-pro-tools-user": ("Gemini 3 Pro", "user"),
    "gemini3-pro-tools-user": ("Gemini 3 Pro", "user"),
    "hardcoded-pipeline": ("Hardcoded", "baseline"),
    "human-expert-agent": ("Human Expert", "baseline"),
    "oracle": ("Oracle", "baseline"),
}

# Models we care about for the main analysis (exclude baselines)
MAIN_MODELS = {"GPT-5", "Sonnet 4.5", "DeepSeek V3", "Gemini 2.5 Pro"}

# Colors for models
MODEL_COLORS = {
    "GPT-5": "#1f77b4",
    "Sonnet 4.5": "#ff7f0e",
    "DeepSeek V3": "#2ca02c",
    "Gemini 2.5 Pro": "#d62728",
    "Gemini 3 Pro": "#9467bd",
    "Hardcoded": "#8c564b",
    "Human Expert": "#7f7f7f",
    "Oracle": "#bcbd22",
}

MODE_MARKERS = {
    "user": "o",
    "benchmark": "^",
    "baseline": "s",
    "unknown": "D",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraceAnalysis:
    """Analysis result for one agent × task × condition."""
    task_id: str
    agent_raw: str
    model: str
    mode: str  # user / benchmark
    reasoning_trace: str = ""
    trace_length: int = 0

    # Plan scores (from reasoning trace keyword matching)
    plan_backbone: int = 0
    plan_sequence: int = 0
    plan_structure: int = 0
    plan_scoring: int = 0

    # Execution scores (from actual tool calls / agent_functions)
    exec_backbone: int = 0
    exec_sequence: int = 0
    exec_structure: int = 0
    exec_scoring: int = 0

    # Context snippets for manual verification
    plan_backbone_context: str = ""
    plan_sequence_context: str = ""
    plan_structure_context: str = ""
    plan_scoring_context: str = ""

    # Existing scores from evaluation
    approach_score: float = 0.0
    orchestration_score: float = 0.0
    quality_score: float = 0.0
    total_score: float = 0.0
    tools_used: list = field(default_factory=list)
    iterations: int = 0

    @property
    def plan_score(self) -> int:
        return self.plan_backbone + self.plan_sequence + self.plan_structure + self.plan_scoring

    @property
    def exec_score(self) -> int:
        return self.exec_backbone + self.exec_sequence + self.exec_structure + self.exec_scoring

    @property
    def gap(self) -> int:
        return self.plan_score - self.exec_score

    @property
    def knowledge_case(self) -> str:
        """Classify into knowledge gap cases."""
        if self.plan_score >= 3 and self.exec_score >= 3:
            return "A_full_knowledge"  # Knows and executes
        elif self.plan_score > self.exec_score and self.plan_score >= 2:
            return "B_tool_gap"  # Knows but can't execute
        else:
            return "C_science_gap"  # Doesn't know


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def extract_plan_mentions(trace: str, step_name: str) -> tuple[int, str]:
    """
    Check if a pipeline step is mentioned in the reasoning trace.

    Returns (binary_score, context_snippet).
    """
    if not trace or len(trace.strip()) < 10:
        return 0, ""

    trace_lower = trace.lower()
    step_def = PIPELINE_STEPS[step_name]

    for pattern in step_def["plan_keywords"]:
        match = re.search(pattern, trace_lower)
        if match:
            # Extract context window (±100 chars)
            start = max(0, match.start() - 100)
            end = min(len(trace), match.end() + 100)
            context = trace[start:end].replace("\n", " ").strip()
            return 1, f"...{context}..."

    return 0, ""


def extract_exec_score(result: dict, step_name: str) -> int:
    """
    Check if a pipeline step was actually executed via tool calls.
    Uses both tool_call_log and approach_metrics.agent_functions.
    """
    step_def = PIPELINE_STEPS[step_name]

    # Check approach_metrics.agent_functions
    agent_functions = result.get("approach_metrics", {}).get("agent_functions", [])
    for fn in step_def["execution_functions"]:
        if fn in agent_functions:
            return 1

    # Check tool_call_log
    tool_calls = result.get("raw_output", {}).get("tool_call_log", [])
    tools_used = result.get("tools_used", [])
    actual_tool_order = result.get("orchestration_metrics", {}).get("actual_tool_order", [])

    all_tools = set(tools_used)
    for tc in tool_calls:
        t = tc.get("tool", "") if isinstance(tc, dict) else ""
        all_tools.add(t.lower())
    for t in actual_tool_order:
        all_tools.add(t.lower())

    for tool_name in step_def["execution_tools"]:
        if tool_name.lower() in all_tools:
            return 1

    return 0


def load_all_results(results_root: str) -> list[TraceAnalysis]:
    """Load all result.json files and perform trace analysis."""
    analyses = []
    seen = set()  # (agent_raw, task_id) to avoid duplicates

    # Prefer full_run directories (most complete), then fallback to individual runs
    search_paths = [
        f"{results_root}/full_run_user/runs/*/agents/*/*/result.json",
        f"{results_root}/full_run_benchmark/runs/*/agents/*/*/result.json",
        f"{results_root}/runs/*/agents/*/*/result.json",
        f"{results_root}/*/runs/*/agents/*/*/result.json",
    ]

    all_paths = []
    for pattern in search_paths:
        all_paths.extend(glob.glob(pattern))

    print(f"Found {len(all_paths)} result files")

    for path in all_paths:
        try:
            # Extract agent and task from path
            parts = path.split("/agents/")
            if len(parts) < 2:
                continue
            agent_task = parts[1].split("/")
            agent_raw = agent_task[0]
            task_id = agent_task[1] if len(agent_task) > 1 else ""

            # Skip if already seen (prefer earlier = full_run)
            key = (agent_raw, task_id)
            if key in seen:
                continue
            seen.add(key)

            # Map to canonical model/mode
            if agent_raw not in AGENT_CANONICAL:
                continue
            model, mode = AGENT_CANONICAL[agent_raw]

            with open(path) as f:
                result = json.load(f)

            trace = result.get("raw_output", {}).get("reasoning_trace", "") or ""

            a = TraceAnalysis(
                task_id=result.get("task_id", task_id),
                agent_raw=agent_raw,
                model=model,
                mode=mode,
                reasoning_trace=trace,
                trace_length=len(trace),
            )

            # Plan scores from reasoning trace
            a.plan_backbone, a.plan_backbone_context = extract_plan_mentions(trace, "backbone_generation")
            a.plan_sequence, a.plan_sequence_context = extract_plan_mentions(trace, "sequence_design")
            a.plan_structure, a.plan_structure_context = extract_plan_mentions(trace, "structure_prediction")
            a.plan_scoring, a.plan_scoring_context = extract_plan_mentions(trace, "scoring_validation")

            # Execution scores from actual tool calls
            a.exec_backbone = extract_exec_score(result, "backbone_generation")
            a.exec_sequence = extract_exec_score(result, "sequence_design")
            a.exec_structure = extract_exec_score(result, "structure_prediction")
            a.exec_scoring = extract_exec_score(result, "scoring_validation")

            # Existing evaluation scores
            a.approach_score = result.get("approach_metrics", {}).get("score", 0)
            a.orchestration_score = result.get("orchestration_metrics", {}).get("score", 0)
            a.quality_score = result.get("quality_metrics", {}).get("score", 0)
            a.total_score = result.get("partial_score", 0) or 0
            a.tools_used = result.get("tools_used", [])
            a.iterations = result.get("iterations", 0)

            analyses.append(a)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            continue

    print(f"Loaded {len(analyses)} valid analyses")
    return analyses


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_model_mode(analyses: list[TraceAnalysis]) -> dict:
    """Group and aggregate by (model, mode)."""
    groups = defaultdict(list)
    for a in analyses:
        groups[(a.model, a.mode)].append(a)

    agg = {}
    for (model, mode), items in groups.items():
        n = len(items)
        if n == 0:
            continue

        # Filter to those with non-empty traces for plan scores
        traced = [a for a in items if a.trace_length > 10]
        n_traced = len(traced)

        agg[(model, mode)] = {
            "n_total": n,
            "n_traced": n_traced,
            # Plan scores (only from traced items)
            "mean_plan_backbone": np.mean([a.plan_backbone for a in traced]) if traced else 0,
            "mean_plan_sequence": np.mean([a.plan_sequence for a in traced]) if traced else 0,
            "mean_plan_structure": np.mean([a.plan_structure for a in traced]) if traced else 0,
            "mean_plan_scoring": np.mean([a.plan_scoring for a in traced]) if traced else 0,
            "mean_plan_score": np.mean([a.plan_score for a in traced]) if traced else 0,
            # Execution scores (from all items)
            "mean_exec_backbone": np.mean([a.exec_backbone for a in items]),
            "mean_exec_sequence": np.mean([a.exec_sequence for a in items]),
            "mean_exec_structure": np.mean([a.exec_structure for a in items]),
            "mean_exec_scoring": np.mean([a.exec_scoring for a in items]),
            "mean_exec_score": np.mean([a.exec_score for a in items]),
            # Gap (only from traced)
            "mean_gap": np.mean([a.gap for a in traced]) if traced else 0,
            # Knowledge cases (only from traced)
            "case_A": sum(1 for a in traced if a.knowledge_case == "A_full_knowledge") / n_traced if n_traced else 0,
            "case_B": sum(1 for a in traced if a.knowledge_case == "B_tool_gap") / n_traced if n_traced else 0,
            "case_C": sum(1 for a in traced if a.knowledge_case == "C_science_gap") / n_traced if n_traced else 0,
            # Quality
            "mean_total_score": np.mean([a.total_score for a in items]),
            "mean_approach_score": np.mean([a.approach_score for a in items]),
        }

    return agg


def compute_per_step_gap(analyses: list[TraceAnalysis]) -> dict:
    """Compute per-step plan-execution gap for each model."""
    groups = defaultdict(list)
    for a in analyses:
        if a.trace_length > 10 and a.model in MAIN_MODELS:
            groups[a.model].append(a)

    step_gaps = {}
    for model, items in groups.items():
        step_gaps[model] = {
            "backbone": np.mean([a.plan_backbone - a.exec_backbone for a in items]),
            "sequence": np.mean([a.plan_sequence - a.exec_sequence for a in items]),
            "structure": np.mean([a.plan_structure - a.exec_structure for a in items]),
            "scoring": np.mean([a.plan_scoring - a.exec_scoring for a in items]),
        }

    return step_gaps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_figures(analyses: list[TraceAnalysis], agg: dict, outdir: Path):
    """Generate all figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    # ----- Figure 1: Plan vs Execution scatter -----
    fig, ax = plt.subplots(figsize=(8, 7))

    for a in analyses:
        if a.model not in MAIN_MODELS or a.trace_length <= 10:
            continue
        color = MODEL_COLORS.get(a.model, "#333")
        marker = MODE_MARKERS.get(a.mode, "o")
        # Add jitter for overlapping points
        jx = np.random.normal(0, 0.08)
        jy = np.random.normal(0, 0.08)
        ax.scatter(
            a.plan_score + jx, a.exec_score + jy,
            c=color, marker=marker, alpha=0.25, s=30, edgecolors="none",
        )

    # Plot aggregated means
    for (model, mode), stats in agg.items():
        if model not in MAIN_MODELS:
            continue
        color = MODEL_COLORS.get(model, "#333")
        marker = MODE_MARKERS.get(mode, "o")
        ax.scatter(
            stats["mean_plan_score"], stats["mean_exec_score"],
            c=color, marker=marker, s=200, edgecolors="black", linewidths=1.5,
            zorder=10,
        )
        # Label
        offset = (5, 5)
        ax.annotate(
            f"{model}\n({mode})",
            (stats["mean_plan_score"], stats["mean_exec_score"]),
            textcoords="offset points", xytext=offset,
            fontsize=8, ha="left",
        )

    # Diagonal line
    ax.plot([0, 4], [0, 4], "k--", alpha=0.3, label="Plan = Execution")
    ax.set_xlabel("Plan Score (scientific knowledge proxy)")
    ax.set_ylabel("Execution Score (tool knowledge proxy)")
    ax.set_title("Plan vs Execution: Scientific Knowledge × Tool Knowledge")
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.3, 4.3)

    # Legend
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m],
               markersize=10, label=m)
        for m in MAIN_MODELS if m in MODEL_COLORS
    ]
    mode_handles = [
        Line2D([0], [0], marker=MODE_MARKERS[m], color="w", markerfacecolor="gray",
               markersize=10, label=f"{m} mode")
        for m in ["user", "benchmark"]
    ]
    ax.legend(handles=model_handles + mode_handles, loc="upper left", fontsize=9)

    # Zone annotations
    ax.text(0.5, 3.5, "Tool gap\n(knows but\ncan't execute)", fontsize=9, ha="center",
            style="italic", color="#666", bbox=dict(boxstyle="round", fc="#fff8f0", alpha=0.7))
    ax.text(3.5, 0.5, "Execution\nwithout\nunderstanding", fontsize=9, ha="center",
            style="italic", color="#666", bbox=dict(boxstyle="round", fc="#f0f8ff", alpha=0.7))
    ax.text(0.5, 0.5, "No\nknowledge", fontsize=9, ha="center",
            style="italic", color="#999")
    ax.text(3.5, 3.5, "Full\nknowledge", fontsize=9, ha="center",
            style="italic", color="#060")

    fig.savefig(outdir / "fig_plan_vs_execution.pdf")
    fig.savefig(outdir / "fig_plan_vs_execution.png")
    plt.close(fig)
    print("  -> fig_plan_vs_execution.pdf")

    # ----- Figure 2: Knowledge gap stacked bar -----
    fig, ax = plt.subplots(figsize=(10, 5))

    model_mode_keys = sorted(
        [(m, mo) for (m, mo) in agg if m in MAIN_MODELS],
        key=lambda x: (x[0], x[1]),
    )
    labels = [f"{m}\n({mo})" for m, mo in model_mode_keys]
    case_a = [agg[(m, mo)]["case_A"] * 100 for m, mo in model_mode_keys]
    case_b = [agg[(m, mo)]["case_B"] * 100 for m, mo in model_mode_keys]
    case_c = [agg[(m, mo)]["case_C"] * 100 for m, mo in model_mode_keys]

    x = np.arange(len(labels))
    w = 0.6
    ax.bar(x, case_a, w, label="A: Full knowledge", color="#2ecc71")
    ax.bar(x, case_b, w, bottom=case_a, label="B: Tool gap", color="#e67e22")
    ax.bar(x, case_c, w, bottom=[a + b for a, b in zip(case_a, case_b)],
           label="C: Science gap", color="#e74c3c")

    ax.set_ylabel("Proportion (%)")
    ax.set_title("Knowledge Gap Profile by Model × Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    fig.savefig(outdir / "fig_knowledge_gap_bar.pdf")
    fig.savefig(outdir / "fig_knowledge_gap_bar.png")
    plt.close(fig)
    print("  -> fig_knowledge_gap_bar.pdf")

    # ----- Figure 3: Backbone generation gap (de novo tasks only) -----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter to de novo tasks only (binder, scaffold, ppi, peptide, dnb_, dnk_, cfd_, cpx_)
    de_novo_prefixes = ("binder_", "scaffold_", "ppi_", "peptide_", "dnb_", "dnk_", "cfd_", "cpx_")
    dn_analyses = [a for a in analyses if a.task_id.startswith(de_novo_prefixes) and a.trace_length > 10]

    dn_groups = defaultdict(list)
    for a in dn_analyses:
        if a.model in MAIN_MODELS:
            dn_groups[(a.model, a.mode)].append(a)

    for (model, mode), items in sorted(dn_groups.items()):
        mention_rate = np.mean([a.plan_backbone for a in items])
        exec_rate = np.mean([a.exec_backbone for a in items])
        color = MODEL_COLORS.get(model, "#333")
        marker = MODE_MARKERS.get(mode, "o")
        ax.scatter(mention_rate, exec_rate, c=color, marker=marker, s=150,
                   edgecolors="black", linewidths=1.5, zorder=10)
        ax.annotate(f"{model}\n({mode})", (mention_rate, exec_rate),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Backbone generation mention rate (plan)")
    ax.set_ylabel("Backbone generation execution rate")
    ax.set_title("Backbone Generation: Plan vs Execution\n(De novo tasks only)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.savefig(outdir / "fig_backbone_gen_gap.pdf")
    fig.savefig(outdir / "fig_backbone_gen_gap.png")
    plt.close(fig)
    print("  -> fig_backbone_gen_gap.pdf")

    # ----- Figure 4: Heatmap (model × step gap) -----
    step_gaps = compute_per_step_gap(analyses)
    if step_gaps:
        models = sorted(step_gaps.keys())
        steps = ["backbone", "sequence", "structure", "scoring"]
        matrix = np.array([[step_gaps[m][s] for s in steps] for m in models])

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-0.5, vmax=0.5, aspect="auto")

        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([s.replace("_", "\n") for s in steps], fontsize=10)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(steps)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10, color="black" if abs(val) < 0.3 else "white")

        ax.set_title("Plan − Execution Gap by Model × Pipeline Step\n(positive = tool gap, negative = exec without plan)")
        fig.colorbar(im, ax=ax, label="Mean gap (plan − exec)")

        fig.savefig(outdir / "fig_gap_heatmap.pdf")
        fig.savefig(outdir / "fig_gap_heatmap.png")
        plt.close(fig)
        print("  -> fig_gap_heatmap.pdf")

    # ----- Figure 5: Guided (user) vs Unguided (benchmark) comparison -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Plan score comparison
    ax = axes[0]
    for model in MAIN_MODELS:
        user_key = (model, "user")
        bench_key = (model, "benchmark")
        if user_key in agg and bench_key in agg:
            color = MODEL_COLORS.get(model, "#333")
            ax.plot(
                [0, 1],
                [agg[bench_key]["mean_plan_score"], agg[user_key]["mean_plan_score"]],
                "o-", color=color, markersize=10, linewidth=2,
                label=model,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Benchmark\n(unguided)", "User\n(guided)"])
    ax.set_ylabel("Mean Plan Score (0-4)")
    ax.set_title("Scientific Knowledge\n(Plan Score)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.2)

    # Right: Execution score comparison
    ax = axes[1]
    for model in MAIN_MODELS:
        user_key = (model, "user")
        bench_key = (model, "benchmark")
        if user_key in agg and bench_key in agg:
            color = MODEL_COLORS.get(model, "#333")
            ax.plot(
                [0, 1],
                [agg[bench_key]["mean_exec_score"], agg[user_key]["mean_exec_score"]],
                "o-", color=color, markersize=10, linewidth=2,
                label=model,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Benchmark\n(unguided)", "User\n(guided)"])
    ax.set_ylabel("Mean Execution Score (0-4)")
    ax.set_title("Tool Knowledge\n(Execution Score)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.2)

    fig.suptitle("Guided vs Unguided: Does User Mode Boost Scientific or Tool Knowledge?", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig_guided_vs_unguided.pdf")
    fig.savefig(outdir / "fig_guided_vs_unguided.png")
    plt.close(fig)
    print("  -> fig_guided_vs_unguided.pdf")

    # ----- Figure 6: Per-step plan & execution rates by model -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]
    step_labels = ["Backbone Generation", "Sequence Design", "Structure Prediction", "Scoring / Validation"]

    for idx, (step, label) in enumerate(zip(steps, step_labels)):
        ax = axes[idx // 2][idx % 2]
        plan_key = f"mean_plan_{step.split('_')[0]}"
        exec_key = f"mean_exec_{step.split('_')[0]}"

        model_mode_list = sorted(
            [(m, mo) for (m, mo) in agg if m in MAIN_MODELS],
            key=lambda x: (x[0], x[1])
        )
        x_labels = [f"{m}\n({mo})" for m, mo in model_mode_list]
        plan_vals = [agg[(m, mo)][plan_key] * 100 for m, mo in model_mode_list]
        exec_vals = [agg[(m, mo)][exec_key] * 100 for m, mo in model_mode_list]

        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w / 2, plan_vals, w, label="Plan (mentioned)", color="#3498db", alpha=0.8)
        ax.bar(x + w / 2, exec_vals, w, label="Executed", color="#e67e22", alpha=0.8)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=0)
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Per-Step Plan vs Execution Rates by Model × Mode", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig_step_plan_exec_rates.pdf")
    fig.savefig(outdir / "fig_step_plan_exec_rates.png")
    plt.close(fig)
    print("  -> fig_step_plan_exec_rates.pdf")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_csv(analyses: list[TraceAnalysis], outdir: Path):
    """Write raw data CSV."""
    csv_path = outdir / "reasoning_trace_summary.csv"
    fields = [
        "task_id", "model", "mode", "trace_length",
        "plan_backbone", "plan_sequence", "plan_structure", "plan_scoring", "plan_score",
        "exec_backbone", "exec_sequence", "exec_structure", "exec_scoring", "exec_score",
        "gap", "knowledge_case",
        "approach_score", "orchestration_score", "quality_score", "total_score",
        "iterations",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in analyses:
            row = {
                "task_id": a.task_id, "model": a.model, "mode": a.mode,
                "trace_length": a.trace_length,
                "plan_backbone": a.plan_backbone, "plan_sequence": a.plan_sequence,
                "plan_structure": a.plan_structure, "plan_scoring": a.plan_scoring,
                "plan_score": a.plan_score,
                "exec_backbone": a.exec_backbone, "exec_sequence": a.exec_sequence,
                "exec_structure": a.exec_structure, "exec_scoring": a.exec_scoring,
                "exec_score": a.exec_score,
                "gap": a.gap, "knowledge_case": a.knowledge_case,
                "approach_score": a.approach_score,
                "orchestration_score": a.orchestration_score,
                "quality_score": a.quality_score, "total_score": a.total_score,
                "iterations": a.iterations,
            }
            writer.writerow(row)

    print(f"  -> {csv_path.name} ({len(analyses)} rows)")
    return csv_path


def generate_context_csv(analyses: list[TraceAnalysis], outdir: Path):
    """Write context snippets for manual verification."""
    csv_path = outdir / "reasoning_trace_contexts.csv"
    fields = [
        "task_id", "model", "mode",
        "plan_backbone", "plan_backbone_context",
        "plan_sequence", "plan_sequence_context",
        "plan_structure", "plan_structure_context",
        "plan_scoring", "plan_scoring_context",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in analyses:
            if a.trace_length <= 10:
                continue
            writer.writerow({
                "task_id": a.task_id, "model": a.model, "mode": a.mode,
                "plan_backbone": a.plan_backbone,
                "plan_backbone_context": a.plan_backbone_context[:300],
                "plan_sequence": a.plan_sequence,
                "plan_sequence_context": a.plan_sequence_context[:300],
                "plan_structure": a.plan_structure,
                "plan_structure_context": a.plan_structure_context[:300],
                "plan_scoring": a.plan_scoring,
                "plan_scoring_context": a.plan_scoring_context[:300],
            })

    print(f"  -> {csv_path.name}")
    return csv_path


def generate_report(analyses: list[TraceAnalysis], agg: dict, outdir: Path):
    """Generate markdown summary report."""
    report_path = outdir / "reasoning_trace_report.md"

    lines = [
        "# Reasoning Trace Analysis: Scientific Knowledge vs Tool Knowledge",
        "",
        "## Summary",
        "",
        f"- **Total analyses**: {len(analyses)}",
        f"- **With non-empty traces**: {sum(1 for a in analyses if a.trace_length > 10)}",
        f"- **Models analyzed**: {len(set(a.model for a in analyses))}",
        f"- **Unique tasks**: {len(set(a.task_id for a in analyses))}",
        "",
        "## Methodology",
        "",
        "We analyze agent reasoning traces (text output before/during tool calls) to separate:",
        "- **Plan Score** (0-4): How many pipeline steps the agent *mentioned* in reasoning",
        "  (backbone generation, sequence design, structure prediction, scoring/validation)",
        "- **Execution Score** (0-4): How many pipeline steps the agent *actually executed*",
        "  via tool calls",
        "- **Gap** = Plan − Execution: positive means \"knew but couldn't execute\"",
        "",
        "### Knowledge Gap Cases",
        "- **Case A (Full Knowledge)**: Plan ≥ 3 AND Execution ≥ 3",
        "- **Case B (Tool Gap)**: Plan > Execution AND Plan ≥ 2",
        "- **Case C (Science Gap)**: Low plan score (doesn't know the right steps)",
        "",
        "## Model × Mode Results",
        "",
        "| Model | Mode | N | N Traced | Plan | Exec | Gap | Case A | Case B | Case C |",
        "|-------|------|---|----------|------|------|-----|--------|--------|--------|",
    ]

    for (model, mode) in sorted(agg.keys()):
        s = agg[(model, mode)]
        lines.append(
            f"| {model} | {mode} | {s['n_total']} | {s['n_traced']} | "
            f"{s['mean_plan_score']:.2f} | {s['mean_exec_score']:.2f} | "
            f"{s['mean_gap']:+.2f} | {s['case_A']*100:.0f}% | "
            f"{s['case_B']*100:.0f}% | {s['case_C']*100:.0f}% |"
        )

    lines.extend([
        "",
        "## Per-Step Gap Analysis (Main Models Only)",
        "",
        "| Model | Backbone Gap | Sequence Gap | Structure Gap | Scoring Gap |",
        "|-------|-------------|--------------|---------------|-------------|",
    ])

    step_gaps = compute_per_step_gap(analyses)
    for model in sorted(step_gaps.keys()):
        g = step_gaps[model]
        lines.append(
            f"| {model} | {g['backbone']:+.2f} | {g['sequence']:+.2f} | "
            f"{g['structure']:+.2f} | {g['scoring']:+.2f} |"
        )

    # Key findings
    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Find which models have biggest tool gap
    tool_gap_models = [(m, mo, s["mean_gap"]) for (m, mo), s in agg.items()
                       if m in MAIN_MODELS and s["n_traced"] > 5]
    tool_gap_models.sort(key=lambda x: -x[2])

    if tool_gap_models:
        lines.append("### Largest Tool Gaps (knew but couldn't execute)")
        for m, mo, gap in tool_gap_models[:5]:
            lines.append(f"- **{m} ({mo})**: gap = {gap:+.2f}")
        lines.append("")

    # Models with smallest gap (aligned)
    aligned_models = [(m, mo, abs(s["mean_gap"]), s["mean_exec_score"])
                      for (m, mo), s in agg.items()
                      if m in MAIN_MODELS and s["n_traced"] > 5]
    aligned_models.sort(key=lambda x: x[2])

    if aligned_models:
        lines.append("### Most Aligned (plan ≈ execution)")
        for m, mo, gap, exec_s in aligned_models[:3]:
            lines.append(f"- **{m} ({mo})**: |gap| = {gap:.2f}, exec = {exec_s:.2f}")
        lines.append("")

    # Guided vs unguided delta
    lines.append("### Guided (User) vs Unguided (Benchmark) Delta")
    for model in sorted(MAIN_MODELS):
        user_key = (model, "user")
        bench_key = (model, "benchmark")
        if user_key in agg and bench_key in agg:
            plan_delta = agg[user_key]["mean_plan_score"] - agg[bench_key]["mean_plan_score"]
            exec_delta = agg[user_key]["mean_exec_score"] - agg[bench_key]["mean_exec_score"]
            lines.append(
                f"- **{model}**: Plan Δ = {plan_delta:+.2f}, Exec Δ = {exec_delta:+.2f}"
            )
            if plan_delta > 0.1 and exec_delta > 0.1:
                lines.append(f"  → Guided mode boosts *both* scientific and tool knowledge")
            elif exec_delta > plan_delta + 0.1:
                lines.append(f"  → Guided mode primarily boosts *tool knowledge*")
            elif plan_delta > exec_delta + 0.1:
                lines.append(f"  → Guided mode primarily boosts *scientific knowledge* (agent learns workflow)")
    lines.append("")

    # Backbone generation analysis for de novo tasks
    de_novo_prefixes = ("binder_", "scaffold_", "ppi_", "peptide_", "dnb_", "dnk_", "cfd_", "cpx_")
    dn_traced = [a for a in analyses if a.task_id.startswith(de_novo_prefixes)
                 and a.trace_length > 10 and a.model in MAIN_MODELS]

    if dn_traced:
        lines.append("### Backbone Generation in De Novo Tasks")
        dn_by_model = defaultdict(list)
        for a in dn_traced:
            dn_by_model[(a.model, a.mode)].append(a)

        lines.append("")
        lines.append("| Model | Mode | N | Mention Rate | Exec Rate | Gap |")
        lines.append("|-------|------|---|-------------|-----------|-----|")
        for (model, mode) in sorted(dn_by_model.keys()):
            items = dn_by_model[(model, mode)]
            mention_rate = np.mean([a.plan_backbone for a in items])
            exec_rate = np.mean([a.exec_backbone for a in items])
            lines.append(
                f"| {model} | {mode} | {len(items)} | "
                f"{mention_rate:.2f} | {exec_rate:.2f} | "
                f"{mention_rate - exec_rate:+.2f} |"
            )
        lines.append("")

    # Gemini special analysis
    gemini_analyses = [a for a in analyses if "Gemini" in a.model and a.trace_length > 10]
    if gemini_analyses:
        lines.append("### Gemini Special Analysis (MCP Tool Limitations)")
        lines.append("")
        lines.append("Gemini cannot use MCP tools directly and falls back to `execute_python`.")
        lines.append("This makes the plan-execution gap particularly interesting:")
        lines.append("")
        for model in ["Gemini 2.5 Pro", "Gemini 3 Pro"]:
            items = [a for a in gemini_analyses if a.model == model]
            if items:
                mean_plan = np.mean([a.plan_score for a in items])
                mean_exec = np.mean([a.exec_score for a in items])
                lines.append(
                    f"- **{model}**: Plan = {mean_plan:.2f}, Exec = {mean_exec:.2f}, "
                    f"Gap = {mean_plan - mean_exec:+.2f}"
                )
                # Check if execute_python is dominant
                py_users = sum(1 for a in items if "execute_python" in (a.tools_used or []))
                lines.append(
                    f"  → {py_users}/{len(items)} tasks used execute_python"
                )
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  -> {report_path.name}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / "results"
    outdir = project_root / "figures"
    outdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Reasoning Trace Analysis: Scientific vs Tool Knowledge")
    print("=" * 60)
    print()

    # 1. Load all results
    print("[1/4] Loading results...")
    analyses = load_all_results(str(results_root))

    # Print summary
    models = defaultdict(int)
    for a in analyses:
        models[(a.model, a.mode)] += 1
    print("\nData summary:")
    for (m, mo), n in sorted(models.items()):
        traced = sum(1 for a in analyses if a.model == m and a.mode == mo and a.trace_length > 10)
        print(f"  {m:20s} ({mo:10s}): {n:4d} results, {traced:4d} with traces")

    # 2. Aggregate
    print("\n[2/4] Aggregating...")
    agg = aggregate_by_model_mode(analyses)

    # 3. Generate CSV
    print("\n[3/4] Generating outputs...")
    generate_csv(analyses, outdir)
    generate_context_csv(analyses, outdir)
    generate_report(analyses, agg, outdir)

    # 4. Generate figures
    print("\n[4/4] Generating figures...")
    create_figures(analyses, agg, outdir)

    print("\n" + "=" * 60)
    print("Done! All outputs in:", outdir)
    print("=" * 60)


if __name__ == "__main__":
    main()
