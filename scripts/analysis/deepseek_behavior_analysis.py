#!/usr/bin/env python3
"""DeepSeek V3 Behavioral Analysis — Why Does It Consistently Outperform?

Analyzes tool call logs to identify observable behavioral differences
between DeepSeek V3 and other LLM agents in BioDesignBench.

Outputs:
    figures/fig_ds_first_tool.pdf          — First tool call distribution
    figures/fig_ds_tool_flow.pdf           — Pipeline stage flow by model
    figures/fig_ds_retry_behavior.pdf      — Retry and iteration patterns
    figures/fig_ds_difficulty_stages.pdf   — Difficulty × stage completion
    figures/fig_ds_error_recovery.pdf      — Error recovery strategies
    results/analysis/deepseek_behavior.csv
    results/analysis/deepseek_behavior_report.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from biodesignbench.taxonomy import get_category
from scripts.analysis.load_results import CONDITION_MAP, load_all
from scripts.analysis.reviewer_defense import MINIMUM_VIABLE_STEPS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = PROJECT_ROOT / "figures"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"

# Models to compare (ordered by overall score)
MODELS = [
    ("DeepSeek V3 user", "DS-V3 user", "#1f77b4"),
    ("DeepSeek V3 benchmark", "DS-V3 bench", "#aec7e8"),
    ("GPT-5 user", "GPT-5 user", "#ff7f0e"),
    ("GPT-5 benchmark", "GPT-5 bench", "#ffbb78"),
    ("Sonnet 4.5 user", "Son-4.5 user", "#2ca02c"),
    ("Sonnet 4.5 benchmark", "Son-4.5 bench", "#98df8a"),
    ("Gemini 2.5 Pro user", "Gem-2.5 user", "#d62728"),
    ("Gemini 2.5 Pro benchmark", "Gem-2.5 bench", "#ff9896"),
    ("Hardcoded Pipeline", "Hardcoded", "#9467bd"),
]

# Canonical pipeline stage mapping: tool → stage
TOOL_TO_STAGE = {
    # Backbone generation
    "generate_backbone": "backbone_gen",
    "design_binder": "backbone_gen",   # composite: includes RFdiffusion
    "rfdiffusion": "backbone_gen",
    "chroma": "backbone_gen",
    # Sequence design
    "optimize_sequence": "seq_design",
    "rosetta_design": "seq_design",    # composite
    "proteinmpnn": "seq_design",
    "ligandmpnn": "seq_design",
    "esm_if": "seq_design",
    "mpnn": "seq_design",
    # Structure prediction
    "predict_structure": "struct_pred",
    "predict_complex": "struct_pred",
    "predict_structure_boltz": "struct_pred",
    "predict_affinity_boltz": "struct_pred",
    "validate_design": "struct_pred",
    "esmfold": "struct_pred",
    "alphafold2": "struct_pred",
    "colabfold": "struct_pred",
    # Scoring/Analysis
    "score_stability": "scoring",
    "rosetta_score": "scoring",
    "rosetta_relax": "scoring",
    "analyze_interface": "scoring",
    "suggest_hotspots": "scoring",
    "energy_minimize": "scoring",
    "esm2": "scoring",
    "pyrosetta": "scoring",
    # Utility
    "execute_python": "code_exec",
    "read_file": "file_io",
    "write_file": "file_io",
}

STAGE_ORDER = ["backbone_gen", "seq_design", "struct_pred", "scoring", "code_exec", "file_io"]
STAGE_LABELS = {
    "backbone_gen": "Backbone\nGeneration",
    "seq_design": "Sequence\nDesign",
    "struct_pred": "Structure\nPrediction",
    "scoring": "Scoring/\nAnalysis",
    "code_exec": "Code\nExecution",
    "file_io": "File I/O",
}


# ═════════════════════════════════════════════════════════════════════════
# Data extraction
# ═════════════════════════════════════════════════════════════════════════


def extract_all_tool_logs() -> pd.DataFrame:
    """Extract tool call logs from all result.json files."""
    rows = []
    task_meta = {}

    for cond_name, info in CONDITION_MAP.items():
        agent_dir = Path(info["path"])
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
            log = result.get("raw_output", {}).get("tool_call_log", [])
            tools_used = result.get("tools_used", [])
            tool_order = result.get("orchestration_metrics", {}).get(
                "actual_tool_order", []
            )
            score = result.get("partial_score", 0)
            iterations = result.get("iterations", 0)

            # Task metadata
            cat = get_category(tid)
            approach = cat.approach.value if cat else "unknown"
            subject = cat.subject.value if cat else "unknown"

            # Extract per-call details
            n_failures = sum(1 for tc in log if not tc.get("success", True))
            n_retries = _count_retries(log)

            # First MCP tool (skip execute_python/read_file)
            first_mcp = _first_mcp_tool(log)
            first_mcp_stage = TOOL_TO_STAGE.get(first_mcp, "other") if first_mcp else "none"

            # Stage coverage
            stages_hit = set()
            for tool in tools_used:
                stage = TOOL_TO_STAGE.get(tool)
                if stage and stage not in ("code_exec", "file_io"):
                    stages_hit.add(stage)

            # Tool order stages
            order_stages = []
            for t in tool_order:
                stage = TOOL_TO_STAGE.get(t)
                if stage:
                    order_stages.append(stage)

            rows.append({
                "task_id": tid,
                "condition": cond_name,
                "llm": info["llm"],
                "mode": info["mode"],
                "score": score,
                "iterations": iterations,
                "n_tool_calls": len(log),
                "n_unique_tools": len(set(tools_used)),
                "n_failures": n_failures,
                "n_retries": n_retries,
                "failure_rate": n_failures / len(log) if log else 0,
                "first_mcp_tool": first_mcp or "none",
                "first_mcp_stage": first_mcp_stage,
                "tools_used": tools_used,
                "tool_order": tool_order,
                "has_backbone_gen": "backbone_gen" in stages_hit,
                "has_seq_design": "seq_design" in stages_hit,
                "has_struct_pred": "struct_pred" in stages_hit,
                "has_scoring": "scoring" in stages_hit,
                "n_design_stages": len(stages_hit),
                "design_approach": approach,
                "molecular_subject": subject,
                "difficulty": _get_difficulty(tid),
                "order_stages": order_stages,
                "error_recovery": _classify_error_recovery(log),
            })

    return pd.DataFrame(rows)


def _first_mcp_tool(log: list) -> str | None:
    """Find the first MCP tool (not execute_python/read_file)."""
    for tc in log:
        tool = tc.get("tool", "")
        if tool not in ("execute_python", "read_file", "write_file"):
            return tool
    return None


def _count_retries(log: list) -> int:
    """Count consecutive calls to the same tool (retries)."""
    retries = 0
    for i in range(1, len(log)):
        if log[i]["tool"] == log[i-1]["tool"] and not log[i-1].get("success", True):
            retries += 1
    return retries


def _classify_error_recovery(log: list) -> str:
    """Classify primary error recovery strategy."""
    if not log:
        return "no_calls"

    failures = [(i, tc) for i, tc in enumerate(log) if not tc.get("success", True)]
    if not failures:
        return "no_errors"

    strategies = Counter()
    for idx, tc in failures:
        if idx + 1 < len(log):
            next_tool = log[idx + 1]["tool"]
            if next_tool == tc["tool"]:
                strategies["retry_same"] += 1
            else:
                strategies["switch_tool"] += 1
        else:
            strategies["give_up"] += 1

    if not strategies:
        return "no_errors"
    return strategies.most_common(1)[0][0]


def _get_difficulty(task_id: str) -> str:
    """Get task difficulty from task JSON."""
    task_file = PROJECT_ROOT / "tasks" / "tier2" / f"{task_id}.json"
    if task_file.exists():
        with open(task_file) as f:
            t = json.load(f)
        return t.get("difficulty", t.get("metadata", {}).get("difficulty", "unknown"))
    return "unknown"


# ═════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═════════════════════════════════════════════════════════════════════════


def analyze_first_tool(df: pd.DataFrame) -> pd.DataFrame:
    """First MCP tool call distribution by model."""
    # De novo tasks only
    de_novo = df[df["design_approach"] == "de_novo"]

    rows = []
    for cond, short, _ in MODELS:
        cdf = de_novo[de_novo["condition"] == cond]
        if len(cdf) == 0:
            continue
        stage_counts = cdf["first_mcp_stage"].value_counts()
        total = len(cdf)
        for stage in STAGE_ORDER + ["none", "other"]:
            rows.append({
                "condition": cond,
                "short": short,
                "stage": stage,
                "count": stage_counts.get(stage, 0),
                "pct": stage_counts.get(stage, 0) / total * 100,
            })
    return pd.DataFrame(rows)


def analyze_pipeline_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline stage completion rates by model."""
    rows = []
    for cond, short, _ in MODELS:
        cdf = df[df["condition"] == cond]
        if len(cdf) == 0:
            continue

        # De novo only for backbone
        de_novo = cdf[cdf["design_approach"] == "de_novo"]

        rows.append({
            "condition": cond,
            "short": short,
            "backbone_gen_rate": de_novo["has_backbone_gen"].mean() * 100 if len(de_novo) > 0 else 0,
            "seq_design_rate": cdf["has_seq_design"].mean() * 100,
            "struct_pred_rate": cdf["has_struct_pred"].mean() * 100,
            "scoring_rate": cdf["has_scoring"].mean() * 100,
            "mean_tool_calls": cdf["n_tool_calls"].mean(),
            "mean_unique_tools": cdf["n_unique_tools"].mean(),
            "mean_iterations": cdf["iterations"].mean(),
            "mean_stages": cdf["n_design_stages"].mean(),
        })
    return pd.DataFrame(rows)


def analyze_difficulty_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline stage completion by difficulty × model."""
    rows = []
    for cond, short, _ in MODELS:
        for diff in ["easy", "medium", "hard"]:
            cdf = df[(df["condition"] == cond) & (df["difficulty"] == diff)]
            if len(cdf) == 0:
                continue
            de_novo = cdf[cdf["design_approach"] == "de_novo"]
            rows.append({
                "condition": cond,
                "short": short,
                "difficulty": diff,
                "n": len(cdf),
                "backbone_gen_rate": de_novo["has_backbone_gen"].mean() * 100 if len(de_novo) > 0 else 0,
                "seq_design_rate": cdf["has_seq_design"].mean() * 100,
                "struct_pred_rate": cdf["has_struct_pred"].mean() * 100,
                "scoring_rate": cdf["has_scoring"].mean() * 100,
                "mean_score": cdf["score"].mean(),
                "mean_tool_calls": cdf["n_tool_calls"].mean(),
            })
    return pd.DataFrame(rows)


def analyze_error_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Error recovery strategy distribution by model."""
    rows = []
    for cond, short, _ in MODELS:
        cdf = df[df["condition"] == cond]
        if len(cdf) == 0:
            continue
        strats = cdf["error_recovery"].value_counts()
        total = len(cdf)
        for strat in ["no_errors", "retry_same", "switch_tool", "give_up", "no_calls"]:
            rows.append({
                "condition": cond,
                "short": short,
                "strategy": strat,
                "count": strats.get(strat, 0),
                "pct": strats.get(strat, 0) / total * 100,
            })
    return pd.DataFrame(rows)


def analyze_benchmark_discovery(df: pd.DataFrame) -> pd.DataFrame:
    """Backbone generation discovery rate in benchmark mode (de novo tasks)."""
    de_novo = df[df["design_approach"] == "de_novo"]
    rows = []
    for cond, short, _ in MODELS:
        cdf = de_novo[de_novo["condition"] == cond]
        if len(cdf) == 0:
            continue
        rows.append({
            "condition": cond,
            "short": short,
            "mode": cdf["mode"].iloc[0],
            "n_de_novo": len(cdf),
            "backbone_gen_rate": cdf["has_backbone_gen"].mean() * 100,
            "seq_design_rate": cdf["has_seq_design"].mean() * 100,
            "struct_pred_rate": cdf["has_struct_pred"].mean() * 100,
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════


def plot_first_tool(first_tool_df: pd.DataFrame):
    """Stacked bar: first MCP tool stage by model (de novo tasks)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models_order = [s for _, s, _ in MODELS]
    stages_to_plot = ["backbone_gen", "seq_design", "struct_pred", "scoring", "none"]
    stage_colors = {
        "backbone_gen": "#2ca02c",
        "seq_design": "#1f77b4",
        "struct_pred": "#ff7f0e",
        "scoring": "#9467bd",
        "none": "#d9d9d9",
    }
    stage_names = {
        "backbone_gen": "Backbone Gen",
        "seq_design": "Sequence Design",
        "struct_pred": "Structure Pred",
        "scoring": "Scoring",
        "none": "No MCP Tool",
    }

    pivot = first_tool_df.pivot_table(
        index="short", columns="stage", values="pct", fill_value=0
    )
    # Reorder
    pivot = pivot.reindex(models_order)
    pivot = pivot.reindex(columns=stages_to_plot, fill_value=0)

    bottom = np.zeros(len(pivot))
    for stage in stages_to_plot:
        if stage in pivot.columns:
            vals = pivot[stage].values
            ax.bar(range(len(pivot)), vals, bottom=bottom, label=stage_names[stage],
                  color=stage_colors.get(stage, "#999"), alpha=0.8)
            bottom += vals

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of De Novo Tasks")
    ax.set_title("First MCP Tool Call Stage (De Novo Tasks Only)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ds_first_tool.pdf")
    fig.savefig(FIGURES_DIR / "fig_ds_first_tool.png")
    plt.close(fig)
    print(f"Saved: fig_ds_first_tool.pdf")


def plot_pipeline_flow(flow_df: pd.DataFrame):
    """Grouped bar: pipeline stage completion rates by model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Stage completion rates
    ax = axes[0]
    models = flow_df["short"].values
    x = np.arange(len(models))
    width = 0.2

    stages = [
        ("backbone_gen_rate", "Backbone Gen", "#2ca02c"),
        ("seq_design_rate", "Sequence Design", "#1f77b4"),
        ("struct_pred_rate", "Structure Pred", "#ff7f0e"),
        ("scoring_rate", "Scoring", "#9467bd"),
    ]

    for i, (col, label, color) in enumerate(stages):
        ax.bar(x + i * width - 1.5 * width, flow_df[col].values,
              width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% Tasks Using Stage")
    ax.set_title("Pipeline Stage Usage")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Tool diversity & iteration metrics
    ax2 = axes[1]
    x2 = np.arange(len(models))
    width2 = 0.25

    ax2.bar(x2 - width2, flow_df["mean_tool_calls"].values, width2,
           label="Avg Tool Calls", color="#1f77b4", alpha=0.8)
    ax2.bar(x2, flow_df["mean_unique_tools"].values, width2,
           label="Avg Unique Tools", color="#ff7f0e", alpha=0.8)
    ax2.bar(x2 + width2, flow_df["mean_stages"].values, width2,
           label="Avg Design Stages", color="#2ca02c", alpha=0.8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Count")
    ax2.set_title("Tool Usage Intensity")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Pipeline Behavior Comparison Across Models",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ds_tool_flow.pdf")
    fig.savefig(FIGURES_DIR / "fig_ds_tool_flow.png")
    plt.close(fig)
    print(f"Saved: fig_ds_tool_flow.pdf")


def plot_difficulty_stages(diff_df: pd.DataFrame):
    """Line plot: difficulty × score + stage completion for key models."""
    key_models = ["DS-V3 user", "GPT-5 user", "Son-4.5 user", "Gem-2.5 user"]
    key_colors = {"DS-V3 user": "#1f77b4", "GPT-5 user": "#ff7f0e",
                  "Son-4.5 user": "#2ca02c", "Gem-2.5 user": "#d62728"}
    diffs = ["easy", "medium", "hard"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Score by difficulty
    ax = axes[0]
    for model in key_models:
        mdf = diff_df[diff_df["short"] == model]
        if len(mdf) == 0:
            continue
        mdf = mdf.set_index("difficulty").reindex(diffs)
        ax.plot(diffs, mdf["mean_score"].values, "o-",
               color=key_colors[model], label=model, linewidth=2, markersize=6)
    ax.set_ylabel("Mean Total Score")
    ax.set_title("Score by Difficulty")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Backbone gen rate by difficulty (de novo)
    ax = axes[1]
    for model in key_models:
        mdf = diff_df[diff_df["short"] == model]
        if len(mdf) == 0:
            continue
        mdf = mdf.set_index("difficulty").reindex(diffs)
        ax.plot(diffs, mdf["backbone_gen_rate"].values, "o-",
               color=key_colors[model], label=model, linewidth=2, markersize=6)
    ax.set_ylabel("Backbone Gen Rate (%)")
    ax.set_title("Backbone Generation by Difficulty\n(De Novo Tasks)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    # Panel 3: Tool calls by difficulty
    ax = axes[2]
    for model in key_models:
        mdf = diff_df[diff_df["short"] == model]
        if len(mdf) == 0:
            continue
        mdf = mdf.set_index("difficulty").reindex(diffs)
        ax.plot(diffs, mdf["mean_tool_calls"].values, "o-",
               color=key_colors[model], label=model, linewidth=2, markersize=6)
    ax.set_ylabel("Mean Tool Calls per Task")
    ax.set_title("Tool Call Volume by Difficulty")
    ax.grid(alpha=0.3)

    fig.suptitle("Difficulty-Invariance Analysis: Why DeepSeek Doesn't Drop on Hard Tasks",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ds_difficulty_stages.pdf")
    fig.savefig(FIGURES_DIR / "fig_ds_difficulty_stages.png")
    plt.close(fig)
    print(f"Saved: fig_ds_difficulty_stages.pdf")


def plot_error_recovery(recovery_df: pd.DataFrame):
    """Stacked bar: error recovery strategy distribution by model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models_order = [s for _, s, _ in MODELS]
    strategies = ["no_errors", "retry_same", "switch_tool", "give_up"]
    strat_colors = {
        "no_errors": "#2ca02c",
        "retry_same": "#1f77b4",
        "switch_tool": "#ff7f0e",
        "give_up": "#d62728",
    }
    strat_labels = {
        "no_errors": "No Errors",
        "retry_same": "Retry Same Tool",
        "switch_tool": "Switch Tool",
        "give_up": "Give Up",
    }

    pivot = recovery_df.pivot_table(
        index="short", columns="strategy", values="pct", fill_value=0
    )
    pivot = pivot.reindex(models_order)
    pivot = pivot.reindex(columns=strategies, fill_value=0)

    bottom = np.zeros(len(pivot))
    for strat in strategies:
        if strat in pivot.columns:
            vals = pivot[strat].values
            ax.bar(range(len(pivot)), vals, bottom=bottom,
                  label=strat_labels[strat], color=strat_colors[strat], alpha=0.8)
            bottom += vals

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of Tasks")
    ax.set_title("Error Recovery Strategy by Model", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ds_error_recovery.pdf")
    fig.savefig(FIGURES_DIR / "fig_ds_error_recovery.png")
    plt.close(fig)
    print(f"Saved: fig_ds_error_recovery.pdf")


# ═════════════════════════════════════════════════════════════════════════
# Report generation
# ═════════════════════════════════════════════════════════════════════════


def generate_report(
    flow_df: pd.DataFrame,
    first_tool_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    df: pd.DataFrame,
) -> str:
    """Generate markdown report."""
    lines = [
        "# DeepSeek V3 Behavioral Analysis",
        "",
        "## Why does DeepSeek V3 consistently outperform other models?",
        "",
        "---",
        "",
        "## 1. Pipeline Stage Usage",
        "",
        "| Model | Backbone Gen (%) | Seq Design (%) | Struct Pred (%) | Scoring (%) | Avg Tools | Avg Stages |",
        "|-------|-----------------|----------------|-----------------|-------------|-----------|------------|",
    ]

    for _, r in flow_df.iterrows():
        lines.append(
            f"| {r['short']} | {r['backbone_gen_rate']:.1f} | {r['seq_design_rate']:.1f} | "
            f"{r['struct_pred_rate']:.1f} | {r['scoring_rate']:.1f} | "
            f"{r['mean_tool_calls']:.1f} | {r['mean_stages']:.1f} |"
        )

    # First tool analysis
    lines += ["", "---", "", "## 2. First MCP Tool (De Novo Tasks)", ""]
    lines.append("| Model | Backbone Gen (%) | Seq Design (%) | Struct Pred (%) | Scoring (%) | None (%) |")
    lines.append("|-------|-----------------|----------------|-----------------|-------------|----------|")

    for cond, short, _ in MODELS:
        cdf = first_tool_df[first_tool_df["condition"] == cond]
        if len(cdf) == 0:
            continue
        bb = cdf[cdf["stage"] == "backbone_gen"]["pct"].values
        sd = cdf[cdf["stage"] == "seq_design"]["pct"].values
        sp = cdf[cdf["stage"] == "struct_pred"]["pct"].values
        sc = cdf[cdf["stage"] == "scoring"]["pct"].values
        nn = cdf[cdf["stage"] == "none"]["pct"].values
        lines.append(
            f"| {short} | {bb[0]:.1f} | {sd[0]:.1f} | {sp[0]:.1f} | "
            f"{sc[0]:.1f} | {nn[0]:.1f} |"
        )

    # Difficulty invariance
    lines += ["", "---", "", "## 3. Difficulty Invariance", ""]
    lines.append("| Model | Easy Score | Medium Score | Hard Score | Drop (Easy→Hard) |")
    lines.append("|-------|-----------|-------------|------------|-------------------|")

    for cond, short, _ in MODELS:
        mdf = diff_df[diff_df["short"] == short].set_index("difficulty")
        if "easy" not in mdf.index or "hard" not in mdf.index:
            continue
        easy = mdf.loc["easy", "mean_score"]
        med = mdf.loc["medium", "mean_score"] if "medium" in mdf.index else np.nan
        hard = mdf.loc["hard", "mean_score"]
        drop = easy - hard
        lines.append(
            f"| {short} | {easy:.1f} | {med:.1f} | {hard:.1f} | {drop:+.1f} |"
        )

    # Benchmark mode discovery
    lines += ["", "---", "", "## 4. Benchmark Mode Tool Discovery", ""]
    lines.append("| Model | Mode | Backbone Gen (%) | Seq Design (%) | Struct Pred (%) |")
    lines.append("|-------|------|-----------------|----------------|-----------------|")

    for _, r in discovery_df.iterrows():
        lines.append(
            f"| {r['short']} | {r['mode']} | {r['backbone_gen_rate']:.1f} | "
            f"{r['seq_design_rate']:.1f} | {r['struct_pred_rate']:.1f} |"
        )

    # Error recovery
    lines += ["", "---", "", "## 5. Error Recovery", ""]
    lines.append("| Model | No Errors (%) | Retry (%) | Switch Tool (%) | Give Up (%) |")
    lines.append("|-------|---------------|-----------|-----------------|-------------|")

    for cond, short, _ in MODELS:
        cdf = recovery_df[recovery_df["condition"] == cond]
        if len(cdf) == 0:
            continue
        ne = cdf[cdf["strategy"] == "no_errors"]["pct"].values
        rs = cdf[cdf["strategy"] == "retry_same"]["pct"].values
        st = cdf[cdf["strategy"] == "switch_tool"]["pct"].values
        gu = cdf[cdf["strategy"] == "give_up"]["pct"].values
        lines.append(
            f"| {short} | {ne[0]:.1f} | {rs[0]:.1f} | {st[0]:.1f} | {gu[0]:.1f} |"
        )

    # Key findings
    lines += [
        "", "---", "",
        "## 6. Key Findings",
        "",
    ]

    # Auto-detect key differences
    ds_user = flow_df[flow_df["short"] == "DS-V3 user"]
    if len(ds_user) > 0:
        ds = ds_user.iloc[0]
        others = flow_df[~flow_df["short"].str.startswith("DS-V3")]

        lines.append(f"1. **Pipeline completeness**: DeepSeek V3 user achieves "
                     f"{ds['backbone_gen_rate']:.0f}% backbone gen rate vs "
                     f"{others['backbone_gen_rate'].mean():.0f}% average for other models.")

        lines.append(f"\n2. **Tool diversity**: DeepSeek uses {ds['mean_unique_tools']:.1f} "
                     f"unique tools/task vs {others['mean_unique_tools'].mean():.1f} for others.")

        # Difficulty invariance
        ds_diff = diff_df[diff_df["short"] == "DS-V3 user"].set_index("difficulty")
        if "easy" in ds_diff.index and "hard" in ds_diff.index:
            ds_drop = ds_diff.loc["easy", "mean_score"] - ds_diff.loc["hard", "mean_score"]
            lines.append(f"\n3. **Difficulty invariance**: DeepSeek drops only {ds_drop:.1f} pts "
                         "from easy→hard, maintaining consistent pipeline execution regardless of task complexity.")

    lines += ["", "---", "", "*Generated by `scripts/analysis/deepseek_behavior_analysis.py`*"]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def main():
    print("Extracting tool call logs...")
    df = extract_all_tool_logs()
    print(f"  {len(df)} task × condition entries")

    print("\n1. Pipeline stage usage...")
    flow_df = analyze_pipeline_flow(df)
    print(flow_df[["short", "backbone_gen_rate", "seq_design_rate",
                   "struct_pred_rate", "mean_unique_tools"]].to_string(index=False))

    print("\n2. First tool call analysis...")
    first_tool_df = analyze_first_tool(df)

    print("\n3. Difficulty × stage analysis...")
    diff_df = analyze_difficulty_stages(df)

    print("\n4. Error recovery analysis...")
    recovery_df = analyze_error_recovery(df)

    print("\n5. Benchmark mode discovery...")
    discovery_df = analyze_benchmark_discovery(df)
    print(discovery_df[["short", "mode", "backbone_gen_rate"]].to_string(index=False))

    # Save CSV
    print("\n6. Saving CSVs...")
    flow_df.to_csv(ANALYSIS_DIR / "deepseek_pipeline_flow.csv", index=False)
    diff_df.to_csv(ANALYSIS_DIR / "deepseek_difficulty_stages.csv", index=False)
    recovery_df.to_csv(ANALYSIS_DIR / "deepseek_error_recovery.csv", index=False)
    discovery_df.to_csv(ANALYSIS_DIR / "deepseek_benchmark_discovery.csv", index=False)

    # Plots
    print("\n7. Generating figures...")
    plot_first_tool(first_tool_df)
    plot_pipeline_flow(flow_df)
    plot_difficulty_stages(diff_df)
    plot_error_recovery(recovery_df)

    # Report
    print("\n8. Generating report...")
    report = generate_report(flow_df, first_tool_df, diff_df, recovery_df, discovery_df, df)
    report_path = ANALYSIS_DIR / "deepseek_behavior_report.md"
    report_path.write_text(report)
    print(f"   Saved: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
