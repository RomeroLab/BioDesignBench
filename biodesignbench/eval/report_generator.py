"""Report generation for benchmark results.

Produces structured reports with figure specifications for visualizing
benchmark outcomes. Matplotlib/seaborn rendering is optional — when
available, ``render_figures()`` produces actual PNG files.

Usage::

    report = generate_report(results, output_dir=Path("reports/"))
    figures = render_figures(results, output_dir=Path("reports/figures/"))
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIGURE_SPECS: dict[str, dict[str, str]] = {
    "task_matrix_heatmap": {
        "title": "Task Matrix: DesignTaskType x BiologicalContext Scores",
        "plot_type": "heatmap",
        "data_source": "category_scores",
        "description": "Heatmap of average scores per taxonomy cell (13 cells)",
    },
    "axis_decomposition": {
        "title": "Score Decomposition by Rubric Component",
        "plot_type": "stacked_bar",
        "data_source": "component_breakdown",
        "description": "Stacked bar chart showing contribution of each rubric component per agent",
    },
    "agent_comparison": {
        "title": "Agent Comparison: Total Scores",
        "plot_type": "box",
        "data_source": "agent_total_scores",
        "description": "Box plot comparing score distributions across agents",
    },
    "difficulty_curves": {
        "title": "Task Difficulty Curves",
        "plot_type": "line",
        "data_source": "task_scores_sorted",
        "description": "Per-agent score curves sorted by task difficulty",
    },
    "failure_modes": {
        "title": "Failure Mode Distribution",
        "plot_type": "bar",
        "data_source": "failure_mode_counts",
        "description": "Bar chart of failure mode frequencies per agent",
    },
    "tool_usage": {
        "title": "Tool Usage Patterns",
        "plot_type": "heatmap",
        "data_source": "tool_usage_matrix",
        "description": "Heatmap of tool usage frequency per agent per task category",
    },
    "trajectory_quality": {
        "title": "Trajectory Quality: Success Rate vs Tool Calls",
        "plot_type": "scatter",
        "data_source": "trajectory_metrics",
        "description": "Scatter plot of trajectory quality metrics",
    },
    "ablation_deltas": {
        "title": "Ablation Impact: Score Deltas from Baseline",
        "plot_type": "bar",
        "data_source": "ablation_results",
        "description": "Bar chart showing score change per ablation vs baseline",
    },
    "perturbation_robustness": {
        "title": "Perturbation Robustness: Score Retention",
        "plot_type": "grouped_bar",
        "data_source": "perturbation_scores",
        "description": "Grouped bar chart of score retention under perturbation levels",
    },
    "statistical_tests": {
        "title": "Pairwise Statistical Significance",
        "plot_type": "heatmap",
        "data_source": "pairwise_p_values",
        "description": "Heatmap of pairwise p-values between agents",
    },
}


def generate_report(
    results: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a structured benchmark report from results.

    Args:
        results: List of evaluation result dicts. If None, returns template.
        output_dir: Optional directory to save report artifacts.

    Returns:
        Dict with: summary, figure_specs, agent_rankings, per_task_scores.
    """
    results = results or []

    # Extract agent scores
    agent_scores: dict[str, list[float]] = {}
    per_task: dict[str, dict[str, float]] = {}

    for r in results:
        agent_id = r.get("agent_id", "unknown")
        task_id = r.get("task_id", "unknown")
        score = r.get("total_score", r.get("partial_score", 0.0))

        agent_scores.setdefault(agent_id, []).append(score)
        per_task.setdefault(task_id, {})[agent_id] = score

    # Compute rankings
    agent_means = {
        agent: sum(scores) / len(scores) if scores else 0.0
        for agent, scores in agent_scores.items()
    }
    rankings = sorted(agent_means.items(), key=lambda x: x[1], reverse=True)

    # Build summary
    summary = {
        "total_evaluations": len(results),
        "num_agents": len(agent_scores),
        "num_tasks": len(per_task),
        "agent_rankings": [
            {"rank": i + 1, "agent_id": agent, "mean_score": round(mean, 1)}
            for i, (agent, mean) in enumerate(rankings)
        ],
    }

    report = {
        "summary": summary,
        "figure_specs": dict(FIGURE_SPECS),
        "agent_scores": {k: round(v, 1) for k, v in agent_means.items()},
        "per_task_scores": per_task,
    }

    # Save to file if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

    return report


def render_figures(
    results: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> list[str]:
    """Render benchmark figures as PNG files using matplotlib.

    Requires matplotlib to be installed. Produces one figure per applicable
    FIGURE_SPEC based on the available data in *results*.

    Args:
        results: List of evaluation result dicts.
        output_dir: Directory to save PNG files. Defaults to current dir.

    Returns:
        List of file paths to rendered figures.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for figure rendering. "
            "Install it with: pip install matplotlib"
        )

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather data
    agent_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        agent_id = r.get("agent_id", "unknown")
        score = r.get("total_score", r.get("partial_score", 0.0))
        agent_scores[agent_id].append(score)

    rendered: list[str] = []

    # 1. Agent comparison box plot
    if len(agent_scores) >= 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(agent_scores.keys())
        data = [agent_scores[a] for a in labels]
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("Score")
        ax.set_title("Agent Comparison: Total Scores")
        ax.set_ylim(0, 105)
        path = output_dir / "agent_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rendered.append(str(path))

    # 2. Score distribution histogram
    if results:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_scores = [
            r.get("total_score", r.get("partial_score", 0.0))
            for r in results
        ]
        ax.hist(all_scores, bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        path = output_dir / "score_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rendered.append(str(path))

    return rendered
