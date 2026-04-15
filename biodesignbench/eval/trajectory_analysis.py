"""Trajectory analysis for agent tool-call sequences.

Analyzes the sequence of tool calls made by an agent to extract
behavioral metrics like tool usage patterns, refinement attempts,
and planning-vs-execution balance.

Usage::

    metrics = analyze_trajectory(tool_call_log)
    comparison = compare_trajectories({"agent_a": log_a, "agent_b": log_b}, "binder_001")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryMetrics:
    """Metrics extracted from an agent's tool-call trajectory."""

    total_tool_calls: int = 0
    unique_tools: int = 0
    success_rate: float = 0.0
    recovery_count: int = 0
    iterations_used: int = 0
    refinement_attempts: int = 0
    planning_vs_execution_ratio: float = 0.0


@dataclass
class ComparisonReport:
    """Comparison of trajectories from multiple agents on a single task."""

    task_id: str
    agent_metrics: dict[str, TrajectoryMetrics] = field(default_factory=dict)


# Tools considered "planning" rather than "execution"
_PLANNING_TOOLS = {
    "web_search", "pubmed_search", "read_file", "write_file",
    "execute_python", "suggest_hotspots", "get_design_status",
}

# Bio execution tools
_EXECUTION_TOOLS = {
    "design_binder", "validate_design", "optimize_sequence",
    "predict_complex", "analyze_interface", "predict_structure",
    "score_stability", "energy_minimize", "generate_backbone",
}


def analyze_trajectory(tool_call_log: list[dict[str, Any]]) -> TrajectoryMetrics:
    """Analyze an agent's tool-call log into trajectory metrics.

    Args:
        tool_call_log: List of dicts with at least "tool" and "success" keys.
            Optional: "iteration", "args_summary".

    Returns:
        TrajectoryMetrics with computed values.
    """
    if not tool_call_log:
        return TrajectoryMetrics()

    total = len(tool_call_log)
    tools_seen = set()
    successes = 0
    recoveries = 0
    refinements = 0
    planning_calls = 0
    execution_calls = 0

    prev_failed_tool = None

    for i, call in enumerate(tool_call_log):
        tool_name = call.get("tool", "")
        success = call.get("success", True)
        tools_seen.add(tool_name)

        if success:
            successes += 1

        # Count recoveries: success after a failure on the same tool
        if success and prev_failed_tool == tool_name:
            recoveries += 1

        if not success:
            prev_failed_tool = tool_name
        else:
            prev_failed_tool = None

        # Count refinement attempts: same tool called again with different args
        if i > 0 and tool_name == tool_call_log[i - 1].get("tool"):
            prev_args = tool_call_log[i - 1].get("args_summary", {})
            curr_args = call.get("args_summary", {})
            if prev_args != curr_args:
                refinements += 1

        # Planning vs execution
        tool_lower = tool_name.lower()
        if tool_lower in _PLANNING_TOOLS:
            planning_calls += 1
        elif tool_lower in _EXECUTION_TOOLS:
            execution_calls += 1

    # Compute iterations (max iteration value seen, or count of calls)
    iterations = max(
        (call.get("iteration", 1) for call in tool_call_log),
        default=1,
    )

    # Planning vs execution ratio
    total_categorized = planning_calls + execution_calls
    ratio = (
        planning_calls / total_categorized
        if total_categorized > 0
        else 0.0
    )

    return TrajectoryMetrics(
        total_tool_calls=total,
        unique_tools=len(tools_seen),
        success_rate=round(successes / total, 3) if total > 0 else 0.0,
        recovery_count=recoveries,
        iterations_used=iterations,
        refinement_attempts=refinements,
        planning_vs_execution_ratio=round(ratio, 3),
    )


def compare_trajectories(
    agent_logs: dict[str, list[dict[str, Any]]],
    task_id: str,
) -> ComparisonReport:
    """Compare tool-call trajectories from multiple agents on the same task.

    Args:
        agent_logs: Dict mapping agent_id to their tool_call_log.
        task_id: The task being compared.

    Returns:
        ComparisonReport with per-agent TrajectoryMetrics.
    """
    report = ComparisonReport(task_id=task_id)
    for agent_id, log in agent_logs.items():
        report.agent_metrics[agent_id] = analyze_trajectory(log)
    return report
