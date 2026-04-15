"""Score agent orchestration: pipeline ordering, intermediate validation, adaptiveness.

Evaluates whether the agent:
1. Executed tools in a logical order matching the expected pipeline
2. Validated intermediate results (e.g. checked pLDDT between steps)
3. Adapted its approach based on intermediate results (changed args between calls)
"""

from __future__ import annotations

from typing import Any

from biodesignbench.eval.metrics.approach import (
    MCP_TOOL_EXPANSION,
    normalize_tool_name,
)

# Expected pipeline ordering per DesignApproach.
EXPECTED_PIPELINES: dict[str, list[str]] = {
    "de_novo": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "redesign": ["proteinmpnn", "esmfold"],
}

# Tools that count as intermediate validation steps
VALIDATION_TOOLS: set[str] = {
    "validate_design",
    "predict_complex",
    "analyze_interface",
    "esmfold",
    "score_stability",
    "rosetta_score",
    "rosetta_interface_score",
    "predict_structure_boltz",
    "predict_affinity_boltz",
}


def _expand_tool_name(tool: str) -> list[str]:
    """Expand an MCP wrapper tool to its underlying bio tools."""
    if tool in MCP_TOOL_EXPANSION:
        underlying = MCP_TOOL_EXPANSION[tool]
        return underlying if underlying else [tool]
    return [tool]


def _extract_ordered_bio_tools(tool_call_log: list[dict[str, Any]]) -> list[str]:
    """Extract ordered list of bio tool names from the tool call log."""
    utility_tools = {"execute_python", "read_file", "write_file"}
    ordered: list[str] = []
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        if tool in utility_tools:
            continue
        expanded = _expand_tool_name(tool)
        for t in expanded:
            ordered.append(normalize_tool_name(t))
    return ordered


def _longest_ordered_subsequence_length(
    actual: list[str], expected: list[str]
) -> int:
    """Find length of longest subsequence of `expected` in order in `actual`."""
    if not expected or not actual:
        return 0
    j = 0
    matched = 0
    for tool in actual:
        k = j
        while k < len(expected):
            if tool == normalize_tool_name(expected[k]):
                matched += 1
                j = k + 1
                break
            k += 1
    return matched


def _count_validation_steps(tool_call_log: list[dict[str, Any]]) -> int:
    """Count how many tool calls are validation/checking steps."""
    count = 0
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        if tool in VALIDATION_TOOLS:
            count += 1
        expanded = _expand_tool_name(tool)
        for t in expanded:
            if t in VALIDATION_TOOLS and tool not in VALIDATION_TOOLS:
                count += 1
    return count


def _has_adaptive_behavior(tool_call_log: list[dict[str, Any]]) -> bool:
    """Check if the agent adapted its approach between iterations."""
    tool_args: dict[str, list[dict]] = {}
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        args = entry.get("args_summary", {})
        if tool not in tool_args:
            tool_args[tool] = []
        tool_args[tool].append(args)

    for tool, args_list in tool_args.items():
        if len(args_list) >= 2:
            for i in range(1, len(args_list)):
                if args_list[i] != args_list[i - 1]:
                    return True
    return False


def get_task_category(task_id: str) -> str | None:
    """Extract category from task_id using taxonomy, with legacy fallback.

    First tries taxonomy-based lookup (returns DesignTaskType.value),
    then falls back to prefix-based matching for old task IDs.
    """
    try:
        from biodesignbench.taxonomy import get_category

        category = get_category(task_id)
        if category is not None:
            return category.task_type.value
    except ImportError:
        pass

    # Legacy fallback: map old prefixes to design approach
    _PREFIX_TO_APPROACH = {
        "binder": "de_novo", "scaffold": "de_novo", "ppi": "de_novo",
        "peptide": "de_novo", "dnb": "de_novo", "dnk": "de_novo",
        "cpx": "de_novo", "cfd": "de_novo",
        "antibody": "redesign", "stability": "redesign",
        "enzyme": "redesign", "fluorescence": "redesign",
        "sqo": "redesign",
    }
    prefix = task_id.split("_")[0]
    return _PREFIX_TO_APPROACH.get(prefix)


def score_orchestration(
    tool_call_log: list[dict[str, Any]],
    task_id: str,
    max_points: int = 15,
) -> dict[str, Any]:
    """Score the agent's multi-step pipeline orchestration.

    Components:
    - Pipeline ordering (50%): Tools executed in correct order
    - Intermediate validation (30%): Validation steps between design steps
    - Adaptive behavior (20%): Adjusted parameters based on results
    """
    if not tool_call_log:
        return {
            "score": 0,
            "max": max_points,
            "pipeline_order_score": 0.0,
            "validation_score": 0.0,
            "adaptive_score": 0.0,
            "details": "No tool calls recorded",
        }

    category = get_task_category(task_id)
    expected_pipeline = EXPECTED_PIPELINES.get(category, [])

    # 1. Pipeline ordering (50%)
    ordered_tools = _extract_ordered_bio_tools(tool_call_log)

    if expected_pipeline:
        matched = _longest_ordered_subsequence_length(ordered_tools, expected_pipeline)
        order_ratio = matched / len(expected_pipeline)
    else:
        order_ratio = 1.0 if ordered_tools else 0.0

    pipeline_points = order_ratio * max_points * 0.5

    # 2. Intermediate validation (30%)
    validation_count = _count_validation_steps(tool_call_log)
    if validation_count >= 2:
        validation_ratio = 1.0
    elif validation_count == 1:
        validation_ratio = 0.6
    else:
        validation_ratio = 0.0

    validation_points = validation_ratio * max_points * 0.3

    # 3. Adaptive behavior (20%)
    adaptive = _has_adaptive_behavior(tool_call_log)
    adaptive_points = max_points * 0.2 if adaptive else 0.0

    total = int(round(pipeline_points + validation_points + adaptive_points))

    return {
        "score": min(total, max_points),
        "max": max_points,
        "pipeline_order_score": round(pipeline_points, 1),
        "validation_score": round(validation_points, 1),
        "adaptive_score": round(adaptive_points, 1),
        "expected_pipeline": expected_pipeline,
        "actual_tool_order": ordered_tools,
        "validation_steps": validation_count,
        "adaptive_behavior": adaptive,
    }
