"""Score agent error recovery: graceful failure handling, retries, recovery.

Evaluates whether the agent:
1. Handled tool failures without crashing the pipeline
2. Retried failed tools with adjusted parameters
3. Eventually produced valid output after encountering errors
"""

from __future__ import annotations

from typing import Any


def _count_failures(tool_call_log: list[dict[str, Any]]) -> int:
    """Count tool calls that failed."""
    return sum(1 for e in tool_call_log if not e.get("success", True))


def _count_retries_with_adjustment(tool_call_log: list[dict[str, Any]]) -> int:
    """Count cases where a failed tool was retried with different arguments.

    A retry-with-adjustment is: same tool name appears after a failure,
    with different args_summary.
    """
    retries = 0
    failed_tools: dict[str, dict] = {}  # tool -> last failed args

    for entry in tool_call_log:
        tool = entry.get("tool", "")
        success = entry.get("success", True)
        args = entry.get("args_summary", {})

        if not success:
            failed_tools[tool] = args
        elif tool in failed_tools:
            # Same tool succeeded after a failure
            if args != failed_tools[tool]:
                retries += 1
            del failed_tools[tool]

    return retries


def _has_recovery(tool_call_log: list[dict[str, Any]]) -> bool:
    """Check if the agent recovered from failures.

    Recovery = at least one failure occurred, and a later tool call succeeded.
    """
    saw_failure = False
    for entry in tool_call_log:
        if not entry.get("success", True):
            saw_failure = True
        elif saw_failure:
            return True
    return False


def _continued_after_failure(tool_call_log: list[dict[str, Any]]) -> bool:
    """Check if the pipeline continued after a failure (didn't crash)."""
    failure_indices = [
        i for i, e in enumerate(tool_call_log) if not e.get("success", True)
    ]
    if not failure_indices:
        return True  # No failures = graceful by default
    last_failure = max(failure_indices)
    return last_failure < len(tool_call_log) - 1


def score_error_recovery(
    tool_call_log: list[dict[str, Any]],
    iterations: int = 1,
    max_points: int = 10,
) -> dict[str, Any]:
    """Score the agent's error recovery behavior.

    Components:
    - Graceful failure handling (40%): Pipeline continued after errors
    - Retry with adjustment (40%): Failed tools retried with different params
    - Recovery success (20%): Eventually produced output after failures

    When no errors occurred, the agent gets full marks (nothing to recover from).

    Args:
        tool_call_log: Ordered list of tool call entries.
        iterations: Total agent iterations (higher = more attempts).
        max_points: Maximum points for this component.

    Returns:
        Dict with: score, max, num_failures, retries_with_adjustment,
        continued_after_failure, recovered.
    """
    if not tool_call_log:
        # No tool calls → no errors → full credit
        return {
            "score": max_points,
            "max": max_points,
            "num_failures": 0,
            "retries_with_adjustment": 0,
            "continued_after_failure": True,
            "recovered": True,
            "details": "No tool calls — full credit by default",
        }

    num_failures = _count_failures(tool_call_log)

    if num_failures == 0:
        # No errors occurred — full marks
        return {
            "score": max_points,
            "max": max_points,
            "num_failures": 0,
            "retries_with_adjustment": 0,
            "continued_after_failure": True,
            "recovered": True,
            "details": "No errors encountered",
        }

    # Errors occurred — score recovery behavior
    continued = _continued_after_failure(tool_call_log)
    retries = _count_retries_with_adjustment(tool_call_log)
    recovered = _has_recovery(tool_call_log)

    # 1. Graceful failure handling (40%)
    graceful_points = max_points * 0.4 if continued else 0.0

    # 2. Retry with adjustment (40%)
    # Full credit if retried at least once with different args
    if retries >= 2:
        retry_ratio = 1.0
    elif retries == 1:
        retry_ratio = 0.7
    elif iterations > 1 and continued:
        # Tried again (new iteration) but didn't change tool args
        retry_ratio = 0.3
    else:
        retry_ratio = 0.0
    retry_points = retry_ratio * max_points * 0.4

    # 3. Recovery success (20%)
    recovery_points = max_points * 0.2 if recovered else 0.0

    total = int(round(graceful_points + retry_points + recovery_points))

    return {
        "score": min(total, max_points),
        "max": max_points,
        "num_failures": num_failures,
        "retries_with_adjustment": retries,
        "continued_after_failure": continued,
        "recovered": recovered,
    }
