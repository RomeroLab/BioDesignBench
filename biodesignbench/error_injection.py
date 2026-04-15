"""Error injection for evaluating agent robustness and recovery.

Injects controlled errors into MCP tool responses to test whether
agents can detect, diagnose, and recover from failures.

Usage::

    injector = ErrorInjector(error_rate=0.3, seed=42)
    errors = injector.get_errors_for_task("binder_001")
    for err in errors:
        modified_response = injector.inject(err.tool_name, original_response, err)
    recovery = get_recovery_score(tool_log, errors, max_points=10)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """Types of errors that can be injected."""
    TIMEOUT = "timeout"
    PARTIAL_FAILURE = "partial_failure"
    INPUT_CORRUPTION = "input_corruption"
    MISLEADING_METRIC = "misleading_metric"
    TOOL_UNAVAILABLE = "tool_unavailable"


# MCP tool names that can receive injected errors
MCP_TOOL_NAMES = [
    "design_binder",
    "validate_design",
    "optimize_sequence",
    "predict_complex",
    "analyze_interface",
    "predict_structure",
    "score_stability",
    "energy_minimize",
    "generate_backbone",
    "rosetta_score",
    "rosetta_relax",
    "rosetta_interface_score",
    "rosetta_design",
    "predict_structure_boltz",
    "predict_affinity_boltz",
]

# Descriptions for each error type
_ERROR_DESCRIPTIONS: dict[ErrorType, str] = {
    ErrorType.TIMEOUT: "Tool call timed out after 120s",
    ErrorType.PARTIAL_FAILURE: "Tool returned partial/incomplete results",
    ErrorType.INPUT_CORRUPTION: "Input PDB/FASTA was silently corrupted",
    ErrorType.MISLEADING_METRIC: "Returned metric values are unreliable (e.g., pLDDT=99 for a bad structure)",
    ErrorType.TOOL_UNAVAILABLE: "Tool is temporarily unavailable (service down)",
}


@dataclass
class InjectedError:
    """A single injected error."""
    error_type: ErrorType
    tool_name: str
    call_index: int
    description: str


class ErrorInjector:
    """Deterministic error injector for testing agent robustness.

    Uses hash(task_id) + seed for per-task determinism, ensuring
    the same task always gets the same errors with the same seed.

    Args:
        error_rate: Probability of injecting at least one error (0.0-1.0).
        seed: Random seed for reproducibility.
    """

    def __init__(self, error_rate: float = 0.3, seed: int = 42):
        self.error_rate = error_rate
        self.seed = seed

    def _task_rng(self, task_id: str) -> random.Random:
        """Create a deterministic RNG for a specific task."""
        h = hashlib.sha256(f"{task_id}:{self.seed}".encode()).hexdigest()
        task_seed = int(h[:16], 16) % (2**32)
        return random.Random(task_seed)

    def get_errors_for_task(self, task_id: str) -> list[InjectedError]:
        """Get the list of errors to inject for a given task.

        Deterministic: same task_id + same seed = same errors.
        Returns 0-3 errors depending on error_rate and task hash.
        """
        rng = self._task_rng(task_id)

        # Decide whether to inject any errors
        if rng.random() > self.error_rate:
            return []

        # Pick 1-3 errors
        num_errors = rng.randint(1, 3)
        error_types = list(ErrorType)

        errors = []
        used_tools: set[str] = set()

        for i in range(num_errors):
            error_type = rng.choice(error_types)
            # Pick a tool that hasn't been used yet
            available_tools = [t for t in MCP_TOOL_NAMES if t not in used_tools]
            if not available_tools:
                break
            tool_name = rng.choice(available_tools)
            used_tools.add(tool_name)

            errors.append(InjectedError(
                error_type=error_type,
                tool_name=tool_name,
                call_index=i,
                description=_ERROR_DESCRIPTIONS[error_type],
            ))

        return errors

    def inject(
        self,
        tool_name: str,
        tool_response: dict[str, Any],
        injected_error: InjectedError,
    ) -> dict[str, Any]:
        """Apply an injected error to a tool response.

        Args:
            tool_name: Name of the tool being called.
            tool_response: Original response from the tool.
            injected_error: Error to inject.

        Returns:
            Modified response dict with the error applied.
        """
        response = dict(tool_response)

        if injected_error.error_type == ErrorType.TIMEOUT:
            return {
                "error": "TimeoutError: Tool call timed out after 120s",
                "success": False,
                "tool": tool_name,
            }

        elif injected_error.error_type == ErrorType.TOOL_UNAVAILABLE:
            return {
                "error": f"ServiceUnavailable: {tool_name} is temporarily down",
                "success": False,
                "tool": tool_name,
            }

        elif injected_error.error_type == ErrorType.PARTIAL_FAILURE:
            response["success"] = True
            response["warning"] = "Partial results returned"
            # Remove some result keys to simulate partial output
            for key in list(response.keys()):
                if key in ("sequences", "structures", "designs"):
                    response[key] = response[key][:1] if isinstance(response[key], list) else response[key]
            return response

        elif injected_error.error_type == ErrorType.INPUT_CORRUPTION:
            response["success"] = True
            response["warning"] = "Input may be corrupted"
            if "pLDDT" in response:
                response["pLDDT"] = max(0, response["pLDDT"] - 30)
            return response

        elif injected_error.error_type == ErrorType.MISLEADING_METRIC:
            response["success"] = True
            # Inflate metrics to misleading values
            if "pLDDT" in response:
                response["pLDDT"] = 99.0
            if "ipTM" in response:
                response["ipTM"] = 0.99
            return response

        return response


def get_recovery_score(
    tool_log: list[dict[str, Any]],
    injected_errors: list[InjectedError],
    max_points: int = 10,
) -> dict[str, Any]:
    """Score the agent's recovery from injected errors.

    Scoring:
    - No injected errors: full marks (nothing to recover from)
    - Per error: did the agent detect it? retry? adjust parameters? succeed after?
    - Full recovery from all errors: high score
    - No recovery attempts: low score

    Args:
        tool_log: List of tool call dicts with "tool", "success", "args_summary" keys.
        injected_errors: Errors that were injected.
        max_points: Maximum recovery score.

    Returns:
        Dict with: score, max, errors_injected, errors_recovered, recovery_details.
    """
    if not injected_errors:
        return {
            "score": max_points,
            "max": max_points,
            "errors_injected": 0,
            "errors_recovered": 0,
            "recovery_details": [],
        }

    per_error_points = max_points / len(injected_errors)
    total = 0.0
    details = []

    for error in injected_errors:
        # Find tool calls matching this error's tool
        tool_calls = [
            (i, call) for i, call in enumerate(tool_log)
            if call.get("tool") == error.tool_name
        ]

        recovered = False
        detection_score = 0.0

        if len(tool_calls) >= 2:
            # Agent retried the tool
            first_call = tool_calls[0][1]
            later_calls = [c for _, c in tool_calls[1:]]

            # Check if any later call succeeded
            if any(c.get("success") for c in later_calls):
                recovered = True
                detection_score = per_error_points
            else:
                # Partial credit for retry attempt
                detection_score = per_error_points * 0.5

            # Check if agent adjusted parameters
            if len(tool_calls) >= 2:
                first_args = first_call.get("args_summary", {})
                for _, later_call in tool_calls[1:]:
                    later_args = later_call.get("args_summary", {})
                    if first_args != later_args:
                        detection_score = min(detection_score + per_error_points * 0.2, per_error_points)
                        break
        elif len(tool_calls) == 1:
            call = tool_calls[0][1]
            if call.get("success"):
                # Single successful call—error may not have been triggered
                detection_score = per_error_points * 0.3

        total += detection_score
        details.append({
            "error_type": error.error_type.value,
            "tool_name": error.tool_name,
            "recovered": recovered,
            "points": round(detection_score, 2),
        })

    return {
        "score": int(round(total)),
        "max": max_points,
        "errors_injected": len(injected_errors),
        "errors_recovered": sum(1 for d in details if d["recovered"]),
        "recovery_details": details,
    }
