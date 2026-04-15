"""Robustness scoring for perturbation stress tests.

Compares normal vs perturbed evaluation results to compute
robustness metrics per agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biodesignbench.eval.results import EvaluationResult, PerturbationSummary


@dataclass
class PerturbationResult:
    """Per-task robustness result."""

    task_id: str
    agent_id: str
    original_score: float
    perturbed_score: float
    perturbation_level: str
    robustness_ratio: float  # perturbed / original (capped at 1.0)
    score_delta: float  # original - perturbed
    graceful_degradation: bool  # True if perturbed_score > 0


def compute_robustness(
    normal_results: list[EvaluationResult],
    perturbed_results: list[EvaluationResult],
) -> list[PerturbationResult]:
    """Compare normal vs perturbed results to compute per-task robustness.

    Args:
        normal_results: Results from normal (unperturbed) evaluation.
        perturbed_results: Results from perturbed evaluation.

    Returns:
        List of PerturbationResult, one per matched task.
    """
    # Build lookup: (task_id, agent_id) -> original score
    normal_lookup: dict[tuple[str, str], float] = {}
    for r in normal_results:
        key = (r.task_id, r.agent_id)
        normal_lookup[key] = r.get_overall_score()

    results = []
    for pr in perturbed_results:
        key = (pr.task_id, pr.agent_id)
        original_score = normal_lookup.get(key, 0.0)
        perturbed_score = pr.get_overall_score()

        if original_score > 0:
            ratio = min(perturbed_score / original_score, 1.0)
        else:
            ratio = 1.0 if perturbed_score == 0 else 0.0

        delta = original_score - perturbed_score

        results.append(
            PerturbationResult(
                task_id=pr.task_id,
                agent_id=pr.agent_id,
                original_score=original_score,
                perturbed_score=perturbed_score,
                perturbation_level=pr.perturbation_level or "unknown",
                robustness_ratio=round(ratio, 4),
                score_delta=round(delta, 2),
                graceful_degradation=perturbed_score > 0,
            )
        )

    return results


def compute_robustness_summary(
    results: list[PerturbationResult],
) -> PerturbationSummary:
    """Aggregate per-task robustness into a summary.

    Args:
        results: List of PerturbationResult for a single agent+level.

    Returns:
        PerturbationSummary with mean robustness, delta, and error handling rate.
    """
    if not results:
        return PerturbationSummary(agent_id="unknown", level="unknown")

    agent_id = results[0].agent_id
    level = results[0].perturbation_level

    n = len(results)
    mean_robustness = sum(r.robustness_ratio for r in results) / n
    mean_delta = sum(r.score_delta for r in results) / n
    graceful_count = sum(1 for r in results if r.graceful_degradation)
    error_handling_rate = graceful_count / n

    return PerturbationSummary(
        agent_id=agent_id,
        level=level,
        tasks_evaluated=n,
        mean_robustness=round(mean_robustness, 4),
        mean_score_delta=round(mean_delta, 2),
        error_handling_rate=round(error_handling_rate, 4),
    )
