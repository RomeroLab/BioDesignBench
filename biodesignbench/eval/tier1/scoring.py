"""Standardized 100-point scoring framework for Tier 1 tasks.

Every Tier 1 task is scored on a 100-point rubric with four components:
  - script_executes (25 pts): Agent code runs without fatal error
  - primary_artifact (25 pts): Main output file is valid
  - secondary_artifacts (25 pts): Metadata/report files are valid
  - content_correctness (25 pts): Ground truth checks pass

Partial credit is awarded within each component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_RUBRIC = {
    "script_executes": 25,
    "primary_artifact": 25,
    "secondary_artifacts": 25,
    "content_correctness": 25,
}


@dataclass
class ScoringRubric:
    """Configurable scoring rubric for a task.

    Each component maps to a maximum point value.
    Sub-checks within a component are weighted proportionally.
    """

    components: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_RUBRIC))

    @property
    def max_score(self) -> int:
        return sum(self.components.values())

    def validate(self) -> None:
        total = sum(self.components.values())
        if total != 100:
            raise ValueError(f"Rubric total must be 100, got {total}")


def partial_credit_range(
    actual: float,
    expected_min: float,
    expected_max: float,
    max_points: int,
) -> int:
    """Award partial credit for a value within an expected range.

    Returns max_points if actual is within [expected_min, expected_max].
    Otherwise, scales linearly based on distance from range.

    Args:
        actual: The actual value to score.
        expected_min: Lower bound of acceptable range.
        expected_max: Upper bound of acceptable range.
        max_points: Maximum points for this check.

    Returns:
        Integer score in [0, max_points].
    """
    if actual <= 0:
        return 0

    if expected_min <= actual <= expected_max:
        return max_points

    # Below range: ratio of actual / expected_min
    if actual < expected_min:
        ratio = actual / expected_min if expected_min > 0 else 0
    else:
        # Above range: ratio of expected_max / actual
        ratio = expected_max / actual if actual > 0 else 0

    return int(max_points * min(max(ratio, 0), 1.0))


def score_artifact(
    validation: dict[str, Any],
    max_points: int,
    *,
    weights: dict[str, float] | None = None,
) -> int:
    """Score a single artifact based on its validation result.

    Default weighting:
      - exists: 40% of max_points
      - format_valid: 30% of max_points (based on no errors)
      - structure: 30% of max_points (based on expected content)

    Args:
        validation: Validation result dict from a validator.
        max_points: Maximum points for this artifact.
        weights: Optional custom weight distribution.

    Returns:
        Integer score in [0, max_points].
    """
    if weights is None:
        weights = {"exists": 0.4, "format_valid": 0.3, "structure": 0.3}

    score = 0.0

    # Check existence
    exists = validation.get("exists", False)
    if exists:
        score += max_points * weights.get("exists", 0.4)
    else:
        return 0  # No points if file doesn't exist

    # Check format validity (absence of errors)
    errors = validation.get("errors", [])
    if not errors:
        score += max_points * weights.get("format_valid", 0.3)
    else:
        # Partial credit: fewer errors = more points
        error_count = len(errors)
        format_points = max_points * weights.get("format_valid", 0.3)
        score += format_points * max(0, 1 - error_count * 0.2)

    # Check structure (varies by validator)
    structure_points = max_points * weights.get("structure", 0.3)
    structure_checks = 0
    structure_pass = 0

    for key in ["all_valid_headers", "all_valid_sequences", "has_required_fields",
                "has_required_columns", "has_required_keys", "valid_json",
                "valid_tree", "valid_png", "valid_yaml", "all_same_length",
                "has_ter_records", "has_end_record", "non_empty", "contains_all"]:
        if key in validation:
            structure_checks += 1
            if validation[key]:
                structure_pass += 1

    if structure_checks > 0:
        score += structure_points * (structure_pass / structure_checks)
    else:
        # No structure checks applicable, award full structure points
        score += structure_points

    return int(min(score, max_points))


def calculate_task_score(
    rubric: ScoringRubric,
    results: dict[str, int],
) -> dict[str, Any]:
    """Calculate final task score from component results.

    Args:
        rubric: The scoring rubric defining max points per component.
        results: Dict mapping component names to actual scores.

    Returns:
        Dict with: breakdown, total, max_possible, percentage.
    """
    breakdown = {}
    for component, max_pts in rubric.components.items():
        actual = min(results.get(component, 0), max_pts)
        breakdown[component] = {"score": actual, "max": max_pts}

    total = sum(v["score"] for v in breakdown.values())
    max_possible = rubric.max_score

    return {
        "breakdown": breakdown,
        "total": total,
        "max_possible": max_possible,
        "percentage": round(total / max_possible * 100, 1) if max_possible > 0 else 0,
    }
