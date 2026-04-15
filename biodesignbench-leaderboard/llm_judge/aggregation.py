"""Score aggregation and merging for LLM judge panel.

Implements:
- Weighted averaging with outlier downweighting
- Algo + LLM score merging with rubric cap enforcement
- Weight split configuration (72/28 algo-LLM)
"""

from __future__ import annotations

import statistics
from typing import Any

from llm_judge.rubrics import JUDGE_DIMENSIONS


# ---------------------------------------------------------------------------
# Weight split: algo + LLM portions per component (must sum to rubric max)
# ---------------------------------------------------------------------------

WEIGHT_SPLIT: dict[str, dict[str, int]] = {
    "approach":      {"algo": 10, "llm": 10},   # 20 total
    "orchestration": {"algo": 7,  "llm": 8},    # 15 total
    "quality":       {"algo": 35, "llm": 0},    # 35 total (no LLM)
    "feasibility":   {"algo": 10, "llm": 5},    # 15 total
    "novelty":       {"algo": 3,  "llm": 2},    # 5 total
    "diversity":     {"algo": 7,  "llm": 3},    # 10 total
}

# Mapping from LLM judge dimension → rubric component
_JUDGE_DIM_TO_COMPONENT: dict[str, str] = {
    "approach_strategy": "approach",
    "orchestration_reasoning": "orchestration",
    "bio_feasibility": "feasibility",
    "novelty_quality": "novelty",
    "diversity_quality": "diversity",
}

# Rubric max per component
_RUBRIC_MAX: dict[str, int] = {
    "approach": 20,
    "orchestration": 15,
    "quality": 35,
    "feasibility": 15,
    "novelty": 5,
    "diversity": 10,
}


def aggregate_judge_scores(
    judge_results: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Aggregate scores from multiple judges with outlier downweighting.

    For each dimension:
    1. Collect raw scores from all judges
    2. Compute median
    3. Downweight outliers (>2 points from median) by 0.5x
    4. Compute weighted average

    Args:
        judge_results: List of per-judge result dicts.
            Each maps dimension_name → {reasoning, score}.

    Returns:
        Aggregated dict mapping dimension_name → {score, reasoning, raw_scores}.

    Raises:
        ValueError: If judge_results is empty.
    """
    if not judge_results:
        raise ValueError("aggregate_judge_scores requires at least one judge result")

    if len(judge_results) == 1:
        # Single judge: pass through directly
        result = {}
        for dim in JUDGE_DIMENSIONS:
            entry = judge_results[0].get(dim, {"score": 0, "reasoning": ""})
            result[dim] = {
                "score": float(entry["score"]),
                "reasoning": entry["reasoning"],
                "raw_scores": [entry["score"]],
            }
        return result

    aggregated = {}
    for dim, info in JUDGE_DIMENSIONS.items():
        raw_scores = []
        reasonings = []
        for jr in judge_results:
            entry = jr.get(dim, {"score": info["max_score"] // 2, "reasoning": ""})
            raw_scores.append(float(entry["score"]))
            reasonings.append(entry.get("reasoning", ""))

        # Outlier detection: downweight scores >2 points from median
        med = statistics.median(raw_scores)
        weights = []
        for s in raw_scores:
            if abs(s - med) > 2.0:
                weights.append(0.5)
            else:
                weights.append(1.0)

        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(raw_scores, weights))
        weight_total = sum(weights)
        avg = weighted_sum / weight_total if weight_total > 0 else 0

        # Clamp to valid range
        avg = max(0, min(avg, info["max_score"]))

        aggregated[dim] = {
            "score": round(avg, 1),
            "reasoning": " | ".join(
                f"[Judge {i+1}] {r}" for i, r in enumerate(reasonings) if r
            ),
            "raw_scores": raw_scores,
        }

    return aggregated


def split_algo_score(
    component: str,
    original_score: float,
    original_max: int,
) -> float:
    """Scale an algorithmic score to its algo-only portion.

    For the hybrid system, algorithmic scores are computed against the
    original rubric max (e.g., approach out of 20), then scaled down
    to the algo-only portion (e.g., 10 out of 20).

    Quality is special: it keeps its full 35 points (no LLM portion).

    Args:
        component: Rubric component name.
        original_score: Score computed against original max.
        original_max: Original rubric max for this component.

    Returns:
        Scaled score for the algo-only portion.
    """
    split = WEIGHT_SPLIT.get(component)
    if split is None:
        return original_score

    algo_max = split["algo"]

    if split["llm"] == 0:
        # No LLM portion — return original score unchanged
        return original_score

    # Scale: (original_score / original_max) * algo_max
    if original_max == 0:
        return 0.0
    ratio = original_score / original_max
    return round(ratio * algo_max, 2)


def merge_algo_and_judge_scores(
    algo_scores: dict[str, float | int],
    judge_scores: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    """Merge algorithmic and LLM judge scores into final component scores.

    Args:
        algo_scores: Dict mapping component → algo-portion score.
            These should already be split via split_algo_score().
        judge_scores: Aggregated judge scores (from aggregate_judge_scores).
            None if LLM judge is disabled.

    Returns:
        Dict mapping component → final merged score (capped at rubric max).
    """
    if judge_scores is None:
        return dict(algo_scores)

    merged = {}
    for component, algo_score in algo_scores.items():
        rubric_max = _RUBRIC_MAX.get(component, 100)

        # Find matching judge dimension
        judge_dim = None
        for jd, comp in _JUDGE_DIM_TO_COMPONENT.items():
            if comp == component:
                judge_dim = jd
                break

        if judge_dim and judge_dim in judge_scores:
            llm_score = judge_scores[judge_dim].get("score", 0)
            if isinstance(llm_score, dict):
                llm_score = llm_score.get("score", 0)
            total = algo_score + llm_score
        else:
            total = algo_score

        merged[component] = min(total, rubric_max)

    return merged
