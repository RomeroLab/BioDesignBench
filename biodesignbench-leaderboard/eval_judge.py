"""LLM Judge orchestration for the leaderboard backend.

Runs the cross-model judge panel on each successfully scored task and
merges the resulting LLM points into the algorithmic component scores
to produce hybrid totals (28 LLM points + 72 algorithmic points = 100).

The judge panel uses 3 judges from different model families with
self-exclusion (PoLL, Verga et al. 2024). Individual judge calls are
synchronous; we process tasks sequentially to keep the API spend
predictable. Provider keys are read from environment variables that
must be configured as HuggingFace Space secrets:

    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_API_KEY
    DEEPSEEK_API_KEY
"""

from __future__ import annotations

import logging
from typing import Any

from llm_judge import (
    LLMJudgePanel,
    detect_agent_family,
    merge_algo_and_judge_scores,
    split_algo_score,
)

logger = logging.getLogger(__name__)


def _build_algo_dict(task_result: dict[str, Any]) -> dict[str, float]:
    """Pull per-component algo scores from a task result.

    Prefers 'cpu_scores' (post-Boltz) but falls back to 'final_scores'
    if it has been computed already.
    """
    if "cpu_scores" in task_result:
        return dict(task_result["cpu_scores"])
    if "final_scores" in task_result:
        return dict(task_result["final_scores"])
    return {
        "approach": 0,
        "orchestration": 0,
        "quality": 0,
        "feasibility": 0,
        "novelty": 0,
        "diversity": 0,
    }


def run_judge_panel(
    per_task_results: dict[str, dict[str, Any]],
    agent_id: str,
    dry_run: bool = False,
    progress_callback=None,
) -> dict[str, dict[str, Any]]:
    """Run the LLM judge panel over every successful task in a submission.

    For each task with a non-empty design output:
      1. Look up the original task prompt (used to give the panel context).
      2. Build a 3-judge panel that excludes the agent's own model family.
      3. Run all judges synchronously and aggregate.
      4. Compute the hybrid component scores by:
           - splitting each algo score into its algo-portion (split_algo_score)
           - adding the matching judge LLM-portion (merge_algo_and_judge_scores)
      5. Store both raw judge results and final hybrid scores on the task.

    The function modifies per_task_results in place and also returns it.

    Args:
        per_task_results: Dict mapping task_id → task result (from the
            dispatcher + boltz post-eval pipeline).
        agent_id: Agent identifier for self-exclusion (e.g., 'gpt5-tools').
        dry_run: If True, judges return midpoint scores without API calls.
        progress_callback: Optional callable(task_id, i, total).

    Returns:
        The same dict, now augmented with 'judge_scores' and 'hybrid_scores'
        per task and 'hybrid_total' on each successful entry.
    """
    from eval_tasks import get_task

    family = detect_agent_family(agent_id)
    panel = LLMJudgePanel(agent_model_family=family, dry_run=dry_run)
    logger.info(
        f"LLM judge panel for agent '{agent_id}' (family={family}): "
        f"{len(panel.judges)} judges, dry_run={dry_run}"
    )

    eligible = [
        tid for tid, r in per_task_results.items()
        if r.get("success") and r.get("sequences")
    ]
    total = len(eligible)

    for i, task_id in enumerate(eligible):
        result = per_task_results[task_id]

        # Pull task prompt for judge context. If the dataset is not
        # reachable (e.g., dev mode without HF_TOKEN) we still run with
        # a placeholder description rather than aborting the whole run.
        task_data = get_task(task_id) or {}
        task_description = task_data.get("prompt_md") or f"BioDesignBench task {task_id}"

        algo_metrics = result.get("agent_metrics", {})
        if "boltz_metrics" in result:
            algo_metrics = {**algo_metrics, **result["boltz_metrics"]}

        try:
            judge_result = panel.evaluate_sync(
                task_description=task_description,
                tool_call_log=result.get("run_log", []),
                designed_sequences=result.get("sequences", []),
                algorithmic_metrics=algo_metrics,
            )
        except Exception as e:
            logger.error(f"Judge panel failed on {task_id}: {e}")
            judge_result = None

        # Build algo-portion dict (split each component down to its algo max)
        algo_full = _build_algo_dict(result)
        rubric_max = {
            "approach": 20, "orchestration": 15, "quality": 35,
            "feasibility": 15, "novelty": 5, "diversity": 10,
        }
        algo_split = {
            comp: split_algo_score(comp, score, rubric_max[comp])
            for comp, score in algo_full.items()
        }

        hybrid = merge_algo_and_judge_scores(algo_split, judge_result)
        hybrid_total = sum(hybrid.values())

        result["judge_scores"] = judge_result
        result["hybrid_scores"] = hybrid
        result["hybrid_total"] = round(hybrid_total, 2)

        if progress_callback:
            progress_callback(task_id, i + 1, total)

        logger.info(
            f"[{i+1}/{total}] {task_id}: hybrid={hybrid_total:.1f}"
        )

    return per_task_results
