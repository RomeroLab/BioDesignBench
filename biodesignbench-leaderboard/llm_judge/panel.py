"""LLM Judge Panel: manages cross-model evaluation with self-exclusion.

Following PoLL (Verga et al., 2024): 3 judges from different model families,
excluding the generating model. Human baselines get all 4 judges.
"""

from __future__ import annotations

from typing import Any

from llm_judge.aggregation import aggregate_judge_scores
from llm_judge.judge import LLMJudge


# ---------------------------------------------------------------------------
# Available judge models (one per family)
# ---------------------------------------------------------------------------

JUDGE_MODELS: list[dict[str, str]] = [
    {
        "family": "anthropic",
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
    },
    {
        "family": "openai",
        "provider": "openai",
        "model": "gpt-5.2",
    },
    {
        "family": "google",
        "provider": "google",
        "model": "gemini-2.5-pro",
    },
    {
        "family": "deepseek",
        "provider": "deepseek",
        "model": "deepseek-chat",
    },
]


# ---------------------------------------------------------------------------
# Agent ID → model family mapping
# ---------------------------------------------------------------------------

_AGENT_FAMILY_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "gemini": "google",
    "deepseek": "deepseek",
    "human": "human",
}


def detect_agent_family(agent_id: str) -> str:
    """Map an agent ID to its model family.

    Args:
        agent_id: Agent identifier (e.g., 'claude-code', 'gpt5-tools-benchmark').

    Returns:
        Family string: 'anthropic', 'openai', 'google', 'deepseek', 'human',
        or 'unknown'.
    """
    agent_lower = agent_id.lower()
    for prefix, family in _AGENT_FAMILY_PREFIXES.items():
        if agent_lower.startswith(prefix):
            return family
    return "unknown"


def get_judge_models(agent_model_family: str) -> list[dict[str, str]]:
    """Select judge models for a given agent, excluding self.

    Args:
        agent_model_family: Family of the agent being evaluated
            ('anthropic', 'openai', 'google', 'deepseek', 'human', 'unknown').

    Returns:
        List of judge model dicts (3 for agents, 4 for human baselines).
    """
    if agent_model_family == "human":
        return list(JUDGE_MODELS)  # All 4 judges

    return [j for j in JUDGE_MODELS if j["family"] != agent_model_family]


class LLMJudgePanel:
    """Cross-model judge panel for protein design evaluation.

    Manages 3 judges (excluding the agent's own model family) and
    aggregates their scores.

    Args:
        agent_model_family: Model family to exclude ('anthropic', etc).
        dry_run: If True, all judges return deterministic midpoint scores.
    """

    def __init__(
        self,
        agent_model_family: str,
        dry_run: bool = False,
    ):
        self.agent_model_family = agent_model_family
        self.dry_run = dry_run
        self.judge_configs = get_judge_models(agent_model_family)
        self.judges = [
            LLMJudge(
                provider=cfg["provider"],
                model=cfg["model"],
                dry_run=dry_run,
            )
            for cfg in self.judge_configs
        ]

    def evaluate_sync(
        self,
        task_description: str,
        tool_call_log: list[dict[str, Any]],
        designed_sequences: list[str],
        algorithmic_metrics: dict[str, Any],
        reference_pipeline: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a design with all judges and aggregate.

        Args:
            task_description: Original task prompt.
            tool_call_log: Agent's tool call sequence.
            designed_sequences: Designed protein sequences.
            algorithmic_metrics: Computed biophysical metrics.
            reference_pipeline: Expected expert pipeline.

        Returns:
            Dict with aggregated scores, judge count, and individual results.
        """
        individual_results = []

        for judge in self.judges:
            result = judge.evaluate_sync(
                task_description=task_description,
                tool_call_log=tool_call_log,
                designed_sequences=designed_sequences,
                algorithmic_metrics=algorithmic_metrics,
                reference_pipeline=reference_pipeline,
            )
            individual_results.append(result)

        aggregated = aggregate_judge_scores(individual_results)

        return {
            **aggregated,
            "judge_count": len(self.judges),
            "individual_judges": [
                {
                    "model": cfg["model"],
                    "family": cfg["family"],
                    "scores": result,
                }
                for cfg, result in zip(self.judge_configs, individual_results)
            ],
        }
