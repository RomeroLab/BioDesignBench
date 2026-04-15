"""Evaluation result data structures."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(UTC)


class EvaluationResult(BaseModel):
    """Result of evaluating a single task."""

    task_id: str
    agent_id: str
    timestamp: datetime = Field(default_factory=_utcnow)

    # Execution status
    success: bool = False
    error_message: str | None = None

    # Tier 1 metrics (coding tasks)
    valid_execution: bool = False
    test_results: dict[str, bool] = Field(default_factory=dict)
    partial_score: float = 0.0  # 0-100

    # Tier 2 metrics (design tasks)
    approach_metrics: dict[str, Any] = Field(default_factory=dict)
    orchestration_metrics: dict[str, Any] = Field(default_factory=dict)
    error_recovery_metrics: dict[str, Any] = Field(default_factory=dict)
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    novelty_metrics: dict[str, Any] = Field(default_factory=dict)
    diversity_metrics: dict[str, Any] = Field(default_factory=dict)
    feasibility_metrics: dict[str, Any] = Field(default_factory=dict)

    # Execution metrics
    execution_time_seconds: float = 0.0
    api_calls: int = 0
    cost_usd: float | None = None
    tools_used: list[str] = Field(default_factory=list)
    iterations: int = 1

    # Raw outputs
    raw_output: dict[str, Any] = Field(default_factory=dict)
    prompt_file: str | None = None  # Path to standardized prompt file used

    # Contamination detection
    contamination_flags: list[str] = Field(default_factory=list)
    contamination_score: float = 0.0
    contamination_evidence: list[str] = Field(default_factory=list)

    # Perturbation metadata
    perturbation_level: str | None = None
    perturbation_details: list[str] = Field(default_factory=list)

    # Failure mode analysis
    failure_modes: list[str] = Field(default_factory=list)

    def get_overall_score(self) -> float:
        """Calculate overall score for this task.

        Uses partial_score for both tiers (0-100 scale).
        """
        if not self.success:
            return 0.0
        return self.partial_score


class AgentSummary(BaseModel):
    """Summary statistics for an agent across all tasks."""

    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    success_rate: float = 0.0

    # Tier 1 aggregates
    tier1_tasks: int = 0
    tier1_success_rate: float = 0.0
    tier1_avg_partial_score: float = 0.0

    # Tier 2 aggregates
    tier2_tasks: int = 0
    tier2_avg_partial_score: float = 0.0
    tier2_avg_approach: float = 0.0
    tier2_avg_orchestration: float = 0.0
    tier2_avg_error_recovery: float = 0.0
    tier2_avg_quality: float = 0.0
    tier2_avg_novelty: float = 0.0
    tier2_avg_diversity: float = 0.0

    # Efficiency metrics
    avg_execution_time: float = 0.0
    total_cost_usd: float = 0.0
    avg_iterations: float = 0.0


class PerturbationSummary(BaseModel):
    """Summary of perturbation robustness for an agent."""

    agent_id: str
    level: str
    tasks_evaluated: int = 0
    mean_robustness: float = 0.0  # mean(perturbed_score / original_score)
    mean_score_delta: float = 0.0  # mean(original - perturbed)
    error_handling_rate: float = 0.0  # fraction that degraded gracefully


class BenchmarkResults(BaseModel):
    """Complete benchmark results across all agents and tasks."""

    benchmark_version: str = "0.1.0"
    run_timestamp: datetime = Field(default_factory=_utcnow)

    # Configuration
    config: dict[str, Any] = Field(default_factory=dict)

    # Results
    results: list[EvaluationResult] = Field(default_factory=list)
    agent_summaries: dict[str, AgentSummary] = Field(default_factory=dict)

    def add_result(self, result: EvaluationResult) -> None:
        """Add a single evaluation result."""
        self.results.append(result)

    def compute_summaries(self) -> None:
        """Compute summary statistics for all agents."""
        from collections import defaultdict

        # Group results by agent
        agent_results: dict[str, list[EvaluationResult]] = defaultdict(list)
        for result in self.results:
            agent_results[result.agent_id].append(result)

        # Compute summaries
        for agent_id, results in agent_results.items():
            summary = AgentSummary(agent_id=agent_id)
            summary.total_tasks = len(results)
            summary.successful_tasks = sum(1 for r in results if r.success)
            summary.success_rate = (
                summary.successful_tasks / summary.total_tasks
                if summary.total_tasks > 0
                else 0.0
            )

            # Tier 1 aggregates
            tier1_results = [r for r in results if r.test_results]
            summary.tier1_tasks = len(tier1_results)
            if tier1_results:
                tier1_successful = sum(1 for r in tier1_results if r.success)
                summary.tier1_success_rate = (
                    tier1_successful / len(tier1_results)
                )
                summary.tier1_avg_partial_score = (
                    sum(r.partial_score for r in tier1_results) / len(tier1_results)
                )

            # Tier 2 aggregates
            tier2_results = [
                r for r in results
                if r.quality_metrics or r.novelty_metrics or r.diversity_metrics
                or r.approach_metrics
            ]
            summary.tier2_tasks = len(tier2_results)
            if tier2_results:
                summary.tier2_avg_partial_score = (
                    sum(r.partial_score for r in tier2_results) / len(tier2_results)
                )

                def _avg(field: str) -> float:
                    vals = [
                        getattr(r, field).get("score", 0)
                        for r in tier2_results if getattr(r, field)
                    ]
                    return sum(vals) / len(vals) if vals else 0.0

                summary.tier2_avg_approach = _avg("approach_metrics")
                summary.tier2_avg_orchestration = _avg("orchestration_metrics")
                summary.tier2_avg_error_recovery = _avg("error_recovery_metrics")
                summary.tier2_avg_quality = _avg("quality_metrics")
                summary.tier2_avg_novelty = _avg("novelty_metrics")
                summary.tier2_avg_diversity = _avg("diversity_metrics")

            # Execution metrics
            summary.avg_execution_time = (
                sum(r.execution_time_seconds for r in results) / len(results)
            )
            summary.total_cost_usd = sum(r.cost_usd or 0 for r in results)
            summary.avg_iterations = sum(r.iterations for r in results) / len(results)

            self.agent_summaries[agent_id] = summary

    def to_leaderboard(self) -> list[dict[str, Any]]:
        """Generate leaderboard data sorted by overall performance."""
        if not self.agent_summaries:
            self.compute_summaries()

        leaderboard = []
        for agent_id, summary in self.agent_summaries.items():
            leaderboard.append(
                {
                    "rank": 0,  # Will be set after sorting
                    "agent_id": agent_id,
                    "success_rate": summary.success_rate,
                    "total_tasks": summary.total_tasks,
                    "avg_time": summary.avg_execution_time,
                    "total_cost": summary.total_cost_usd,
                }
            )

        # Sort by success rate descending
        leaderboard.sort(key=lambda x: x["success_rate"], reverse=True)

        # Assign ranks
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard
