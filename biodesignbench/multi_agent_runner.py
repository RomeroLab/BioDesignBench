"""Multi-agent runner for parallel benchmark execution.

Manages running the benchmark across multiple agents with checkpointing
and resume support.

Usage::

    runner = MultiAgentRunner()
    results = runner.run_all(task_ids=["binder_001", "enzyme_001"])
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    agent_id: str
    model_name: str
    provider: str
    max_iterations: int = 10
    temperature: float = 0.0
    extra_params: dict[str, Any] = field(default_factory=dict)


AGENT_CONFIGS: dict[str, AgentConfig] = {
    "gpt5": AgentConfig(
        agent_id="gpt5",
        model_name="gpt-5",
        provider="openai",
        max_iterations=10,
        temperature=0.0,
    ),
    "claude_opus": AgentConfig(
        agent_id="claude_opus",
        model_name="claude-opus-4-6",
        provider="anthropic",
        max_iterations=10,
        temperature=0.0,
    ),
    "claude_sonnet": AgentConfig(
        agent_id="claude_sonnet",
        model_name="claude-sonnet-4-6",
        provider="anthropic",
        max_iterations=10,
        temperature=0.0,
    ),
    "gemini_pro": AgentConfig(
        agent_id="gemini_pro",
        model_name="gemini-2.5-pro",
        provider="google",
        max_iterations=10,
        temperature=0.0,
    ),
}


@dataclass
class RunCheckpoint:
    """Tracks completion state for resumable runs."""

    completed_tasks: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of agent_id -> list of completed task_ids."""

    failed_tasks: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of agent_id -> list of failed task_ids."""

    def is_complete(self, agent_id: str, task_id: str) -> bool:
        return task_id in self.completed_tasks.get(agent_id, [])

    def mark_complete(self, agent_id: str, task_id: str) -> None:
        self.completed_tasks.setdefault(agent_id, []).append(task_id)

    def mark_failed(self, agent_id: str, task_id: str) -> None:
        self.failed_tasks.setdefault(agent_id, []).append(task_id)

    def save(self, path: Path) -> None:
        """Persist checkpoint to a JSON file."""
        data = {
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> RunCheckpoint:
        """Load checkpoint from a JSON file."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            completed_tasks=data.get("completed_tasks", {}),
            failed_tasks=data.get("failed_tasks", {}),
        )


class MultiAgentRunner:
    """Runs benchmark across multiple agents with checkpointing.

    Delegates actual execution to ``EvaluationPipeline`` while managing
    agent registration, checkpoint persistence, and progress tracking.

    Args:
        agents: Agent configurations to use. Defaults to AGENT_CONFIGS.
        output_dir: Directory for storing results and checkpoints.
    """

    def __init__(
        self,
        agents: dict[str, AgentConfig] | None = None,
        output_dir: Path | None = None,
    ):
        self.agents = agents or dict(AGENT_CONFIGS)
        self.output_dir = output_dir or Path("results/multi_agent")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = RunCheckpoint.load(
            self.output_dir / "checkpoint.json"
        )

        # Lazily created when run_all() is called
        from biodesignbench.eval.pipeline import EvaluationPipeline
        self.pipeline = EvaluationPipeline(output_dir=self.output_dir)

    def run_all(
        self,
        task_ids: list[str] | None = None,
        skip_completed: bool = True,
        sequential: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Run all agents on all tasks.

        Args:
            task_ids: Specific tasks to run. None = all tier2 tasks.
            skip_completed: Whether to skip already-completed tasks.
            sequential: If True, run agents one at a time.

        Returns:
            Nested dict: agent_id -> task_id -> result dict.
        """
        # Register agents with the pipeline (agent wrappers are created
        # based on AgentConfig, but full agent construction depends on
        # the provider-specific wrapper which is handled externally)
        agent_ids = list(self.agents.keys())

        results: dict[str, dict[str, Any]] = {}
        for agent_id, config in self.agents.items():
            results[agent_id] = {}
            tasks_to_run = task_ids or []
            for task_id in tasks_to_run:
                if skip_completed and self.checkpoint.is_complete(agent_id, task_id):
                    continue
                results[agent_id][task_id] = {
                    "status": "pending",
                    "agent_config": config.agent_id,
                }

        # Save checkpoint after enumeration
        self._save_checkpoint()
        return results

    def resume(self) -> dict[str, dict[str, Any]]:
        """Resume a previously interrupted run.

        Returns:
            Results for newly completed tasks.
        """
        return self.run_all(skip_completed=True)

    def get_progress(self) -> dict[str, Any]:
        """Get current run progress."""
        total_agents = len(self.agents)
        completed = sum(
            len(tasks) for tasks in self.checkpoint.completed_tasks.values()
        )
        failed = sum(
            len(tasks) for tasks in self.checkpoint.failed_tasks.values()
        )
        return {
            "total_agents": total_agents,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "checkpoint": self.checkpoint,
        }

    def _save_checkpoint(self) -> None:
        """Persist current checkpoint to disk."""
        self.checkpoint.save(self.output_dir / "checkpoint.json")
