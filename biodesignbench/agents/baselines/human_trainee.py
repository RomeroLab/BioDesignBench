"""Human trainee baseline for BioDesignBench."""

import json
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput
from biodesignbench.tasks.schema import Task, TaskTier


class HumanTraineeBaseline(AgentInterface):
    """
    Human trainee baseline - graduate student performance.

    This baseline represents mid-range comparison.
    It loads results from graduate student participants
    who had 1 week effort per task.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize human trainee baseline.

        Args:
            data_dir: Directory containing human trainee results
        """
        self.data_dir = Path(data_dir) if data_dir else self._default_data_dir()
        self._results_cache: dict[str, dict[str, Any]] = {}

    def _default_data_dir(self) -> Path:
        """Get default data directory."""
        return Path(__file__).parent.parent.parent.parent / "data" / "baselines" / "human_trainee"

    def get_info(self) -> AgentInfo:
        """Return agent metadata."""
        return AgentInfo(
            agent_id="human-trainee",
            name="Human Trainee",
            version="1.0.0",
            description="Mid-range baseline from graduate student participants",
            provider="baseline",
            model="human",
            is_bio_specific=True,
            capabilities=["trainee_level", "learning"],
        )

    def setup(self) -> None:
        """Load human trainee results."""
        self._load_results()

    def teardown(self) -> None:
        """Clear results cache."""
        self._results_cache.clear()

    async def solve(self, task: Task, output_dir: Path | None = None) -> AgentOutput:
        """
        Return human trainee result for task.

        Args:
            task: Task object with description, inputs, constraints

        Returns:
            AgentOutput with human trainee solution
        """
        # Load results if not cached
        if not self._results_cache:
            self._load_results()

        # Get result for this task
        result = self._results_cache.get(task.task_id, {})

        if task.tier == TaskTier.TIER1:
            return AgentOutput(
                code=result.get("code", "# No human trainee code available"),
                artifacts=result.get("artifacts", []),
                tools_used=result.get("tools_used", ["human"]),
                api_calls=0,
                iterations=1,
                reasoning_trace=result.get(
                    "reasoning", "Human trainee solution (1 week effort)."
                ),
            )
        else:
            return AgentOutput(
                designs=result.get("designs", []),
                tools_used=result.get("tools_used", ["human"]),
                api_calls=0,
                iterations=1,
                reasoning_trace=result.get(
                    "reasoning", "Human trainee design (1 week effort)."
                ),
            )

    def _load_results(self) -> None:
        """Load human trainee results from data directory."""
        if not self.data_dir.exists():
            # Create directory and placeholder
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._create_placeholder_results()
            return

        # Load all JSON files in data directory
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "task_id" in data:
                        self._results_cache[data["task_id"]] = data
                    elif isinstance(data, list):
                        for item in data:
                            if "task_id" in item:
                                self._results_cache[item["task_id"]] = item
            except (json.JSONDecodeError, IOError):
                pass

    def _create_placeholder_results(self) -> None:
        """Create placeholder results file."""
        placeholder = {
            "description": "Human trainee baseline results",
            "source": "Graduate student study",
            "effort_per_task": "1 week",
            "tasks": [],
        }
        placeholder_file = self.data_dir / "placeholder.json"
        with open(placeholder_file, "w") as f:
            json.dump(placeholder, f, indent=2)

    def has_result(self, task_id: str) -> bool:
        """Check if human trainee result exists for task."""
        if not self._results_cache:
            self._load_results()
        return task_id in self._results_cache

    def get_precomputed_result(self, task_id: str) -> dict | None:
        """Return precomputed EvaluationResult dict if available."""
        if not self._results_cache:
            self._load_results()
        result = self._results_cache.get(task_id)
        if result and "partial_score" in result:
            return result
        return None

    def get_effort_info(self, task_id: str) -> dict[str, Any] | None:
        """Get effort information for human trainee result."""
        if not self._results_cache:
            self._load_results()
        result = self._results_cache.get(task_id, {})
        return result.get("effort_info")
