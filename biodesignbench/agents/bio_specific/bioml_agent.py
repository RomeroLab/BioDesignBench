"""AIDE (formerly BioML) agent wrapper for BioDesignBench."""

import logging
import os
import re
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput
from biodesignbench.tasks.schema import Task, TaskTier

logger = logging.getLogger(__name__)

_AIDE_SYSTEM_PROMPT = """You are an ML-focused bioinformatics agent specializing in:
- Machine learning for biological data (protein, genomic, molecular)
- Systematic code solution search with metric feedback
- Data preprocessing and feature engineering for bio datasets
- Model training, evaluation, and result analysis

Write a complete Python script to solve the given task.
Use standard libraries: biopython, numpy, pandas, scipy, scikit-learn.
Output ONLY the Python code, no explanations."""


class BioMLAgent(AgentInterface):
    """
    AIDE agent wrapper (registered as 'bioml-agent' for backward compatibility).

    AIDE (WecoAI/aideml) uses tree search over code solutions with metric
    feedback. When the aideml package is installed, it wraps the real agent.
    Otherwise, it falls back to an LLM with ML-focused system prompt +
    Docker sandbox execution.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 300,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("AIDE_MODEL", "claude-sonnet-4-20250514")
        self.timeout_seconds = timeout_seconds
        self._api_calls = 0

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="bioml-agent",
            name="AIDE (BioML)",
            version="1.0.0",
            description="ML-focused agent using tree search over code solutions",
            provider="weco",
            model=self.model,
            is_bio_specific=True,
            capabilities=[
                "code_execution",
                "ml_workflow",
                "tree_search",
                "metric_feedback",
            ],
        )

    def setup(self) -> None:
        try:
            import aide  # noqa: F401
        except ImportError:
            raise ImportError(
                "aideml package required: pip install aideml. "
                "Fallback to plain OpenAI API is disabled because it would be "
                "identical to gpt5-tools, making the comparison meaningless."
            )

    def teardown(self) -> None:
        self._api_calls = 0

    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="aide_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        self._provision_inputs(task, output_dir)

        # Require real AIDE package -- fallback to OpenAI is meaningless for benchmarking
        try:
            return await self._solve_with_aide(task, output_dir)
        except ImportError:
            raise ImportError(
                "aideml package required: pip install aideml. "
                "Fallback to plain OpenAI API is disabled because it would be "
                "identical to gpt5-tools, making the comparison meaningless."
            )

    async def _solve_with_aide(self, task: Task, output_dir: Path) -> AgentOutput:
        """Solve using the real AIDE (aideml) package."""
        import aide

        prompt = self.get_standard_prompt(task)
        self._api_calls += 1

        experiment = aide.Experiment(
            data_dir=str(output_dir),
            goal=prompt,
            eval="code",
        )
        # aide.Experiment.run(steps) returns Solution(code=..., valid_metric=...)
        solution = experiment.run(steps=10)

        code = solution.code if solution else ""
        output_files = [
            f.name
            for f in output_dir.iterdir()
            if f.name != "script.py" and f.is_file()
        ]
        artifacts = [str(output_dir / f) for f in output_files]

        return AgentOutput(
            code=code,
            artifacts=artifacts,
            output_dir=str(output_dir),
            tools_used=["aide"],
            api_calls=self._api_calls,
            iterations=10,
            reasoning_trace="",
        )

    async def _fallback_solve(self, task: Task, output_dir: Path) -> AgentOutput:
        """Fallback: LLM + ML-focused system prompt + DockerExecutor."""
        from biodesignbench.sandbox.executor import DockerExecutor

        executor = DockerExecutor(timeout_seconds=self.timeout_seconds)
        prompt = self.get_standard_prompt(task)

        code = self._generate_code(prompt)
        if not code:
            return AgentOutput(output_dir=str(output_dir), api_calls=self._api_calls)

        result = executor.execute_in_dir(code=code, output_dir=output_dir)
        output_files = result.output_files
        artifacts = [str(output_dir / f) for f in output_files]

        return AgentOutput(
            code=code,
            artifacts=artifacts,
            output_dir=str(output_dir),
            tools_used=["aide-fallback"],
            api_calls=self._api_calls,
            iterations=1,
        )

    def _generate_code(self, prompt: str) -> str:
        """Generate code using LLM backend."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            self._api_calls += 1
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _AIDE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            code = response.choices[0].message.content or ""
            match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
            return match.group(1) if match else code
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return ""

