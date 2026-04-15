"""STELLA agent wrapper for BioDesignBench."""

import logging
import os
import re
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput
from biodesignbench.tasks.schema import Task, TaskTier

logger = logging.getLogger(__name__)

_STELLA_SYSTEM_PROMPT = """You are a scientific literature-informed AI agent specializing in
computational biology. You approach problems by:
1. Considering relevant scientific literature and established methods
2. Using well-cited bioinformatics tools and databases
3. Following best practices from published protocols

Your expertise includes:
- Literature-informed sequence analysis and structure prediction
- Database integration (PubMed, UniProt, PDB, NCBI)
- Reproducible computational biology workflows
- Scientific data format handling (FASTA, PDB, CSV, Newick)

Write a complete Python script to solve the given task.
Use standard libraries: biopython, requests, numpy, pandas, scipy.
Output ONLY the Python code, no explanations."""


class STELLAAgent(AgentInterface):
    """
    STELLA agent wrapper - Scientific Literature Agent.

    STELLA (zaixizhang/STELLA) is a multi-agent system with Manager,
    Developer, and Critic roles for scientific tasks. When the stella
    package is installed, it wraps the real agent. Otherwise, it falls
    back to an LLM with a literature-focused system prompt + Docker sandbox.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 300,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("STELLA_MODEL", "gpt-4o")
        self.timeout_seconds = timeout_seconds
        self._api_calls = 0

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="stella",
            name="STELLA",
            version="1.0.0",
            description="Scientific literature agent with citation-aware reasoning",
            provider="stella",
            model=self.model,
            is_bio_specific=True,
            capabilities=[
                "literature_search",
                "citation_analysis",
                "knowledge_synthesis",
                "code_execution",
            ],
        )

    def setup(self) -> None:
        try:
            import stella  # noqa: F401
        except ImportError:
            raise ImportError(
                "STELLA package required: pip install stella. "
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

            output_dir = Path(tempfile.mkdtemp(prefix="stella_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        self._provision_inputs(task, output_dir)

        # Require real STELLA package -- fallback to OpenAI is meaningless for benchmarking
        try:
            return await self._solve_with_stella(task, output_dir)
        except ImportError:
            raise ImportError(
                "STELLA package required: pip install stella. "
                "Fallback to plain OpenAI API is disabled because it would be "
                "identical to gpt5-tools, making the comparison meaningless."
            )

    async def _solve_with_stella(self, task: Task, output_dir: Path) -> AgentOutput:
        """Solve using the real STELLA package."""
        from stella import STELLAOrchestrator

        prompt = self.get_standard_prompt(task)
        self._api_calls += 1

        orchestrator = STELLAOrchestrator(model=self.model)
        result = orchestrator.run(prompt, output_dir=str(output_dir))

        code = result.get("code", "")
        output_files = [
            f.name for f in output_dir.iterdir() if f.name != "script.py" and f.is_file()
        ]
        artifacts = [str(output_dir / f) for f in output_files]

        return AgentOutput(
            code=code,
            artifacts=artifacts,
            output_dir=str(output_dir),
            tools_used=["stella"],
            api_calls=self._api_calls,
            iterations=result.get("iterations", 1),
            reasoning_trace=result.get("reasoning", ""),
        )

    async def _fallback_solve(self, task: Task, output_dir: Path) -> AgentOutput:
        """Fallback: LLM + literature-focused system prompt + DockerExecutor."""
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
            tools_used=["stella-fallback"],
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
                    {"role": "system", "content": _STELLA_SYSTEM_PROMPT},
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

