"""Biomni agent wrapper for BioDesignBench."""

import contextlib
import logging
import os
import re
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput
from biodesignbench.tasks.schema import Task, TaskTier

logger = logging.getLogger(__name__)

_BIOMNI_SYSTEM_PROMPT = """You are a biomedical AI agent with deep expertise in molecular biology,
genomics, structural biology, and bioinformatics. You specialize in:
- Protein sequence analysis and structure prediction
- Genomic data retrieval and processing (NCBI, UniProt, PDB)
- Biological file format handling (FASTA, PDB, GFF, VCF)
- Molecular visualization and analysis
- Biological database querying

Write a complete Python script to solve the given bioinformatics task.
Use standard libraries: biopython, requests, numpy, pandas, scipy.
Output ONLY the Python code, no explanations."""


class BiomniAgent(AgentInterface):
    """
    Biomni agent wrapper.

    Biomni (snap-stanford/Biomni) is a bio-specific agent with specialized
    biological tools and knowledge. When the biomni package is installed,
    it wraps the real agent. Otherwise, it falls back to an LLM with a
    bio-domain system prompt + Docker sandbox execution.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 300,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("BIOMNI_MODEL", "gpt-4o")
        self.timeout_seconds = timeout_seconds
        self._api_calls = 0
        self._biomni_available = None
        self._a1_class = None  # Cached A1 class (loaded once)
        self._a1_instance = None  # Cached A1 instance (avoids re-downloading data lake)

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="biomni",
            name="Biomni",
            version="1.0.0",
            description="Multi-modal bio agent with biological domain expertise",
            provider="stanford",
            model=self.model,
            is_bio_specific=True,
            capabilities=[
                "code_execution",
                "biological_knowledge",
                "structure_analysis",
                "sequence_analysis",
                "literature_search",
            ],
        )

    def setup(self) -> None:
        try:
            from biomni.agent import A1

            self._a1_class = A1
        except ImportError as e:
            raise ImportError(
                f"biomni package required (needs Python>=3.11): "
                f"pip install git+https://github.com/snap-stanford/Biomni.git@main. "
                f"Original error: {e}"
            )

    def teardown(self) -> None:
        self._api_calls = 0
        self._a1_class = None
        self._a1_instance = None

    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="biomni_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        self._provision_inputs(task, output_dir)

        # Require real biomni package -- fallback to OpenAI is meaningless for benchmarking
        try:
            return await self._solve_with_biomni(task, output_dir)
        except ImportError:
            raise ImportError(
                "biomni package required: pip install biomni>=0.0.6. "
                "Fallback to plain OpenAI API is disabled because it would be "
                "identical to gpt5-tools, making the comparison meaningless."
            )

    async def _solve_with_biomni(self, task: Task, output_dir: Path) -> AgentOutput:
        """Solve using the real biomni package (A1 instance cached from setup).

        Contamination caveat
        ---------------------
        Biomni runs in the host process (not inside Docker) and retains its
        ``literature_search`` capability, meaning it can potentially access
        the source papers/issues that tasks are derived from. Mitigations:

        1. The ``task`` passed here is **sanitized** by the pipeline
           (source, DOI, tools_expected, tags, test_file, ground_truth
           are all redacted before the agent sees them).
        2. The standardized prompt includes **anti-contamination
           instructions** telling the agent not to look up known solutions.
        3. After the agent finishes, the pipeline runs **post-hoc
           contamination detection** on reasoning traces and code,
           flagging DOI patterns, GitHub issue URLs, and source paper
           name/author references.

        Full network isolation is **not** enforced for Biomni because its
        ``literature_search`` tool legitimately needs HTTP access.
        """
        if self._a1_class is None:
            from biomni.agent import A1

            self._a1_class = A1

        # Reuse the cached A1 instance to avoid re-downloading the data lake.
        # Only create a new instance on the first call.
        if self._a1_instance is None:
            self._a1_instance = self._a1_class(path=str(output_dir), llm=self.model)
        else:
            # Update the working directory for this task
            self._a1_instance.path = str(output_dir)

        prompt = self.get_standard_prompt(task)
        self._api_calls += 1

        # Run A1 with cwd set to output_dir so any files it writes
        # (outside its own path parameter) land in the task output, not the project root.
        prev_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            result = self._a1_instance.go(prompt)
        finally:
            os.chdir(prev_cwd)

        # Collect output files produced by biomni
        output_files = [
            f.name
            for f in output_dir.iterdir()
            if f.name != "script.py" and f.is_file()
        ]
        artifacts = [str(output_dir / f) for f in output_files]

        code = ""
        reasoning = ""
        if isinstance(result, dict):
            code = result.get("code", "")
            reasoning = result.get("reasoning", str(result))
        elif isinstance(result, str):
            reasoning = result

        return AgentOutput(
            code=code,
            artifacts=artifacts,
            output_dir=str(output_dir),
            tools_used=["biomni"],
            api_calls=self._api_calls,
            iterations=1,
            reasoning_trace=reasoning,
        )

    async def _fallback_solve(self, task: Task, output_dir: Path) -> AgentOutput:
        """Fallback: LLM + bio system prompt + DockerExecutor."""
        from biodesignbench.sandbox.executor import DockerExecutor

        executor = DockerExecutor(timeout_seconds=self.timeout_seconds)
        prompt = self.get_standard_prompt(task)

        # Generate code via LLM
        code = self._generate_code(prompt)
        if not code:
            return AgentOutput(output_dir=str(output_dir), api_calls=self._api_calls)

        # Execute in sandbox
        result = executor.execute_in_dir(code=code, output_dir=output_dir)
        output_files = result.output_files
        artifacts = [str(output_dir / f) for f in output_files]

        return AgentOutput(
            code=code,
            artifacts=artifacts,
            output_dir=str(output_dir),
            tools_used=["biomni-fallback"],
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
                    {"role": "system", "content": _BIOMNI_SYSTEM_PROMPT},
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

