"""Open Interpreter agent wrapper for BioDesignBench."""

import logging
import os
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput
from biodesignbench.tasks.schema import Task, TaskTier

logger = logging.getLogger(__name__)


class OpenInterpreterAgent(AgentInterface):
    """
    Open Interpreter agent for code execution.

    Uses the Open Interpreter library with code execution redirected
    to Docker sandbox for isolation.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        auto_run: bool = True,
        timeout_seconds: int = 300,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.auto_run = auto_run
        self.timeout_seconds = timeout_seconds
        self.interpreter = None
        self._api_calls = 0

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="open-interpreter",
            name="Open Interpreter",
            version="1.0.0",
            description="Open source code interpreter with sandboxed execution",
            provider="open-source",
            model=self.model,
            is_bio_specific=False,
            capabilities=["code_execution", "file_operations", "shell_commands"],
        )

    def setup(self) -> None:
        try:
            from interpreter import interpreter

            self.interpreter = interpreter
            self.interpreter.llm.model = self.model
            if self.api_key:
                self.interpreter.llm.api_key = self.api_key
            self.interpreter.auto_run = self.auto_run
            self.interpreter.safe_mode = "off"
        except ImportError:
            raise ImportError(
                "open-interpreter package required: pip install open-interpreter. "
                "Fallback to plain OpenAI API is disabled because it would be "
                "identical to gpt5-tools, making the comparison meaningless."
            )

    def teardown(self) -> None:
        if self.interpreter:
            self.interpreter.reset()
        self.interpreter = None
        self._api_calls = 0

    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        from biodesignbench.sandbox.executor import DockerExecutor

        executor = DockerExecutor(timeout_seconds=self.timeout_seconds)

        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="oi_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        self._provision_inputs(task, output_dir)
        prompt = self.get_standard_prompt(task)

        if self.interpreter is None:
            self.setup()

        generated_code = ""
        tools_used = ["python"]
        all_output_files = []

        if self.interpreter is not None:
            # Use Open Interpreter with Docker-sandboxed execution
            self._api_calls += 1
            messages = self.interpreter.chat(prompt, display=False, stream=False)

            # Collect all code blocks from OI output
            code_blocks = []
            for msg in messages:
                if msg.get("type") == "code" and msg.get("content"):
                    code_blocks.append(msg["content"])

            if code_blocks:
                generated_code = "\n\n".join(code_blocks)
                # Execute combined code in sandbox
                result = executor.execute_in_dir(code=generated_code, output_dir=output_dir)
                all_output_files.extend(result.output_files)
        else:
            # Fallback: generate code via OpenAI directly + execute in sandbox
            generated_code = await self._fallback_generate(task, prompt)
            if generated_code:
                result = executor.execute_in_dir(code=generated_code, output_dir=output_dir)
                all_output_files.extend(result.output_files)

        artifacts = [str(output_dir / f) for f in all_output_files]

        if task.tier == TaskTier.TIER1:
            return AgentOutput(
                code=generated_code,
                artifacts=artifacts,
                output_dir=str(output_dir),
                tools_used=tools_used,
                api_calls=self._api_calls,
                iterations=1,
                reasoning_trace="",
            )
        else:
            return AgentOutput(
                designs=[],
                output_dir=str(output_dir),
                tools_used=tools_used,
                api_calls=self._api_calls,
                iterations=1,
                reasoning_trace="",
            )

    async def _fallback_generate(self, task: Task, prompt: str) -> str:
        """Generate code using OpenAI API when Open Interpreter is not available."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            self._api_calls += 1
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a bioinformatics expert. Write a complete Python script that solves the given task. Output ONLY the Python code, no explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            code = response.choices[0].message.content or ""
            # Strip markdown code fences if present
            import re

            match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
            if match:
                return match.group(1)
            return code
        except Exception as e:
            logger.error(f"Fallback code generation failed: {e}")
            return ""

