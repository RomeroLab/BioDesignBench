"""Claude Code agent wrapper for BioDesignBench."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput, ToolCallEntry, summarize_tool_args
from biodesignbench.tasks.schema import Task, TaskTier

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(AgentInterface):
    """
    Claude Code agent using Anthropic's Claude with tool use.

    Uses the Anthropic SDK to interact with Claude models that support
    tool use for code execution and file operations. Code is executed
    in a Docker sandbox (or subprocess fallback).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_seconds: int = 300,
        tool_provider: Any | None = None,
        tool_mode: str | None = None,
        system_prompt_suffix: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.tool_provider = tool_provider
        self.tool_mode = tool_mode
        self.system_prompt_suffix = system_prompt_suffix
        self.client = None
        self._api_calls = 0

    def get_info(self) -> AgentInfo:
        agent_id = "claude-code"
        if self.tool_mode:
            agent_id = f"claude-code-{self.tool_mode}"
        return AgentInfo(
            agent_id=agent_id,
            name="Claude Code",
            version="1.0.0",
            description="Claude with tool use for code execution",
            provider="anthropic",
            model=self.model,
            is_bio_specific=False,
            capabilities=["code_execution", "tool_use", "file_operations"],
        )

    def setup(self) -> None:
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def teardown(self) -> None:
        self.client = None
        self._api_calls = 0

    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        if self.client is None:
            self.setup()

        from biodesignbench.sandbox.executor import DockerExecutor

        executor = DockerExecutor(timeout_seconds=self.timeout_seconds)

        # Setup output directory
        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="claude_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        _solve_start = time.monotonic()
        _total_input_tokens = 0
        _total_output_tokens = 0

        # Provision input files via symlinks
        self._provision_inputs(task, output_dir)

        prompt = self.get_standard_prompt(task)
        tools = self._get_tools()
        system_prompt = self._get_system_prompt(task, output_dir)
        if self.system_prompt_suffix:
            system_prompt += "\n\n" + self.system_prompt_suffix

        messages = [{"role": "user", "content": prompt}]
        tools_used = []
        tool_call_log: list[ToolCallEntry] = []
        iterations = 0
        max_iterations = 50
        generated_code = ""
        all_output_files = []

        while iterations < max_iterations:
            iterations += 1
            self._api_calls += 1

            api_kwargs: dict[str, Any] = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                tools=tools,
            )
            if system_prompt:
                api_kwargs["system"] = system_prompt
            response = self.client.messages.create(**api_kwargs)

            if hasattr(response, "usage") and response.usage:
                _total_input_tokens += getattr(response.usage, "input_tokens", 0)
                _total_output_tokens += getattr(response.usage, "output_tokens", 0)

            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    code_blocks = self._extract_code_blocks(block.text)
                    if code_blocks:
                        generated_code = code_blocks[-1]
                    assistant_content.append(block)
                elif block.type == "tool_use":
                    tools_used.append(block.name)
                    assistant_content.append(block)

            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "tool_use":
                tool_results = await self._execute_tools(
                    assistant_content, executor, output_dir, all_output_files,
                    tool_call_log, iterations,
                )
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        # If we have generated_code but never executed it, execute now
        if generated_code and not all_output_files:
            result = executor.execute_in_dir(code=generated_code, output_dir=output_dir)
            all_output_files.extend(result.output_files)

        artifacts = [str(output_dir / f) for f in all_output_files]

        _wall_time = round(time.monotonic() - _solve_start, 3)
        _tool_exec = round(sum(e.duration_seconds for e in tool_call_log), 3)

        if task.tier == TaskTier.TIER1:
            return AgentOutput(
                code=generated_code,
                artifacts=artifacts,
                output_dir=str(output_dir),
                tools_used=list(set(tools_used)),
                tool_call_log=tool_call_log,
                api_calls=self._api_calls,
                iterations=iterations,
                reasoning_trace=self._extract_reasoning(messages),
                wall_time_seconds=_wall_time,
                tool_execution_seconds=_tool_exec,
                total_input_tokens=_total_input_tokens,
                total_output_tokens=_total_output_tokens,
            )
        else:
            designs = self._extract_designs_from_dir(output_dir)
            return AgentOutput(
                designs=designs,
                output_dir=str(output_dir),
                tools_used=list(set(tools_used)),
                tool_call_log=tool_call_log,
                api_calls=self._api_calls,
                iterations=iterations,
                reasoning_trace=self._extract_reasoning(messages),
                wall_time_seconds=_wall_time,
                tool_execution_seconds=_tool_exec,
                total_input_tokens=_total_input_tokens,
                total_output_tokens=_total_output_tokens,
            )

    def _get_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "execute_python",
                "description": "Execute Python code in a sandbox. Code runs in /workspace with access to input files. Output files should be written to the current directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "read_file",
                "description": "Read contents of a file in the working directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"}
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file in the working directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        ]
        if self.tool_provider:
            tools.extend(self.tool_provider.get_tool_definitions_anthropic(mode=self.tool_mode))
        return tools

    async def _execute_tools(
        self,
        content: list[Any],
        executor: Any,
        output_dir: Path,
        all_output_files: list[str],
        tool_call_log: list[ToolCallEntry],
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        for block in content:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue

            tool_input = block.input
            tool_name = block.name

            _tool_start = time.monotonic()

            if tool_name == "execute_python":
                exec_result = executor.execute_in_dir(
                    code=tool_input["code"], output_dir=output_dir
                )
                all_output_files.extend(exec_result.output_files)
                response_text = (
                    f"Exit code: {exec_result.return_code}\n"
                    f"Stdout:\n{exec_result.stdout[:4000]}\n"
                    f"Stderr:\n{exec_result.stderr[:2000]}\n"
                    f"Output files: {exec_result.output_files}"
                )
            elif tool_name == "read_file":
                filepath = output_dir / tool_input["path"]
                try:
                    response_text = filepath.read_text()[:8000]
                except Exception as e:
                    response_text = f"Error reading file: {e}"
            elif tool_name == "write_file":
                filepath = output_dir / tool_input["path"]
                try:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_text(tool_input["content"])
                    if filepath.name not in all_output_files:
                        all_output_files.append(filepath.name)
                    response_text = f"File written: {tool_input['path']}"
                except Exception as e:
                    response_text = f"Error writing file: {e}"
            elif self.tool_provider and self.tool_provider.is_tool_available(tool_name, mode=self.tool_mode):
                result = await self.tool_provider.call_tool(tool_name, tool_input, output_dir, mode=self.tool_mode)
                response_text = json.dumps(result, indent=2)[:8000]
            else:
                response_text = f"Unknown tool: {tool_name}"

            _tool_duration = round(time.monotonic() - _tool_start, 3)

            # Record structured tool call for orchestration scoring
            is_error = response_text.startswith("Error") or "Error" in response_text[:50]
            summary, values = summarize_tool_args(tool_input)
            tool_call_log.append(ToolCallEntry(
                tool=tool_name,
                iteration=iteration,
                success=not is_error,
                error=response_text[:200] if is_error else None,
                args_summary=summary,
                args_values=values,
                duration_seconds=_tool_duration,
            ))

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": response_text,
                }
            )

        return results

    def _extract_reasoning(self, messages: list[dict[str, Any]]) -> str:
        reasoning = []
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "type") and block.type == "text":
                            text = getattr(block, "text", "")
                            if isinstance(text, str):
                                reasoning.append(text)
                elif isinstance(content, str):
                    reasoning.append(content)
        return "\n\n".join(reasoning)
