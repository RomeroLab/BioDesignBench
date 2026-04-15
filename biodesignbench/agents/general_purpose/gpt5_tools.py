"""GPT-5 with tools agent wrapper for BioDesignBench."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput, ToolCallEntry, summarize_tool_args
from biodesignbench.tasks.schema import DesignTask, Task, TaskTier

logger = logging.getLogger(__name__)


class GPT5ToolsAgent(AgentInterface):
    """
    GPT-5 agent using OpenAI's function calling API.

    Uses the OpenAI SDK to interact with GPT-5 models that support
    function calling for tool use. Code is executed in a Docker sandbox.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_seconds: int = 300,
        tool_provider: Any | None = None,
        reasoning_effort: str | None = None,
        tool_mode: str | None = None,
        system_prompt_suffix: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.tool_provider = tool_provider
        self.reasoning_effort = reasoning_effort
        self.tool_mode = tool_mode
        self.system_prompt_suffix = system_prompt_suffix
        self.client = None
        self._api_calls = 0

    def get_info(self) -> AgentInfo:
        if self.reasoning_effort:
            agent_id = "gpt5.2-reasoning"
            name = f"GPT-5.2 Reasoning ({self.reasoning_effort})"
        else:
            agent_id = "gpt5-tools"
            name = "GPT-5 + Tools"
        if self.tool_mode:
            agent_id = f"{agent_id}-{self.tool_mode}"
        return AgentInfo(
            agent_id=agent_id,
            name=name,
            version="1.0.0",
            description="GPT-5 with function calling for tool use",
            provider="openai",
            model=self.model,
            is_bio_specific=False,
            capabilities=["code_execution", "function_calling", "file_operations"],
        )

    def setup(self) -> None:
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

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

        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="gpt5_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        _solve_start = time.monotonic()
        _total_input_tokens = 0
        _total_output_tokens = 0

        # Copy (not symlink) so files survive inside Docker sandbox containers
        self._provision_inputs(task, output_dir, use_symlinks=False)

        # Build list of locally available input files for the system prompt
        local_files = [
            f.name for f in output_dir.iterdir()
            if f.is_file() and f.suffix in (".pdb", ".fasta", ".fa", ".json", ".csv")
        ]

        prompt = self.get_standard_prompt(task)
        tools = self._get_tools()

        system_prompt = self._get_system_prompt(task, output_dir, local_files)
        if self.system_prompt_suffix:
            system_prompt += "\n\n" + self.system_prompt_suffix
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
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
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=self.max_tokens,
                user=self._get_user_identifier(task),
            )
            if self.reasoning_effort:
                # Reasoning models don't support temperature — only default (1) is allowed
                api_kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                api_kwargs["temperature"] = self.temperature
            response = self.client.chat.completions.create(**api_kwargs)

            if response.usage:
                _total_input_tokens += response.usage.prompt_tokens or 0
                _total_output_tokens += response.usage.completion_tokens or 0

            assistant_message = response.choices[0].message

            if assistant_message.content:
                code_blocks = self._extract_code_blocks(assistant_message.content)
                if code_blocks:
                    generated_code = code_blocks[-1]

            messages.append(assistant_message.model_dump())

            # If response was truncated, skip tool call processing
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning("Response truncated (finish_reason=length), skipping tool calls")
                break

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tools_used.append(tool_call.function.name)
                    tool_response, tool_duration = await self._execute_tool(
                        tool_call, executor, output_dir, all_output_files
                    )
                    # Record structured tool call for orchestration scoring
                    is_error = tool_response.startswith("Error") or "Error" in tool_response[:50]
                    try:
                        args = json.loads(tool_call.function.arguments)
                        summary, values = summarize_tool_args(args)
                    except json.JSONDecodeError:
                        summary, values = {"_raw": "str"}, {}
                    tool_call_log.append(ToolCallEntry(
                        tool=tool_call.function.name,
                        iteration=iterations,
                        success=not is_error,
                        error=tool_response[:200] if is_error else None,
                        args_summary=summary,
                        args_values=values,
                        duration_seconds=tool_duration,
                    ))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_response,
                        }
                    )
            else:
                break

        # Execute any final generated code if not yet executed
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

    def _get_system_prompt(
        self,
        task: Task | None = None,
        output_dir: Path | None = None,
        local_files: list[str] | None = None,
    ) -> str:
        from biodesignbench.tools.protein_design_provider import get_system_guidance

        base = (
            "You are a bioinformatics expert. Write Python code to solve bioinformatics "
            "problems. Use standard libraries like biopython, requests, numpy, and pandas. "
            "Ensure your code is executable and produces the required output files. "
            "Use the execute_python tool to run code and verify results."
        )
        if self.tool_provider:
            base += "\n\n" + get_system_guidance(mode=self.tool_mode)
            # Tell the agent the absolute path for MCP tool args (target_pdb etc.)
            if output_dir and local_files:
                pdb_files = [f for f in local_files if f.endswith(".pdb")]
                if pdb_files:
                    abs_paths = [str(Path(output_dir).resolve() / f) for f in pdb_files]
                    base += (
                        f"\n\nThe following PDB files are available locally and should be "
                        f"passed as absolute paths to protein design tools (e.g. target_pdb): "
                        f"{', '.join(abs_paths)}"
                    )
        # Always mention local files so the agent doesn't try to download
        if local_files:
            base += (
                f"\n\nLocal input files available in the working directory: "
                f"{', '.join(local_files)}. "
                f"Do NOT download these files from the internet - they are already available."
            )
        return base

    def _get_user_identifier(self, task: Task) -> str:
        """Generate a stable, hashed user identifier for OpenAI safety tracking.

        OpenAI requires a unique `user` parameter per end-user to help detect
        and prevent abuse. Since BioDesignBench is a benchmark harness (not a
        multi-user app), we hash the runner identity + task ID to produce a
        stable, non-PII identifier.
        """
        raw = os.environ.get("BIODESIGNBENCH_USER_ID", os.environ.get("USER", "bench"))
        raw = f"{raw}:{task.task_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _get_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code in a sandbox. Write output files to the current directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"}
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file in the working directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file in the working directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]
        if self.tool_provider:
            tools.extend(self.tool_provider.get_tool_definitions_openai(mode=self.tool_mode))
        return tools

    async def _execute_tool(
        self,
        tool_call: Any,
        executor: Any,
        output_dir: Path,
        all_output_files: list[str],
    ) -> tuple[str, float]:
        """Execute a single tool call and return (response_string, duration_seconds)."""
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed tool call JSON for {tool_call.function.name}: {e}")
            return f"Error: malformed function arguments (truncated JSON): {e}", 0.0

        name = tool_call.function.name
        _tool_start = time.monotonic()

        if name == "execute_python":
            result = executor.execute_in_dir(code=args["code"], output_dir=output_dir)
            all_output_files.extend(result.output_files)
            response_text = (
                f"Exit code: {result.return_code}\n"
                f"Stdout:\n{result.stdout[:4000]}\n"
                f"Stderr:\n{result.stderr[:2000]}\n"
                f"Output files: {result.output_files}"
            )
        elif name == "read_file":
            filepath = output_dir / args["path"]
            try:
                response_text = filepath.read_text()[:8000]
            except Exception as e:
                response_text = f"Error reading file: {e}"
        elif name == "write_file":
            filepath = output_dir / args["path"]
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(args["content"])
                if filepath.name not in all_output_files:
                    all_output_files.append(filepath.name)
                response_text = f"File written: {args['path']}"
            except Exception as e:
                response_text = f"Error writing file: {e}"
        elif self.tool_provider and self.tool_provider.is_tool_available(name, mode=self.tool_mode):
            # Resolve relative PDB paths to absolute so MCP Docker container can find them
            for key in ("target_pdb", "complex_pdb", "expected_structure"):
                if key in args and args[key] and not Path(args[key]).is_absolute():
                    candidate = output_dir / args[key]
                    if candidate.exists():
                        args[key] = str(candidate.resolve())
            result = await self.tool_provider.call_tool(name, args, output_dir, mode=self.tool_mode)
            response_text = json.dumps(result, indent=2)[:8000]
        else:
            response_text = f"Unknown tool: {name}"

        _duration = round(time.monotonic() - _tool_start, 3)
        return response_text, _duration

    def _extract_reasoning(self, messages: list[dict[str, Any]]) -> str:
        reasoning = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    reasoning.append(content)
        return "\n\n".join(reasoning)
