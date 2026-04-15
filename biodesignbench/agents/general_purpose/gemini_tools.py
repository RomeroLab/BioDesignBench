"""Gemini with tools agent wrapper for BioDesignBench."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput, ToolCallEntry, summarize_tool_args
from biodesignbench.tasks.schema import Task, TaskTier

logger = logging.getLogger(__name__)


class GeminiToolsAgent(AgentInterface):
    """
    Gemini agent using Google's genai SDK with Vertex AI support.

    Uses the google-genai SDK to interact with Gemini models that support
    function declarations for tool use. Code is executed in a Docker sandbox.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_seconds: int = 300,
        tool_provider: Any | None = None,
        tool_mode: str | None = None,
        thinking_budget: int | None = None,
        project: str | None = None,
        location: str | None = None,
        system_prompt_suffix: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        # project=None → read env var; project="" → force API key mode (no Vertex AI)
        self.project = project if project is not None else os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.tool_provider = tool_provider
        self.tool_mode = tool_mode
        self.thinking_budget = thinking_budget
        self.system_prompt_suffix = system_prompt_suffix
        self.client = None
        self._api_calls = 0

    def get_info(self) -> AgentInfo:
        agent_id = "gemini-tools"
        if self.tool_mode:
            agent_id = f"gemini-tools-{self.tool_mode}"
        return AgentInfo(
            agent_id=agent_id,
            name="Gemini + Tools",
            version="2.0.0",
            description="Gemini with function declarations for tool use (Vertex AI)",
            provider="google",
            model=self.model_name,
            is_bio_specific=False,
            capabilities=["code_execution", "function_calling", "multimodal", "file_operations"],
        )

    def setup(self) -> None:
        try:
            from google import genai

            if self.project:
                # Vertex AI mode
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
            else:
                # Google AI Studio mode (API key)
                self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )

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

        from google.genai import types
        from biodesignbench.sandbox.executor import DockerExecutor

        executor = DockerExecutor(timeout_seconds=self.timeout_seconds)

        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="gemini_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        self._provision_inputs(task, output_dir)

        prompt = self.get_standard_prompt(task)
        system_guidance = self._get_system_prompt(task, output_dir)
        if self.system_prompt_suffix:
            system_guidance += "\n\n" + self.system_prompt_suffix
        tools_config = self._get_tools_config()

        config_kwargs: dict[str, Any] = {
            "tools": tools_config,
            "temperature": self.temperature,
        }
        if system_guidance:
            config_kwargs["system_instruction"] = system_guidance
        if self.thinking_budget and self.thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        tools_used = []
        tool_call_log: list[ToolCallEntry] = []
        iterations = 0
        max_iterations = 50
        generated_code = ""
        all_output_files = []
        reasoning_parts: list[str] = []

        self._api_calls += 1
        response = self._send_with_retry(chat, prompt)
        iterations += 1

        while iterations < max_iterations:
            has_function_call = False
            function_responses = []

            parts = (
                response.candidates[0].content.parts
                if response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
                else []
            )
            for part in parts:
                if part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    tools_used.append(fc.name)
                    args = dict(fc.args) if fc.args else {}
                    tool_response = await self._execute_function(
                        fc.name, args, executor, output_dir, all_output_files,
                        tool_call_log, iterations,
                    )
                    function_responses.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"result": tool_response},
                        )
                    )
                elif part.text:
                    reasoning_parts.append(part.text)
                    code_blocks = self._extract_code_blocks(part.text)
                    if code_blocks:
                        generated_code = code_blocks[-1]

            if not has_function_call:
                break

            if function_responses:
                self._api_calls += 1
                response = self._send_with_retry(chat, function_responses)
                iterations += 1

        # Execute any final generated code if not yet executed
        if generated_code and not all_output_files:
            result = executor.execute_in_dir(code=generated_code, output_dir=output_dir)
            all_output_files.extend(result.output_files)

        artifacts = [str(output_dir / f) for f in all_output_files]
        reasoning_trace = "\n\n".join(reasoning_parts)

        if task.tier == TaskTier.TIER1:
            return AgentOutput(
                code=generated_code,
                artifacts=artifacts,
                output_dir=str(output_dir),
                tools_used=list(set(tools_used)),
                tool_call_log=tool_call_log,
                api_calls=self._api_calls,
                iterations=iterations,
                reasoning_trace=reasoning_trace,
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
                reasoning_trace=reasoning_trace,
            )

    @staticmethod
    def _send_with_retry(chat: Any, message: Any, max_retries: int = 5) -> Any:
        """Send a message with exponential backoff on rate limit errors."""
        delay = 2.0
        for attempt in range(max_retries):
            try:
                return chat.send_message(message)
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    logger.warning(f"Rate limited (attempt {attempt + 1}), waiting {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise

    async def _execute_function(
        self,
        name: str,
        args: dict[str, Any],
        executor: Any,
        output_dir: Path,
        all_output_files: list[str],
        tool_call_log: list[ToolCallEntry],
        iteration: int,
    ) -> str:
        """Execute a function call and return response string."""
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
            result = await self.tool_provider.call_tool(name, args, output_dir, mode=self.tool_mode)
            response_text = json.dumps(result, indent=2)[:8000]
        else:
            response_text = f"Unknown tool: {name}"

        # Record structured tool call for orchestration scoring
        is_error = response_text.startswith("Error") or "Error" in response_text[:50]
        summary, values = summarize_tool_args(args)
        tool_call_log.append(ToolCallEntry(
            tool=name,
            iteration=iteration,
            success=not is_error,
            error=response_text[:200] if is_error else None,
            args_summary=summary,
            args_values=values,
        ))

        return response_text

    def _get_tools_config(self) -> list[Any]:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name="execute_python",
                description="Execute Python code in a sandbox",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "code": types.Schema(
                            type="STRING",
                            description="Python code to execute",
                        )
                    },
                    required=["code"],
                ),
            ),
            types.FunctionDeclaration(
                name="read_file",
                description="Read contents of a file",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "path": types.Schema(
                            type="STRING",
                            description="Path to the file",
                        )
                    },
                    required=["path"],
                ),
            ),
            types.FunctionDeclaration(
                name="write_file",
                description="Write content to a file",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "path": types.Schema(
                            type="STRING",
                            description="Path to the file",
                        ),
                        "content": types.Schema(
                            type="STRING",
                            description="Content to write",
                        ),
                    },
                    required=["path", "content"],
                ),
            ),
        ]
        # Merge all function declarations into a single Tool object.
        # Gemini ignores tools in the second Tool group when multiple
        # Tool objects are provided, so all declarations must live together.
        if self.tool_provider:
            mcp_tools = self.tool_provider.get_tool_definitions_gemini(mode=self.tool_mode)
            for tool_obj in mcp_tools:
                if hasattr(tool_obj, "function_declarations") and tool_obj.function_declarations:
                    declarations.extend(tool_obj.function_declarations)
        return [types.Tool(function_declarations=declarations)]
