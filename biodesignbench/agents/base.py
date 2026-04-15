"""Base agent interface."""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from biodesignbench.tasks.schema import Task


class AgentInfo(BaseModel):
    """Agent metadata."""

    agent_id: str
    name: str
    version: str
    description: str = ""
    provider: str = ""  # e.g., "anthropic", "openai", "custom"
    model: str = ""  # e.g., "claude-3-opus", "gpt-4"
    is_bio_specific: bool = False  # True for bio-specialized agents
    capabilities: list[str] = []  # e.g., ["code_execution", "web_search"]


class ToolCallEntry(BaseModel):
    """Single tool call record for orchestration/error recovery scoring."""

    tool: str
    iteration: int = 1
    success: bool = True
    error: str | None = None
    args_summary: dict[str, Any] = {}  # Key args (not full content)
    args_values: dict[str, Any] = {}   # Numeric/bool values preserved
    duration_seconds: float = 0.0


def summarize_tool_args(args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build args_summary (types) and args_values (numeric/bool actuals).

    - int, float, bool → stored as-is in args_values
    - short str (≤60 chars, no newlines) → stored as-is (paths, enum values)
    - long str / bytes → type + length
    - list → stored if all elements are simple scalars, else type + length
    - everything else → type name only
    """
    summary: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for k, v in args.items():
        summary[k] = type(v).__name__
        if isinstance(v, (int, float, bool)):
            values[k] = v
        elif isinstance(v, str):
            if len(v) <= 60 and "\n" not in v:
                values[k] = v
            else:
                values[k] = f"str(len={len(v)})"
        elif isinstance(v, (list, tuple)):
            if len(v) <= 20 and all(isinstance(x, (int, float, str, bool)) for x in v):
                values[k] = list(v)
            else:
                values[k] = f"list(len={len(v)})"
        # dicts and other complex types → skip (type only in summary)
    return summary, values


class AgentOutput(BaseModel):
    """Standard output format from agent."""

    # For coding tasks
    code: str = ""
    artifacts: list[str] = []  # Generated file paths
    output_dir: str = ""  # Directory containing output artifacts

    # For design tasks
    designs: list[dict[str, Any]] = []

    # Execution metadata
    tools_used: list[str] = []
    tool_call_log: list[ToolCallEntry] = []  # Ordered log for orchestration scoring
    api_calls: int = 0
    iterations: int = 1
    reasoning_trace: str = ""  # Optional: agent's reasoning

    # Cost/time instrumentation (Experiment E)
    wall_time_seconds: float = 0.0
    tool_execution_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class AgentInterface(ABC):
    """
    Base interface that all agents must implement.

    Example:
        class MyAgent(AgentInterface):
            def get_info(self) -> AgentInfo:
                return AgentInfo(
                    agent_id="my-agent",
                    name="My Custom Agent",
                    version="1.0.0",
                )

            async def solve(self, task: Task, output_dir: Path | None = None) -> AgentOutput:
                # Agent implementation
                return AgentOutput(code="print('hello')")
    """

    @abstractmethod
    def get_info(self) -> AgentInfo:
        """
        Return agent metadata.

        Returns:
            AgentInfo with agent details
        """
        pass

    @abstractmethod
    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        """
        Solve a benchmark task.

        Args:
            task: Task object with description, inputs, constraints
            output_dir: Directory for output artifacts. If None, a temp dir is used.
            input_dir: Directory containing input files. If None, uses InputManager.

        Returns:
            AgentOutput with results and execution metadata

        Raises:
            Exception: If agent fails to complete task
        """
        pass

    def _provision_inputs(
        self, task: Task, output_dir: Path, *, use_symlinks: bool = True
    ) -> list[Path]:
        """Provision task input files into the output directory.

        Replaces per-agent _copy_input_files(). Uses InputManager for
        centralized path resolution and symlink-based provisioning.
        """
        from biodesignbench.eval.input_manager import InputManager

        return InputManager().provision_inputs(task, output_dir, use_symlinks=use_symlinks)

    def setup(self) -> None:
        """
        Optional setup before running tasks.
        Override to initialize resources, load models, etc.
        """
        pass

    def teardown(self) -> None:
        """
        Optional cleanup after running tasks.
        Override to release resources.
        """
        pass

    def get_standard_prompt(self, task: Task) -> str:
        """Load the standardized prompt for a task.

        All agents should use this to get the canonical task description,
        ensuring fair comparison across different agent implementations.

        Args:
            task: Task object (uses task.task_id and task.tier to find the prompt file).

        Returns:
            Markdown-formatted prompt string.
        """
        from biodesignbench.tasks.prompt_generator import load_prompt

        return load_prompt(task.task_id, tier=task.tier.value)

    # ------------------------------------------------------------------
    # Shared helpers — subclasses inherit these; override only if needed.
    # ------------------------------------------------------------------

    def _get_system_prompt(
        self,
        task: Task | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> str:
        """Build mode-aware system prompt with PDB file hints."""
        from biodesignbench.tools.protein_design_provider import get_system_guidance

        base = (
            "You are a bioinformatics expert. Write Python code to solve bioinformatics "
            "problems. Ensure your code is executable and produces the required output files. "
            "Use the execute_python tool to run code and verify results."
        )
        tool_provider = getattr(self, "tool_provider", None)
        tool_mode = getattr(self, "tool_mode", None)
        if tool_provider:
            base += "\n\n" + get_system_guidance(mode=tool_mode)
            if output_dir:
                pdb_files = [
                    f.name for f in Path(output_dir).iterdir()
                    if f.is_file() and f.suffix == ".pdb"
                ] if Path(output_dir).exists() else []
                if pdb_files:
                    abs_paths = [str(Path(output_dir).resolve() / f) for f in pdb_files]
                    base += (
                        f"\n\nThe following PDB files are available locally and should be "
                        f"passed as absolute paths to protein design tools (e.g. target_pdb): "
                        f"{', '.join(abs_paths)}"
                    )
        return base

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract Python code blocks from markdown-formatted text."""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        return re.findall(pattern, text, re.DOTALL)

    @staticmethod
    def _extract_designs_from_dir(output_dir: Path | None) -> list[dict[str, Any]]:
        """Extract designs from designed_sequences.fasta + metrics.json on disk."""
        designs: list[dict[str, Any]] = []
        if output_dir is None:
            return designs

        output_dir = Path(output_dir)
        fasta_path = output_dir / "designed_sequences.fasta"
        metrics_path = output_dir / "metrics.json"

        # Parse FASTA sequences
        sequences: dict[str, str] = {}
        if fasta_path.exists():
            current_name = ""
            current_seq_parts: list[str] = []
            for line in fasta_path.read_text().splitlines():
                line = line.strip()
                if line.startswith(">"):
                    if current_name and current_seq_parts:
                        sequences[current_name] = "".join(current_seq_parts)
                    current_name = line[1:].split()[0]
                    current_seq_parts = []
                elif line:
                    current_seq_parts.append(line)
            if current_name and current_seq_parts:
                sequences[current_name] = "".join(current_seq_parts)

        # Parse metrics
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                pass

        # Combine into design dicts
        for name, seq in sequences.items():
            design: dict[str, Any] = {"name": name, "sequence": seq}
            if name in metrics:
                design["metrics"] = metrics[name]
            designs.append(design)

        return designs


class DummyAgent(AgentInterface):
    """Dummy agent for testing."""

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="dummy",
            name="Dummy Agent",
            version="0.0.1",
            description="A dummy agent for testing",
        )

    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        """Return empty output for any task."""
        return AgentOutput(
            code="# Dummy implementation\npass",
            iterations=1,
        )
