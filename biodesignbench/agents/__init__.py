"""Agent interfaces for BioDesignBench."""

from functools import partial

from biodesignbench.interventions import FORCED_DEPTH_PROMPT, LOW_DIVERSITY_CONTROL_PROMPT

from biodesignbench.agents.base import (
    AgentInfo,
    AgentInterface,
    AgentOutput,
    DummyAgent,
)

# General-purpose LLM agents
from biodesignbench.agents.general_purpose import (
    ClaudeCodeAgent,
    DeepSeekAgent,
    GeminiToolsAgent,
    GPT5ToolsAgent,
    OpenInterpreterAgent,
)

# Bio-specific agents
from biodesignbench.agents.bio_specific import (
    BioMLAgent,
    BiomniAgent,
    STELLAAgent,
)

# Baselines
from biodesignbench.agents.baselines import (
    HardcodedPipelineAgent,
    HumanExpertAgent,
    HumanExpertBaseline,
    HumanTraineeBaseline,
    ScriptedPipelineBaseline,
)

__all__ = [
    # Base classes
    "AgentInterface",
    "AgentInfo",
    "AgentOutput",
    "DummyAgent",
    # General-purpose LLM agents
    "ClaudeCodeAgent",
    "DeepSeekAgent",
    "GPT5ToolsAgent",
    "GeminiToolsAgent",
    "OpenInterpreterAgent",
    # Bio-specific agents
    "BiomniAgent",
    "STELLAAgent",
    "BioMLAgent",
    # Baselines
    "ScriptedPipelineBaseline",
    "HumanExpertBaseline",
    "HumanTraineeBaseline",
    "HardcodedPipelineAgent",
    "HumanExpertAgent",
]

# Runnable agents (make API calls, execute code)
AGENT_REGISTRY = {
    # General-purpose (mode inherits from --tool-mode flag)
    "claude-code": ClaudeCodeAgent,
    "gpt5-tools": GPT5ToolsAgent,
    "gpt5.2-reasoning": partial(GPT5ToolsAgent, reasoning_effort="high"),
    "gemini-tools": GeminiToolsAgent,
    "open-interpreter": OpenInterpreterAgent,
    # General-purpose — benchmark mode (atomic tools only)
    "claude-code-benchmark": partial(ClaudeCodeAgent, tool_mode="benchmark"),
    "gpt5-tools-benchmark": partial(GPT5ToolsAgent, tool_mode="benchmark"),
    "gpt5.2-reasoning-benchmark": partial(GPT5ToolsAgent, reasoning_effort="high", tool_mode="benchmark"),
    "gemini-tools-benchmark": partial(GeminiToolsAgent, tool_mode="benchmark"),
    # General-purpose — user mode (all tools including composites)
    "claude-code-user": partial(ClaudeCodeAgent, tool_mode="user"),
    "gpt5-tools-user": partial(GPT5ToolsAgent, tool_mode="user"),
    "gpt5.2-reasoning-user": partial(GPT5ToolsAgent, reasoning_effort="high", tool_mode="user"),
    "gemini-tools-user": partial(GeminiToolsAgent, tool_mode="user"),
    # --- New agents (2026-03) ---
    # Claude Sonnet 4.5
    "sonnet-4.5-tools": partial(ClaudeCodeAgent, model="claude-sonnet-4-5-20250929"),
    "sonnet-4.5-tools-benchmark": partial(ClaudeCodeAgent, model="claude-sonnet-4-5-20250929", tool_mode="benchmark"),
    "sonnet-4.5-tools-user": partial(ClaudeCodeAgent, model="claude-sonnet-4-5-20250929", tool_mode="user"),
    # Gemini 2.5 Pro via Vertex AI (thinking off for fair comparison)
    "gemini-2.5-pro-tools": partial(GeminiToolsAgent, model="gemini-2.5-pro", thinking_budget=0),
    "gemini-2.5-pro-tools-benchmark": partial(GeminiToolsAgent, model="gemini-2.5-pro", thinking_budget=0, tool_mode="benchmark"),
    "gemini-2.5-pro-tools-user": partial(GeminiToolsAgent, model="gemini-2.5-pro", thinking_budget=0, tool_mode="user"),
    # Gemini 3 Pro via Vertex AI global region (thinking off for fair comparison)
    "gemini-3-pro-tools": partial(GeminiToolsAgent, model="gemini-3-pro-preview", thinking_budget=0, location="global"),
    "gemini-3-pro-tools-benchmark": partial(GeminiToolsAgent, model="gemini-3-pro-preview", thinking_budget=0, location="global", tool_mode="benchmark"),
    "gemini-3-pro-tools-user": partial(GeminiToolsAgent, model="gemini-3-pro-preview", thinking_budget=0, location="global", tool_mode="user"),
    # DeepSeek V3
    "deepseek-v3-tools": DeepSeekAgent,
    "deepseek-v3-tools-benchmark": partial(DeepSeekAgent, tool_mode="benchmark"),
    "deepseek-v3-tools-user": partial(DeepSeekAgent, tool_mode="user"),
    # --- Intervention experiments (2026-04) ---
    "deepseek-v3-forced-depth": partial(
        DeepSeekAgent, tool_mode="benchmark",
        system_prompt_suffix=FORCED_DEPTH_PROMPT,
    ),
    "deepseek-v3-low-diversity": partial(
        DeepSeekAgent, tool_mode="benchmark",
        system_prompt_suffix=LOW_DIVERSITY_CONTROL_PROMPT,
    ),
    "gpt5-tools-forced-depth": partial(
        GPT5ToolsAgent, tool_mode="benchmark",
        system_prompt_suffix=FORCED_DEPTH_PROMPT,
    ),
    "gpt5-tools-low-diversity": partial(
        GPT5ToolsAgent, tool_mode="benchmark",
        system_prompt_suffix=LOW_DIVERSITY_CONTROL_PROMPT,
    ),
    # Bio-specific
    "biomni": BiomniAgent,
    "stella": STELLAAgent,
    "bioml-agent": BioMLAgent,
    # Dummy for testing
    "dummy": DummyAgent,
}

# Base IDs for general-purpose agents (without mode suffix)
_BASE_GENERAL_PURPOSE = {
    "claude-code", "gpt5-tools", "gpt5.2-reasoning", "gemini-tools",
    "sonnet-4.5-tools", "gemini-2.5-pro-tools", "deepseek-v3-tools",
}

# Baselines (pre-computed results, not run during benchmark)
BASELINE_REGISTRY = {
    "scripted-baseline": ScriptedPipelineBaseline,
    "human-expert": HumanExpertBaseline,
    "human-trainee": HumanTraineeBaseline,
    "hardcoded-pipeline": HardcodedPipelineAgent,
    "human-expert-agent": HumanExpertAgent,
    "human-expert-shallow": partial(HumanExpertAgent, sampling_config_name="shallow"),
}


def get_agent(agent_id: str, **kwargs) -> AgentInterface:
    """
    Get agent instance by ID.

    Args:
        agent_id: Agent identifier (e.g., "claude-code", "biomni")
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        Initialized agent instance

    Raises:
        ValueError: If agent_id is not found in registry
    """
    all_registry = {**AGENT_REGISTRY, **BASELINE_REGISTRY}
    if agent_id not in all_registry:
        available = ", ".join(sorted(all_registry.keys()))
        raise ValueError(f"Unknown agent: {agent_id}. Available: {available}")
    return all_registry[agent_id](**kwargs)


def list_agents(include_baselines: bool = False) -> list[str]:
    """List available agent IDs.

    Args:
        include_baselines: If True, include pre-computed baselines.
    """
    agents = sorted(AGENT_REGISTRY.keys())
    if include_baselines:
        agents.extend(sorted(BASELINE_REGISTRY.keys()))
    return agents
