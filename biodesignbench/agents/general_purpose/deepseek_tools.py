"""DeepSeek V3 agent wrapper for BioDesignBench."""

import logging
import os
from typing import Any

from biodesignbench.agents.base import AgentInfo
from biodesignbench.agents.general_purpose.gpt5_tools import GPT5ToolsAgent

logger = logging.getLogger(__name__)

# DeepSeek uses OpenAI-compatible API
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekAgent(GPT5ToolsAgent):
    """
    DeepSeek V3 agent using OpenAI-compatible API.

    DeepSeek provides an OpenAI-compatible endpoint, so this class
    reuses GPT5ToolsAgent logic with a different base_url and defaults.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        max_tokens: int = 16384,
        temperature: float = 0.0,
        timeout_seconds: int = 300,
        tool_provider: Any | None = None,
        tool_mode: str | None = None,
        system_prompt_suffix: str | None = None,
    ):
        super().__init__(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            tool_provider=tool_provider,
            reasoning_effort=None,
            tool_mode=tool_mode,
            system_prompt_suffix=system_prompt_suffix,
        )

    def get_info(self) -> AgentInfo:
        agent_id = "deepseek-v3-tools"
        if self.tool_mode:
            agent_id = f"deepseek-v3-tools-{self.tool_mode}"
        return AgentInfo(
            agent_id=agent_id,
            name="DeepSeek V3 + Tools",
            version="1.0.0",
            description="DeepSeek V3 with function calling via OpenAI-compatible API",
            provider="deepseek",
            model=self.model,
            is_bio_specific=False,
            capabilities=["code_execution", "function_calling", "file_operations"],
        )

    def setup(self) -> None:
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=_DEEPSEEK_BASE_URL,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def _get_user_identifier(self, task: Any) -> str:
        """DeepSeek doesn't require user tracking; return a simple ID."""
        return f"biodesignbench:{task.task_id}"
