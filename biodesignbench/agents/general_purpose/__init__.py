"""General-purpose LLM agents."""

from biodesignbench.agents.general_purpose.claude_code import ClaudeCodeAgent
from biodesignbench.agents.general_purpose.deepseek_tools import DeepSeekAgent
from biodesignbench.agents.general_purpose.gemini_tools import GeminiToolsAgent
from biodesignbench.agents.general_purpose.gpt5_tools import GPT5ToolsAgent
from biodesignbench.agents.general_purpose.open_interpreter import OpenInterpreterAgent

__all__ = [
    "ClaudeCodeAgent",
    "DeepSeekAgent",
    "GPT5ToolsAgent",
    "GeminiToolsAgent",
    "OpenInterpreterAgent",
]
