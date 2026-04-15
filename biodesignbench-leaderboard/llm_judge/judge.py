"""Single LLM judge: wraps one API call to evaluate a design attempt.

Supports Anthropic, OpenAI, Google, and DeepSeek providers.
In dry_run mode, returns deterministic midpoint scores without API calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_judge.rubrics import (
    JUDGE_DIMENSIONS,
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)


def _midpoint_scores() -> dict[str, dict[str, Any]]:
    """Return deterministic midpoint scores for dry-run mode."""
    result = {}
    for dim, info in JUDGE_DIMENSIONS.items():
        mid = info["max_score"] // 2
        if info["max_score"] % 2 == 1 and mid * 2 < info["max_score"]:
            # For odd max (5, 3), floor division gives correct 50%
            pass
        result[dim] = {
            "reasoning": f"[Dry run] Midpoint score for {dim}.",
            "score": mid,
        }
    return result


def parse_judge_response(raw_text: str) -> dict[str, dict[str, Any]]:
    """Parse LLM judge response into structured scores.

    Handles:
    - Direct JSON response
    - JSON inside markdown code blocks
    - Out-of-range score clamping
    - Invalid JSON fallback to midpoint scores

    Args:
        raw_text: Raw LLM response text.

    Returns:
        Dict mapping dimension names to {reasoning, score}.
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw_text

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try finding any JSON object in the text
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
            except json.JSONDecodeError:
                return _midpoint_scores()
        else:
            return _midpoint_scores()

    # Validate and clamp scores
    result = {}
    for dim, info in JUDGE_DIMENSIONS.items():
        if dim in data and isinstance(data[dim], dict):
            score = data[dim].get("score", info["max_score"] // 2)
            if isinstance(score, (int, float)):
                score = max(0, min(score, info["max_score"]))
            else:
                score = info["max_score"] // 2
            reasoning = data[dim].get("reasoning", "")
            result[dim] = {"reasoning": str(reasoning), "score": score}
        else:
            # Missing dimension — use midpoint
            result[dim] = {
                "reasoning": f"[Fallback] Dimension {dim} missing from judge response.",
                "score": info["max_score"] // 2,
            }

    return result


class LLMJudge:
    """Single LLM judge that evaluates a protein design attempt.

    Args:
        provider: API provider ('anthropic', 'openai', 'google', 'deepseek').
        model: Model identifier string.
        dry_run: If True, return deterministic scores without API calls.
        api_key: Optional API key override.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        dry_run: bool = False,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.dry_run = dry_run
        self.api_key = api_key
        self.api_calls = 0
        self._client = None

    def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is not None:
            return self._client

        import os

        if self.provider == "anthropic":
            import anthropic

            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)
        elif self.provider == "openai":
            from openai import OpenAI

            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key)
        elif self.provider == "google":
            from google import genai

            key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key)
        elif self.provider == "deepseek":
            from openai import OpenAI

            key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
            self._client = OpenAI(
                api_key=key, base_url="https://api.deepseek.com"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def _call_api(self, system: str, user: str) -> str:
        """Make a single API call and return raw text response."""
        client = self._get_client()
        self.api_calls += 1

        if self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text

        elif self.provider in ("openai", "deepseek"):
            # GPT-5+ uses max_completion_tokens; older models use max_tokens
            token_param = (
                "max_completion_tokens" if "gpt-5" in self.model or "o3" in self.model or "o4" in self.model
                else "max_tokens"
            )
            response = client.chat.completions.create(
                model=self.model,
                **{token_param: 4096},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content

        elif self.provider == "google":
            response = client.models.generate_content(
                model=self.model,
                contents=f"{system}\n\n{user}",
            )
            return response.text

        raise ValueError(f"Unsupported provider: {self.provider}")

    def evaluate_sync(
        self,
        task_description: str,
        tool_call_log: list[dict[str, Any]],
        designed_sequences: list[str],
        algorithmic_metrics: dict[str, Any],
        reference_pipeline: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate a design attempt synchronously.

        Args:
            task_description: Original task prompt.
            tool_call_log: Agent's tool call sequence.
            designed_sequences: Designed protein sequences.
            algorithmic_metrics: Computed biophysical metrics.
            reference_pipeline: Expected expert pipeline.

        Returns:
            Dict mapping dimension names to {reasoning, score}.
        """
        if self.dry_run:
            return _midpoint_scores()

        prompt = build_judge_prompt(
            task_description=task_description,
            tool_call_log=tool_call_log,
            designed_sequences=designed_sequences,
            algorithmic_metrics=algorithmic_metrics,
            reference_pipeline=reference_pipeline,
        )

        raw_response = self._call_api(JUDGE_SYSTEM_PROMPT, prompt)
        return parse_judge_response(raw_response)
