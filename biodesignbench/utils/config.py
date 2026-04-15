"""Configuration loader for BioDesignBench."""

import os
from pathlib import Path


def load_env(env_file: str | Path | None = None) -> None:
    """Load environment variables from .env file.

    Args:
        env_file: Path to .env file. Defaults to .env in project root.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    if env_file is None:
        # Walk up from this file to find project root .env
        env_file = Path(__file__).resolve().parents[2] / ".env"

    if Path(env_file).exists():
        load_dotenv(env_file, override=False)


def get_api_key(provider: str) -> str | None:
    """Get API key for a provider from environment.

    Args:
        provider: One of 'anthropic', 'openai', 'google'.

    Returns:
        API key string or None if not set.
    """
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {provider}. Expected: {list(key_map.keys())}")
    return os.environ.get(env_var)


def get_sandbox_config() -> dict:
    """Get Docker sandbox configuration from environment.

    Returns:
        Dict with sandbox configuration.
    """
    return {
        "image": os.environ.get("SANDBOX_IMAGE", "biodesignbench-sandbox"),
        "timeout": int(os.environ.get("SANDBOX_TIMEOUT", "300")),
        "memory_limit": os.environ.get("SANDBOX_MEMORY_LIMIT", "4g"),
    }


def get_bio_agent_model(agent: str) -> str:
    """Get the LLM backend model for a bio-specific agent.

    Args:
        agent: One of 'biomni', 'stella', 'aide'.

    Returns:
        Model identifier string (e.g., 'gpt-4o').
    """
    model_map = {
        "biomni": "BIOMNI_MODEL",
        "stella": "STELLA_MODEL",
        "aide": "AIDE_MODEL",
    }
    env_var = model_map.get(agent)
    if env_var is None:
        raise ValueError(f"Unknown bio agent: {agent}. Expected: {list(model_map.keys())}")
    return os.environ.get(env_var, "gpt-4o")
