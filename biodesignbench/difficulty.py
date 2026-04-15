"""Task difficulty classification for BioDesignBench.

Provides a DifficultyLevel enum and utilities for classifying tasks
by difficulty based on metadata, constraints, and taxonomy category.

Difficulty is determined by:
1. Explicit ``difficulty`` field in task JSON metadata (if present)
2. Heuristic classification based on task category and constraints
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from biodesignbench.taxonomy import get_category


class DifficultyLevel(str, Enum):
    """Task difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Heuristic rules for difficulty classification
# ---------------------------------------------------------------------------

# Tasks explicitly tagged in task JSONs or templates
_EXPLICIT_DIFFICULTY: dict[str, DifficultyLevel] = {}

# Category-based defaults (cells that tend to be harder)
_HARD_CATEGORIES = {"cpx_sig", "cpx_str", "cfd_enz", "cfd_sig"}
_EASY_CATEGORIES = {"sqo_str", "sqo_flu", "dnk_str"}


def _classify_heuristic(task_id: str, metadata: dict[str, Any] | None = None) -> DifficultyLevel:
    """Classify difficulty heuristically from task ID and metadata."""
    # Check explicit difficulty in metadata
    if metadata:
        explicit = metadata.get("difficulty")
        if explicit:
            try:
                return DifficultyLevel(explicit.lower())
            except ValueError:
                pass

    # Check category-based heuristic
    category = get_category(task_id)
    if category:
        cat_id = category.category_id
        if cat_id in _HARD_CATEGORIES:
            return DifficultyLevel.HARD
        if cat_id in _EASY_CATEGORIES:
            return DifficultyLevel.EASY

    # Default to medium
    return DifficultyLevel.MEDIUM


def get_difficulty(task_id: str, metadata: dict[str, Any] | None = None) -> DifficultyLevel:
    """Get difficulty level for a task.

    Args:
        task_id: Task identifier (old or new format).
        metadata: Optional task metadata dict for explicit difficulty field.

    Returns:
        DifficultyLevel enum value.
    """
    if task_id in _EXPLICIT_DIFFICULTY:
        return _EXPLICIT_DIFFICULTY[task_id]
    return _classify_heuristic(task_id, metadata)


def get_difficulty_distribution(task_dir: Path | None = None) -> dict[str, int]:
    """Count tasks by difficulty level across all task JSONs.

    Args:
        task_dir: Directory containing task JSONs. Defaults to ``tasks/tier2/``.

    Returns:
        Dict mapping difficulty level name to count.
    """
    if task_dir is None:
        task_dir = Path("tasks/tier2")

    dist: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}

    if not task_dir.exists():
        return dist

    for json_file in sorted(task_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            metadata = data.get("metadata", {})
            task_id = data.get("task_id", json_file.stem)
        except (json.JSONDecodeError, KeyError):
            task_id = json_file.stem
            metadata = {}

        level = get_difficulty(task_id, metadata)
        dist[level.value] += 1

    return dist
