"""Base test class for all 75 Tier 1 tasks.

Provides common fixtures, validation helpers, and scoring infrastructure.
Individual test classes inherit from BaseTier1Test and override class attributes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from biodesignbench.eval.tier1.ground_truth import load_ground_truth
from biodesignbench.eval.tier1.scoring import (
    ScoringRubric,
    calculate_task_score,
    partial_credit_range,
    score_artifact,
)
from biodesignbench.eval.tier1 import validators


class BaseTier1Test:
    """Base class for Tier 1 task tests.

    Subclasses MUST set:
        TASK_ID: str - the task identifier (e.g., 'format_001')
        EXPECTED_ARTIFACTS: list[str] - expected output filenames

    Subclasses MAY override:
        SCORING_RUBRIC: dict - custom point allocation
        GROUND_TRUTH: dict - inline ground truth (overrides file-based)
    """

    TASK_ID: ClassVar[str] = ""
    EXPECTED_ARTIFACTS: ClassVar[list[str]] = []
    SCORING_RUBRIC: ClassVar[dict[str, int]] = {
        "script_executes": 25,
        "primary_artifact": 25,
        "secondary_artifacts": 25,
        "content_correctness": 25,
    }
    GROUND_TRUTH: ClassVar[dict[str, Any]] = {}

    @pytest.fixture
    def task_id(self) -> str:
        return self.TASK_ID

    @pytest.fixture
    def ground_truth(self) -> dict[str, Any]:
        """Load ground truth from file, with class-level override."""
        if self.GROUND_TRUTH:
            return self.GROUND_TRUTH
        return load_ground_truth(self.TASK_ID)

    @pytest.fixture
    def rubric(self) -> ScoringRubric:
        return ScoringRubric(components=dict(self.SCORING_RUBRIC))

    def get_output_path(self, output_dir: Path, filename: str) -> Path:
        """Get path to an expected output file."""
        return output_dir / filename

    def assert_file_exists(self, output_dir: Path, filename: str) -> Path:
        """Assert a file exists and return its path."""
        path = self.get_output_path(output_dir, filename)
        assert path.exists(), f"Expected output file not found: {filename}"
        return path

    def calculate_score(self, output_dir: Path) -> dict[str, Any]:
        """Calculate the full scoring rubric for this task.

        Override in subclasses for task-specific scoring.
        Default implementation checks artifact existence.
        """
        rubric = ScoringRubric(components=dict(self.SCORING_RUBRIC))
        results: dict[str, int] = {}

        # script_executes: check if any artifacts were produced
        artifacts_exist = any(
            (output_dir / a).exists() for a in self.EXPECTED_ARTIFACTS
        )
        results["script_executes"] = rubric.components["script_executes"] if artifacts_exist else 0

        # primary_artifact: validate first artifact
        if self.EXPECTED_ARTIFACTS:
            primary_path = output_dir / self.EXPECTED_ARTIFACTS[0]
            if primary_path.exists():
                results["primary_artifact"] = rubric.components["primary_artifact"]
            else:
                results["primary_artifact"] = 0
        else:
            results["primary_artifact"] = 0

        # secondary_artifacts: validate remaining artifacts
        secondary = self.EXPECTED_ARTIFACTS[1:]
        if secondary:
            found = sum(1 for a in secondary if (output_dir / a).exists())
            ratio = found / len(secondary)
            results["secondary_artifacts"] = int(
                rubric.components["secondary_artifacts"] * ratio
            )
        else:
            results["secondary_artifacts"] = rubric.components["secondary_artifacts"]

        # content_correctness: subclasses override for specific checks
        results["content_correctness"] = 0

        return calculate_task_score(rubric, results)
