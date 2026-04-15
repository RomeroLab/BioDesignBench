"""Bridge between EvaluationPipeline and pytest for Tier 1 tests.

Runs pytest on task-specific test files and parses results into
EvaluationResult fields with partial credit scoring.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


class Tier1TestRunner:
    """Runs pytest on Tier 1 test files and collects results."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.tests_dir = self.project_root / "tests" / "tier1"

    def run_tests(
        self,
        task_id: str,
        output_dir: Path,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        """Run pytest for a specific task and return structured results.

        Args:
            task_id: Task identifier (e.g., 'format_001').
            output_dir: Directory containing agent's output files.
            timeout_seconds: Maximum time for test execution.

        Returns:
            Dict with: passed, failed, skipped, errors, total, partial_score,
            test_results (per-test pass/fail), output (raw pytest output).
        """
        test_file = self.tests_dir / f"test_{task_id}.py"

        result = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "total": 0,
            "partial_score": 0.0,
            "test_results": {},
            "output": "",
            "error_message": None,
        }

        if not test_file.exists():
            result["error_message"] = f"Test file not found: {test_file}"
            return result

        # Set environment variable for dual-mode fixture
        env = os.environ.copy()
        env["BIODESIGNBENCH_OUTPUT_DIR"] = str(output_dir)

        try:
            proc = subprocess.run(
                [
                    "python", "-m", "pytest",
                    str(test_file),
                    "-v",
                    "--tb=short",
                    "-q",
                    "--no-header",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(self.project_root),
                env=env,
            )

            result["output"] = proc.stdout + proc.stderr

            # Parse pytest output for test results
            result["test_results"] = self._parse_test_results(proc.stdout)

            # Parse summary line (e.g., "5 passed, 2 failed, 1 skipped")
            summary = self._parse_summary(proc.stdout + proc.stderr)
            result.update(summary)

            # Calculate partial score
            result["partial_score"] = self._calculate_partial_score(result)

        except subprocess.TimeoutExpired:
            result["error_message"] = f"Tests timed out after {timeout_seconds}s"
        except Exception as e:
            result["error_message"] = str(e)

        return result

    def _parse_test_results(self, output: str) -> dict[str, bool]:
        """Parse per-test pass/fail from verbose pytest output."""
        results = {}
        # Match lines like: test_file.py::TestClass::test_name PASSED
        pattern = re.compile(r"::(\w+)::(\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)")
        for match in pattern.finditer(output):
            class_name, test_name, status = match.groups()
            key = f"{class_name}::{test_name}"
            results[key] = status == "PASSED"
        return results

    def _parse_summary(self, output: str) -> dict[str, int]:
        """Parse pytest summary line for counts."""
        summary = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0}

        # Match patterns like "5 passed", "2 failed", "1 skipped"
        for status in ["passed", "failed", "skipped", "error"]:
            match = re.search(rf"(\d+)\s+{status}", output)
            if match:
                key = "errors" if status == "error" else status
                summary[key] = int(match.group(1))

        summary["total"] = (
            summary["passed"] + summary["failed"]
            + summary["skipped"] + summary["errors"]
        )
        return summary

    def _calculate_partial_score(self, result: dict[str, Any]) -> float:
        """Calculate 0-100 partial score from test results.

        Scoring: (passed / (passed + failed)) * 100
        Skipped tests are excluded from denominator.
        """
        scoreable = result["passed"] + result["failed"]
        if scoreable == 0:
            return 0.0
        return round(result["passed"] / scoreable * 100, 1)

    def get_test_file(self, task_id: str) -> Path | None:
        """Get the test file path for a task, or None if not found."""
        test_file = self.tests_dir / f"test_{task_id}.py"
        return test_file if test_file.exists() else None

    def list_available_tests(self) -> list[str]:
        """List all task IDs that have test files."""
        task_ids = []
        for f in sorted(self.tests_dir.glob("test_*.py")):
            # Extract task_id from test_<task_id>.py
            task_id = f.stem[5:]  # Remove 'test_' prefix
            task_ids.append(task_id)
        return task_ids
