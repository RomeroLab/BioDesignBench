"""Ground truth loader for Tier 2 design tasks.

Loads evaluation thresholds and reference sequences from
data/tier2/ground_truth/ JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default ground truth directory
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_GT_DIR = _PROJECT_ROOT / "data" / "tier2" / "ground_truth"


def load_ground_truth(
    task_id: str,
    ground_truth_dir: Path | None = None,
) -> dict[str, Any]:
    """Load ground truth data for a Tier 2 task.

    Args:
        task_id: Task identifier (e.g., 'binder_001').
        ground_truth_dir: Override directory. Defaults to data/tier2/ground_truth/.

    Returns:
        Parsed ground truth dict.

    Raises:
        FileNotFoundError: If ground truth file doesn't exist.
    """
    gt_dir = ground_truth_dir or _DEFAULT_GT_DIR
    gt_path = gt_dir / f"{task_id}.json"

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    with open(gt_path) as f:
        return json.load(f)


def get_evaluation_thresholds(
    task_id: str,
    ground_truth_dir: Path | None = None,
) -> dict[str, float]:
    """Extract evaluation_thresholds from ground truth.

    Returns:
        Dict mapping threshold names to values.
        Returns empty dict if thresholds not present.
    """
    gt = load_ground_truth(task_id, ground_truth_dir)
    return gt.get("evaluation_thresholds", {})


def get_reference_sequence(
    task_id: str,
    task_data: dict[str, Any] | None = None,
) -> str | None:
    """Get the wild-type/reference sequence for a task.

    First checks task_data['target']['sequence'], then falls back
    to the ground truth file's known_sequence field if available.

    Args:
        task_id: Task identifier.
        task_data: Pre-loaded task JSON dict. If None, attempts to
            load from tasks/tier2/<task_id>.json.

    Returns:
        Reference sequence string or None.
    """
    # Try task data first
    if task_data is not None:
        target = task_data.get("target", {})
        seq = target.get("sequence")
        if seq:
            return seq

    # Try loading task JSON
    task_dir = _PROJECT_ROOT / "tasks" / "tier2"
    task_path = task_dir / f"{task_id}.json"
    if task_path.exists():
        with open(task_path) as f:
            data = json.load(f)
        seq = data.get("target", {}).get("sequence")
        if seq:
            return seq

    return None
