"""Ground truth loader for Tier 1 tasks."""

import json
from pathlib import Path
from typing import Any

# Default ground truth directory
GROUND_TRUTH_DIR = Path(__file__).resolve().parents[3] / "data" / "tier1" / "ground_truth"


def load_ground_truth(task_id: str, ground_truth_dir: Path | None = None) -> dict[str, Any]:
    """Load ground truth data for a task.

    Args:
        task_id: Task identifier (e.g., 'format_001').
        ground_truth_dir: Override directory. Defaults to data/tier1/ground_truth/.

    Returns:
        Ground truth dict, or empty dict if file not found.
    """
    gt_dir = ground_truth_dir or GROUND_TRUTH_DIR
    gt_path = gt_dir / f"{task_id}.json"

    if not gt_path.exists():
        return {}

    with open(gt_path) as f:
        return json.load(f)


def save_ground_truth(
    task_id: str,
    data: dict[str, Any],
    ground_truth_dir: Path | None = None,
) -> Path:
    """Save ground truth data for a task.

    Args:
        task_id: Task identifier.
        data: Ground truth dict to save.
        ground_truth_dir: Override directory.

    Returns:
        Path to saved file.
    """
    gt_dir = ground_truth_dir or GROUND_TRUTH_DIR
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / f"{task_id}.json"

    with open(gt_path, "w") as f:
        json.dump(data, f, indent=2)

    return gt_path
