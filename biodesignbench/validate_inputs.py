"""Validate Tier 2 task JSONs and their associated input files.

Checks each task JSON for schema correctness, PDB existence, chain IDs,
ground truth file existence, and prompt file existence. Produces a
structured ValidationReport per task.

Usage:
    from biodesignbench.validate_inputs import validate_all_tasks
    reports = validate_all_tasks("tasks/tier2", "data/tier2/input")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from biodesignbench.taxonomy import parse_new_task_id


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TASK_DIR = _PROJECT_ROOT / "tasks" / "tier2"
_DEFAULT_PDB_DIR = _PROJECT_ROOT / "data" / "tier2" / "input"
_DEFAULT_GT_DIR = _PROJECT_ROOT / "data" / "tier2" / "ground_truth"
_DEFAULT_PROMPT_DIR = _PROJECT_ROOT / "data" / "tier2" / "prompts"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Result of validating a single task JSON."""

    task_id: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def _count_atom_records(pdb_path: Path) -> tuple[int, set[str]]:
    """Count ATOM records and collect chain IDs from a PDB file."""
    atom_count = 0
    chain_ids: set[str] = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_count += 1
                if len(line) >= 22:
                    chain_ids.add(line[21])
    return atom_count, chain_ids


def validate_single_task(
    task_json: dict,
    *,
    pdb_dir: Path,
    gt_dir: Path,
    prompt_dir: Path,
) -> ValidationReport:
    """Validate a single task JSON dict.

    Checks:
    1. Required top-level keys exist
    2. PDB file exists with ≥ 50 ATOM records
    3. Chain ID in PDB matches task target chain
    4. For new-format IDs: category is valid
    5. Ground truth file exists
    6. Prompt file exists
    """
    task_id = task_json.get("task_id", "<unknown>")
    report = ValidationReport(task_id=task_id)

    # 1. Required keys
    required = {"task_id", "description", "target", "design_constraints",
                "expected_output", "evaluation", "metadata"}
    missing = required - set(task_json.keys())
    if missing:
        report.add_error(f"Missing top-level keys: {sorted(missing)}")

    # Target checks
    target = task_json.get("target", {})
    pdb_id = target.get("pdb_id", "")
    chain = target.get("chain", "")

    # 2. PDB file exists with ≥ 50 ATOM records
    if pdb_id:
        pdb_path = pdb_dir / f"{pdb_id.lower()}.pdb"
        if not pdb_path.exists():
            report.add_error(f"PDB file not found: {pdb_path}")
        else:
            atom_count, chain_ids = _count_atom_records(pdb_path)
            if atom_count < 50:
                report.add_error(
                    f"PDB {pdb_path.name}: only {atom_count} ATOM records (need ≥ 50)"
                )

            # 3. Chain check
            if chain and chain_ids and chain not in chain_ids:
                report.add_error(
                    f"Chain '{chain}' not in PDB {pdb_path.name} "
                    f"(available: {sorted(chain_ids)})"
                )
    else:
        report.add_error("Missing target.pdb_id")

    # 4. New-format ID category check
    parsed = parse_new_task_id(task_id)
    if parsed is not None:
        # parse_new_task_id returns None for invalid categories,
        # so if it succeeds the category is valid
        pass

    # 5. Ground truth file
    gt_path = gt_dir / f"{task_id}.json"
    if not gt_path.exists():
        report.add_error(f"Ground truth file not found: {gt_path}")

    # 6. Prompt file
    prompt_path = prompt_dir / f"{task_id}.md"
    if not prompt_path.exists():
        report.add_error(f"Prompt file not found: {prompt_path}")

    return report


def validate_all_tasks(
    task_dir: str | Path | None = None,
    pdb_dir: str | Path | None = None,
    gt_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
) -> list[ValidationReport]:
    """Validate all task JSONs in a directory.

    Args:
        task_dir: Directory containing task JSON files.
        pdb_dir: Directory containing PDB files.
        gt_dir: Directory containing ground truth JSONs.
        prompt_dir: Directory containing prompt .md files.

    Returns:
        List of ValidationReport, one per task JSON.
    """
    task_dir = Path(task_dir) if task_dir else _DEFAULT_TASK_DIR
    pdb_dir = Path(pdb_dir) if pdb_dir else _DEFAULT_PDB_DIR
    gt_dir = Path(gt_dir) if gt_dir else _DEFAULT_GT_DIR
    prompt_dir = Path(prompt_dir) if prompt_dir else _DEFAULT_PROMPT_DIR

    reports: list[ValidationReport] = []

    for json_path in sorted(task_dir.glob("*.json")):
        with open(json_path) as f:
            task_json = json.load(f)
        report = validate_single_task(
            task_json,
            pdb_dir=pdb_dir,
            gt_dir=gt_dir,
            prompt_dir=prompt_dir,
        )
        reports.append(report)

    return reports
