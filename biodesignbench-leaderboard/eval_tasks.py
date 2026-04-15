"""Load hidden benchmark tasks from a private HuggingFace Dataset.

Each task row contains:
  - task_id:          e.g., "dnb_sig_001"
  - task_json:        Full task definition (JSON string)
  - ground_truth:     Ground truth thresholds + reference (JSON string)
  - prompt_md:        Task prompt in Markdown
  - pdb_data:         Base64-encoded PDB file (if needed)
  - pdb_filename:     Original PDB filename (e.g., "7n1j.pdb")
  - oracle_sequences: JSON list of oracle sequences (for non-binding tasks)

Falls back to local files in development (when BDB_USE_LOCAL=1).

HF Dataset: RomeroLab-Duke/biodesignbench-hidden-tasks (private)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

TASKS_DATASET = os.environ.get(
    "BDB_TASKS_DATASET",
    "RomeroLab-Duke/biodesignbench-hidden-tasks",
)
HF_TOKEN = os.environ.get("HF_TOKEN")
USE_LOCAL = os.environ.get("BDB_USE_LOCAL", "0") == "1"

# Local paths (for development)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TASKS_DIR = _PROJECT_ROOT / "tasks" / "tier2"
_GT_DIR = _PROJECT_ROOT / "data" / "tier2" / "ground_truth"
_PROMPTS_DIR = _PROJECT_ROOT / "data" / "tier2" / "prompts"
_INPUT_DIR = _PROJECT_ROOT / "data" / "tier2" / "input"
_ORACLE_PATH = _PROJECT_ROOT / "data" / "oracle" / "sequences.json"
_TOOL_SCHEMAS_PATH = Path(__file__).parent / "mcp_tool_schemas.json"

# Public task IDs (for development/testing — not hidden)
# One per major category: binding (dnb_ab), non-binding (sqo_enz), complex (cpx_sig)
PUBLIC_TASK_IDS = {"dnb_ab_001", "sqo_enz_005", "cpx_sig_001"}


# ---------------------------------------------------------------------------
#  HF Dataset loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_from_hf() -> dict[str, dict[str, Any]]:
    """Load all tasks from the private HF Dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            TASKS_DATASET,
            split="train",
            token=HF_TOKEN,
        )
        tasks = {}
        for row in ds:
            task_id = row["task_id"]
            tasks[task_id] = {
                "task_id": task_id,
                "task_json": json.loads(row["task_json"]),
                "ground_truth": json.loads(row["ground_truth"]),
                "prompt_md": row["prompt_md"],
                "pdb_data": row.get("pdb_data"),
                "pdb_filename": row.get("pdb_filename"),
                "oracle_sequences": json.loads(row.get("oracle_sequences", "[]")),
            }
        logger.info(f"Loaded {len(tasks)} tasks from HF Dataset")
        return tasks
    except Exception as e:
        logger.error(f"Failed to load tasks from HF: {e}")
        return {}


@lru_cache(maxsize=1)
def _load_from_local() -> dict[str, dict[str, Any]]:
    """Load tasks from local project files (development mode)."""
    tasks = {}

    # Load oracle data
    oracle_data = {}
    if _ORACLE_PATH.exists():
        with open(_ORACLE_PATH) as f:
            oracle_data = json.load(f)

    # Enumerate task files
    if not _TASKS_DIR.exists():
        logger.warning(f"Tasks directory not found: {_TASKS_DIR}")
        return tasks

    for task_path in sorted(_TASKS_DIR.glob("*.json")):
        task_id = task_path.stem
        try:
            with open(task_path) as f:
                task_json = json.load(f)

            # Ground truth
            gt_path = _GT_DIR / f"{task_id}.json"
            ground_truth = {}
            if gt_path.exists():
                with open(gt_path) as f:
                    ground_truth = json.load(f)

            # Prompt
            prompt_path = _PROMPTS_DIR / f"{task_id}.md"
            prompt_md = ""
            if prompt_path.exists():
                prompt_md = prompt_path.read_text()

            # PDB data
            pdb_data = None
            pdb_filename = None
            input_pdb = task_json.get("input_pdb") or task_json.get("pdb_file")
            if input_pdb:
                pdb_path = _INPUT_DIR / input_pdb
                if pdb_path.exists():
                    pdb_data = base64.b64encode(pdb_path.read_bytes()).decode()
                    pdb_filename = input_pdb

            # Oracle sequences
            oracle_entry = oracle_data.get(task_id, {})
            oracle_seqs = oracle_entry.get("sequences", []) if isinstance(oracle_entry, dict) else []

            tasks[task_id] = {
                "task_id": task_id,
                "task_json": task_json,
                "ground_truth": ground_truth,
                "prompt_md": prompt_md,
                "pdb_data": pdb_data,
                "pdb_filename": pdb_filename,
                "oracle_sequences": oracle_seqs,
            }
        except Exception as e:
            logger.warning(f"Failed to load task {task_id}: {e}")

    logger.info(f"Loaded {len(tasks)} tasks from local files")
    return tasks


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def load_all_tasks() -> dict[str, dict[str, Any]]:
    """Load all benchmark tasks.

    Returns:
        Dict mapping task_id → task data dict.
    """
    if USE_LOCAL:
        return _load_from_local()
    return _load_from_hf()


def get_task(task_id: str) -> dict[str, Any] | None:
    """Load a single task by ID."""
    tasks = load_all_tasks()
    return tasks.get(task_id)


def get_hidden_task_ids() -> list[str]:
    """Get the list of hidden (non-public) task IDs."""
    tasks = load_all_tasks()
    return sorted(tid for tid in tasks if tid not in PUBLIC_TASK_IDS)


def get_all_task_ids() -> list[str]:
    """Get all task IDs (public + hidden)."""
    return sorted(load_all_tasks().keys())


def get_public_task_ids() -> list[str]:
    """Get the 3 public task IDs for development."""
    tasks = load_all_tasks()
    return sorted(tid for tid in tasks if tid in PUBLIC_TASK_IDS)


@lru_cache(maxsize=1)
def load_tool_schemas() -> list[dict[str, Any]]:
    """Load the 17 MCP tool schemas for task payloads."""
    if _TOOL_SCHEMAS_PATH.exists():
        with open(_TOOL_SCHEMAS_PATH) as f:
            return json.load(f)
    return []


def build_task_payload(task_id: str) -> dict[str, Any] | None:
    """Build the payload to send to a submitter's endpoint.

    Returns:
        Dict with: task_id, task_description, available_tools,
        input_files, design_constraints, max_steps, timeout_sec.
        Returns None if task not found.
    """
    task = get_task(task_id)
    if task is None:
        return None

    task_json = task["task_json"]
    prompt = task["prompt_md"]

    # Build input files (base64-encoded PDBs)
    input_files = {}
    if task.get("pdb_data") and task.get("pdb_filename"):
        input_files[task["pdb_filename"]] = task["pdb_data"]

    # Extract constraints from task JSON
    constraints = task_json.get("design_constraints", {})
    max_designs = task_json.get("max_designs", 10)

    return {
        "task_id": task_id,
        "task_description": prompt,
        "available_tools": load_tool_schemas(),
        "input_files": input_files,
        "design_constraints": {
            **constraints,
            "max_designs": max_designs,
        },
        "max_steps": 50,
        "timeout_sec": 300,
    }
