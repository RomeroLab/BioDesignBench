"""Centralized input resolution and provisioning for benchmark tasks."""

import logging
import os
import shutil
from pathlib import Path

from biodesignbench.tasks.schema import CodingTask, DesignTask, Task, TaskTier

logger = logging.getLogger(__name__)


class InputManager:
    """Resolves task input paths and provides read-only input access.

    Handles three path formats found in task JSONs:
    - Absolute: /home/.../data/tier1/input/1atp.pdb
    - Prefixed: data/tier1/input/1atp.pdb
    - Bare: 1atp.pdb
    """

    def __init__(self, project_root: Path | None = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        self._project_root = Path(project_root)
        self.tier1_input_dir = self._project_root / "data" / "tier1" / "input"
        self.tier2_input_dir = self._project_root / "data" / "tier2" / "input"

    def resolve_inputs(self, task: Task) -> list[Path]:
        """Resolve all input paths to absolute paths (pure query, no side effects).

        For CodingTask: resolves input_data.files (relative, bare, or absolute).
        For DesignTask: resolves target.pdb_id to tier2 input PDB.

        Returns only paths that exist on disk.
        """
        resolved = []

        if isinstance(task, CodingTask):
            for filepath in task.input_data.files:
                path = self._resolve_single_path(filepath, task.tier)
                if path is not None and path.exists():
                    resolved.append(path)
        elif isinstance(task, DesignTask):
            if task.target.pdb_id:
                pdb_path = self.tier2_input_dir / f"{task.target.pdb_id.lower()}.pdb"
                if pdb_path.exists():
                    resolved.append(pdb_path)

        return resolved

    def provision_inputs(
        self, task: Task, working_dir: Path, *, use_symlinks: bool = True
    ) -> list[Path]:
        """Make inputs accessible in working_dir via symlinks (copy fallback).

        Idempotent: skips files already provisioned.

        Args:
            task: The task whose inputs to provision.
            working_dir: Directory where inputs should be accessible.
            use_symlinks: If True, create symlinks. If False, copy files.

        Returns:
            List of paths to provisioned files in working_dir.
        """
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        resolved = self.resolve_inputs(task)
        provisioned = []

        for src in resolved:
            dst = working_dir / src.name
            if dst.exists() or dst.is_symlink():
                # Already provisioned (idempotent)
                provisioned.append(dst)
                continue

            if use_symlinks:
                try:
                    os.symlink(src, dst)
                except OSError:
                    # Symlink not supported (e.g., Windows without privileges)
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

            provisioned.append(dst)

        return provisioned

    def get_input_dir(self, task: Task) -> Path:
        """Return canonical input directory for task's tier."""
        if task.tier == TaskTier.TIER2:
            return self.tier2_input_dir
        return self.tier1_input_dir

    def _resolve_single_path(self, filepath: str, tier: TaskTier) -> Path | None:
        """Resolve a single file path to an absolute path.

        Handles three formats:
        - Absolute: returns as-is
        - Prefixed (data/tier1/input/...): resolves relative to project root
        - Bare (filename only): resolves relative to tier's input dir
        """
        path = Path(filepath)

        # Absolute path
        if path.is_absolute():
            return path

        # Prefixed path (data/tier1/input/... or data/tier2/input/...)
        if str(filepath).startswith("data/"):
            resolved = self._project_root / filepath
            return resolved

        # Bare filename - resolve to tier's input dir
        input_dir = self.tier2_input_dir if tier == TaskTier.TIER2 else self.tier1_input_dir
        return input_dir / filepath
