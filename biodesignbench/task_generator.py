"""Task expansion framework — TaskSpec, validation, and conversion.

Provides the machinery to define new Tier 2 tasks via ``TaskSpec`` dataclass,
validate them against the taxonomy and PDB files, and convert them into the
standard DesignTask JSON + ground truth JSON formats consumed by the
evaluation pipeline.

Usage:
    spec = TaskSpec(task_id="dnb_sig_006", ...)
    errors = validate_task_spec(spec, pdb_base_dir=Path("data/tier2/input"), all_ids=set())
    task_json = spec_to_task_json(spec, random.Random(42))
    ground_truth = spec_to_ground_truth(spec)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biodesignbench.eval.metrics.approach import MCP_TOOL_EXPANSION, TOOL_CATEGORIES
from biodesignbench.taxonomy import (
    BiologicalContext,
    DesignTaskType,
    TaskCategory,
    VALID_CATEGORIES,
    _CORE_TOOLS,
    _PRIMARY_METRIC,
    is_valid_category,
    parse_new_task_id,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical set of 11 MCP tool names from protein_design_provider.py.
MCP_TOOL_NAMES: frozenset[str] = frozenset(MCP_TOOL_EXPANSION.keys())

#: All recognized tool names (MCP wrappers + low-level bio tools).
_ALL_TOOL_NAMES: set[str] = set(MCP_TOOL_EXPANSION.keys()) | set(TOOL_CATEGORIES.keys())

#: Amino acid alphabet and weights (same as generate_tier2_content.py).
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_W = [8, 1, 5, 6, 4, 7, 2, 5, 6, 9, 2, 4, 5, 4, 5, 7, 5, 7, 1, 3]

#: Reasonable ranges for common evaluation thresholds.
_THRESHOLD_RANGES: dict[str, tuple[float, float]] = {
    "pLDDT_good": (50, 100),
    "ipTM_good": (0.5, 1.0),
    "kd_nM_good": (0.001, 10000),
    "max_seq_identity_good": (0.0, 1.0),
    "predicted_ddG_good": (-20.0, 5.0),
    "active_site_rmsd_good": (0.0, 5.0),
    "kd_improvement_fold": (1, 1000),
}

#: Default eval thresholds per primary metric.
_DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "ipTM": {
        "pLDDT_good": 80,
        "ipTM_good": 0.8,
        "kd_nM_good": 100,
        "max_seq_identity_good": 0.3,
    },
    "pLDDT": {
        "pLDDT_good": 80,
        "max_seq_identity_good": 0.3,
    },
}

_NEW_ID_RE = re.compile(r"^[a-z]{2,3}_[a-z]{2,3}_\d{3}$")


# ---------------------------------------------------------------------------
# TaskSpec dataclass
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Generation-time specification for a Tier 2 design task.

    This is the input to the expansion pipeline — it gets converted into
    a DesignTask JSON (for the agent) and a ground truth JSON (for the
    evaluator) via ``spec_to_task_json()`` and ``spec_to_ground_truth()``.
    """

    # Identity
    task_id: str
    task_type: DesignTaskType
    biological_context: BiologicalContext
    difficulty: str  # "easy" | "medium" | "hard"

    # Target
    target_name: str
    target_pdb_id: str
    target_chain: str
    target_seq_len: int
    binding_site_residues: list[int] | None = None
    additional_info: dict[str, Any] = field(default_factory=dict)

    # Prompt text
    description: str = ""

    # Design constraints
    length_range: tuple[int, int] | None = None
    excluded_residues: list[str] = field(default_factory=list)
    required_residues: dict[int, str] = field(default_factory=dict)
    max_designs: int = 10
    additional_constraints: dict[str, Any] = field(default_factory=dict)

    # Expected tools (auto-populated from taxonomy if empty)
    expected_tools: list[str] = field(default_factory=list)

    # Ground truth
    gt_kd_nM: float | None = None
    gt_tm_C: float | None = None
    gt_known_sequence: str | None = None
    gt_expression: bool | None = None
    gt_additional: dict[str, float] = field(default_factory=dict)

    # Evaluation thresholds (auto-populated if empty)
    eval_thresholds: dict[str, float] = field(default_factory=dict)

    # Metadata
    source: str = "synthetic"
    doi: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def category(self) -> TaskCategory:
        """Return the TaskCategory for this spec."""
        return TaskCategory(self.task_type, self.biological_context)

    @property
    def category_id(self) -> str:
        """Short identifier, e.g. 'dnb_sig'."""
        return self.category.category_id

    def __post_init__(self) -> None:
        """Auto-fill expected_tools and eval_thresholds from taxonomy."""
        if not self.expected_tools:
            self.expected_tools = list(_CORE_TOOLS.get(self.task_type, []))

        if not self.eval_thresholds:
            primary = _PRIMARY_METRIC.get(self.task_type, "pLDDT")
            self.eval_thresholds = dict(_DEFAULT_THRESHOLDS.get(primary, {}))


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------


def gen_seq(n: int, rng: random.Random) -> str:
    """Generate a random amino acid sequence of length *n*.

    The first residue is always methionine ('M'). Amino acid frequencies
    are weighted to approximate natural abundance.

    Args:
        n: Desired sequence length (≥ 1).
        rng: Seeded Random instance for reproducibility.

    Returns:
        Amino acid sequence string of length *n*.
    """
    seq = rng.choices(_AA, weights=_AA_W, k=n)
    seq[0] = "M"
    return "".join(seq)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_task_spec(
    spec: TaskSpec,
    *,
    pdb_base_dir: Path,
    all_ids: set[str],
) -> list[str]:
    """Validate a TaskSpec against structural/semantic constraints.

    Returns a list of error strings (empty = valid).

    Checks:
    1.  task_id format (new-format regex)
    2.  task_id prefix matches task_type × biological_context
    3.  task_type × context in VALID_CATEGORIES
    4.  PDB file exists with ≥ 50 ATOM records
    5.  target_chain in PDB chain IDs
    6.  expected_tools are recognized
    7.  eval_thresholds within reasonable ranges
    8.  No duplicate task_id
    9.  length_range valid (if present)
    10. difficulty in {easy, medium, hard}
    11. description ≥ 50 chars
    """
    errors: list[str] = []

    # 1. task_id format
    if not _NEW_ID_RE.match(spec.task_id):
        errors.append(
            f"task_id '{spec.task_id}' does not match new-format pattern "
            f"(expected ^[a-z]{{2,3}}_[a-z]{{2,3}}_\\d{{3}}$)"
        )

    # 2. task_id prefix matches spec enums
    parsed = parse_new_task_id(spec.task_id)
    if parsed is not None:
        p_type, p_ctx, _ = parsed
        if p_type != spec.task_type or p_ctx != spec.biological_context:
            errors.append(
                f"task_id prefix mismatch: '{spec.task_id}' implies "
                f"{p_type.short}_{p_ctx.short} but spec has "
                f"{spec.task_type.short}_{spec.biological_context.short}"
            )

    # 3. valid category
    if not is_valid_category(spec.task_type, spec.biological_context):
        errors.append(
            f"Invalid category: {spec.task_type.short}_{spec.biological_context.short} "
            f"is not in VALID_CATEGORIES"
        )

    # 4. PDB file exists with ≥ 50 ATOM records
    pdb_path = pdb_base_dir / f"{spec.target_pdb_id.lower()}.pdb"
    if not pdb_path.exists():
        errors.append(f"PDB file not found: {pdb_path}")
    else:
        atom_count = 0
        chain_ids: set[str] = set()
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_count += 1
                    if len(line) >= 22:
                        chain_ids.add(line[21])
        if atom_count < 50:
            errors.append(
                f"PDB {pdb_path.name} has only {atom_count} ATOM records (need ≥ 50)"
            )

        # 5. target_chain in PDB
        if chain_ids and spec.target_chain not in chain_ids:
            errors.append(
                f"Chain '{spec.target_chain}' not found in PDB {pdb_path.name} "
                f"(available: {sorted(chain_ids)})"
            )

    # 6. expected_tools recognized
    for tool in spec.expected_tools:
        if tool not in _ALL_TOOL_NAMES:
            errors.append(f"Unrecognized tool '{tool}' in expected_tools")

    # 7. eval_thresholds within range
    for key, value in spec.eval_thresholds.items():
        if key in _THRESHOLD_RANGES:
            lo, hi = _THRESHOLD_RANGES[key]
            if not (lo <= value <= hi):
                errors.append(
                    f"Threshold '{key}' = {value} out of range [{lo}, {hi}]"
                )

    # 8. duplicate ID
    if spec.task_id in all_ids:
        errors.append(f"Duplicate task_id: '{spec.task_id}'")

    # 9. length_range
    if spec.length_range is not None:
        lo, hi = spec.length_range
        if lo <= 0 or hi <= 0:
            errors.append(f"length_range values must be > 0 (got {spec.length_range})")
        elif lo > hi:
            errors.append(f"length_range min ({lo}) > max ({hi})")

    # 10. difficulty
    if spec.difficulty not in {"easy", "medium", "hard"}:
        errors.append(
            f"Invalid difficulty '{spec.difficulty}' (must be easy/medium/hard)"
        )

    # 11. description length
    if len(spec.description) < 50:
        errors.append(
            f"Description too short ({len(spec.description)} chars, need ≥ 50)"
        )

    return errors


# ---------------------------------------------------------------------------
# Conversion: TaskSpec → DesignTask JSON
# ---------------------------------------------------------------------------


def spec_to_task_json(spec: TaskSpec, rng: random.Random) -> dict[str, Any]:
    """Convert a TaskSpec to a DesignTask-compatible JSON dict.

    Follows the same structure as ``scripts/generate_tier2_content.py``'s
    ``build_task_json()`` so that existing evaluation code can consume it.
    """
    seq = gen_seq(spec.target_seq_len, rng)

    target: dict[str, Any] = {
        "name": spec.target_name,
        "pdb_id": spec.target_pdb_id,
        "sequence": seq,
        "chain": spec.target_chain,
        "binding_site_residues": spec.binding_site_residues,
        "additional_info": spec.additional_info,
    }

    design_constraints: dict[str, Any] = {
        "length_range": list(spec.length_range) if spec.length_range else None,
        "excluded_residues": spec.excluded_residues,
        "required_residues": {str(k): v for k, v in spec.required_residues.items()},
        "max_designs": spec.max_designs,
        "additional": spec.additional_constraints,
    }

    metrics = ["pLDDT", "ipTM", "predicted_kd", "max_sequence_identity", "diversity"]

    gt_data: dict[str, Any] | None = {
        "known_sequence": spec.gt_known_sequence,
        "experimental_kd_nM": spec.gt_kd_nM,
        "experimental_tm": spec.gt_tm_C,
        "experimental_expression": spec.gt_expression,
        "additional_metrics": spec.gt_additional,
    }

    return {
        "task_id": spec.task_id,
        "task_type": spec.category_id,
        "tier": "tier2",
        "description": spec.description,
        "target": target,
        "design_constraints": design_constraints,
        "expected_output": {
            "format": "design_bundle",
            "required_files": ["designed_sequences.fasta", "metrics.json"],
        },
        "evaluation": {
            "method": "metrics",
            "test_file": None,
            "metrics": metrics,
            "ground_truth": gt_data,
        },
        "constraints": {
            "time_limit_minutes": 120,
            "knowledge_cutoff": "2024-09-01",
            "max_api_calls": None,
            "max_cost_usd": None,
        },
        "metadata": {
            "difficulty": spec.difficulty,
            "source": spec.source,
            "doi": spec.doi,
            "tools_expected": spec.expected_tools,
            "tags": spec.tags,
        },
    }


# ---------------------------------------------------------------------------
# Conversion: TaskSpec → Ground Truth JSON
# ---------------------------------------------------------------------------


def spec_to_ground_truth(spec: TaskSpec) -> dict[str, Any]:
    """Convert a TaskSpec to a ground truth JSON dict.

    Follows the same structure as ``scripts/generate_tier2_content.py``'s
    ``build_ground_truth()``.
    """
    experimental: dict[str, Any] = {
        "kd_nM": spec.gt_kd_nM,
        "tm_C": spec.gt_tm_C,
        "expression": spec.gt_expression,
    }
    experimental.update(spec.gt_additional)

    return {
        "task_id": spec.task_id,
        "description": (
            f"Experimental ground truth for {spec.target_name} "
            f"{spec.task_type.value.replace('_', ' ')}"
        ),
        "source_paper": spec.source,
        "doi": spec.doi,
        "target": {
            "name": spec.target_name,
            "pdb_id": spec.target_pdb_id,
        },
        "experimental_results": experimental,
        "evaluation_thresholds": dict(spec.eval_thresholds),
        "notes": f"From {spec.source}.",
    }
