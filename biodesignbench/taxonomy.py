"""Two-axis taxonomy for Tier 2 design tasks.

Axes:
    DesignApproach  — how the agent designs (2 types)
    MolecularSubject — what molecule is being designed (5 subjects)

Valid combinations form a 2x5 matrix with 8 populated cells
(binder x redesign and fluorescent_protein x de_novo are near-empty).
Each cell maps to a TaskCategory with expected tools and primary metric.

This module is standalone (stdlib-only imports) so it can be used early
in the import chain without pulling in heavy dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DesignApproach(str, Enum):
    """How the agent designs — generate from scratch or modify existing."""

    DE_NOVO = "de_novo"
    REDESIGN = "redesign"

    @property
    def short(self) -> str:
        return _APPROACH_SHORT[self]


class MolecularSubject(str, Enum):
    """What molecule is being designed."""

    ANTIBODY = "antibody"
    ENZYME = "enzyme"
    BINDER = "binder"
    SCAFFOLD = "scaffold"
    FLUORESCENT_PROTEIN = "fluorescent_protein"

    @property
    def short(self) -> str:
        return _SUBJECT_SHORT[self]


_APPROACH_SHORT: dict[DesignApproach, str] = {
    DesignApproach.DE_NOVO: "dn",
    DesignApproach.REDESIGN: "rd",
}

_SUBJECT_SHORT: dict[MolecularSubject, str] = {
    MolecularSubject.ANTIBODY: "ab",
    MolecularSubject.ENZYME: "enz",
    MolecularSubject.BINDER: "bnd",
    MolecularSubject.SCAFFOLD: "scf",
    MolecularSubject.FLUORESCENT_PROTEIN: "fp",
}

# Reverse lookups: short name -> enum
_SHORT_TO_APPROACH: dict[str, DesignApproach] = {v: k for k, v in _APPROACH_SHORT.items()}
_SHORT_TO_SUBJECT: dict[str, MolecularSubject] = {v: k for k, v in _SUBJECT_SHORT.items()}


# ---------------------------------------------------------------------------
# TaskCategory dataclass
# ---------------------------------------------------------------------------

# Core tools expected per design approach
_CORE_TOOLS: dict[DesignApproach, list[str]] = {
    DesignApproach.DE_NOVO: [
        "rfdiffusion",
        "proteinmpnn",
        "alphafold2",
    ],
    DesignApproach.REDESIGN: [
        "proteinmpnn",
        "esmfold",
        "alphafold2",
    ],
}

# Primary quality metric per design approach
_PRIMARY_METRIC: dict[DesignApproach, str] = {
    DesignApproach.DE_NOVO: "ipTM",
    DesignApproach.REDESIGN: "pLDDT",
}


@dataclass(frozen=True)
class TaskCategory:
    """A valid cell in the DesignApproach x MolecularSubject matrix."""

    approach: DesignApproach
    subject: MolecularSubject

    @property
    def category_id(self) -> str:
        """Short identifier, e.g. 'dn_bnd'."""
        return f"{self.approach.short}_{self.subject.short}"

    @property
    def expected_core_tools(self) -> list[str]:
        return list(_CORE_TOOLS[self.approach])

    @property
    def primary_quality_metric(self) -> str:
        return _PRIMARY_METRIC[self.approach]

    # Backward-compatible aliases
    @property
    def task_type(self) -> DesignApproach:
        return self.approach

    @property
    def context(self) -> MolecularSubject:
        return self.subject


# ---------------------------------------------------------------------------
# Valid categories (8 populated cells out of 2x5=10)
# ---------------------------------------------------------------------------

VALID_CATEGORIES: list[TaskCategory] = [
    # de_novo (5)
    TaskCategory(DesignApproach.DE_NOVO, MolecularSubject.ANTIBODY),
    TaskCategory(DesignApproach.DE_NOVO, MolecularSubject.ENZYME),
    TaskCategory(DesignApproach.DE_NOVO, MolecularSubject.BINDER),
    TaskCategory(DesignApproach.DE_NOVO, MolecularSubject.SCAFFOLD),
    TaskCategory(DesignApproach.DE_NOVO, MolecularSubject.FLUORESCENT_PROTEIN),
    # redesign (5)
    TaskCategory(DesignApproach.REDESIGN, MolecularSubject.ANTIBODY),
    TaskCategory(DesignApproach.REDESIGN, MolecularSubject.ENZYME),
    TaskCategory(DesignApproach.REDESIGN, MolecularSubject.SCAFFOLD),
    TaskCategory(DesignApproach.REDESIGN, MolecularSubject.FLUORESCENT_PROTEIN),
    # Note: BINDER x REDESIGN is empty (no tasks) but kept valid for future use
    TaskCategory(DesignApproach.REDESIGN, MolecularSubject.BINDER),
]

_CATEGORY_BY_ID: dict[str, TaskCategory] = {c.category_id: c for c in VALID_CATEGORIES}


# ---------------------------------------------------------------------------
# OLD -> NEW task ID mapping (76 tasks)
# ---------------------------------------------------------------------------

OLD_TO_NEW_MAPPING: dict[str, str] = {
    # ── de_novo × antibody (4) ──
    "dnb_ab_001": "dn_ab_001",
    "dnb_ab_002": "dn_ab_002",
    "dnb_ab_003": "dn_ab_003",
    "dnb_ab_005": "dn_ab_004",
    # ── de_novo × enzyme (2) ──
    "cfd_enz_005": "dn_enz_001",
    "cpx_enz_001": "dn_enz_002",
    # ── de_novo × binder (19) ──
    "binder_003": "dn_bnd_001",
    "binder_005": "dn_bnd_002",
    "binder_007": "dn_bnd_003",
    "binder_008": "dn_bnd_004",
    "peptide_001": "dn_bnd_005",
    "peptide_002": "dn_bnd_006",
    "dnb_enz_001": "dn_bnd_007",
    "dnb_enz_002": "dn_bnd_008",
    "dnb_enz_003": "dn_bnd_009",
    "dnb_sig_007": "dn_bnd_010",
    "dnb_sig_008": "dn_bnd_011",
    "cpx_sig_001": "dn_bnd_012",
    "cpx_sig_002": "dn_bnd_013",
    "cpx_sig_003": "dn_bnd_014",
    "cpx_sig_004": "dn_bnd_015",
    "cpx_sig_005": "dn_bnd_016",
    "cpx_sig_006": "dn_bnd_017",
    "cpx_sig_007": "dn_bnd_018",
    "cpx_sig_008": "dn_bnd_019",
    # ── de_novo × scaffold (21) ──
    "scaffold_001": "dn_scf_001",
    "scaffold_002": "dn_scf_002",
    "scaffold_003": "dn_scf_003",
    "dnk_str_004": "dn_scf_004",
    "dnk_str_005": "dn_scf_005",
    "dnk_str_006": "dn_scf_006",
    "ppi_002": "dn_scf_007",
    "cpx_str_007": "dn_scf_008",
    "cfd_sig_002": "dn_scf_009",
    "cfd_sig_003": "dn_scf_010",
    "cfd_sig_004": "dn_scf_011",
    "cfd_sig_005": "dn_scf_012",
    "cfd_sig_006": "dn_scf_013",
    "cfd_str_001": "dn_scf_014",
    "cfd_str_002": "dn_scf_015",
    "ppi_001": "dn_scf_016",
    "ppi_003": "dn_scf_017",
    "ppi_004": "dn_scf_018",
    "cpx_str_003": "dn_scf_019",
    "cpx_str_004": "dn_scf_020",
    "cpx_str_008": "dn_scf_021",
    # ── de_novo × fluorescent_protein (1) ──
    "cfd_flu_004": "dn_fp_001",
    # ── redesign × antibody (5) ──
    "antibody_001": "rd_ab_001",
    "antibody_002": "rd_ab_002",
    "antibody_003": "rd_ab_003",
    "antibody_004": "rd_ab_004",
    "sqo_ab_006": "rd_ab_005",
    # ── redesign × enzyme (10) ──
    "enzyme_001": "rd_enz_001",
    "enzyme_002": "rd_enz_002",
    "stability_002": "rd_enz_003",
    "sqo_enz_005": "rd_enz_004",
    "sqo_enz_006": "rd_enz_005",
    "cfd_enz_006": "rd_enz_006",
    "sqo_sig_001": "rd_enz_007",
    "sqo_sig_002": "rd_enz_008",
    "sqo_sig_003": "rd_enz_009",
    "cpx_enz_002": "rd_enz_010",
    # ── redesign × scaffold (4) ──
    "stability_003": "rd_scf_001",
    "sqo_str_005": "rd_scf_002",
    "cpx_str_005": "rd_scf_003",
    "cpx_str_006": "rd_scf_004",
    # ── redesign × fluorescent_protein (10) ──
    "stability_001": "rd_fp_001",
    "fluorescence_001": "rd_fp_002",
    "sqo_flu_002": "rd_fp_003",
    "sqo_flu_003": "rd_fp_004",
    "sqo_flu_004": "rd_fp_005",
    "sqo_flu_005": "rd_fp_006",
    "sqo_flu_006": "rd_fp_007",
    "cfd_flu_003": "rd_fp_008",
    "cfd_flu_005": "rd_fp_009",
    "cfd_flu_006": "rd_fp_010",
}

_NEW_TO_OLD_MAPPING: dict[str, str] = {v: k for k, v in OLD_TO_NEW_MAPPING.items()}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

_NEW_ID_RE = re.compile(r"^([a-z]{2})_([a-z]{2,3})_(\d{3})$")


def get_category(task_id: str) -> Optional[TaskCategory]:
    """Get the TaskCategory for a task ID (old or new format).

    Returns None if the task ID is not recognized.
    """
    # Try old format first
    if task_id in OLD_TO_NEW_MAPPING:
        new_id = OLD_TO_NEW_MAPPING[task_id]
        cat_id = new_id.rsplit("_", 1)[0]
        return _CATEGORY_BY_ID.get(cat_id)

    # Try new format
    m = _NEW_ID_RE.match(task_id)
    if m:
        cat_id = f"{m.group(1)}_{m.group(2)}"
        return _CATEGORY_BY_ID.get(cat_id)

    return None


def get_new_task_id(old_task_id: str) -> Optional[str]:
    """Map an old task ID to its new two-axis ID.

    Returns None if the old ID is not recognized.
    """
    return OLD_TO_NEW_MAPPING.get(old_task_id)


def get_old_task_id(new_task_id: str) -> Optional[str]:
    """Map a new two-axis task ID back to the original ID.

    Returns None if the new ID is not recognized.
    """
    return _NEW_TO_OLD_MAPPING.get(new_task_id)


def is_valid_category(approach: DesignApproach, subject: MolecularSubject) -> bool:
    """Check whether a (approach, subject) pair is a valid cell."""
    cat_id = f"{approach.short}_{subject.short}"
    return cat_id in _CATEGORY_BY_ID


def parse_new_task_id(
    task_id: str,
) -> Optional[tuple[DesignApproach, MolecularSubject, int]]:
    """Parse a new-format task ID into its components.

    Returns (DesignApproach, MolecularSubject, sequence_number) or None
    if the ID is invalid or doesn't map to a valid category.
    """
    m = _NEW_ID_RE.match(task_id)
    if not m:
        return None

    approach_short, subject_short, num_str = m.group(1), m.group(2), m.group(3)

    approach = _SHORT_TO_APPROACH.get(approach_short)
    subject = _SHORT_TO_SUBJECT.get(subject_short)

    if approach is None or subject is None:
        return None

    if not is_valid_category(approach, subject):
        return None

    return approach, subject, int(num_str)


def get_task_distribution() -> dict[str, int]:
    """Count how many existing tasks fall into each valid category.

    Returns a dict mapping category_id -> count for all valid cells.
    """
    dist: dict[str, int] = {c.category_id: 0 for c in VALID_CATEGORIES}
    for new_id in OLD_TO_NEW_MAPPING.values():
        cat_id = new_id.rsplit("_", 1)[0]
        if cat_id in dist:
            dist[cat_id] += 1
    return dist


# ---------------------------------------------------------------------------
# Task type normalization (maps old category names to canonical enum values)
# ---------------------------------------------------------------------------

_OLD_TYPE_TO_CANONICAL: dict[str, str] = {
    # Old DesignTaskType values -> new DesignApproach
    "de_novo_binder": "de_novo",
    "de_novo_backbone": "de_novo",
    "complex_engineering": "de_novo",
    "conformational_design": "de_novo",
    "sequence_optimization": "redesign",
    # Old category prefix names
    "binder": "de_novo",
    "antibody": "redesign",
    "peptide": "de_novo",
    "stability": "redesign",
    "enzyme": "redesign",
    "fluorescence": "redesign",
    "scaffold": "de_novo",
    "ppi": "de_novo",
}

# Also accept the canonical enum values as-is
_CANONICAL_VALUES = {e.value for e in DesignApproach}


def normalize_task_type(task_type: str) -> str:
    """Map old-format task_type names to canonical DesignApproach values.

    Old task JSONs use category names like ``"binder"``, ``"antibody"``,
    ``"scaffold"`` as ``task_type``.  This function normalizes them to the
    canonical enum values (``"de_novo"``, ``"redesign"``).

    Args:
        task_type: Raw task_type value from a task JSON.

    Returns:
        Canonical design approach string.  Returns *task_type* unchanged if
        it is already canonical or not recognized.
    """
    lower = task_type.lower().strip()
    if lower in _CANONICAL_VALUES:
        return lower
    return _OLD_TYPE_TO_CANONICAL.get(lower, task_type)


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

# Allow old import names to work during migration
DesignTaskType = DesignApproach
BiologicalContext = MolecularSubject
