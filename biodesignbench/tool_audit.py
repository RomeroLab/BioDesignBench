"""Tool boundary audit for Tier 2 tasks.

Verifies that every task's oracle methodology can be evaluated using
the available MCP protein design tools.  Tasks requiring tools outside
the MCP boundary (e.g. SCUBA-D, BoltzGen) should have been removed
already; this module provides a systematic check.

Usage::

    from biodesignbench.tool_audit import audit_all_tasks, AuditVerdict
    results = audit_all_tasks()
    assert all(r.verdict == AuditVerdict.KEEP for r in results)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class AuditVerdict(str, Enum):
    """Whether a task should be kept or removed based on tool coverage."""

    KEEP = "keep"
    REMOVE = "remove"


# Maps oracle methodology labels → whether MCP tools can evaluate them.
# True = covered by MCP server tools, False = not covered.
ORACLE_METHOD_TO_MCP: dict[str, bool] = {
    # Core pipeline tools
    "PyRosetta": True,
    "Rosetta": True,
    "RFdiffusion": True,
    "RFdiffusion+ProteinMPNN+AF2": True,
    "ProteinMPNN": True,
    "AlphaFold2": True,
    "ESMFold": True,
    # Scoring / optimization
    "ESM2": True,
    "ESM2-guided": True,
    "Directed Evo": True,
    "Directed Evolution": True,
    "DirEvo": True,
    "OpenMM": True,
    "Boltz": True,
    # NOT covered
    "SCUBA-D": False,
    "BoltzGen": False,
}


@dataclass
class TaskAuditResult:
    """Result of auditing a single task."""

    task_id: str
    oracle_method: str
    verdict: AuditVerdict
    reason: str


def audit_task(task_id: str, oracle_method: str) -> TaskAuditResult:
    """Audit a single task for MCP tool coverage.

    Args:
        task_id: Task identifier.
        oracle_method: Oracle methodology label (e.g. "PyRosetta").

    Returns:
        TaskAuditResult with verdict and reason.
    """
    # Normalize method name
    method = oracle_method.strip()

    # Check exact match first
    if method in ORACLE_METHOD_TO_MCP:
        if ORACLE_METHOD_TO_MCP[method]:
            return TaskAuditResult(
                task_id=task_id,
                oracle_method=method,
                verdict=AuditVerdict.KEEP,
                reason=f"Method '{method}' is covered by MCP tools",
            )
        else:
            return TaskAuditResult(
                task_id=task_id,
                oracle_method=method,
                verdict=AuditVerdict.REMOVE,
                reason=f"Method '{method}' is NOT covered by MCP tools",
            )

    # Check case-insensitive match
    method_lower = method.lower()
    for known, covered in ORACLE_METHOD_TO_MCP.items():
        if known.lower() == method_lower:
            verdict = AuditVerdict.KEEP if covered else AuditVerdict.REMOVE
            return TaskAuditResult(
                task_id=task_id,
                oracle_method=method,
                verdict=verdict,
                reason=f"Method '{method}' (matched as '{known}') {'is' if covered else 'is NOT'} covered",
            )

    # Unknown method — flag for review but default to KEEP
    return TaskAuditResult(
        task_id=task_id,
        oracle_method=method,
        verdict=AuditVerdict.KEEP,
        reason=f"Method '{method}' not in audit registry; defaulting to KEEP",
    )


# Methodology assignments for all tasks (from taxonomy_matrix.md)
_TASK_METHODS: dict[str, str] = {
    # De Novo Binder × Signaling (9) — all Rosetta except binder_008
    "binder_001": "Rosetta",
    "binder_002": "Rosetta",
    "binder_003": "Rosetta",
    "binder_005": "Rosetta",
    "binder_007": "Rosetta",
    "binder_008": "RFdiffusion",
    "peptide_001": "Rosetta",
    "peptide_002": "Rosetta",
    "peptide_003": "Rosetta",
    # De Novo Binder × Antibody (5 restored)
    "dnb_ab_001": "RFdiffusion",
    "dnb_ab_002": "RFdiffusion",
    "dnb_ab_003": "RFdiffusion",
    "dnb_ab_004": "RFdiffusion",
    "dnb_ab_005": "RFdiffusion",
    # De Novo Binder × Enzyme (3 new)
    "dnb_enz_001": "RFdiffusion",
    "dnb_enz_002": "RFdiffusion",
    "dnb_enz_003": "RFdiffusion",
    # De Novo Binder × Signaling (buffer)
    "dnb_sig_007": "RFdiffusion",
    "dnb_sig_008": "RFdiffusion",
    # Seq Optimize × Antibody (5)
    "antibody_001": "Directed Evo",
    "antibody_002": "Directed Evo",
    "antibody_003": "Directed Evo",
    "antibody_004": "Directed Evo",
    "sqo_ab_006": "Directed Evo",
    # Seq Optimize × Enzyme (2)
    "sqo_enz_005": "Directed Evo",
    "sqo_enz_006": "Directed Evo",
    # Seq Optimize × Structural (4)
    "sqo_str_005": "Rosetta",
    "stability_001": "Directed Evo",
    "stability_002": "Directed Evo",
    "stability_003": "Rosetta",
    # Seq Optimize × Signaling (3 new)
    "sqo_sig_001": "Directed Evo",
    "sqo_sig_002": "Directed Evo",
    "sqo_sig_003": "Directed Evo",
    # Seq Optimize × Fluorescence (6)
    "fluorescence_001": "Directed Evo",
    "sqo_flu_002": "Directed Evo",
    "sqo_flu_003": "Directed Evo",
    "sqo_flu_004": "Directed Evo",
    "sqo_flu_005": "Directed Evo",
    "sqo_flu_006": "Directed Evo",
    # De Novo Structure × Structural (6)
    "dnk_str_004": "Rosetta",
    "dnk_str_005": "Rosetta",
    "dnk_str_006": "Rosetta",
    "scaffold_001": "Rosetta",
    "scaffold_002": "Rosetta",
    "scaffold_003": "Rosetta",
    # Complex × Enzyme (2 new)
    "cpx_enz_001": "Rosetta",
    "cpx_enz_002": "Rosetta",
    # Complex × Signaling (6 + 2 buffer)
    "cpx_sig_001": "Rosetta",
    "cpx_sig_002": "Rosetta",
    "cpx_sig_003": "Rosetta",
    "cpx_sig_004": "RFdiffusion",
    "cpx_sig_005": "Rosetta",
    "cpx_sig_006": "Rosetta",
    "cpx_sig_007": "RFdiffusion",
    "cpx_sig_008": "RFdiffusion",
    # Complex × Structural (8)
    "cpx_str_003": "Rosetta",
    "cpx_str_004": "Rosetta",
    "cpx_str_005": "Rosetta",
    "cpx_str_006": "Rosetta",
    "ppi_001": "Rosetta",
    "ppi_002": "Rosetta",
    "ppi_003": "Rosetta",
    "ppi_004": "Rosetta",
    "cpx_str_007": "RFdiffusion",
    "cpx_str_008": "RFdiffusion",
    # Conformational × Enzyme (4)
    "enzyme_001": "Rosetta",
    "enzyme_002": "Rosetta",
    "cfd_enz_005": "Directed Evo",
    "cfd_enz_006": "Directed Evo",
    # Conformational × Signaling (5)
    "cfd_sig_002": "Rosetta",
    "cfd_sig_003": "RFdiffusion",
    "cfd_sig_004": "Rosetta",
    "cfd_sig_005": "Rosetta",
    "cfd_sig_006": "RFdiffusion",
    # Conformational × Structural (2 new)
    "cfd_str_001": "Rosetta",
    "cfd_str_002": "Rosetta",
    # Conformational × Fluorescence (4)
    "cfd_flu_003": "Directed Evo",
    "cfd_flu_004": "Rosetta",
    "cfd_flu_005": "Directed Evo",
    "cfd_flu_006": "Directed Evo",
}


def audit_all_tasks(
    task_methods: Optional[dict[str, str]] = None,
) -> list[TaskAuditResult]:
    """Audit all known tasks for MCP tool coverage.

    Args:
        task_methods: Optional override of task → method mapping.
            Defaults to the built-in _TASK_METHODS registry.

    Returns:
        List of TaskAuditResult objects, one per task.
    """
    methods = task_methods if task_methods is not None else _TASK_METHODS
    return [audit_task(tid, method) for tid, method in sorted(methods.items())]
