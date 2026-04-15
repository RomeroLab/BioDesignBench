"""Standalone 100-point scoring rubric for BioDesignBench Tier 2 design tasks.

This file is a **self-contained extraction** of the scoring logic from the
``biodesignbench`` package.  It has **zero external dependencies** (stdlib only)
so it can run on HuggingFace Spaces without installing the full package.

Modules consolidated:
  - biodesignbench/taxonomy.py
  - biodesignbench/eval/metrics/sequence.py
  - biodesignbench/eval/metrics/approach.py
  - biodesignbench/eval/metrics/orchestration.py
  - biodesignbench/eval/tier2/scoring.py
  - biodesignbench/eval/tier2/oracle.py  (oracle loading stub)

Six scoring components (sum = 100):
  approach      (20 pts)  — Tool/methodology selection
  orchestration (15 pts)  — Pipeline ordering + intermediate validation
  quality       (35 pts)  — 3-tier continuous scoring (structure/interface/physics)
  feasibility   (15 pts)  — Valid AAs, length, composition + biophysical checks
  novelty       ( 5 pts)  — Sequence identity to known sequences
  diversity     (10 pts)  — Number + diversity of designs
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import combinations
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Taxonomy  (from biodesignbench/taxonomy.py)
# ═══════════════════════════════════════════════════════════════════════════════


class DesignTaskType(str, Enum):
    """What the agent does."""

    DE_NOVO_BINDER = "de_novo_binder"
    SEQUENCE_OPTIMIZATION = "sequence_optimization"
    DE_NOVO_BACKBONE = "de_novo_backbone"
    COMPLEX_ENGINEERING = "complex_engineering"
    CONFORMATIONAL_DESIGN = "conformational_design"

    @property
    def short(self) -> str:
        return _TASK_TYPE_SHORT[self]


class BiologicalContext(str, Enum):
    """Domain knowledge required."""

    ANTIBODY = "antibody"
    ENZYME = "enzyme"
    SIGNALING = "signaling"
    STRUCTURAL = "structural"
    FLUORESCENT = "fluorescent"
    THERAPEUTIC = "therapeutic"

    @property
    def short(self) -> str:
        return _CONTEXT_SHORT[self]


_TASK_TYPE_SHORT: dict[DesignTaskType, str] = {
    DesignTaskType.DE_NOVO_BINDER: "dnb",
    DesignTaskType.SEQUENCE_OPTIMIZATION: "sqo",
    DesignTaskType.DE_NOVO_BACKBONE: "dnk",
    DesignTaskType.COMPLEX_ENGINEERING: "cpx",
    DesignTaskType.CONFORMATIONAL_DESIGN: "cfd",
}

_CONTEXT_SHORT: dict[BiologicalContext, str] = {
    BiologicalContext.ANTIBODY: "ab",
    BiologicalContext.ENZYME: "enz",
    BiologicalContext.SIGNALING: "sig",
    BiologicalContext.STRUCTURAL: "str",
    BiologicalContext.FLUORESCENT: "flu",
    BiologicalContext.THERAPEUTIC: "thr",
}

_SHORT_TO_TASK_TYPE: dict[str, DesignTaskType] = {v: k for k, v in _TASK_TYPE_SHORT.items()}
_SHORT_TO_CONTEXT: dict[str, BiologicalContext] = {v: k for k, v in _CONTEXT_SHORT.items()}

# Core tools expected per task type
_CORE_TOOLS: dict[DesignTaskType, list[str]] = {
    DesignTaskType.DE_NOVO_BINDER: ["rfdiffusion", "proteinmpnn", "alphafold2"],
    DesignTaskType.SEQUENCE_OPTIMIZATION: ["proteinmpnn", "esmfold", "alphafold2"],
    DesignTaskType.DE_NOVO_BACKBONE: ["rfdiffusion", "proteinmpnn", "alphafold2"],
    DesignTaskType.COMPLEX_ENGINEERING: ["rfdiffusion", "proteinmpnn", "alphafold2"],
    DesignTaskType.CONFORMATIONAL_DESIGN: ["esmfold", "proteinmpnn", "alphafold2"],
}

_PRIMARY_METRIC: dict[DesignTaskType, str] = {
    DesignTaskType.DE_NOVO_BINDER: "ipTM",
    DesignTaskType.SEQUENCE_OPTIMIZATION: "pLDDT",
    DesignTaskType.DE_NOVO_BACKBONE: "pLDDT",
    DesignTaskType.COMPLEX_ENGINEERING: "ipTM",
    DesignTaskType.CONFORMATIONAL_DESIGN: "pLDDT",
}


@dataclass(frozen=True)
class TaskCategory:
    """A valid cell in the DesignTaskType × BiologicalContext matrix."""

    task_type: DesignTaskType
    context: BiologicalContext

    @property
    def category_id(self) -> str:
        return f"{self.task_type.short}_{self.context.short}"

    @property
    def expected_core_tools(self) -> list[str]:
        return list(_CORE_TOOLS[self.task_type])

    @property
    def primary_quality_metric(self) -> str:
        return _PRIMARY_METRIC[self.task_type]


VALID_CATEGORIES: list[TaskCategory] = [
    # de_novo_binder (4)
    TaskCategory(DesignTaskType.DE_NOVO_BINDER, BiologicalContext.ANTIBODY),
    TaskCategory(DesignTaskType.DE_NOVO_BINDER, BiologicalContext.ENZYME),
    TaskCategory(DesignTaskType.DE_NOVO_BINDER, BiologicalContext.SIGNALING),
    TaskCategory(DesignTaskType.DE_NOVO_BINDER, BiologicalContext.THERAPEUTIC),
    # sequence_optimization (5)
    TaskCategory(DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.ANTIBODY),
    TaskCategory(DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.ENZYME),
    TaskCategory(DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.SIGNALING),
    TaskCategory(DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.STRUCTURAL),
    TaskCategory(DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.FLUORESCENT),
    # de_novo_backbone (1)
    TaskCategory(DesignTaskType.DE_NOVO_BACKBONE, BiologicalContext.STRUCTURAL),
    # complex_engineering (3)
    TaskCategory(DesignTaskType.COMPLEX_ENGINEERING, BiologicalContext.ENZYME),
    TaskCategory(DesignTaskType.COMPLEX_ENGINEERING, BiologicalContext.SIGNALING),
    TaskCategory(DesignTaskType.COMPLEX_ENGINEERING, BiologicalContext.STRUCTURAL),
    # conformational_design (4)
    TaskCategory(DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.ENZYME),
    TaskCategory(DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.SIGNALING),
    TaskCategory(DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.STRUCTURAL),
    TaskCategory(DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.FLUORESCENT),
]

_CATEGORY_BY_ID: dict[str, TaskCategory] = {c.category_id: c for c in VALID_CATEGORIES}

# OLD → NEW task ID mapping (30 tasks)
OLD_TO_NEW_MAPPING: dict[str, str] = {
    "binder_001": "dnb_sig_001", "binder_003": "dnb_sig_002",
    "binder_005": "dnb_sig_003", "binder_007": "dnb_sig_004",
    "ppi_004": "dnb_sig_005",
    "binder_002": "dnb_thr_001", "binder_006": "dnb_thr_002",
    "binder_008": "dnb_thr_003", "peptide_001": "dnb_thr_004",
    "peptide_002": "dnb_thr_005", "peptide_003": "dnb_thr_006",
    "antibody_001": "sqo_ab_001", "antibody_002": "sqo_ab_002",
    "antibody_003": "sqo_ab_003", "antibody_004": "sqo_ab_004",
    "antibody_005": "sqo_ab_005",
    "stability_002": "sqo_enz_001", "enzyme_001": "sqo_enz_002",
    "enzyme_002": "sqo_enz_003", "enzyme_003": "sqo_enz_004",
    "stability_003": "sqo_str_001", "stability_004": "sqo_str_002",
    "stability_001": "sqo_flu_001",
    "scaffold_001": "dnk_str_001", "scaffold_002": "dnk_str_002",
    "scaffold_003": "dnk_str_003",
    "ppi_001": "cpx_str_001", "ppi_002": "cpx_str_002",
    "ppi_003": "cfd_sig_001",
    "fluorescence_001": "cfd_flu_001",
}
_NEW_TO_OLD_MAPPING: dict[str, str] = {v: k for k, v in OLD_TO_NEW_MAPPING.items()}

_NEW_ID_RE = re.compile(r"^([a-z]{2,3})_([a-z]{2,3})_(\d{3})$")

_OLD_TYPE_TO_CANONICAL: dict[str, str] = {
    "binder": "de_novo_binder", "antibody": "de_novo_binder",
    "peptide": "de_novo_binder", "stability": "sequence_optimization",
    "enzyme": "sequence_optimization", "fluorescence": "sequence_optimization",
    "scaffold": "de_novo_backbone", "ppi": "complex_engineering",
}
_CANONICAL_VALUES = {e.value for e in DesignTaskType}


def get_category(task_id: str) -> Optional[TaskCategory]:
    """Get the TaskCategory for a task ID (old or new format)."""
    if task_id in OLD_TO_NEW_MAPPING:
        new_id = OLD_TO_NEW_MAPPING[task_id]
        cat_id = new_id.rsplit("_", 1)[0]
        return _CATEGORY_BY_ID.get(cat_id)
    m = _NEW_ID_RE.match(task_id)
    if m:
        cat_id = f"{m.group(1)}_{m.group(2)}"
        return _CATEGORY_BY_ID.get(cat_id)
    return None


def get_new_task_id(old_task_id: str) -> Optional[str]:
    return OLD_TO_NEW_MAPPING.get(old_task_id)


def get_old_task_id(new_task_id: str) -> Optional[str]:
    return _NEW_TO_OLD_MAPPING.get(new_task_id)


def is_valid_category(task_type: DesignTaskType, context: BiologicalContext) -> bool:
    cat_id = f"{task_type.short}_{context.short}"
    return cat_id in _CATEGORY_BY_ID


def parse_new_task_id(
    task_id: str,
) -> Optional[tuple[DesignTaskType, BiologicalContext, int]]:
    m = _NEW_ID_RE.match(task_id)
    if not m:
        return None
    task_short, ctx_short, num_str = m.group(1), m.group(2), m.group(3)
    task_type = _SHORT_TO_TASK_TYPE.get(task_short)
    context = _SHORT_TO_CONTEXT.get(ctx_short)
    if task_type is None or context is None:
        return None
    if not is_valid_category(task_type, context):
        return None
    return task_type, context, int(num_str)


def normalize_task_type(task_type: str) -> str:
    lower = task_type.lower().strip()
    if lower in _CANONICAL_VALUES:
        return lower
    return _OLD_TYPE_TO_CANONICAL.get(lower, task_type)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Sequence Metrics  (from biodesignbench/eval/metrics/sequence.py)
# ═══════════════════════════════════════════════════════════════════════════════

_KD_SCALE: dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute fractional sequence identity between two sequences."""
    if not seq1 or not seq2:
        return 0.0
    s1, s2 = seq1.upper(), seq2.upper()
    if len(s1) == len(s2):
        return sum(a == b for a, b in zip(s1, s2)) / len(s1)
    short, long = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    best = 0.0
    for offset in range(len(long) - len(short) + 1):
        matches = sum(a == b for a, b in zip(short, long[offset:offset + len(short)]))
        identity = matches / len(short)
        if identity > best:
            best = identity
    return best


def max_identity_to_reference(designs: list[str], reference: str) -> float:
    if not designs or not reference:
        return 0.0
    return max(sequence_identity(d, reference) for d in designs)


def mean_pairwise_diversity(sequences: list[str]) -> float:
    if len(sequences) < 2:
        return 0.0
    total = 0.0
    count = 0
    for s1, s2 in combinations(sequences, 2):
        total += 1.0 - sequence_identity(s1, s2)
        count += 1
    return total / count if count > 0 else 0.0


def sequence_entropy(sequences: list[str], truncate: bool = False) -> float:
    if len(sequences) < 2:
        return 0.0
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        if not truncate:
            return 0.0
        seq_len = min(lengths)
        sequences = [s[:seq_len] for s in sequences]
    else:
        seq_len = lengths.pop()
    if seq_len == 0:
        return 0.0
    n = len(sequences)
    total_entropy = 0.0
    for pos in range(seq_len):
        counts: dict[str, int] = {}
        for seq in sequences:
            aa = seq[pos].upper()
            counts[aa] = counts.get(aa, 0) + 1
        pos_entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / n
                pos_entropy -= p * math.log(p)
        total_entropy += pos_entropy / math.log(20)
    return total_entropy / seq_len


def validate_amino_acids(sequence: str) -> dict:
    if not sequence or not sequence.strip():
        return {"valid": False, "invalid_chars": set(), "fraction_valid": 0.0}
    upper = sequence.upper()
    chars = set(upper)
    invalid = chars - STANDARD_AAS
    valid_count = sum(1 for c in upper if c in STANDARD_AAS)
    return {
        "valid": len(invalid) == 0,
        "invalid_chars": invalid,
        "fraction_valid": valid_count / len(upper),
    }


def check_length_constraints(
    sequence: str,
    length_range: tuple[int, int] | None,
) -> dict:
    length = len(sequence)
    if length_range is None:
        return {"length": length, "within_range": True, "range": None}
    min_len, max_len = length_range
    return {
        "length": length,
        "within_range": min_len <= length <= max_len,
        "range": length_range,
    }


def hydrophobicity_profile(sequence: str) -> dict:
    if not sequence:
        return {"mean": 0.0, "std": 0.0, "fraction_hydrophobic": 0.0, "min": 0.0, "max": 0.0}
    values = [_KD_SCALE.get(aa.upper(), 0.0) for aa in sequence]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    hydrophobic_count = sum(1 for v in values if v > 0)
    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "fraction_hydrophobic": round(hydrophobic_count / n, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def count_mutations(wt: str, designed: str) -> int:
    if len(wt) != len(designed):
        return -1
    return sum(a != b for a, b in zip(wt.upper(), designed.upper()))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Approach Scoring  (from biodesignbench/eval/metrics/approach.py)
# ═══════════════════════════════════════════════════════════════════════════════


class DesignFunction(str, Enum):
    """Functional capabilities that tools provide."""

    BACKBONE_GENERATION = "backbone_generation"
    SEQUENCE_DESIGN = "sequence_design"
    STRUCTURE_PREDICTION = "structure_prediction"
    COMPLEX_PREDICTION = "complex_prediction"
    INTERFACE_ANALYSIS = "interface_analysis"
    STABILITY_SCORING = "stability_scoring"
    ENERGY_MINIMIZATION = "energy_minimization"
    HOTSPOT_IDENTIFICATION = "hotspot_identification"
    SEQUENCE_SCORING = "sequence_scoring"
    PHYSICS_VALIDATION = "physics_validation"


TOOL_CATEGORIES: dict[str, str] = {
    "alphafold2": "structure_prediction", "alphafold": "structure_prediction",
    "af2": "structure_prediction", "esmfold": "structure_prediction",
    "openfold": "structure_prediction", "boltz": "structure_prediction",
    "colabfold": "structure_prediction", "omegafold": "structure_prediction",
    "rosettafold": "structure_prediction",
    "proteinmpnn": "sequence_design", "mpnn": "sequence_design",
    "esm_if": "sequence_design", "ligandmpnn": "sequence_design",
    "rfdiffusion": "backbone_generation", "rfdiff": "backbone_generation",
    "chroma": "backbone_generation", "framediff": "backbone_generation",
    "foldingdiff": "backbone_generation",
    "rosetta": "energy_optimization", "pyrosetta": "energy_optimization",
    "foldx": "energy_optimization", "openmm": "energy_optimization",
    "amber": "energy_optimization", "esm2": "energy_optimization",
    "foldseek": "structure_search", "dali": "structure_search",
    "tmalign": "structure_search",
}

MCP_TOOL_EXPANSION: dict[str, list[str]] = {
    "design_binder": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "validate_design": ["esmfold", "alphafold2"],
    "optimize_sequence": ["proteinmpnn"],
    "predict_complex": ["alphafold2"],
    "analyze_interface": ["pyrosetta"],
    "predict_structure": ["esmfold", "alphafold2"],
    "score_stability": ["esm2"],
    "energy_minimize": ["openmm"],
    "suggest_hotspots": [],
    "get_design_status": [],
    "generate_backbone": ["rfdiffusion"],
    "rosetta_score": ["pyrosetta"],
    "rosetta_relax": ["pyrosetta"],
    "rosetta_interface_score": ["pyrosetta"],
    "rosetta_design": ["pyrosetta"],
    "predict_structure_boltz": ["boltz"],
    "predict_affinity_boltz": ["boltz"],
}

TOOL_TO_FUNCTION: dict[str, set[DesignFunction]] = {
    # MCP wrappers
    "design_binder": {DesignFunction.BACKBONE_GENERATION, DesignFunction.SEQUENCE_DESIGN, DesignFunction.STRUCTURE_PREDICTION},
    "validate_design": {DesignFunction.STRUCTURE_PREDICTION},
    "optimize_sequence": {DesignFunction.SEQUENCE_DESIGN},
    "predict_complex": {DesignFunction.COMPLEX_PREDICTION, DesignFunction.STRUCTURE_PREDICTION},
    "analyze_interface": {DesignFunction.INTERFACE_ANALYSIS},
    "predict_structure": {DesignFunction.STRUCTURE_PREDICTION},
    "score_stability": {DesignFunction.STABILITY_SCORING},
    "energy_minimize": {DesignFunction.ENERGY_MINIMIZATION},
    "suggest_hotspots": {DesignFunction.HOTSPOT_IDENTIFICATION},
    "get_design_status": set(),
    "generate_backbone": {DesignFunction.BACKBONE_GENERATION},
    "rosetta_score": {DesignFunction.PHYSICS_VALIDATION},
    "rosetta_relax": {DesignFunction.ENERGY_MINIMIZATION},
    "rosetta_interface_score": {DesignFunction.INTERFACE_ANALYSIS},
    "rosetta_design": {DesignFunction.SEQUENCE_DESIGN},
    "predict_structure_boltz": {DesignFunction.STRUCTURE_PREDICTION},
    "predict_affinity_boltz": {DesignFunction.COMPLEX_PREDICTION, DesignFunction.INTERFACE_ANALYSIS},
    # Bio-level tools
    "rfdiffusion": {DesignFunction.BACKBONE_GENERATION},
    "proteinmpnn": {DesignFunction.SEQUENCE_DESIGN},
    "alphafold2": {DesignFunction.STRUCTURE_PREDICTION, DesignFunction.COMPLEX_PREDICTION},
    "alphafold": {DesignFunction.STRUCTURE_PREDICTION, DesignFunction.COMPLEX_PREDICTION},
    "esmfold": {DesignFunction.STRUCTURE_PREDICTION},
    "esm2": {DesignFunction.STABILITY_SCORING, DesignFunction.SEQUENCE_SCORING},
    "pyrosetta": {DesignFunction.ENERGY_MINIMIZATION, DesignFunction.PHYSICS_VALIDATION, DesignFunction.INTERFACE_ANALYSIS},
    "rosetta": {DesignFunction.ENERGY_MINIMIZATION, DesignFunction.PHYSICS_VALIDATION, DesignFunction.INTERFACE_ANALYSIS},
    "openmm": {DesignFunction.ENERGY_MINIMIZATION},
    "boltz": {DesignFunction.STRUCTURE_PREDICTION, DesignFunction.COMPLEX_PREDICTION},
    "foldx": {DesignFunction.STABILITY_SCORING, DesignFunction.PHYSICS_VALIDATION},
    "colabfold": {DesignFunction.STRUCTURE_PREDICTION, DesignFunction.COMPLEX_PREDICTION},
    "foldseek": {DesignFunction.STRUCTURE_PREDICTION},
    "chroma": {DesignFunction.BACKBONE_GENERATION},
    "ligandmpnn": {DesignFunction.SEQUENCE_DESIGN},
    "esm_if": {DesignFunction.SEQUENCE_DESIGN},
    "mpnn": {DesignFunction.SEQUENCE_DESIGN},
}


class _TaskTypeDict(dict):
    """Dict that accepts both DesignTaskType enum and string keys."""

    def __init__(self, raw: dict[str, set[DesignFunction]]):
        super().__init__()
        self._raw = raw
        for k, v in raw.items():
            super().__setitem__(k, v)

    def __contains__(self, key):
        k = key.value if hasattr(key, "value") else key
        return super().__contains__(k)

    def __getitem__(self, key):
        k = key.value if hasattr(key, "value") else key
        return super().__getitem__(k)

    def get(self, key, default=None):
        k = key.value if hasattr(key, "value") else key
        return super().get(k, default)


REQUIRED_FUNCTIONS = _TaskTypeDict({
    "de_novo_binder": {DesignFunction.BACKBONE_GENERATION, DesignFunction.SEQUENCE_DESIGN, DesignFunction.STRUCTURE_PREDICTION},
    "sequence_optimization": {DesignFunction.SEQUENCE_DESIGN, DesignFunction.STRUCTURE_PREDICTION},
    "de_novo_backbone": {DesignFunction.BACKBONE_GENERATION, DesignFunction.SEQUENCE_DESIGN, DesignFunction.STRUCTURE_PREDICTION},
    "complex_engineering": {DesignFunction.SEQUENCE_DESIGN, DesignFunction.COMPLEX_PREDICTION},
    "conformational_design": {DesignFunction.SEQUENCE_DESIGN, DesignFunction.STRUCTURE_PREDICTION},
})

BONUS_FUNCTIONS = _TaskTypeDict({
    "de_novo_binder": {DesignFunction.COMPLEX_PREDICTION, DesignFunction.INTERFACE_ANALYSIS, DesignFunction.ENERGY_MINIMIZATION, DesignFunction.HOTSPOT_IDENTIFICATION},
    "sequence_optimization": {DesignFunction.STABILITY_SCORING, DesignFunction.ENERGY_MINIMIZATION, DesignFunction.PHYSICS_VALIDATION},
    "de_novo_backbone": {DesignFunction.ENERGY_MINIMIZATION, DesignFunction.PHYSICS_VALIDATION},
    "complex_engineering": {DesignFunction.BACKBONE_GENERATION, DesignFunction.INTERFACE_ANALYSIS, DesignFunction.ENERGY_MINIMIZATION, DesignFunction.STRUCTURE_PREDICTION},
    "conformational_design": {DesignFunction.STABILITY_SCORING, DesignFunction.ENERGY_MINIMIZATION, DesignFunction.COMPLEX_PREDICTION},
})

_GENERATION_TOOLS: set[str] = {
    "rfdiffusion", "proteinmpnn", "design_binder", "optimize_sequence",
    "generate_backbone", "rosetta_design", "chroma", "ligandmpnn",
    "esm_if", "mpnn",
}

_VALIDATION_TOOLS: set[str] = {
    "esmfold", "alphafold2", "validate_design", "predict_structure",
    "predict_complex", "score_stability", "rosetta_score",
    "rosetta_interface_score", "predict_structure_boltz",
    "predict_affinity_boltz", "analyze_interface",
}

_REFINEMENT_TOOLS: set[str] = {
    "energy_minimize", "rosetta_relax", "openmm", "pyrosetta", "rosetta",
}


def expand_mcp_tools(tools: list[str]) -> list[str]:
    """Expand MCP wrapper tool names to their underlying bio tools."""
    seen: set[str] = set()
    expanded: list[str] = []
    for tool in tools:
        if tool in MCP_TOOL_EXPANSION:
            underlying = MCP_TOOL_EXPANSION[tool]
            if not underlying:
                if tool not in seen:
                    expanded.append(tool)
                    seen.add(tool)
            else:
                for ut in underlying:
                    if ut not in seen:
                        expanded.append(ut)
                        seen.add(ut)
        else:
            if tool not in seen:
                expanded.append(tool)
                seen.add(tool)
    return expanded


def normalize_tool_name(tool: str) -> str:
    return tool.lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def get_tool_category(tool: str) -> str | None:
    normalized = normalize_tool_name(tool)
    for name, category in TOOL_CATEGORIES.items():
        if normalize_tool_name(name) == normalized:
            return category
    return None


def _extract_functions_from_tools(tools: list[str]) -> set[DesignFunction]:
    functions: set[DesignFunction] = set()
    for tool in tools:
        if tool in TOOL_TO_FUNCTION:
            functions.update(TOOL_TO_FUNCTION[tool])
        else:
            norm = normalize_tool_name(tool)
            for known, funcs in TOOL_TO_FUNCTION.items():
                if normalize_tool_name(known) == norm:
                    functions.update(funcs)
                    break
    return functions


def _check_validation(tools_used: list[str]) -> float:
    if not tools_used:
        return 0.0
    has_generation = False
    has_validation_after_generation = False
    has_any_validation = False
    for tool in tools_used:
        if tool in _GENERATION_TOOLS:
            has_generation = True
        if tool in _VALIDATION_TOOLS:
            has_any_validation = True
            if has_generation:
                has_validation_after_generation = True
    if has_validation_after_generation:
        return 4.0
    if has_any_validation:
        return 2.0
    return 0.0


def _check_refinement(tools_used: list[str]) -> float:
    if not tools_used:
        return 0.0
    for tool in tools_used:
        if tool in _REFINEMENT_TOOLS:
            return 4.0
    counts = Counter(tools_used)
    for tool, count in counts.items():
        if count >= 2 and (tool in _GENERATION_TOOLS or tool in _VALIDATION_TOOLS):
            return 4.0
    return 0.0


def _score_approach_legacy(
    tools_used: list[str],
    tools_expected: list[str],
    max_points: int = 20,
) -> dict:
    if not tools_expected:
        return {
            "score": max_points, "max": max_points,
            "breakdown": [], "tools_matched": [], "tools_missing": [],
            "mode": "legacy",
        }
    expanded_used = expand_mcp_tools(tools_used)
    per_tool = max_points / len(tools_expected)
    used_normalized = [normalize_tool_name(t) for t in expanded_used]
    used_categories = [get_tool_category(t) for t in expanded_used]
    total = 0.0
    breakdown = []
    matched = []
    missing = []
    for expected in tools_expected:
        expected_norm = normalize_tool_name(expected)
        expected_cat = get_tool_category(expected)
        if expected_norm in used_normalized:
            total += per_tool
            breakdown.append({"tool": expected, "match": "exact", "points": per_tool})
            matched.append(expected)
        elif expected_cat and expected_cat in used_categories:
            points = per_tool * 0.7
            total += points
            breakdown.append({"tool": expected, "match": "category", "points": points})
            matched.append(expected)
        else:
            breakdown.append({"tool": expected, "match": "none", "points": 0})
            missing.append(expected)
    return {
        "score": int(round(total)), "max": max_points,
        "breakdown": breakdown, "tools_matched": matched,
        "tools_missing": missing, "mode": "legacy",
    }


def score_approach(
    tools_used: list[str],
    tools_expected: list[str],
    max_points: int = 20,
    task_type: DesignTaskType | str | None = None,
) -> dict:
    """Score the agent's tool/methodology selection."""
    if task_type is None:
        return _score_approach_legacy(tools_used, tools_expected, max_points)

    tt_key = task_type.value if hasattr(task_type, "value") else str(task_type)
    scale = max_points / 20.0
    func_max = 12.0 * scale

    agent_functions = _extract_functions_from_tools(tools_used)
    required = REQUIRED_FUNCTIONS.get(tt_key, set())
    bonus = BONUS_FUNCTIONS.get(tt_key, set())

    if required:
        covered_required = agent_functions & required
        required_ratio = len(covered_required) / len(required)
    else:
        required_ratio = 1.0 if agent_functions else 0.0
        covered_required = set()

    covered_bonus = agent_functions & bonus
    bonus_count = min(len(covered_bonus), 3)
    func_score = (required_ratio * 9.0 + bonus_count * 1.0) * scale
    func_score = min(func_score, func_max)

    val_score = _check_validation(tools_used) * scale
    ref_score = _check_refinement(tools_used) * scale

    total = min(func_score + val_score + ref_score, float(max_points))

    return {
        "score": int(round(total)), "max": max_points, "mode": "function",
        "function_coverage": round(func_score, 1),
        "validation_inclusion": round(val_score, 1),
        "iterative_refinement": round(ref_score, 1),
        "required_functions": sorted(f.value for f in required),
        "covered_required": sorted(f.value for f in covered_required),
        "covered_bonus": sorted(f.value for f in covered_bonus),
        "agent_functions": sorted(f.value for f in agent_functions),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Orchestration Scoring  (from biodesignbench/eval/metrics/orchestration.py)
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_PIPELINES: dict[str, list[str]] = {
    "de_novo_binder": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "sequence_optimization": ["proteinmpnn", "esmfold"],
    "de_novo_backbone": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "complex_engineering": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "conformational_design": ["proteinmpnn", "esmfold"],
    # Old category names (backward compat)
    "binder": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "antibody": ["proteinmpnn", "esmfold"],
    "stability": ["proteinmpnn", "esmfold"],
    "enzyme": ["rfdiffusion", "proteinmpnn", "esmfold"],
}

ORCHESTRATION_VALIDATION_TOOLS: set[str] = {
    "validate_design", "predict_complex", "analyze_interface",
    "esmfold", "score_stability", "rosetta_score",
    "rosetta_interface_score", "predict_structure_boltz",
    "predict_affinity_boltz",
}


def _expand_tool_name(tool: str) -> list[str]:
    if tool in MCP_TOOL_EXPANSION:
        underlying = MCP_TOOL_EXPANSION[tool]
        return underlying if underlying else [tool]
    return [tool]


def _extract_ordered_bio_tools(tool_call_log: list[dict[str, Any]]) -> list[str]:
    utility_tools = {"execute_python", "read_file", "write_file"}
    ordered: list[str] = []
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        if tool in utility_tools:
            continue
        expanded = _expand_tool_name(tool)
        for t in expanded:
            ordered.append(normalize_tool_name(t))
    return ordered


def _longest_ordered_subsequence_length(
    actual: list[str], expected: list[str]
) -> int:
    if not expected or not actual:
        return 0
    j = 0
    matched = 0
    for tool in actual:
        k = j
        while k < len(expected):
            if tool == normalize_tool_name(expected[k]):
                matched += 1
                j = k + 1
                break
            k += 1
    return matched


def _count_validation_steps(tool_call_log: list[dict[str, Any]]) -> int:
    count = 0
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        if tool in ORCHESTRATION_VALIDATION_TOOLS:
            count += 1
        expanded = _expand_tool_name(tool)
        for t in expanded:
            if t in ORCHESTRATION_VALIDATION_TOOLS and tool not in ORCHESTRATION_VALIDATION_TOOLS:
                count += 1
    return count


def _has_adaptive_behavior(tool_call_log: list[dict[str, Any]]) -> bool:
    tool_args: dict[str, list[dict]] = {}
    for entry in tool_call_log:
        tool = entry.get("tool", "")
        args = entry.get("args_summary", {})
        if tool not in tool_args:
            tool_args[tool] = []
        tool_args[tool].append(args)
    for tool, args_list in tool_args.items():
        if len(args_list) >= 2:
            for i in range(1, len(args_list)):
                if args_list[i] != args_list[i - 1]:
                    return True
    return False


def _get_task_category_for_orchestration(task_id: str) -> str | None:
    """Extract category from task_id using taxonomy, with legacy fallback."""
    category = get_category(task_id)
    if category is not None:
        return category.task_type.value
    for cat in ("binder", "antibody", "stability", "enzyme"):
        if task_id.startswith(cat):
            return cat
    return None


def score_orchestration(
    tool_call_log: list[dict[str, Any]],
    task_id: str,
    max_points: int = 15,
) -> dict[str, Any]:
    """Score the agent's multi-step pipeline orchestration."""
    if not tool_call_log:
        return {
            "score": 0, "max": max_points,
            "pipeline_order_score": 0.0, "validation_score": 0.0,
            "adaptive_score": 0.0, "details": "No tool calls recorded",
        }

    category = _get_task_category_for_orchestration(task_id)
    expected_pipeline = EXPECTED_PIPELINES.get(category, [])

    ordered_tools = _extract_ordered_bio_tools(tool_call_log)
    if expected_pipeline:
        matched = _longest_ordered_subsequence_length(ordered_tools, expected_pipeline)
        order_ratio = matched / len(expected_pipeline)
    else:
        order_ratio = 1.0 if ordered_tools else 0.0

    pipeline_points = order_ratio * max_points * 0.5

    validation_count = _count_validation_steps(tool_call_log)
    if validation_count >= 2:
        validation_ratio = 1.0
    elif validation_count == 1:
        validation_ratio = 0.6
    else:
        validation_ratio = 0.0
    validation_points = validation_ratio * max_points * 0.3

    adaptive = _has_adaptive_behavior(tool_call_log)
    adaptive_points = max_points * 0.2 if adaptive else 0.0

    total = int(round(pipeline_points + validation_points + adaptive_points))

    return {
        "score": min(total, max_points), "max": max_points,
        "pipeline_order_score": round(pipeline_points, 1),
        "validation_score": round(validation_points, 1),
        "adaptive_score": round(adaptive_points, 1),
        "expected_pipeline": expected_pipeline,
        "actual_tool_order": ordered_tools,
        "validation_steps": validation_count,
        "adaptive_behavior": adaptive,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Quality + Scoring  (from biodesignbench/eval/tier2/scoring.py)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DESIGN_RUBRIC = {
    "approach": 20, "orchestration": 15, "quality": 35,
    "feasibility": 15, "novelty": 5, "diversity": 10,
}

METRIC_RANGES: dict[str, tuple[float, float]] = {
    "pLDDT": (0, 100), "pTM": (0, 1), "ipTM": (0, 1),
    "i_pAE": (0, 50), "predicted_kd": (0, 1e6),
    "predicted_ddG": (-100, 100), "active_site_rmsd": (0, 50),
    "max_sequence_identity": (0, 1), "TM_score": (0, 1),
}

THRESHOLD_TO_METRIC: dict[str, tuple[str, str]] = {
    "pLDDT_good": ("pLDDT", "higher_is_better"),
    "ipTM_good": ("ipTM", "higher_is_better"),
    "kd_nM_good": ("predicted_kd", "lower_is_better"),
    "predicted_ddG_good": ("predicted_ddG", "lower_is_better"),
    "active_site_rmsd_good": ("active_site_rmsd", "lower_is_better"),
}

# Tier A: Structure Confidence
_TIER_A_THRESHOLDS: dict[str, dict[str, float]] = {
    "pLDDT": {"pass": 65, "good": 80, "excellent": 90},
    "pTM": {"pass": 0.45, "good": 0.65, "excellent": 0.80},
}

# Tier B: Interface Confidence (binding only)
_TIER_B_THRESHOLDS: dict[str, dict[str, float]] = {
    "ipTM": {"pass": 0.15, "good": 0.40, "excellent": 0.70},
    "i_pAE": {"pass": 25.0, "good": 15.0, "excellent": 8.0},
}
_TIER_B_DIRECTIONS: dict[str, str] = {"i_pAE": "lower_is_better"}

# Tier C: Interface Physics
_TIER_C_METRICS: dict[str, tuple[str, str]] = {
    "kd_nM_good": ("predicted_kd", "lower_is_better"),
    "predicted_ddG_good": ("predicted_ddG", "lower_is_better"),
    "active_site_rmsd_good": ("active_site_rmsd", "lower_is_better"),
}
_TIER_C_PHYSICS: dict[str, dict[str, float]] = {
    "buried_surface_area": {"pass": 800, "good": 1500, "excellent": 2500},
    "hydrogen_bonds": {"pass": 5, "good": 15, "excellent": 30},
}

_TIER_A_BASE = 15
_TIER_B_BASE = 10
_TIER_C_BASE = 10
_QUALITY_BASE = _TIER_A_BASE + _TIER_B_BASE + _TIER_C_BASE  # 35

_BINDING_TASK_TYPES: set[DesignTaskType] = {
    DesignTaskType.DE_NOVO_BINDER,
    DesignTaskType.COMPLEX_ENGINEERING,
}
_BINDING_OLD_PREFIXES: set[str] = {"binder", "antibody", "ppi", "peptide"}


def _is_binding_task(task_id: str | None) -> bool:
    if not task_id:
        return False
    cat = get_category(task_id)
    if cat is not None:
        return cat.task_type in _BINDING_TASK_TYPES
    prefix = task_id.split("_")[0]
    return prefix in _BINDING_OLD_PREFIXES


def _get_tier_weights(
    task_id: str | None = None,
    max_points: int = 35,
) -> tuple[int, int, int]:
    if not task_id:
        scale = max_points / _QUALITY_BASE if _QUALITY_BASE > 0 else 0
        return (
            int(round(_TIER_A_BASE * scale)),
            int(round(_TIER_B_BASE * scale)),
            int(round(_TIER_C_BASE * scale)),
        )
    is_binding = _is_binding_task(task_id)
    cat = get_category(task_id)
    if cat is None and not is_binding:
        scale = max_points / _QUALITY_BASE if _QUALITY_BASE > 0 else 0
        return (
            int(round(_TIER_A_BASE * scale)),
            int(round(_TIER_B_BASE * scale)),
            int(round(_TIER_C_BASE * scale)),
        )
    if is_binding:
        ratio_a = 12 / 35
        ratio_b = 18 / 35
        a = int(round(max_points * ratio_a))
        b = int(round(max_points * ratio_b))
        c = max_points - a - b
        return (a, b, c)
    else:
        ratio_a = 25 / 35
        ratio_b = 10 / 35
        a = int(round(max_points * ratio_a))
        b = int(round(max_points * ratio_b))
        c = max_points - a - b
        return (a, b, c)


def _continuous_score(
    value: float,
    thresholds: dict[str, float],
    direction: str = "higher_is_better",
) -> float:
    """Return continuous fraction [0.0, 1.0] via linear interpolation."""
    p, g, e = thresholds["pass"], thresholds["good"], thresholds["excellent"]

    if direction == "lower_is_better":
        floor = p + abs(p) * 0.3 if p != 0 else 0.3
        if value <= e:
            return 1.0
        if value >= floor:
            return 0.0
        if value <= g:
            span = g - e
            if span == 0:
                return 1.0
            return 0.66 + (g - value) / span * 0.34
        if value <= p:
            span = p - g
            if span == 0:
                return 0.66
            return 0.33 + (p - value) / span * 0.33
        span = floor - p
        if span == 0:
            return 0.0
        return 0.33 * (floor - value) / span

    # higher_is_better
    floor = p * 0.7
    if value >= e:
        return 1.0
    if value <= floor:
        return 0.0
    if value >= g:
        span = e - g
        if span == 0:
            return 1.0
        return 0.66 + (value - g) / span * 0.34
    if value >= p:
        span = g - p
        if span == 0:
            return 0.66
        return 0.33 + (value - p) / span * 0.33
    span = p - floor
    if span == 0:
        return 0.0
    return 0.33 * (value - floor) / span


# Category-specific quality metrics (17 valid taxonomy cells)
QUALITY_METRICS: dict[tuple[DesignTaskType, BiologicalContext], dict[str, Any]] = {
    # de_novo_binder (4 cells)
    (DesignTaskType.DE_NOVO_BINDER, BiologicalContext.ANTIBODY): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.75, "good": 0.50, "pass": 0.20},
        "secondary_metrics": ["pLDDT", "predicted_kd"],
    },
    (DesignTaskType.DE_NOVO_BINDER, BiologicalContext.SIGNALING): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_kd"],
    },
    (DesignTaskType.DE_NOVO_BINDER, BiologicalContext.THERAPEUTIC): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_kd"],
    },
    (DesignTaskType.DE_NOVO_BINDER, BiologicalContext.ENZYME): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_kd", "active_site_rmsd"],
    },
    # sequence_optimization (5 cells)
    (DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.ANTIBODY): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 90, "good": 80, "pass": 65},
        "secondary_metrics": ["ipTM", "max_sequence_identity"],
    },
    (DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.ENZYME): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 90, "good": 80, "pass": 65},
        "secondary_metrics": ["predicted_ddG", "active_site_rmsd"],
    },
    (DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.STRUCTURAL): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 92, "good": 82, "pass": 68},
        "secondary_metrics": ["TM_score", "predicted_ddG"],
    },
    (DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.FLUORESCENT): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 88, "good": 78, "pass": 62},
        "secondary_metrics": ["predicted_ddG", "max_sequence_identity"],
    },
    (DesignTaskType.SEQUENCE_OPTIMIZATION, BiologicalContext.SIGNALING): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 90, "good": 80, "pass": 65},
        "secondary_metrics": ["ipTM", "predicted_ddG"],
    },
    # de_novo_backbone (1 cell)
    (DesignTaskType.DE_NOVO_BACKBONE, BiologicalContext.STRUCTURAL): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 88, "good": 78, "pass": 60},
        "secondary_metrics": ["TM_score", "predicted_ddG"],
    },
    # complex_engineering (3 cells)
    (DesignTaskType.COMPLEX_ENGINEERING, BiologicalContext.SIGNALING): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.72, "good": 0.48, "pass": 0.20},
        "secondary_metrics": ["pLDDT", "predicted_kd"],
    },
    (DesignTaskType.COMPLEX_ENGINEERING, BiologicalContext.STRUCTURAL): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.72, "good": 0.48, "pass": 0.20},
        "secondary_metrics": ["pLDDT", "TM_score"],
    },
    (DesignTaskType.COMPLEX_ENGINEERING, BiologicalContext.ENZYME): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_kd", "active_site_rmsd"],
    },
    # conformational_design (4 cells)
    (DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.ENZYME): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 88, "good": 78, "pass": 62},
        "secondary_metrics": ["predicted_ddG", "active_site_rmsd"],
    },
    (DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.SIGNALING): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 85, "good": 75, "pass": 60},
        "secondary_metrics": ["ipTM", "predicted_kd"],
    },
    (DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.FLUORESCENT): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 85, "good": 75, "pass": 60},
        "secondary_metrics": ["predicted_ddG", "max_sequence_identity"],
    },
    (DesignTaskType.CONFORMATIONAL_DESIGN, BiologicalContext.STRUCTURAL): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 88, "good": 78, "pass": 62},
        "secondary_metrics": ["TM_score", "predicted_ddG"],
    },
}


def get_quality_config(task_id: str) -> dict[str, Any] | None:
    category = get_category(task_id)
    if category is None:
        return None
    key = (category.task_type, category.context)
    return QUALITY_METRICS.get(key)


@dataclass
class DesignScoringRubric:
    components: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_DESIGN_RUBRIC))

    @property
    def max_score(self) -> int:
        return sum(self.components.values())

    def validate(self) -> None:
        total = sum(self.components.values())
        if total != 100:
            raise ValueError(f"Rubric total must be 100, got {total}")


def _has_reasonable_composition(seq: str, min_length: int = 20) -> bool:
    upper = seq.upper()
    if len(upper) < min_length:
        return False
    unique_aas = len(set(upper))
    if unique_aas < 5:
        return False
    counts = Counter(upper)
    max_fraction = max(counts.values()) / len(upper)
    if max_fraction > 0.5:
        return False
    ala_fraction = counts.get("A", 0) / len(upper)
    if ala_fraction > 0.3:
        return False
    hp = hydrophobicity_profile(upper)
    if hp["mean"] > 2.0:
        return False
    return True


def validate_metric_range(name: str, value: float) -> bool:
    if name not in METRIC_RANGES:
        return True
    low, high = METRIC_RANGES[name]
    return low <= value <= high


# Functional Similarity thresholds for non-binding Tier B
_FUNCTIONAL_SIM_DEFAULTS: dict[DesignTaskType, dict[str, float]] = {
    DesignTaskType.SEQUENCE_OPTIMIZATION: {"pass": 0.40, "good": 0.60, "excellent": 0.85},
    DesignTaskType.CONFORMATIONAL_DESIGN: {"pass": 0.15, "good": 0.30, "excellent": 0.50},
    DesignTaskType.DE_NOVO_BACKBONE: {"pass": 0.10, "good": 0.20, "excellent": 0.40},
}


def _derive_functional_sim_thresholds(value: float) -> dict[str, float]:
    return {
        "pass": value * 0.5,
        "good": value,
        "excellent": min(value * 2, 1.0),
    }


def _get_functional_sim_thresholds(
    thresholds: dict[str, float],
    task_id: str,
) -> dict[str, float] | None:
    if _is_binding_task(task_id):
        return None
    gt_value = thresholds.get("max_seq_identity_good")
    if gt_value is not None:
        return _derive_functional_sim_thresholds(gt_value)
    cat = get_category(task_id)
    if cat is None:
        return None
    return _FUNCTIONAL_SIM_DEFAULTS.get(cat.task_type)


def _score_functional_similarity(
    designs: list[str],
    oracle_sequences: list[str],
    thresholds: dict[str, float],
) -> float | None:
    if not designs or not oracle_sequences:
        return None
    best_identity = 0.0
    for design in designs:
        for oracle in oracle_sequences:
            ident = sequence_identity(design, oracle)
            if ident > best_identity:
                best_identity = ident
    return _continuous_score(best_identity, thresholds, "higher_is_better")


def score_quality(
    agent_metrics: dict[str, float],
    thresholds: dict[str, float],
    max_points: int = 35,
    task_id: str | None = None,
    designs: list[str] | None = None,
    oracle_sequences: list[str] | None = None,
) -> dict[str, Any]:
    """Score quality using 3-tier continuous system."""
    valid_metrics = {
        k: v for k, v in agent_metrics.items() if validate_metric_range(k, v)
    }
    for extra_key in ("buried_surface_area", "hydrogen_bonds"):
        if extra_key in agent_metrics and extra_key not in valid_metrics:
            val = agent_metrics[extra_key]
            if isinstance(val, (int, float)) and val >= 0:
                valid_metrics[extra_key] = float(val)

    tier_a_max, tier_b_max, tier_c_max = _get_tier_weights(task_id, max_points)
    is_binding = _is_binding_task(task_id)

    overrides: dict[str, dict[str, float]] = {}
    if task_id:
        config = get_quality_config(task_id)
        if config and "thresholds" in config:
            primary = config["primary_metric"]
            overrides[primary] = config["thresholds"]

    # Tier A: Structure Confidence
    tier_a_scores: dict[str, float] = {}
    for metric, default_thresh in _TIER_A_THRESHOLDS.items():
        if metric in valid_metrics:
            thresh = overrides.get(metric, default_thresh)
            tier_a_scores[metric] = _continuous_score(
                valid_metrics[metric], thresh, "higher_is_better"
            )
    tier_a_pts = (sum(tier_a_scores.values()) / len(tier_a_scores)) * tier_a_max if tier_a_scores else 0.0

    # Tier B: Interface or Functional Similarity
    tier_b_scores: dict[str, float] = {}
    tier_b_pts = 0.0
    _use_functional_sim = (
        tier_b_max > 0
        and task_id is not None
        and not is_binding
        and get_category(task_id) is not None
    )

    if tier_b_max > 0:
        if _use_functional_sim:
            if designs and oracle_sequences:
                func_thresh = _get_functional_sim_thresholds(thresholds, task_id)
                if func_thresh is not None:
                    frac = _score_functional_similarity(designs, oracle_sequences, func_thresh)
                    if frac is not None:
                        tier_b_pts = frac * tier_b_max
                        tier_b_scores["oracle_identity"] = frac
        else:
            for metric, default_thresh in _TIER_B_THRESHOLDS.items():
                if metric in valid_metrics:
                    thresh = overrides.get(metric, default_thresh)
                    direction = _TIER_B_DIRECTIONS.get(metric, "higher_is_better")
                    tier_b_scores[metric] = _continuous_score(
                        valid_metrics[metric], thresh, direction
                    )
            if tier_b_scores:
                tier_b_pts = (sum(tier_b_scores.values()) / len(tier_b_scores)) * tier_b_max

    # Tier C: Interface Physics
    tier_c_fractions: list[float] = []
    tier_c_breakdown: list[dict] = []

    if tier_c_max > 0:
        if is_binding:
            for metric_key, phys_thresh in _TIER_C_PHYSICS.items():
                if metric_key in valid_metrics:
                    frac = _continuous_score(valid_metrics[metric_key], phys_thresh, "higher_is_better")
                    tier_c_fractions.append(frac)
                    tier_c_breakdown.append({
                        "threshold": metric_key, "metric": metric_key,
                        "value": valid_metrics[metric_key],
                        "threshold_value": phys_thresh, "fraction": round(frac, 3),
                    })

        for thresh_key, (metric_key, direction) in _TIER_C_METRICS.items():
            if thresh_key in thresholds and metric_key in valid_metrics:
                threshold_val = thresholds[thresh_key]
                agent_val = valid_metrics[metric_key]
                margin = abs(threshold_val) * 0.5 if threshold_val != 0 else 1.0
                if direction == "lower_is_better":
                    gt_thresh = {
                        "pass": threshold_val + margin,
                        "good": threshold_val,
                        "excellent": threshold_val - margin,
                    }
                else:
                    gt_thresh = {
                        "pass": threshold_val - margin,
                        "good": threshold_val,
                        "excellent": threshold_val + margin,
                    }
                frac = _continuous_score(agent_val, gt_thresh, direction)
                tier_c_fractions.append(frac)
                tier_c_breakdown.append({
                    "threshold": thresh_key, "metric": metric_key,
                    "value": agent_val, "threshold_value": threshold_val,
                    "fraction": round(frac, 3),
                })

    tier_c_pts = (sum(tier_c_fractions) / len(tier_c_fractions)) * tier_c_max if tier_c_fractions else 0.0

    total = min(tier_a_pts + tier_b_pts + tier_c_pts, max_points)
    metrics_evaluated = len(tier_a_scores) + len(tier_b_scores) + len(tier_c_fractions)

    return {
        "score": int(round(total)), "max": max_points,
        "tier_a": round(tier_a_pts, 1), "tier_b": round(tier_b_pts, 1),
        "tier_c": round(tier_c_pts, 1),
        "metrics_evaluated": metrics_evaluated,
        "breakdown": {
            "structure": tier_a_scores, "interface": tier_b_scores,
            "physics": tier_c_breakdown,
        },
    }


def score_novelty(
    designs: list[str],
    reference_seq: str | None,
    thresholds: dict[str, float],
    max_points: int = 5,
) -> dict[str, Any]:
    """Score novelty by computing sequence identity to reference."""
    if not designs:
        return {"score": 0, "max": max_points, "max_identity": 0.0, "identity_threshold": None}

    identity_threshold = thresholds.get("max_seq_identity_good")
    max_id = max_identity_to_reference(designs, reference_seq) if reference_seq else 0.0

    if identity_threshold is None:
        if reference_seq:
            novelty_ratio = 1.0 - max_id
            score = int(round(max_points * min(novelty_ratio * 2, 1.0)))
        else:
            score = max_points
    elif identity_threshold >= 0.9:
        if max_id >= identity_threshold:
            score = max_points
        elif max_id >= identity_threshold * 0.9:
            score = int(round(max_points * 0.7))
        else:
            score = int(round(max_points * 0.3))
    else:
        if max_id <= identity_threshold:
            score = max_points
        elif max_id <= identity_threshold * 1.5:
            score = int(round(max_points * 0.5))
        else:
            score = int(round(max_points * 0.2))

    return {
        "score": min(score, max_points), "max": max_points,
        "max_identity": round(max_id, 3), "identity_threshold": identity_threshold,
    }


def score_diversity(
    designs: list[str],
    max_designs: int = 10,
    max_points: int = 5,
) -> dict[str, Any]:
    """Score diversity of designs."""
    if not designs:
        return {"score": 0, "max": max_points, "num_designs": 0, "pairwise_diversity": 0.0, "entropy": 0.0}

    num = len(designs)
    count_fraction = min(num / max_designs, 1.0) if max_designs > 0 else 1.0
    diversity = mean_pairwise_diversity(designs)
    entropy = sequence_entropy(designs)

    count_score = count_fraction * max_points * 0.4
    diversity_score = diversity * max_points * 0.4
    entropy_score = entropy * max_points * 0.2
    total = int(round(count_score + diversity_score + entropy_score))

    return {
        "score": min(total, max_points), "max": max_points,
        "num_designs": num, "pairwise_diversity": round(diversity, 3),
        "entropy": round(entropy, 3),
    }


def score_feasibility(
    designs: list[str],
    constraints: dict[str, Any],
    max_points: int = 25,
) -> dict[str, Any]:
    """Score feasibility of designed sequences."""
    if not designs:
        return {"score": 0, "max": max_points, "aa_validity": 0.0, "length_validity": 0.0, "composition_check": 0.0}

    per_check = max_points / 3
    length_range = constraints.get("length_range")
    if isinstance(length_range, list):
        length_range = tuple(length_range)

    comp_min_length = 20
    if length_range and length_range[1] < 20:
        comp_min_length = max(length_range[0], 5)

    aa_valid_count = sum(1 for seq in designs if validate_amino_acids(seq)["valid"])
    aa_fraction = aa_valid_count / len(designs)

    length_valid_count = sum(1 for seq in designs if check_length_constraints(seq, length_range)["within_range"])
    length_fraction = length_valid_count / len(designs)

    composition_ok = sum(1 for seq in designs if _has_reasonable_composition(seq, min_length=comp_min_length))
    composition_fraction = composition_ok / len(designs)

    aa_score = aa_fraction * per_check
    length_score = length_fraction * per_check
    comp_score = composition_fraction * per_check
    total = int(round(aa_score + length_score + comp_score))

    return {
        "score": min(total, max_points), "max": max_points,
        "aa_validity": round(aa_fraction, 3),
        "length_validity": round(length_fraction, 3),
        "composition_check": round(composition_fraction, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Design Gate + Final Score
# ═══════════════════════════════════════════════════════════════════════════════

_DESIGN_GATE_ZEROED = {"quality", "novelty", "diversity", "feasibility"}
_DESIGN_GATE_CAP = 30


def apply_design_gate(
    component_scores: dict[str, int],
    num_designs: int,
) -> dict[str, int]:
    """If no designs produced, cap total at 30."""
    if num_designs >= 1:
        return dict(component_scores)
    gated = dict(component_scores)
    for key in _DESIGN_GATE_ZEROED:
        gated[key] = 0
    remaining_sum = sum(v for k, v in gated.items() if k not in _DESIGN_GATE_ZEROED)
    if remaining_sum > _DESIGN_GATE_CAP:
        scale = _DESIGN_GATE_CAP / remaining_sum
        for key in gated:
            if key not in _DESIGN_GATE_ZEROED:
                gated[key] = int(round(gated[key] * scale))
    return gated


def calculate_design_score(
    rubric: DesignScoringRubric,
    results: dict[str, int],
) -> dict[str, Any]:
    """Calculate final design task score from component results."""
    breakdown = {}
    for component, max_pts in rubric.components.items():
        actual = min(results.get(component, 0), max_pts)
        breakdown[component] = {"score": actual, "max": max_pts}
    total = sum(v["score"] for v in breakdown.values())
    max_possible = rubric.max_score
    return {
        "breakdown": breakdown,
        "total": total,
        "max_possible": max_possible,
        "percentage": round(total / max_possible * 100, 1) if max_possible > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Full Task Scorer (high-level API for eval pipeline)
# ═══════════════════════════════════════════════════════════════════════════════


def score_submission_task(
    task_id: str,
    sequences: list[str],
    run_log: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    agent_metrics: dict[str, float] | None = None,
    oracle_sequences: list[str] | None = None,
) -> dict[str, Any]:
    """Score a single task submission end-to-end.

    This is the main entry point for the evaluation pipeline.

    Args:
        task_id: Task identifier (e.g., "dnb_sig_001").
        sequences: Designed amino acid sequences from the agent.
        run_log: Tool call log from the agent.
        ground_truth: Ground truth dict with thresholds, reference_sequence,
            design_constraints, tools_expected, max_designs.
        agent_metrics: Optional metrics reported by the agent or from Boltz
            (e.g., {"pLDDT": 85.0, "ipTM": 0.35}).
        oracle_sequences: Optional oracle sequences for functional similarity.

    Returns:
        Dict with: total_score, component_scores, details, num_designs.
    """
    if agent_metrics is None:
        agent_metrics = {}

    # Extract fields from ground truth
    thresholds = ground_truth.get("thresholds", {})
    reference_seq = ground_truth.get("reference_sequence")
    constraints = ground_truth.get("design_constraints", {})
    tools_expected = ground_truth.get("tools_expected", [])
    max_designs = ground_truth.get("max_designs", 10)

    # Get task category for function-based scoring
    cat = get_category(task_id)
    task_type = cat.task_type if cat else None

    # Extract tools used from run_log
    tools_used = [entry.get("tool", "") for entry in run_log if entry.get("tool")]

    # Score all 6 components
    approach_result = score_approach(
        tools_used=tools_used,
        tools_expected=tools_expected,
        task_type=task_type,
    )
    orchestration_result = score_orchestration(
        tool_call_log=run_log,
        task_id=task_id,
    )
    quality_result = score_quality(
        agent_metrics=agent_metrics,
        thresholds=thresholds,
        task_id=task_id,
        designs=sequences,
        oracle_sequences=oracle_sequences,
    )
    feasibility_result = score_feasibility(
        designs=sequences,
        constraints=constraints,
    )
    novelty_result = score_novelty(
        designs=sequences,
        reference_seq=reference_seq,
        thresholds=thresholds,
    )
    diversity_result = score_diversity(
        designs=sequences,
        max_designs=max_designs,
    )

    # Build component scores dict
    component_scores = {
        "approach": approach_result["score"],
        "orchestration": orchestration_result["score"],
        "quality": quality_result["score"],
        "feasibility": feasibility_result["score"],
        "novelty": novelty_result["score"],
        "diversity": diversity_result["score"],
    }

    # Apply design gate
    num_designs = len(sequences)
    gated = apply_design_gate(component_scores, num_designs)
    total = sum(gated.values())

    return {
        "total_score": total,
        "component_scores": gated,
        "num_designs": num_designs,
        "details": {
            "approach": approach_result,
            "orchestration": orchestration_result,
            "quality": quality_result,
            "feasibility": feasibility_result,
            "novelty": novelty_result,
            "diversity": diversity_result,
        },
    }


def aggregate_scores(
    per_task_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-task scores into an overall submission result.

    If `eval_judge.run_judge_panel()` has been run beforehand each task
    will carry `hybrid_scores` and `hybrid_total`; in that case we use
    the hybrid (algo + LLM judge, capped at rubric max) as the canonical
    score. Otherwise we fall back to the algo-only `component_scores` /
    `total_score` produced by the dispatcher + Boltz pipeline.
    """
    if not per_task_scores:
        return {
            "overall_score": 0.0,
            "component_scores": {c: 0.0 for c in DEFAULT_DESIGN_RUBRIC},
            "taxonomy_scores": {},
            "tasks_completed": 0,
            "tasks_total": 0,
            "tasks_with_zero": 0,
        }

    totals = {c: 0.0 for c in DEFAULT_DESIGN_RUBRIC}
    n = len(per_task_scores)
    tasks_with_zero = 0
    used_hybrid = False

    # Taxonomy breakdown
    taxonomy_scores: dict[str, dict[str, list[float]]] = {}

    for task_id, result in per_task_scores.items():
        if "hybrid_scores" in result and "hybrid_total" in result:
            comp_scores = result["hybrid_scores"]
            total_score = result["hybrid_total"]
            used_hybrid = True
        else:
            comp_scores = result.get("component_scores", {})
            total_score = result.get("total_score", 0.0)

        if total_score == 0:
            tasks_with_zero += 1

        for comp, val in comp_scores.items():
            totals[comp] += val

        # Taxonomy mapping
        cat = get_category(task_id)
        if cat:
            tt = cat.task_type.value
            ctx = cat.context.short
            taxonomy_scores.setdefault(tt, {}).setdefault(ctx, []).append(total_score)

    # Average components
    avg_components = {c: round(v / n, 1) for c, v in totals.items()}
    overall = round(sum(avg_components.values()), 1)

    # Average taxonomy scores
    taxonomy_avg: dict[str, dict[str, float]] = {}
    for tt, contexts in taxonomy_scores.items():
        taxonomy_avg[tt] = {}
        for ctx, scores in contexts.items():
            taxonomy_avg[tt][ctx] = round(sum(scores) / len(scores), 1)

    return {
        "overall_score": overall,
        "component_scores": avg_components,
        "taxonomy_scores": taxonomy_avg,
        "tasks_completed": n,
        "tasks_total": n,
        "tasks_with_zero": tasks_with_zero,
        "scoring_mode": "hybrid" if used_hybrid else "algo",
    }
