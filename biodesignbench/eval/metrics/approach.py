"""Score whether an agent chose appropriate tools for the design problem.

Two scoring modes:

1. **Function-based** (new, preferred): When ``task_type`` is provided,
   scoring evaluates which *design functions* the agent's tools cover
   (e.g. backbone_generation, sequence_design, structure_prediction)
   rather than checking exact tool names.

2. **Legacy** (backward-compatible): When ``task_type is None``, falls
   back to exact-tool-matching with category-level partial credit.

Tool Naming Layers
~~~~~~~~~~~~~~~~~~
BioDesignBench uses a two-layer tool naming scheme:

1. **Bio-level names** — canonical names for the underlying algorithms
   (e.g. ``rfdiffusion``, ``proteinmpnn``, ``esmfold``, ``alphafold2``).

2. **MCP wrapper names** — high-level tool names exposed to agents via the
   MCP protein design server (e.g. ``design_binder``, ``validate_design``).

``MCP_TOOL_EXPANSION`` maps wrapper names → bio-level names.
``TOOL_TO_FUNCTION`` maps both layers → ``DesignFunction`` enums.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biodesignbench.taxonomy import DesignApproach


# ---------------------------------------------------------------------------
# DesignFunction enum
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool category mappings (backward compat)
# ---------------------------------------------------------------------------

TOOL_CATEGORIES: dict[str, str] = {
    # Structure prediction
    "alphafold2": "structure_prediction",
    "alphafold": "structure_prediction",
    "af2": "structure_prediction",
    "esmfold": "structure_prediction",
    "openfold": "structure_prediction",
    "boltz": "structure_prediction",
    "colabfold": "structure_prediction",
    "omegafold": "structure_prediction",
    "rosettafold": "structure_prediction",
    # Sequence design
    "proteinmpnn": "sequence_design",
    "mpnn": "sequence_design",
    "esm_if": "sequence_design",
    "ligandmpnn": "sequence_design",
    # Backbone generation
    "rfdiffusion": "backbone_generation",
    "rfdiff": "backbone_generation",
    "chroma": "backbone_generation",
    "framediff": "backbone_generation",
    "foldingdiff": "backbone_generation",
    # Energy optimization
    "rosetta": "energy_optimization",
    "pyrosetta": "energy_optimization",
    "foldx": "energy_optimization",
    "openmm": "energy_optimization",
    "amber": "energy_optimization",
    "esm2": "energy_optimization",
    # Structure search
    "foldseek": "structure_search",
    "dali": "structure_search",
    "tmalign": "structure_search",
}


# ---------------------------------------------------------------------------
# MCP wrapper expansion
# ---------------------------------------------------------------------------

MCP_TOOL_EXPANSION: dict[str, list[str]] = {
    "design_binder": ["rfdiffusion", "proteinmpnn", "esmfold"],
    "design_fold": ["rfdiffusion", "proteinmpnn", "alphafold2"],
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
    "design_sequence": ["proteinmpnn"],
}


# ---------------------------------------------------------------------------
# TOOL_TO_FUNCTION — maps every tool name to its DesignFunction(s)
# ---------------------------------------------------------------------------

TOOL_TO_FUNCTION: dict[str, set[DesignFunction]] = {
    # MCP wrappers
    "design_binder": {
        DesignFunction.BACKBONE_GENERATION,
        DesignFunction.SEQUENCE_DESIGN,
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "design_fold": {
        DesignFunction.BACKBONE_GENERATION,
        DesignFunction.SEQUENCE_DESIGN,
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "validate_design": {
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "optimize_sequence": {
        DesignFunction.SEQUENCE_DESIGN,
    },
    "predict_complex": {
        DesignFunction.COMPLEX_PREDICTION,
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "analyze_interface": {
        DesignFunction.INTERFACE_ANALYSIS,
    },
    "predict_structure": {
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "score_stability": {
        DesignFunction.STABILITY_SCORING,
    },
    "energy_minimize": {
        DesignFunction.ENERGY_MINIMIZATION,
    },
    "suggest_hotspots": {
        DesignFunction.HOTSPOT_IDENTIFICATION,
    },
    "get_design_status": set(),  # utility
    "generate_backbone": {
        DesignFunction.BACKBONE_GENERATION,
    },
    "rosetta_score": {
        DesignFunction.PHYSICS_VALIDATION,
    },
    "rosetta_relax": {
        DesignFunction.ENERGY_MINIMIZATION,
    },
    "rosetta_interface_score": {
        DesignFunction.INTERFACE_ANALYSIS,
    },
    "rosetta_design": {
        DesignFunction.SEQUENCE_DESIGN,
    },
    "design_sequence": {
        DesignFunction.SEQUENCE_DESIGN,
    },
    "predict_structure_boltz": {
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "predict_affinity_boltz": {
        DesignFunction.COMPLEX_PREDICTION,
        DesignFunction.INTERFACE_ANALYSIS,
    },
    # Bio-level tools
    "rfdiffusion": {
        DesignFunction.BACKBONE_GENERATION,
    },
    "proteinmpnn": {
        DesignFunction.SEQUENCE_DESIGN,
    },
    "alphafold2": {
        DesignFunction.STRUCTURE_PREDICTION,
        DesignFunction.COMPLEX_PREDICTION,
    },
    "alphafold": {
        DesignFunction.STRUCTURE_PREDICTION,
        DesignFunction.COMPLEX_PREDICTION,
    },
    "esmfold": {
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "esm2": {
        DesignFunction.STABILITY_SCORING,
        DesignFunction.SEQUENCE_SCORING,
    },
    "pyrosetta": {
        DesignFunction.ENERGY_MINIMIZATION,
        DesignFunction.PHYSICS_VALIDATION,
        DesignFunction.INTERFACE_ANALYSIS,
    },
    "rosetta": {
        DesignFunction.ENERGY_MINIMIZATION,
        DesignFunction.PHYSICS_VALIDATION,
        DesignFunction.INTERFACE_ANALYSIS,
    },
    "openmm": {
        DesignFunction.ENERGY_MINIMIZATION,
    },
    "boltz": {
        DesignFunction.STRUCTURE_PREDICTION,
        DesignFunction.COMPLEX_PREDICTION,
    },
    "foldx": {
        DesignFunction.STABILITY_SCORING,
        DesignFunction.PHYSICS_VALIDATION,
    },
    "colabfold": {
        DesignFunction.STRUCTURE_PREDICTION,
        DesignFunction.COMPLEX_PREDICTION,
    },
    "foldseek": {
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "chroma": {
        DesignFunction.BACKBONE_GENERATION,
    },
    "ligandmpnn": {
        DesignFunction.SEQUENCE_DESIGN,
    },
    "esm_if": {
        DesignFunction.SEQUENCE_DESIGN,
    },
    "mpnn": {
        DesignFunction.SEQUENCE_DESIGN,
    },
}


# ---------------------------------------------------------------------------
# REQUIRED_FUNCTIONS and BONUS_FUNCTIONS per DesignTaskType
# ---------------------------------------------------------------------------

# Lazy import to avoid circular dependency — we import the actual enum
# values only when needed.  The keys use string values matching the enum.

def _get_task_type_enum(name: str):
    """Lazy import of DesignApproach enum member."""
    from biodesignbench.taxonomy import DesignApproach
    return DesignApproach(name)


# Using string keys internally, exposed as DesignTaskType via property
_REQUIRED_FUNCTIONS_RAW: dict[str, set[DesignFunction]] = {
    "de_novo": {
        DesignFunction.BACKBONE_GENERATION,
        DesignFunction.SEQUENCE_DESIGN,
        DesignFunction.STRUCTURE_PREDICTION,
    },
    "redesign": {
        DesignFunction.SEQUENCE_DESIGN,
        DesignFunction.STRUCTURE_PREDICTION,
    },
}

_BONUS_FUNCTIONS_RAW: dict[str, set[DesignFunction]] = {
    "de_novo": {
        DesignFunction.COMPLEX_PREDICTION,
        DesignFunction.INTERFACE_ANALYSIS,
        DesignFunction.ENERGY_MINIMIZATION,
        DesignFunction.HOTSPOT_IDENTIFICATION,
    },
    "redesign": {
        DesignFunction.STABILITY_SCORING,
        DesignFunction.ENERGY_MINIMIZATION,
        DesignFunction.PHYSICS_VALIDATION,
        DesignFunction.COMPLEX_PREDICTION,
    },
}


class _TaskTypeDict(dict):
    """Dict that accepts both DesignTaskType enum and string keys."""

    def __init__(self, raw: dict[str, set[DesignFunction]]):
        super().__init__()
        self._raw = raw
        # Pre-populate with string keys
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


REQUIRED_FUNCTIONS = _TaskTypeDict(_REQUIRED_FUNCTIONS_RAW)
BONUS_FUNCTIONS = _TaskTypeDict(_BONUS_FUNCTIONS_RAW)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Tools that count as "generation/design" steps (for validation ordering)
_GENERATION_TOOLS: set[str] = {
    "rfdiffusion", "proteinmpnn", "design_binder", "design_fold",
    "optimize_sequence", "generate_backbone", "design_sequence",
    "rosetta_design", "chroma", "ligandmpnn", "esm_if", "mpnn",
}

# Tools that count as "validation" steps
_VALIDATION_TOOLS: set[str] = {
    "esmfold", "alphafold2", "validate_design", "predict_structure",
    "predict_complex", "score_stability", "rosetta_score",
    "rosetta_interface_score", "predict_structure_boltz",
    "predict_affinity_boltz", "analyze_interface",
}

# Tools that count as "refinement" (energy minimization / relaxation)
_REFINEMENT_TOOLS: set[str] = {
    "energy_minimize", "rosetta_relax", "openmm", "pyrosetta", "rosetta",
}


def expand_mcp_tools(tools: list[str]) -> list[str]:
    """Expand MCP wrapper tool names to their underlying bio tools.

    Non-MCP tools pass through unchanged.  Duplicates are removed while
    preserving order (first occurrence wins).
    """
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
    """Normalize a tool name for comparison: lowercase, strip whitespace."""
    return tool.lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def get_tool_category(tool: str) -> str | None:
    """Get the category for a tool name."""
    normalized = normalize_tool_name(tool)
    for name, category in TOOL_CATEGORIES.items():
        if normalize_tool_name(name) == normalized:
            return category
    return None


def _extract_functions_from_tools(tools: list[str]) -> set[DesignFunction]:
    """Map a list of tool names to the set of DesignFunctions they cover.

    Handles both MCP wrapper names and bio-level names.
    Unknown tools are silently ignored.
    """
    functions: set[DesignFunction] = set()
    for tool in tools:
        # Direct lookup
        if tool in TOOL_TO_FUNCTION:
            functions.update(TOOL_TO_FUNCTION[tool])
        # Try normalized lookup
        else:
            norm = normalize_tool_name(tool)
            for known, funcs in TOOL_TO_FUNCTION.items():
                if normalize_tool_name(known) == norm:
                    functions.update(funcs)
                    break
    return functions


def _check_validation(tools_used: list[str]) -> float:
    """Check if validation occurs after a generation step.

    Returns:
        4.0 — validation after generation (ideal)
        2.0 — validation present but no prior generation
        0.0 — no validation tools
    """
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
    """Check for iterative refinement in the pipeline.

    Returns:
        4.0 — refinement/relaxation present, OR same design tool called 2+
        0.0 — no refinement detected
    """
    if not tools_used:
        return 0.0

    # Check for explicit refinement tools
    for tool in tools_used:
        if tool in _REFINEMENT_TOOLS:
            return 4.0

    # Check for same tool called multiple times (iterative design)
    counts = Counter(tools_used)
    for tool, count in counts.items():
        if count >= 2 and (tool in _GENERATION_TOOLS or tool in _VALIDATION_TOOLS):
            return 4.0

    return 0.0


# ---------------------------------------------------------------------------
# Main scoring functions
# ---------------------------------------------------------------------------


def _score_approach_legacy(
    tools_used: list[str],
    tools_expected: list[str],
    max_points: int = 20,
) -> dict:
    """Legacy scoring: exact tool matching with category partial credit.

    Kept for backward compatibility when task_type is not available.
    """
    if not tools_expected:
        return {
            "score": max_points,
            "max": max_points,
            "breakdown": [],
            "tools_matched": [],
            "tools_missing": [],
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
        "score": int(round(total)),
        "max": max_points,
        "breakdown": breakdown,
        "tools_matched": matched,
        "tools_missing": missing,
        "mode": "legacy",
    }


def score_approach(
    tools_used: list[str],
    tools_expected: list[str],
    max_points: int = 20,
    task_type: DesignApproach | str | None = None,
) -> dict:
    """Score the agent's tool/methodology selection.

    When *task_type* is provided, uses function-based scoring:
      - function_coverage (12 pts): Required + bonus function coverage
      - validation_inclusion (4 pts): Validation after generation
      - iterative_refinement (4 pts): Energy minimization or repeated design

    When *task_type* is ``None``, falls back to legacy exact-tool matching.

    Args:
        tools_used: List of tool names the agent actually used.
        tools_expected: List of tool names expected for this task.
        max_points: Maximum points for the approach component.
        task_type: DesignTaskType enum or string value. None for legacy mode.

    Returns:
        Dict with: score, max, mode, and scoring details.
    """
    # Legacy fallback
    if task_type is None:
        return _score_approach_legacy(tools_used, tools_expected, max_points)

    # Resolve task_type to string key
    tt_key = task_type.value if hasattr(task_type, "value") else str(task_type)

    # Scale point allocations proportionally
    scale = max_points / 20.0
    func_max = 12.0 * scale
    val_max = 4.0 * scale
    ref_max = 4.0 * scale

    # Extract functions from tools
    agent_functions = _extract_functions_from_tools(tools_used)

    required = REQUIRED_FUNCTIONS.get(tt_key, set())
    bonus = BONUS_FUNCTIONS.get(tt_key, set())

    # --- Function coverage (12 pts base) ---
    if required:
        covered_required = agent_functions & required
        required_ratio = len(covered_required) / len(required)
    else:
        required_ratio = 1.0 if agent_functions else 0.0
        covered_required = set()

    covered_bonus = agent_functions & bonus
    bonus_count = min(len(covered_bonus), 3)

    # 9 pts for required coverage + 1 pt per bonus (up to 3)
    func_score = (required_ratio * 9.0 + bonus_count * 1.0) * scale
    func_score = min(func_score, func_max)

    # --- Validation inclusion (4 pts base) ---
    val_score = _check_validation(tools_used) * scale

    # --- Iterative refinement (4 pts base) ---
    ref_score = _check_refinement(tools_used) * scale

    total = func_score + val_score + ref_score
    total = min(total, float(max_points))

    return {
        "score": int(round(total)),
        "max": max_points,
        "mode": "function",
        "function_coverage": round(func_score, 1),
        "validation_inclusion": round(val_score, 1),
        "iterative_refinement": round(ref_score, 1),
        "required_functions": sorted(f.value for f in required),
        "covered_required": sorted(f.value for f in covered_required),
        "covered_bonus": sorted(f.value for f in covered_bonus),
        "agent_functions": sorted(f.value for f in agent_functions),
    }
