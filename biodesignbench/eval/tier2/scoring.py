"""100-point scoring rubric for Tier 2 design tasks.

Six components:
  - approach (20 pts):        Tool/methodology selection
  - orchestration (15 pts):   Pipeline ordering + intermediate validation
  - quality (35 pts):         3-tier continuous scoring (structure/interface/physics)
  - feasibility (15 pts):     Valid AAs, length, composition + biophysical checks
  - novelty (5 pts):          Sequence identity to known sequences
  - diversity (10 pts):       Number + diversity of designs

Quality scoring uses continuous linear interpolation (no cliff effects):
  Non-binding tasks: Tier A = 25, Tier B = 10 (functional similarity), Tier C = 0
  Binding tasks:     Tier A = 12, Tier B = 18, Tier C = 5

  Tier A: Structure confidence — pLDDT, pTM
  Tier B: Interface confidence (binding) or functional similarity (non-binding)
  Tier C: Interface physics — BSA, H-bonds, ddG, Kd, RMSD (binding only)

Binding vs non-binding is determined by ground truth thresholds: a task is
binding if its ground truth contains ``ipTM_good``.  This decouples scoring
from the taxonomy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from biodesignbench.eval.metrics.sequence import (
    check_length_constraints,
    hydrophobicity_profile,
    max_identity_to_reference,
    mean_pairwise_diversity,
    sequence_entropy,
    sequence_identity,
    validate_amino_acids,
)
from biodesignbench.taxonomy import (
    DesignApproach,
    MolecularSubject,
    get_category,
)


DEFAULT_DESIGN_RUBRIC = {
    "approach": 20,
    "orchestration": 15,
    "quality": 35,
    "feasibility": 15,
    "novelty": 5,
    "diversity": 10,
}

# Valid ranges for agent-reported metric values
METRIC_RANGES: dict[str, tuple[float, float]] = {
    "pLDDT": (0, 100),
    "pTM": (0, 1),
    "ipTM": (0, 1),
    "i_pAE": (0, 50),
    "predicted_kd": (0, 1e6),
    "predicted_ddG": (-100, 100),
    "active_site_rmsd": (0, 50),
    "max_sequence_identity": (0, 1),
    "TM_score": (0, 1),
}

# Map ground truth threshold keys to (metric_key, direction)
# Retained for backward compatibility; score_quality uses tier-based scoring.
THRESHOLD_TO_METRIC: dict[str, tuple[str, str]] = {
    "pLDDT_good": ("pLDDT", "higher_is_better"),
    "ipTM_good": ("ipTM", "higher_is_better"),
    "kd_nM_good": ("predicted_kd", "lower_is_better"),
    "predicted_ddG_good": ("predicted_ddG", "lower_is_better"),
    "active_site_rmsd_good": ("active_site_rmsd", "lower_is_better"),
}


# ---------------------------------------------------------------------------
# Graduated quality thresholds (BindCraft/Germinal-inspired)
# ---------------------------------------------------------------------------

# Tier A: Structure Confidence (pLDDT, pTM)
_TIER_A_THRESHOLDS: dict[str, dict[str, float]] = {
    "pLDDT": {"pass": 65, "good": 80, "excellent": 90},
    "pTM": {"pass": 0.45, "good": 0.65, "excellent": 0.80},
}

# Tier B: Interface Confidence (ipTM, i_pAE)
# Recalibrated to realistic AF2-Multimer ranges (median ipTM ~ 0.14)
_TIER_B_THRESHOLDS: dict[str, dict[str, float]] = {
    "ipTM": {"pass": 0.15, "good": 0.40, "excellent": 0.70},
    "i_pAE": {"pass": 25.0, "good": 15.0, "excellent": 8.0},
}

# Tier B: direction overrides (default higher_is_better)
_TIER_B_DIRECTIONS: dict[str, str] = {
    "i_pAE": "lower_is_better",
}

# Tier C: Interface Physics — uses ground truth thresholds
_TIER_C_METRICS: dict[str, tuple[str, str]] = {
    "kd_nM_good": ("predicted_kd", "lower_is_better"),
    "predicted_ddG_good": ("predicted_ddG", "lower_is_better"),
    "active_site_rmsd_good": ("active_site_rmsd", "lower_is_better"),
}

# Tier C: Physics-based interface metrics (BSA + H-bonds)
# Scored for binding tasks only; falls back to ddG/Kd from ground truth thresholds.
_TIER_C_PHYSICS: dict[str, dict[str, float]] = {
    "buried_surface_area": {"pass": 800, "good": 1500, "excellent": 2500},
    "hydrogen_bonds": {"pass": 5, "good": 15, "excellent": 30},
}

# Base point allocation (out of 35) — legacy; overridden by _get_tier_weights
_TIER_A_BASE = 15
_TIER_B_BASE = 10
_TIER_C_BASE = 10
_QUALITY_BASE = _TIER_A_BASE + _TIER_B_BASE + _TIER_C_BASE  # 35


_BINDING_THRESHOLD_KEYS = frozenset({
    "ipTM_good", "kd_nM_good",
})


def _is_binding_task(
    task_id: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> bool:
    """Determine whether a task is a binding (interface) task.

    Uses ground truth thresholds as the primary signal, decoupling binding
    detection from the taxonomy.  Any of the binding-related threshold keys
    (ipTM_good, kd_nM_good, predicted_ddG_good, active_site_rmsd_good)
    indicates a binding task.

    Falls back to category-specific QUALITY_METRICS when thresholds are
    empty: if the category's primary metric is ``ipTM``, the task is binding.

    Args:
        task_id: Task identifier for category-based fallback.
        thresholds: Ground truth evaluation_thresholds dict.

    Returns:
        True if the task involves a protein-protein interface.
    """
    if thresholds is not None and thresholds:
        return bool(_BINDING_THRESHOLD_KEYS & thresholds.keys())
    # Fallback: check category-specific QUALITY_METRICS
    if task_id is not None:
        category = get_category(task_id)
        if category is not None:
            key = (category.approach, category.subject)
            qm = QUALITY_METRICS.get(key)
            if qm and qm.get("primary_metric") == "ipTM":
                return True
    return False


def _get_tier_weights(
    task_id: str | None = None,
    max_points: int = 35,
    thresholds: dict[str, float] | None = None,
) -> tuple[int, int, int]:
    """Return (tier_a, tier_b, tier_c) point allocations based on task type.

    Non-binding tasks:
        Tier A = 25, Tier B = 10 (functional similarity), Tier C = 0.

    Binding tasks:
        Tier A = 12, Tier B = 18, Tier C = 5 (sum = 35).

    Unknown tasks (no thresholds, no task_id): fall back to old balanced 15/10/10.
    """
    if thresholds is None and task_id is None:
        # No info at all → legacy balanced
        scale = max_points / _QUALITY_BASE if _QUALITY_BASE > 0 else 0
        return (
            int(round(_TIER_A_BASE * scale)),
            int(round(_TIER_B_BASE * scale)),
            int(round(_TIER_C_BASE * scale)),
        )

    is_binding = _is_binding_task(task_id, thresholds)

    if is_binding:
        # Binding: A=12, B=18, C=5 for max_points=35
        ratio_a = 12 / 35
        ratio_b = 18 / 35
        a = int(round(max_points * ratio_a))
        b = int(round(max_points * ratio_b))
        c = max_points - a - b
        return (a, b, c)
    else:
        # Non-binding: A=25, B=10 (functional similarity via oracle), C=0
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
    """Return continuous fraction [0.0, 1.0] via linear interpolation.

    Eliminates cliff effects from the old graduated scoring.  Four bands:

    - Below floor (pass * 0.7 for higher_is_better, pass * 1.3 for
      lower_is_better): **0.0**
    - Floor -> pass: **0.0 -> 0.33** (linear)
    - Pass -> good: **0.33 -> 0.66** (linear)
    - Good -> excellent: **0.66 -> 1.0** (linear)
    - Above excellent: **1.0**

    Args:
        value: Metric value to evaluate.
        thresholds: Dict with 'pass', 'good', 'excellent' keys.
        direction: 'higher_is_better' or 'lower_is_better'.

    Returns:
        Continuous fraction of points earned in [0.0, 1.0].
    """
    p, g, e = thresholds["pass"], thresholds["good"], thresholds["excellent"]

    if direction == "lower_is_better":
        # Invert: lower value = better -> map to higher-is-better space
        # floor is 30% worse than pass threshold (handles negative values)
        floor = p + abs(p) * 0.3 if p != 0 else 0.3
        # Map: value -> effective (floor->0, pass->0.33, good->0.66, excellent->1.0)
        if value <= e:
            return 1.0
        if value >= floor:
            return 0.0
        if value <= g:
            # Between excellent and good: 0.66 -> 1.0
            span = g - e
            if span == 0:
                return 1.0
            return 0.66 + (g - value) / span * 0.34
        if value <= p:
            # Between good and pass: 0.33 -> 0.66
            span = p - g
            if span == 0:
                return 0.66
            return 0.33 + (p - value) / span * 0.33
        # Between pass and floor: 0.0 -> 0.33
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
        # Between good and excellent: 0.66 -> 1.0
        span = e - g
        if span == 0:
            return 1.0
        return 0.66 + (value - g) / span * 0.34
    if value >= p:
        # Between pass and good: 0.33 -> 0.66
        span = g - p
        if span == 0:
            return 0.66
        return 0.33 + (value - p) / span * 0.33
    # Between floor and pass: 0.0 -> 0.33
    span = p - floor
    if span == 0:
        return 0.0
    return 0.33 * (value - floor) / span


# ---------------------------------------------------------------------------
# Category-specific quality metrics (2x5 taxonomy, 10 cells)
# ---------------------------------------------------------------------------

QUALITY_METRICS: dict[
    tuple[DesignApproach, MolecularSubject], dict[str, Any]
] = {
    # ── de_novo (5 cells) ─────────────────────────────────────────────
    # ipTM recalibrated to realistic AF2-Multimer ranges
    (DesignApproach.DE_NOVO, MolecularSubject.ANTIBODY): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.75, "good": 0.50, "pass": 0.20},
        "secondary_metrics": ["pLDDT", "predicted_kd"],
    },
    (DesignApproach.DE_NOVO, MolecularSubject.ENZYME): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_kd", "active_site_rmsd"],
    },
    (DesignApproach.DE_NOVO, MolecularSubject.BINDER): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_kd"],
    },
    (DesignApproach.DE_NOVO, MolecularSubject.SCAFFOLD): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 88, "good": 78, "pass": 60},
        "secondary_metrics": ["TM_score", "predicted_ddG"],
    },
    (DesignApproach.DE_NOVO, MolecularSubject.FLUORESCENT_PROTEIN): {
        "primary_metric": "ipTM",
        "thresholds": {"excellent": 0.70, "good": 0.45, "pass": 0.18},
        "secondary_metrics": ["pLDDT", "predicted_ddG"],
    },
    # ── redesign (5 cells) ────────────────────────────────────────────
    (DesignApproach.REDESIGN, MolecularSubject.ANTIBODY): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 90, "good": 80, "pass": 65},
        "secondary_metrics": ["ipTM", "max_sequence_identity"],
    },
    (DesignApproach.REDESIGN, MolecularSubject.ENZYME): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 90, "good": 80, "pass": 65},
        "secondary_metrics": ["predicted_ddG", "active_site_rmsd"],
    },
    (DesignApproach.REDESIGN, MolecularSubject.SCAFFOLD): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 92, "good": 82, "pass": 68},
        "secondary_metrics": ["TM_score", "predicted_ddG"],
    },
    (DesignApproach.REDESIGN, MolecularSubject.FLUORESCENT_PROTEIN): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 88, "good": 78, "pass": 62},
        "secondary_metrics": ["predicted_ddG", "max_sequence_identity"],
    },
    (DesignApproach.REDESIGN, MolecularSubject.BINDER): {
        "primary_metric": "pLDDT",
        "thresholds": {"excellent": 90, "good": 80, "pass": 65},
        "secondary_metrics": ["ipTM", "predicted_kd"],
    },
}


def get_quality_config(task_id: str) -> dict[str, Any] | None:
    """Look up category-specific quality metrics for a task ID.

    Works with both old-format IDs (``"binder_001"``) and new two-axis IDs
    (``"dn_bnd_001"``).  Returns ``None`` if the task is not recognized
    or does not map to a valid taxonomy cell.
    """
    category = get_category(task_id)
    if category is None:
        return None
    key = (category.approach, category.subject)
    return QUALITY_METRICS.get(key)


@dataclass
class DesignScoringRubric:
    """Configurable scoring rubric for design tasks."""

    components: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DESIGN_RUBRIC)
    )

    @property
    def max_score(self) -> int:
        return sum(self.components.values())

    def validate(self) -> None:
        total = sum(self.components.values())
        if total != 100:
            raise ValueError(f"Rubric total must be 100, got {total}")


def _has_reasonable_composition(seq: str, min_length: int = 20) -> bool:
    """Check if sequence has reasonable amino acid composition.

    Rejects:
    - Length < *min_length* residues (default 20, lowered for peptide tasks)
    - Fewer than 5 unique amino acids (for sequences >= min_length)
    - Any single amino acid > 50% of sequence
    - Alanine content > 30% (BindCraft-inspired; MPNNsol naturally < 15%)
    - Mean Kyte-Doolittle hydrophobicity > 2.0 (aggregation-prone)

    Args:
        seq: Amino acid sequence string.
        min_length: Minimum sequence length for composition check.

    Returns:
        True if the composition is reasonable for a designed protein.
    """
    upper = seq.upper()
    if len(upper) < min_length:
        return False

    unique_aas = len(set(upper))
    if unique_aas < 5:
        return False

    # Check no single AA dominates (> 50%)
    counts = Counter(upper)
    max_fraction = max(counts.values()) / len(upper)
    if max_fraction > 0.5:
        return False

    # Alanine content check (> 30% is biologically unrealistic for design)
    ala_fraction = counts.get("A", 0) / len(upper)
    if ala_fraction > 0.3:
        return False

    # Mean hydrophobicity check (overly hydrophobic -> aggregation-prone)
    hp = hydrophobicity_profile(upper)
    if hp["mean"] > 2.0:
        return False

    return True


def validate_metric_range(name: str, value: float) -> bool:
    """Check if a metric value falls within its valid range."""
    if name not in METRIC_RANGES:
        return True  # Unknown metrics pass by default
    low, high = METRIC_RANGES[name]
    return low <= value <= high


# ---------------------------------------------------------------------------
# Functional Similarity (oracle-based, non-binding Tier B)
# ---------------------------------------------------------------------------

# Design-approach default thresholds for oracle identity when
# max_seq_identity_good is missing from ground truth.
_FUNCTIONAL_SIM_DEFAULTS: dict[DesignApproach, dict[str, float]] = {
    DesignApproach.REDESIGN: {
        "pass": 0.40, "good": 0.60, "excellent": 0.85,
    },
    DesignApproach.DE_NOVO: {
        "pass": 0.10, "good": 0.20, "excellent": 0.40,
    },
}


def _derive_functional_sim_thresholds(value: float) -> dict[str, float]:
    """Derive pass/good/excellent from a single ``max_seq_identity_good`` value.

    - pass = value * 0.5
    - good = value
    - excellent = min(value * 2, 1.0)

    Args:
        value: The ``max_seq_identity_good`` threshold from ground truth.

    Returns:
        Dict with ``pass``, ``good``, ``excellent`` keys.
    """
    return {
        "pass": value * 0.5,
        "good": value,
        "excellent": min(value * 2, 1.0),
    }


def _get_functional_sim_thresholds(
    thresholds: dict[str, float],
    task_id: str,
) -> dict[str, float] | None:
    """Get functional similarity thresholds for a task.

    Priority:
    1. Ground truth ``max_seq_identity_good`` -> derive pass/good/excellent
    2. Design-approach defaults from ``_FUNCTIONAL_SIM_DEFAULTS``
    3. None (binding tasks or unrecognized IDs)

    Args:
        thresholds: Ground truth thresholds dict.
        task_id: Task identifier.

    Returns:
        Threshold dict or None if not applicable.
    """
    # Binding tasks don't use functional similarity
    if _is_binding_task(task_id, thresholds):
        return None

    # Priority 1: ground truth value
    gt_value = thresholds.get("max_seq_identity_good")
    if gt_value is not None:
        return _derive_functional_sim_thresholds(gt_value)

    # Priority 2: design-approach defaults
    cat = get_category(task_id)
    if cat is None:
        return None
    return _FUNCTIONAL_SIM_DEFAULTS.get(cat.approach)


def _score_functional_similarity(
    designs: list[str],
    oracle_sequences: list[str],
    thresholds: dict[str, float],
) -> float | None:
    """Score functional similarity of designs to oracle sequences.

    Computes the best sequence identity between any design and any oracle
    sequence, then maps it through ``_continuous_score`` to get a [0, 1]
    fraction.

    Args:
        designs: Agent-produced design sequences.
        oracle_sequences: Publication-validated oracle sequences.
        thresholds: Dict with pass/good/excellent identity thresholds.

    Returns:
        Fraction [0.0, 1.0] or None if scoring is impossible
        (no designs or no oracle sequences).
    """
    if not designs or not oracle_sequences:
        return None

    # Best identity: max over all (design, oracle) pairs
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
    """Score quality using 3-tier continuous system.

    Tier A -- Structure Confidence:
        pLDDT and pTM evaluated against continuous thresholds.
        If only one metric available, it gets all Tier A points.

    Tier B -- Interface Confidence (binding) or Functional Similarity (non-binding):
        Binding: ipTM and i_pAE. ONLY scored when present.
        Non-binding: Oracle sequence identity. Requires *designs* and
        *oracle_sequences* parameters.

    Tier C -- Interface Physics:
        BSA, H-bonds (from analyze_interface) for binding tasks.
        Falls back to predicted_kd, predicted_ddG, active_site_rmsd
        from ground truth thresholds.

    Point allocation depends on ground truth (binding detection via ipTM_good):
        Non-binding -> A=25, B=10 (functional similarity), C=0.
        Binding -> A=12, B=18, C=5 (of 35).

    Category-specific overrides via ``QUALITY_METRICS`` replace generic
    thresholds for the primary metric when *task_id* is provided.

    Args:
        agent_metrics: Metrics reported by the agent (e.g., {"pLDDT": 85.0}).
        thresholds: Ground truth thresholds (e.g., {"kd_nM_good": 100}).
        max_points: Maximum points for quality component (default 35).
        task_id: Optional task ID for category-specific threshold overrides.
        designs: Optional list of design sequences for oracle comparison.
        oracle_sequences: Optional list of oracle (publication-validated) sequences.

    Returns:
        Dict with: score, max, tier_a, tier_b, tier_c, metrics_evaluated,
        breakdown.
    """
    # Pre-validate metric ranges: exclude out-of-range values
    valid_metrics = {
        k: v for k, v in agent_metrics.items() if validate_metric_range(k, v)
    }
    # BSA and H-bonds are not in METRIC_RANGES — pass them through
    for extra_key in ("buried_surface_area", "hydrogen_bonds"):
        if extra_key in agent_metrics and extra_key not in valid_metrics:
            val = agent_metrics[extra_key]
            if isinstance(val, (int, float)) and val >= 0:
                valid_metrics[extra_key] = float(val)

    # Get task-type-aware tier weights (now using ground truth thresholds)
    tier_a_max, tier_b_max, tier_c_max = _get_tier_weights(
        task_id, max_points, thresholds
    )
    is_binding = _is_binding_task(task_id, thresholds)

    # Load category-specific threshold overrides
    overrides: dict[str, dict[str, float]] = {}
    if task_id:
        config = get_quality_config(task_id)
        if config and "thresholds" in config:
            primary = config["primary_metric"]
            overrides[primary] = config["thresholds"]

    # --- Tier A: Structure Confidence ---
    tier_a_scores: dict[str, float] = {}
    for metric, default_thresh in _TIER_A_THRESHOLDS.items():
        if metric in valid_metrics:
            thresh = overrides.get(metric, default_thresh)
            tier_a_scores[metric] = _continuous_score(
                valid_metrics[metric], thresh, "higher_is_better"
            )

    if tier_a_scores:
        tier_a_pts = (
            sum(tier_a_scores.values()) / len(tier_a_scores)
        ) * tier_a_max
    else:
        tier_a_pts = 0.0

    # --- Tier B: Interface Confidence (binding) or Functional Similarity (non-binding) ---
    tier_b_scores: dict[str, float] = {}
    tier_b_pts = 0.0

    # Determine whether to use functional similarity for Tier B:
    # Only for known non-binding tasks (task_id set and recognized).
    _use_functional_sim = (
        tier_b_max > 0
        and task_id is not None
        and not is_binding
        and get_category(task_id) is not None
    )

    if tier_b_max > 0:
        if _use_functional_sim:
            # Non-binding: functional similarity via oracle sequences
            if designs and oracle_sequences:
                func_thresh = _get_functional_sim_thresholds(thresholds, task_id)
                if func_thresh is not None:
                    frac = _score_functional_similarity(
                        designs, oracle_sequences, func_thresh
                    )
                    if frac is not None:
                        tier_b_pts = frac * tier_b_max
                        tier_b_scores["oracle_identity"] = frac
        else:
            # Binding or legacy (no task_id): score ipTM and i_pAE
            for metric, default_thresh in _TIER_B_THRESHOLDS.items():
                if metric in valid_metrics:
                    thresh = overrides.get(metric, default_thresh)
                    direction = _TIER_B_DIRECTIONS.get(metric, "higher_is_better")
                    tier_b_scores[metric] = _continuous_score(
                        valid_metrics[metric], thresh, direction
                    )
            if tier_b_scores:
                tier_b_pts = (
                    sum(tier_b_scores.values()) / len(tier_b_scores)
                ) * tier_b_max

    # --- Tier C: Interface Physics ---
    tier_c_fractions: list[float] = []
    tier_c_breakdown: list[dict] = []

    if tier_c_max > 0:
        # First: score BSA + H-bonds (from analyze_interface) for binding tasks
        if is_binding:
            for metric_key, phys_thresh in _TIER_C_PHYSICS.items():
                if metric_key in valid_metrics:
                    frac = _continuous_score(
                        valid_metrics[metric_key], phys_thresh, "higher_is_better"
                    )
                    tier_c_fractions.append(frac)
                    tier_c_breakdown.append({
                        "threshold": metric_key,
                        "metric": metric_key,
                        "value": valid_metrics[metric_key],
                        "threshold_value": phys_thresh,
                        "fraction": round(frac, 3),
                    })

        # Second: score ground-truth threshold metrics (ddG, Kd, RMSD)
        for thresh_key, (metric_key, direction) in _TIER_C_METRICS.items():
            if thresh_key in thresholds and metric_key in valid_metrics:
                threshold_val = thresholds[thresh_key]
                agent_val = valid_metrics[metric_key]

                # Build graduated thresholds from single ground-truth value
                # Use abs-based margins to handle negative values (e.g., ddG)
                margin = abs(threshold_val) * 0.5 if threshold_val != 0 else 1.0
                if direction == "lower_is_better":
                    # pass is worst tolerable (higher), excellent is best (lower)
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
                    "threshold": thresh_key,
                    "metric": metric_key,
                    "value": agent_val,
                    "threshold_value": threshold_val,
                    "fraction": round(frac, 3),
                })

    if tier_c_fractions:
        tier_c_pts = (
            sum(tier_c_fractions) / len(tier_c_fractions)
        ) * tier_c_max
    else:
        tier_c_pts = 0.0

    total = tier_a_pts + tier_b_pts + tier_c_pts
    total = min(total, max_points)
    metrics_evaluated = (
        len(tier_a_scores) + len(tier_b_scores) + len(tier_c_fractions)
    )

    return {
        "score": int(round(total)),
        "max": max_points,
        "tier_a": round(tier_a_pts, 1),
        "tier_b": round(tier_b_pts, 1),
        "tier_c": round(tier_c_pts, 1),
        "metrics_evaluated": metrics_evaluated,
        "breakdown": {
            "structure": tier_a_scores,
            "interface": tier_b_scores,
            "physics": tier_c_breakdown,
        },
    }


def score_novelty(
    designs: list[str],
    reference_seq: str | None,
    thresholds: dict[str, float],
    max_points: int = 5,
) -> dict[str, Any]:
    """Score novelty by computing sequence identity to reference.

    Lower identity = more novel = higher score for de novo design.
    For stability tasks (max_seq_identity_good near 1.0), high identity is desired.

    Args:
        designs: List of designed sequences.
        reference_seq: Wild-type/reference sequence.
        thresholds: Ground truth thresholds.
        max_points: Maximum points for novelty component.

    Returns:
        Dict with: score, max, max_identity, identity_threshold.
    """
    if not designs:
        return {
            "score": 0, "max": max_points,
            "max_identity": 0.0, "identity_threshold": None,
        }

    identity_threshold = thresholds.get("max_seq_identity_good")

    if reference_seq:
        max_id = max_identity_to_reference(designs, reference_seq)
    else:
        max_id = 0.0

    if identity_threshold is None:
        # No threshold -> score based on novelty (lower identity = better)
        if reference_seq:
            novelty_ratio = 1.0 - max_id
            score = int(round(max_points * min(novelty_ratio * 2, 1.0)))
        else:
            score = max_points  # No reference = full points
    elif identity_threshold >= 0.9:
        # Stability-type: high identity expected (minor mutations)
        if max_id >= identity_threshold:
            score = max_points
        elif max_id >= identity_threshold * 0.9:
            score = int(round(max_points * 0.7))
        else:
            score = int(round(max_points * 0.3))
    else:
        # Design-type: identity should be BELOW threshold (novel)
        if max_id <= identity_threshold:
            score = max_points
        elif max_id <= identity_threshold * 1.5:
            score = int(round(max_points * 0.5))
        else:
            score = int(round(max_points * 0.2))

    return {
        "score": min(score, max_points),
        "max": max_points,
        "max_identity": round(max_id, 3),
        "identity_threshold": identity_threshold,
    }


def score_diversity(
    designs: list[str],
    max_designs: int = 10,
    max_points: int = 5,
) -> dict[str, Any]:
    """Score diversity of designs.

    Components:
    - 65% for mean pairwise diversity
    - 35% for sequence entropy

    Returns:
        Dict with: score, max, num_designs, pairwise_diversity, entropy.
    """
    if not designs:
        return {
            "score": 0, "max": max_points,
            "num_designs": 0, "pairwise_diversity": 0.0, "entropy": 0.0,
        }

    num = len(designs)
    diversity = mean_pairwise_diversity(designs)
    entropy = sequence_entropy(designs)

    # Score based purely on sequence diversity (not design count).
    # Tasks don't specify how many designs to produce, so counting
    # would unfairly penalise agents that submit fewer designs.
    diversity_score = diversity * max_points * 0.65
    entropy_score = entropy * max_points * 0.35

    total = int(round(diversity_score + entropy_score))

    return {
        "score": min(total, max_points),
        "max": max_points,
        "num_designs": num,
        "pairwise_diversity": round(diversity, 3),
        "entropy": round(entropy, 3),
    }


def score_feasibility(
    designs: list[str],
    constraints: dict[str, Any],
    max_points: int = 25,
) -> dict[str, Any]:
    """Score feasibility of designed sequences.

    Components (equal weight):
    - Valid amino acids (all standard AAs)
    - Length within constraints
    - Composition (not too hydrophobic, not homopolymeric)

    Args:
        designs: List of designed sequences.
        constraints: Design constraints including length_range.
        max_points: Maximum points for feasibility component.

    Returns:
        Dict with: score, max, aa_validity, length_validity, composition_check.
    """
    if not designs:
        return {
            "score": 0, "max": max_points,
            "aa_validity": 0.0, "length_validity": 0.0, "composition_check": 0.0,
        }

    per_check = max_points / 3
    length_range = constraints.get("length_range")
    if isinstance(length_range, list):
        length_range = tuple(length_range)

    # Adapt composition min_length from length_range for peptide tasks
    comp_min_length = 20
    if length_range and length_range[1] < 20:
        # Task allows short peptides -- lower composition threshold
        comp_min_length = max(length_range[0], 5)

    # 1. AA validity
    aa_valid_count = 0
    for seq in designs:
        result = validate_amino_acids(seq)
        if result["valid"]:
            aa_valid_count += 1
    aa_fraction = aa_valid_count / len(designs)

    # 2. Length validity
    length_valid_count = 0
    for seq in designs:
        result = check_length_constraints(seq, length_range)
        if result["within_range"]:
            length_valid_count += 1
    length_fraction = length_valid_count / len(designs)

    # 3. Composition check: reasonable AA composition
    composition_ok = 0
    for seq in designs:
        if _has_reasonable_composition(seq, min_length=comp_min_length):
            composition_ok += 1
    composition_fraction = composition_ok / len(designs)

    aa_score = aa_fraction * per_check
    length_score = length_fraction * per_check
    comp_score = composition_fraction * per_check

    total = int(round(aa_score + length_score + comp_score))

    return {
        "score": min(total, max_points),
        "max": max_points,
        "aa_validity": round(aa_fraction, 3),
        "length_validity": round(length_fraction, 3),
        "composition_check": round(composition_fraction, 3),
    }


_DESIGN_GATE_ZEROED = {"quality", "novelty", "diversity", "feasibility"}
_DESIGN_GATE_CAP = 30


def apply_design_gate(
    component_scores: dict[str, int],
    num_designs: int,
) -> dict[str, int]:
    """Apply design-gated scoring: if no designs produced, cap total at 30.

    When ``num_designs == 0`` the agent produced no valid sequences.
    Quality, novelty, diversity, and feasibility are zeroed out and
    the remaining components (approach, orchestration) are
    proportionally scaled so their sum does not exceed 30.

    Args:
        component_scores: Dict mapping component names to scores.
        num_designs: Number of valid designs produced by the agent.

    Returns:
        New dict with gated scores.
    """
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
    """Calculate final design task score from component results.

    Args:
        rubric: Scoring rubric with max points per component.
        results: Dict mapping component names to actual scores.

    Returns:
        Dict with: breakdown, total, max_possible, percentage.
    """
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
