"""Deep failure mode analysis for Tier 2 protein design evaluations.

Classifies each evaluation into specific failure modes, extracts reasoning
quality signals, and produces structured analysis for paper tables.

Usage::

    analyzer = FailureModeAnalyzer()
    analysis = analyzer.analyze_run("results/runs/run_20260211_220512_578312")
    print(analyzer.to_markdown_tables(analysis))
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from biodesignbench.eval.metrics.sequence import STANDARD_AAS
from biodesignbench.eval.results import EvaluationResult


# ---------------------------------------------------------------------------
# Failure Mode Taxonomy
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    EXECUTION = "execution"
    DESIGN = "design"
    CONSTRAINT = "constraint"
    TOOL = "tool"
    REASONING = "reasoning"  # positive signals, not failures


class FailureMode(str, Enum):
    # Execution-level
    NO_OUTPUT = "NO_OUTPUT"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    NETWORK_BLOCKED = "NETWORK_BLOCKED"
    SAFETY_REFUSAL = "SAFETY_REFUSAL"
    RATE_LIMITED = "RATE_LIMITED"
    PARSE_ERROR = "PARSE_ERROR"
    TIMEOUT = "TIMEOUT"

    # Design-level
    RANDOM_SEQUENCES = "RANDOM_SEQUENCES"
    MOCK_DATA = "MOCK_DATA"
    INVALID_AA = "INVALID_AA"
    REPETITIVE_PATTERN = "REPETITIVE_PATTERN"
    POLY_AA = "POLY_AA"
    FABRICATED_METRICS = "FABRICATED_METRICS"
    ELLIPSIS_SEQUENCE = "ELLIPSIS_SEQUENCE"

    # Constraint-level
    CONSTRAINT_IGNORED_LENGTH = "CONSTRAINT_IGNORED_LENGTH"
    CONSTRAINT_IGNORED_ACTIVE_SITE = "CONSTRAINT_IGNORED_ACTIVE_SITE"
    CONSTRAINT_IGNORED_CHROMOPHORE = "CONSTRAINT_IGNORED_CHROMOPHORE"
    CONSTRAINT_IGNORED_CDR = "CONSTRAINT_IGNORED_CDR"
    CONSTRAINT_IGNORED_CYSTEINE = "CONSTRAINT_IGNORED_CYSTEINE"
    CONSTRAINT_IGNORED_IDENTITY = "CONSTRAINT_IGNORED_IDENTITY"

    # Tool-level
    NO_BIO_TOOLS = "NO_BIO_TOOLS"
    TOOL_AVAILABLE_NOT_USED = "TOOL_AVAILABLE_NOT_USED"
    CODE_ONLY_APPROACH = "CODE_ONLY_APPROACH"

    # Reasoning quality (positive signals)
    GOOD_PLAN_NO_EXECUTION = "GOOD_PLAN_NO_EXECUTION"
    CORRECT_TOOL_SELECTION = "CORRECT_TOOL_SELECTION"
    BIOLOGICAL_AWARENESS = "BIOLOGICAL_AWARENESS"
    STRUCTURAL_AWARENESS = "STRUCTURAL_AWARENESS"

    # Extended failure modes (v2)
    TOOL_SELECTION_ERROR = "TOOL_SELECTION_ERROR"
    PARAMETER_ERROR = "PARAMETER_ERROR"
    OUTPUT_PARSING_ERROR = "OUTPUT_PARSING_ERROR"
    MISSING_VALIDATION = "MISSING_VALIDATION"
    DOMAIN_KNOWLEDGE_GAP = "DOMAIN_KNOWLEDGE_GAP"
    ITERATION_FAILURE = "ITERATION_FAILURE"
    PIPELINE_ORDERING = "PIPELINE_ORDERING"
    RESOURCE_MISMANAGEMENT = "RESOURCE_MISMANAGEMENT"
    CONTEXT_MISUNDERSTANDING = "CONTEXT_MISUNDERSTANDING"


FAILURE_CATEGORY: dict[FailureMode, FailureCategory] = {
    FailureMode.NO_OUTPUT: FailureCategory.EXECUTION,
    FailureMode.EMPTY_OUTPUT: FailureCategory.EXECUTION,
    FailureMode.NETWORK_BLOCKED: FailureCategory.EXECUTION,
    FailureMode.SAFETY_REFUSAL: FailureCategory.EXECUTION,
    FailureMode.RATE_LIMITED: FailureCategory.EXECUTION,
    FailureMode.PARSE_ERROR: FailureCategory.EXECUTION,
    FailureMode.TIMEOUT: FailureCategory.EXECUTION,
    FailureMode.RANDOM_SEQUENCES: FailureCategory.DESIGN,
    FailureMode.MOCK_DATA: FailureCategory.DESIGN,
    FailureMode.INVALID_AA: FailureCategory.DESIGN,
    FailureMode.REPETITIVE_PATTERN: FailureCategory.DESIGN,
    FailureMode.POLY_AA: FailureCategory.DESIGN,
    FailureMode.FABRICATED_METRICS: FailureCategory.DESIGN,
    FailureMode.ELLIPSIS_SEQUENCE: FailureCategory.DESIGN,
    FailureMode.CONSTRAINT_IGNORED_LENGTH: FailureCategory.CONSTRAINT,
    FailureMode.CONSTRAINT_IGNORED_ACTIVE_SITE: FailureCategory.CONSTRAINT,
    FailureMode.CONSTRAINT_IGNORED_CHROMOPHORE: FailureCategory.CONSTRAINT,
    FailureMode.CONSTRAINT_IGNORED_CDR: FailureCategory.CONSTRAINT,
    FailureMode.CONSTRAINT_IGNORED_CYSTEINE: FailureCategory.CONSTRAINT,
    FailureMode.CONSTRAINT_IGNORED_IDENTITY: FailureCategory.CONSTRAINT,
    FailureMode.NO_BIO_TOOLS: FailureCategory.TOOL,
    FailureMode.TOOL_AVAILABLE_NOT_USED: FailureCategory.TOOL,
    FailureMode.CODE_ONLY_APPROACH: FailureCategory.TOOL,
    FailureMode.GOOD_PLAN_NO_EXECUTION: FailureCategory.REASONING,
    FailureMode.CORRECT_TOOL_SELECTION: FailureCategory.REASONING,
    FailureMode.BIOLOGICAL_AWARENESS: FailureCategory.REASONING,
    FailureMode.STRUCTURAL_AWARENESS: FailureCategory.REASONING,
    # Extended failure modes (v2)
    FailureMode.TOOL_SELECTION_ERROR: FailureCategory.TOOL,
    FailureMode.PARAMETER_ERROR: FailureCategory.TOOL,
    FailureMode.OUTPUT_PARSING_ERROR: FailureCategory.EXECUTION,
    FailureMode.MISSING_VALIDATION: FailureCategory.DESIGN,
    FailureMode.DOMAIN_KNOWLEDGE_GAP: FailureCategory.REASONING,
    FailureMode.ITERATION_FAILURE: FailureCategory.EXECUTION,
    FailureMode.PIPELINE_ORDERING: FailureCategory.TOOL,
    FailureMode.RESOURCE_MISMANAGEMENT: FailureCategory.EXECUTION,
    FailureMode.CONTEXT_MISUNDERSTANDING: FailureCategory.REASONING,
}


# ---------------------------------------------------------------------------
# Report Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FailureModeReport:
    """Analysis report for a single evaluation."""

    task_id: str
    agent_id: str
    failure_modes: list[str] = field(default_factory=list)
    reasoning_signals: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    severity: str = "none"  # "none", "minor", "major", "critical"
    score: float = 0.0
    summary: str = ""


@dataclass
class AgentFailureProfile:
    """Aggregated failure profile for one agent across all tasks."""

    agent_id: str
    total_evaluations: int = 0
    failure_mode_counts: dict[str, int] = field(default_factory=dict)
    failure_category_counts: dict[str, int] = field(default_factory=dict)
    reasoning_signal_counts: dict[str, int] = field(default_factory=dict)
    mean_score: float = 0.0
    severity_distribution: dict[str, int] = field(default_factory=dict)
    per_category_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class FailureAnalysisResult:
    """Complete failure analysis output."""

    run_id: str = "unknown"
    evaluations: list[FailureModeReport] = field(default_factory=list)
    agent_profiles: dict[str, AgentFailureProfile] = field(default_factory=dict)
    heatmap_data: dict[str, dict[str, int]] = field(default_factory=dict)
    category_heatmap: dict[str, dict[str, float]] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------

_RANDOM_CODE_PATTERNS = [
    re.compile(r"random\.choice\s*\(.*(?:amino|aa|residue)", re.IGNORECASE),
    re.compile(r"random\.choices\s*\(.*(?:amino|aa|residue)", re.IGNORECASE),
    re.compile(r"random\.sample\s*\(.*(?:amino|aa|residue)", re.IGNORECASE),
    re.compile(r"''\s*\.join\s*\(\s*random\.choice", re.IGNORECASE),
]

_MOCK_PATTERNS = [
    re.compile(r"(?i)mock(?:sequence|data|value|metric|result)"),
    re.compile(r"(?i)placeholder"),
    re.compile(r"(?i)#\s*(?:mock|fake|dummy)"),
    re.compile(r"(?i)for\s+demonstration"),
    re.compile(r"(?i)hypothetical\s+(?:metric|value|stability|Tm|pLDDT|score)"),
]

_NETWORK_BLOCKED_PATTERNS = [
    re.compile(r"(?i)connection\s+(?:refused|timed?\s*out|error)"),
    re.compile(r"(?i)unable\s+to\s+(?:connect|download|fetch|access)"),
    re.compile(r"(?i)network\s+(?:error|unreachable)"),
    re.compile(r"(?i)requests?\.exceptions"),
]

_SAFETY_REFUSAL_PATTERNS = [
    re.compile(r"(?i)(?:safety|content)\s+(?:filter|policy|restriction)"),
    re.compile(r"(?i)limited\s+access.*for\s+safety\s+reasons"),
    re.compile(r"(?i)i\s+(?:cannot|can't|won't)\s+(?:help|assist).*(?:virus|pathogen|weapon|toxin)"),
]

_RATE_LIMIT_PATTERNS = [
    re.compile(r"(?i)rate\s+limit"),
    re.compile(r"\b429\b"),
    re.compile(r"(?i)too\s+many\s+requests"),
    re.compile(r"(?i)quota\s+exceeded"),
    re.compile(r"(?i)resource\s+exhausted"),
]

_BIO_TOOL_MENTIONS = {
    "rfdiffusion": re.compile(r"(?i)\brf[-_]?diffusion\b"),
    "proteinmpnn": re.compile(r"(?i)\bprotein[-_]?mpnn\b"),
    "alphafold": re.compile(r"(?i)\balpha[-_]?fold\s*2?\b"),
    "esmfold": re.compile(r"(?i)\besm[-_]?fold\b"),
    "rosetta": re.compile(r"(?i)\b(?:rosetta|pyrosetta)\b"),
    "foldseek": re.compile(r"(?i)\bfoldseek\b"),
    "boltz": re.compile(r"(?i)\bboltz\b"),
    "openmm": re.compile(r"(?i)\bopen[-_]?mm\b"),
    "foldx": re.compile(r"(?i)\bfoldx\b"),
}

_CONSTRAINT_AWARENESS = {
    "active_site": re.compile(r"(?i)\b(?:active\s+site|catalytic\s+residue)"),
    "chromophore": re.compile(r"(?i)\b(?:chromophore|fluorophore|Ser65|Tyr66|Gly67)"),
    "cdr": re.compile(r"(?i)\b(?:CDR|complementarity[-\s]?determining)"),
    "disulfide": re.compile(r"(?i)\b(?:disulfide|cysteine\s+bridge|S-S\s+bond)"),
    "binding_site": re.compile(r"(?i)\b(?:binding\s+site|interface\s+residue|hotspot)"),
}

_METHODOLOGY_PATTERNS = {
    "de_novo": re.compile(r"(?i)\bde\s+novo\b"),
    "directed_evolution": re.compile(r"(?i)\bdirected\s+evolution\b"),
    "rational_design": re.compile(r"(?i)\brational\s+design\b"),
    "backbone_generation": re.compile(r"(?i)\bbackbone\s+generation\b"),
    "sequence_design": re.compile(r"(?i)\bsequence\s+design\b"),
    "structure_prediction": re.compile(r"(?i)\bstructure\s+prediction\b"),
    "energy_minimization": re.compile(r"(?i)\benergy\s+minimiz"),
}

_CODE_ONLY_TOOLS = {"execute_python", "python", "write_file", "read_file", "bash", "aide"}


# ---------------------------------------------------------------------------
# Detection Functions
# ---------------------------------------------------------------------------


def detect_execution_failures(
    result: EvaluationResult,
    output_dir: Path | None = None,
) -> list[tuple[str, list[str]]]:
    """Detect execution-level failures."""
    failures: list[tuple[str, list[str]]] = []
    trace = result.raw_output.get("reasoning_trace", "")
    code = result.raw_output.get("code", "")
    error = result.error_message or ""
    combined = f"{trace}\n{code}\n{error}"

    # TIMEOUT
    if "timed out" in error.lower():
        failures.append((FailureMode.TIMEOUT.value, [error[:200]]))

    # NO_OUTPUT
    num_designs = result.diversity_metrics.get("num_designs", 0)
    if num_designs == 0:
        if output_dir:
            fasta = output_dir / "designed_sequences.fasta"
            if not fasta.exists():
                failures.append((
                    FailureMode.NO_OUTPUT.value,
                    ["No designed_sequences.fasta found"],
                ))
        else:
            failures.append((
                FailureMode.NO_OUTPUT.value,
                ["num_designs=0, no output_dir"],
            ))

    # EMPTY_OUTPUT
    if output_dir:
        fasta = output_dir / "designed_sequences.fasta"
        if fasta.exists() and fasta.stat().st_size == 0:
            failures.append((
                FailureMode.EMPTY_OUTPUT.value,
                ["FASTA file exists but is 0 bytes"],
            ))

    # NETWORK_BLOCKED
    for pat in _NETWORK_BLOCKED_PATTERNS:
        m = pat.search(combined)
        if m:
            failures.append((FailureMode.NETWORK_BLOCKED.value, [m.group()[:100]]))
            break

    # SAFETY_REFUSAL
    for pat in _SAFETY_REFUSAL_PATTERNS:
        m = pat.search(combined)
        if m:
            failures.append((FailureMode.SAFETY_REFUSAL.value, [m.group()[:100]]))
            break

    # RATE_LIMITED
    for pat in _RATE_LIMIT_PATTERNS:
        m = pat.search(combined)
        if m:
            failures.append((FailureMode.RATE_LIMITED.value, [m.group()[:100]]))
            break

    # PARSE_ERROR
    if "json" in error.lower() and ("error" in error.lower() or "unterminated" in error.lower()):
        failures.append((FailureMode.PARSE_ERROR.value, [error[:200]]))

    return failures


def detect_design_failures(
    result: EvaluationResult,
    output_dir: Path | None = None,
    task_data: dict[str, Any] | None = None,
) -> list[tuple[str, list[str]]]:
    """Detect design-level failures from output artifacts and reasoning."""
    failures: list[tuple[str, list[str]]] = []
    trace = result.raw_output.get("reasoning_trace", "")
    code = result.raw_output.get("code", "")
    combined = f"{trace}\n{code}"

    # MOCK_DATA
    for pat in _MOCK_PATTERNS:
        m = pat.search(combined)
        if m:
            failures.append((
                FailureMode.MOCK_DATA.value,
                [f"Mock pattern: {m.group()[:80]}"],
            ))
            break

    # RANDOM_SEQUENCES
    for pat in _RANDOM_CODE_PATTERNS:
        m = pat.search(combined)
        if m:
            failures.append((
                FailureMode.RANDOM_SEQUENCES.value,
                [f"Random generation: {m.group()[:80]}"],
            ))
            break

    # Sequence-level checks from FASTA
    sequences = _read_raw_fasta(output_dir)
    if not sequences:
        return failures

    # ELLIPSIS_SEQUENCE
    ellipsis_seqs = [s for s in sequences if "..." in s]
    if ellipsis_seqs:
        failures.append((
            FailureMode.ELLIPSIS_SEQUENCE.value,
            [f"{len(ellipsis_seqs)} sequences contain literal ellipsis"],
        ))

    # INVALID_AA
    invalid_count = 0
    for seq in sequences:
        clean = seq.replace(".", "").replace("*", "").replace("-", "")
        if clean and set(clean.upper()) - STANDARD_AAS:
            invalid_count += 1
    if invalid_count > 0:
        failures.append((
            FailureMode.INVALID_AA.value,
            [f"{invalid_count}/{len(sequences)} sequences have invalid AAs"],
        ))

    # REPETITIVE_PATTERN
    for seq in sequences:
        if _has_repeating_motif(seq):
            failures.append((
                FailureMode.REPETITIVE_PATTERN.value,
                [f"Repeating motif in length-{len(seq)} sequence"],
            ))
            break

    # POLY_AA
    for seq in sequences:
        if len(seq) >= 20:
            counts = Counter(seq.upper())
            max_frac = max(counts.values()) / len(seq)
            if max_frac > 0.7:
                dominant = counts.most_common(1)[0]
                failures.append((
                    FailureMode.POLY_AA.value,
                    [f"Poly-{dominant[0]}: {max_frac:.0%} of length-{len(seq)}"],
                ))
                break

    # FABRICATED_METRICS
    fab_evidence = _detect_fabricated_metrics(result, output_dir)
    if fab_evidence:
        failures.append((FailureMode.FABRICATED_METRICS.value, fab_evidence))

    return failures


def detect_constraint_failures(
    result: EvaluationResult,
    task_data: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> list[tuple[str, list[str]]]:
    """Detect constraint-level failures by comparing designs against task constraints."""
    failures: list[tuple[str, list[str]]] = []
    if not task_data:
        return failures

    constraints = task_data.get("design_constraints", {})
    additional = constraints.get("additional", {})
    target = task_data.get("target", {})

    sequences = _read_raw_fasta(output_dir)
    if not sequences:
        return failures

    # LENGTH
    length_range = constraints.get("length_range")
    if length_range:
        min_len, max_len = length_range
        violations = [s for s in sequences if not (min_len <= len(s) <= max_len)]
        if violations:
            failures.append((
                FailureMode.CONSTRAINT_IGNORED_LENGTH.value,
                [f"{len(violations)}/{len(sequences)} outside [{min_len}, {max_len}]"],
            ))

    # ACTIVE SITE
    active_site = additional.get("active_site_residues")
    ref_seq = target.get("sequence")
    if active_site and ref_seq and additional.get("preserve_active_site"):
        for seq in sequences:
            if len(seq) == len(ref_seq):
                for pos in active_site:
                    idx = pos - 1
                    if 0 <= idx < len(seq) and seq[idx] != ref_seq[idx]:
                        failures.append((
                            FailureMode.CONSTRAINT_IGNORED_ACTIVE_SITE.value,
                            [f"Position {pos} mutated: {ref_seq[idx]}->{seq[idx]}"],
                        ))
                        return failures  # one evidence is enough
                break

    # CHROMOPHORE
    chromophore = additional.get("chromophore_residues")
    if chromophore and ref_seq and additional.get("preserve_chromophore"):
        for seq in sequences:
            if len(seq) == len(ref_seq):
                for pos in chromophore:
                    idx = pos - 1
                    if 0 <= idx < len(seq) and seq[idx] != ref_seq[idx]:
                        failures.append((
                            FailureMode.CONSTRAINT_IGNORED_CHROMOPHORE.value,
                            [f"Chromophore residue {pos} mutated"],
                        ))
                        return failures
                break

    return failures


def detect_tool_failures(
    result: EvaluationResult,
    task_data: dict[str, Any] | None = None,
) -> list[tuple[str, list[str]]]:
    """Detect tool usage failures."""
    failures: list[tuple[str, list[str]]] = []
    tools_used = result.tools_used or []
    tools_expected = (task_data or {}).get("metadata", {}).get("tools_expected", [])

    from biodesignbench.eval.metrics.approach import TOOL_CATEGORIES, normalize_tool_name

    bio_tools_used = [
        t for t in tools_used
        if normalize_tool_name(t) in {normalize_tool_name(k) for k in TOOL_CATEGORIES}
    ]

    # NO_BIO_TOOLS
    if not bio_tools_used and tools_expected:
        failures.append((
            FailureMode.NO_BIO_TOOLS.value,
            [f"Expected: {tools_expected}, Used: {tools_used}"],
        ))

    # TOOL_AVAILABLE_NOT_USED
    if tools_expected and bio_tools_used:
        used_norm = {normalize_tool_name(t) for t in tools_used}
        for expected in tools_expected:
            if normalize_tool_name(expected) not in used_norm:
                failures.append((
                    FailureMode.TOOL_AVAILABLE_NOT_USED.value,
                    [f"Expected tool '{expected}' not used"],
                ))

    # CODE_ONLY_APPROACH
    if tools_used and all(t.lower() in _CODE_ONLY_TOOLS for t in tools_used):
        failures.append((
            FailureMode.CODE_ONLY_APPROACH.value,
            [f"Only code tools: {tools_used}"],
        ))

    return failures


def analyze_reasoning_trace(
    result: EvaluationResult,
    task_data: dict[str, Any] | None = None,
) -> list[tuple[str, list[str]]]:
    """Extract reasoning quality signals (positive, not failures)."""
    signals: list[tuple[str, list[str]]] = []
    trace = result.raw_output.get("reasoning_trace", "")
    if not trace:
        return signals

    tools_expected = (task_data or {}).get("metadata", {}).get("tools_expected", [])

    # CORRECT_TOOL_SELECTION
    mentioned_tools = []
    for tool_name, pattern in _BIO_TOOL_MENTIONS.items():
        if pattern.search(trace):
            mentioned_tools.append(tool_name)

    if tools_expected and mentioned_tools:
        from biodesignbench.eval.metrics.approach import normalize_tool_name

        expected_norm = {normalize_tool_name(t) for t in tools_expected}
        mentioned_norm = {normalize_tool_name(t) for t in mentioned_tools}
        overlap = expected_norm & mentioned_norm
        if overlap:
            signals.append((
                FailureMode.CORRECT_TOOL_SELECTION.value,
                [f"Correctly mentioned: {sorted(overlap)}"],
            ))

    # GOOD_PLAN_NO_EXECUTION
    if mentioned_tools and result.partial_score == 0:
        signals.append((
            FailureMode.GOOD_PLAN_NO_EXECUTION.value,
            [f"Described {mentioned_tools} but produced no valid output"],
        ))

    # BIOLOGICAL_AWARENESS
    awareness_hits = []
    for concept, pattern in _CONSTRAINT_AWARENESS.items():
        if pattern.search(trace):
            awareness_hits.append(concept)
    if awareness_hits:
        signals.append((
            FailureMode.BIOLOGICAL_AWARENESS.value,
            [f"Bio awareness: {', '.join(awareness_hits)}"],
        ))

    # STRUCTURAL_AWARENESS
    methodology_hits = []
    for method, pattern in _METHODOLOGY_PATTERNS.items():
        if pattern.search(trace):
            methodology_hits.append(method)
    if methodology_hits:
        signals.append((
            FailureMode.STRUCTURAL_AWARENESS.value,
            [f"Methodology: {', '.join(methodology_hits)}"],
        ))

    return signals


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _read_raw_fasta(output_dir: Path | None) -> list[str]:
    """Read raw sequences from FASTA without filtering."""
    if not output_dir:
        return []
    fasta_path = Path(output_dir) / "designed_sequences.fasta"
    if not fasta_path.exists():
        return []

    sequences: list[str] = []
    current_seq = ""
    for line in fasta_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
            current_seq = ""
        else:
            current_seq += line
    if current_seq:
        sequences.append(current_seq)
    return sequences


def _has_repeating_motif(
    seq: str, min_motif: int = 3, min_repeats: int = 5
) -> bool:
    """Check if sequence contains a repeating motif of length >= min_motif."""
    if len(seq) < min_motif * min_repeats:
        return False
    upper = seq.upper()
    for motif_len in range(min_motif, min(10, len(upper) // min_repeats + 1)):
        for start in range(len(upper) - motif_len * min_repeats + 1):
            motif = upper[start : start + motif_len]
            consecutive = 1
            pos = start + motif_len
            while pos + motif_len <= len(upper):
                if upper[pos : pos + motif_len] == motif:
                    consecutive += 1
                    pos += motif_len
                else:
                    break
            if consecutive >= min_repeats:
                return True
    return False


def _detect_fabricated_metrics(
    result: EvaluationResult,
    output_dir: Path | None,
) -> list[str]:
    """Detect fabricated/invented metric values."""
    evidence: list[str] = []
    if not output_dir:
        return evidence

    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return evidence

    try:
        data = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return evidence

    if not isinstance(data, dict):
        return evidence

    # Check dict of design_id -> {metric: value}
    if all(isinstance(v, dict) for v in data.values()) and len(data) > 1:
        metric_sets = [json.dumps(v, sort_keys=True) for v in data.values()]
        if len(set(metric_sets)) == 1:
            evidence.append(
                f"All {len(metric_sets)} designs have identical metrics: "
                f"{list(data.values())[0]}"
            )

    return evidence


def _determine_severity(
    failure_modes: list[str],
    reasoning_signals: list[str],
    score: float,
) -> str:
    """Determine overall severity of failures."""
    if not failure_modes:
        return "none"

    critical = {
        FailureMode.NO_OUTPUT.value,
        FailureMode.TIMEOUT.value,
        FailureMode.SAFETY_REFUSAL.value,
    }
    major = {
        FailureMode.MOCK_DATA.value,
        FailureMode.FABRICATED_METRICS.value,
        FailureMode.RANDOM_SEQUENCES.value,
        FailureMode.NO_BIO_TOOLS.value,
    }

    mode_set = set(failure_modes)
    if mode_set & critical:
        return "critical"
    if mode_set & major:
        return "major"
    if len(failure_modes) >= 3:
        return "major"
    return "minor"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def classify_failure(
    result: EvaluationResult,
    output_dir: Path | None = None,
    task_data: dict[str, Any] | None = None,
) -> list[str]:
    """Classify failure modes for a single evaluation result.

    Convenience wrapper that runs all four detection functions and returns
    the combined list of failure mode strings (no reasoning signals).

    Args:
        result: The evaluation result to analyze.
        output_dir: Optional path to the output directory.
        task_data: Optional task JSON data.

    Returns:
        List of FailureMode value strings (e.g., ["NO_OUTPUT", "NO_BIO_TOOLS"]).
    """
    detections: list[tuple[str, list[str]]] = []
    detections.extend(detect_execution_failures(result, output_dir))
    detections.extend(detect_design_failures(result, output_dir, task_data))
    detections.extend(detect_constraint_failures(result, task_data, output_dir))
    detections.extend(detect_tool_failures(result, task_data))
    return [mode for mode, _ in detections]


def generate_failure_report(
    result: EvaluationResult,
    output_dir: Path | None = None,
) -> FailureModeReport:
    """Generate a failure mode report for a single evaluation result.

    Convenience wrapper that creates a FailureModeAnalyzer and calls
    analyze_single. Uses default task/ground-truth directories.

    Args:
        result: The evaluation result to analyze.
        output_dir: Optional path to the output directory.

    Returns:
        FailureModeReport with detected failures and reasoning signals.
    """
    analyzer = FailureModeAnalyzer()
    return analyzer.analyze_single(result, output_dir)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class FailureModeAnalyzer:
    """Analyze failure modes across Tier 2 design evaluations.

    Usage::

        analyzer = FailureModeAnalyzer()
        analysis = analyzer.analyze_run("results/runs/run_20260211_220512_578312")
        print(analyzer.to_markdown_tables(analysis))
    """

    def __init__(
        self,
        tasks_dir: Path | None = None,
        ground_truth_dir: Path | None = None,
    ):
        _project_root = Path(__file__).resolve().parents[2]
        self.tasks_dir = tasks_dir or _project_root / "tasks" / "tier2"
        self.ground_truth_dir = (
            ground_truth_dir or _project_root / "data" / "tier2" / "ground_truth"
        )
        self._task_cache: dict[str, dict[str, Any]] = {}

    def _load_task_data(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self._task_cache:
            return self._task_cache[task_id]
        path = self.tasks_dir / f"{task_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        self._task_cache[task_id] = data
        return data

    @staticmethod
    def _get_task_category(task_id: str) -> str:
        return task_id.rsplit("_", 1)[0] if "_" in task_id else task_id

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def analyze_single(
        self,
        result: EvaluationResult,
        output_dir: Path | None = None,
    ) -> FailureModeReport:
        """Analyze a single evaluation result."""
        task_data = self._load_task_data(result.task_id)

        detections: list[tuple[str, list[str]]] = []
        detections.extend(detect_execution_failures(result, output_dir))
        detections.extend(detect_design_failures(result, output_dir, task_data))
        detections.extend(detect_constraint_failures(result, task_data, output_dir))
        detections.extend(detect_tool_failures(result, task_data))

        reasoning = analyze_reasoning_trace(result, task_data)

        failure_modes = [mode for mode, _ in detections]
        reasoning_signals = [sig for sig, _ in reasoning]

        evidence: dict[str, list[str]] = {}
        for mode, ev_list in detections + reasoning:
            evidence.setdefault(mode, []).extend(ev_list)

        severity = _determine_severity(
            failure_modes, reasoning_signals, result.partial_score
        )

        parts = [f"Score {result.partial_score:.0f}/100"]
        if failure_modes:
            parts.append(f"failures: {', '.join(failure_modes[:3])}")
        if reasoning_signals:
            parts.append(f"reasoning: {', '.join(reasoning_signals[:2])}")

        return FailureModeReport(
            task_id=result.task_id,
            agent_id=result.agent_id,
            failure_modes=failure_modes,
            reasoning_signals=reasoning_signals,
            evidence=evidence,
            severity=severity,
            score=result.partial_score,
            summary="; ".join(parts),
        )

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_results(
        self,
        results: list[EvaluationResult],
        run_dir: Path | None = None,
    ) -> FailureAnalysisResult:
        """Analyze all results from a benchmark run."""
        run_id = run_dir.name if run_dir else "unknown"
        analysis = FailureAnalysisResult(run_id=run_id)

        for result in results:
            if not self._is_tier2(result):
                continue

            output_dir = None
            if run_dir:
                candidate = (
                    run_dir / "agents" / result.agent_id / result.task_id / "output"
                )
                if candidate.exists():
                    output_dir = candidate

            report = self.analyze_single(result, output_dir)
            analysis.evaluations.append(report)

        analysis.agent_profiles = self._compute_agent_profiles(analysis.evaluations)
        analysis.heatmap_data = self._compute_heatmap(analysis.evaluations)
        analysis.category_heatmap = self._compute_category_heatmap(analysis.evaluations)
        analysis.insights = self._generate_insights(analysis)
        return analysis

    def analyze_run(self, run_dir: str | Path) -> FailureAnalysisResult:
        """Load and analyze a complete run directory."""
        run_dir = Path(run_dir)
        results = self._load_run_results(run_dir)
        return self.analyze_results(results, run_dir)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load_run_results(self, run_dir: Path) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        agents_dir = run_dir / "agents"
        if not agents_dir.exists():
            return results
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            for task_dir in sorted(agent_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                result_path = task_dir / "result.json"
                if result_path.exists():
                    with open(result_path) as f:
                        data = json.load(f)
                    results.append(EvaluationResult.model_validate(data))
        return results

    @staticmethod
    def _is_tier2(result: EvaluationResult) -> bool:
        return bool(
            result.quality_metrics
            or result.novelty_metrics
            or result.diversity_metrics
            or result.approach_metrics
            or result.feasibility_metrics
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _compute_agent_profiles(
        self, evaluations: list[FailureModeReport]
    ) -> dict[str, AgentFailureProfile]:
        by_agent: dict[str, list[FailureModeReport]] = defaultdict(list)
        for report in evaluations:
            by_agent[report.agent_id].append(report)

        profiles: dict[str, AgentFailureProfile] = {}
        for agent_id, reports in by_agent.items():
            p = AgentFailureProfile(agent_id=agent_id)
            p.total_evaluations = len(reports)
            p.mean_score = sum(r.score for r in reports) / max(len(reports), 1)

            for report in reports:
                for mode in report.failure_modes:
                    p.failure_mode_counts[mode] = p.failure_mode_counts.get(mode, 0) + 1
                    cat = FAILURE_CATEGORY.get(
                        FailureMode(mode), FailureCategory.EXECUTION
                    ).value
                    p.failure_category_counts[cat] = (
                        p.failure_category_counts.get(cat, 0) + 1
                    )
                for sig in report.reasoning_signals:
                    p.reasoning_signal_counts[sig] = (
                        p.reasoning_signal_counts.get(sig, 0) + 1
                    )
                p.severity_distribution[report.severity] = (
                    p.severity_distribution.get(report.severity, 0) + 1
                )

            cat_scores: dict[str, list[float]] = defaultdict(list)
            for report in reports:
                cat_scores[self._get_task_category(report.task_id)].append(report.score)
            p.per_category_scores = {
                c: sum(s) / len(s) for c, s in cat_scores.items()
            }

            profiles[agent_id] = p
        return profiles

    def _compute_heatmap(
        self, evaluations: list[FailureModeReport]
    ) -> dict[str, dict[str, int]]:
        hm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for report in evaluations:
            for mode in report.failure_modes:
                hm[report.agent_id][mode] += 1
        return {k: dict(v) for k, v in hm.items()}

    def _compute_category_heatmap(
        self, evaluations: list[FailureModeReport]
    ) -> dict[str, dict[str, float]]:
        data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for report in evaluations:
            cat = self._get_task_category(report.task_id)
            data[cat][report.agent_id].append(report.score)
        return {
            cat: {a: sum(s) / len(s) for a, s in agents.items()}
            for cat, agents in data.items()
        }

    # ------------------------------------------------------------------
    # Insight generation
    # ------------------------------------------------------------------

    def _generate_insights(self, analysis: FailureAnalysisResult) -> list[str]:
        insights: list[str] = []
        total = max(len(analysis.evaluations), 1)

        # Universal failures
        all_modes: Counter[str] = Counter()
        for report in analysis.evaluations:
            for mode in report.failure_modes:
                all_modes[mode] += 1

        for mode, count in all_modes.most_common(5):
            pct = count / total * 100
            if pct >= 50:
                insights.append(
                    f"Universal failure: {mode} affects {pct:.0f}% "
                    f"of evaluations ({count}/{total})"
                )

        # Agent-specific
        for agent_id, profile in analysis.agent_profiles.items():
            if profile.per_category_scores:
                best_cat = max(
                    profile.per_category_scores,
                    key=profile.per_category_scores.get,  # type: ignore[arg-type]
                )
                best_score = profile.per_category_scores[best_cat]
                if best_score > 0:
                    insights.append(
                        f"{agent_id} performs best on {best_cat} "
                        f"tasks (avg {best_score:.1f}/100)"
                    )

            gap = profile.reasoning_signal_counts.get(
                FailureMode.GOOD_PLAN_NO_EXECUTION.value, 0
            )
            if gap > 0:
                insights.append(
                    f"{agent_id}: knowledge-execution gap in "
                    f"{gap}/{profile.total_evaluations} tasks"
                )

        return insights

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def to_dict(self, analysis: FailureAnalysisResult) -> dict[str, Any]:
        """Serialize analysis to JSON-compatible dict."""
        return {
            "run_id": analysis.run_id,
            "summary": {
                "total_evaluations": len(analysis.evaluations),
                "agents": list(analysis.agent_profiles.keys()),
                "insights": analysis.insights,
            },
            "evaluations": [
                {
                    "task_id": r.task_id,
                    "agent_id": r.agent_id,
                    "score": r.score,
                    "severity": r.severity,
                    "failure_modes": r.failure_modes,
                    "reasoning_signals": r.reasoning_signals,
                    "evidence": r.evidence,
                    "summary": r.summary,
                }
                for r in analysis.evaluations
            ],
            "agent_profiles": {
                aid: {
                    "total_evaluations": p.total_evaluations,
                    "mean_score": round(p.mean_score, 1),
                    "failure_mode_counts": p.failure_mode_counts,
                    "failure_category_counts": p.failure_category_counts,
                    "reasoning_signal_counts": p.reasoning_signal_counts,
                    "severity_distribution": p.severity_distribution,
                    "per_category_scores": {
                        k: round(v, 1) for k, v in p.per_category_scores.items()
                    },
                }
                for aid, p in analysis.agent_profiles.items()
            },
            "heatmap": analysis.heatmap_data,
            "category_heatmap": {
                cat: {a: round(s, 1) for a, s in agents.items()}
                for cat, agents in analysis.category_heatmap.items()
            },
        }

    def to_markdown_tables(self, analysis: FailureAnalysisResult) -> str:
        """Generate markdown tables for a paper appendix."""
        lines: list[str] = []
        agents = sorted(analysis.agent_profiles.keys())

        # Table 1: Agent Summary
        lines.append("## Table 1: Agent Failure Summary\n")
        lines.append(
            "| Agent | Score | Critical | Major | Minor | Top Failure |"
        )
        lines.append("|-------|------:|:--------:|:-----:|:-----:|-------------|")
        for aid in agents:
            p = analysis.agent_profiles[aid]
            top = (
                max(p.failure_mode_counts, key=p.failure_mode_counts.get)  # type: ignore[arg-type]
                if p.failure_mode_counts
                else "none"
            )
            lines.append(
                f"| {aid} | {p.mean_score:.1f} "
                f"| {p.severity_distribution.get('critical', 0)} "
                f"| {p.severity_distribution.get('major', 0)} "
                f"| {p.severity_distribution.get('minor', 0)} "
                f"| {top} |"
            )

        # Table 2: Heatmap
        all_modes = sorted(
            {m for counts in analysis.heatmap_data.values() for m in counts}
        )
        if all_modes:
            lines.append("\n## Table 2: Failure Mode Heatmap\n")
            lines.append("| Agent | " + " | ".join(all_modes) + " |")
            lines.append(
                "|-------| " + " | ".join("---:" for _ in all_modes) + " |"
            )
            for aid in agents:
                counts = analysis.heatmap_data.get(aid, {})
                row = " | ".join(str(counts.get(m, 0)) for m in all_modes)
                lines.append(f"| {aid} | {row} |")

        # Table 3: Category x Agent scores
        categories = sorted(analysis.category_heatmap.keys())
        if categories:
            lines.append("\n## Table 3: Category x Agent Average Score\n")
            lines.append("| Category | " + " | ".join(agents) + " |")
            lines.append(
                "|----------| " + " | ".join("---:" for _ in agents) + " |"
            )
            for cat in categories:
                scores = analysis.category_heatmap.get(cat, {})
                row = " | ".join(f"{scores.get(a, 0):.1f}" for a in agents)
                lines.append(f"| {cat} | {row} |")

        # Insights
        if analysis.insights:
            lines.append("\n## Key Insights\n")
            for insight in analysis.insights:
                lines.append(f"- {insight}")

        return "\n".join(lines)
