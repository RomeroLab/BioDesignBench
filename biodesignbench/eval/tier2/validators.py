"""Design-specific output validators for Tier 2 tasks.

Validates that agent-produced design outputs (FASTA sequences, metrics JSON)
conform to expected formats and constraints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from biodesignbench.eval.metrics.sequence import STANDARD_AAS
from biodesignbench.eval.tier1.validators import validate_fasta, validate_json

_PLACEHOLDER_PATTERNS = re.compile(
    r"(?i)(mock|placeholder|todo|sample|test_seq|dummy|filler)",
)


def _is_garbage_sequence(seq: str) -> bool:
    """Detect garbage/placeholder sequences that should be excluded.

    Returns True if the sequence is likely garbage:
    - Empty or too short (< 5 residues)
    - Contains non-standard amino acid characters
    - All-identical characters
    - Fewer than 3 unique AAs in sequences > 10 residues
    - Contains placeholder text patterns
    - Contains '...' ellipsis pattern

    Args:
        seq: Amino acid sequence string.

    Returns:
        True if the sequence is garbage and should be excluded.
    """
    if not seq or len(seq) < 5:
        return True

    upper = seq.upper()

    # Check for non-standard characters (digits, dots, B, Z, etc.)
    # Allow X (unknown AA, valid IUPAC) and / (chain separator in antibodies)
    invalid = set(upper) - STANDARD_AAS - {"X", "/"}
    if invalid:
        return True

    # All-identical (homopolymeric)
    if len(set(upper)) == 1:
        return True

    # Too few unique AAs for longer sequences
    if len(upper) > 10 and len(set(upper)) < 3:
        return True

    # Placeholder text patterns (case-insensitive)
    if _PLACEHOLDER_PATTERNS.search(seq):
        return True

    # Ellipsis pattern
    if "..." in seq:
        return True

    return False


def extract_designs_from_fasta(path: Path) -> list[dict]:
    """Parse a FASTA file into a list of design dicts.

    Returns:
        List of {id: str, sequence: str} dicts. Empty list if file missing.
    """
    if not path.exists():
        return []

    designs = []
    current_id: str | None = None
    current_seq = ""

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    designs.append({"id": current_id, "sequence": current_seq})
                current_id = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line

    if current_id is not None:
        designs.append({"id": current_id, "sequence": current_seq})

    # Strip internal whitespace from sequences (common agent formatting error)
    for d in designs:
        d["sequence"] = "".join(d["sequence"].split())

    # Filter out garbage sequences before returning
    designs = [d for d in designs if not _is_garbage_sequence(d["sequence"])]

    return designs


_METRIC_KEY_MAP: dict[str, str] = {
    "plddt": "pLDDT",
    "pldt": "pLDDT",
    "iptm": "ipTM",
    "ptm": "pTM",
    "i_pae": "i_pAE",
    "ipae": "i_pAE",
    "predicted_kd": "predicted_kd",
    "kd": "predicted_kd",
    "predicted_ddg": "predicted_ddG",
    "ddg": "predicted_ddG",
    "active_site_rmsd": "active_site_rmsd",
    "tm_score": "TM_score",
    "tmscore": "TM_score",
}


def _normalize_metric_key(key: str) -> str:
    """Map common agent metric key variants to standard names."""
    return _METRIC_KEY_MAP.get(key.lower(), key)


def _average_design_metrics(designs: list[dict]) -> dict[str, float]:
    """Average numeric metrics across a list of per-design dicts."""
    aggregated: dict[str, list[float]] = {}
    for entry in designs:
        if isinstance(entry, dict):
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    aggregated.setdefault(k, []).append(float(v))
    return {k: sum(vs) / len(vs) for k, vs in aggregated.items() if vs}


def extract_metrics_from_json(path: Path) -> dict[str, float]:
    """Extract and average metrics from a metrics JSON file.

    Supports three formats:
    1. Dict with metric keys mapping to values: {"pLDDT": 85.0, ...}
    2. List of per-design dicts: [{"pLDDT": 85.0}, {"pLDDT": 90.0}]
       In this case, values are averaged across designs.
    3. Dict with nested "designs" list: {"designs": [{"plddt": 96}, ...]}
       Per-design metrics are averaged; top-level numerics are included.

    Metric keys are normalized to standard names (e.g., plddt -> pLDDT).

    Returns:
        Dict of metric name -> float value. Empty dict if file missing/invalid.
    """
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    raw: dict[str, float] = {}

    if isinstance(data, dict):
        # Check for nested designs list
        designs_list = data.get("designs")
        if isinstance(designs_list, list) and designs_list:
            raw = _average_design_metrics(designs_list)
        else:
            # Flat dict of metrics
            raw = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    elif isinstance(data, list) and data:
        raw = _average_design_metrics(data)

    # Normalize keys to standard names.
    # More specific keys (iptm) should override less specific alternatives.
    # Process in two passes: first generic, then specific overrides.
    # NOTE: "ptm" maps to "pTM" (monomer fold confidence), NOT "ipTM" (interface).
    normalized: dict[str, float] = {}
    specific_keys = {"iptm", "i_pae", "ipae", "predicted_kd", "predicted_ddg", "active_site_rmsd"}
    for k, v in raw.items():
        if k.lower() not in specific_keys:
            std_key = _normalize_metric_key(k)
            normalized[std_key] = v
    for k, v in raw.items():
        if k.lower() in specific_keys:
            std_key = _normalize_metric_key(k)
            normalized[std_key] = v

    return normalized


def validate_design_fasta(
    path: Path,
    max_designs: int = 10,
    length_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Validate a design FASTA file with design-specific checks.

    Extends the base FASTA validator with:
    - Max design count enforcement
    - Length range checks per sequence
    - Duplicate sequence detection

    Returns:
        Dict with all base FASTA fields plus: within_length_range (bool),
        length_violations (list), duplicate_sequences (list),
        num_unique_sequences (int).
    """
    result = validate_fasta(
        path,
        alphabet="protein",
        min_sequences=1,
        max_sequences=max_designs,
    )

    # Design-specific additions
    result["within_length_range"] = True
    result["length_violations"] = []
    result["duplicate_sequences"] = []
    result["num_unique_sequences"] = 0

    if not result["exists"]:
        return result

    designs = extract_designs_from_fasta(path)
    sequences = [d["sequence"] for d in designs]
    result["num_unique_sequences"] = len(set(sequences))

    # Check for duplicate sequences
    seen: dict[str, str] = {}
    for d in designs:
        seq = d["sequence"].upper()
        if seq in seen:
            result["duplicate_sequences"].append(
                f"{d['id']} duplicates {seen[seq]}"
            )
        else:
            seen[seq] = d["id"]

    # Length range checks
    if length_range is not None:
        min_len, max_len = length_range
        for d in designs:
            seq_len = len(d["sequence"])
            if not (min_len <= seq_len <= max_len):
                result["within_length_range"] = False
                result["length_violations"].append(
                    f"{d['id']}: length {seq_len} outside [{min_len}, {max_len}]"
                )

    return result


def validate_metrics_json(
    path: Path,
    expected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a metrics JSON file with design-specific checks.

    Args:
        path: Path to metrics.json.
        expected_metrics: List of expected metric names.

    Returns:
        Dict with base JSON fields plus: has_expected_metrics (bool),
        missing_metrics (list), metrics (dict).
    """
    result = validate_json(path, expected_type="any")

    result["has_expected_metrics"] = True
    result["missing_metrics"] = []
    result["metrics"] = {}

    if not result["valid_json"]:
        return result

    metrics = extract_metrics_from_json(path)
    result["metrics"] = metrics

    if expected_metrics:
        for metric in expected_metrics:
            if metric not in metrics:
                result["has_expected_metrics"] = False
                result["missing_metrics"].append(metric)

    return result


def validate_design_output(
    output_dir: Path,
    required_files: list[str] | None = None,
    max_designs: int = 10,
    length_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Complete validation of a design task output directory.

    Args:
        output_dir: Directory containing agent outputs.
        required_files: List of required filenames.
        max_designs: Maximum number of designs allowed.
        length_range: (min, max) sequence length constraint.

    Returns:
        Dict with: dir_exists (bool), files_found (list), files_missing (list),
        fasta_validation (dict|None), metrics_validation (dict|None), errors (list).
    """
    if required_files is None:
        required_files = ["designed_sequences.fasta", "metrics.json"]

    result: dict[str, Any] = {
        "dir_exists": False,
        "files_found": [],
        "files_missing": [],
        "fasta_validation": None,
        "metrics_validation": None,
        "errors": [],
    }

    if not output_dir.exists():
        result["errors"].append(f"Output directory not found: {output_dir}")
        return result

    result["dir_exists"] = True

    for fname in required_files:
        fpath = output_dir / fname
        if fpath.exists():
            result["files_found"].append(fname)
        else:
            result["files_missing"].append(fname)

    # Validate FASTA if present
    fasta_path = output_dir / "designed_sequences.fasta"
    if fasta_path.exists():
        result["fasta_validation"] = validate_design_fasta(
            fasta_path, max_designs=max_designs, length_range=length_range,
        )

    # Validate metrics JSON if present
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        result["metrics_validation"] = validate_metrics_json(metrics_path)

    return result
