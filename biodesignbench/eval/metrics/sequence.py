"""Pure Python sequence-level metrics for protein design evaluation.

No external dependencies required. All functions work on plain strings.
"""

from __future__ import annotations

import math
from itertools import combinations


# Kyte-Doolittle hydrophobicity scale
_KD_SCALE: dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute fractional sequence identity between two sequences.

    For same-length sequences, computes positional match fraction.
    For different-length sequences, uses sliding window of the shorter
    over the longer and returns the best match.

    Returns:
        Float in [0, 1]. Returns 0.0 if either sequence is empty.
    """
    if not seq1 or not seq2:
        return 0.0

    s1 = seq1.upper()
    s2 = seq2.upper()

    if len(s1) == len(s2):
        matches = sum(a == b for a, b in zip(s1, s2))
        return matches / len(s1)

    # Sliding window: slide shorter over longer
    short, long = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    best = 0.0
    for offset in range(len(long) - len(short) + 1):
        matches = sum(
            a == b for a, b in zip(short, long[offset:offset + len(short)])
        )
        identity = matches / len(short)
        if identity > best:
            best = identity
    return best


def max_identity_to_reference(designs: list[str], reference: str) -> float:
    """Return the maximum sequence identity of any design to the reference.

    Args:
        designs: List of designed sequences.
        reference: Reference/wild-type sequence.

    Returns:
        Float in [0, 1]. Returns 0.0 if designs is empty.
    """
    if not designs or not reference:
        return 0.0
    return max(sequence_identity(d, reference) for d in designs)


def mean_pairwise_diversity(sequences: list[str]) -> float:
    """Compute mean pairwise diversity across all sequence pairs.

    Diversity = 1 - identity for each pair.

    Returns:
        Float in [0, 1]. Returns 0.0 for 0 or 1 sequences.
    """
    if len(sequences) < 2:
        return 0.0

    total = 0.0
    count = 0
    for s1, s2 in combinations(sequences, 2):
        total += 1.0 - sequence_identity(s1, s2)
        count += 1

    return total / count if count > 0 else 0.0


def sequence_entropy(sequences: list[str], truncate: bool = False) -> float:
    """Compute mean per-position Shannon entropy across aligned sequences.

    All sequences must be the same length (unless *truncate* is True).
    Uses log base 20 (max entropy = 1.0 when all 20 amino acids are
    equally likely).

    Args:
        sequences: List of amino acid sequences.
        truncate: If True, truncate all sequences to the shortest length
            instead of returning 0.0 for unequal-length inputs.

    Returns:
        Float in [0, 1]. Returns 0.0 if sequences have different lengths
        (when truncate=False) or fewer than 2 sequences.
    """
    if len(sequences) < 2:
        return 0.0

    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        if not truncate:
            return 0.0
        # Truncate to shortest length
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

        # Normalize by log(20) so max = 1.0
        total_entropy += pos_entropy / math.log(20)

    return total_entropy / seq_len


def validate_amino_acids(sequence: str) -> dict:
    """Check that a sequence contains only standard amino acids.

    Returns:
        Dict with: valid (bool), invalid_chars (set), fraction_valid (float).
    """
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
    """Check if a sequence length falls within the given range.

    Args:
        sequence: Amino acid sequence.
        length_range: (min_length, max_length) or None for no constraint.

    Returns:
        Dict with: length (int), within_range (bool), range (tuple|None).
    """
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
    """Compute Kyte-Doolittle hydrophobicity statistics.

    Returns:
        Dict with: mean, std, fraction_hydrophobic (KD > 0), min, max.
        Returns all zeros for empty sequence.
    """
    if not sequence:
        return {
            "mean": 0.0, "std": 0.0, "fraction_hydrophobic": 0.0,
            "min": 0.0, "max": 0.0,
        }

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
    """Count the number of differing positions between two same-length sequences.

    Returns:
        Number of mutations. Returns -1 if sequences differ in length.
    """
    if len(wt) != len(designed):
        return -1
    return sum(a != b for a, b in zip(wt.upper(), designed.upper()))
