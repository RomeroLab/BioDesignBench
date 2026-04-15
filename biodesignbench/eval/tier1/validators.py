"""Reusable output format validators for Tier 1 tasks.

Each validator returns a standardized dict with validation results
that can be fed directly into the scoring framework.
"""

import csv
import io
import json
import re
import struct
from pathlib import Path
from typing import Any


# Standard amino acid alphabets
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_ALPHABET_EXTENDED = PROTEIN_ALPHABET | set("BJOUXZ*-")
DNA_ALPHABET = set("ACGT")
DNA_ALPHABET_EXTENDED = DNA_ALPHABET | set("NRYSWKMBDHV-")
RNA_ALPHABET = set("ACGU")
RNA_ALPHABET_EXTENDED = RNA_ALPHABET | set("NRYSWKMBDHV-")

# 3-letter to 1-letter amino acid mapping
AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def validate_fasta(
    path: Path,
    *,
    alphabet: str = "protein",
    min_sequences: int = 0,
    max_sequences: int | None = None,
    header_pattern: str | None = None,
    expected_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a FASTA file.

    Args:
        path: Path to FASTA file.
        alphabet: One of 'protein', 'protein_ext', 'dna', 'dna_ext', 'rna', 'rna_ext'.
        min_sequences: Minimum expected sequence count.
        max_sequences: Maximum expected sequence count (None = no limit).
        header_pattern: Optional regex the header line must match.
        expected_ids: Optional list of expected sequence IDs.

    Returns:
        Dict with: exists, num_sequences, all_valid_headers, all_valid_sequences,
        unique_ids, sequence_lengths, errors.
    """
    alpha_map = {
        "protein": PROTEIN_ALPHABET,
        "protein_ext": PROTEIN_ALPHABET_EXTENDED,
        "dna": DNA_ALPHABET,
        "dna_ext": DNA_ALPHABET_EXTENDED,
        "rna": RNA_ALPHABET,
        "rna_ext": RNA_ALPHABET_EXTENDED,
    }
    valid_chars = alpha_map.get(alphabet, PROTEIN_ALPHABET)

    result: dict[str, Any] = {
        "exists": False,
        "num_sequences": 0,
        "all_valid_headers": True,
        "all_valid_sequences": True,
        "unique_ids": set(),
        "duplicate_ids": [],
        "sequence_lengths": {},
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True
    current_id: str | None = None
    current_seq = ""

    def _finish_sequence() -> None:
        nonlocal current_seq
        if current_id is None:
            return
        seq_upper = current_seq.upper()
        result["sequence_lengths"][current_id] = len(seq_upper)
        if seq_upper:
            invalid = set(seq_upper) - valid_chars
            if invalid:
                result["all_valid_sequences"] = False
                result["errors"].append(
                    f"Invalid characters in {current_id}: {invalid}"
                )
        current_seq = ""

    try:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    _finish_sequence()
                    result["num_sequences"] += 1

                    # Parse header
                    header_match = re.match(r">(\S+)", line)
                    if header_match:
                        current_id = header_match.group(1)
                        if current_id in result["unique_ids"]:
                            result["duplicate_ids"].append(current_id)
                        result["unique_ids"].add(current_id)
                    else:
                        result["all_valid_headers"] = False
                        result["errors"].append(f"Invalid header at line {line_num}")
                        current_id = f"unknown_{line_num}"
                        result["unique_ids"].add(current_id)

                    # Check header pattern
                    if header_pattern and not re.match(header_pattern, line):
                        result["all_valid_headers"] = False
                        result["errors"].append(
                            f"Header at line {line_num} doesn't match pattern"
                        )
                else:
                    current_seq += line

            _finish_sequence()
    except Exception as e:
        result["errors"].append(f"Read error: {e}")

    # Count checks
    if min_sequences and result["num_sequences"] < min_sequences:
        result["errors"].append(
            f"Too few sequences: {result['num_sequences']} < {min_sequences}"
        )
    if max_sequences is not None and result["num_sequences"] > max_sequences:
        result["errors"].append(
            f"Too many sequences: {result['num_sequences']} > {max_sequences}"
        )

    # Expected IDs check
    if expected_ids is not None:
        missing = set(expected_ids) - result["unique_ids"]
        if missing:
            result["errors"].append(f"Missing expected IDs: {missing}")

    return result


def validate_pdb(
    path: Path,
    *,
    expected_chains: list[str] | None = None,
    min_atoms: int = 0,
    check_bfactors: bool = False,
) -> dict[str, Any]:
    """Validate a PDB file.

    Returns:
        Dict with: exists, num_atoms, num_residues, chain_ids, residues_per_chain,
        has_bfactors, has_ter_records, has_end_record, coordinate_ranges, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "num_atoms": 0,
        "num_hetatm": 0,
        "num_residues": 0,
        "chain_ids": set(),
        "residues_per_chain": {},
        "has_bfactors": False,
        "has_ter_records": False,
        "has_end_record": False,
        "coordinate_ranges": {"x": [float("inf"), float("-inf")],
                              "y": [float("inf"), float("-inf")],
                              "z": [float("inf"), float("-inf")]},
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True
    seen_residues: set[tuple[str, int]] = set()

    try:
        with open(path) as f:
            for line in f:
                record = line[:6].strip()

                if record == "ATOM" or record == "HETATM":
                    if record == "ATOM":
                        result["num_atoms"] += 1
                    else:
                        result["num_hetatm"] += 1

                    try:
                        chain = line[21].strip()
                        resnum = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        bfactor = float(line[60:66].strip()) if len(line) >= 66 else 0.0

                        if chain:
                            result["chain_ids"].add(chain)
                        res_key = (chain, resnum)
                        if res_key not in seen_residues:
                            seen_residues.add(res_key)
                            result["residues_per_chain"].setdefault(chain, 0)
                            result["residues_per_chain"][chain] += 1

                        # Coordinate ranges
                        for coord, val in [("x", x), ("y", y), ("z", z)]:
                            result["coordinate_ranges"][coord][0] = min(
                                result["coordinate_ranges"][coord][0], val
                            )
                            result["coordinate_ranges"][coord][1] = max(
                                result["coordinate_ranges"][coord][1], val
                            )

                        if bfactor != 0.0:
                            result["has_bfactors"] = True
                    except (ValueError, IndexError) as e:
                        result["errors"].append(f"Parse error in ATOM record: {e}")

                elif record == "TER":
                    result["has_ter_records"] = True
                elif record == "END":
                    result["has_end_record"] = True

    except Exception as e:
        result["errors"].append(f"Read error: {e}")

    result["num_residues"] = len(seen_residues)
    result["chain_ids"] = sorted(result["chain_ids"])

    if expected_chains is not None:
        actual = set(result["chain_ids"])
        expected = set(expected_chains)
        if actual != expected:
            result["errors"].append(
                f"Chain mismatch: expected {expected}, got {actual}"
            )

    if min_atoms and result["num_atoms"] < min_atoms:
        result["errors"].append(
            f"Too few atoms: {result['num_atoms']} < {min_atoms}"
        )

    return result


def validate_json(
    path: Path,
    *,
    required_fields: list[str] | None = None,
    expected_type: str = "any",
    min_entries: int = 0,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Validate a JSON file.

    Args:
        path: Path to JSON file.
        required_fields: Fields each entry (if list) or root (if dict) must have.
        expected_type: Expected root type ('list', 'dict', 'any').
        min_entries: Minimum entries (for list root).
        max_entries: Maximum entries (for list root).

    Returns:
        Dict with: exists, valid_json, root_type, num_entries, has_required_fields,
        missing_fields, data, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "valid_json": False,
        "root_type": None,
        "num_entries": 0,
        "has_required_fields": True,
        "missing_fields": [],
        "data": None,
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    try:
        with open(path) as f:
            data = json.load(f)
        result["valid_json"] = True
        result["data"] = data
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
        return result

    # Root type check
    if isinstance(data, list):
        result["root_type"] = "list"
        result["num_entries"] = len(data)
    elif isinstance(data, dict):
        result["root_type"] = "dict"
        result["num_entries"] = len(data)
    else:
        result["root_type"] = type(data).__name__
        result["num_entries"] = 1

    if expected_type != "any" and result["root_type"] != expected_type:
        result["errors"].append(
            f"Expected root type '{expected_type}', got '{result['root_type']}'"
        )

    # Required fields check
    if required_fields:
        if result["root_type"] == "list":
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    for field in required_fields:
                        if field not in entry:
                            result["has_required_fields"] = False
                            result["missing_fields"].append(f"[{i}].{field}")
                else:
                    result["has_required_fields"] = False
                    result["errors"].append(f"Entry [{i}] is not a dict")
        elif result["root_type"] == "dict":
            for field in required_fields:
                if field not in data:
                    result["has_required_fields"] = False
                    result["missing_fields"].append(field)

    # Count checks for list type
    if result["root_type"] == "list":
        if min_entries and result["num_entries"] < min_entries:
            result["errors"].append(
                f"Too few entries: {result['num_entries']} < {min_entries}"
            )
        if max_entries is not None and result["num_entries"] > max_entries:
            result["errors"].append(
                f"Too many entries: {result['num_entries']} > {max_entries}"
            )

    return result


def validate_csv(
    path: Path,
    *,
    delimiter: str = ",",
    required_columns: list[str] | None = None,
    min_rows: int = 0,
    max_rows: int | None = None,
    has_header: bool = True,
) -> dict[str, Any]:
    """Validate a CSV/TSV file.

    Returns:
        Dict with: exists, num_rows, headers, has_required_columns,
        missing_columns, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "num_rows": 0,
        "headers": [],
        "has_required_columns": True,
        "missing_columns": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    try:
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            result["errors"].append("File is empty")
            return result

        if has_header:
            result["headers"] = rows[0]
            result["num_rows"] = len(rows) - 1
        else:
            result["num_rows"] = len(rows)

        if required_columns and has_header:
            header_set = set(result["headers"])
            for col in required_columns:
                if col not in header_set:
                    result["has_required_columns"] = False
                    result["missing_columns"].append(col)

        if min_rows and result["num_rows"] < min_rows:
            result["errors"].append(
                f"Too few rows: {result['num_rows']} < {min_rows}"
            )
        if max_rows is not None and result["num_rows"] > max_rows:
            result["errors"].append(
                f"Too many rows: {result['num_rows']} > {max_rows}"
            )

    except Exception as e:
        result["errors"].append(f"Read error: {e}")

    return result


def validate_tsv(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Validate a TSV file. Delegates to validate_csv with tab delimiter."""
    return validate_csv(path, delimiter="\t", **kwargs)


def validate_newick(path: Path, *, expected_leaves: list[str] | None = None) -> dict[str, Any]:
    """Validate a Newick tree file.

    Returns:
        Dict with: exists, valid_tree, num_leaves, leaf_names, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "valid_tree": False,
        "num_leaves": 0,
        "leaf_names": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    try:
        content = path.read_text().strip()
        if not content:
            result["errors"].append("File is empty")
            return result

        # Basic Newick validation: must end with semicolon
        if not content.endswith(";"):
            result["errors"].append("Newick string must end with ';'")
            return result

        # Check balanced parentheses
        depth = 0
        for ch in content:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                result["errors"].append("Unbalanced parentheses")
                return result
        if depth != 0:
            result["errors"].append("Unbalanced parentheses")
            return result

        result["valid_tree"] = True

        # Extract leaf names (tokens not inside parentheses, before : or ,)
        # Simple regex approach: find names between commas/parens
        tree_str = content.rstrip(";")
        # Remove branch lengths
        tree_str = re.sub(r":[0-9eE.+-]+", "", tree_str)
        # Remove parentheses
        tree_str = tree_str.replace("(", ",").replace(")", ",")
        # Split and filter
        leaves = [t.strip() for t in tree_str.split(",") if t.strip()]
        result["leaf_names"] = leaves
        result["num_leaves"] = len(leaves)

        if expected_leaves is not None:
            actual = set(leaves)
            expected = set(expected_leaves)
            if actual != expected:
                result["errors"].append(
                    f"Leaf mismatch: missing {expected - actual}, extra {actual - expected}"
                )

    except Exception as e:
        result["errors"].append(f"Read error: {e}")

    return result


def validate_png(path: Path) -> dict[str, Any]:
    """Validate a PNG image file.

    Returns:
        Dict with: exists, valid_png, width, height, errors.
    """
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    result: dict[str, Any] = {
        "exists": False,
        "valid_png": False,
        "width": 0,
        "height": 0,
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    try:
        with open(path, "rb") as f:
            header = f.read(24)

        if len(header) < 24:
            result["errors"].append("File too small to be a valid PNG")
            return result

        if header[:8] != PNG_MAGIC:
            result["errors"].append("Invalid PNG magic bytes")
            return result

        result["valid_png"] = True
        # IHDR chunk: width and height are at bytes 16-23
        width, height = struct.unpack(">II", header[16:24])
        result["width"] = width
        result["height"] = height

    except Exception as e:
        result["errors"].append(f"Read error: {e}")

    return result


def validate_yaml(
    path: Path,
    *,
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a YAML file.

    Returns:
        Dict with: exists, valid_yaml, data, has_required_keys, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "valid_yaml": False,
        "data": None,
        "has_required_keys": True,
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        result["valid_yaml"] = True
        result["data"] = data
    except ImportError:
        result["errors"].append("PyYAML not installed")
        return result
    except Exception as e:
        result["errors"].append(f"Invalid YAML: {e}")
        return result

    if required_keys and isinstance(data, dict):
        for key in required_keys:
            if key not in data:
                result["has_required_keys"] = False
                result["errors"].append(f"Missing required key: {key}")

    return result


def validate_msa(
    path: Path,
    *,
    format: str = "fasta",
    alphabet: str = "protein_ext",
    min_sequences: int = 2,
) -> dict[str, Any]:
    """Validate a multiple sequence alignment file.

    Args:
        path: Path to MSA file.
        format: Alignment format ('fasta', 'clustal', 'a3m', 'stockholm').
        alphabet: Character alphabet for validation.
        min_sequences: Minimum number of sequences.

    Returns:
        Dict with: exists, num_sequences, alignment_length, all_same_length,
        gap_fraction, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "num_sequences": 0,
        "alignment_length": 0,
        "all_same_length": True,
        "gap_fraction": 0.0,
        "sequence_ids": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    # For FASTA-based MSA formats, parse as FASTA with gaps allowed
    gap_alphabet = {"protein_ext", "dna_ext", "rna_ext"}
    if alphabet not in gap_alphabet:
        alphabet = alphabet + "_ext" if alphabet + "_ext" in {
            "protein_ext", "dna_ext", "rna_ext"
        } else "protein_ext"

    alpha_map = {
        "protein_ext": PROTEIN_ALPHABET_EXTENDED,
        "dna_ext": DNA_ALPHABET_EXTENDED,
        "rna_ext": RNA_ALPHABET_EXTENDED,
    }
    valid_chars = alpha_map.get(alphabet, PROTEIN_ALPHABET_EXTENDED)

    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq = ""

    try:
        with open(path) as f:
            content = f.read()

        if format in ("fasta", "a3m"):
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        sequences[current_id] = current_seq
                    match = re.match(r">(\S+)", line)
                    current_id = match.group(1) if match else f"seq_{len(sequences)}"
                    current_seq = ""
                else:
                    current_seq += line
            if current_id is not None:
                sequences[current_id] = current_seq
        elif format == "stockholm":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#") or line.startswith("//") or not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    seq_id, seq = parts
                    sequences.setdefault(seq_id, "")
                    sequences[seq_id] += seq
        else:
            result["errors"].append(f"Unsupported MSA format: {format}")
            return result

    except Exception as e:
        result["errors"].append(f"Read error: {e}")
        return result

    result["num_sequences"] = len(sequences)
    result["sequence_ids"] = list(sequences.keys())

    if not sequences:
        result["errors"].append("No sequences found")
        return result

    # Check alignment consistency
    lengths = [len(s) for s in sequences.values()]
    result["alignment_length"] = lengths[0] if lengths else 0
    result["all_same_length"] = len(set(lengths)) == 1

    if not result["all_same_length"]:
        result["errors"].append(
            f"Inconsistent lengths: {min(lengths)}-{max(lengths)}"
        )

    # Gap fraction
    total_chars = sum(lengths)
    gap_chars = sum(s.count("-") for s in sequences.values())
    result["gap_fraction"] = gap_chars / total_chars if total_chars > 0 else 0.0

    if result["num_sequences"] < min_sequences:
        result["errors"].append(
            f"Too few sequences: {result['num_sequences']} < {min_sequences}"
        )

    return result


def validate_text_file(
    path: Path,
    *,
    min_lines: int = 0,
    max_lines: int | None = None,
    contains: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a generic text file.

    Returns:
        Dict with: exists, num_lines, non_empty, contains_all, errors.
    """
    result: dict[str, Any] = {
        "exists": False,
        "num_lines": 0,
        "non_empty": False,
        "contains_all": True,
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    result["exists"] = True

    try:
        content = path.read_text()
        lines = content.splitlines()
        result["num_lines"] = len(lines)
        result["non_empty"] = bool(content.strip())

        if contains:
            for term in contains:
                if term not in content:
                    result["contains_all"] = False
                    result["errors"].append(f"Missing expected content: '{term}'")

        if min_lines and result["num_lines"] < min_lines:
            result["errors"].append(
                f"Too few lines: {result['num_lines']} < {min_lines}"
            )
        if max_lines is not None and result["num_lines"] > max_lines:
            result["errors"].append(
                f"Too many lines: {result['num_lines']} > {max_lines}"
            )

    except Exception as e:
        result["errors"].append(f"Read error: {e}")

    return result
