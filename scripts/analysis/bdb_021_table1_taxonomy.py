#!/usr/bin/env python3
"""BDB-021: Table 1 -- Task Taxonomy Matrix for NMI paper.

Generates a 2x5 taxonomy table showing the distribution of 76 Tier 2 design
tasks across DesignApproach (de_novo, redesign) and MolecularSubject
(antibody, enzyme, binder, scaffold, fluorescent_protein).

Outputs:
    results/analysis/table1_taxonomy.csv
    results/analysis/table1_taxonomy_latex.tex

Usage:
    python scripts/analysis/bdb_021_table1_taxonomy.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from biodesignbench.taxonomy import (
    DesignApproach,
    MolecularSubject,
    OLD_TO_NEW_MAPPING,
    get_category,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPROACHES: list[DesignApproach] = [DesignApproach.DE_NOVO, DesignApproach.REDESIGN]
SUBJECTS: list[MolecularSubject] = [
    MolecularSubject.ANTIBODY,
    MolecularSubject.ENZYME,
    MolecularSubject.BINDER,
    MolecularSubject.SCAFFOLD,
    MolecularSubject.FLUORESCENT_PROTEIN,
]

APPROACH_DISPLAY: dict[str, str] = {
    "de_novo": "De novo",
    "redesign": "Redesign",
}

SUBJECT_DISPLAY: dict[str, str] = {
    "antibody": "Antibody",
    "enzyme": "Enzyme",
    "binder": "Binder",
    "scaffold": "Scaffold",
    "fluorescent_protein": "Fluorescent Protein",
}

SUBJECT_DISPLAY_SHORT: dict[str, str] = {
    "antibody": "Antibody",
    "enzyme": "Enzyme",
    "binder": "Binder",
    "scaffold": "Scaffold",
    "fluorescent_protein": "Fluor.\\ Prot.",
}


# ---------------------------------------------------------------------------
# Task metadata loading
# ---------------------------------------------------------------------------


def _load_task_jsons() -> dict[str, dict[str, Any]]:
    """Load all Tier 2 task JSONs into a dict keyed by task_id.

    Returns:
        Mapping from task_id to full task JSON dict.
    """
    tasks_dir = PROJECT_ROOT / "tasks" / "tier2"
    tasks: dict[str, dict[str, Any]] = {}
    for f in sorted(tasks_dir.glob("*.json")):
        with open(f) as fh:
            t = json.load(fh)
        tasks[t["task_id"]] = t
    return tasks


def _first_sentence(description: str) -> str:
    """Extract the first sentence from a task description.

    Args:
        description: Full task description text.

    Returns:
        First sentence (up to the first period), or the full text if no
        period is found.
    """
    if not description:
        return ""
    idx = description.find(".")
    if idx >= 0:
        return description[: idx + 1]
    return description


# ---------------------------------------------------------------------------
# Core: build the taxonomy table data
# ---------------------------------------------------------------------------


def build_taxonomy_table() -> list[dict[str, Any]]:
    """Build the 2x5 taxonomy table as a list of row dicts.

    Each row represents one cell of the DesignApproach x MolecularSubject
    matrix and contains:
        - approach: str (e.g. "de_novo")
        - subject: str (e.g. "antibody")
        - count: int
        - example_task_id: str (first task alphabetically, or "---" if empty)
        - example_description: str (first sentence)
        - main_task_type: str (task_type field from JSON)

    Returns:
        List of 10 row dicts (2 approaches x 5 subjects).
    """
    task_jsons = _load_task_jsons()

    # Group old task IDs by (approach, subject)
    cell_tasks: dict[tuple[str, str], list[str]] = {}
    for approach in APPROACHES:
        for subject in SUBJECTS:
            cell_tasks[(approach.value, subject.value)] = []

    for old_id in OLD_TO_NEW_MAPPING:
        cat = get_category(old_id)
        if cat is None:
            continue
        key = (cat.approach.value, cat.subject.value)
        if key in cell_tasks:
            cell_tasks[key].append(old_id)

    # Build rows
    rows: list[dict[str, Any]] = []
    for approach in APPROACHES:
        for subject in SUBJECTS:
            key = (approach.value, subject.value)
            task_ids = sorted(cell_tasks[key])
            count = len(task_ids)

            if count == 0:
                rows.append(
                    {
                        "approach": approach.value,
                        "subject": subject.value,
                        "count": 0,
                        "example_task_id": "---",
                        "example_description": "",
                        "main_task_type": "---",
                    }
                )
            else:
                # Pick first task alphabetically as representative example
                example_id = task_ids[0]
                task_json = task_jsons.get(example_id, {})
                desc = task_json.get("description", "")
                task_type = task_json.get("task_type", "unknown")

                rows.append(
                    {
                        "approach": approach.value,
                        "subject": subject.value,
                        "count": count,
                        "example_task_id": example_id,
                        "example_description": _first_sentence(desc),
                        "main_task_type": task_type,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save the taxonomy table as a CSV file.

    Args:
        rows: List of row dicts from build_taxonomy_table().
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "approach",
        "subject",
        "count",
        "example_task_id",
        "example_description",
        "main_task_type",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Output: LaTeX
# ---------------------------------------------------------------------------


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text.

    Args:
        text: Raw text string.

    Returns:
        LaTeX-safe string with special characters escaped.
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def save_latex(rows: list[dict[str, Any]], path: Path) -> None:
    """Save the taxonomy table as a LaTeX file in NMI format.

    The table has:
    - Rows: De novo, Redesign, Total
    - Columns: Antibody, Enzyme, Binder, Scaffold, Fluorescent Protein, Total
    - Each cell shows count and a representative example
    - Empty cells (rd_bnd=0) shown as "---"

    Args:
        rows: List of row dicts from build_taxonomy_table().
        path: Output .tex path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Index rows by (approach, subject) for easy lookup
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        lookup[(r["approach"], r["subject"])] = r

    # Compute totals
    subject_totals: dict[str, int] = {}
    for subj in SUBJECTS:
        subject_totals[subj.value] = sum(
            lookup.get((a.value, subj.value), {}).get("count", 0) for a in APPROACHES
        )

    approach_totals: dict[str, int] = {}
    for appr in APPROACHES:
        approach_totals[appr.value] = sum(
            lookup.get((appr.value, s.value), {}).get("count", 0) for s in SUBJECTS
        )

    grand_total = sum(approach_totals.values())

    # Build LaTeX
    n_cols = len(SUBJECTS) + 2  # approach col + 5 subjects + total
    col_spec = "l" + "c" * (len(SUBJECTS) + 1)

    lines: list[str] = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Task taxonomy matrix. Each cell shows the number of tasks")
    lines.append(r"and a representative example. The benchmark spans two design approaches")
    lines.append(r"(de novo generation vs.\ redesign of existing sequences) across five")
    lines.append(r"molecular subjects.}")
    lines.append(r"\label{tab:taxonomy}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    header_cells = [""]
    for subj in SUBJECTS:
        header_cells.append(r"\textbf{" + SUBJECT_DISPLAY_SHORT[subj.value] + "}")
    header_cells.append(r"\textbf{Total}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for appr in APPROACHES:
        row_cells: list[str] = [r"\textbf{" + APPROACH_DISPLAY[appr.value] + "}"]
        for subj in SUBJECTS:
            cell = lookup.get((appr.value, subj.value))
            if cell is None or cell["count"] == 0:
                row_cells.append("---")
            else:
                count = cell["count"]
                example_id = _escape_latex(cell["example_task_id"])
                row_cells.append(f"{count}")
        # Row total
        row_cells.append(str(approach_totals[appr.value]))
        lines.append(" & ".join(row_cells) + r" \\")

    # Total row
    lines.append(r"\midrule")
    total_cells: list[str] = [r"\textbf{Total}"]
    for subj in SUBJECTS:
        total_cells.append(str(subject_totals[subj.value]))
    total_cells.append(str(grand_total))
    lines.append(" & ".join(total_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(
        r"\raggedright\footnotesize\textit{Note:} The redesign $\times$ binder cell "
        r"is empty (---) because existing binder redesign tasks were reclassified "
        r"under enzyme or scaffold subjects based on their primary design objective."
    )
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Pretty-print to stdout
# ---------------------------------------------------------------------------


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print the taxonomy matrix to stdout in a readable format.

    Args:
        rows: List of row dicts from build_taxonomy_table().
    """
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        lookup[(r["approach"], r["subject"])] = r

    # Column widths
    subj_labels = [SUBJECT_DISPLAY[s.value] for s in SUBJECTS]
    col_w = max(len(s) for s in subj_labels + ["Total"]) + 2

    # Header
    header = f"{'':18s}"
    for s in SUBJECTS:
        header += f"{SUBJECT_DISPLAY[s.value]:>{col_w}s}"
    header += f"{'Total':>{col_w}s}"
    print(header)
    print("-" * len(header))

    # Data rows
    for appr in APPROACHES:
        label = APPROACH_DISPLAY[appr.value]
        row_str = f"{label:18s}"
        row_total = 0
        for subj in SUBJECTS:
            cell = lookup.get((appr.value, subj.value))
            if cell is None or cell["count"] == 0:
                row_str += f"{'---':>{col_w}s}"
            else:
                row_str += f"{cell['count']:>{col_w}d}"
                row_total += cell["count"]
        row_str += f"{row_total:>{col_w}d}"
        print(row_str)

    # Total row
    print("-" * len(header))
    total_str = f"{'Total':18s}"
    grand = 0
    for subj in SUBJECTS:
        col_total = sum(
            lookup.get((a.value, subj.value), {}).get("count", 0) for a in APPROACHES
        )
        total_str += f"{col_total:>{col_w}d}"
        grand += col_total
    total_str += f"{grand:>{col_w}d}"
    print(total_str)

    # Example details
    print("\nRepresentative examples per cell:")
    print("-" * 80)
    for appr in APPROACHES:
        for subj in SUBJECTS:
            cell = lookup.get((appr.value, subj.value))
            if cell is None or cell["count"] == 0:
                continue
            cat_id = f"{appr.short}_{subj.short}"
            desc = cell["example_description"]
            if len(desc) > 70:
                desc = desc[:67] + "..."
            print(
                f"  {cat_id:8s} ({cell['count']:2d}) "
                f"{cell['example_task_id']:20s} {desc}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate Table 1: Task Taxonomy Matrix."""
    out_dir = PROJECT_ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_taxonomy_table()

    # Save outputs
    csv_path = out_dir / "table1_taxonomy.csv"
    tex_path = out_dir / "table1_taxonomy_latex.tex"

    save_csv(rows, csv_path)
    save_latex(rows, tex_path)

    # Print to stdout
    print("=" * 80)
    print("Table 1: Task Taxonomy Matrix (2x5)")
    print("=" * 80)
    print()
    print_table(rows)
    print()
    print(f"CSV saved to:   {csv_path}")
    print(f"LaTeX saved to: {tex_path}")


if __name__ == "__main__":
    main()
