"""Post-hoc data contamination detection.

Scans agent reasoning traces and generated code for evidence that the
agent accessed source papers, DOIs, or GitHub issues associated with the
task.  Produces a ``ContaminationReport`` with flags, evidence snippets,
and a 0-1 contamination score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# DOI regex: matches "10.<registrant>/<suffix>"
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\"']+")

# GitHub issue/PR URL regex
_GITHUB_ISSUE_RE = re.compile(
    r"https?://github\.com/[^\s/]+/[^\s/]+/(?:issues|pull)/\d+"
)

# Known source paper names and author patterns that indicate the agent
# identified the task's origin.  Each entry is (pattern, label).
_SOURCE_PAPER_PATTERNS: list[tuple[str, str]] = [
    (r"\bBindCraft\b.*\bPacesa\b", "BindCraft (Pacesa)"),
    (r"\bPacesa\s+et\s+al", "Pacesa et al."),
    (r"\bAlphaProteo\b.*\bZambaldi\b", "AlphaProteo (Zambaldi)"),
    (r"\bZambaldi\s+et\s+al", "Zambaldi et al."),
    (r"\bAbBiBench\b", "AbBiBench"),
]


@dataclass
class ContaminationReport:
    """Result of contamination scanning for a single agent-task pair."""

    task_id: str
    agent_id: str
    flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    contamination_score: float = 0.0


def detect_contamination(
    task_id: str,
    agent_id: str,
    reasoning_trace: str,
    code: str,
    task_source: str | None = None,
    task_doi: str | None = None,
) -> ContaminationReport:
    """Scan agent output for contamination signals.

    Args:
        task_id: The task identifier.
        agent_id: The agent identifier.
        reasoning_trace: Agent's reasoning / chain-of-thought text.
        code: Agent-generated code.
        task_source: The task's ``metadata.source`` from the *original*
            (unsanitized) task, e.g. ``"BindCraft (Pacesa et al. 2024)"``.
        task_doi: The task's ``metadata.doi`` from the *original* task.

    Returns:
        A ``ContaminationReport`` with flags, evidence, and score.
    """
    report = ContaminationReport(task_id=task_id, agent_id=agent_id)
    combined_text = f"{reasoning_trace}\n{code}"

    if not combined_text.strip():
        return report

    score = 0.0

    # 1. Check for exact DOI match
    if task_doi:
        if task_doi in combined_text:
            report.flags.append("exact_doi_match")
            report.evidence.append(f"Exact task DOI found: {task_doi}")
            score += 0.9

    # 2. Check for any DOI pattern (even if not the task's own DOI)
    doi_matches = _DOI_RE.findall(combined_text)
    # Filter out the already-caught exact match
    other_dois = [d for d in doi_matches if d != task_doi]
    if other_dois and "exact_doi_match" not in report.flags:
        # Any DOI reference is mildly suspicious (could be legitimate citation)
        report.flags.append("doi_pattern_found")
        report.evidence.append(f"DOI references found: {other_dois[:3]}")
        score += 0.3

    # 3. Check for GitHub issue/PR URLs
    gh_matches = _GITHUB_ISSUE_RE.findall(combined_text)
    if gh_matches:
        report.flags.append("github_issue_url")
        report.evidence.append(f"GitHub issue/PR URLs found: {gh_matches[:3]}")
        score += 0.5

    # 4. Check for source paper name/author references
    if task_source and task_source not in ("synthetic", "github_issue"):
        for pattern, label in _SOURCE_PAPER_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                report.flags.append("source_paper_name")
                report.evidence.append(
                    f"Source paper reference detected: {label}"
                )
                score += 0.7
                break  # One source paper match is enough

    report.contamination_score = min(score, 1.0)
    return report
