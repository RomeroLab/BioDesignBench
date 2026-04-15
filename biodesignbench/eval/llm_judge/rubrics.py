"""Structured rubric prompts for LLM judge evaluation.

Each judge evaluates 5 dimensions with explicit score-level descriptors
following the Prometheus (ICLR 2024) rubric-based approach.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Judge dimensions with max scores matching the LLM portion of the split
# ---------------------------------------------------------------------------

JUDGE_DIMENSIONS: dict[str, dict[str, Any]] = {
    "approach_strategy": {
        "max_score": 10,
        "description": "Strategic quality of tool/methodology selection",
    },
    "orchestration_reasoning": {
        "max_score": 8,
        "description": "Pipeline logic, error handling, and adaptive reasoning",
    },
    "bio_feasibility": {
        "max_score": 5,
        "description": "Biological plausibility beyond sequence-level checks",
    },
    "novelty_quality": {
        "max_score": 2,
        "description": "Meaningful innovation vs accidental variation",
    },
    "diversity_quality": {
        "max_score": 3,
        "description": "Functional diversity of design strategies",
    },
}


JUDGE_SYSTEM_PROMPT = (
    "You are an expert protein design evaluator with deep knowledge of "
    "computational protein engineering, including backbone generation "
    "(RFdiffusion, Chroma), sequence design (ProteinMPNN, LigandMPNN), "
    "structure prediction (AlphaFold2, ESMFold, Boltz), and interface "
    "analysis (PyRosetta, FoldX). You evaluate AI agent protein design "
    "attempts against a structured rubric. Score each dimension "
    "independently. Provide reasoning BEFORE your score. Be critical "
    "but fair — a score of 5/10 means average, not bad."
)


_RUBRIC_TEXT = """\
### Approach Strategy (0-10 pts)
- 9-10: Selects optimal tools for this specific target; demonstrates deep \
understanding of design strategy (e.g., chooses RFdiffusion hotspot \
conditioning for epitope-specific binder, not generic backbone generation)
- 7-8:  Appropriate tool selection with minor suboptimalities
- 5-6:  Reasonable tools but misses key steps or uses generic strategy
- 3-4:  Partially appropriate; missing critical tools for this task type
- 0-2:  Inappropriate or random tool selection

### Orchestration Reasoning (0-8 pts)
- 7-8: Logical pipeline with error handling, iterative refinement based on \
intermediate results, clear adaptive reasoning
- 5-6: Correct ordering with some validation but limited adaptation
- 3-4: Basic pipeline but missing intermediate checks or illogical ordering
- 0-2: No clear pipeline logic; tools called without reasoning

### Biological Feasibility (0-5 pts)
- 4-5: Designs are biologically plausible — CDR loops appropriate for \
target, active site geometry consistent, no obvious steric clashes
- 2-3: Generally plausible with minor concerns
- 0-1: Biologically implausible designs (e.g., all-alanine core, \
impossible disulfide patterns)

### Novelty Quality (0-2 pts)
- 2: Novel design represents meaningful innovation (new fold, creative \
binding mode) not just random mutations
- 1: Some novelty but appears accidental rather than designed
- 0: No meaningful novelty; trivially similar to reference or random

### Diversity Quality (0-3 pts)
- 3: Multiple designs explore different binding modes/conformations/\
strategies — functionally diverse, not just sequence variants
- 1-2: Some diversity but designs are minor variants of each other
- 0: No meaningful diversity; essentially one design repeated
"""


def build_judge_prompt(
    task_description: str,
    tool_call_log: list[dict[str, Any]],
    designed_sequences: list[str],
    algorithmic_metrics: dict[str, Any],
    reference_pipeline: list[str] | None = None,
) -> str:
    """Build the user prompt for LLM judge evaluation.

    Args:
        task_description: The original design task prompt.
        tool_call_log: Sequence of tool calls with args.
        designed_sequences: FASTA-format designed sequences.
        algorithmic_metrics: Computed metrics (pLDDT, ipTM, etc).
        reference_pipeline: Expected expert pipeline for this task type.

    Returns:
        Formatted prompt string for the judge LLM.
    """
    sections = []

    # Task description
    sections.append(f"## Task Description\n{task_description}")

    # Reference pipeline (for approach/orchestration context)
    if reference_pipeline:
        pipeline_str = " → ".join(reference_pipeline)
        sections.append(
            f"## Reference Pipeline (Expert-Validated)\n{pipeline_str}"
        )

    # Tool call log
    if tool_call_log:
        log_lines = []
        for i, entry in enumerate(tool_call_log, 1):
            tool = entry.get("tool", "unknown")
            args = entry.get("args_summary", {})
            args_str = json.dumps(args, default=str) if args else "{}"
            log_lines.append(f"{i}. {tool}({args_str})")
        sections.append(
            "## Agent's Tool Call Log\n" + "\n".join(log_lines)
        )
    else:
        sections.append("## Agent's Tool Call Log\nNo tool calls recorded.")

    # Designed sequences
    if designed_sequences:
        seq_lines = []
        for i, seq in enumerate(designed_sequences[:10], 1):  # Cap at 10
            display = seq[:80] + "..." if len(seq) > 80 else seq
            seq_lines.append(f">design_{i} (len={len(seq)})\n{display}")
        sections.append(
            f"## Designed Sequences ({len(designed_sequences)} total)\n"
            + "\n".join(seq_lines)
        )
    else:
        sections.append("## Designed Sequences\nNo sequences produced.")

    # Algorithmic metrics (read-only context)
    if algorithmic_metrics:
        metrics_str = json.dumps(algorithmic_metrics, indent=2, default=str)
        sections.append(
            f"## Algorithmic Metrics (Read-Only Context)\n```json\n{metrics_str}\n```"
        )

    # Scoring rubric
    sections.append(f"## Scoring Rubric\n{_RUBRIC_TEXT}")

    # Output format instruction
    output_format = {
        dim: {"reasoning": "...", "score": f"0-{info['max_score']}"}
        for dim, info in JUDGE_DIMENSIONS.items()
    }
    sections.append(
        "## Required Output Format\n"
        "Evaluate each dimension. For each:\n"
        "1. Cite specific evidence from the agent's work\n"
        "2. Reason about quality relative to the rubric\n"
        "3. Assign a score\n\n"
        "Respond in JSON format:\n"
        f"```json\n{json.dumps(output_format, indent=2)}\n```"
    )

    return "\n\n".join(sections)
