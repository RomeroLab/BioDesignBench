"""LLM-based plan evaluation: judge whether agent's reasoning trace
demonstrates understanding of each pipeline step.

Replaces keyword matching with LLM assessment of 4 pipeline steps:
  backbone_generation, sequence_design, structure_prediction, scoring_validation

Each step scored as 0 or 1 per judge, aggregated across 3-4 judges via majority vote.
"""
from __future__ import annotations

import json
import re
from typing import Any

from biodesignbench.eval.llm_judge.judge import LLMJudge

PLAN_EVAL_SYSTEM = """You are an expert protein design evaluator. Your task is to assess whether an AI agent's reasoning trace demonstrates awareness and planning of specific protein design pipeline steps.

You have deep knowledge of:
- RFdiffusion for backbone generation
- ProteinMPNN for inverse folding / sequence design
- AlphaFold2, ESMFold, Boltz for structure prediction
- Rosetta for energy scoring and validation

Be strict: the agent must show genuine understanding or intent to use a step, not just mention a keyword in passing. Look for evidence that the agent planned to perform the step as part of its design strategy."""

PLAN_EVAL_PROMPT_TEMPLATE = """## Task
{task_description}

## Agent's Reasoning Trace
{reasoning_trace}

## Pipeline Steps to Evaluate

For each step below, determine whether the agent's reasoning trace shows that the agent **planned or intended** to perform this step. Score 1 if the agent demonstrates clear awareness and intent, 0 if not.

1. **backbone_generation**: Did the agent plan to generate a de novo protein backbone/scaffold? (e.g., using RFdiffusion, backbone diffusion, scaffold generation, de novo structure design)

2. **sequence_design**: Did the agent plan to design/optimize amino acid sequences for the structure? (e.g., using ProteinMPNN, inverse folding, sequence optimization, fixed-backbone design)

3. **structure_prediction**: Did the agent plan to predict/validate the 3D structure of designed sequences? (e.g., using AlphaFold2, ESMFold, Boltz, checking pLDDT/pTM, fold confidence)

4. **scoring_validation**: Did the agent plan to score the design's energy/stability? (e.g., using Rosetta, energy minimization, interface analysis, ddG calculation, binding energy)

## Response Format
Return a JSON object with exactly this structure:
```json
{{
    "backbone_generation": {{"planned": 0 or 1, "evidence": "brief quote or reason"}},
    "sequence_design": {{"planned": 0 or 1, "evidence": "brief quote or reason"}},
    "structure_prediction": {{"planned": 0 or 1, "evidence": "brief quote or reason"}},
    "scoring_validation": {{"planned": 0 or 1, "evidence": "brief quote or reason"}}
}}
```
"""

STEPS = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]


def parse_plan_response(raw_text: str) -> dict[str, int]:
    """Parse LLM response into per-step binary scores."""
    # Try JSON extraction
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw_text

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
            except json.JSONDecodeError:
                return {s: 0 for s in STEPS}
        else:
            return {s: 0 for s in STEPS}

    result = {}
    for step in STEPS:
        if step in data and isinstance(data[step], dict):
            val = data[step].get("planned", 0)
            result[step] = 1 if val == 1 or val is True else 0
        else:
            result[step] = 0
    return result


def evaluate_plan_single(
    judge: LLMJudge,
    task_description: str,
    reasoning_trace: str,
) -> dict[str, int]:
    """Evaluate plan with a single judge."""
    if not reasoning_trace or len(reasoning_trace.strip()) < 10:
        return {s: 0 for s in STEPS}

    if judge.dry_run:
        return {s: 0 for s in STEPS}

    # Cap trace length
    trace = reasoning_trace[:4000]
    prompt = PLAN_EVAL_PROMPT_TEMPLATE.format(
        task_description=task_description[:1000],
        reasoning_trace=trace,
    )

    raw = judge._call_api(PLAN_EVAL_SYSTEM, prompt)
    return parse_plan_response(raw)


def evaluate_plan_panel(
    judges: list[LLMJudge],
    task_description: str,
    reasoning_trace: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate plan with multiple judges, aggregate via majority vote.

    Returns dict mapping step → {planned: 0/1, votes: [per-judge], n_judges: int}.
    """
    if not reasoning_trace or len(reasoning_trace.strip()) < 10:
        return {
            s: {"planned": 0, "votes": [0] * len(judges), "n_judges": len(judges)}
            for s in STEPS
        }

    all_votes: dict[str, list[int]] = {s: [] for s in STEPS}
    for judge in judges:
        result = evaluate_plan_single(judge, task_description, reasoning_trace)
        for step in STEPS:
            all_votes[step].append(result.get(step, 0))

    aggregated = {}
    for step in STEPS:
        votes = all_votes[step]
        planned = 1 if sum(votes) > len(votes) / 2 else 0  # majority vote
        aggregated[step] = {
            "planned": planned,
            "votes": votes,
            "n_judges": len(judges),
        }
    return aggregated
