"""Tier 2 Design Evaluator - orchestrates all scoring components.

Supports hybrid algorithmic + LLM judge scoring when ``llm_judge_enabled``
is True.  In hybrid mode, algorithmic scores are split into their algo
portion (72 pts) and LLM judge scores fill the remaining 28 pts.
Quality remains 100% algorithmic (35 pts).

Usage:
    evaluator = Tier2Evaluator()
    result = evaluator.evaluate(
        task_id="binder_001",
        output_dir=Path("results/claude/binder_001"),
        tools_used=["rfdiffusion", "proteinmpnn", "alphafold2"],
        tools_expected=["rfdiffusion", "proteinmpnn", "alphafold2"],
    )

    # With LLM judge (dry run):
    evaluator = Tier2Evaluator(llm_judge_enabled=True, llm_judge_dry_run=True)
    result = evaluator.evaluate(
        task_id="binder_001",
        output_dir=Path("..."),
        agent_id="claude-code",
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from biodesignbench.eval.metrics.approach import score_approach
from biodesignbench.eval.metrics.orchestration import score_orchestration
from biodesignbench.eval.tier2.ground_truth import (
    get_evaluation_thresholds,
    get_reference_sequence,
    load_ground_truth,
)
from biodesignbench.eval.tier2.oracle import get_oracle_sequences
from biodesignbench.eval.tier2.scoring import (
    DesignScoringRubric,
    apply_design_gate,
    calculate_design_score,
    score_diversity,
    score_feasibility,
    score_novelty,
    score_quality,
)
from biodesignbench.eval.tier2.validators import (
    extract_designs_from_fasta,
    extract_metrics_from_json,
    validate_design_output,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Tier2Evaluator:
    """Orchestrates Tier 2 design evaluation.

    Args:
        rubric: Scoring rubric (default 100-point design rubric).
        ground_truth_dir: Override directory for ground truth files.
        llm_judge_enabled: If True, run LLM judge panel for subjective
            dimensions (approach, orchestration, feasibility, novelty,
            diversity).
        llm_judge_dry_run: If True, LLM judges return deterministic
            midpoint scores without making API calls.
    """

    def __init__(
        self,
        rubric: DesignScoringRubric | None = None,
        ground_truth_dir: Path | None = None,
        llm_judge_enabled: bool = False,
        llm_judge_dry_run: bool = False,
    ):
        self.rubric = rubric or DesignScoringRubric()
        self.ground_truth_dir = ground_truth_dir
        self.llm_judge_enabled = llm_judge_enabled
        self.llm_judge_dry_run = llm_judge_dry_run

    def evaluate(
        self,
        task_id: str,
        output_dir: Path,
        tools_used: list[str] | None = None,
        tools_expected: list[str] | None = None,
        tool_call_log: list[dict] | None = None,
        iterations: int = 1,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation for a design task.

        Args:
            task_id: Task identifier.
            output_dir: Directory containing agent outputs.
            tools_used: Tools the agent actually used.
            tools_expected: Tools expected for this task type.
            tool_call_log: Structured log of tool calls.
            iterations: Number of design iterations.
            agent_id: Agent identifier (needed for LLM judge panel
                to determine which model family to exclude).

        Returns:
            Dict with: total_score, breakdown, validation, metrics, errors.
        """
        tools_used = tools_used or []
        tools_expected = tools_expected or []
        errors: list[str] = []

        # Load task data
        task_data = self._load_task_data(task_id)
        constraints = task_data.get("design_constraints", {}) if task_data else {}
        max_designs = constraints.get("max_designs", 10)
        length_range = constraints.get("length_range")
        if isinstance(length_range, list):
            length_range = tuple(length_range)

        # Load ground truth thresholds
        try:
            thresholds = get_evaluation_thresholds(
                task_id, self.ground_truth_dir
            )
        except FileNotFoundError:
            thresholds = {}
            errors.append(f"Ground truth not found for {task_id}")

        # Get reference sequence
        reference_seq = get_reference_sequence(task_id, task_data)

        # Validate output directory
        required_files = ["designed_sequences.fasta", "metrics.json"]
        if task_data:
            expected_output = task_data.get("expected_output", {})
            required_files = expected_output.get("required_files", required_files)

        validation = validate_design_output(
            output_dir,
            required_files=required_files,
            max_designs=max_designs,
            length_range=length_range,
        )

        # Extract designs and metrics
        fasta_path = output_dir / "designed_sequences.fasta"
        metrics_path = output_dir / "metrics.json"

        designs_raw = extract_designs_from_fasta(fasta_path)
        design_sequences = [d["sequence"] for d in designs_raw]
        agent_metrics = extract_metrics_from_json(metrics_path)

        # Score each component
        from biodesignbench.taxonomy import get_category

        category = get_category(task_id)
        task_type_enum = category.task_type if category else None
        approach_result = score_approach(
            tools_used, tools_expected,
            max_points=self.rubric.components.get("approach", 20),
            task_type=task_type_enum,
        )

        orchestration_result = score_orchestration(
            tool_call_log or [],
            task_id=task_id,
            max_points=self.rubric.components.get("orchestration", 15),
        )

        # Load oracle sequences for functional similarity (non-binding Tier B)
        oracle_seqs = get_oracle_sequences(task_id)

        quality_result = score_quality(
            agent_metrics, thresholds,
            max_points=self.rubric.components.get("quality", 35),
            task_id=task_id,
            designs=design_sequences or None,
            oracle_sequences=oracle_seqs or None,
        )

        novelty_result = score_novelty(
            design_sequences, reference_seq, thresholds,
            max_points=self.rubric.components.get("novelty", 5),
        )

        diversity_result = score_diversity(
            design_sequences,
            max_designs=max_designs,
            max_points=self.rubric.components.get("diversity", 5),
        )

        feasibility_result = score_feasibility(
            design_sequences, constraints,
            max_points=self.rubric.components.get("feasibility", 15),
        )

        # Calculate component scores
        component_scores = {
            "approach": approach_result["score"],
            "orchestration": orchestration_result["score"],
            "quality": quality_result["score"],
            "novelty": novelty_result["score"],
            "diversity": diversity_result["score"],
            "feasibility": feasibility_result["score"],
        }

        # --- LLM Judge Integration ---
        llm_judge_result = None
        if self.llm_judge_enabled:
            llm_judge_result = self._run_llm_judge(
                task_data=task_data,
                task_id=task_id,
                tool_call_log=tool_call_log or [],
                design_sequences=design_sequences,
                agent_metrics=agent_metrics,
                agent_id=agent_id,
                component_scores=component_scores,
            )

        # Apply design gate: cap at 30 if no designs produced
        component_scores = apply_design_gate(component_scores, len(designs_raw))

        final = calculate_design_score(self.rubric, component_scores)

        result = {
            "total_score": final["total"],
            "percentage": final["percentage"],
            "breakdown": final["breakdown"],
            "validation": validation,
            "metrics": {
                "approach": approach_result,
                "orchestration": orchestration_result,
                "quality": quality_result,
                "novelty": novelty_result,
                "diversity": diversity_result,
                "feasibility": feasibility_result,
            },
            "agent_metrics": agent_metrics,
            "num_designs": len(designs_raw),
            "errors": errors,
        }

        if llm_judge_result is not None:
            result["llm_judge"] = llm_judge_result

        return result

    def _run_llm_judge(
        self,
        task_data: dict | None,
        task_id: str,
        tool_call_log: list[dict],
        design_sequences: list[str],
        agent_metrics: dict[str, Any],
        agent_id: str | None,
        component_scores: dict[str, int],
    ) -> dict[str, Any]:
        """Run LLM judge panel and merge scores into component_scores.

        Modifies component_scores in place: splits algo portion, adds LLM
        judge scores.

        Returns:
            LLM judge result dict for reporting.
        """
        from biodesignbench.eval.llm_judge.aggregation import (
            merge_algo_and_judge_scores,
            split_algo_score,
        )
        from biodesignbench.eval.llm_judge.panel import (
            LLMJudgePanel,
            detect_agent_family,
        )
        from biodesignbench.eval.metrics.orchestration import EXPECTED_PIPELINES

        # Determine agent family for judge exclusion
        family = detect_agent_family(agent_id) if agent_id else "unknown"

        # Build task description from task data
        task_desc = ""
        if task_data:
            task_desc = task_data.get("description", "")

        # Get reference pipeline
        from biodesignbench.taxonomy import get_category
        category = get_category(task_id)
        ref_pipeline = None
        if category:
            ref_pipeline = EXPECTED_PIPELINES.get(category.task_type.value)

        # Run panel
        panel = LLMJudgePanel(
            agent_model_family=family,
            dry_run=self.llm_judge_dry_run,
        )

        logger.info(
            "Running LLM judge panel: %d judges (family=%s, dry_run=%s)",
            len(panel.judges), family, self.llm_judge_dry_run,
        )

        panel_result = panel.evaluate_sync(
            task_description=task_desc,
            tool_call_log=tool_call_log,
            designed_sequences=design_sequences,
            algorithmic_metrics=agent_metrics,
            reference_pipeline=ref_pipeline,
        )

        # Split algo scores to their reduced portion
        split_scores = {}
        for comp, score in component_scores.items():
            original_max = self.rubric.components.get(comp, 0)
            split_scores[comp] = split_algo_score(comp, score, original_max)

        # Merge algo + LLM judge scores
        merged = merge_algo_and_judge_scores(split_scores, panel_result)

        # Update component_scores in place with merged values
        for comp in component_scores:
            component_scores[comp] = int(round(merged.get(comp, component_scores[comp])))

        return panel_result

    def _load_task_data(self, task_id: str) -> dict[str, Any] | None:
        """Load task JSON data."""
        task_path = _PROJECT_ROOT / "tasks" / "tier2" / f"{task_id}.json"
        if not task_path.exists():
            return None
        with open(task_path) as f:
            return json.load(f)
