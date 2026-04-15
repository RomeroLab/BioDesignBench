"""LLM-as-a-Judge scoring for BioDesignBench Tier 2 evaluation.

Provides cross-model LLM judge panels that evaluate subjective dimensions
(approach, orchestration, feasibility, novelty, diversity) while quality
metrics remain 100% algorithmic.

Usage:
    from biodesignbench.eval.llm_judge import LLMJudgePanel

    panel = LLMJudgePanel(agent_model_family="anthropic", dry_run=True)
    result = panel.evaluate_sync(
        task_description="Design a binder for IL-6R",
        tool_call_log=[...],
        designed_sequences=["MKVL..."],
        algorithmic_metrics={"pLDDT": 82.5},
    )
"""

from biodesignbench.eval.llm_judge.aggregation import (
    WEIGHT_SPLIT,
    aggregate_judge_scores,
    merge_algo_and_judge_scores,
    split_algo_score,
)
from biodesignbench.eval.llm_judge.judge import LLMJudge, parse_judge_response
from biodesignbench.eval.llm_judge.panel import (
    LLMJudgePanel,
    detect_agent_family,
    get_judge_models,
)
from biodesignbench.eval.llm_judge.rubrics import (
    JUDGE_DIMENSIONS,
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)

__all__ = [
    "LLMJudge",
    "LLMJudgePanel",
    "JUDGE_DIMENSIONS",
    "JUDGE_SYSTEM_PROMPT",
    "WEIGHT_SPLIT",
    "aggregate_judge_scores",
    "build_judge_prompt",
    "detect_agent_family",
    "get_judge_models",
    "merge_algo_and_judge_scores",
    "parse_judge_response",
    "split_algo_score",
]
