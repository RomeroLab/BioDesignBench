"""Statistical analysis for benchmark score comparisons.

Provides significance testing between agents using non-parametric tests
(Wilcoxon, Mann-Whitney U) and ANOVA for multi-agent comparisons.

Usage::

    scores = {"gpt5": [45, 52, 38], "claude": [67, 72, 55]}
    report = compute_significance(scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class StatReport:
    """Statistical comparison report."""

    n_agents: int = 0
    p_value: float | None = None
    cohens_d: float | None = None
    test_name: str = ""
    bonferroni_corrected: float | None = None
    anova_p: float | None = None
    pairwise_results: list[dict[str, Any]] = field(default_factory=list)


def _cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
    pooled_std = ((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def compute_significance(
    scores: dict[str, list[float]],
) -> StatReport:
    """Compute statistical significance between agent scores.

    For 2 agents: uses Mann-Whitney U (unpaired) or Wilcoxon (paired if equal size).
    For 3+ agents: uses one-way ANOVA followed by pairwise tests.

    Args:
        scores: Dict mapping agent_id to list of task scores.

    Returns:
        StatReport with test results.
    """
    agents = sorted(scores.keys())
    n_agents = len(agents)

    if n_agents < 2:
        return StatReport(n_agents=n_agents, test_name="none")

    report = StatReport(n_agents=n_agents)

    if not HAS_SCIPY:
        report.test_name = "scipy_unavailable"
        return report

    # For 2 agents: pairwise test
    if n_agents == 2:
        g1 = scores[agents[0]]
        g2 = scores[agents[1]]

        if len(g1) < 3 or len(g2) < 3:
            report.test_name = "insufficient_data"
            return report

        # Use Mann-Whitney U (non-parametric, unpaired)
        stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        report.test_name = "mann_whitney_u"
        report.p_value = float(p)
        report.cohens_d = round(_cohens_d(g1, g2), 3)
        report.pairwise_results = [{
            "agent_a": agents[0],
            "agent_b": agents[1],
            "p_value": float(p),
            "cohens_d": report.cohens_d,
            "test": "mann_whitney_u",
        }]
        return report

    # For 3+ agents: ANOVA + pairwise
    all_groups = [scores[a] for a in agents]

    # Check minimum data
    if any(len(g) < 3 for g in all_groups):
        report.test_name = "insufficient_data"
        return report

    # One-way ANOVA
    f_stat, anova_p = scipy_stats.f_oneway(*all_groups)
    report.anova_p = float(anova_p)
    report.test_name = "anova_with_pairwise"

    # Pairwise Mann-Whitney U with Bonferroni correction
    n_comparisons = n_agents * (n_agents - 1) // 2
    pairwise = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            g1 = scores[agents[i]]
            g2 = scores[agents[j]]
            stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            corrected_p = min(float(p) * n_comparisons, 1.0)
            pairwise.append({
                "agent_a": agents[i],
                "agent_b": agents[j],
                "p_value": float(p),
                "bonferroni_p": corrected_p,
                "cohens_d": round(_cohens_d(g1, g2), 3),
                "test": "mann_whitney_u",
            })

    report.pairwise_results = pairwise
    if pairwise:
        report.p_value = min(r["p_value"] for r in pairwise)
        report.bonferroni_corrected = min(r["bonferroni_p"] for r in pairwise)

    return report
