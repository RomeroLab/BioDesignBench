"""Ablation framework for controlled benchmark experiments.

Defines ablation presets that systematically disable or modify agent
capabilities to measure the contribution of individual components.

Usage::

    from biodesignbench.ablations import ABLATION_PRESETS, AblationConfig

    config = ABLATION_PRESETS["no_design_tools"]
    # Apply to agent configuration before running benchmark
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes:
        name: Short identifier for the ablation.
        description: Human-readable description of what is ablated.
        disable_tools: List of tool names to disable during the run.
        max_iterations: Override for maximum iterations (None = use default).
        system_prompt_override: Optional system prompt replacement.
        inject_domain_knowledge: Whether to inject domain-specific knowledge
            into the prompt (True = add bio knowledge, False = strip it).
    """

    name: str
    description: str
    disable_tools: list[str] = field(default_factory=list)
    max_iterations: int | None = None
    system_prompt_override: str | None = None
    inject_domain_knowledge: bool | None = None


ABLATION_PRESETS: dict[str, AblationConfig] = {
    "no_design_tools": AblationConfig(
        name="no_design_tools",
        description="Disable all protein design tools (RFdiffusion, ProteinMPNN), keep only analysis tools",
        disable_tools=["design_binder", "optimize_sequence", "generate_backbone"],
    ),
    "no_literature": AblationConfig(
        name="no_literature",
        description="Disable literature/database search tools",
        disable_tools=["web_search", "pubmed_search", "uniprot_search"],
    ),
    "minimal_tools": AblationConfig(
        name="minimal_tools",
        description="Only allow structure prediction and sequence design (no diffusion)",
        disable_tools=[
            "design_binder",
            "generate_backbone",
            "analyze_interface",
            "energy_minimize",
            "suggest_hotspots",
        ],
    ),
    "max_iterations_5": AblationConfig(
        name="max_iterations_5",
        description="Limit agent to 5 iterations maximum",
        max_iterations=5,
    ),
    "max_iterations_20": AblationConfig(
        name="max_iterations_20",
        description="Allow agent up to 20 iterations",
        max_iterations=20,
    ),
    "domain_knowledge_prompt": AblationConfig(
        name="domain_knowledge_prompt",
        description="Inject domain-specific protein design knowledge into system prompt",
        inject_domain_knowledge=True,
    ),
    "no_domain_knowledge": AblationConfig(
        name="no_domain_knowledge",
        description="Strip all domain-specific knowledge from prompts",
        inject_domain_knowledge=False,
    ),
}
