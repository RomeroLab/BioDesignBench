"""Task template registry for Tier 2 expansion.

Each template module covers one DesignTaskType and defines TaskSpec
instances for the taxonomy cells it owns. Template modules are skeletons
until populated with curated PDB targets.

Usage:
    from biodesignbench.task_templates import generate_all_specs
    specs = generate_all_specs()  # list[TaskSpec]
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

from biodesignbench.task_templates import cfd, cpx, dnb, dnk, sqo

if TYPE_CHECKING:
    from biodesignbench.task_generator import TaskSpec

#: Ordered list of all template modules.
TEMPLATE_MODULES: list[types.ModuleType] = [dnb, sqo, dnk, cpx, cfd]


def generate_all_specs() -> list[TaskSpec]:
    """Collect TaskSpecs from all template modules.

    Returns:
        Combined list of TaskSpec instances from all registered modules.
    """
    specs: list[TaskSpec] = []
    for mod in TEMPLATE_MODULES:
        specs.extend(mod.generate_specs())
    return specs
