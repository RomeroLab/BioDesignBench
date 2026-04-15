"""De novo backbone task templates.

Cells covered:
    dnk_str — 3 existing, 3 new (de novo scaffold generation)
"""

from __future__ import annotations

from biodesignbench.task_generator import TaskSpec
from biodesignbench.taxonomy import DesignApproach, MolecularSubject


def generate_specs() -> list[TaskSpec]:
    """Return TaskSpec instances for de_novo_backbone tasks."""
    DNK = DesignApproach.DE_NOVO
    STR = MolecularSubject.SCAFFOLD

    return [
        TaskSpec(
            task_id="dn_scf_001",
            task_type=DNK,
            biological_context=STR,
            difficulty="easy",
            target_name="TIM barrel (reference topology)",
            target_pdb_id="1BTM",
            target_chain="A",
            target_seq_len=247,
            description=(
                "Design a de novo TIM barrel (beta/alpha)_8 fold using RFdiffusion. "
                "The reference structure PDB 1BTM provides the target topology. "
                "TIM barrels are the most common enzyme fold and were among the first "
                "topologies successfully designed de novo by Huang et al. Science 2016."
            ),
            length_range=(200, 260),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"tm_score_to_ref": 0.85},
            source="Huang et al. Science 2016",
            doi="10.1126/science.aaf8405",
            tags=["de_novo_backbone", "TIM_barrel", "beta_alpha", "reference_topology"],
        ),
        TaskSpec(
            task_id="dn_scf_002",
            task_type=DNK,
            biological_context=STR,
            difficulty="medium",
            target_name="Rossmann fold (reference topology)",
            target_pdb_id="2FM3",
            target_chain="A",
            target_seq_len=182,
            description=(
                "Design a de novo Rossmann fold protein using backbone generation "
                "tools. PDB 2FM3 provides the reference NAD-binding Rossmann topology. "
                "The Rossmann fold is a common dinucleotide-binding domain that has been "
                "successfully designed de novo by Dou et al. Nature 2018."
            ),
            length_range=(160, 200),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"tm_score_to_ref": 0.80},
            source="Dou et al. Nature 2018",
            doi="10.1038/s41586-018-0509-0",
            tags=["de_novo_backbone", "Rossmann_fold", "dinucleotide_binding", "reference_topology"],
        ),
        TaskSpec(
            task_id="dn_scf_003",
            task_type=DNK,
            biological_context=STR,
            difficulty="hard",
            target_name="NTF2-like fold (reference topology)",
            target_pdb_id="1OAI",
            target_chain="A",
            target_seq_len=127,
            description=(
                "Design a de novo NTF2-like fold protein with a curved beta-sheet "
                "and alpha-helical core. PDB 1OAI provides the reference NTF2 topology. "
                "This complex alpha+beta fold was designed by Marcos et al. Science 2017 "
                "and represents a significant challenge in backbone generation."
            ),
            length_range=(110, 140),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"tm_score_to_ref": 0.75},
            source="Marcos et al. Science 2017",
            doi="10.1126/science.aao0285",
            tags=["de_novo_backbone", "NTF2_fold", "alpha_beta", "reference_topology"],
        ),
    ]
