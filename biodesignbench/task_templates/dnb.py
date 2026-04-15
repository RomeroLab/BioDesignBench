"""De novo binder task templates.

Cells covered:
    dnb_ab  — 5 restored (antibody/nanobody binders)
    dnb_enz — 0 existing, 3 new (enzyme inhibitor binders)
    dnb_sig — 5 existing, 3 new (signaling binders, +2 RFdiffusion buffer)
    dnb_thr — 6 existing, 0 new (therapeutic binders)
"""

from __future__ import annotations

from biodesignbench.task_generator import TaskSpec
from biodesignbench.taxonomy import DesignApproach, MolecularSubject


def generate_specs() -> list[TaskSpec]:
    """Return TaskSpec instances for de_novo_binder tasks."""
    DNB = DesignApproach.DE_NOVO
    AB = MolecularSubject.ANTIBODY
    ENZ = MolecularSubject.ENZYME
    SIG = MolecularSubject.BINDER

    return [
        # ----- dnb_ab: De novo nanobody binder design -----
        TaskSpec(
            task_id="dn_ab_001",
            task_type=DNB,
            biological_context=AB,
            difficulty="easy",
            target_name="Anti-lysozyme nanobody (VHH)",
            target_pdb_id="1ZVH",
            target_chain="A",
            target_seq_len=129,
            binding_site_residues=[33, 50, 52, 53, 54, 55, 56, 98, 100, 101, 102, 103],
            description=(
                "Design a de novo nanobody (VHH) that binds hen egg-white lysozyme "
                "at the active-site cleft. The reference structure (PDB 1ZVH) shows "
                "cAbLys3 engaging the catalytic residues via a long CDR3 loop. "
                "Generate binders with sub-micromolar predicted affinity."
            ),
            length_range=(110, 140),
            gt_kd_nM=16.0,
            gt_tm_C=68.0,
            gt_expression=True,
            source="Desmyter et al. JBC 2001",
            doi="10.1074/jbc.M105056200",
            tags=["nanobody", "lysozyme", "VHH", "CDR3"],
        ),
        TaskSpec(
            task_id="dn_ab_002",
            task_type=DNB,
            biological_context=AB,
            difficulty="medium",
            target_name="Anti-GFP nanobody (enhancer)",
            target_pdb_id="3OGO",
            target_chain="A",
            target_seq_len=238,
            binding_site_residues=[142, 143, 144, 145, 146, 165, 167, 168, 203, 205, 206],
            description=(
                "Design a de novo nanobody that binds GFP and enhances its "
                "fluorescence (GFP enhancer). The reference nanobody (PDB 3OGO) "
                "contacts GFP near the chromophore to rigidify its local environment. "
                "Aim for tight binding without quenching fluorescence."
            ),
            length_range=(110, 140),
            gt_kd_nM=0.59,
            gt_tm_C=72.0,
            gt_expression=True,
            source="Kirchhofer et al. Nat Struct Mol Biol 2010",
            doi="10.1038/nsmb.1727",
            tags=["nanobody", "GFP", "enhancer", "fluorescence"],
        ),
        TaskSpec(
            task_id="dn_ab_003",
            task_type=DNB,
            biological_context=AB,
            difficulty="medium",
            target_name="Anti-EGFR nanobody 7D12",
            target_pdb_id="4KRL",
            target_chain="A",
            target_seq_len=645,
            binding_site_residues=[333, 334, 335, 357, 358, 359, 405, 406, 407, 408, 412],
            description=(
                "Design a de novo nanobody targeting the extracellular domain III "
                "of human EGFR. The reference nanobody 7D12 (PDB 4KRL) blocks EGF "
                "binding and inhibits receptor dimerization. Designs should compete "
                "with the natural EGF binding site on domain III."
            ),
            length_range=(110, 140),
            gt_kd_nM=5.1,
            gt_tm_C=65.0,
            gt_expression=True,
            source="Schmitz et al. Structure 2013",
            doi="10.1016/j.str.2013.05.013",
            tags=["nanobody", "EGFR", "cancer", "receptor"],
        ),
        TaskSpec(
            task_id="dn_ab_004",
            task_type=DNB,
            biological_context=AB,
            difficulty="hard",
            target_name="Anti-SARS-CoV-2 RBD nanobody Nb6",
            target_pdb_id="7KKJ",
            target_chain="A",
            target_seq_len=201,
            binding_site_residues=[
                417, 446, 449, 453, 455, 456, 475, 486, 487, 489, 493, 496, 498, 500, 501, 505,
            ],
            description=(
                "Design a de novo nanobody that neutralizes SARS-CoV-2 by binding "
                "the receptor-binding domain (RBD) of the Spike protein. Reference "
                "Nb6 (PDB 7KKJ) blocks ACE2 engagement. Designs must achieve tight "
                "binding (<100 nM) and maintain stability against viral evolution."
            ),
            length_range=(110, 140),
            gt_kd_nM=2.0,
            gt_tm_C=66.0,
            gt_expression=True,
            source="Schoof et al. Science 2020",
            doi="10.1126/science.abe3255",
            tags=["nanobody", "SARS-CoV-2", "RBD", "neutralization"],
        ),
        TaskSpec(
            task_id="dn_ab_005",
            task_type=DNB,
            biological_context=AB,
            difficulty="hard",
            target_name="Anti-CD38 nanobody",
            target_pdb_id="4CMH",
            target_chain="A",
            target_seq_len=300,
            binding_site_residues=[68, 69, 70, 82, 83, 116, 117, 226, 237, 271, 272, 274],
            description=(
                "Design a de novo nanobody targeting human CD38, a validated "
                "target in multiple myeloma therapy. PDB 4CMH shows a nanobody "
                "engaging the catalytic domain of CD38. Designs should achieve "
                "sub-micromolar affinity and block enzymatic activity."
            ),
            length_range=(110, 140),
            gt_kd_nM=10.0,
            gt_tm_C=70.0,
            gt_expression=True,
            source="Fumey et al. MAbs 2017",
            doi="10.1080/19420862.2017.1294873",
            tags=["nanobody", "CD38", "multiple_myeloma", "therapeutic"],
        ),
        # ----- dnb_enz: Enzyme inhibitor binder design -----
        TaskSpec(
            task_id="dn_enz_001",
            task_type=DNB,
            biological_context=ENZ,
            difficulty="easy",
            target_name="Trypsin protease inhibitor binder",
            target_pdb_id="1PPE",
            target_chain="E",
            target_seq_len=223,
            binding_site_residues=[189, 190, 191, 192, 195, 213, 214, 215, 216, 217, 226],
            description=(
                "Design a de novo protein binder that inhibits bovine trypsin by "
                "blocking the active site cleft. PDB 1PPE shows the trypsin-PSTI "
                "complex. The binder should mimic the canonical serine protease "
                "inhibitor loop geometry for tight active-site engagement."
            ),
            length_range=(60, 100),
            gt_kd_nM=20.0,
            gt_tm_C=65.0,
            gt_expression=True,
            source="Marquart et al. Acta Cryst 1983",
            doi="10.1107/S0567740883003986",
            tags=["binder", "trypsin", "protease", "inhibitor", "RFdiffusion"],
        ),
        TaskSpec(
            task_id="dn_enz_002",
            task_type=DNB,
            biological_context=ENZ,
            difficulty="medium",
            target_name="CDK2 kinase inhibitor binder",
            target_pdb_id="1FIN",
            target_chain="A",
            target_seq_len=298,
            binding_site_residues=[10, 11, 12, 13, 14, 80, 81, 82, 83, 84, 85, 86, 145, 146],
            description=(
                "Design a de novo protein binder targeting the ATP-binding site "
                "of cyclin-dependent kinase 2 (CDK2). PDB 1FIN shows the CDK2-cyclin A "
                "complex. The binder should compete with ATP and cyclin binding "
                "to inhibit CDK2 kinase activity for cancer therapy applications."
            ),
            length_range=(60, 100),
            gt_kd_nM=50.0,
            gt_tm_C=60.0,
            gt_expression=True,
            source="Jeffrey et al. Nature 1995",
            doi="10.1038/376313a0",
            tags=["binder", "CDK2", "kinase", "cancer", "RFdiffusion"],
        ),
        TaskSpec(
            task_id="dn_enz_003",
            task_type=DNB,
            biological_context=ENZ,
            difficulty="hard",
            target_name="ACE2 protease binder",
            target_pdb_id="1R42",
            target_chain="A",
            target_seq_len=597,
            binding_site_residues=[345, 346, 371, 374, 375, 378, 393, 505, 506, 509, 510],
            description=(
                "Design a de novo protein binder targeting the catalytic domain "
                "of angiotensin-converting enzyme 2 (ACE2). PDB 1R42 shows ACE2 "
                "in the closed conformation. The binder should block the active "
                "site zinc-dependent peptidase domain and demonstrate tight binding."
            ),
            length_range=(60, 100),
            gt_kd_nM=100.0,
            gt_tm_C=58.0,
            gt_expression=True,
            source="Towler et al. JBC 2004",
            doi="10.1074/jbc.M313446200",
            tags=["binder", "ACE2", "protease", "RFdiffusion", "enzyme_inhibitor"],
        ),
        # ----- dnb_sig: buffer (5 existing + 3 new, +2 RFdiffusion) -----
        TaskSpec(
            task_id="dn_bnd_009",
            task_type=DNB,
            biological_context=SIG,
            difficulty="medium",
            target_name="EphB2 receptor ectodomain",
            target_pdb_id="3ETP",
            target_chain="A",
            target_seq_len=210,
            binding_site_residues=[38, 39, 64, 66, 68, 70, 106, 108, 110, 112],
            description=(
                "Design a de novo protein binder targeting the EphB2 receptor "
                "ectodomain ligand-binding domain. EphB2 is a receptor tyrosine "
                "kinase involved in axon guidance and vasculogenesis. The PDB 3ETP "
                "structure shows the ephrin-binding channel that designs should target."
            ),
            length_range=(60, 100),
            gt_kd_nM=50.0,
            gt_tm_C=62.0,
            gt_expression=True,
            source="Himanen et al. PNAS 2010",
            doi="10.1073/pnas.0911536107",
            tags=["binder", "EphB2", "receptor_tyrosine_kinase", "signaling"],
        ),
        TaskSpec(
            task_id="dn_bnd_010",
            task_type=DNB,
            biological_context=SIG,
            difficulty="medium",
            target_name="FGFR2 receptor binder (RFdiffusion)",
            target_pdb_id="1EV2",
            target_chain="A",
            target_seq_len=276,
            binding_site_residues=[160, 162, 164, 167, 170, 205, 207, 210, 252, 255, 258],
            description=(
                "Design a de novo protein binder targeting the D2-D3 domains of "
                "FGFR2 using RFdiffusion backbone generation. PDB 1EV2 shows the "
                "FGFR2-FGF2 complex. The binder should compete with FGF2 for "
                "receptor engagement to modulate FGF signaling."
            ),
            length_range=(60, 100),
            gt_kd_nM=30.0,
            gt_tm_C=64.0,
            gt_expression=True,
            source="Plotnikov et al. Cell 2000",
            doi="10.1016/S0092-8674(00)80851-X",
            tags=["binder", "FGFR2", "RFdiffusion", "signaling"],
        ),
        TaskSpec(
            task_id="dn_bnd_011",
            task_type=DNB,
            biological_context=SIG,
            difficulty="hard",
            target_name="PD-1 receptor binder (RFdiffusion)",
            target_pdb_id="4ZQK",
            target_chain="A",
            target_seq_len=145,
            binding_site_residues=[64, 66, 68, 69, 73, 74, 75, 78, 126, 128, 130, 131, 132],
            description=(
                "Design a de novo protein binder targeting the PD-1 receptor "
                "using RFdiffusion backbone generation. PDB 4ZQK shows the PD-1 "
                "extracellular domain. The binder should block PD-L1 engagement "
                "for checkpoint immunotherapy applications."
            ),
            length_range=(60, 100),
            gt_kd_nM=15.0,
            gt_tm_C=62.0,
            gt_expression=True,
            source="Zak et al. Structure 2015",
            doi="10.1016/j.str.2015.09.010",
            tags=["binder", "PD-1", "RFdiffusion", "immunotherapy", "checkpoint"],
        ),
    ]
