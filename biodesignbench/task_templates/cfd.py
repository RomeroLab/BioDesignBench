"""Conformational design task templates.

Cells covered:
    cfd_enz — 0 existing, 6 new (enzyme conformational design)
    cfd_sig — 1 existing, 5 new (signaling switches/sensors)
    cfd_str — 0 existing, 2 new (conformational structural switches)
    cfd_flu — 1 existing, 5 new (fluorescent protein switches)
"""

from __future__ import annotations

from biodesignbench.task_generator import TaskSpec
from biodesignbench.taxonomy import DesignApproach, MolecularSubject


def generate_specs() -> list[TaskSpec]:
    """Return TaskSpec instances for conformational_design tasks."""
    CFD = DesignApproach.DE_NOVO
    ENZ = MolecularSubject.ENZYME
    SIG = MolecularSubject.BINDER
    STR = MolecularSubject.SCAFFOLD
    FLU = MolecularSubject.FLUORESCENT_PROTEIN

    return [
        # ----- cfd_enz: Enzyme conformational design -----
        TaskSpec(
            task_id="dn_enz_006",
            task_type=CFD,
            biological_context=ENZ,
            difficulty="easy",
            target_name="Adenylate kinase (open/closed)",
            target_pdb_id="4AKE",
            target_chain="A",
            target_seq_len=214,
            binding_site_residues=[13, 14, 15, 16, 17, 30, 31, 32, 119, 120, 121, 170, 171],
            description=(
                "Design conformational variants of E. coli adenylate kinase that "
                "shift the open-closed equilibrium. PDB 4AKE shows the open state "
                "with the LID and NMP domains extended. This is a classic model for "
                "studying conformational transitions in enzymes with large domain motions."
            ),
            length_range=(200, 220),
            gt_kd_nM=None,
            gt_tm_C=52.0,
            gt_expression=True,
            gt_additional={"kcat_per_s": 312.0},
            source="Muller et al. Proteins 1996",
            doi="10.1002/(SICI)1097-0134(199606)25:2<I>",
            tags=["conformational", "adenylate_kinase", "domain_motion", "open_closed"],
        ),
        TaskSpec(
            task_id="dn_enz_007",
            task_type=CFD,
            biological_context=ENZ,
            difficulty="easy",
            target_name="Maltose-binding protein (hinge motion)",
            target_pdb_id="1OMP",
            target_chain="A",
            target_seq_len=370,
            binding_site_residues=[11, 12, 13, 14, 15, 65, 66, 111, 152, 155, 230, 235, 340],
            description=(
                "Design conformational variants of maltose-binding protein (MBP) "
                "that modulate the hinge-bending motion between open and closed states. "
                "PDB 1OMP shows the maltose-bound closed conformation. MBP is widely "
                "used as a biosensor scaffold due to its large conformational change."
            ),
            length_range=(360, 380),
            gt_kd_nM=None,
            gt_tm_C=62.7,
            gt_expression=True,
            gt_additional={"maltose_kd_uM": 1.2},
            source="Sharff et al. Biochemistry 1992",
            doi="10.1021/bi00158a015",
            tags=["conformational", "MBP", "hinge_motion", "biosensor"],
        ),
        TaskSpec(
            task_id="dn_enz_008",
            task_type=CFD,
            biological_context=ENZ,
            difficulty="medium",
            target_name="Dihydrofolate reductase (DHFR)",
            target_pdb_id="1RX2",
            target_chain="A",
            target_seq_len=159,
            binding_site_residues=[5, 7, 27, 28, 31, 32, 52, 54, 57, 94, 100, 113],
            description=(
                "Design conformational variants of E. coli DHFR that alter the "
                "catalytic cycle dynamics. PDB 1RX2 shows DHFR with folate bound. "
                "DHFR is a model enzyme for studying how conformational dynamics "
                "couples to catalysis across multiple intermediate states."
            ),
            length_range=(150, 170),
            gt_kd_nM=None,
            gt_tm_C=52.4,
            gt_expression=True,
            gt_additional={"kcat_per_s": 12.0},
            source="Sawaya & Kraut Biochemistry 1997",
            doi="10.1021/bi962927q",
            tags=["conformational", "DHFR", "catalytic_cycle", "dynamics"],
        ),
        TaskSpec(
            task_id="dn_enz_009",
            task_type=CFD,
            biological_context=ENZ,
            difficulty="medium",
            target_name="Citrate synthase (open/closed)",
            target_pdb_id="1CTS",
            target_chain="A",
            target_seq_len=437,
            binding_site_residues=[274, 275, 314, 318, 319, 320, 338, 375, 376, 380, 381, 421],
            description=(
                "Design conformational variants of pig citrate synthase that "
                "modulate the large domain rotation between open and closed states. "
                "PDB 1CTS shows the closed conformation with bound substrates. "
                "The 18-degree domain closure is one of the largest known enzyme "
                "conformational changes."
            ),
            length_range=(430, 450),
            gt_kd_nM=None,
            gt_tm_C=55.0,
            gt_expression=True,
            gt_additional={"kcat_per_s": 58.0},
            source="Remington et al. JMB 1982",
            doi="10.1016/0022-2836(82)90266-8",
            tags=["conformational", "citrate_synthase", "domain_rotation", "enzyme"],
        ),
        TaskSpec(
            task_id="dn_enz_010",
            task_type=CFD,
            biological_context=ENZ,
            difficulty="hard",
            target_name="Calcineurin (autoinhibited + active)",
            target_pdb_id="1AUI",
            target_chain="A",
            target_seq_len=390,
            binding_site_residues=[
                90, 92, 118, 119, 120, 122, 150, 152, 199, 243, 280, 282, 310, 311, 314,
            ],
            description=(
                "Design conformational switches based on calcineurin that toggle "
                "between autoinhibited and active states. PDB 1AUI shows the "
                "calcineurin A/B heterodimer. The autoinhibitory domain blocks the "
                "active site and is displaced by calmodulin binding. This mechanism "
                "can be exploited for designed allosteric control."
            ),
            length_range=(100, 150),
            gt_kd_nM=None,
            gt_tm_C=60.0,
            gt_expression=True,
            gt_additional={"kcat_per_s": 0.8},
            source="Kissinger et al. Nature 1995",
            doi="10.1038/378641a0",
            tags=["conformational", "calcineurin", "autoinhibition", "phosphatase"],
        ),
        TaskSpec(
            task_id="dn_enz_011",
            task_type=CFD,
            biological_context=ENZ,
            difficulty="hard",
            target_name="Allosteric glucokinase (GCK)",
            target_pdb_id="1V4S",
            target_chain="A",
            target_seq_len=465,
            binding_site_residues=[
                151, 152, 168, 169, 204, 205, 206, 225, 227, 230, 254, 256, 287, 290,
            ],
            description=(
                "Design conformational variants of human glucokinase (GCK) that "
                "modulate its allosteric activation. PDB 1V4S shows the active "
                "conformation. GCK is a key glucose sensor in pancreatic beta cells; "
                "activating mutations cause hyperinsulinemia while inactivating ones "
                "cause MODY2 diabetes. Engineer the allosteric mechanism."
            ),
            length_range=(100, 150),
            gt_kd_nM=None,
            gt_tm_C=52.0,
            gt_expression=True,
            gt_additional={"glucose_km_mM": 8.0},
            source="Kamata et al. Structure 2004",
            doi="10.1016/j.str.2004.02.005",
            tags=["conformational", "glucokinase", "allosteric", "diabetes"],
        ),
        # ----- cfd_sig: Signaling switches/sensors -----
        TaskSpec(
            task_id="dn_bnd_012",
            task_type=CFD,
            biological_context=SIG,
            difficulty="easy",
            target_name="LOCKR degradation switch",
            target_pdb_id="6OB5",
            target_chain="A",
            target_seq_len=365,
            description=(
                "Design variants of the LOCKR (Latching Orthogonal Cage/Key pRotein) "
                "degradation switch with altered switching dynamics. PDB 6OB5 shows "
                "the closed LOCKR cage. The LOCKR system by Langan et al. uses a key "
                "peptide to unlatch a caged degron for targeted protein degradation."
            ),
            length_range=(100, 150),
            gt_kd_nM=100.0,
            gt_tm_C=70.0,
            gt_expression=True,
            source="Langan et al. Nature 2019",
            doi="10.1038/s41586-019-1432-8",
            tags=["conformational", "LOCKR", "switch", "degradation", "designed"],
        ),
        TaskSpec(
            task_id="dn_bnd_013",
            task_type=CFD,
            biological_context=SIG,
            difficulty="medium",
            target_name="LATCH coiled-coil switch",
            target_pdb_id="6XN6",
            target_chain="A",
            target_seq_len=256,
            description=(
                "Design variants of the LATCH coiled-coil switch with tunable "
                "activation thresholds. PDB 6XN6 shows the designed coiled-coil "
                "switch mechanism by Lajoie et al. The system uses competitive "
                "binding to create a bistable protein switch for synthetic biology."
            ),
            length_range=(80, 130),
            gt_kd_nM=50.0,
            gt_tm_C=75.0,
            gt_expression=True,
            source="Lajoie et al. PNAS 2020",
            doi="10.1073/pnas.2000557117",
            tags=["conformational", "LATCH", "coiled_coil", "switch", "synthetic_biology"],
        ),
        TaskSpec(
            task_id="dn_bnd_014",
            task_type=CFD,
            biological_context=SIG,
            difficulty="medium",
            target_name="Cage-key sensor (de novo)",
            target_pdb_id="7CBC",
            target_chain="A",
            target_seq_len=220,
            binding_site_residues=[40, 43, 44, 47, 80, 83, 84, 87, 120, 123, 124, 127],
            description=(
                "Design cage-key protein sensor variants with novel analyte "
                "specificity. PDB 7CBC shows the designed cage-key system from "
                "Quijano-Rubio et al. The cage sequesters a functional peptide "
                "that is released upon key binding. Redesign the key-cage interface "
                "for orthogonal sensing applications."
            ),
            length_range=(80, 120),
            gt_kd_nM=20.0,
            gt_tm_C=72.0,
            gt_expression=True,
            source="Quijano-Rubio et al. Nature 2021",
            doi="10.1038/s41586-021-03258-z",
            tags=["conformational", "cage_key", "sensor", "de_novo", "biosensor"],
        ),
        TaskSpec(
            task_id="dn_bnd_015",
            task_type=CFD,
            biological_context=SIG,
            difficulty="hard",
            target_name="Designed allosteric protein switch",
            target_pdb_id="5CWA",
            target_chain="A",
            target_seq_len=198,
            description=(
                "Design allosteric protein switches based on the SWTCH system "
                "from Ha et al. PDB 5CWA shows a computationally designed protein "
                "that undergoes a large conformational change upon effector binding. "
                "Create variants with distinct input-output relationships for use "
                "as modular signaling components."
            ),
            length_range=(80, 120),
            gt_kd_nM=200.0,
            gt_tm_C=65.0,
            gt_expression=True,
            source="Ha et al. PNAS 2015",
            doi="10.1073/pnas.1507847112",
            tags=["conformational", "allosteric", "switch", "designed", "modular"],
        ),
        TaskSpec(
            task_id="dn_bnd_016",
            task_type=CFD,
            biological_context=SIG,
            difficulty="hard",
            target_name="LOCKR-Caspase drug-responsive circuit",
            target_pdb_id="7S1A",
            target_chain="A",
            target_seq_len=340,
            description=(
                "Design a drug-responsive LOCKR-Caspase circuit with programmable "
                "cell death activation. PDB 7S1A shows the LOCKR-based caspase "
                "activation system from Ng et al. The circuit responds to a small "
                "molecule to trigger apoptosis. Redesign for altered drug sensitivity "
                "and tighter OFF-state control."
            ),
            length_range=(80, 140),
            gt_kd_nM=30.0,
            gt_tm_C=68.0,
            gt_expression=True,
            source="Ng et al. Science 2024",
            doi="10.1126/science.adj0863",
            tags=["conformational", "LOCKR", "caspase", "circuit", "drug_responsive"],
        ),
        # ----- cfd_str: Conformational structural switches -----
        TaskSpec(
            task_id="dn_scf_010",
            task_type=CFD,
            biological_context=STR,
            difficulty="medium",
            target_name="Fold-switching protein RfaH",
            target_pdb_id="5OND",
            target_chain="A",
            target_seq_len=162,
            description=(
                "Design conformational variants of the fold-switching protein RfaH. "
                "PDB 5OND shows the alpha-helical C-terminal domain that refolds into "
                "a beta-barrel upon release from the N-terminal domain. RfaH is one of "
                "the few known fold-switching proteins and represents a key challenge "
                "in conformational design."
            ),
            length_range=(80, 120),
            gt_tm_C=55.0,
            gt_expression=True,
            gt_additional={"fold_switch_kd_uM": 5.0},
            source="Burmann et al. Cell 2012",
            doi="10.1016/j.cell.2012.09.037",
            tags=["conformational", "fold_switch", "RfaH", "metamorphic"],
        ),
        TaskSpec(
            task_id="dn_scf_011",
            task_type=CFD,
            biological_context=STR,
            difficulty="hard",
            target_name="Designed conformational switch (Janus protein)",
            target_pdb_id="6EI4",
            target_chain="A",
            target_seq_len=95,
            description=(
                "Design a Janus-type conformational switch protein that populates "
                "two distinct folds (alpha-helix bundle and beta-sheet). PDB 6EI4 "
                "shows a computationally designed fold-switching protein. This is "
                "among the hardest challenges in protein design: engineering a single "
                "sequence that encodes two distinct stable folds."
            ),
            length_range=(85, 110),
            gt_tm_C=50.0,
            gt_expression=True,
            gt_additional={"fold_ratio_alpha_beta": 1.0},
            source="Ambroggio & Bhatt-Kuhlman Structure 2006",
            doi="10.1016/j.str.2006.03.016",
            tags=["conformational", "Janus", "fold_switch", "two_folds", "designed"],
        ),
        # ----- cfd_flu: Fluorescent protein switches -----
        TaskSpec(
            task_id="dn_fp_001",
            task_type=CFD,
            biological_context=FLU,
            difficulty="easy",
            target_name="Split GFP (GFP1-10 + GFP11)",
            target_pdb_id="4KF5",
            target_chain="A",
            target_seq_len=229,
            description=(
                "Design split GFP variants with improved complementation "
                "efficiency and brightness. PDB 4KF5 shows the self-complementing "
                "GFP1-10/GFP11 system by Cabantous et al. Optimize the split "
                "interface for faster assembly kinetics while maintaining the "
                "conditional fluorescence readout."
            ),
            length_range=(200, 240),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.60, "complementation_half_life_min": 30.0},
            source="Cabantous et al. Sci Rep 2013",
            doi="10.1038/srep02854",
            tags=["conformational", "split_GFP", "complementation", "biosensor"],
        ),
        TaskSpec(
            task_id="dn_fp_002",
            task_type=CFD,
            biological_context=FLU,
            difficulty="easy",
            target_name="cpGFP (circularly permuted GFP)",
            target_pdb_id="3EVP",
            target_chain="A",
            target_seq_len=241,
            description=(
                "Design circularly permuted GFP (cpGFP) variants with improved "
                "dynamic range for use in genetically encoded indicators. PDB 3EVP "
                "shows a cpGFP structure. cpGFP is the core of GCaMP and other "
                "biosensors; optimizing its conformational sensitivity to insert "
                "domain motion is critical for sensor performance."
            ),
            length_range=(230, 250),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.45, "dynamic_range_fold": 8.0},
            source="Nakai et al. Nat Biotechnol 2001",
            doi="10.1038/84397",
            tags=["conformational", "cpGFP", "circular_permutation", "biosensor"],
        ),
        TaskSpec(
            task_id="dn_fp_003",
            task_type=CFD,
            biological_context=FLU,
            difficulty="medium",
            target_name="GCaMP6s calcium indicator",
            target_pdb_id="3WLC",
            target_chain="A",
            target_seq_len=440,
            binding_site_residues=[
                20, 31, 56, 57, 58, 59, 60, 93, 95, 97, 100, 104, 105, 136, 141,
            ],
            description=(
                "Design GCaMP6s variants with improved calcium sensitivity and "
                "kinetics for in vivo neural imaging. PDB 3WLC shows the GCaMP "
                "structure with cpGFP fused to calmodulin and M13 peptide. "
                "Optimize the allosteric coupling between calcium binding and "
                "fluorescence change."
            ),
            length_range=(430, 460),
            gt_kd_nM=144.0,
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"dynamic_range_fold": 63.0},
            source="Chen et al. Nature 2013",
            doi="10.1038/nature12354",
            tags=["conformational", "GCaMP6s", "calcium", "indicator", "neuroscience"],
        ),
        TaskSpec(
            task_id="dn_fp_004",
            task_type=CFD,
            biological_context=FLU,
            difficulty="medium",
            target_name="iLOV (LOV-domain fluorescent protein)",
            target_pdb_id="4EES",
            target_chain="A",
            target_seq_len=110,
            description=(
                "Design iLOV variants with enhanced fluorescence and oxygen-"
                "independent maturation. PDB 4EES shows the iLOV domain derived "
                "from Arabidopsis phototropin 2. iLOV uses FMN as chromophore "
                "instead of forming a chromophore from the polypeptide, enabling "
                "fluorescence under anaerobic conditions."
            ),
            length_range=(100, 120),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.44, "extinction_coeff": 13900.0},
            source="Chapman et al. JACS 2008",
            doi="10.1021/ja801728a",
            tags=["conformational", "iLOV", "LOV_domain", "FMN", "oxygen_independent"],
        ),
        TaskSpec(
            task_id="dn_fp_005",
            task_type=CFD,
            biological_context=FLU,
            difficulty="hard",
            target_name="miRFP670 near-infrared FP",
            target_pdb_id="5VIQ",
            target_chain="A",
            target_seq_len=315,
            description=(
                "Design miRFP670 variants with improved brightness and "
                "photostability for deep-tissue imaging. PDB 5VIQ shows the "
                "engineered bacteriophytochrome-derived near-infrared fluorescent "
                "protein. Optimize the biliverdin chromophore environment to enhance "
                "quantum yield while maintaining the far-red/NIR emission spectrum."
            ),
            length_range=(300, 330),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.12, "extinction_coeff": 87400.0},
            source="Shcherbakova et al. Nat Commun 2016",
            doi="10.1038/ncomms12405",
            tags=["conformational", "miRFP670", "near_infrared", "biliverdin", "deep_tissue"],
        ),
    ]
