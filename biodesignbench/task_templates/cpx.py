"""Complex engineering task templates.

Cells covered:
    cpx_enz — 0 existing, 2 new (enzyme complex engineering)
    cpx_sig — 0 existing, 6 new (signaling complex engineering)
    cpx_str — 2 existing, 6 new (structural protein assemblies, +2 RFdiffusion buffer)
"""

from __future__ import annotations

from biodesignbench.task_generator import TaskSpec
from biodesignbench.taxonomy import DesignApproach, MolecularSubject


def generate_specs() -> list[TaskSpec]:
    """Return TaskSpec instances for complex_engineering tasks."""
    CPX = DesignApproach.DE_NOVO
    ENZ = MolecularSubject.ENZYME
    SIG = MolecularSubject.BINDER
    STR = MolecularSubject.SCAFFOLD

    return [
        # ----- cpx_enz: Enzyme complex engineering -----
        TaskSpec(
            task_id="dn_enz_004",
            task_type=CPX,
            biological_context=ENZ,
            difficulty="medium",
            target_name="Aldolase-TIM barrel metabolon",
            target_pdb_id="1ADO",
            target_chain="A",
            target_seq_len=363,
            binding_site_residues=[33, 34, 35, 80, 107, 108, 109, 146, 147, 148, 179, 226, 229],
            description=(
                "Engineer a designed enzyme complex (metabolon) between fructose-"
                "1,6-bisphosphate aldolase and a TIM-barrel enzyme. PDB 1ADO shows "
                "rabbit muscle aldolase. Design a protein interface that enables "
                "substrate channeling between the two active sites for improved "
                "metabolic flux."
            ),
            length_range=(80, 120),
            gt_kd_nM=100.0,
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"substrate_channeling_pct": 60.0},
            source="Blom & Sygusch JMB 1997",
            doi="10.1006/jmbi.1996.0857",
            tags=["complex", "aldolase", "metabolon", "substrate_channeling"],
        ),
        TaskSpec(
            task_id="dn_enz_005",
            task_type=CPX,
            biological_context=ENZ,
            difficulty="hard",
            target_name="Glutamate dehydrogenase hexamer interface",
            target_pdb_id="1NR7",
            target_chain="A",
            target_seq_len=501,
            binding_site_residues=[82, 83, 84, 90, 91, 245, 250, 260, 262, 393, 395, 446],
            description=(
                "Engineer the hexameric interface of bovine glutamate dehydrogenase "
                "(GDH) to create a controllable enzyme complex. PDB 1NR7 shows the "
                "GDH hexamer. Design interface mutations that modulate allosteric "
                "regulation and oligomerization for metabolic engineering."
            ),
            length_range=(80, 120),
            gt_kd_nM=50.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Smith et al. JMB 2001",
            doi="10.1016/S0022-2836(01)00889-0",
            tags=["complex", "GDH", "hexamer", "allosteric", "enzyme"],
        ),
        # ----- cpx_sig: Signaling complex engineering -----
        TaskSpec(
            task_id="dn_bnd_001",
            task_type=CPX,
            biological_context=SIG,
            difficulty="easy",
            target_name="ERK2-MKP3 phosphatase complex",
            target_pdb_id="1HZM",
            target_chain="A",
            target_seq_len=360,
            binding_site_residues=[160, 161, 162, 163, 164, 316, 318, 319, 320, 321],
            description=(
                "Engineer the protein-protein interface between ERK2 (MAPK1) and "
                "its phosphatase MKP3. PDB 1HZM shows ERK2 in the active state. "
                "Design interface mutations that modulate the ERK2-MKP3 interaction "
                "to create a controllable MAPK signaling pathway element."
            ),
            length_range=(80, 120),
            gt_kd_nM=400.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Camps et al. Science 2000",
            doi="10.1126/science.286.5449.2514",
            tags=["complex", "MAPK", "ERK2", "phosphatase", "signaling"],
        ),
        TaskSpec(
            task_id="dn_bnd_002",
            task_type=CPX,
            biological_context=SIG,
            difficulty="easy",
            target_name="14-3-3 + BAD peptide complex",
            target_pdb_id="1SA0",
            target_chain="A",
            target_seq_len=232,
            binding_site_residues=[56, 57, 58, 125, 126, 127, 128, 129, 172, 173, 175, 180],
            description=(
                "Engineer the 14-3-3 protein interface to selectively bind the "
                "BAD phosphopeptide with enhanced specificity. PDB 1SA0 shows the "
                "14-3-3/BAD complex. 14-3-3 proteins are hub proteins in apoptosis "
                "signaling; modulating their binding specificity is therapeutically relevant."
            ),
            length_range=(60, 100),
            gt_kd_nM=200.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Ottmann et al. Structure 2007",
            doi="10.1016/j.str.2007.01.011",
            tags=["complex", "14-3-3", "BAD", "apoptosis", "phosphopeptide"],
        ),
        TaskSpec(
            task_id="dn_bnd_003",
            task_type=CPX,
            biological_context=SIG,
            difficulty="medium",
            target_name="TNF-alpha homotrimer interface",
            target_pdb_id="1TNF",
            target_chain="A",
            target_seq_len=157,
            binding_site_residues=[29, 31, 33, 84, 85, 86, 87, 143, 144, 145, 146, 147],
            description=(
                "Engineer the TNF-alpha homotrimer interface to create variants "
                "with altered oligomerization properties. PDB 1TNF shows the "
                "canonical TNF trimer. Design mutations that modulate trimerization "
                "affinity to create dominant-negative TNF variants for therapeutic use."
            ),
            length_range=(60, 100),
            gt_kd_nM=5.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Eck & Sprang Science 1989",
            doi="10.1016/S0021-9258(17)36684-3",
            tags=["complex", "TNF", "homotrimer", "cytokine", "interface"],
        ),
        TaskSpec(
            task_id="dn_bnd_004",
            task_type=CPX,
            biological_context=SIG,
            difficulty="medium",
            target_name="Calmodulin-CaMKII complex",
            target_pdb_id="1CDM",
            target_chain="A",
            target_seq_len=148,
            binding_site_residues=[19, 22, 25, 26, 29, 36, 72, 76, 79, 80, 83, 105, 109, 112, 113],
            description=(
                "Engineer the calmodulin (CaM) interface for altered target "
                "selectivity toward CaM-kinase II (CaMKII). PDB 1CDM shows calcium-"
                "loaded calmodulin. Design CaM variants with enhanced specificity "
                "for CaMKII over other CaM-binding partners in calcium signaling."
            ),
            length_range=(60, 100),
            gt_kd_nM=1.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Meador et al. Science 1993",
            doi="10.1126/science.8378350",
            tags=["complex", "calmodulin", "CaMKII", "calcium", "signaling"],
        ),
        TaskSpec(
            task_id="dn_bnd_005",
            task_type=CPX,
            biological_context=SIG,
            difficulty="hard",
            target_name="KRAS G12D-effector complex",
            target_pdb_id="4DSO",
            target_chain="A",
            target_seq_len=169,
            binding_site_residues=[12, 13, 25, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40],
            description=(
                "Engineer a protein that selectively binds the oncogenic KRAS G12D "
                "mutant over wild-type KRAS. PDB 4DSO shows the KRAS-effector "
                "complex. KRAS G12D is a key driver in pancreatic cancer; selective "
                "inhibition requires distinguishing a single amino acid change at the "
                "effector-binding interface."
            ),
            length_range=(60, 120),
            gt_kd_nM=50.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Hunter et al. Mol Cell 2015",
            doi="10.1016/j.molcel.2014.10.025",
            tags=["complex", "KRAS", "G12D", "oncogene", "selective_binding"],
        ),
        TaskSpec(
            task_id="dn_bnd_006",
            task_type=CPX,
            biological_context=SIG,
            difficulty="hard",
            target_name="Wnt3a-Frizzled CRD complex",
            target_pdb_id="6AHY",
            target_chain="A",
            target_seq_len=352,
            binding_site_residues=[77, 78, 79, 80, 130, 131, 132, 133, 187, 209, 210, 211, 237, 238],
            description=(
                "Engineer the Wnt3a-Frizzled8 CRD interface to create Wnt pathway "
                "modulators. PDB 6AHY shows the Wnt3a-Fzd8 complex with its unique "
                "lipid-mediated interaction. Design protein variants that can act as "
                "Wnt agonists or antagonists for regenerative medicine applications."
            ),
            length_range=(60, 120),
            gt_kd_nM=15.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Hirai et al. Nat Struct Mol Biol 2019",
            doi="10.1038/s41594-019-0216-z",
            tags=["complex", "Wnt", "Frizzled", "signaling", "pathway_modulator"],
        ),
        # ----- cpx_str: Structural protein assemblies -----
        TaskSpec(
            task_id="dn_scf_004",
            task_type=CPX,
            biological_context=STR,
            difficulty="easy",
            target_name="Designed two-component nanocage I53-50",
            target_pdb_id="6P6F",
            target_chain="A",
            target_seq_len=211,
            binding_site_residues=[76, 77, 78, 96, 97, 98, 99, 100, 101, 148, 149, 150, 151],
            description=(
                "Engineer a two-component icosahedral nanocage based on the I53-50 "
                "design by Bale et al. PDB 6P6F shows the 120-subunit assembly. "
                "Optimize the designed interface between the pentameric and trimeric "
                "components for improved assembly yield and thermostability."
            ),
            length_range=(80, 120),
            gt_kd_nM=None,
            gt_tm_C=80.0,
            gt_expression=True,
            gt_additional={"assembly_yield_pct": 85.0},
            source="Bale et al. Science 2016",
            doi="10.1126/science.aaf8818",
            tags=["nanocage", "I53-50", "two_component", "icosahedral"],
        ),
        TaskSpec(
            task_id="dn_scf_005",
            task_type=CPX,
            biological_context=STR,
            difficulty="medium",
            target_name="Designed symmetric tetramer",
            target_pdb_id="3R3O",
            target_chain="A",
            target_seq_len=97,
            binding_site_residues=[15, 16, 17, 18, 19, 45, 46, 47, 48, 82, 83, 84, 85],
            description=(
                "Engineer a C4-symmetric homotetramer based on the design by King "
                "et al. PDB 3R3O shows a computationally designed four-helix bundle "
                "tetramer. Optimize the symmetric interface for tighter assembly and "
                "improved thermal stability of the quaternary structure."
            ),
            length_range=(80, 110),
            gt_kd_nM=None,
            gt_tm_C=85.0,
            gt_expression=True,
            gt_additional={"assembly_yield_pct": 90.0},
            source="King et al. Science 2012",
            doi="10.1126/science.1219738",
            tags=["oligomer", "tetramer", "symmetric", "designed_assembly"],
        ),
        TaskSpec(
            task_id="dn_scf_006",
            task_type=CPX,
            biological_context=STR,
            difficulty="medium",
            target_name="ATC-HL3 hetero-oligomer",
            target_pdb_id="7A4B",
            target_chain="A",
            target_seq_len=153,
            binding_site_residues=[20, 23, 24, 27, 55, 58, 62, 65, 100, 103, 107, 110],
            description=(
                "Engineer a heteromeric protein assembly based on the ATC-HL3 "
                "design from Chen et al. PDB 7A4B shows the designed heterotrimeric "
                "coiled-coil assembly. Optimize the hetero-specific interface to "
                "improve selectivity and prevent homo-oligomerization."
            ),
            length_range=(60, 100),
            gt_kd_nM=None,
            gt_tm_C=78.0,
            gt_expression=True,
            gt_additional={"assembly_yield_pct": 75.0},
            source="Chen et al. Nature 2022",
            doi="10.1038/s41586-022-04654-9",
            tags=["hetero-oligomer", "coiled_coil", "specificity", "designed_assembly"],
        ),
        TaskSpec(
            task_id="dn_scf_007",
            task_type=CPX,
            biological_context=STR,
            difficulty="hard",
            target_name="Designed icosahedral nanocage I32-28",
            target_pdb_id="5IM4",
            target_chain="A",
            target_seq_len=280,
            binding_site_residues=[
                50, 51, 52, 53, 90, 91, 92, 93, 150, 151, 152, 200, 201, 202, 203,
            ],
            description=(
                "Engineer a one-component icosahedral protein nanocage based on "
                "the I32-28 design by Hsia et al. PDB 5IM4 shows the 60-subunit "
                "assembly. Optimize the designed interface for robust self-assembly "
                "and cargo encapsulation. This is one of the largest designed cages."
            ),
            length_range=(80, 130),
            gt_kd_nM=None,
            gt_tm_C=75.0,
            gt_expression=True,
            gt_additional={"assembly_yield_pct": 70.0},
            source="Hsia et al. Nature 2016",
            doi="10.1038/nature18010",
            tags=["nanocage", "icosahedral", "I32-28", "one_component"],
        ),
        # ----- cpx_str buffer: +2 RFdiffusion methodology tasks -----
        TaskSpec(
            task_id="dn_scf_008",
            task_type=CPX,
            biological_context=STR,
            difficulty="medium",
            target_name="Designed coiled-coil trimer (RFdiffusion)",
            target_pdb_id="4DZM",
            target_chain="A",
            target_seq_len=66,
            binding_site_residues=[8, 11, 15, 18, 22, 25, 29, 32, 43, 46, 50, 53],
            description=(
                "Engineer a designed coiled-coil homotrimer using RFdiffusion for "
                "backbone generation. PDB 4DZM shows the GCN4 leucine zipper trimer. "
                "Redesign the hydrophobic core and interhelical contacts for improved "
                "thermostability and specificity of the trimeric state."
            ),
            length_range=(60, 80),
            gt_kd_nM=None,
            gt_tm_C=90.0,
            gt_expression=True,
            source="Harbury et al. Science 1993",
            doi="10.1126/science.8248779",
            tags=["coiled_coil", "trimer", "RFdiffusion", "designed_assembly"],
        ),
        TaskSpec(
            task_id="dn_scf_009",
            task_type=CPX,
            biological_context=STR,
            difficulty="hard",
            target_name="Heterodimeric helix bundle (RFdiffusion)",
            target_pdb_id="6MRS",
            target_chain="A",
            target_seq_len=80,
            binding_site_residues=[12, 15, 16, 19, 23, 26, 47, 50, 54, 57, 61, 64],
            description=(
                "Design a heterodimeric four-helix bundle using RFdiffusion backbone "
                "generation with ProteinMPNN sequence design. PDB 6MRS shows a designed "
                "heterospecific protein pair. Create an orthogonal heterodimer with "
                "strong specificity against homodimerization."
            ),
            length_range=(70, 100),
            gt_kd_nM=None,
            gt_tm_C=85.0,
            gt_expression=True,
            source="Chen et al. Nature 2019",
            doi="10.1038/s41586-018-0802-y",
            tags=["heterodimer", "helix_bundle", "RFdiffusion", "specificity"],
        ),
        # ----- cpx_sig buffer: +2 RFdiffusion methodology tasks -----
        TaskSpec(
            task_id="dn_bnd_007",
            task_type=CPX,
            biological_context=SIG,
            difficulty="medium",
            target_name="Designed SH2-peptide complex (RFdiffusion)",
            target_pdb_id="1SHC",
            target_chain="A",
            target_seq_len=114,
            binding_site_residues=[35, 36, 37, 55, 56, 57, 58, 63, 64, 65, 69, 82, 95],
            description=(
                "Engineer the Src SH2 domain-phosphopeptide interface using "
                "RFdiffusion-based backbone design. PDB 1SHC shows the SH2-peptide "
                "complex. Design binder proteins that modulate SH2 domain signaling "
                "with programmable phosphotyrosine selectivity."
            ),
            length_range=(60, 100),
            gt_kd_nM=200.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Waksman et al. Cell 1993",
            doi="10.1016/0092-8674(93)90487-B",
            tags=["complex", "SH2", "phosphopeptide", "RFdiffusion", "signaling"],
        ),
        TaskSpec(
            task_id="dn_bnd_008",
            task_type=CPX,
            biological_context=SIG,
            difficulty="hard",
            target_name="Designed PDZ domain complex (RFdiffusion)",
            target_pdb_id="1BE9",
            target_chain="A",
            target_seq_len=94,
            binding_site_residues=[16, 17, 23, 25, 26, 27, 32, 50, 71, 72, 73, 76, 78],
            description=(
                "Engineer the PDZ domain-peptide interface using RFdiffusion "
                "backbone generation. PDB 1BE9 shows the PSD-95 PDZ3 domain. "
                "Design allosteric modulators that control PDZ-mediated protein "
                "scaffolding in synaptic signaling."
            ),
            length_range=(60, 100),
            gt_kd_nM=100.0,
            gt_tm_C=None,
            gt_expression=True,
            source="Doyle et al. Cell 1996",
            doi="10.1016/S0092-8674(00)80468-3",
            tags=["complex", "PDZ", "scaffolding", "RFdiffusion", "synapse"],
        ),
    ]
