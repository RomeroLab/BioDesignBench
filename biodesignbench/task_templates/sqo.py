"""Sequence optimization task templates.

Cells covered:
    sqo_ab  — 5 existing, 1 new (antibody CDR optimization)
    sqo_enz — 4 existing, 2 new (enzyme stability/activity)
    sqo_sig — 0 existing, 3 new (signaling protein optimization)
    sqo_str — 2 existing, 4 new (structural protein stability)
    sqo_flu — 1 existing, 5 new (fluorescent protein optimization)
"""

from __future__ import annotations

from biodesignbench.task_generator import TaskSpec
from biodesignbench.taxonomy import DesignApproach, MolecularSubject


def generate_specs() -> list[TaskSpec]:
    """Return TaskSpec instances for sequence_optimization tasks."""
    SQO = DesignApproach.REDESIGN
    AB = MolecularSubject.ANTIBODY
    ENZ = MolecularSubject.ENZYME
    SIG = MolecularSubject.BINDER
    STR = MolecularSubject.SCAFFOLD
    FLU = MolecularSubject.FLUORESCENT_PROTEIN

    return [
        # ----- sqo_ab: buffer (5 existing + 1 new) -----
        TaskSpec(
            task_id="rd_ab_001",
            task_type=SQO,
            biological_context=AB,
            difficulty="medium",
            target_name="Durvalumab anti-PD-L1 antibody",
            target_pdb_id="5X8L",
            target_chain="A",
            target_seq_len=230,
            binding_site_residues=[33, 50, 52, 53, 54, 55, 56, 98, 100, 101, 102, 103],
            description=(
                "Optimize the CDR regions of durvalumab, an FDA-approved anti-PD-L1 "
                "checkpoint inhibitor. PDB 5X8L shows the Fab-PD-L1 complex. "
                "Improve binding affinity while preserving developability "
                "(aggregation resistance and expression levels)."
            ),
            length_range=(220, 240),
            gt_kd_nM=0.67,
            gt_tm_C=74.0,
            gt_expression=True,
            source="Tan et al. MAbs 2018",
            doi="10.1080/19420862.2018.1433975",
            tags=["antibody", "PD-L1", "checkpoint", "CDR_optimization"],
        ),
        # ----- sqo_enz: enzyme optimization -----
        TaskSpec(
            task_id="rd_enz_001",
            task_type=SQO,
            biological_context=ENZ,
            difficulty="medium",
            target_name="Cytochrome P450 BM3 (CYP102A1)",
            target_pdb_id="1FAG",
            target_chain="A",
            target_seq_len=455,
            binding_site_residues=[47, 51, 75, 87, 88, 181, 184, 188, 260, 263, 264, 328, 330, 400],
            description=(
                "Optimize the heme domain of cytochrome P450 BM3 for improved "
                "thermostability and broadened substrate specificity. The PDB 1FAG "
                "structure shows the wild-type fatty acid hydroxylase. Directed "
                "evolution has identified key stabilizing mutations at positions "
                "near the substrate access channel."
            ),
            length_range=(440, 470),
            gt_kd_nM=None,
            gt_tm_C=57.0,
            gt_expression=True,
            gt_additional={"kcat_per_s": 17000.0},
            source="Wong et al. Biotechnol Bioeng 2007",
            doi="10.1002/bit.21562",
            tags=["enzyme", "P450", "thermostability", "substrate_specificity"],
        ),
        TaskSpec(
            task_id="rd_enz_002",
            task_type=SQO,
            biological_context=ENZ,
            difficulty="hard",
            target_name="Kemp eliminase HG3.17",
            target_pdb_id="3NZ1",
            target_chain="A",
            target_seq_len=254,
            binding_site_residues=[50, 52, 101, 136, 176, 178, 201, 210, 233],
            description=(
                "Optimize the computationally designed Kemp eliminase HG3.17 for "
                "increased catalytic efficiency (kcat/Km). PDB 3NZ1 shows the "
                "TIM-barrel scaffold with the designed active site. Mutations should "
                "improve transition-state stabilization while maintaining fold stability."
            ),
            length_range=(245, 265),
            gt_kd_nM=None,
            gt_tm_C=63.5,
            gt_expression=True,
            gt_additional={"kcat_per_s": 700.0, "kcat_km_per_M_per_s": 230000.0},
            source="Khersonsky et al. JACS 2012",
            doi="10.1021/ja3037367",
            tags=["enzyme", "Kemp_eliminase", "designed_enzyme", "catalysis"],
        ),
        # ----- sqo_sig: signaling protein optimization -----
        TaskSpec(
            task_id="rd_bnd_001",
            task_type=SQO,
            biological_context=SIG,
            difficulty="easy",
            target_name="Interleukin-2 (IL-2) stability",
            target_pdb_id="1M47",
            target_chain="A",
            target_seq_len=133,
            description=(
                "Optimize human IL-2 for improved thermostability and extended "
                "half-life. PDB 1M47 shows IL-2 bound to IL-2Ralpha. IL-2 is a "
                "key cytokine for cancer immunotherapy but has poor stability. "
                "Design mutations that improve Tm without disrupting receptor binding."
            ),
            length_range=(125, 140),
            gt_tm_C=55.0,
            gt_expression=True,
            gt_additional={"kd_nM_receptor": 10.0},
            source="Wang et al. Science 2005",
            doi="10.1126/science.1117893",
            tags=["cytokine", "IL-2", "stability", "immunotherapy"],
        ),
        TaskSpec(
            task_id="rd_bnd_002",
            task_type=SQO,
            biological_context=SIG,
            difficulty="medium",
            target_name="Interferon-alpha2 (IFNa2) optimization",
            target_pdb_id="1RH2",
            target_chain="A",
            target_seq_len=166,
            description=(
                "Optimize interferon-alpha2 for improved thermostability and "
                "pharmacokinetics. PDB 1RH2 shows the IFNa2 structure. Type I "
                "interferons are critical antiviral cytokines but are limited by "
                "instability and short half-life. Engineer stability-enhancing mutations."
            ),
            length_range=(160, 175),
            gt_tm_C=58.0,
            gt_expression=True,
            source="Radhakrishnan et al. Structure 1996",
            doi="10.1016/S0969-2126(96)00162-5",
            tags=["cytokine", "interferon", "stability", "antiviral"],
        ),
        TaskSpec(
            task_id="rd_bnd_003",
            task_type=SQO,
            biological_context=SIG,
            difficulty="hard",
            target_name="Erythropoietin (EPO) thermostability",
            target_pdb_id="1BUY",
            target_chain="A",
            target_seq_len=166,
            description=(
                "Optimize erythropoietin (EPO) for improved thermostability and "
                "resistance to aggregation. PDB 1BUY shows the EPO-receptor complex. "
                "EPO is a four-helix bundle cytokine used to treat anemia but is "
                "prone to aggregation. Design stabilizing mutations while preserving "
                "receptor binding and biological activity."
            ),
            length_range=(160, 175),
            gt_tm_C=52.0,
            gt_expression=True,
            gt_additional={"kd_nM_receptor": 1.0},
            source="Syed et al. Nature 1998",
            doi="10.1038/25940",
            tags=["cytokine", "EPO", "stability", "aggregation"],
        ),
        # ----- sqo_str: structural protein stability -----
        TaskSpec(
            task_id="rd_scf_001",
            task_type=SQO,
            biological_context=STR,
            difficulty="easy",
            target_name="Villin headpiece HP35",
            target_pdb_id="1YRF",
            target_chain="A",
            target_seq_len=35,
            description=(
                "Optimize the villin headpiece subdomain HP35 for improved "
                "thermostability. HP35 (PDB 1YRF) is one of the smallest and "
                "fastest-folding proteins known, making it a benchmark for protein "
                "design. Mutations should increase Tm while preserving the three-helix "
                "bundle fold."
            ),
            length_range=(30, 40),
            gt_tm_C=73.0,
            gt_expression=True,
            source="Kubelka et al. JACS 2003",
            doi="10.1021/ja0360133",
            tags=["miniprotein", "villin", "fast_folding", "thermostability"],
        ),
        TaskSpec(
            task_id="rd_scf_002",
            task_type=SQO,
            biological_context=STR,
            difficulty="easy",
            target_name="WW domain FiP35",
            target_pdb_id="2F21",
            target_chain="A",
            target_seq_len=35,
            description=(
                "Optimize the WW domain FiP35 for improved thermostability. "
                "The WW domain (PDB 2F21) is a 35-residue beta-sheet miniprotein "
                "that recognizes proline-rich motifs. FiP35 is one of the fastest "
                "folding proteins and a key benchmark for computational protein design."
            ),
            length_range=(30, 40),
            gt_tm_C=77.0,
            gt_expression=True,
            source="Liu et al. PNAS 2008",
            doi="10.1073/pnas.0711908105",
            tags=["miniprotein", "WW_domain", "beta_sheet", "fast_folding"],
        ),
        TaskSpec(
            task_id="rd_scf_003",
            task_type=SQO,
            biological_context=STR,
            difficulty="medium",
            target_name="Top7 (computationally designed protein)",
            target_pdb_id="1QYS",
            target_chain="A",
            target_seq_len=93,
            description=(
                "Optimize Top7, the first protein designed with a novel fold not "
                "found in nature (PDB 1QYS, Kuhlman et al. Science 2003). Improve "
                "thermostability and expression while preserving the unique "
                "alpha/beta topology. This is a landmark target in computational "
                "protein design."
            ),
            length_range=(85, 100),
            gt_tm_C=99.0,
            gt_expression=True,
            source="Kuhlman et al. Science 2003",
            doi="10.1126/science.1089427",
            tags=["designed_protein", "novel_fold", "Top7", "thermostability"],
        ),
        TaskSpec(
            task_id="rd_scf_004",
            task_type=SQO,
            biological_context=STR,
            difficulty="hard",
            target_name="Trp-cage miniprotein TC5b",
            target_pdb_id="1L2Y",
            target_chain="A",
            target_seq_len=20,
            description=(
                "Optimize the Trp-cage miniprotein TC5b, a 20-residue construct "
                "that is the smallest known stably folding protein (PDB 1L2Y). "
                "Improve thermostability beyond the wild-type Tm of 42C while "
                "maintaining the characteristic tryptophan-cage topology. Very "
                "challenging due to the minimal sequence length."
            ),
            length_range=(18, 25),
            gt_tm_C=42.0,
            gt_expression=True,
            source="Neidigh et al. Nat Struct Biol 2002",
            doi="10.1038/nsb798",
            tags=["miniprotein", "Trp_cage", "ultrafast_folding", "thermostability"],
        ),
        # ----- sqo_flu: fluorescent protein optimization -----
        TaskSpec(
            task_id="rd_fp_001",
            task_type=SQO,
            biological_context=FLU,
            difficulty="easy",
            target_name="mCherry red fluorescent protein",
            target_pdb_id="2H5Q",
            target_chain="A",
            target_seq_len=236,
            description=(
                "Optimize mCherry for improved brightness, photostability, and "
                "maturation kinetics. PDB 2H5Q shows the mCherry beta-barrel "
                "structure. Focus mutations near the chromophore environment to "
                "improve quantum yield while maintaining the red emission wavelength."
            ),
            length_range=(225, 245),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.22, "extinction_coeff": 72000.0},
            source="Shaner et al. Nat Biotechnol 2004",
            doi="10.1038/nbt1037",
            tags=["fluorescent_protein", "mCherry", "red", "brightness"],
        ),
        TaskSpec(
            task_id="rd_fp_002",
            task_type=SQO,
            biological_context=FLU,
            difficulty="easy",
            target_name="YFP (yellow fluorescent protein)",
            target_pdb_id="1YFP",
            target_chain="A",
            target_seq_len=238,
            description=(
                "Optimize YFP for improved brightness and reduced sensitivity to "
                "chloride and pH. PDB 1YFP shows the GFP-derived yellow variant "
                "with the T203Y mutation. Designs should improve photostability "
                "while maintaining the yellow emission at approximately 527 nm."
            ),
            length_range=(230, 245),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.61, "extinction_coeff": 83400.0},
            source="Wachter et al. Structure 1998",
            doi="10.1016/S0969-2126(98)00131-9",
            tags=["fluorescent_protein", "YFP", "yellow", "GFP_variant"],
        ),
        TaskSpec(
            task_id="rd_fp_003",
            task_type=SQO,
            biological_context=FLU,
            difficulty="medium",
            target_name="mTurquoise2 (CFP variant)",
            target_pdb_id="4B5Y",
            target_chain="A",
            target_seq_len=239,
            description=(
                "Optimize mTurquoise2 for improved brightness and monoexponential "
                "fluorescence lifetime, important for FRET applications. PDB 4B5Y "
                "shows the optimized cyan variant with quantum yield 0.93. Further "
                "improve photostability and maturation at 37C."
            ),
            length_range=(230, 245),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.93, "extinction_coeff": 30000.0},
            source="Goedhart et al. Nat Commun 2012",
            doi="10.1038/ncomms1544",
            tags=["fluorescent_protein", "mTurquoise2", "cyan", "FRET"],
        ),
        TaskSpec(
            task_id="rd_fp_004",
            task_type=SQO,
            biological_context=FLU,
            difficulty="medium",
            target_name="mNeonGreen fluorescent protein",
            target_pdb_id="5LTR",
            target_chain="A",
            target_seq_len=237,
            description=(
                "Optimize mNeonGreen, one of the brightest monomeric fluorescent "
                "proteins derived from the lancelet Branchiostoma lanceolatum. "
                "PDB 5LTR shows its unique beta-barrel fold. Improve photostability "
                "and folding efficiency while maintaining the high quantum yield."
            ),
            length_range=(230, 245),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.80, "extinction_coeff": 116000.0},
            source="Shaner et al. Nat Methods 2013",
            doi="10.1038/nmeth.2413",
            tags=["fluorescent_protein", "mNeonGreen", "green", "bright"],
        ),
        TaskSpec(
            task_id="rd_fp_005",
            task_type=SQO,
            biological_context=FLU,
            difficulty="hard",
            target_name="mScarlet red fluorescent protein",
            target_pdb_id="5LK4",
            target_chain="A",
            target_seq_len=232,
            description=(
                "Optimize mScarlet, the brightest available monomeric red "
                "fluorescent protein. PDB 5LK4 shows the structure engineered by "
                "rational design and machine learning. Improve photostability and "
                "maturation speed while maintaining the record-high quantum yield of 0.70."
            ),
            length_range=(225, 240),
            gt_tm_C=None,
            gt_expression=True,
            gt_additional={"quantum_yield": 0.70, "extinction_coeff": 100000.0},
            source="Bindels et al. Nat Methods 2017",
            doi="10.1038/nmeth.4074",
            tags=["fluorescent_protein", "mScarlet", "red", "bright"],
        ),
    ]
