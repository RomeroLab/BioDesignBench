"""Causal depth intervention definitions for BioDesignBench.

Defines the 18-task representative subset (2 per populated taxonomy cell,
selected nearest to cell-median total score), intervention prompts, and
condition enum for forced-depth experiments.

Selection criterion: within each of the 9 populated cells of the
DesignApproach x MolecularSubject taxonomy, the 2 tasks closest to the
cell-median total score were chosen. binder_003 is included as the sole
case-study task from the main analysis (Fig 6c). cfd_flu_004 is the only
task in dn_fp and is included by necessity.
"""

from __future__ import annotations

from enum import Enum


class InterventionCondition(Enum):
    """Experimental conditions for causal depth intervention."""

    BASELINE = "baseline"
    FORCED_DEPTH = "forced_depth"
    LOW_DIVERSITY_CONTROL = "low_diversity_control"


# 18-task representative subset: 2 per populated taxonomy cell (9 cells)
# Format: old task IDs (used throughout the codebase and result files)
INTERVENTION_SUBSET: list[str] = [
    # dn_ab (median=46.1): closest 2
    "dnb_ab_003",        # dn_ab_003, dist=1.7, medium
    "dnb_ab_001",        # dn_ab_001, dist=1.7, easy
    # dn_enz (median=41.7): only 2 tasks exist
    "cfd_enz_005",       # dn_enz_001, dist=9.1, easy
    "cpx_enz_001",       # dn_enz_002, dist=9.1, medium
    # dn_bnd (median=50.7): closest 2 + case study
    "cpx_sig_001",       # dn_bnd_012, dist=0.0 (exact median), easy
    "cpx_sig_008",       # dn_bnd_019, dist=0.8, hard
    "binder_003",        # dn_bnd_001, dist=14.1, case study from Fig 6c
    # dn_scf (median=42.4): closest 2
    "dnk_str_005",       # dn_scf_005, dist=0.0 (exact median), medium
    "cpx_str_007",       # dn_scf_008, dist=0.8, medium
    # dn_fp (median=52.2): only 1 task exists
    "cfd_flu_004",       # dn_fp_001, dist=0.0, hard
    # rd_ab (median=46.2): exact median + DS-exceeds-expert case
    "antibody_004",      # rd_ab_004, dist=0.0 (exact median), hard
    "sqo_ab_006",        # rd_ab_005, dist=15.4, medium
    # rd_enz (median=47.0): closest 2 (tied at dist=4.8)
    "sqo_enz_005",       # rd_enz_004, dist=4.8, medium
    "sqo_sig_002",       # rd_enz_008, dist=4.8, medium
    # rd_scf (median=41.2): closest 2
    "stability_003",     # rd_scf_001, dist=2.4, easy
    "cpx_str_005",       # rd_scf_003, dist=2.5, medium
    # rd_fp (median=53.2): closest 2
    "fluorescence_001",  # rd_fp_002, dist=3.5, medium
    "cfd_flu_003",       # rd_fp_008, dist=3.8, easy
]


FORCED_DEPTH_PROMPT = """\
## MANDATORY EVALUATION PROTOCOL

You must follow this generate-evaluate-filter protocol for every design task.

### Step 1: Candidate generation
Generate at least 5 independent design candidates. Use separate tool calls \
for each candidate rather than requesting all in a single call, to ensure \
structural diversity across backbone samples.

### Step 2: Multi-metric evaluation
For EACH candidate, evaluate using at least 4 distinct assessment categories \
appropriate to the task. Choose from:
  - Structural confidence (e.g., predict_structure, validate_design)
  - Binding/interface quality (e.g., predict_complex, analyze_interface)
  - Energetic stability (e.g., rosetta_score, energy_minimize)
  - Sequence plausibility (e.g., score_stability)
  - Rosetta physics (e.g., rosetta_relax, rosetta_interface_score)
Record all metric values for every candidate.

### Step 3: Composite ranking and filtering
Rank all candidates using a composite score across the metrics collected. \
Output ONLY the top 3 candidates to designed_sequences.fasta. \
Report metrics for ALL candidates (including rejected ones) in metrics.json \
under a "all_candidates" key, so the filtering decision is auditable.

Do NOT skip evaluation steps or output un-evaluated candidates.
"""


LOW_DIVERSITY_CONTROL_PROMPT = """\
## MANDATORY EVALUATION PROTOCOL

You must follow this generate-evaluate-filter protocol for every design task.

### Step 1: Candidate generation
Generate at least 5 independent design candidates. Use separate tool calls \
for each candidate rather than requesting all in a single call.

### Step 2: Structure-focused validation
For EACH candidate, perform thorough structural assessment only:
  - Run structure prediction (predict_structure or validate_design) and record pLDDT, pTM
  - Run a second structure prediction with different parameters or model to cross-check
  - If the task involves a complex, run predict_complex and record ipTM
  - Re-validate the top candidates with an additional structure prediction call
Focus all evaluation effort on structural confidence metrics.

### Step 3: Filter by structural confidence and output top 3
Rank candidates by pLDDT (and ipTM if applicable). Output ONLY the top 3 \
to designed_sequences.fasta. Report all metrics in metrics.json.

Do NOT skip evaluation steps or output un-evaluated candidates.
"""
