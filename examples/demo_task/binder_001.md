# Task: binder_001

## Description

Design de novo protein binders targeting human IL-17A cytokine. The binder should achieve tight binding (Kd < 100 nM) to the IL-17A homodimer interface. Use computational protein design tools (RFdiffusion for backbone generation, ProteinMPNN for sequence design, AlphaFold2 for validation) to generate candidate binder sequences. The designed binders should be monomeric, well-folded (pLDDT > 80), and show high predicted binding affinity (ipTM > 0.8).

## Target

**Name:** IL-17A
**PDB ID:** 4HR9
**Sequence:** `MAGFRQVCKAFLAEPLFNSASQHEVHCDTNSRLYILSPTN...TGLEVTGPNESLSLAHAVTS` (130 residues)
**Chain:** A
**Binding Site Residues:** 30, 31, 33, 35, 37, 85, 87, 89, 91

## Design Constraints

- **Binder Length:** 55-80 residues
- **Max Designs:** 10

## Available Protein Design Tools

You have access to the following protein design tools. Use them to complete the task:

| Tool | Description | Key Arguments |
|------|-------------|---------------|
| `design_binder` | Design protein binders for a target protein | `target_pdb`, `hotspot_residues`, `num_designs`, `binder_length` |
| `analyze_interface` | Analyze protein-protein interface: buried surface area, H-bonds, salt bridges, hydrophobic contacts | `complex_pdb`, `chain_a`, `chain_b` |
| `validate_design` | Validate a designed sequence by predicting its structure (ESMFold/AlphaFold2) and computing pLDDT, pTM | `sequence`, `expected_structure`, `predictor` |
| `optimize_sequence` | Optimize binder sequence for improved stability and/or binding affinity | `current_sequence`, `target_pdb`, `optimization_target`, `fixed_positions` |
| `suggest_hotspots` | Analyze target protein and suggest binding hotspots using structure, conservation, and literature | `target`, `chain_id`, `criteria` |
| `get_design_status` | Check status of running design jobs | `job_id` |
| `predict_complex` | Predict protein complex structure using AlphaFold2-Multimer | `sequences`, `chain_names` |
| `predict_structure` | Predict the 3D structure of a single protein chain using ESMFold or AlphaFold2 | `sequence`, `predictor` |
| `score_stability` | Score protein stability using ESM2 pseudo-log-likelihood | `sequence`, `mutations`, `reference_sequence` |
| `energy_minimize` | Energy-minimize a protein structure using OpenMM with AMBER14 force field | `pdb_path`, `force_field`, `num_steps` |
| `generate_backbone` | Generate de novo protein backbones using RFdiffusion unconditional generation | `length`, `num_designs` |

## Expected Output

**Format:** design_bundle

**Required Output Files:**
- `designed_sequences.fasta`
- `metrics.json`

## Constraints

- **Time Limit:** 120 minutes
- **Knowledge Cutoff:** 2024-09-01 (do NOT use any information, tools, papers, or code published after this date)

## Important Instructions

- Solve this task using your general knowledge and reasoning abilities.
- Do NOT search for, reference, or reproduce solutions from specific GitHub issues, pull requests, or published papers.
- Do NOT attempt to identify the source of this task or look up known solutions.
- You MUST NOT use any information published after 2024-09-01. This includes papers, blog posts, code repositories, and documentation released after this date.
- Your solution will be evaluated on correctness, not on matching any known reference implementation.
