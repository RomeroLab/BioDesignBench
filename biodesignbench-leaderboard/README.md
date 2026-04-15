---
title: BioDesignBench Leaderboard
emoji: "\U0001F9EC"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
license: mit
---

# BioDesignBench Leaderboard

Evaluating LLM Agents on Protein Design via MCP Tools.

**Romero Lab, Duke University**

## Overview

BioDesignBench evaluates LLM agents as orchestrators of multi-step *stochastic*
protein-design pipelines. This leaderboard tracks agent performance across
**76 design tasks** spanning a **2 × 5 design matrix** (de novo design vs
redesign × five molecular families: antibody, binder, enzyme, scaffold,
fluorescent protein, **9 occupied cells**), scored on a 100-point hybrid rubric:
**72 algorithmic points** (Boltz-2 verification + sequence/feasibility metrics)
plus **28 LLM-judge points** (3-judge panel with self-exclusion).

The six rubric components are Approach, Orchestration, Quality, Feasibility,
Novelty, and Diversity. See the *About* tab for the full methodology and the
*Depth Gap* tab for evaluation-depth interventions.

## Features

- **Overall Leaderboard** — Mixed-ranking table with human baselines and LLM agents
- **Taxonomy Heatmap** — Per-cell scores across the 9 occupied cells of the 2 × 5 design matrix
- **Component Analysis** — Radar and bar charts comparing the 6 scoring components
- **Guidance Effect** — Paired comparison of the same LLM in unguided (atomic tools) vs guided (composite workflows) mode
- **Depth Gap** — Forced-depth and low-diversity intervention results
- **About** — Methodology, submission guide, and citation info

## Backend pipeline phases

Submission processing runs in 4 admin-controlled phases:

| Phase | Step | Status | Notes |
|---|---|---|---|
| **A** | Dispatch tasks → CPU scoring | live | HTTP POST to submitter endpoint, validate, score 5/6 components |
| **B** | Boltz-2 structure verification | code-ready | Needs ZeroGPU hardware + uncommented `torch`/`boltz` deps |
| **C** | LLM judge panel (28-pt hybrid) | live | 3-judge PoLL with self-exclusion, requires API key secrets |
| **D** | Finalize + publish to leaderboard | live | Aggregates hybrid scores, writes back to submissions dataset |

### Phase B activation checklist

To wire up Boltz-2 verification on this Space:

1. **Switch hardware** in HF Space settings → Hardware → `zero-a10g`
   (requires HF Pro / Enterprise).
2. **Edit `requirements.txt`** and uncomment the two lines:
   ```
   torch>=2.2
   boltz>=0.4
   ```
3. **Verify secrets** are set: `HF_TOKEN` (private dataset),
   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
   `DEEPSEEK_API_KEY`.
4. Restart the Space. The first build will pull ~2GB of CUDA wheels.

On `cpu-basic` hardware the Phase B predictors return a structured
failure dict with `success=False` and an actionable error message
instead of crashing the dispatcher.
