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

## Bringing your own MCP tools

BioDesignBench is "bring your own LLM, optionally bring your own tools."
The **Custom MCP** submission mode lets you evaluate any 17-tool
implementation against the identical 76 tasks, the identical agent
harness, and the identical scoring rubric used by the paper's reference
runs. Our [`protein-design-mcp`](https://github.com/RomeroLab/protein-design-mcp)
is just one reference implementation.

### The contract

Your MCP server is a public HTTPS endpoint that accepts POST requests:

```
POST https://your-mcp.example.com/
Authorization: Bearer <optional shared token>
Content-Type: application/json

{
  "name": "predict_structure",
  "arguments": {"sequence": "MKKL..."}
}
```

The response must be a JSON object. Report errors in a top-level
`error` field rather than via HTTP status codes so the agent loop can
see the reason.

### Tool schemas

The 17 reference tool names plus full JSON Schema for each argument
set live in [`mcp_tool_schemas.json`](./mcp_tool_schemas.json). Your
implementation must accept the same tool names and argument shapes —
the leaderboard's agent loop picks tools by name from this list.

### Reference implementation

See [`RomeroLab/protein-design-mcp`](https://github.com/RomeroLab/protein-design-mcp)
for the lab's reference implementation. It is published as a Docker
image and a Modal app; you can fork it, replace individual tool
handlers, and redeploy.

### Hosting options

Any public HTTPS endpoint works — we only check that the contract is
satisfied:

| Option | Pros | Cons |
|---|---|---|
| **Modal** (serverless GPU) | Cheap pay-per-use, auto-scale | Modal account needed |
| **AWS / GCP / Azure VM** | Full control, reuse existing cloud | 24/7 billing or manual shutdown |
| **Runpod / Lambda Labs / Vast** | Cheap GPU rentals | Manual spin-up per submission |
| **Kubernetes / HPC** | Reuse on-prem GPU | Ops overhead |
| **ngrok + local GPU** | $0, fastest iteration | URL is ephemeral |

The lab's reference MCP is hosted on Modal for serverless
pay-per-use cost control; your submission does not have to match.

### Minimal stub

A ~150-line FastAPI template you can fork is at
[`example_mcp_server.py`](./example_mcp_server.py). Replace the
`handle_*` stubs with your implementations, deploy, and paste the URL
into the submission form's **Advanced: Custom MCP** section.


## Backend pipeline phases

Submission processing runs in 4 admin-controlled phases:

| Phase | Step | Status | Notes |
|---|---|---|---|
| **A** | Dispatch tasks → CPU scoring | live | HTTP POST to submitter endpoint, validate, score 5/6 components |
| **B** | Boltz-2 structure verification | live (Modal) | Modal-hosted A10G companion app provisions GPU on demand |
| **C** | LLM judge panel (28-pt hybrid) | live | 3-judge PoLL with self-exclusion, requires API key secrets |
| **D** | Finalize + publish to leaderboard | live | Aggregates hybrid scores, writes back to submissions dataset |

### Phase B architecture (Modal companion app)

The HF Space runs on `cpu-basic` and cannot host Boltz directly, so
Phase B uses a Modal-deployed sidecar (`modal_boltz_app.py`) that:

- pre-builds an image with `boltz==2.2.1`, `torch==2.10`, NVIDIA
  cuequivariance kernels, and FastAPI;
- exposes a single web endpoint at
  `https://<workspace>--bdb-boltz-predict.modal.run`;
- spins up an A10G on demand, runs `boltz predict` (via the same CLI
  the dev pipeline uses), and returns confidence metrics;
- auto-stops after 5 minutes idle so the lab is only billed for active
  inference time (~$0.06 per task at A10G rates).

The HF Space is just an HTTP client (`eval_boltz.py`); design sequences
are POSTed to the Modal endpoint with a shared bearer token. To
deploy the sidecar (one time):

```bash
cd biodesignbench-leaderboard
modal deploy modal_boltz_app.py
```

Then set these HF Space secrets:

```
MODAL_BOLTZ_URL    https://<workspace>--bdb-boltz-predict.modal.run
MODAL_BOLTZ_TOKEN  matches the modal secret `bdb-boltz-shared` TOKEN
```

If `MODAL_BOLTZ_URL` is unset, Phase B predictors return a structured
failure dict with `success=False` and an actionable error message
instead of crashing the dispatcher.
