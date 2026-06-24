# BioDesignBench

> **Evaluating LLM-Driven Protein Design: Agents Lack Iterative Evaluation Depth**
> Jeonghyeon Kim & Philip Romero — Romero Lab, Duke University
>
> 📄 **Paper:** _coming soon_ &nbsp;&middot;&nbsp;
> 🏆 **Leaderboard:** [`RomeroLab-Duke/BioDesignBench-Leaderboard`](https://huggingface.co/spaces/RomeroLab-Duke/BioDesignBench-Leaderboard) &nbsp;&middot;&nbsp;
> 🧬 **Reference MCP server:** [`jasonkim8652/protein-design-mcp`](https://github.com/jasonkim8652/protein-design-mcp) &middot; `pip install protein-design-mcp`

BioDesignBench is a benchmark for testing whether tool-augmented LLM agents can
orchestrate the **stochastic, multi-step pipelines of computational protein
design**. Where existing chemistry-agent and code-agent benchmarks evaluate
deterministic tool chains, we focus on the qualitatively different setting in
which generative tools (RFdiffusion, ProteinMPNN, Boltz-2) sample from
distributions over structures and sequences and a competent practitioner must
generate multiple candidates and **screen them across complementary
biophysical metrics** before a design is viable.

We evaluate four frontier LLMs (DeepSeek V3, GPT-5, Claude Sonnet 4.5,
Gemini 2.5 Pro) under guided and unguided MCP-tool presentation modes against
deterministic and human baselines on **76 expert-curated tasks** drawn from
2024–2026 literature. The headline finding: top-tier agents now beat a
hardcoded pipeline, but **invoke evaluation tools at only 14% of expert depth**,
and workflow guidance rescues coverage without rescuing depth.

```
                                                     Hybrid score (100 pts)
    Human Oracle                ████████████████████  74.9
    Human Expert                █████████████████      61.3
    DeepSeek V3 (unguided)      █████████████████      60.4
    DeepSeek V3 (guided)        ████████████████       58.5
    GPT-5 (unguided)            ███████████████        55.6
    GPT-5 (guided)              ███████████████        55.3
    Hardcoded Pipeline          ███████████████        54.2
    Claude Sonnet 4.5 (guided)  ██████████████         50.2
    Claude Sonnet 4.5 (unguid)  ████████████           41.2
    Gemini 2.5 Pro              ██                      8.4
```

## Three principal findings

1. **Top-tier LLM agents now beat a deterministic pipeline.** DeepSeek V3 and
   GPT-5 surpass a hand-engineered hardcoded pipeline (54.2) under both modes.
   Autonomous protein-design orchestration is no longer infeasible.
2. **Coverage–depth dissociation.** Workflow guidance closes the *coverage* gap
   (Rescue Index up to +3.01) but leaves *utilisation depth* unchanged
   (Rescue Index ≈ 0). Better tool docs cannot teach iterative depth.
3. **Evaluation depth, not tool knowledge, is the bottleneck.** Across 836
   task–condition observations, evaluation depth per candidate correlates with
   total score at *ρ* = 0.685 (*p* < 10⁻¹¹⁷). LLM agents generate backbone
   candidates at expert-level rates but evaluate each one at **14% of expert
   depth**. Forced-depth interventions confirm this is causal.

## Why the task data is **not** in this repo

To prevent contamination of future language models, the 76 task specifications,
their input PDBs, ground truth, and oracle outputs are deliberately **not
released here**. The benchmark is hosted as a private HuggingFace dataset and
agents are evaluated through the public submission flow at the leaderboard URL
above. The repo contains:

- the **scoring & evaluation pipeline** (`biodesignbench/eval/`)
- the **agent harness, baselines, and bio-specific agent wrappers**
  (`biodesignbench/agents/`)
- the **MCP tool provider** that maps the 17 reference tools to
  Anthropic / OpenAI / Gemini function-calling schemas
  (`biodesignbench/tools/`)
- the **2 × 5 taxonomy module** (`biodesignbench/taxonomy.py`)
- the **LLM judge** for the 28-point rubric portion
  (`biodesignbench/eval/llm_judge/`)
- **all paper figure-generating analysis scripts** (`scripts/analysis/`)
- the **HuggingFace Space leaderboard backend** (`biodesignbench-leaderboard/`)
- a **public demo task** for reviewer reproducibility (`examples/demo_task/`)

Anything that would let you reconstruct a task — input files, prompts, ground
truth, baseline outputs, results CSVs — is held privately by Romero Lab and
served at evaluation time only. Researchers requiring per-task data for
replication studies may contact the corresponding author under a data use
agreement.

## Repository layout

```
BioDesignBench/
├── biodesignbench/                # Python package
│   ├── taxonomy.py                # 2 × 5 design matrix (DesignApproach × MolecularSubject)
│   ├── eval/                      # 100-point scoring pipeline
│   │   ├── tier1/                 #   Bio-coding tasks (unit-test style)
│   │   ├── tier2/                 #   Design tasks (4D metrics + Boltz-2 verification)
│   │   ├── metrics/               #   approach / orchestration / quality / etc.
│   │   ├── llm_judge/             #   28-pt LLM judge panel (PoLL with self-exclusion)
│   │   └── pipeline.py            #   Top-level orchestration
│   ├── agents/                    # Agent harness
│   │   ├── general_purpose/       #   GPT-5, Claude Sonnet, Gemini, DeepSeek wrappers
│   │   ├── bio_specific/          #   Biomni / STELLA / BioML wrappers
│   │   └── baselines/             #   Hardcoded pipeline + human-expert agent
│   ├── tools/                     # 17-tool MCP provider with mode toggle
│   ├── interventions.py           # Forced-depth & low-diversity intervention specs
│   └── tool_audit.py              # Tool-call trace analysis
├── biodesignbench-leaderboard/    # Gradio HuggingFace Space (backend + UI)
├── examples/demo_task/            # Public demo task for reviewer reproducibility
├── scripts/analysis/              # All paper figure / SI analysis scripts (60 files)
├── docker/sandbox/                # Sandbox image for executing agent-generated code
├── docs/PRD.md                    # Project requirements document
├── pyproject.toml
└── environment.yml
```

## System requirements

- **Operating systems tested:** Ubuntu 22.04 LTS, macOS 14 (Sonoma).
- **Python:** 3.11 (pinned in `environment.yml` and `pyproject.toml`).
- **Required non-standard hardware:** NVIDIA GPU (A10G or comparable) for
  RFdiffusion, Boltz-2, ESMFold, and ProteinMPNN. The scoring pipeline,
  analysis scripts, and figure-generation code run on CPU.
- **Typical install time on a normal desktop:** 10 to 15 minutes for the conda
  environment; approximately 30 minutes total when including pip extras and
  the `protein-design-mcp` Docker image pull.
- **Key dependency versions** (full pin list in `pyproject.toml` and
  `environment.yml`): NumPy ≥ 1.24, pandas ≥ 2.0, SciPy ≥ 1.10,
  scikit-learn ≥ 1.3, biopython ≥ 1.81, PyTorch ≥ 2.0, matplotlib ≥ 3.7,
  seaborn ≥ 0.12, anthropic SDK ≥ 0.75, openai ≥ 1.12,
  google-generativeai ≥ 0.8.

## Quickstart (developers)

### 1. Install

```bash
git clone https://github.com/RomeroLab/BioDesignBench.git
cd BioDesignBench

# Conda environment (CPU only — no protein-design GPU tools)
conda env create -f environment.yml
conda activate biodesignbench

# Editable install with optional extras
pip install -e ".[dev,agents]"
```

For the GPU-side protein-design tools (RFdiffusion, ProteinMPNN, Boltz-2,
PyRosetta, AF2), install the reference MCP server:

```bash
pip install protein-design-mcp
# Source, Dockerfiles, and Modal deploy template:
#   https://github.com/jasonkim8652/protein-design-mcp
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY
```

### 3. Inspect the scoring pipeline

```python
from biodesignbench.eval.pipeline import score_design
from biodesignbench.taxonomy import get_category, DesignApproach, MolecularSubject

# 2 × 5 taxonomy
cat = get_category("dn_bnd_001")
print(cat.approach, cat.subject)
# DesignApproach.DE_NOVO, MolecularSubject.BINDER

# Score a hypothetical design (without task data, only the rubric pipeline)
help(score_design)
```

### 4. Run an analysis script

All paper figures and SI analyses are reproducible from the canonical
score CSVs (held privately). Each script in `scripts/analysis/` is named
after the figure it produces:

```
scripts/analysis/bdb_022_fig2_leaderboard.py        # Figure 2: leaderboard
scripts/analysis/bdb_023_fig3_mode_comparison.py    # Figure 3: coverage–depth dissociation
scripts/analysis/bdb_050_variance_decomposition.py  # Figure 5: variance partition
scripts/analysis/bdb_060_contamination.py           # SI Figure 9: contamination
```

## Demo

A worked example using a public trypsin-binder design task is shipped in
`examples/demo_task/` so reviewers and new users can run the scoring
pipeline end to end without access to the private benchmark tasks. To
run it:

```bash
biodesignbench score \
  --task examples/demo_task/trypsin_binder.json \
  --output examples/demo_output/
```

**Expected output:** a JSON file in `examples/demo_output/` containing
the six rubric component scores (Approach, Orchestration, Quality,
Feasibility, Novelty, Diversity) summing to a total out of 100, a
per-task scoring log, and the predicted complex structure as a PDB
file.

**Expected run time on a normal desktop:** approximately 2 minutes for
the scoring pipeline alone (using pre-computed structures shipped with
the demo); approximately 10 minutes when also running Boltz-2 structure
verification on a single A10G GPU.

The demo task is fully public and does not overlap with any of the 76
private benchmark tasks, so running it does not compromise the
contamination defense described above.

## Submitting an agent for evaluation

Submissions are accepted through the **HuggingFace Space**:

👉 https://huggingface.co/spaces/RomeroLab-Duke/BioDesignBench-Leaderboard

Unlike most agent benchmarks, **submitters do not host an HTTP endpoint**.
The 76 task descriptions never leave Romero Lab infrastructure. You provide:

- an **LLM provider + API key** — we run the BioDesignBench agent loop
  against your chosen model (Anthropic / OpenAI / Google / DeepSeek)
  inside the leaderboard backend. Your key is scrubbed from our records
  immediately after the dispatch phase.
- *(optional)* a **custom MCP URL** if you want to evaluate your own
  tool implementations. Otherwise, the agent calls our reference
  protein-design-mcp endpoint.

Each submission carries a unique canary token embedded as an HTML
comment in every task prompt, so we can retrospectively detect leakage
if any future model regurgitates it.

### Bring your own tools (Custom MCP)

If you want to benchmark a new tool implementation (a faster structure
predictor, a different diffusion backbone, your own stability model)
against the same 76 tasks / same scoring rubric used by the paper, stand
up an HTTPS endpoint satisfying the MCP contract and paste the URL into
the submission form's **Advanced: Custom MCP** section:

- **Contract + hosting options**:
  [`biodesignbench-leaderboard/README.md`](biodesignbench-leaderboard/README.md#bringing-your-own-mcp-tools)
- **Minimal FastAPI stub (~150 lines)**:
  [`biodesignbench-leaderboard/example_mcp_server.py`](biodesignbench-leaderboard/example_mcp_server.py)
- **Reference implementation to fork**:
  [`jasonkim8652/protein-design-mcp`](https://github.com/jasonkim8652/protein-design-mcp)
  (PyPI: `protein-design-mcp`; Modal deploy template included in
  `deploy/modal_app.py`)

The MCP server — ours or yours — only ever sees operational tool
arguments (sequences, PDB paths, hotspot residues). It never sees the
raw task prompt or evaluation criteria.

**Rate limit:** 1 submission per calendar month per organization.
LLM-judge API costs are paid by Romero Lab; please be considerate.

### Backend pipeline status

| Phase | Step | Status |
|---|---|---|
| A | Dispatch tasks → CPU scoring (5/6 components) | live |
| B | Boltz-2 structure verification | live (Modal-hosted A10G sidecar) |
| C | LLM-judge panel (28-pt hybrid) | live |
| D | Finalize + publish | live |

See [`biodesignbench-leaderboard/README.md`](biodesignbench-leaderboard/README.md)
for the Modal companion-app deployment notes.

## Citation

```bibtex
@article{biodesignbench2026,
  title  = {Evaluating LLM-Driven Protein Design:
            Agents Lack Iterative Evaluation Depth},
  author = {Kim, Jeonghyeon and Romero, Philip},
  year   = {2026},
}
```

## License

Code: MIT. Task content (held privately): not licensed for redistribution.

## Contact

- Jeonghyeon Kim — `jeonghyeon.kim@duke.edu`
- Philip Romero — `philip.romero@duke.edu`
