"""Protein design tool provider that wraps the protein-design-mcp server.

Manages an MCP server subprocess (stdio transport) and provides tool
definitions in Anthropic, OpenAI, and Google Gemini formats so that
general-purpose LLM agents can invoke protein design tools uniformly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 17 protein design tools with their JSON Schema parameter definitions
# ---------------------------------------------------------------------------

PROTEIN_DESIGN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "design_binder",
        "description": (
            "End-to-end binder design pipeline: RFdiffusion (conditional backbone) -> "
            "ProteinMPNN (sequence design) -> ESMFold (validation). "
            "Convenience wrapper — same result as calling generate_backbone, "
            "design_sequence, and predict_structure individually."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_pdb": {
                    "type": "string",
                    "description": "Path to target protein PDB file",
                },
                "hotspot_residues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Target residues for binder interface, e.g. ['A45', 'A46']"
                    ),
                },
                "num_designs": {
                    "type": "integer",
                    "description": "Number of designs to generate (default: 10)",
                    "default": 10,
                },
                "binder_length": {
                    "type": "integer",
                    "description": "Binder length in residues (default: 80)",
                    "default": 80,
                },
            },
            "required": ["target_pdb", "hotspot_residues"],
        },
    },
    {
        "name": "design_fold",
        "description": (
            "End-to-end de novo fold design pipeline: RFdiffusion (unconditional backbone) -> "
            "ProteinMPNN (sequence design) -> AlphaFold2 (structure validation). "
            "Convenience wrapper — same result as calling generate_backbone, "
            "design_sequence, and predict_structure individually."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "description": "Backbone length in residues",
                },
                "num_designs": {
                    "type": "integer",
                    "description": "Number of designs to generate (default: 10)",
                    "default": 10,
                },
                "num_sequences_per_backbone": {
                    "type": "integer",
                    "description": "ProteinMPNN sequences per backbone (default: 4)",
                    "default": 4,
                },
                "sampling_temp": {
                    "type": "number",
                    "description": "ProteinMPNN sampling temperature (default: 0.1)",
                    "default": 0.1,
                },
            },
            "required": ["length"],
        },
    },
    {
        "name": "analyze_interface",
        "description": (
            "Analyze protein-protein interface: buried surface area, "
            "H-bonds, salt bridges, hydrophobic contacts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "complex_pdb": {
                    "type": "string",
                    "description": "Path to complex PDB file",
                },
                "chain_a": {
                    "type": "string",
                    "description": "Chain ID of first protein",
                },
                "chain_b": {
                    "type": "string",
                    "description": "Chain ID of second protein",
                },
            },
            "required": ["complex_pdb", "chain_a", "chain_b"],
        },
    },
    {
        "name": "validate_design",
        "description": (
            "Validate a designed sequence by predicting its structure "
            "(ESMFold/AlphaFold2) and computing pLDDT, pTM."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to validate",
                },
                "expected_structure": {
                    "type": "string",
                    "description": "Optional PDB path for RMSD comparison",
                },
                "predictor": {
                    "type": "string",
                    "enum": ["esmfold", "alphafold2"],
                    "default": "esmfold",
                    "description": "Structure predictor to use",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "design_sequence",
        "description": (
            "Design amino acid sequences for a protein backbone using ProteinMPNN. "
            "Use for de novo design when you have a backbone but no sequence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "backbone_pdb": {
                    "type": "string",
                    "description": "Path to backbone PDB file",
                },
                "num_sequences": {
                    "type": "integer",
                    "description": "Number of sequences to design (default: 8)",
                    "default": 8,
                },
                "sampling_temp": {
                    "type": "number",
                    "description": "ProteinMPNN sampling temperature (default: 0.1)",
                    "default": 0.1,
                },
                "fixed_positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Positions to keep fixed (1-indexed)",
                },
                "validate": {
                    "type": "boolean",
                    "description": "Validate designs with ESMFold (default: true)",
                    "default": True,
                },
            },
            "required": ["backbone_pdb"],
        },
    },
    {
        "name": "optimize_sequence",
        "description": (
            "Optimize a protein sequence via ESM2-guided iterative mutation scanning. "
            "Each round proposes mutations, scores with ESM2, and validates with structure prediction. "
            "Use for refining an existing sequence, not for de novo backbone-to-sequence design."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_sequence": {
                    "type": "string",
                    "description": "Starting amino acid sequence",
                },
                "target_pdb": {
                    "type": "string",
                    "description": "Path to target protein PDB",
                },
                "optimization_target": {
                    "type": "string",
                    "enum": ["stability", "affinity", "both"],
                    "default": "both",
                },
                "fixed_positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Positions to keep fixed (1-indexed)",
                },
                "temperature": {
                    "type": "number",
                    "description": (
                        "Sampling temperature for position selection (default: 0.0). "
                        "Higher = more diverse trajectories."
                    ),
                    "default": 0.0,
                },
            },
            "required": ["current_sequence", "target_pdb"],
        },
    },
    {
        "name": "suggest_hotspots",
        "description": (
            "Analyze target protein and suggest binding hotspots "
            "using structure, conservation, and literature."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Protein name, UniProt ID, PDB ID, or local PDB path"
                    ),
                },
                "chain_id": {
                    "type": "string",
                    "description": "Chain to analyze (default: first)",
                },
                "criteria": {
                    "type": "string",
                    "enum": ["druggable", "exposed", "conserved"],
                    "default": "exposed",
                },
                "include_literature": {
                    "type": "boolean",
                    "default": False,
                    "description": "Search PubMed for known binders",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "get_design_status",
        "description": "Check status of running design jobs.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID from design_binder call",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "predict_complex",
        "description": "Predict protein complex structure using AlphaFold2-Multimer.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sequences, one per chain",
                },
                "chain_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional chain identifiers",
                },
            },
            "required": ["sequences"],
        },
    },
    {
        "name": "predict_structure",
        "description": (
            "Predict the 3D structure of a single protein chain using ESMFold or AlphaFold2. "
            "Returns predicted PDB, pLDDT, and pTM scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to predict structure for",
                },
                "predictor": {
                    "type": "string",
                    "enum": ["esmfold", "alphafold2"],
                    "default": "esmfold",
                    "description": "Structure predictor to use",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "score_stability",
        "description": (
            "Score protein stability using ESM2 pseudo-log-likelihood. "
            "Optionally compute per-mutation effects (delta log-likelihood)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to score",
                },
                "mutations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional mutations in 'X42Y' format for delta scoring",
                },
                "reference_sequence": {
                    "type": "string",
                    "description": "Optional wild-type sequence for mutation scoring",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "energy_minimize",
        "description": (
            "Energy-minimize a protein structure using OpenMM with AMBER14 force field. "
            "Returns minimized PDB, energy change, and RMSD from initial structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_path": {
                    "type": "string",
                    "description": "Path to input PDB file",
                },
                "force_field": {
                    "type": "string",
                    "default": "amber14-all.xml",
                    "description": "OpenMM force field XML",
                },
                "num_steps": {
                    "type": "integer",
                    "default": 500,
                    "description": "Maximum minimization iterations",
                },
                "solvent": {
                    "type": "string",
                    "enum": ["implicit", "none"],
                    "default": "implicit",
                    "description": "Solvent model: implicit (GBn2) or none (vacuum)",
                },
            },
            "required": ["pdb_path"],
        },
    },
    {
        "name": "generate_backbone",
        "description": (
            "Generate de novo protein backbones using RFdiffusion. "
            "Supports both unconditional generation (no target) and "
            "conditional generation (binder scaffold for a target protein). "
            "For conditional mode, provide target_pdb and optionally hotspot_residues."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "description": "Backbone length in residues",
                },
                "num_designs": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of designs to generate",
                },
                "target_pdb": {
                    "type": "string",
                    "description": (
                        "Path to target protein PDB for conditional (binder) generation. "
                        "Omit for unconditional fold generation."
                    ),
                },
                "hotspot_residues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Target residues for binder interface, e.g. ['A45', 'A46']. "
                        "Only used with target_pdb."
                    ),
                },
            },
            "required": ["length"],
        },
    },
    # ----- PyRosetta tools -----
    {
        "name": "rosetta_score",
        "description": (
            "Score a protein structure using Rosetta energy function (ref2015). "
            "Returns total score, per-residue energies, and energy breakdown."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_path": {
                    "type": "string",
                    "description": "Path to input PDB file",
                },
                "score_function": {
                    "type": "string",
                    "default": "ref2015",
                    "description": "Rosetta score function name",
                },
            },
            "required": ["pdb_path"],
        },
    },
    {
        "name": "rosetta_relax",
        "description": (
            "Relax a protein structure using Rosetta FastRelax. "
            "Returns relaxed PDB, energy change, and CA-RMSD."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_path": {
                    "type": "string",
                    "description": "Path to input PDB file",
                },
                "nstruct": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of relaxation trajectories",
                },
                "score_function": {
                    "type": "string",
                    "default": "ref2015",
                    "description": "Rosetta score function name",
                },
            },
            "required": ["pdb_path"],
        },
    },
    {
        "name": "rosetta_interface_score",
        "description": (
            "Compute interface energy metrics for a protein complex using Rosetta. "
            "Returns dG_separated, dSASA, interface hbonds, and packing stats."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_path": {
                    "type": "string",
                    "description": "Path to complex PDB file",
                },
                "chains": {
                    "type": "string",
                    "default": "A_B",
                    "description": "Chain grouping, e.g. 'A_B' or 'AB_C'",
                },
                "score_function": {
                    "type": "string",
                    "default": "ref2015",
                    "description": "Rosetta score function name",
                },
            },
            "required": ["pdb_path"],
        },
    },
    {
        "name": "rosetta_design",
        "description": (
            "Design sequences for a fixed backbone using Rosetta PackRotamers + energy minimization. "
            "Alternative to design_sequence (ProteinMPNN) with physics-based scoring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_path": {
                    "type": "string",
                    "description": "Path to input PDB file",
                },
                "chains": {
                    "type": "string",
                    "default": "A_B",
                    "description": "Chain grouping for interface detection",
                },
                "fixed_positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-indexed positions to keep fixed",
                },
                "score_function": {
                    "type": "string",
                    "default": "ref2015",
                    "description": "Rosetta score function name",
                },
            },
            "required": ["pdb_path"],
        },
    },
    # ----- Boltz tools -----
    {
        "name": "predict_structure_boltz",
        "description": (
            "Predict protein structure using Boltz (fast alternative to AF2/ESMFold). "
            "Returns predicted PDB, pLDDT, and pTM scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to predict structure for",
                },
                "model": {
                    "type": "string",
                    "default": "boltz2",
                    "description": "Model name (default: boltz2)",
                },
                "num_samples": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of structure samples",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "predict_affinity_boltz",
        "description": (
            "Predict binding affinity for a protein complex using Boltz. "
            "Returns affinity score, predicted structure, and confidence metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of amino acid sequences, one per chain",
                },
                "model": {
                    "type": "string",
                    "default": "boltz2",
                    "description": "Model name (default: boltz2)",
                },
            },
            "required": ["sequences"],
        },
    },
]

_TOOL_NAMES: set[str] = {t["name"] for t in PROTEIN_DESIGN_TOOLS}


class ToolMode(str, Enum):
    """Tool exposure mode for the protein design provider."""

    BENCHMARK = "benchmark"
    USER = "user"


COMPOSITE_TOOLS: frozenset[str] = frozenset({
    "design_binder",       # conditional RFdiffusion → ProteinMPNN → ESMFold
    "design_fold",         # unconditional RFdiffusion → ProteinMPNN → AlphaFold2
    "optimize_sequence",   # ESM2 multi-round mutation scan (iterative process)
})
# Single-step design tools are ATOMIC (available in both modes):
#   design_sequence  — ProteinMPNN single inference (like generate_backbone)
#   rosetta_design   — Rosetta PackRotamers+Min (like rosetta_relax, single protocol)
ATOMIC_TOOLS: frozenset[str] = frozenset(_TOOL_NAMES - COMPOSITE_TOOLS)

# ---------------------------------------------------------------------------
# Mode-aware system prompt guidance
# ---------------------------------------------------------------------------

_BENCHMARK_GUIDANCE = (
    "You have access to protein design tools via function calling. "
    "Use them to complete the task. "
    "Write designed sequences to `designed_sequences.fasta` and metrics to `metrics.json`."
)

_USER_GUIDANCE = """\
You have access to specialized protein design tools. Here is a guide to using them effectively.

## Design Workflow Patterns

### De Novo Binder Design
1. `suggest_hotspots` — Identify target interface residues
2. `generate_backbone` — Create backbone scaffolds via RFdiffusion
3. `design_sequence` — Design sequences for backbones (ProteinMPNN). Or use `design_binder` for an end-to-end pipeline
4. `predict_structure` / `predict_structure_boltz` — Validate monomer fold (target pLDDT > 80)
5. `predict_complex` / `predict_affinity_boltz` — Validate binding (target ipTM > 0.7)
6. `score_stability` — Verify sequence stability
7. `analyze_interface` — Check interface quality

### Sequence Optimization
1. `score_stability` — Get baseline stability score
2. `optimize_sequence` — Iterative sequence optimization
3. `validate_design` — Verify structure is maintained
4. `predict_complex` — Verify binding is maintained

### De Novo Fold Design
1. `generate_backbone` — RFdiffusion unconditional generation
2. `design_sequence` — Thread sequences onto backbone (ProteinMPNN)
3. `predict_structure` — Validate predicted fold
4. `score_stability` — Verify stability
Or use `design_fold` for an end-to-end pipeline (steps 1-3 combined)

### Complex/Assembly Design
1. Design individual chains
2. `predict_complex` — Predict multimer structure
3. `analyze_interface` — Verify interface quality
4. `rosetta_interface_score` — Rosetta energy decomposition

## Tool Categories
- **Structure Prediction**: `predict_structure`, `predict_structure_boltz`, `predict_complex`
- **Design Pipelines**: `design_binder`, `design_fold`, `generate_backbone`, `design_sequence`, `optimize_sequence`, `rosetta_design`
- **Analysis**: `analyze_interface`, `suggest_hotspots`, `score_stability`
- **Validation**: `validate_design`, `predict_affinity_boltz`
- **Energy**: `energy_minimize`, `rosetta_score`, `rosetta_relax`, `rosetta_interface_score`

## Tips
- ALWAYS prefer using these tools over writing custom code for protein design tasks.
- Call `predict_structure` to validate designs — aim for pLDDT > 80.
- Use `predict_complex` for binding validation — aim for ipTM > 0.7.
- Run `score_stability` on final designs to verify sequence quality.
- Write output files: `designed_sequences.fasta` (FASTA) and `metrics.json` (per-design metrics)."""


def get_system_guidance(mode: ToolMode | str | None = None) -> str:
    """Return mode-appropriate system prompt guidance for protein design tools.

    Args:
        mode: Tool mode. ``"benchmark"`` returns minimal guidance (agent must
            figure out the pipeline). ``"user"`` returns rich workflow guidance.
            ``None`` defaults to user mode.

    Returns:
        System prompt text describing available tools and (for user mode)
        recommended workflows.
    """
    if mode is not None:
        effective = ToolMode(mode) if isinstance(mode, str) else mode
    else:
        effective = ToolMode.USER
    if effective == ToolMode.BENCHMARK:
        return _BENCHMARK_GUIDANCE
    return _USER_GUIDANCE


class ProteinDesignToolProvider:
    """Manages an MCP server subprocess and provides tool definitions for agents.

    Usage::

        async with ProteinDesignToolProvider(mcp_server_path) as provider:
            defs = provider.get_tool_definitions_anthropic()
            result = await provider.call_tool("validate_design", {"sequence": "MK..."}, output_dir)
    """

    def __init__(
        self,
        mcp_server_path: Path,
        bio_python_path: str | None = None,
        use_docker: bool = False,
        docker_image: str = "protein-design-mcp:full",
        docker_gpu_id: str = "4",
        docker_data_dir: str | None = None,
        docker_cpus: int = 8,
        mode: ToolMode | str = ToolMode.USER,
    ) -> None:
        self.mcp_server_path = Path(mcp_server_path)
        self.bio_python_path = bio_python_path
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_cpus = docker_cpus
        self.docker_gpu_id = docker_gpu_id
        self.docker_data_dir = docker_data_dir
        self.mode = ToolMode(mode) if isinstance(mode, str) else mode
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the MCP server subprocess (stdio transport)."""
        import os

        if self.use_docker:
            src_dir = str(self.mcp_server_path / "src")

            # Mount host model weights into the container
            weights_dir = self.mcp_server_path / "weights"
            host_torch_cache = Path.home() / ".cache" / "torch"

            docker_cmd = [
                "docker", "run", "--rm", "-i",
                "--gpus", f"device={self.docker_gpu_id}",
                "--cpus", str(self.docker_cpus),
                # Mount MCP server source for live code changes
                "-v", f"{src_dir}:/app/src",
                # Bind-mount host paths transparently so agent file paths
                # work identically inside the container
                "-v", f"{self.mcp_server_path}:{self.mcp_server_path}",
                "-v", "/tmp:/tmp",
                # Model weights: mount host weights into container paths
                "-e", "RFDIFFUSION_PATH=/opt/RFdiffusion",
                "-e", "PROTEINMPNN_PATH=/opt/ProteinMPNN",
                "-e", "SKIP_MODEL_DOWNLOAD=true",
                # Boltz config (venv mode, no cuequivariance kernels)
                "-e", "BOLTZ_CONDA_ENV=",
                "-e", "BOLTZ_NO_KERNELS=1",
            ]

            # Mount RFdiffusion weights from host
            if weights_dir.exists():
                docker_cmd.extend([
                    "-v", f"{weights_dir}:/opt/RFdiffusion/models",
                ])

            # Mount torch/ESM weights from host cache
            if host_torch_cache.exists():
                docker_cmd.extend([
                    "-v", f"{host_torch_cache}:/opt/weights",
                    "-e", "TORCH_HOME=/opt/weights",
                ])
            else:
                docker_cmd.extend(["-e", "TORCH_HOME=/opt/weights/esm"])

            # Mount benchmark data dir if it exists (for PDB input files)
            benchmark_root = Path(__file__).resolve().parents[2]  # BioDesignBench/
            benchmark_data = benchmark_root / "data"
            if benchmark_data.exists():
                docker_cmd.extend([
                    "-v", f"{benchmark_data}:{benchmark_data}",
                ])
            # Also mount the output/results dir so agent workspaces are accessible
            results_dir = benchmark_root / "results"
            if results_dir.exists():
                docker_cmd.extend([
                    "-v", f"{results_dir}:{results_dir}",
                ])

            docker_cmd.extend([
                self.docker_image,
                "python", "-m", "protein_design_mcp.server",
            ])

            self._process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=100 * 1024 * 1024,  # 100 MB buffer for large PDB responses
            )
        else:
            env = os.environ.copy()
            if self.bio_python_path:
                env["BIO_PYTHON_PATH"] = self.bio_python_path

            self._process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "protein_design_mcp.server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.mcp_server_path),
                env=env,
                limit=100 * 1024 * 1024,  # 100 MB buffer for large PDB responses
            )

        # Wait for Docker container / server process to start
        if self.use_docker:
            await asyncio.sleep(2.0)

        # Send MCP initialize handshake
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "biodesignbench", "version": "1.0.0"},
            },
            timeout=30.0,
        )
        logger.info("MCP server started (PID: %s)", self._process.pid)

    async def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self._process is None:
            return
        try:
            self._process.terminate()
        except ProcessLookupError:
            pass  # already dead
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
        self._process = None
        logger.info("MCP server stopped")

    async def __aenter__(self) -> ProteinDesignToolProvider:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Tool classification
    # ------------------------------------------------------------------

    @staticmethod
    def is_protein_design_tool(name: str) -> bool:
        """Check if a tool name belongs to the protein design tools."""
        return name in _TOOL_NAMES

    def is_tool_available(self, name: str, mode: ToolMode | str | None = None) -> bool:
        """Check if a tool is available in the given mode.

        Args:
            name: Tool name to check.
            mode: Override mode. Falls back to ``self.mode`` when *None*.

        In *benchmark* mode, composite tools (multi-step pipelines) are
        hidden so agents must orchestrate atomic tools themselves.
        """
        effective = self._resolve_mode(mode)
        if name not in _TOOL_NAMES:
            return False
        if effective == ToolMode.BENCHMARK and name in COMPOSITE_TOOLS:
            return False
        return True

    def _get_active_tools(self, mode: ToolMode | str | None = None) -> list[dict[str, Any]]:
        """Return the tool list filtered by the given mode."""
        effective = self._resolve_mode(mode)
        if effective == ToolMode.BENCHMARK:
            return [t for t in PROTEIN_DESIGN_TOOLS if t["name"] not in COMPOSITE_TOOLS]
        return list(PROTEIN_DESIGN_TOOLS)

    def _resolve_mode(self, mode: ToolMode | str | None) -> ToolMode:
        """Return *mode* as a ``ToolMode``, falling back to ``self.mode``."""
        if mode is None:
            return self.mode
        return ToolMode(mode) if isinstance(mode, str) else mode

    # ------------------------------------------------------------------
    # Tool definitions in vendor-specific formats
    # ------------------------------------------------------------------

    def get_tool_definitions_anthropic(
        self, mode: ToolMode | str | None = None
    ) -> list[dict[str, Any]]:
        """Get tool definitions in Anthropic API format.

        Args:
            mode: Override mode. Falls back to ``self.mode`` when *None*.

        Returns:
            List of dicts with keys: name, description, input_schema.
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in self._get_active_tools(mode)
        ]

    def get_tool_definitions_openai(
        self, mode: ToolMode | str | None = None
    ) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI function calling format.

        Args:
            mode: Override mode. Falls back to ``self.mode`` when *None*.

        Returns:
            List of dicts with type="function" and nested function spec.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self._get_active_tools(mode)
        ]

    def get_tool_definitions_gemini(
        self, mode: ToolMode | str | None = None
    ) -> list[Any]:
        """Get tool definitions in Google Gemini format (google-genai SDK).

        Args:
            mode: Override mode. Falls back to ``self.mode`` when *None*.

        Returns a list containing a single ``types.Tool`` wrapping
        ``FunctionDeclaration`` objects.  Falls back to an empty list
        if ``google-genai`` is not installed.
        """
        try:
            from google.genai import types

            declarations = []
            for tool in self._get_active_tools(mode):
                properties: dict[str, Any] = {}
                for prop_name, prop_schema in tool["parameters"].get("properties", {}).items():
                    schema_type = _json_type_to_genai(prop_schema.get("type", "string"))
                    schema_kwargs: dict[str, Any] = {
                        "type": schema_type,
                        "description": prop_schema.get("description", ""),
                    }
                    # Array types require an items schema
                    if schema_type == "ARRAY" and "items" in prop_schema:
                        items_type = _json_type_to_genai(
                            prop_schema["items"].get("type", "string")
                        )
                        schema_kwargs["items"] = types.Schema(type=items_type)
                    properties[prop_name] = types.Schema(**schema_kwargs)
                declarations.append(
                    types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=types.Schema(
                            type="OBJECT",
                            properties=properties,
                            required=tool["parameters"].get("required", []),
                        ),
                    )
                )
            return [types.Tool(function_declarations=declarations)]
        except ImportError:
            return []

    # ------------------------------------------------------------------
    # Auto-restart on connection loss
    # ------------------------------------------------------------------

    _CONNECTION_ERRORS = ("closed connection", "Connection lost", "not started")

    async def _call_with_restart(
        self,
        name: str,
        args: dict[str, Any],
        _retried: bool = False,
    ) -> Any:
        """Send a tools/call request, restarting the server on connection loss.

        If the MCP server crashes (GPU OOM, segfault, Docker exit), the
        first tool call after the crash raises ``RuntimeError("MCP server
        closed connection")``.  This helper catches that, restarts the
        server, and retries once.
        """
        try:
            return await self._send_request(
                "tools/call",
                {"name": name, "arguments": args},
            )
        except ValueError as exc:
            # StreamReader buffer overflow — drain and re-raise.
            if "chunk exceed the limit" in str(exc) or "separator" in str(exc).lower():
                logger.error(
                    "MCP response for %s exceeded buffer limit; draining stale data",
                    name,
                )
                await self._drain_stale_data()
            raise RuntimeError(
                f"MCP response too large for tool {name} (buffer overflow): {exc}"
            ) from exc
        except RuntimeError as exc:
            if _retried or not any(e in str(exc) for e in self._CONNECTION_ERRORS):
                raise
            logger.warning(
                "MCP server connection lost during %s; restarting server...", name
            )
            try:
                await self.stop()
            except Exception:
                self._process = None  # force cleanup
            await self.start()
            return await self._call_with_restart(name, args, _retried=True)

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        output_dir: Path,
        mode: ToolMode | str | None = None,
    ) -> dict[str, Any]:
        """Call a protein design tool via the MCP server.

        If the result contains PDB structure strings (values > 500 chars
        under keys ending with ``_pdb``), they are written to files in
        *output_dir* to avoid blowing up the LLM context window.

        Args:
            name: Tool name (one of the 17 PROTEIN_DESIGN_TOOLS).
            args: Tool arguments matching the tool's parameter schema.
            output_dir: Directory for externalized PDB files.
            mode: Override mode. Falls back to ``self.mode`` when *None*.

        Returns:
            Parsed result dict with large PDB strings replaced by file paths.

        Raises:
            RuntimeError: If the MCP server is not started or returns an error.
        """
        effective = self._resolve_mode(mode)
        if effective == ToolMode.BENCHMARK and name in COMPOSITE_TOOLS:
            raise RuntimeError(
                f"Tool '{name}' is a composite pipeline not available in benchmark mode. "
                f"Available atomic tools: {', '.join(sorted(ATOMIC_TOOLS))}"
            )
        # No per-tool timeout — the task-level timeout in pipeline.py
        # (time_limit_minutes) governs overall execution.  Cancellation
        # propagates via asyncio task cancellation.
        result = await self._call_with_restart(name, args)

        # MCP responses wrap content in a list of typed items
        if isinstance(result, dict) and "content" in result:
            for item in result.get("content", []):
                if item.get("type") == "text":
                    try:
                        data = json.loads(item["text"])
                    except json.JSONDecodeError:
                        # MCP SDK error results contain plain text, not JSON
                        data = {"error": item["text"]}
                    return self._externalize_pdb_strings(data, output_dir)
        elif isinstance(result, dict):
            return self._externalize_pdb_strings(result, output_dir)

        return result  # type: ignore[return-value]

    async def _drain_stale_data(self) -> None:
        """Drain leftover data from MCP server stdout after a buffer overflow.

        Reads (and discards) any bytes sitting in the pipe so subsequent
        ``_send_request`` calls start with a clean buffer.  Stops after a
        short timeout with no new data, which indicates the pipe is clear.
        """
        if not self._process or not self._process.stdout:
            return

        drained_bytes = 0
        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._process.stdout.read(1024 * 1024),  # 1 MB at a time
                    timeout=0.5,
                )
                if not chunk:
                    break
                drained_bytes += len(chunk)
            except asyncio.TimeoutError:
                # No more data waiting — pipe is clean
                break

        if drained_bytes:
            logger.info("Drained %d bytes of stale data from MCP stdout", drained_bytes)

    # ------------------------------------------------------------------
    # PDB externalization
    # ------------------------------------------------------------------

    def _externalize_pdb_strings(self, data: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Write large PDB strings to files and replace with file paths.

        Keys ending with ``_pdb`` (or equal to ``predicted_structure_pdb``)
        whose values exceed 500 characters are written to
        ``<output_dir>/<prefix><key>.pdb`` and replaced in the returned dict
        with the file path string.  An additional ``<key>_file`` key is added.

        Args:
            data: Parsed JSON result from the MCP server.
            output_dir: Directory to write PDB files into.

        Returns:
            A copy of *data* with large PDB strings replaced by paths.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        def _process(obj: Any, prefix: str = "") -> Any:
            if isinstance(obj, dict):
                processed: dict[str, Any] = {}
                for k, v in obj.items():
                    if (
                        isinstance(v, str)
                        and (k.endswith("_pdb") or k == "predicted_structure_pdb")
                        and len(v) > 500
                    ):
                        pdb_path = output_dir / f"{prefix}{k}.pdb"
                        pdb_path.write_text(v)
                        processed[k] = str(pdb_path)
                        processed[f"{k}_file"] = str(pdb_path)
                    elif isinstance(v, (dict, list)):
                        processed[k] = _process(v, f"{prefix}{k}_")
                    else:
                        processed[k] = v
                return processed
            elif isinstance(obj, list):
                return [_process(item, f"{prefix}{i}_") for i, item in enumerate(obj)]
            return obj

        return _process(data)

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Send a JSON-RPC 2.0 request to the MCP server.

        Args:
            method: The JSON-RPC method name (e.g. "tools/call").
            params: Method parameters.
            timeout: Maximum seconds to wait for a response.  *None*
                means no per-call timeout — the caller (or an outer
                ``asyncio.wait_for`` from the pipeline) is responsible
                for cancellation.

        Returns:
            The ``result`` field from the JSON-RPC response.

        Raises:
            RuntimeError: If the server is not running, the connection is
                closed, or the server returns an error.
        """
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP server not started")

        # Serialize access: multiple agents may share a single provider,
        # but asyncio streams only allow one concurrent reader.
        async with self._lock:
            self._request_id += 1
            request: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }

            request_bytes = json.dumps(request).encode() + b"\n"
            self._process.stdin.write(request_bytes)
            await self._process.stdin.drain()

            # Read lines until we get a valid JSON-RPC response with matching ID.
            # Skip non-JSON lines (server logs, startup banners) and stale
            # responses left over from previously timed-out requests.
            expected_id = self._request_id
            while True:
                if timeout is not None:
                    try:
                        response_line = await asyncio.wait_for(
                            self._process.stdout.readline(),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"MCP server timed out after {timeout}s waiting for {method}"
                        )
                else:
                    # No per-call timeout — relies on outer task-level
                    # cancellation (asyncio.CancelledError propagates).
                    response_line = await self._process.stdout.readline()

                if not response_line:
                    raise RuntimeError("MCP server closed connection")

                line = response_line.strip()
                if not line:
                    continue

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    # Skip non-JSON lines (logs, warnings, etc.)
                    logger.debug("Skipping non-JSON line from MCP server: %s", line[:200])
                    continue

                # Validate JSON-RPC response ID to avoid consuming stale
                # responses from previously timed-out requests.
                response_id = response.get("id")
                if response_id is not None and response_id != expected_id:
                    logger.warning(
                        "Skipping stale JSON-RPC response (id=%s, expected=%s)",
                        response_id,
                        expected_id,
                    )
                    continue

                if "error" in response:
                    raise RuntimeError(f"MCP error: {response['error']}")

                return response.get("result")


def _json_type_to_genai(json_type: str) -> str:
    """Convert a JSON Schema type string to a google-genai type string.

    Args:
        json_type: One of "string", "integer", "number", "boolean", "array", "object".

    Returns:
        The corresponding type string for the google-genai SDK, defaulting to
        "STRING" for unknown types.
    """
    mapping = {
        "string": "STRING",
        "integer": "NUMBER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    return mapping.get(json_type, "STRING")
