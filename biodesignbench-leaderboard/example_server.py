"""Reference FastAPI server for BioDesignBench submitters.

This example shows how to implement the API endpoint that BioDesignBench
will call during benchmarking. Replace the mock agent logic with your
actual LLM agent + MCP tool pipeline.

Usage:
    pip install fastapi uvicorn
    python example_server.py

    # Or with uvicorn directly:
    uvicorn example_server:app --host 0.0.0.0 --port 8000

Your endpoint will receive POST requests at /api/run with the task payload.

Task Payload Format:
    {
        "task_id": "dnb_sig_001",
        "task_description": "Design a de novo binder for...",
        "available_tools": [... 17 tool schemas ...],
        "input_files": {"7n1j.pdb": "<base64>"},
        "design_constraints": {"length_range": [80, 150], "max_designs": 10},
        "max_steps": 50,
        "timeout_sec": 300
    }

Expected Response Format:
    {
        "sequences": ["MKKL...", "MFQR..."],
        "run_log": [{"step": 1, "tool": "suggest_hotspots", "success": true}, ...],
        "total_steps": 12,
        "total_time_sec": 142.5,
        "metrics": {}
    }
"""

from __future__ import annotations

import base64
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="BioDesignBench Example Agent",
    description="Reference implementation for benchmark submission",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
#  Request/Response models
# ---------------------------------------------------------------------------


class TaskPayload(BaseModel):
    task_id: str
    task_description: str
    available_tools: list[dict[str, Any]] = []
    input_files: dict[str, str] = {}  # filename -> base64 data
    design_constraints: dict[str, Any] = {}
    max_steps: int = 50
    timeout_sec: int = 300


class AgentResponse(BaseModel):
    sequences: list[str]
    run_log: list[dict[str, Any]]
    total_steps: int
    total_time_sec: float
    metrics: dict[str, Any] = {}


# ---------------------------------------------------------------------------
#  Mock agent (replace with your real agent)
# ---------------------------------------------------------------------------

# Standard amino acids for mock sequence generation
_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _generate_mock_sequence(length: int) -> str:
    """Generate a random protein sequence with reasonable composition."""
    # Weight towards common amino acids
    weights = [
        7, 2, 5, 6, 4, 7, 2, 5, 6, 9,  # A C D E F G H I K L
        2, 4, 5, 4, 5, 7, 6, 7, 1, 3,   # M N P Q R S T V W Y
    ]
    return "".join(random.choices(_AAS, weights=weights, k=length))


def mock_agent(payload: TaskPayload) -> AgentResponse:
    """Mock agent that generates random but valid designs.

    Replace this with your actual LLM agent + MCP tool pipeline.
    This mock demonstrates the expected response format.
    """
    start = time.monotonic()

    # Determine design parameters
    constraints = payload.design_constraints
    length_range = constraints.get("length_range", [80, 150])
    max_designs = constraints.get("max_designs", 10)
    num_designs = min(max_designs, 5)  # Generate 5 for this mock

    # "Decode" input PDB files (in a real agent, you'd use these)
    for filename, b64_data in payload.input_files.items():
        pdb_bytes = base64.b64decode(b64_data)
        # In a real agent: save to temp file and pass to MCP tools

    # Simulate a multi-step design pipeline
    run_log = [
        {
            "step": 1,
            "tool": "suggest_hotspots",
            "success": True,
            "args_summary": {"target": "from_pdb"},
            "output_summary": "Found 5 hotspot residues",
        },
        {
            "step": 2,
            "tool": "generate_backbone",
            "success": True,
            "args_summary": {"length": length_range[0]},
            "output_summary": f"Generated {num_designs} backbones",
        },
        {
            "step": 3,
            "tool": "optimize_sequence",
            "success": True,
            "args_summary": {"optimization_target": "both"},
            "output_summary": f"Optimized {num_designs} sequences",
        },
        {
            "step": 4,
            "tool": "predict_structure",
            "success": True,
            "args_summary": {"predictor": "esmfold"},
            "output_summary": "Predicted structures for all designs",
        },
        {
            "step": 5,
            "tool": "validate_design",
            "success": True,
            "args_summary": {},
            "output_summary": "Validated all designs",
        },
    ]

    # Generate mock sequences
    min_len, max_len = length_range
    sequences = [
        _generate_mock_sequence(random.randint(min_len, max_len))
        for _ in range(num_designs)
    ]

    elapsed = time.monotonic() - start

    return AgentResponse(
        sequences=sequences,
        run_log=run_log,
        total_steps=len(run_log),
        total_time_sec=round(elapsed, 2),
        metrics={},  # Agent-reported metrics (optional)
    )


# ---------------------------------------------------------------------------
#  API endpoint
# ---------------------------------------------------------------------------


@app.post("/api/run", response_model=AgentResponse)
async def run_task(payload: TaskPayload) -> AgentResponse:
    """Run a single benchmark task.

    This is the endpoint that BioDesignBench will POST to during benchmarking.
    Replace mock_agent() with your actual agent logic.
    """
    return mock_agent(payload)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "agent": "example-mock"}


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("Starting BioDesignBench example server...")
    print("POST endpoint: http://localhost:8000/api/run")
    print("Health check:  http://localhost:8000/health")
    print()
    print("Replace mock_agent() with your real agent logic.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
