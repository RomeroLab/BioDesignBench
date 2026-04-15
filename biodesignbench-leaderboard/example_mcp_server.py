"""Minimal reference implementation of the BioDesignBench MCP contract.

Submitters who want to evaluate their own tool implementations against
the BioDesignBench leaderboard can fork this file, plug in their own
logic for each tool, deploy it behind a public HTTPS URL, and paste
that URL into the submission form under "Advanced: Custom MCP".

The leaderboard's agent loop will POST each tool invocation to your
endpoint and treat the JSON response as the tool result. The MCP never
sees the raw task description or evaluation criteria — only the
operational arguments the agent chooses to pass (a protein sequence,
a PDB path, a set of hotspot residues, etc.).

Contract:

    POST <your-url>/
    Authorization: Bearer <optional shared token>
    Content-Type: application/json

    {
      "name": "<tool_name>",          # one of the 17 tool names
      "arguments": { ... }            # per-tool JSON object
    }

    Response: arbitrary JSON object describing the tool output.
    Errors should be reported in a top-level "error" field rather
    than via HTTP status codes.

Install:
    pip install fastapi uvicorn

Run (locally, with ngrok for a public URL):
    uvicorn example_mcp_server:app --host 0.0.0.0 --port 8000
    ngrok http 8000

Deploy (Modal example):
    modal deploy example_mcp_server.py  # see bdb-boltz as a template

Tool name list (17 total) — the full JSON Schema for each lives in
`mcp_tool_schemas.json` in this repo:

    design_binder, analyze_interface, validate_design, optimize_sequence,
    suggest_hotspots, get_design_status, predict_complex, predict_structure,
    score_stability, energy_minimize, generate_backbone, rosetta_score,
    rosetta_relax, rosetta_interface_score, rosetta_design,
    predict_structure_boltz, predict_affinity_boltz
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BioDesignBench MCP (example stub)")

SHARED_TOKEN = os.environ.get("BDB_MCP_TOKEN", "")


class MCPRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


# ---------------------------------------------------------------------------
#  Tool handlers (REPLACE THESE STUBS WITH YOUR ACTUAL IMPLEMENTATIONS)
# ---------------------------------------------------------------------------


def handle_predict_structure(args: dict) -> dict:
    """Predict the structure of a single protein sequence.

    Real implementations would call AlphaFold2, ESMFold, Boltz, etc.
    """
    sequence = args.get("sequence") or ""
    if not sequence:
        return {"error": "predict_structure requires a 'sequence' argument"}
    # TODO: replace this stub with your structure predictor.
    return {
        "pdb": ">dummy\nATOM ...",
        "pLDDT": 0.0,
        "pTM": 0.0,
        "note": "stub implementation -- replace with your predictor",
    }


def handle_design_binder(args: dict) -> dict:
    """Design a protein binder against a target. Real implementations
    would call RFdiffusion followed by ProteinMPNN and AlphaFold2."""
    return {"error": "design_binder not implemented in this stub"}


def handle_score_stability(args: dict) -> dict:
    """Score single-point stability. Real implementations might call
    Rosetta, DDG_predictor, or a learned model."""
    return {"error": "score_stability not implemented in this stub"}


TOOL_HANDLERS = {
    "predict_structure": handle_predict_structure,
    "design_binder": handle_design_binder,
    "score_stability": handle_score_stability,
    # ... add handlers for the other 14 tools here
}


# ---------------------------------------------------------------------------
#  Dispatcher
# ---------------------------------------------------------------------------


@app.post("/")
def call_tool(
    req: MCPRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    if SHARED_TOKEN:
        bearer = (authorization or "").removeprefix("Bearer ").strip()
        if bearer != SHARED_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

    handler = TOOL_HANDLERS.get(req.name)
    if handler is None:
        return {
            "error": f"Unknown tool: {req.name}",
            "available": sorted(TOOL_HANDLERS.keys()),
        }

    try:
        return handler(req.arguments)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": req.name}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "implemented_tools": sorted(TOOL_HANDLERS.keys()),
        "note": (
            "This is the reference stub. Replace the handle_* functions "
            "with your actual tool implementations before submitting to "
            "the BioDesignBench leaderboard."
        ),
    }
