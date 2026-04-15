"""In-process agent runner for contamination-safe BioDesignBench submissions.

Replaces the old endpoint-based dispatcher: the HF Space leaderboard
runs the agent loop here, using:

  - the submitter's LLM provider + API key (passed at submission time,
    never persisted -- forwarded straight to the provider SDK and
    discarded after the run)
  - either our reference protein-design-mcp endpoint or a custom
    submitter-provided MCP URL (the latter is opt-in, off by default)

This keeps the 76 task descriptions inside Romero Lab infrastructure;
the only path data leaves the lab is via the submitter's chosen LLM
provider API call. The MCP server (custom or reference) sees only
operational tool arguments -- never the raw task prompt or evaluation
criteria.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 50
TOOL_HTTP_TIMEOUT = 600  # per-tool-call timeout (seconds)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert computational protein engineer participating in "
    "the BioDesignBench evaluation. Your goal is to design protein "
    "sequences that satisfy the user's task description by orchestrating "
    "the available protein-design tools. Iterate: generate candidates, "
    "evaluate them with multiple metrics (structure prediction, energy, "
    "interface analysis), refine, and only output your final designs "
    "after thorough validation. When you are done, end your final "
    "message with a fasta-style block containing the designed sequences:\n"
    "```fasta\n>design_1\nMKKL...\n>design_2\nMFQR...\n```"
)


# ---------------------------------------------------------------------------
#  Tool-schema loading
# ---------------------------------------------------------------------------


def load_tool_schemas() -> list[dict]:
    """Load the 17 reference MCP tool schemas."""
    p = Path(__file__).parent / "mcp_tool_schemas.json"
    with open(p) as f:
        return json.load(f)


def to_anthropic_tools(schemas: list[dict]) -> list[dict]:
    """Convert leaderboard tool schemas to Anthropic's `tools` format."""
    out = []
    for s in schemas:
        out.append({
            "name": s["name"],
            "description": s.get("description", ""),
            "input_schema": s.get("parameters") or s.get("input_schema") or {},
        })
    return out


def to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Convert to OpenAI's `tools` format (also used by DeepSeek)."""
    out = []
    for s in schemas:
        out.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters") or s.get("input_schema") or {},
            },
        })
    return out


# ---------------------------------------------------------------------------
#  MCP HTTP client (one tool call per POST)
# ---------------------------------------------------------------------------


def call_mcp_tool(
    mcp_url: str,
    mcp_token: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """POST a single tool call to the MCP server.

    The server contract mirrors `protein-design-mcp/deploy/modal_app.py`:
        POST /  body: {"name": "<tool>", "arguments": {...}}
        200 OK  body: {...tool result...}
    """
    if not mcp_url:
        return {"error": "MCP endpoint not configured (PROTEIN_MCP_URL unset)"}

    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed in leaderboard image"}

    headers = {"Content-Type": "application/json"}
    if mcp_token:
        headers["Authorization"] = f"Bearer {mcp_token}"

    payload = {"name": tool_name, "arguments": arguments}

    try:
        resp = httpx.post(
            mcp_url, json=payload, headers=headers, timeout=TOOL_HTTP_TIMEOUT,
        )
    except Exception as e:
        return {"error": f"MCP POST failed: {e}"}

    if resp.status_code != 200:
        return {"error": f"MCP HTTP {resp.status_code}: {resp.text[:300]}"}

    try:
        return resp.json()
    except Exception as e:
        return {"error": f"MCP returned non-JSON: {e}"}


# ---------------------------------------------------------------------------
#  Sequence extraction from agent's final answer
# ---------------------------------------------------------------------------


_FASTA_BLOCK = re.compile(r"```fasta\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_FASTA_HEADER = re.compile(r"^>\S+", re.MULTILINE)
_AA_LINE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY*\-]+$", re.MULTILINE)
_AA_INLINE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY]{20,})\b")


def extract_sequences(text: str, max_designs: int = 50) -> list[str]:
    """Pull amino acid sequences out of the agent's final assistant text.

    Looks for fenced fasta blocks first; falls back to inline AA strings
    of length >= 20.
    """
    if not text:
        return []

    seqs: list[str] = []

    for block in _FASTA_BLOCK.findall(text):
        # Split on header lines and concatenate body lines per record
        records: list[list[str]] = []
        current: list[str] | None = None
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    records.append(current)
                current = []
                continue
            if current is None:
                current = []
            current.append(line)
        if current:
            records.append(current)
        for rec in records:
            joined = "".join(rec).replace(" ", "").replace("-", "").replace("*", "")
            if len(joined) >= 20 and set(joined) <= set("ACDEFGHIKLMNPQRSTVWY"):
                seqs.append(joined)

    if not seqs:
        for m in _AA_INLINE.finditer(text):
            seqs.append(m.group(1))

    # Dedup while preserving order
    seen = set()
    deduped = []
    for s in seqs:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
        if len(deduped) >= max_designs:
            break
    return deduped


# ---------------------------------------------------------------------------
#  Anthropic agent loop
# ---------------------------------------------------------------------------


def _run_anthropic(
    api_key: str,
    model: str,
    system: str,
    user: str,
    tool_schemas: list[dict],
    mcp_url: str,
    mcp_token: str,
    max_steps: int,
) -> tuple[list[dict], str]:
    """Anthropic Claude tool-calling loop."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    tools = to_anthropic_tools(tool_schemas)

    messages: list[dict] = [{"role": "user", "content": user}]
    run_log: list[dict] = []
    final_text = ""

    for step in range(max_steps):
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=messages,
            tools=tools,
        )

        # Append assistant turn (Anthropic content blocks are passed back as-is)
        messages.append(
            {"role": "assistant",
             "content": [block.model_dump() for block in resp.content]}
        )

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        for b in resp.content:
            if b.type == "text":
                final_text = b.text

        if not tool_uses or resp.stop_reason == "end_turn":
            break

        tool_results = []
        for tu in tool_uses:
            t0 = time.monotonic()
            result = call_mcp_tool(mcp_url, mcp_token, tu.name, tu.input)
            dt = round(time.monotonic() - t0, 2)
            run_log.append({
                "step": step,
                "tool": tu.name,
                "arguments": tu.input,
                "success": "error" not in result,
                "latency_sec": dt,
                "result_summary": str(result)[:500],
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result)[:8000],
            })

        messages.append({"role": "user", "content": tool_results})

    return run_log, final_text


# ---------------------------------------------------------------------------
#  OpenAI / DeepSeek agent loop (DeepSeek uses the openai-compatible API)
# ---------------------------------------------------------------------------


def _run_openai_compat(
    api_key: str,
    model: str,
    system: str,
    user: str,
    tool_schemas: list[dict],
    mcp_url: str,
    mcp_token: str,
    max_steps: int,
    base_url: str | None = None,
) -> tuple[list[dict], str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    tools = to_openai_tools(tool_schemas)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    run_log: list[dict] = []
    final_text = ""

    token_param = (
        "max_completion_tokens" if any(p in model for p in ("gpt-5", "o3", "o4")) else "max_tokens"
    )

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **{token_param: 8192},
        )
        msg = resp.choices[0].message

        # Append assistant turn
        assistant_dict: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_dict["content"] = msg.content
            final_text = msg.content
        if msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_dict)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            t0 = time.monotonic()
            result = call_mcp_tool(mcp_url, mcp_token, tc.function.name, args)
            dt = round(time.monotonic() - t0, 2)
            run_log.append({
                "step": step,
                "tool": tc.function.name,
                "arguments": args,
                "success": "error" not in result,
                "latency_sec": dt,
                "result_summary": str(result)[:500],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)[:8000],
            })

    return run_log, final_text


# ---------------------------------------------------------------------------
#  Google Gemini agent loop
# ---------------------------------------------------------------------------


def _run_google(
    api_key: str,
    model: str,
    system: str,
    user: str,
    tool_schemas: list[dict],
    mcp_url: str,
    mcp_token: str,
    max_steps: int,
) -> tuple[list[dict], str]:
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)

    function_decls = []
    for s in tool_schemas:
        params = s.get("parameters") or {}
        function_decls.append(
            gtypes.FunctionDeclaration(
                name=s["name"],
                description=s.get("description", ""),
                parameters=params,
            )
        )
    tools = [gtypes.Tool(function_declarations=function_decls)]

    history = [gtypes.Content(role="user", parts=[gtypes.Part(text=f"{system}\n\n{user}")])]
    run_log: list[dict] = []
    final_text = ""

    for step in range(max_steps):
        resp = client.models.generate_content(
            model=model,
            contents=history,
            config=gtypes.GenerateContentConfig(tools=tools),
        )
        cand = resp.candidates[0]
        history.append(cand.content)

        function_calls = []
        for part in cand.content.parts:
            if getattr(part, "function_call", None):
                function_calls.append(part.function_call)
            elif getattr(part, "text", None):
                final_text = part.text

        if not function_calls:
            break

        function_responses = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            t0 = time.monotonic()
            result = call_mcp_tool(mcp_url, mcp_token, fc.name, args)
            dt = round(time.monotonic() - t0, 2)
            run_log.append({
                "step": step, "tool": fc.name, "arguments": args,
                "success": "error" not in result, "latency_sec": dt,
                "result_summary": str(result)[:500],
            })
            function_responses.append(
                gtypes.Part(function_response=gtypes.FunctionResponse(
                    name=fc.name, response=result,
                ))
            )

        history.append(gtypes.Content(role="user", parts=function_responses))

    return run_log, final_text


# ---------------------------------------------------------------------------
#  Provider dispatch
# ---------------------------------------------------------------------------


SUPPORTED_PROVIDERS = {"anthropic", "openai", "deepseek", "google"}


def run_agent_on_task(
    provider: str,
    api_key: str,
    model: str,
    task_prompt: str,
    mcp_url: str,
    mcp_token: str = "",
    system_prompt: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Run a single agent submission against one task.

    The contract mirrors what the old HTTP dispatcher returned, so the
    rest of the scoring pipeline (eval_dispatcher.score_cpu_components,
    eval_boltz.run_boltz_posteval, eval_judge.run_judge_panel) works
    unchanged.

    Returns:
        {
          "success": bool,
          "sequences": [str, ...],
          "run_log": [{step, tool, arguments, success, latency_sec, ...}],
          "total_steps": int,
          "total_time_sec": float,
          "metrics": {},  # reserved for future use
          "error": str (only when success=False),
        }
    """
    if provider not in SUPPORTED_PROVIDERS:
        return {
            "success": False,
            "error": f"Unknown provider '{provider}'. Use one of: {sorted(SUPPORTED_PROVIDERS)}",
        }

    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    schemas = load_tool_schemas()
    start = time.monotonic()

    try:
        if provider == "anthropic":
            run_log, final_text = _run_anthropic(
                api_key, model, system, task_prompt, schemas,
                mcp_url, mcp_token, max_steps,
            )
        elif provider == "openai":
            run_log, final_text = _run_openai_compat(
                api_key, model, system, task_prompt, schemas,
                mcp_url, mcp_token, max_steps,
            )
        elif provider == "deepseek":
            run_log, final_text = _run_openai_compat(
                api_key, model, system, task_prompt, schemas,
                mcp_url, mcp_token, max_steps,
                base_url="https://api.deepseek.com",
            )
        elif provider == "google":
            run_log, final_text = _run_google(
                api_key, model, system, task_prompt, schemas,
                mcp_url, mcp_token, max_steps,
            )
        else:
            return {"success": False, "error": "unreachable"}
    except Exception as e:
        elapsed = round(time.monotonic() - start, 1)
        logger.exception(f"Agent loop crashed for provider={provider}")
        return {
            "success": False,
            "error": f"Agent loop crashed: {type(e).__name__}: {str(e)[:400]}",
            "total_time_sec": elapsed,
        }

    sequences = extract_sequences(final_text)
    elapsed = round(time.monotonic() - start, 1)
    return {
        "success": True,
        "sequences": sequences,
        "run_log": run_log,
        "total_steps": len(run_log),
        "total_time_sec": elapsed,
        "metrics": {},
    }
