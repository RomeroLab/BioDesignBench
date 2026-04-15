"""In-process task dispatcher for contamination-safe submissions.

Replaces the previous HTTP-endpoint dispatcher: instead of POSTing each
task to a submitter-hosted endpoint (which leaked task content), this
runs the agent loop here in the leaderboard backend using:

  - the submitter's LLM provider + API key (transient, scrubbed after run)
  - the reference protein-design-mcp endpoint (PROTEIN_MCP_URL secret)
    or, if the submitter opted in, their own custom MCP URL

For each of the 73 hidden tasks:
  1. Build the task payload (now includes a per-submission canary token).
  2. Run the agent loop in process via eval_agent.run_agent_on_task().
  3. Compute CPU-side scores (approach, orchestration, feasibility,
     novelty, diversity); quality is deferred to the Boltz post-eval.
  4. Save the per-task result back to the submission queue.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Sequence/log limits (reused from the old HTTP validator)
MAX_SEQUENCES = 50
MAX_SEQUENCE_LENGTH = 2000
MAX_LOG_ENTRIES = 200
DISPATCH_TIMEOUT = 600  # per-task agent-loop budget (seconds)


def _validate_agent_output(result: dict[str, Any]) -> tuple[bool, str]:
    """Sanity-check the result returned by eval_agent.run_agent_on_task."""
    if not isinstance(result, dict):
        return False, "agent result must be a dict"
    if not result.get("success"):
        return False, result.get("error", "agent loop reported failure")
    sequences = result.get("sequences") or []
    if not isinstance(sequences, list):
        return False, "sequences must be a list"
    if len(sequences) > MAX_SEQUENCES:
        return False, f"too many sequences: {len(sequences)} > {MAX_SEQUENCES}"
    for i, s in enumerate(sequences):
        if not isinstance(s, str):
            return False, f"sequences[{i}] must be a string"
        if not s:
            return False, f"sequences[{i}] is empty"
        if len(s) > MAX_SEQUENCE_LENGTH:
            return False, f"sequences[{i}] too long: {len(s)} > {MAX_SEQUENCE_LENGTH}"
    run_log = result.get("run_log") or []
    if not isinstance(run_log, list):
        return False, "run_log must be a list"
    if len(run_log) > MAX_LOG_ENTRIES:
        return False, f"too many run_log entries: {len(run_log)} > {MAX_LOG_ENTRIES}"
    return True, ""


def _resolve_mcp(submission: dict[str, Any]) -> tuple[str, str]:
    """Pick the MCP endpoint for this submission.

    Custom MCP URL takes priority if the submitter opted in; otherwise
    we fall back to the lab-hosted reference protein-design-mcp at
    PROTEIN_MCP_URL (set as an HF Space secret).
    """
    custom_url = (submission.get("custom_mcp_url") or "").strip()
    if custom_url:
        return custom_url, (submission.get("custom_mcp_token") or "").strip()
    return (
        os.environ.get("PROTEIN_MCP_URL", "").strip(),
        os.environ.get("PROTEIN_MCP_TOKEN", "").strip(),
    )


def score_cpu_components(
    task_id: str,
    sequences: list[str],
    run_log: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    oracle_sequences: list[str] | None = None,
) -> dict[str, Any]:
    """Run CPU-only scoring components for one task.

    Quality is deferred until the Boltz post-eval supplies pLDDT/ipTM
    metrics; the other 5 components are computed here.
    """
    from eval_scorer import (
        get_category,
        score_approach,
        score_diversity,
        score_feasibility,
        score_novelty,
        score_orchestration,
    )

    thresholds = ground_truth.get("thresholds", {})
    reference_seq = ground_truth.get("reference_sequence")
    constraints = ground_truth.get("design_constraints", {})
    tools_expected = ground_truth.get("tools_expected", [])
    max_designs = ground_truth.get("max_designs", 10)

    cat = get_category(task_id)
    task_type = cat.task_type if cat else None
    tools_used = [e.get("tool", "") for e in run_log if e.get("tool")]

    approach_result = score_approach(
        tools_used=tools_used, tools_expected=tools_expected, task_type=task_type,
    )
    orchestration_result = score_orchestration(
        tool_call_log=run_log, task_id=task_id,
    )
    feasibility_result = score_feasibility(designs=sequences, constraints=constraints)
    novelty_result = score_novelty(
        designs=sequences, reference_seq=reference_seq, thresholds=thresholds,
    )
    diversity_result = score_diversity(designs=sequences, max_designs=max_designs)

    return {
        "task_id": task_id,
        "num_designs": len(sequences),
        "sequences": sequences,
        "cpu_scores": {
            "approach": approach_result["score"],
            "orchestration": orchestration_result["score"],
            "feasibility": feasibility_result["score"],
            "novelty": novelty_result["score"],
            "diversity": diversity_result["score"],
        },
        "cpu_details": {
            "approach": approach_result,
            "orchestration": orchestration_result,
            "feasibility": feasibility_result,
            "novelty": novelty_result,
            "diversity": diversity_result,
        },
        "quality_pending": True,
        "oracle_sequences": oracle_sequences or [],
        "ground_truth_thresholds": thresholds,
    }


def dispatch_all_tasks(
    submission_id: str,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """Run the agent loop on every hidden task for one submission.

    Loads the submission record (including the transient api_key),
    picks the MCP endpoint, runs eval_agent.run_agent_on_task() per
    task, scores the CPU components, and persists each per-task result
    back into the submission record. Scrubs the api_key and custom MCP
    token from the record at the end.
    """
    from eval_agent import run_agent_on_task
    from eval_queue import (
        get_submission, save_task_result, scrub_credentials, update_status,
    )
    from eval_tasks import build_task_payload, get_hidden_task_ids, get_task

    sub = get_submission(submission_id)
    if sub is None:
        logger.error(f"dispatch_all_tasks: submission {submission_id} not found")
        return []

    api_key = (sub.get("api_key") or "").strip()
    if not api_key:
        update_status(submission_id, "failed",
                      error_message="api_key missing or already scrubbed")
        return []

    provider = sub.get("provider") or ""
    model = sub.get("model_name") or ""
    canary_token = sub.get("canary_token") or ""
    mcp_url, mcp_token = _resolve_mcp(sub)

    task_ids = get_hidden_task_ids()
    total = len(task_ids)
    update_status(submission_id, "dispatching", tasks_total=total)

    results: list[dict[str, Any]] = []

    try:
        for i, task_id in enumerate(task_ids):
            payload = build_task_payload(task_id, canary_token=canary_token)
            if payload is None:
                results.append({
                    "task_id": task_id, "success": False, "error": "Task not found",
                })
                save_task_result(submission_id, task_id, results[-1])
                continue

            t0 = time.monotonic()
            try:
                agent_result = run_agent_on_task(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    task_prompt=payload["task_description"],
                    mcp_url=mcp_url,
                    mcp_token=mcp_token,
                )
            except Exception as e:
                logger.exception(f"agent loop crashed for task {task_id}")
                agent_result = {
                    "success": False,
                    "error": f"agent loop crashed: {type(e).__name__}: {str(e)[:300]}",
                }
            latency = round(time.monotonic() - t0, 1)

            ok, err = _validate_agent_output(agent_result)
            if not ok:
                result = {
                    "task_id": task_id, "success": False, "error": err,
                    "latency_sec": latency,
                }
                results.append(result)
                save_task_result(submission_id, task_id, result)
            else:
                task_data = get_task(task_id) or {}
                ground_truth = task_data.get("ground_truth", {}) if task_data else {}
                oracle_seqs = task_data.get("oracle_sequences", []) if task_data else []

                cpu_result = score_cpu_components(
                    task_id=task_id,
                    sequences=agent_result["sequences"],
                    run_log=agent_result["run_log"],
                    ground_truth=ground_truth,
                    oracle_sequences=oracle_seqs,
                )
                cpu_result["latency_sec"] = latency
                cpu_result["success"] = True
                cpu_result["agent_metrics"] = agent_result.get("metrics", {})
                cpu_result["agent_total_steps"] = agent_result.get("total_steps", 0)
                results.append(cpu_result)
                save_task_result(submission_id, task_id, cpu_result)

            if progress_callback:
                progress_callback(task_id, i + 1, total, results[-1])

            logger.info(
                f"[{i+1}/{total}] {task_id}: "
                f"{'OK' if results[-1].get('success') else 'FAIL'} "
                f"({results[-1].get('latency_sec', 0):.1f}s)"
            )
    finally:
        # Always scrub credentials, regardless of success/failure
        scrub_credentials(submission_id)

    return results
