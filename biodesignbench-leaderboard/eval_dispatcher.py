"""HTTP task dispatcher — sends benchmark tasks to submitter endpoints.

For each of 76 tasks:
  1. Build task payload (prompt + tools + PDB data)
  2. POST to submitter's endpoint with timeout
  3. Validate response format
  4. Run CPU-only scoring (approach, orchestration, feasibility, novelty, diversity)
  5. Save results to submission queue

CPU scoring runs immediately; quality scoring waits for Boltz post-eval.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Response validation limits
MAX_SEQUENCES = 50
MAX_SEQUENCE_LENGTH = 2000
MAX_LOG_ENTRIES = 200
DISPATCH_TIMEOUT = 300  # seconds per task


# ---------------------------------------------------------------------------
#  Response validation
# ---------------------------------------------------------------------------


def validate_response(response: dict[str, Any]) -> tuple[bool, str]:
    """Validate the submitter's response format.

    Expected format:
    {
        "sequences": ["MKKL...", ...],
        "run_log": [{"step": 1, "tool": "...", "success": true, ...}, ...],
        "total_steps": 12,
        "total_time_sec": 142.5,
        "metrics": {}
    }

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response must be a JSON object"

    # sequences (required)
    sequences = response.get("sequences")
    if not isinstance(sequences, list):
        return False, "Missing or invalid 'sequences' field (must be a list)"

    if len(sequences) > MAX_SEQUENCES:
        return False, f"Too many sequences: {len(sequences)} > {MAX_SEQUENCES}"

    for i, seq in enumerate(sequences):
        if not isinstance(seq, str):
            return False, f"sequences[{i}] must be a string"
        if len(seq) > MAX_SEQUENCE_LENGTH:
            return False, f"sequences[{i}] too long: {len(seq)} > {MAX_SEQUENCE_LENGTH}"
        if len(seq) == 0:
            return False, f"sequences[{i}] is empty"

    # run_log (required)
    run_log = response.get("run_log")
    if not isinstance(run_log, list):
        return False, "Missing or invalid 'run_log' field (must be a list)"

    if len(run_log) > MAX_LOG_ENTRIES:
        return False, f"Too many log entries: {len(run_log)} > {MAX_LOG_ENTRIES}"

    for i, entry in enumerate(run_log):
        if not isinstance(entry, dict):
            return False, f"run_log[{i}] must be a dict"
        if "tool" not in entry:
            return False, f"run_log[{i}] missing 'tool' field"

    # Optional fields — validate types if present
    if "total_steps" in response:
        if not isinstance(response["total_steps"], (int, float)):
            return False, "'total_steps' must be a number"

    if "total_time_sec" in response:
        if not isinstance(response["total_time_sec"], (int, float)):
            return False, "'total_time_sec' must be a number"

    return True, ""


# ---------------------------------------------------------------------------
#  Single task dispatch
# ---------------------------------------------------------------------------


async def dispatch_single_task(
    endpoint_url: str,
    task_payload: dict[str, Any],
    timeout: int = DISPATCH_TIMEOUT,
) -> dict[str, Any]:
    """Send a single task to the submitter's endpoint.

    Args:
        endpoint_url: Submitter's POST endpoint URL.
        task_payload: Task payload from eval_tasks.build_task_payload().
        timeout: Request timeout in seconds.

    Returns:
        Dict with: success, task_id, response (if success), error (if failed),
        latency_sec.
    """
    import httpx

    task_id = task_payload["task_id"]
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                endpoint_url,
                json=task_payload,
                headers={"Content-Type": "application/json"},
            )
            latency = time.monotonic() - start

            if resp.status_code != 200:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "latency_sec": round(latency, 1),
                }

            try:
                data = resp.json()
            except Exception:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": "Response is not valid JSON",
                    "latency_sec": round(latency, 1),
                }

            is_valid, error_msg = validate_response(data)
            if not is_valid:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": f"Invalid response: {error_msg}",
                    "latency_sec": round(latency, 1),
                }

            return {
                "success": True,
                "task_id": task_id,
                "response": data,
                "latency_sec": round(latency, 1),
            }

    except httpx.TimeoutException:
        latency = time.monotonic() - start
        return {
            "success": False,
            "task_id": task_id,
            "error": f"Timeout after {timeout}s",
            "latency_sec": round(latency, 1),
        }
    except Exception as e:
        latency = time.monotonic() - start
        return {
            "success": False,
            "task_id": task_id,
            "error": f"Connection error: {str(e)[:200]}",
            "latency_sec": round(latency, 1),
        }


# ---------------------------------------------------------------------------
#  CPU scoring (runs immediately, no GPU needed)
# ---------------------------------------------------------------------------


def score_cpu_components(
    task_id: str,
    sequences: list[str],
    run_log: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    oracle_sequences: list[str] | None = None,
) -> dict[str, Any]:
    """Run CPU-only scoring components.

    Scores: approach, orchestration, feasibility, novelty, diversity.
    Quality scoring is deferred until Boltz post-eval provides pLDDT/ipTM.

    Args:
        task_id: Task identifier.
        sequences: Designed sequences from submitter.
        run_log: Tool call log from submitter.
        ground_truth: Ground truth data for this task.
        oracle_sequences: Oracle sequences for non-binding tasks.

    Returns:
        Dict with partial scores and metadata for later Boltz completion.
    """
    from eval_scorer import (
        get_category,
        score_approach,
        score_diversity,
        score_feasibility,
        score_novelty,
        score_orchestration,
    )

    # Extract fields
    thresholds = ground_truth.get("thresholds", {})
    reference_seq = ground_truth.get("reference_sequence")
    constraints = ground_truth.get("design_constraints", {})
    tools_expected = ground_truth.get("tools_expected", [])
    max_designs = ground_truth.get("max_designs", 10)

    cat = get_category(task_id)
    task_type = cat.task_type if cat else None
    tools_used = [e.get("tool", "") for e in run_log if e.get("tool")]

    approach_result = score_approach(
        tools_used=tools_used,
        tools_expected=tools_expected,
        task_type=task_type,
    )
    orchestration_result = score_orchestration(
        tool_call_log=run_log,
        task_id=task_id,
    )
    feasibility_result = score_feasibility(
        designs=sequences,
        constraints=constraints,
    )
    novelty_result = score_novelty(
        designs=sequences,
        reference_seq=reference_seq,
        thresholds=thresholds,
    )
    diversity_result = score_diversity(
        designs=sequences,
        max_designs=max_designs,
    )

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
        "quality_pending": True,  # Needs Boltz post-eval
        "oracle_sequences": oracle_sequences or [],
        "ground_truth_thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
#  Full dispatch pipeline
# ---------------------------------------------------------------------------


async def dispatch_all_tasks(
    submission_id: str,
    endpoint_url: str,
    progress_callback=None,
) -> Generator[dict[str, Any], None, None]:
    """Dispatch all hidden tasks to a submitter endpoint.

    Yields progress updates as each task completes. Saves results
    to the submission queue incrementally.

    Args:
        submission_id: Submission ID for queue tracking.
        endpoint_url: Submitter's POST endpoint.
        progress_callback: Optional callback(task_id, i, total, result)
            for streaming progress updates.

    Returns:
        List of per-task results.
    """
    from eval_queue import save_task_result, update_status
    from eval_tasks import build_task_payload, get_hidden_task_ids, get_task

    task_ids = get_hidden_task_ids()
    total = len(task_ids)
    results = []

    update_status(submission_id, "dispatching", tasks_total=total)

    for i, task_id in enumerate(task_ids):
        # Build payload
        payload = build_task_payload(task_id)
        if payload is None:
            result = {
                "task_id": task_id,
                "success": False,
                "error": "Task not found",
            }
            results.append(result)
            save_task_result(submission_id, task_id, result)
            continue

        # Dispatch
        dispatch_result = await dispatch_single_task(endpoint_url, payload)

        if dispatch_result["success"]:
            # Run CPU scoring
            task_data = get_task(task_id)
            ground_truth = task_data["ground_truth"] if task_data else {}
            oracle_seqs = task_data.get("oracle_sequences", []) if task_data else []

            response = dispatch_result["response"]
            cpu_result = score_cpu_components(
                task_id=task_id,
                sequences=response["sequences"],
                run_log=response["run_log"],
                ground_truth=ground_truth,
                oracle_sequences=oracle_seqs,
            )
            cpu_result["latency_sec"] = dispatch_result["latency_sec"]
            cpu_result["success"] = True
            cpu_result["agent_metrics"] = response.get("metrics", {})
            results.append(cpu_result)
            save_task_result(submission_id, task_id, cpu_result)
        else:
            result = {
                "task_id": task_id,
                "success": False,
                "error": dispatch_result["error"],
                "latency_sec": dispatch_result.get("latency_sec"),
            }
            results.append(result)
            save_task_result(submission_id, task_id, result)

        if progress_callback:
            progress_callback(task_id, i + 1, total, results[-1])

        logger.info(
            f"[{i+1}/{total}] {task_id}: "
            f"{'OK' if results[-1].get('success') else 'FAIL'} "
            f"({results[-1].get('latency_sec', 0):.1f}s)"
        )

    return results
