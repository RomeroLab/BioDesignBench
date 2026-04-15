"""Boltz-2 structure verification client (Phase B).

The HF Space leaderboard runs on cpu-basic, so it cannot host Boltz
directly. This module is a thin HTTP client that POSTs design sequences
to a Modal-deployed companion app (`modal_boltz_app.py`), which
provisions an A10G on demand, runs `boltz predict`, and returns
confidence metrics.

Two prediction modes (selected automatically by `run_boltz_posteval`):
  - Monomer (non-binding tasks)   -> pLDDT, pTM
  - Complex (binding tasks)       -> pLDDT, pTM, ipTM, i_pAE

Required HF Space secrets (set out-of-band via the leaderboard admin):
  MODAL_BOLTZ_URL    https://<workspace>--bdb-boltz-predict.modal.run
  MODAL_BOLTZ_TOKEN  shared bearer token matching the modal secret TOKEN

If `MODAL_BOLTZ_URL` is unset the predictors return a structured
failure dict with `success=False` and an actionable error message
rather than crashing the dispatcher.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Batch sizes large enough to amortize Modal cold-start, small enough
# to stay under the 1700s function timeout.
MONOMER_CHUNK_SIZE = 20
COMPLEX_CHUNK_SIZE = 10
HTTP_TIMEOUT_SEC = 1700


_NOT_CONFIGURED = (
    "Modal Boltz endpoint not configured. Set MODAL_BOLTZ_URL (and "
    "MODAL_BOLTZ_TOKEN) on the HF Space, or deploy the companion app "
    "with `modal deploy modal_boltz_app.py`."
)


def _modal_url() -> str | None:
    return os.environ.get("MODAL_BOLTZ_URL", "").strip() or None


def _modal_token() -> str:
    return os.environ.get("MODAL_BOLTZ_TOKEN", "").strip()


def _failure(error: str, complex_keys: bool = False) -> dict[str, Any]:
    out = {"pLDDT": 0.0, "pTM": 0.0, "success": False, "error": error}
    if complex_keys:
        out.update({"ipTM": 0.0, "i_pAE": 0.0})
    return out


def _post_predictions(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """POST a list of prediction items to the Modal endpoint.

    Returns a dict mapping each item's `name` to a metric dict, with
    structured failure entries on error.
    """
    url = _modal_url()
    if not url:
        return {item["name"]: _failure(_NOT_CONFIGURED) for item in items}

    try:
        import httpx
    except ImportError:
        return {
            item["name"]: _failure("httpx not installed in leaderboard image")
            for item in items
        }

    headers = {"Content-Type": "application/json"}
    payload = {"token": _modal_token(), "items": items}

    try:
        resp = httpx.post(
            url, json=payload, headers=headers, timeout=HTTP_TIMEOUT_SEC,
        )
    except Exception as e:
        return {item["name"]: _failure(f"Modal POST failed: {e}") for item in items}

    if resp.status_code != 200:
        return {
            item["name"]: _failure(f"Modal HTTP {resp.status_code}: {resp.text[:200]}")
            for item in items
        }

    try:
        body = resp.json()
    except Exception as e:
        return {item["name"]: _failure(f"Modal returned non-JSON: {e}") for item in items}

    if "error" in body:
        msg = body["error"]
        return {item["name"]: _failure(f"Modal: {msg}") for item in items}

    results = body.get("results", {})
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        name = item["name"]
        out[name] = results.get(name) or _failure(
            "Modal returned no result for this item"
        )
    return out


def predict_monomer_batch(sequences: list[str]) -> list[dict[str, float]]:
    """Predict structures for a batch of monomer sequences."""
    items = [
        {"name": f"mono_{i}", "kind": "monomer", "sequences": [seq]}
        for i, seq in enumerate(sequences[:MONOMER_CHUNK_SIZE])
    ]
    by_name = _post_predictions(items)
    return [by_name[item["name"]] for item in items]


def predict_complex_batch(
    pairs: list[tuple[str, str]],
) -> list[dict[str, float]]:
    """Predict structures for a batch of (binder, target) pairs."""
    items = [
        {"name": f"cmplx_{i}", "kind": "complex", "sequences": [b, t]}
        for i, (b, t) in enumerate(pairs[:COMPLEX_CHUNK_SIZE])
    ]
    by_name = _post_predictions(items)
    return [by_name[item["name"]] for item in items]


def run_boltz_posteval(
    per_task_results: dict[str, dict[str, Any]],
    progress_callback=None,
) -> dict[str, dict[str, Any]]:
    """Run Boltz post-assessment on every task that needs it.

    For each successful task:
      - Non-binding: pick the first design -> monomer prediction
      - Binding: pick the first design + target sequence -> complex prediction
      - Merge Boltz metrics into existing results
      - Re-score the quality component
    """
    from eval_scorer import _is_binding_task

    monomer_tasks: list[tuple[str, str]] = []
    complex_tasks: list[tuple[str, str, str]] = []

    for task_id, result in per_task_results.items():
        if not result.get("success") or not result.get("quality_pending"):
            continue
        sequences = result.get("sequences", [])
        if not sequences:
            continue
        best_seq = sequences[0]

        if _is_binding_task(task_id):
            target_seq = (
                result.get("ground_truth_thresholds", {}).get("target_sequence")
            )
            if target_seq:
                complex_tasks.append((task_id, best_seq, target_seq))
            else:
                monomer_tasks.append((task_id, best_seq))
        else:
            monomer_tasks.append((task_id, best_seq))

    total = len(monomer_tasks) + len(complex_tasks)
    done = 0

    for chunk_start in range(0, len(monomer_tasks), MONOMER_CHUNK_SIZE):
        chunk = monomer_tasks[chunk_start:chunk_start + MONOMER_CHUNK_SIZE]
        seqs = [seq for _, seq in chunk]
        boltz_results = predict_monomer_batch(seqs)
        for (task_id, _), metrics in zip(chunk, boltz_results):
            if metrics.get("success"):
                _merge_boltz_metrics(per_task_results[task_id], metrics)
            done += 1
            if progress_callback:
                progress_callback(task_id, done, total, metrics)

    for chunk_start in range(0, len(complex_tasks), COMPLEX_CHUNK_SIZE):
        chunk = complex_tasks[chunk_start:chunk_start + COMPLEX_CHUNK_SIZE]
        pairs = [(binder, target) for _, binder, target in chunk]
        boltz_results = predict_complex_batch(pairs)
        for (task_id, _, _), metrics in zip(chunk, boltz_results):
            if metrics.get("success"):
                _merge_boltz_metrics(per_task_results[task_id], metrics)
            done += 1
            if progress_callback:
                progress_callback(task_id, done, total, metrics)

    return per_task_results


def _merge_boltz_metrics(
    task_result: dict[str, Any],
    boltz_metrics: dict[str, Any],
) -> None:
    """Merge Boltz prediction metrics into a task result and re-score quality."""
    from eval_scorer import apply_design_gate, score_quality

    merged_metrics = task_result.get("agent_metrics", {}).copy()
    for key in ("pLDDT", "pTM", "ipTM", "i_pAE"):
        if key in boltz_metrics and boltz_metrics[key] > 0:
            merged_metrics[key] = boltz_metrics[key]

    quality_result = score_quality(
        agent_metrics=merged_metrics,
        thresholds=task_result.get("ground_truth_thresholds", {}),
        task_id=task_result.get("task_id", ""),
        designs=task_result.get("sequences"),
        oracle_sequences=task_result.get("oracle_sequences"),
    )

    task_result["boltz_metrics"] = boltz_metrics
    task_result["quality_pending"] = False

    if "cpu_scores" in task_result:
        task_result["cpu_scores"]["quality"] = quality_result["score"]
        component_scores = dict(task_result["cpu_scores"])
        gated = apply_design_gate(component_scores, task_result.get("num_designs", 0))
        task_result["final_scores"] = gated
        task_result["total_score"] = sum(gated.values())

    if "cpu_details" in task_result:
        task_result["cpu_details"]["quality"] = quality_result
