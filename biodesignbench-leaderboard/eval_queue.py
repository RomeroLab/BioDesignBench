"""Submission queue management using HuggingFace Datasets.

Manages the lifecycle of benchmark submissions:
  pending → approved → dispatching → boltz → scoring → complete / failed

Rate limiting: 1 submission per calendar month per organization.
LLM-judge API costs are paid by Romero Lab, so the limit is intentionally low.

HF Dataset: RomeroLab-Duke/biodesignbench-submissions (private)
Schema: Each row is a submission with per-task results stored as JSON.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SUBMISSIONS_DATASET = os.environ.get(
    "BDB_SUBMISSIONS_DATASET",
    "RomeroLab-Duke/biodesignbench-submissions",
)
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_SUBMISSIONS_PER_MONTH = 1

# Submission status progression
VALID_STATUSES = {
    "pending",
    "approved",
    "dispatching",
    "boltz",
    "scoring",
    "complete",
    "failed",
    "rejected",
}


# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------


def _make_submission_row(
    agent_name: str,
    organization: str,
    provider: str,
    model_name: str,
    api_key: str,
    description: str = "",
    custom_mcp_url: str = "",
    custom_mcp_token: str = "",
    canary_token: str = "",
) -> dict[str, Any]:
    """Create a new submission row.

    The submitter's `api_key` is stored on the row only between
    submission and dispatch; `scrub_credentials()` removes it
    immediately after the agent loop completes (or fails).
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        "submission_id": str(uuid.uuid4())[:12],
        "agent_name": agent_name,
        "organization": organization,
        "provider": provider,
        "model_name": model_name,
        # Transient credentials -- scrubbed after dispatch
        "api_key": api_key,
        "custom_mcp_url": custom_mcp_url,
        "custom_mcp_token": custom_mcp_token,
        "description": description,
        "mcp_custom": bool(custom_mcp_url),
        "canary_token": canary_token,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "tasks_dispatched": 0,
        "tasks_total": 76,
        "tasks_boltz_done": 0,
        "overall_score": None,
        "component_scores": None,
        "taxonomy_scores": None,
        "per_task_results": "{}",  # JSON string of task_id → result
        "error_message": None,
    }


# ---------------------------------------------------------------------------
#  Queue operations (HF Datasets API)
# ---------------------------------------------------------------------------


def _get_dataset():
    """Load the submissions dataset from HF Hub."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            SUBMISSIONS_DATASET,
            split="train",
            token=HF_TOKEN,
        )
        return ds
    except Exception as e:
        logger.warning(f"Could not load submissions dataset: {e}")
        return None


def _save_rows(rows: list[dict[str, Any]]) -> bool:
    """Save rows back to HF Dataset."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi

        ds = Dataset.from_list(rows)
        ds.push_to_hub(
            SUBMISSIONS_DATASET,
            token=HF_TOKEN,
            private=True,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save submissions: {e}")
        return False


def _load_all_rows() -> list[dict[str, Any]]:
    """Load all submission rows as a list of dicts."""
    ds = _get_dataset()
    if ds is None:
        return []
    return [dict(row) for row in ds]


SUPPORTED_PROVIDERS = {"anthropic", "openai", "deepseek", "google"}


def submit(
    agent_name: str,
    organization: str,
    provider: str,
    model_name: str,
    api_key: str,
    description: str = "",
    custom_mcp_url: str = "",
    custom_mcp_token: str = "",
) -> dict[str, Any]:
    """Create a new submission.

    Returns:
        Dict with submission_id and status, or error message.
    """
    if not agent_name or not organization or not model_name or not api_key:
        return {"error": "agent_name, organization, model_name, and api_key are required"}

    if provider not in SUPPORTED_PROVIDERS:
        return {"error": f"provider must be one of {sorted(SUPPORTED_PROVIDERS)}"}

    if custom_mcp_url and not custom_mcp_url.startswith(("http://", "https://")):
        return {"error": "custom_mcp_url must start with http:// or https://"}

    error = check_rate_limit(organization)
    if error:
        return {"error": error}

    canary = uuid.uuid4().hex[:16]

    row = _make_submission_row(
        agent_name=agent_name,
        organization=organization,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        description=description,
        custom_mcp_url=custom_mcp_url,
        custom_mcp_token=custom_mcp_token,
        canary_token=canary,
    )

    rows = _load_all_rows()
    rows.append(row)

    if _save_rows(rows):
        return {
            "submission_id": row["submission_id"],
            "status": "pending",
            "canary_token": canary,
            "message": "Submission created. Awaiting admin approval.",
        }
    return {"error": "Failed to save submission. Please try again."}


def scrub_credentials(submission_id: str) -> bool:
    """Remove the submitter's api_key (and custom MCP token) from a row.

    Called immediately after the dispatch phase, regardless of whether
    the agent loop succeeded. The api_key is forwarded directly from the
    submission form to the agent loop and is never needed again after
    that single use.
    """
    rows = _load_all_rows()
    found = False
    for row in rows:
        if row.get("submission_id") == submission_id:
            row["api_key"] = ""
            row["custom_mcp_token"] = ""
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if not found:
        logger.error(f"scrub_credentials: submission {submission_id} not found")
        return False
    return _save_rows(rows)


def check_rate_limit(organization: str) -> str | None:
    """Check if an organization has exceeded the monthly submission limit.

    Returns:
        Error message string if rate limited, None if OK.
    """
    rows = _load_all_rows()
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")

    monthly_count = 0
    for row in rows:
        if row.get("organization", "").lower() != organization.lower():
            continue
        if row.get("status") in ("rejected", "failed"):
            continue
        created = row.get("created_at", "")
        if created.startswith(current_month):
            monthly_count += 1

    if monthly_count >= MAX_SUBMISSIONS_PER_MONTH:
        return (
            f"Organization '{organization}' has reached the limit of "
            f"{MAX_SUBMISSIONS_PER_MONTH} submissions for {current_month}."
        )
    return None


def update_status(
    submission_id: str,
    status: str,
    **extra_fields: Any,
) -> bool:
    """Update a submission's status and optional extra fields.

    Args:
        submission_id: The submission to update.
        status: New status (must be in VALID_STATUSES).
        **extra_fields: Additional fields to update (e.g., tasks_dispatched=10).

    Returns:
        True if updated successfully.
    """
    if status not in VALID_STATUSES:
        logger.error(f"Invalid status: {status}")
        return False

    rows = _load_all_rows()
    found = False
    for row in rows:
        if row.get("submission_id") == submission_id:
            row["status"] = status
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            for k, v in extra_fields.items():
                if k in row:
                    row[k] = v
            found = True
            break

    if not found:
        logger.error(f"Submission {submission_id} not found")
        return False

    return _save_rows(rows)


def save_task_result(
    submission_id: str,
    task_id: str,
    result: dict[str, Any],
) -> bool:
    """Save a per-task result to the submission.

    Args:
        submission_id: The submission to update.
        task_id: Task identifier.
        result: Score result dict from eval_scorer.score_submission_task().

    Returns:
        True if saved successfully.
    """
    rows = _load_all_rows()
    for row in rows:
        if row.get("submission_id") == submission_id:
            per_task = json.loads(row.get("per_task_results", "{}"))
            per_task[task_id] = result
            row["per_task_results"] = json.dumps(per_task)
            row["tasks_dispatched"] = len(per_task)
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            return _save_rows(rows)

    logger.error(f"Submission {submission_id} not found")
    return False


def get_submission(submission_id: str) -> dict[str, Any] | None:
    """Get a single submission by ID."""
    rows = _load_all_rows()
    for row in rows:
        if row.get("submission_id") == submission_id:
            return row
    return None


def get_pending_submissions() -> list[dict[str, Any]]:
    """Get all submissions awaiting admin approval."""
    return [r for r in _load_all_rows() if r.get("status") == "pending"]


def get_approved_submissions() -> list[dict[str, Any]]:
    """Get all approved submissions ready for dispatch."""
    return [r for r in _load_all_rows() if r.get("status") == "approved"]


def get_all_submissions() -> list[dict[str, Any]]:
    """Get all submissions for the admin panel."""
    return _load_all_rows()


def finalize_submission(
    submission_id: str,
    overall_score: float,
    component_scores: dict[str, float],
    taxonomy_scores: dict[str, dict[str, float]],
) -> bool:
    """Finalize a submission with aggregated scores.

    Args:
        submission_id: The submission to finalize.
        overall_score: Overall score (0-100).
        component_scores: Dict of component → averaged score.
        taxonomy_scores: Nested dict of task_type → context → avg score.

    Returns:
        True if finalized successfully.
    """
    return update_status(
        submission_id,
        status="complete",
        overall_score=overall_score,
        component_scores=json.dumps(component_scores),
        taxonomy_scores=json.dumps(taxonomy_scores),
    )
