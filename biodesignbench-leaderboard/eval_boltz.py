"""Boltz structure prediction for post-assessment scoring.

Uses @spaces.GPU decorator for ZeroGPU on HuggingFace Spaces.

Two prediction modes:
  - Monomer: Non-binding tasks -> pLDDT, pTM
  - Complex: Binding tasks (binder + target) -> ipTM, i_pAE

Batch chunking respects ZeroGPU time limits (~180-240s per burst).

Phase B activation checklist (must all be true to actually run Boltz):
  1. HF Space hardware switched to a GPU tier (zero-a10g recommended).
  2. requirements.txt has `torch` and `boltz` uncommented.
  3. HF_TOKEN secret set on the Space (for the private hidden-tasks dataset).
On a cpu-basic Space the predictors return a structured failure dict
with `success=False` and an actionable error message rather than
crashing the dispatcher.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Chunking limits for ZeroGPU (free tier: ~300s max per burst)
MONOMER_CHUNK_SIZE = 5    # ~30-60s per monomer
COMPLEX_CHUNK_SIZE = 2    # ~60-120s per complex
MAX_GPU_TIME = 240         # safety margin under 300s ZeroGPU limit


# ---------------------------------------------------------------------------
#  Boltz prediction (GPU-accelerated)
# ---------------------------------------------------------------------------


_BOLTZ_NOT_INSTALLED = (
    "Boltz / torch not available on this Space. To enable Phase B, "
    "switch the Space hardware to ZeroGPU (zero-a10g) and uncomment the "
    "torch + boltz lines in requirements.txt."
)


def _predict_monomer(sequence: str) -> dict[str, float]:
    """Predict structure of a single protein sequence using Boltz.

    Returns:
        Dict with: pLDDT, pTM (or a structured failure dict).
    """
    try:
        import torch  # noqa: F401
        from boltz import Boltz
    except ImportError:
        logger.warning(_BOLTZ_NOT_INSTALLED)
        return {
            "pLDDT": 0.0, "pTM": 0.0,
            "success": False, "error": _BOLTZ_NOT_INSTALLED,
        }
    try:
        model = Boltz.from_pretrained("boltz2")
        result = model.predict(sequence)

        plddt = float(result.confidence.plddt.mean())
        ptm = float(result.confidence.ptm)

        return {
            "pLDDT": round(plddt, 2),
            "pTM": round(ptm, 4),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Boltz monomer prediction failed: {e}")
        return {"pLDDT": 0.0, "pTM": 0.0, "success": False, "error": str(e)}


def _predict_complex(
    binder_seq: str,
    target_seq: str,
) -> dict[str, float]:
    """Predict complex structure and binding metrics using Boltz.

    Returns:
        Dict with: ipTM, i_pAE, pLDDT, pTM (or a structured failure dict).
    """
    try:
        import torch  # noqa: F401
        from boltz import Boltz
    except ImportError:
        logger.warning(_BOLTZ_NOT_INSTALLED)
        return {
            "pLDDT": 0.0, "pTM": 0.0, "ipTM": 0.0, "i_pAE": 0.0,
            "success": False, "error": _BOLTZ_NOT_INSTALLED,
        }
    try:
        model = Boltz.from_pretrained("boltz2")
        result = model.predict([binder_seq, target_seq])

        plddt = float(result.confidence.plddt.mean())
        ptm = float(result.confidence.ptm)
        iptm = float(result.confidence.iptm) if hasattr(result.confidence, "iptm") else 0.0
        ipae = float(result.confidence.ipae) if hasattr(result.confidence, "ipae") else 0.0

        return {
            "pLDDT": round(plddt, 2),
            "pTM": round(ptm, 4),
            "ipTM": round(iptm, 4),
            "i_pAE": round(ipae, 2),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Boltz complex prediction failed: {e}")
        return {
            "pLDDT": 0.0, "pTM": 0.0, "ipTM": 0.0, "i_pAE": 0.0,
            "success": False, "error": str(e),
        }


# ---------------------------------------------------------------------------
#  GPU-decorated entry points (for HF Spaces with ZeroGPU)
# ---------------------------------------------------------------------------

try:
    import spaces

    @spaces.GPU(duration=MAX_GPU_TIME)
    def predict_monomer_batch(sequences: list[str]) -> list[dict[str, float]]:
        """Predict structures for a batch of monomer sequences.

        Decorated with @spaces.GPU for ZeroGPU allocation.

        Args:
            sequences: List of amino acid sequences (max MONOMER_CHUNK_SIZE).

        Returns:
            List of prediction result dicts with pLDDT, pTM.
        """
        results = []
        for seq in sequences[:MONOMER_CHUNK_SIZE]:
            results.append(_predict_monomer(seq))
        return results

    @spaces.GPU(duration=MAX_GPU_TIME)
    def predict_complex_batch(
        pairs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """Predict structures for a batch of binder-target pairs.

        Args:
            pairs: List of (binder_seq, target_seq) tuples.

        Returns:
            List of prediction result dicts with ipTM, i_pAE, pLDDT, pTM.
        """
        results = []
        for binder, target in pairs[:COMPLEX_CHUNK_SIZE]:
            results.append(_predict_complex(binder, target))
        return results

except ImportError:
    # Not running on HF Spaces -- provide un-decorated versions
    def predict_monomer_batch(sequences: list[str]) -> list[dict[str, float]]:
        return [_predict_monomer(seq) for seq in sequences[:MONOMER_CHUNK_SIZE]]

    def predict_complex_batch(
        pairs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        return [_predict_complex(b, t) for b, t in pairs[:COMPLEX_CHUNK_SIZE]]


# ---------------------------------------------------------------------------
#  High-level assessment API
# ---------------------------------------------------------------------------


def run_boltz_posteval(
    per_task_results: dict[str, dict[str, Any]],
    progress_callback=None,
) -> dict[str, dict[str, Any]]:
    """Run Boltz post-assessment on all tasks that need it.

    For each task:
      - Non-binding: pick best design -> monomer prediction
      - Binding: pick best design + target sequence -> complex prediction
      - Merge Boltz metrics into existing results
      - Re-score quality component

    Args:
        per_task_results: Dict of task_id -> dispatch result (from dispatcher).
        progress_callback: Optional callback(task_id, i, total, metrics).

    Returns:
        Updated per_task_results with Boltz metrics and final quality scores.
    """
    from eval_scorer import _is_binding_task, score_quality

    # Separate tasks into monomer and complex batches
    monomer_tasks = []
    complex_tasks = []

    for task_id, result in per_task_results.items():
        if not result.get("success") or not result.get("quality_pending"):
            continue

        sequences = result.get("sequences", [])
        if not sequences:
            continue

        best_seq = sequences[0]  # Use first design for Boltz

        if _is_binding_task(task_id):
            # Need target sequence from ground truth
            target_seq = result.get("ground_truth_thresholds", {}).get("target_sequence")
            if target_seq:
                complex_tasks.append((task_id, best_seq, target_seq))
            else:
                # Fall back to monomer if no target
                monomer_tasks.append((task_id, best_seq))
        else:
            monomer_tasks.append((task_id, best_seq))

    total = len(monomer_tasks) + len(complex_tasks)
    done = 0

    # Process monomer tasks in chunks
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

    # Process complex tasks in chunks
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
    boltz_metrics: dict[str, float],
) -> None:
    """Merge Boltz prediction metrics into a task result and re-score quality.

    Modifies task_result in-place.
    """
    from eval_scorer import apply_design_gate, score_quality

    # Merge Boltz metrics with any agent-reported metrics
    merged_metrics = task_result.get("agent_metrics", {}).copy()
    for key in ("pLDDT", "pTM", "ipTM", "i_pAE"):
        if key in boltz_metrics and boltz_metrics[key] > 0:
            merged_metrics[key] = boltz_metrics[key]

    # Re-score quality with Boltz metrics
    quality_result = score_quality(
        agent_metrics=merged_metrics,
        thresholds=task_result.get("ground_truth_thresholds", {}),
        task_id=task_result.get("task_id", ""),
        designs=task_result.get("sequences"),
        oracle_sequences=task_result.get("oracle_sequences"),
    )

    # Update scores
    task_result["boltz_metrics"] = boltz_metrics
    task_result["quality_pending"] = False

    if "cpu_scores" in task_result:
        task_result["cpu_scores"]["quality"] = quality_result["score"]

    # Compute final gated score
    if "cpu_scores" in task_result:
        component_scores = dict(task_result["cpu_scores"])
        gated = apply_design_gate(component_scores, task_result.get("num_designs", 0))
        task_result["final_scores"] = gated
        task_result["total_score"] = sum(gated.values())

    if "cpu_details" in task_result:
        task_result["cpu_details"]["quality"] = quality_result
