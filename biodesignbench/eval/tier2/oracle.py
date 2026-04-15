"""Oracle sequence loader for functional similarity scoring.

Oracle sequences are publication-validated designs extracted from the
source papers for each task.  They serve as the gold standard for
measuring whether an agent's designs are functionally relevant (as
opposed to merely well-folded).

The file ``data/oracle/sequences.json`` maps task_id → entry dict
with a ``"sequences"`` list of amino acid strings.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ORACLE_PATH = Path(__file__).resolve().parents[3] / "data" / "oracle" / "sequences.json"


@lru_cache(maxsize=1)
def _load_oracle_data() -> dict:
    """Load and cache the entire oracle sequences file."""
    if not _ORACLE_PATH.exists():
        return {}
    with open(_ORACLE_PATH) as f:
        return json.load(f)


def get_oracle_sequences(task_id: str) -> list[str]:
    """Return oracle sequences for a given task ID.

    Args:
        task_id: Task identifier (e.g., ``"sqo_enz_005"``).

    Returns:
        List of amino acid sequences.  Empty list if the task has no
        oracle entry or the file is missing.
    """
    data = _load_oracle_data()
    entry = data.get(task_id, {})
    if not isinstance(entry, dict):
        return []
    seqs = entry.get("sequences", [])
    if not isinstance(seqs, list):
        return []
    return [s for s in seqs if isinstance(s, str) and len(s) > 0]
