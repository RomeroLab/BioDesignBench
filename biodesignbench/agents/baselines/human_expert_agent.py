"""Expert-level sample-and-filter pipeline baseline for BioDesignBench Tier 2.

Implements the BindCraft/RFdiffusion-style oversampling paradigm:

    1. Massive oversampling  (50 backbones × 8 sequences = 400 candidates)
    2. Composition filtering (pure Python, no tool call)
    3. Monomer validation    (ESMFold, cheap GPU)
    4. Complex prediction    (AF2-Multimer, expensive GPU, only for survivors)
    5. Rosetta scoring       (CPU)
    6. Composite ranking     (weighted multi-metric)
    7. Diversity selection   (sequence-identity clustering → top-N)

Each DesignTaskType maps to a specialized pipeline variant:

    de_novo_binder:       design_binder(50) → validate → predict_complex → analyze_interface → rosetta → rank → top-3
    sequence_optimization: optimize_sequence(×60) → validate → score_stability → rosetta → rank → top-3
    de_novo_backbone:     generate_backbone(50) → optimize_sequence(×8) → validate → self-consistency → rosetta → top-3
    complex_engineering:  design_binder(50) → validate → predict_complex → analyze_interface → rosetta → rank → top-3
    conformational_design: optimize_sequence(×60) → validate → energy_minimize → score_stability → rosetta → rank → top-3

This deterministic baseline uses NO LLM reasoning — arguments are extracted
directly from task JSON fields and fed through a fixed sequence of MCP tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput, ToolCallEntry, summarize_tool_args
from biodesignbench.tasks.schema import Task, TaskTier
from biodesignbench.taxonomy import DesignApproach, MolecularSubject, get_category

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sampling configuration per task type
# ---------------------------------------------------------------------------

SAMPLING_CONFIG: dict[str, dict[str, Any]] = {
    "de_novo_binder": {
        "num_backbones": 5,          # Reduced from 10 (still ~15-30 min)
        "seqs_per_backbone": 2,
        "filter_plddt": 75.0,
        "top_n_complex": 5,          # Reduced from 10
        "top_n_interface": 3,        # Reduced from 5
        "top_n_rosetta": 3,          # Reduced from 5
        "final_n": 3,
    },
    "sequence_optimization": {
        "targets": ["stability", "affinity", "both"],
        "seqs_per_target": 3,        # Reduced from 5 (3×3=9 calls)
        "filter_plddt": 75.0,
        "top_n_rosetta": 5,          # Reduced from 10
        "final_n": 3,
    },
    "de_novo_backbone": {
        "num_backbones": 5,          # Reduced from 10
        "seqs_per_backbone": 2,
        "filter_plddt": 75.0,
        "filter_sctm": 0.5,
        "top_n_rosetta": 3,          # Reduced from 5
        "final_n": 3,
    },
    "complex_engineering": {
        "num_backbones": 5,          # Reduced from 10
        "seqs_per_backbone": 2,
        "filter_plddt": 75.0,
        "top_n_complex": 5,          # Reduced from 10
        "top_n_interface": 3,        # Reduced from 5
        "top_n_rosetta": 3,          # Reduced from 5
        "final_n": 3,
    },
    "conformational_design": {
        "targets": ["stability", "affinity", "both"],
        "seqs_per_target": 3,        # Reduced from 5 (3×3=9 calls)
        "filter_plddt": 73.0,
        "top_n_rosetta": 5,          # Reduced from 10
        "final_n": 3,
    },
}


# Shallow config: mimics typical LLM agent depth
# - Fewer candidates (2 backbones, 1 seq each)
# - Minimal evaluation funnel (1 candidate through each stage)
# - No filtering (final_n = num candidates, output everything)
# NOTE: top_n_* values must be ≥1 so the pipeline doesn't return empty lists.
# The depth reduction comes from the narrow funnel (1 per stage), not skipping stages.
SHALLOW_SAMPLING_CONFIG: dict[str, dict[str, Any]] = {
    "de_novo_binder": {
        "num_backbones": 2,
        "seqs_per_backbone": 1,
        "filter_plddt": 0.0,
        "top_n_complex": 1,
        "top_n_interface": 1,
        "top_n_rosetta": 1,
        "final_n": 2,
    },
    "sequence_optimization": {
        "targets": ["stability"],
        "seqs_per_target": 3,
        "filter_plddt": 0.0,
        "top_n_rosetta": 1,
        "final_n": 3,
    },
    "de_novo_backbone": {
        "num_backbones": 2,
        "seqs_per_backbone": 1,
        "filter_plddt": 0.0,
        "filter_sctm": 0.0,
        "top_n_rosetta": 1,
        "final_n": 2,
    },
    "complex_engineering": {
        "num_backbones": 2,
        "seqs_per_backbone": 1,
        "filter_plddt": 0.0,
        "top_n_complex": 1,
        "top_n_interface": 1,
        "top_n_rosetta": 1,
        "final_n": 2,
    },
    "conformational_design": {
        "targets": ["stability"],
        "seqs_per_target": 3,
        "filter_plddt": 0.0,
        "top_n_rosetta": 1,
        "final_n": 3,
    },
}


def _adaptive_seqs_per_target(seq_len: int, base: int = 5) -> int:
    """Reduce seqs_per_target for long sequences to fit 120min timeout.

    Total HE budget per task: optimize(N) + validate(N) + score(N) + rosetta(~5).
    Each optimize_sequence: ~0.3s × seq_len.  Each validate: ~0.1s × seq_len.
    """
    if seq_len <= 100:
        return base        # 3×5=15 opt → ~8min + validate ~3min
    elif seq_len <= 200:
        return 3           # 3×3=9 opt → ~11min + validate ~4min
    elif seq_len <= 300:
        return 2           # 3×2=6 opt → ~11min + validate ~4min
    elif seq_len <= 500:
        return 1           # 3×1=3 opt → ~8min + validate ~3min
    else:
        return 1           # 3×1=3 opt → ~10min + validate ~4min


def _adaptive_top_n_rosetta(seq_len: int, base: int = 10) -> int:
    """Reduce rosetta top-N for long sequences (rosetta ~2min/call)."""
    if seq_len <= 150:
        return base
    elif seq_len <= 300:
        return 5
    elif seq_len <= 500:
        return 3
    else:
        return 2


def _adaptive_num_designs(
    binder_length: int, base_count: int = 10, target_length: int = 0
) -> int:
    """Reduce design count for large proteins to avoid RFdiffusion timeout.

    RFdiffusion scales roughly as O((L_target + L_binder)²) in time.
    Both binder length and target size contribute significantly.
    """
    count = base_count
    # Reduce for large binders
    if binder_length > 150:
        count = max(3, count // 3)
    elif binder_length > 100:
        count = max(5, count // 2)
    # Reduce for large targets (dominant cost factor for enzymes)
    if target_length > 500:
        count = max(2, count // 3)
    elif target_length > 300:
        count = max(3, count // 2)
    return count

# Per-tool-call timeout (seconds).  Prevents a single hung MCP process
# (e.g. GPU OOM in RFdiffusion) from consuming the entire task budget.
TOOL_TIMEOUT = 1800  # 30 minutes (fallback for tests / backward compat)


def _adaptive_tool_timeout(seq_len: int) -> int:
    """Adaptive per-tool-call timeout based on sequence length.

    Short sequences complete faster; a tighter timeout catches hangs sooner.
    Long sequences legitimately need more time for ESM-2/ESMFold.
    """
    if seq_len <= 200:
        return 600   # 10 minutes
    elif seq_len <= 400:
        return 1200  # 20 minutes
    else:
        return 1800  # 30 minutes

# Valid amino acid one-letter codes (X = unknown, accepted by AlphaFold)
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWXY")


class HumanExpertAgent(AgentInterface):
    """Expert-level pipeline: sample → filter → rank → select.

    Implements BindCraft/RFdiffusion-style oversampling with
    multi-stage filtering for near-optimal design selection.
    """

    def __init__(
        self,
        tool_provider: Any | None = None,
        docker_image: str = "protein-design-mcp:full",
        sampling_config_name: str = "deep",
    ):
        self.tool_provider = tool_provider
        self.docker_image = docker_image
        self.sampling_config_name = sampling_config_name
        if sampling_config_name == "shallow":
            self._config = SHALLOW_SAMPLING_CONFIG
        else:
            self._config = SAMPLING_CONFIG
        self._tool_call_log: list[ToolCallEntry] = []
        self._tools_used: list[str] = []
        self._iteration: int = 0

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="human-expert-agent",
            name="Human Expert Agent",
            version="1.0.0",
            description=(
                "Expert-level sample-and-filter pipeline baseline "
                "(massive oversampling → multi-stage filtering → diversity selection)"
            ),
            provider="baseline",
            model="none",
            is_bio_specific=True,
            capabilities=["sample_and_filter", "protein_design_tools", "deterministic"],
        )

    def setup(self) -> None:
        if self.tool_provider is None:
            raise RuntimeError(
                "HumanExpertAgent requires a ProteinDesignToolProvider. "
                "Pass tool_provider= at construction time."
            )

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def solve(
        self,
        task: Task,
        output_dir: Path | None = None,
        input_dir: Path | None = None,
    ) -> AgentOutput:
        if task.tier != TaskTier.TIER2:
            return AgentOutput(
                reasoning_trace="HumanExpertAgent only supports Tier 2 design tasks.",
            )

        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="expert_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Provision input PDB files
        self._provision_inputs(task, output_dir, use_symlinks=False)

        # Reset per-task state
        self._tool_call_log = []
        self._tools_used = []
        self._iteration = 0

        # Determine pipeline from task type
        category = get_category(task.task_id)
        if category is None:
            return AgentOutput(
                reasoning_trace=f"Cannot determine category for {task.task_id}",
            )

        approach = category.approach
        subject = category.subject
        start_time = time.time()

        # Adaptive timeout based on target sequence length.
        # Capped by TOOL_TIMEOUT so test patches (TOOL_TIMEOUT=0.1) still work.
        target_seq = task.target.sequence or ""
        self._current_timeout = min(_adaptive_tool_timeout(len(target_seq)), TOOL_TIMEOUT)
        logger.info(
            "Adaptive timeout: %ds (seq_len=%d)", self._current_timeout, len(target_seq)
        )

        # Dispatch: approach + subject → specific pipeline
        # Note: cfd_ prefix only applies to enzyme/FP subjects, NOT scaffolds
        is_cfd_task = task.task_id.startswith("cfd_")

        if approach == DesignApproach.DE_NOVO:
            if subject == MolecularSubject.SCAFFOLD:
                pipeline_fn = self._pipeline_dnk
            elif subject in (MolecularSubject.BINDER, MolecularSubject.ANTIBODY):
                pipeline_fn = self._pipeline_dnb
            elif subject in (MolecularSubject.ENZYME, MolecularSubject.FLUORESCENT_PROTEIN):
                pipeline_fn = self._pipeline_cfd
            elif is_cfd_task:
                pipeline_fn = self._pipeline_cfd
            else:
                pipeline_fn = self._pipeline_cpx
        else:
            # REDESIGN: use specialized pipeline for antibodies (CDR-focused)
            if subject == MolecularSubject.ANTIBODY:
                pipeline_fn = self._pipeline_sqo_antibody
            else:
                pipeline_fn = self._pipeline_sqo

        logger.info(
            "Running expert %s/%s pipeline for %s",
            approach.short,
            subject.short,
            task.task_id,
        )

        designs = await pipeline_fn(task, output_dir)
        elapsed = time.time() - start_time

        self._write_outputs(designs, task, output_dir, elapsed)

        return AgentOutput(
            designs=designs,
            output_dir=str(output_dir),
            tools_used=list(set(self._tools_used)),
            tool_call_log=self._tool_call_log,
            api_calls=len(self._tool_call_log),
            iterations=self._iteration,
            reasoning_trace=(
                f"Expert {approach.short}/{subject.short} pipeline: "
                f"{len(self._tool_call_log)} tool calls, "
                f"{len(designs)} final designs"
            ),
            wall_time_seconds=round(elapsed, 3),
            tool_execution_seconds=round(
                sum(e.duration_seconds for e in self._tool_call_log), 3
            ),
        )

    # ------------------------------------------------------------------
    # Tool call helper
    # ------------------------------------------------------------------

    async def _call_tool(
        self,
        name: str,
        args: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Call an MCP tool and record the call.

        Uses adaptive timeout from ``_current_timeout`` (set per-task in
        ``solve()``) or falls back to ``TOOL_TIMEOUT``.
        """
        timeout = getattr(self, "_current_timeout", TOOL_TIMEOUT)
        self._iteration += 1
        self._tools_used.append(name)

        summary, values = summarize_tool_args(args)
        _tool_start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.tool_provider.call_tool(
                    name,
                    args,
                    output_dir,
                    mode="user",
                ),
                timeout=timeout,
            )
            _duration = round(time.monotonic() - _tool_start, 3)
            self._tool_call_log.append(
                ToolCallEntry(
                    tool=name,
                    iteration=self._iteration,
                    success=True,
                    args_summary=summary,
                    args_values=values,
                    duration_seconds=_duration,
                )
            )
            logger.info("  [%d] %s → OK (%.1fs)", self._iteration, name, _duration)
            return result
        except asyncio.TimeoutError:
            _duration = round(time.monotonic() - _tool_start, 3)
            error_msg = f"timeout after {timeout}s"
            self._tool_call_log.append(
                ToolCallEntry(
                    tool=name,
                    iteration=self._iteration,
                    success=False,
                    error=error_msg,
                    args_summary=summary,
                    args_values=values,
                    duration_seconds=_duration,
                )
            )
            logger.warning(
                "  [%d] %s → TIMEOUT after %ds", self._iteration, name, timeout
            )
            return {"error": error_msg}
        except Exception as exc:
            _duration = round(time.monotonic() - _tool_start, 3)
            error_msg = str(exc)[:200]
            self._tool_call_log.append(
                ToolCallEntry(
                    tool=name,
                    iteration=self._iteration,
                    success=False,
                    error=error_msg,
                    args_summary=summary,
                    args_values=values,
                    duration_seconds=_duration,
                )
            )
            logger.warning("  [%d] %s → FAILED: %s", self._iteration, name, error_msg)
            return {"error": error_msg}

    # ------------------------------------------------------------------
    # Parallel tool call helper
    # ------------------------------------------------------------------

    async def _call_tools_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        output_dir: Path,
        max_concurrency: int = 3,
    ) -> list[dict[str, Any]]:
        """Call multiple MCP tools in parallel with bounded concurrency.

        Args:
            calls: List of (tool_name, args) tuples.
            output_dir: Output directory for tool results.
            max_concurrency: Maximum concurrent tool calls (GPU contention limit).

        Returns:
            List of results in the same order as ``calls``.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self._call_tool(name, args, output_dir)

        tasks = [_bounded_call(name, args) for name, args in calls]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # ------------------------------------------------------------------
    # Fallback: generate_backbone → design_sequence
    # ------------------------------------------------------------------

    async def _fallback_backbone_design(
        self,
        length: int,
        num_backbones: int,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Fallback when ``design_binder`` fails or returns empty.

        Uses ``generate_backbone`` to create scaffolds, then
        ``design_sequence`` (ProteinMPNN) to thread sequences onto them.
        Falls back to scanning output dir for backbone PDBs if paths are stale.
        """
        logger.info(
            "  Fallback: generate_backbone(length=%d, n=%d) → design_sequence",
            length, num_backbones,
        )
        bb_result = await self._call_tool(
            "generate_backbone",
            {"length": length, "num_designs": num_backbones},
            output_dir,
        )

        # Collect backbone PDB paths (from result + output dir scan)
        bb_pdbs: list[str] = []
        if "error" not in bb_result:
            backbones = bb_result.get("backbones", [])
            if not backbones and bb_result.get("pdb_path"):
                backbones = [{"pdb_path": bb_result["pdb_path"]}]
            for bb in backbones[:5]:
                p = bb.get("pdb_path", "")
                if p:
                    bb_pdbs.append(p)

        # Also scan output dir for any backbone PDBs (handles Docker path issues)
        if not bb_pdbs:
            bb_pdbs = self._find_backbone_pdbs_in_dir(output_dir)
            if bb_pdbs:
                logger.info("  Fallback: found %d backbone PDBs in output dir", len(bb_pdbs))

        designs: list[dict[str, Any]] = []
        for bb_pdb in bb_pdbs[:5]:
            seq_result = await self._call_tool(
                "design_sequence",
                {"backbone_pdb": bb_pdb, "num_sequences": 2, "sampling_temp": 0.1},
                output_dir,
            )
            if "error" in seq_result:
                continue
            for s in seq_result.get("sequences", []):
                seq = s.get("sequence", "") if isinstance(s, dict) else s
                if seq and set(seq) <= _VALID_AA:
                    designs.append({"sequence": seq, "source": "fallback_backbone"})
        return designs

    # ------------------------------------------------------------------
    # Argument extraction helpers (reused from HardcodedPipelineAgent)
    # ------------------------------------------------------------------

    def _get_pdb_path(self, task: Task, output_dir: Path) -> str:
        pdb_id = task.target.pdb_id
        if pdb_id:
            candidate = output_dir / f"{pdb_id.lower()}.pdb"
            if candidate.exists():
                return str(candidate.resolve())
        for f in output_dir.iterdir():
            if f.suffix == ".pdb" and f.is_file():
                return str(f.resolve())
        return ""

    def _get_hotspot_residues(self, task: Task) -> list[str]:
        chain = task.target.chain or "A"
        residues = task.target.binding_site_residues or []
        return [f"{chain}{r}" for r in residues]

    def _get_binder_length(self, task: Task) -> int:
        if task.design_constraints.length_range:
            lo, hi = task.design_constraints.length_range
            return (lo + hi) // 2
        return 70

    def _get_target_sequence(self, task: Task) -> str:
        return task.target.sequence or ""

    @staticmethod
    def _extract_optimized_sequence(opt_result: dict[str, Any]) -> str:
        seq = opt_result.get("optimized_sequence", "")
        if seq:
            return seq
        seq = opt_result.get("best_sequence", "")
        if seq:
            return seq
        seq = opt_result.get("sequence", "")
        if seq:
            return seq
        candidates = opt_result.get("optimized_sequences", [])
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                return first.get("sequence", "")
            return str(first)
        return ""

    @staticmethod
    def _estimate_cdr_positions(sequence: str) -> list[int]:
        """Estimate CDR positions for antibody variable domains.

        Uses Chothia-like heuristics for VH and VL domains.
        Returns 1-indexed positions to MUTATE (CDR residues).
        """
        n = len(sequence)

        # For single-chain Fv (~110-130 aa), use standard Chothia CDR ranges
        if n <= 140:
            # VH-like CDR ranges (Chothia numbering approximation)
            cdrs = list(range(26, 33))   # CDR1: ~26-32
            cdrs += list(range(52, 57))  # CDR2: ~52-56
            cdrs += list(range(95, 103)) # CDR3: ~95-102
            return [p for p in cdrs if p <= n]

        # For scFv (~240-260 aa), VH + linker + VL
        if n <= 280:
            vh_len = n // 2  # approximate split
            cdrs = []
            # VH CDRs
            cdrs += list(range(26, 33))
            cdrs += list(range(52, 57))
            cdrs += list(range(95, 103))
            # VL CDRs (offset by VH length + linker ~15)
            vl_offset = vh_len + 15
            cdrs += [vl_offset + p for p in range(24, 35)]   # L-CDR1
            cdrs += [vl_offset + p for p in range(50, 57)]   # L-CDR2
            cdrs += [vl_offset + p for p in range(89, 98)]   # L-CDR3
            return [p for p in cdrs if p <= n]

        # For full-length antibody (>280 aa), use broader CDR windows
        # and also include CDRs from second chain (Fab or full IgG)
        cdrs = []
        for chain_offset in range(0, n, n // 2):
            cdrs += [chain_offset + p for p in range(26, 33)]
            cdrs += [chain_offset + p for p in range(52, 57)]
            cdrs += [chain_offset + p for p in range(95, 103)]
        return [p for p in cdrs if 1 <= p <= n]

    @staticmethod
    def _get_framework_positions(sequence: str, cdr_positions: list[int]) -> list[int]:
        """Return 1-indexed positions that are NOT CDRs (framework residues)."""
        cdr_set = set(cdr_positions)
        return [i for i in range(1, len(sequence) + 1) if i not in cdr_set]

    @staticmethod
    def _extract_backbones(bb_result: dict[str, Any]) -> list[dict[str, Any]]:
        backbones = bb_result.get("designs", [])
        if not backbones:
            backbones = bb_result.get("backbones", [])
        if not backbones and "pdb_path" in bb_result:
            backbones = [{"pdb_path": bb_result["pdb_path"]}]
        if not backbones:
            for key, val in bb_result.items():
                if isinstance(val, str) and val.endswith(".pdb"):
                    backbones.append({"pdb_path": val})
        return backbones

    @staticmethod
    def _resolve_backbone_pdb(bb: dict[str, Any] | str) -> str:
        if isinstance(bb, str):
            return bb
        for key in ("backbone_pdb", "backbone_pdb_file", "pdb_path"):
            candidate = bb.get(key, "")
            if candidate and Path(candidate).exists():
                return candidate
        return bb.get("pdb_path", "") or bb.get("backbone_pdb", "")

    @staticmethod
    def _find_backbone_pdbs_in_dir(output_dir: Path) -> list[str]:
        """Scan output dir for backbone PDB files (fallback for Docker paths)."""
        return sorted(str(f) for f in output_dir.glob("*backbone*.pdb"))

    # ------------------------------------------------------------------
    # Filtering & ranking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_composition(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove candidates with unreasonable amino acid composition.

        Checks:
        - Non-empty sequence
        - All characters are valid amino acids
        - No single AA > 50% of sequence
        - Alanine < 30% (common failure mode)
        - Top-3 AAs < 70% of sequence (catches L/P/E-dominated garbage)

        Returns empty list if all candidates fail — returning garbage is
        worse than returning 0 designs.
        """
        strict = []
        for c in candidates:
            seq = c.get("sequence", "")
            if not seq:
                continue
            seq_upper = seq.upper()

            # Check valid characters (hard requirement)
            if not all(ch in _VALID_AA for ch in seq_upper):
                continue

            counts = Counter(seq_upper)
            length = len(seq_upper)

            # No single AA > 50%
            if any(cnt / length > 0.5 for cnt in counts.values()):
                continue

            # Top-3 AAs < 70% (catches biased garbage like LPE repeats)
            top3_frac = sum(c for _, c in counts.most_common(3)) / length
            if top3_frac > 0.7:
                continue

            # Alanine < 30%
            if counts.get("A", 0) / length > 0.3:
                continue

            strict.append(c)

        if not strict and candidates:
            logger.warning(
                "Composition filter removed all %d candidates; "
                "returning empty list (no fallback to garbage)",
                len(candidates),
            )
        return strict

    @staticmethod
    def _filter_by_plddt(
        candidates: list[dict[str, Any]],
        threshold: float,
        reference_plddt: float | None = None,
    ) -> list[dict[str, Any]]:
        """Keep candidates above a pLDDT threshold.

        Two modes:
        - **Absolute** (de novo): pLDDT >= ``threshold`` (e.g. 75).
        - **Relative** (redesign): if ``reference_plddt`` is given, the
          effective threshold is ``max(reference_plddt - 10, 30)``.
          This prevents ESMFold family bias from killing all candidates
          for proteins that ESMFold simply can't predict well.

        If all candidates are below threshold, falls back to the top-3 by
        pLDDT so that long-sequence tasks still produce output.
        """
        if reference_plddt is not None:
            effective = max(reference_plddt - 10, 30)
            logger.info(
                "pLDDT filter: relative mode (ref=%.1f → threshold=%.1f)",
                reference_plddt, effective,
            )
        else:
            effective = threshold

        strict = [c for c in candidates if c.get("plddt", 0) >= effective]
        if not strict and candidates:
            # Fallback: return top-3 ranked by pLDDT
            ranked = sorted(candidates, key=lambda c: c.get("plddt", 0), reverse=True)
            strict = ranked[:3]
            best = ranked[0].get("plddt", 0)
            logger.warning(
                "pLDDT filter (>= %.0f) softened: returning top-%d "
                "(best pLDDT=%.1f) instead of empty",
                effective, len(strict), best,
            )
        return strict

    @staticmethod
    def _rank_composite(
        candidates: list[dict[str, Any]], weights: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Rank candidates by weighted composite score.

        Each metric is min-max normalized to [0, 1] before weighting.
        Missing values get score 0 for that metric.
        """
        if not candidates:
            return []

        # Collect min/max for normalization
        metric_ranges: dict[str, tuple[float, float]] = {}
        for key in weights:
            vals = [c.get(key, 0) for c in candidates if c.get(key, 0) != 0]
            if vals:
                metric_ranges[key] = (min(vals), max(vals))
            else:
                metric_ranges[key] = (0, 1)

        # Score each candidate
        for c in candidates:
            score = 0.0
            for key, weight in weights.items():
                val = c.get(key, 0)
                lo, hi = metric_ranges[key]
                if hi > lo:
                    normalized = (val - lo) / (hi - lo)
                else:
                    normalized = 1.0 if val > 0 else 0.0
                score += weight * normalized
            c["composite_score"] = score

        return sorted(candidates, key=lambda c: c.get("composite_score", 0), reverse=True)

    @staticmethod
    def _compute_sequence_identity(seq1: str, seq2: str) -> float:
        """Compute pairwise sequence identity (fraction of matching positions)."""
        if not seq1 or not seq2:
            return 0.0
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return 0.0
        matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        return matches / max_len

    @staticmethod
    def _select_diverse_top_n(
        ranked: list[dict[str, Any]],
        n: int,
        identity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Select top-N diverse designs by greedy clustering.

        Picks the best-scoring candidate, then skips candidates too similar
        (sequence identity > threshold) to already-selected ones.
        """
        if not ranked:
            return []
        if len(ranked) <= n:
            return ranked[:n]

        selected: list[dict[str, Any]] = [ranked[0]]

        for candidate in ranked[1:]:
            if len(selected) >= n:
                break
            seq = candidate.get("sequence", "")
            too_similar = False
            for chosen in selected:
                identity = HumanExpertAgent._compute_sequence_identity(
                    seq, chosen.get("sequence", "")
                )
                if identity > identity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(candidate)

        # If we couldn't fill n slots due to diversity constraint,
        # fill remaining from ranked list (skip already-selected)
        if len(selected) < n:
            selected_seqs = {s.get("sequence", "") for s in selected}
            for candidate in ranked:
                if len(selected) >= n:
                    break
                if candidate.get("sequence", "") not in selected_seqs:
                    selected.append(candidate)
                    selected_seqs.add(candidate.get("sequence", ""))

        return selected[:n]

    # ------------------------------------------------------------------
    # Pipeline 1: De Novo Binder (DNB)
    # ------------------------------------------------------------------

    async def _pipeline_dnb(
        self, task: Task, output_dir: Path
    ) -> list[dict[str, Any]]:
        cfg = self._config["de_novo_binder"]
        pdb_path = self._get_pdb_path(task, output_dir)
        hotspots = self._get_hotspot_residues(task)
        binder_length = self._get_binder_length(task)
        target_seq = self._get_target_sequence(task)

        # Suggest hotspots if not provided
        if not hotspots and pdb_path:
            hs_result = await self._call_tool(
                "suggest_hotspots",
                {
                    "target": pdb_path,
                    "chain_id": task.target.chain or "A",
                    "criteria": "exposed",
                },
                output_dir,
            )
            if "error" not in hs_result:
                for hs in hs_result.get("hotspots", [])[:5]:
                    res_id = hs.get("residue_id") or hs.get("residue", "")
                    if res_id:
                        hotspots.append(str(res_id))

        if not hotspots:
            chain = task.target.chain or "A"
            hotspots = [f"{chain}{i}" for i in range(30, 40)]

        # Stage 1: Oversampling with design_binder (adaptive count)
        num_designs = _adaptive_num_designs(
            binder_length, cfg["num_backbones"], target_length=len(target_seq)
        )
        logger.info(
            "  DNB: design_binder(num_designs=%d, binder_length=%d, target_length=%d)",
            num_designs, binder_length, len(target_seq),
        )

        binder_result = await self._call_tool(
            "design_binder",
            {
                "target_pdb": pdb_path,
                "hotspot_residues": hotspots,
                "num_designs": num_designs,
                "binder_length": binder_length,
            },
            output_dir,
        )

        raw_designs: list[dict[str, Any]] = []
        has_error = "error" in binder_result
        if not has_error:
            raw_designs = binder_result.get("designs", [])
            if not raw_designs and "sequence" in binder_result:
                raw_designs = [binder_result]
            logger.info("  DNB: raw_designs=%d", len(raw_designs))

        # Extract sequences from raw designs
        candidates: list[dict[str, Any]] = []
        for d in raw_designs:
            seq = d.get("sequence", "")
            if not seq:
                continue
            # design_binder may return target+binder concat; extract binder portion
            if len(seq) > binder_length * 1.5:
                seq = seq[-binder_length:]
                logger.warning(
                    "  DNB: extracted binder portion (%d residues) from concat seq (%d)",
                    len(seq), len(d.get("sequence", "")),
                )
            candidates.append({
                "sequence": seq,
                "pipeline_metrics": {k: v for k, v in d.items() if k != "sequence"},
            })
        logger.info(
            "  DNB: candidates=%d (will_retry=%s)",
            len(candidates), not candidates and num_designs > 3,
        )

        # Retry with fewer designs if first attempt returned no sequences
        if not candidates and num_designs > 3:
            logger.info("  DNB: retry design_binder with num_designs=3")
            retry_result = await self._call_tool(
                "design_binder",
                {
                    "target_pdb": pdb_path,
                    "hotspot_residues": hotspots,
                    "num_designs": 3,
                    "binder_length": binder_length,
                },
                output_dir,
            )
            if "error" not in retry_result:
                for d in retry_result.get("designs", []):
                    seq = d.get("sequence", "")
                    if seq:
                        candidates.append({
                            "sequence": seq,
                            "pipeline_metrics": {k: v for k, v in d.items() if k != "sequence"},
                        })
                if not candidates and "sequence" in retry_result:
                    candidates.append({"sequence": retry_result["sequence"]})

        # Rescue: backbone PDBs without sequences → call design_sequence
        if not candidates:
            bb_pdbs = []
            for d in raw_designs:
                bb_pdb = self._resolve_backbone_pdb(d)
                if bb_pdb and Path(bb_pdb).exists():
                    bb_pdbs.append(bb_pdb)
            if not bb_pdbs:
                bb_pdbs = self._find_backbone_pdbs_in_dir(output_dir)
                if bb_pdbs:
                    logger.info("  Rescue: found %d backbone PDBs in output dir", len(bb_pdbs))
            for bb_pdb in bb_pdbs[:5]:
                seq_result = await self._call_tool(
                    "design_sequence",
                    {"backbone_pdb": bb_pdb, "num_sequences": 1},
                    output_dir,
                )
                if "error" not in seq_result:
                    for s in seq_result.get("sequences", []):
                        seq = s.get("sequence", "") if isinstance(s, dict) else s
                        if seq and set(seq) <= _VALID_AA:
                            candidates.append({"sequence": seq, "source": "rescue_design_seq"})

        # Last resort: generate_backbone → design_sequence
        if not candidates:
            logger.info("  DNB: all binder attempts failed; fallback to backbone pipeline")
            fallback_designs = await self._fallback_backbone_design(
                binder_length, 3, output_dir
            )
            for d in fallback_designs:
                seq = d.get("sequence", "")
                if seq:
                    candidates.append({"sequence": seq, "source": "fallback_backbone"})

        # Stage 2: Composition filter (pure Python, free)
        candidates = self._filter_by_composition(candidates)

        if not candidates:
            return []

        # Stage 3: Monomer validation (ESMFold) — PARALLEL
        val_calls = [
            ("validate_design", {"sequence": c["sequence"], "predictor": "esmfold"})
            for c in candidates
        ]
        val_results = await self._call_tools_parallel(val_calls, output_dir)
        for c, val_result in zip(candidates, val_results):
            if "error" not in val_result:
                c["plddt"] = val_result.get("plddt", 0)
                c["ptm"] = val_result.get("ptm", 0)
            else:
                c["plddt"] = 0
                c["ptm"] = 0

        # Stage 4: pLDDT filter
        candidates = self._filter_by_plddt(candidates, cfg["filter_plddt"])

        if not candidates:
            return []

        # Stage 5: Complex prediction (AF2-Multimer) for top-N by pLDDT — PARALLEL
        candidates.sort(key=lambda c: c.get("plddt", 0), reverse=True)
        top_for_complex = candidates[: cfg["top_n_complex"]]

        cx_calls = []
        for c in top_for_complex:
            seqs = [c["sequence"]]
            chain_names = ["design"]
            if target_seq:
                seqs.append(target_seq)
                chain_names.append("target")
            cx_calls.append(("predict_complex", {"sequences": seqs, "chain_names": chain_names}))

        cx_results = await self._call_tools_parallel(cx_calls, output_dir, max_concurrency=2)
        for c, cx_result in zip(top_for_complex, cx_results):
            if "error" not in cx_result:
                c["iptm"] = cx_result.get("iptm", 0)
                c["cx_plddt"] = cx_result.get("plddt", 0)
            else:
                c["iptm"] = 0

        # Stage 6: Interface analysis for top-N by ipTM — PARALLEL
        top_for_complex.sort(key=lambda c: c.get("iptm", 0), reverse=True)
        top_for_interface = top_for_complex[: cfg["top_n_interface"]]

        iface_calls = [
            ("analyze_interface", {"sequence": c["sequence"], "target_sequence": target_seq or ""})
            for c in top_for_interface
        ]
        iface_results = await self._call_tools_parallel(iface_calls, output_dir)
        for c, iface_result in zip(top_for_interface, iface_results):
            if "error" not in iface_result:
                c["bsa"] = iface_result.get("bsa", 0)
                c["hbonds"] = iface_result.get("hbonds", 0)

        # Stage 7: Rosetta refinement for top-N — PARALLEL (relax + score per candidate)
        top_for_rosetta = top_for_interface[: cfg["top_n_rosetta"]]

        rosetta_calls = []
        for c in top_for_rosetta:
            rosetta_calls.append(("rosetta_relax", {"sequence": c["sequence"]}))
            rosetta_calls.append(("rosetta_score", {"sequence": c["sequence"]}))

        rosetta_results = await self._call_tools_parallel(rosetta_calls, output_dir)
        for i, c in enumerate(top_for_rosetta):
            relax_result = rosetta_results[i * 2]
            score_result = rosetta_results[i * 2 + 1]
            if "error" not in relax_result:
                c["relaxed_pdb"] = relax_result.get("relaxed_pdb", "")
            if "error" not in score_result:
                c["energy"] = score_result.get("total_score", 0)

        # Stage 8: Composite ranking
        ranked = self._rank_composite(
            top_for_rosetta,
            weights={"iptm": 0.4, "plddt": 0.3, "bsa": 0.2, "energy": 0.1},
        )

        # Stage 9: Diversity-weighted selection
        selected = self._select_diverse_top_n(ranked, cfg["final_n"])

        # Format output
        designs = []
        for i, c in enumerate(selected):
            designs.append({
                "name": f"design_{i + 1:03d}",
                "sequence": c["sequence"],
                "length": len(c["sequence"]),
                "quality": {
                    "pLDDT": c.get("plddt", 0),
                    "pTM": c.get("ptm", 0),
                    "ipTM": c.get("iptm", 0),
                },
                "pipeline_metrics": c.get("pipeline_metrics", {}),
            })

        return designs

    # ------------------------------------------------------------------
    # Pipeline 2a: Antibody CDR-focused Redesign (SQO-Ab)
    # ------------------------------------------------------------------

    async def _pipeline_sqo_antibody(
        self, task: Task, output_dir: Path
    ) -> list[dict[str, Any]]:
        """Antibody-specific redesign: CDR-focused mutation + ProteinMPNN.

        Two parallel strategies:
        A) optimize_sequence with fixed_positions = framework (only mutate CDRs)
        B) design_sequence (ProteinMPNN) with fixed_positions = framework

        Merge candidates, filter, and rank.
        """
        cfg = self._config["sequence_optimization"]
        pdb_path = self._get_pdb_path(task, output_dir)
        target_seq = self._get_target_sequence(task)

        if not target_seq:
            return []

        # Identify CDR vs framework positions
        cdr_positions = self._estimate_cdr_positions(target_seq)
        framework_positions = self._get_framework_positions(target_seq, cdr_positions)

        logger.info(
            "  SQO-Ab: seq_len=%d, CDR positions=%d, framework=%d",
            len(target_seq), len(cdr_positions), len(framework_positions),
        )

        # Stage 1: Predict reference structure (for optimize_sequence)
        pred_result = await self._call_tool(
            "predict_structure",
            {"sequence": target_seq, "predictor": "esmfold"},
            output_dir,
        )
        predicted_pdb = ""
        ref_plddt: float | None = None
        if "error" not in pred_result:
            predicted_pdb = (
                pred_result.get("predicted_structure_pdb")
                or pred_result.get("pdb_path")
                or ""
            )
            ref_plddt = pred_result.get("plddt", 0)
        structure_pdb = (
            predicted_pdb
            if predicted_pdb and Path(predicted_pdb).exists()
            else pdb_path
        )

        candidates: list[dict[str, Any]] = []

        # Strategy A: CDR-focused optimize_sequence (fix framework)
        seqs_per_target = _adaptive_seqs_per_target(len(target_seq), cfg["seqs_per_target"])
        opt_calls = [
            ("optimize_sequence", {
                "current_sequence": target_seq,
                "target_pdb": structure_pdb,
                "optimization_target": opt_target,
                "fixed_positions": framework_positions,
            })
            for opt_target in cfg["targets"]
            for _ in range(seqs_per_target)
        ]
        opt_targets_flat = [t for t in cfg["targets"] for _ in range(seqs_per_target)]
        logger.info(
            "  SQO-Ab strategy A: %d CDR-focused optimize calls",
            len(opt_calls),
        )
        opt_results = await self._call_tools_parallel(opt_calls, output_dir)
        for opt_result, opt_target in zip(opt_results, opt_targets_flat):
            if "error" not in opt_result:
                seq = self._extract_optimized_sequence(opt_result)
                if seq:
                    candidates.append({
                        "sequence": seq,
                        "opt_target": opt_target,
                        "strategy": "cdr_optimize",
                    })

        # Strategy B: ProteinMPNN redesign with fixed framework
        # Prefer the INPUT PDB (crystal structure) over ESMFold prediction
        # because ProteinMPNN works best on experimental structures
        mpnn_pdb = pdb_path if pdb_path and Path(pdb_path).exists() else structure_pdb
        if mpnn_pdb and Path(mpnn_pdb).exists():
            logger.info(
                "  SQO-Ab strategy B: ProteinMPNN redesign on %s (fix %d framework positions)",
                "input PDB" if mpnn_pdb == pdb_path else "ESMFold PDB",
                len(framework_positions),
            )
            mpnn_result = await self._call_tool(
                "design_sequence",
                {
                    "backbone_pdb": mpnn_pdb,
                    "num_sequences": 5,
                    "sampling_temp": 0.2,
                    "fixed_positions": framework_positions,
                    "validate": False,  # skip ESMFold inside ProteinMPNN (validate later)
                },
                output_dir,
            )
            if "error" not in mpnn_result:
                for d in mpnn_result.get("designs", []):
                    seq = d.get("sequence", "")
                    if seq and set(seq) <= _VALID_AA:
                        candidates.append({
                            "sequence": seq,
                            "plddt": d.get("plddt", 0),
                            "ptm": d.get("ptm", 0),
                            "strategy": "proteinmpnn",
                        })

        logger.info(
            "  SQO-Ab: %d candidates (optimize=%d, mpnn=%d)",
            len(candidates),
            sum(1 for c in candidates if c.get("strategy") == "cdr_optimize"),
            sum(1 for c in candidates if c.get("strategy") == "proteinmpnn"),
        )

        # Stage 3: Composition filter
        candidates = self._filter_by_composition(candidates)

        if not candidates:
            return []

        # Stage 4: Monomer validation — PARALLEL
        # Skip candidates that already have pLDDT from ProteinMPNN+ESMFold
        needs_validation = [c for c in candidates if not c.get("plddt")]
        if needs_validation:
            val_calls = [
                ("validate_design", {"sequence": c["sequence"], "predictor": "esmfold"})
                for c in needs_validation
            ]
            val_results = await self._call_tools_parallel(val_calls, output_dir)
            for c, val_result in zip(needs_validation, val_results):
                if "error" not in val_result:
                    c["plddt"] = val_result.get("plddt", 0)
                    c["ptm"] = val_result.get("ptm", 0)
                else:
                    c["plddt"] = 0

        # Stage 5: pLDDT filter (relative to WT for redesign)
        candidates = self._filter_by_plddt(
            candidates, cfg["filter_plddt"], reference_plddt=ref_plddt,
        )

        if not candidates:
            return []

        # Stage 6: Rosetta refinement — PARALLEL
        candidates.sort(key=lambda c: c.get("plddt", 0), reverse=True)
        top_for_rosetta = candidates[: _adaptive_top_n_rosetta(len(target_seq), cfg["top_n_rosetta"])]

        rosetta_calls = []
        for c in top_for_rosetta:
            rosetta_calls.append(("rosetta_relax", {"sequence": c["sequence"]}))
            rosetta_calls.append(("rosetta_score", {"sequence": c["sequence"]}))

        rosetta_results = await self._call_tools_parallel(rosetta_calls, output_dir)
        for i, c in enumerate(top_for_rosetta):
            relax_result = rosetta_results[i * 2]
            score_result = rosetta_results[i * 2 + 1]
            if "error" not in relax_result:
                c["relaxed_pdb"] = relax_result.get("relaxed_pdb", "")
            if "error" not in score_result:
                c["energy"] = score_result.get("total_score", 0)

        # Stage 7: Composite ranking
        ranked = self._rank_composite(
            top_for_rosetta,
            weights={"plddt": 0.4, "ptm": 0.3, "energy": 0.3},
        )

        # Stage 8: Diversity selection
        selected = self._select_diverse_top_n(ranked, cfg["final_n"])

        designs = []
        for i, c in enumerate(selected):
            designs.append({
                "name": f"design_{i + 1:03d}",
                "sequence": c["sequence"],
                "length": len(c["sequence"]),
                "quality": {
                    "pLDDT": c.get("plddt", 0),
                    "pTM": c.get("ptm", 0),
                },
                "strategy": c.get("strategy", "unknown"),
            })

        return designs

    # ------------------------------------------------------------------
    # Pipeline 2: Sequence Optimization (SQO)
    # ------------------------------------------------------------------

    async def _pipeline_sqo(
        self, task: Task, output_dir: Path
    ) -> list[dict[str, Any]]:
        cfg = self._config["sequence_optimization"]
        pdb_path = self._get_pdb_path(task, output_dir)
        target_seq = self._get_target_sequence(task)

        if not target_seq:
            return []

        # Stage 1: Predict reference structure
        pred_result = await self._call_tool(
            "predict_structure",
            {"sequence": target_seq, "predictor": "esmfold"},
            output_dir,
        )
        predicted_pdb = ""
        ref_plddt: float | None = None
        if "error" not in pred_result:
            predicted_pdb = (
                pred_result.get("predicted_structure_pdb")
                or pred_result.get("pdb_path")
                or ""
            )
            ref_plddt = pred_result.get("plddt", 0)
        structure_pdb = (
            predicted_pdb
            if predicted_pdb and Path(predicted_pdb).exists()
            else pdb_path
        )

        # Stage 2: Primary — ProteinMPNN redesign on backbone structure
        # ProteinMPNN produces structurally stable sequences by design;
        # optimize_sequence (ESM-2 marginal) is used as supplementary.
        candidates: list[dict[str, Any]] = []

        mpnn_pdb = pdb_path if pdb_path and Path(pdb_path).exists() else structure_pdb
        if mpnn_pdb and Path(mpnn_pdb).exists():
            logger.info("  SQO: ProteinMPNN redesign on %s",
                        "input PDB" if mpnn_pdb == pdb_path else "ESMFold PDB")
            mpnn_result = await self._call_tool(
                "design_sequence",
                {
                    "backbone_pdb": mpnn_pdb,
                    "num_sequences": 5,
                    "sampling_temp": 0.2,
                    "validate": False,
                },
                output_dir,
            )
            if "error" not in mpnn_result:
                for d in mpnn_result.get("designs", []):
                    seq = d.get("sequence", "")
                    if seq and set(seq) <= _VALID_AA:
                        candidates.append({
                            "sequence": seq,
                            "plddt": d.get("plddt", 0),
                            "ptm": d.get("ptm", 0),
                            "opt_target": "proteinmpnn",
                        })

        # Supplementary: optimize_sequence for additional diversity
        targets = cfg["targets"]
        seqs_per_target = _adaptive_seqs_per_target(len(target_seq), cfg["seqs_per_target"])
        logger.info(
            "  SQO: + %d optimize_sequence calls for diversity (seq_len=%d)",
            len(targets) * seqs_per_target, len(target_seq),
        )

        opt_calls = [
            ("optimize_sequence", {
                "current_sequence": target_seq,
                "target_pdb": structure_pdb,
                "optimization_target": opt_target,
            })
            for opt_target in targets
            for _ in range(seqs_per_target)
        ]
        opt_targets_flat = [t for t in targets for _ in range(seqs_per_target)]
        opt_results = await self._call_tools_parallel(opt_calls, output_dir)
        for opt_result, opt_target in zip(opt_results, opt_targets_flat):
            if "error" not in opt_result:
                seq = self._extract_optimized_sequence(opt_result)
                if seq:
                    candidates.append({"sequence": seq, "opt_target": opt_target})

        # Stage 3: Composition filter
        candidates = self._filter_by_composition(candidates)

        if not candidates:
            return []

        # Stage 4: Monomer validation — PARALLEL (skip if already validated by ProteinMPNN)
        needs_validation = [c for c in candidates if not c.get("plddt")]
        if needs_validation:
            val_calls = [
                ("validate_design", {"sequence": c["sequence"], "predictor": "esmfold"})
                for c in needs_validation
            ]
            val_results = await self._call_tools_parallel(val_calls, output_dir)
            for c, val_result in zip(needs_validation, val_results):
                if "error" not in val_result:
                    c["plddt"] = val_result.get("plddt", 0)
                    c["ptm"] = val_result.get("ptm", 0)
                else:
                    c["plddt"] = 0

        # Stage 5: pLDDT filter (relative to WT for redesign)
        candidates = self._filter_by_plddt(
            candidates, cfg["filter_plddt"], reference_plddt=ref_plddt,
        )

        if not candidates:
            return []

        # Stage 6: Stability scoring — PARALLEL
        stab_calls = [
            ("score_stability", {"sequence": c["sequence"], "reference_sequence": target_seq})
            for c in candidates
        ]
        stab_results = await self._call_tools_parallel(stab_calls, output_dir)
        for c, stab_result in zip(candidates, stab_results):
            if "error" not in stab_result:
                c["delta_pll"] = stab_result.get("delta_pll", 0)
                c["pll_score"] = stab_result.get("pll_score", 0)

        # Stage 7: Rosetta refinement — PARALLEL
        candidates.sort(key=lambda c: c.get("delta_pll", 0), reverse=True)
        top_for_rosetta = candidates[: _adaptive_top_n_rosetta(len(target_seq), cfg["top_n_rosetta"])]

        rosetta_calls = []
        for c in top_for_rosetta:
            rosetta_calls.append(("rosetta_relax", {"sequence": c["sequence"]}))
            rosetta_calls.append(("rosetta_score", {"sequence": c["sequence"]}))

        rosetta_results = await self._call_tools_parallel(rosetta_calls, output_dir)
        for i, c in enumerate(top_for_rosetta):
            relax_result = rosetta_results[i * 2]
            score_result = rosetta_results[i * 2 + 1]
            if "error" not in relax_result:
                c["relaxed_pdb"] = relax_result.get("relaxed_pdb", "")
            if "error" not in score_result:
                c["energy"] = score_result.get("total_score", 0)

        # Stage 8: Composite ranking
        ranked = self._rank_composite(
            top_for_rosetta,
            weights={"plddt": 0.4, "delta_pll": 0.3, "energy": 0.3},
        )

        # Stage 9: Diversity selection
        selected = self._select_diverse_top_n(ranked, cfg["final_n"])

        designs = []
        for i, c in enumerate(selected):
            designs.append({
                "name": f"design_{i + 1:03d}",
                "sequence": c["sequence"],
                "length": len(c["sequence"]),
                "quality": {
                    "pLDDT": c.get("plddt", 0),
                    "pTM": c.get("ptm", 0),
                },
                "stability": {
                    "pll_score": c.get("pll_score", 0),
                    "delta_pll": c.get("delta_pll", 0),
                },
            })

        return designs

    # ------------------------------------------------------------------
    # Pipeline 3: De Novo Backbone (DNK)
    # ------------------------------------------------------------------

    async def _pipeline_dnk(
        self, task: Task, output_dir: Path
    ) -> list[dict[str, Any]]:
        cfg = self._config["de_novo_backbone"]
        target_length = self._get_binder_length(task)

        # Stage 1: Generate backbones (adaptive count)
        num_backbones = _adaptive_num_designs(target_length, cfg["num_backbones"])
        logger.info("  DNK: generate_backbone(num_designs=%d, length=%d)", num_backbones, target_length)

        bb_result = await self._call_tool(
            "generate_backbone",
            {"length": target_length, "num_designs": num_backbones},
            output_dir,
        )

        if "error" in bb_result:
            return []

        backbones = self._extract_backbones(bb_result)

        # Stage 2: Sequence design for each backbone using ProteinMPNN
        candidates: list[dict[str, Any]] = []

        for bb in backbones:
            bb_pdb = self._resolve_backbone_pdb(bb)
            if not bb_pdb or not Path(bb_pdb).exists():
                continue

            ds_result = await self._call_tool(
                "design_sequence",
                {
                    "backbone_pdb": bb_pdb,
                    "num_sequences": cfg["seqs_per_backbone"],
                    "sampling_temp": 0.1,
                    "validate": True,
                },
                output_dir,
            )
            if "error" not in ds_result:
                for d in ds_result.get("designs", []):
                    seq = d.get("sequence", "")
                    if seq:
                        candidates.append({
                            "sequence": seq,
                            "backbone_pdb": bb_pdb,
                            "plddt": d.get("plddt", 0),
                            "ptm": d.get("ptm", 0),
                        })

        # Stage 3: Composition filter
        candidates = self._filter_by_composition(candidates)

        if not candidates:
            return []

        # Stage 4: Monomer validation (ESMFold) — PARALLEL
        val_calls = [
            ("validate_design", {"sequence": c["sequence"], "predictor": "esmfold"})
            for c in candidates
        ]
        val_results = await self._call_tools_parallel(val_calls, output_dir)
        for c, val_result in zip(candidates, val_results):
            if "error" not in val_result:
                c["plddt"] = val_result.get("plddt", 0)
                c["ptm"] = val_result.get("ptm", 0)
            else:
                c["plddt"] = 0

        # Stage 5: pLDDT filter
        candidates = self._filter_by_plddt(candidates, cfg["filter_plddt"])

        if not candidates:
            return []

        # Stage 6: Self-consistency check — PARALLEL
        sc_calls = [
            ("predict_structure", {"sequence": c["sequence"], "predictor": "esmfold"})
            for c in candidates
        ]
        sc_results = await self._call_tools_parallel(sc_calls, output_dir)
        for c, sc_result in zip(candidates, sc_results):
            if "error" not in sc_result:
                c["sc_plddt"] = sc_result.get("plddt", 0)
                c["sc_ptm"] = sc_result.get("ptm", 0)
            else:
                c["sc_plddt"] = 0
                c["sc_ptm"] = 0

        # Filter by self-consistency (high pLDDT from re-prediction)
        candidates = [
            c for c in candidates if c.get("sc_plddt", 0) >= cfg["filter_plddt"]
        ]

        if not candidates:
            return []

        # Stage 7: Rosetta scoring — PARALLEL
        candidates.sort(key=lambda c: c.get("plddt", 0), reverse=True)
        rosetta_n = _adaptive_top_n_rosetta(target_length, cfg["top_n_rosetta"])
        top_for_rosetta = candidates[: rosetta_n * 5]

        score_calls = [
            ("rosetta_score", {"sequence": c["sequence"]})
            for c in top_for_rosetta
        ]
        score_results = await self._call_tools_parallel(score_calls, output_dir)
        for c, score_result in zip(top_for_rosetta, score_results):
            if "error" not in score_result:
                c["energy"] = score_result.get("total_score", 0)

        # Stage 8: Energy minimize for top-N — PARALLEL
        top_for_rosetta.sort(key=lambda c: c.get("energy", 0))
        top_for_minimize = top_for_rosetta[: rosetta_n]

        em_calls = []
        em_indices = []
        for i, c in enumerate(top_for_minimize):
            bb_pdb = c.get("backbone_pdb", "")
            if bb_pdb and Path(bb_pdb).exists():
                em_calls.append(("energy_minimize", {"pdb_path": bb_pdb, "num_steps": 500}))
                em_indices.append(i)

        if em_calls:
            em_results = await self._call_tools_parallel(em_calls, output_dir)
            for idx, em_result in zip(em_indices, em_results):
                if "error" not in em_result:
                    top_for_minimize[idx]["rmsd"] = em_result.get("rmsd", 0)

        # Stage 9: Rank by pLDDT (primary)
        ranked = sorted(top_for_minimize, key=lambda c: c.get("plddt", 0), reverse=True)

        # Stage 10: Diversity selection (30% identity cluster threshold for backbones)
        selected = self._select_diverse_top_n(ranked, cfg["final_n"], identity_threshold=0.3)

        designs = []
        for i, c in enumerate(selected):
            designs.append({
                "name": f"design_{i + 1:03d}",
                "sequence": c["sequence"],
                "length": len(c["sequence"]),
                "backbone_pdb": c.get("backbone_pdb", ""),
                "quality": {
                    "pLDDT": c.get("plddt", 0),
                    "pTM": c.get("ptm", 0),
                    "scTM": c.get("sc_ptm", 0),
                },
            })

        return designs

    # ------------------------------------------------------------------
    # Pipeline 4: Complex Engineering (CPX)
    # ------------------------------------------------------------------

    async def _pipeline_cpx(
        self, task: Task, output_dir: Path
    ) -> list[dict[str, Any]]:
        cfg = self._config["complex_engineering"]
        pdb_path = self._get_pdb_path(task, output_dir)
        hotspots = self._get_hotspot_residues(task)
        binder_length = self._get_binder_length(task)
        target_seq = self._get_target_sequence(task)

        # Suggest hotspots if needed
        if not hotspots and pdb_path:
            hs_result = await self._call_tool(
                "suggest_hotspots",
                {
                    "target": pdb_path,
                    "chain_id": task.target.chain or "A",
                    "criteria": "exposed",
                },
                output_dir,
            )
            if "error" not in hs_result:
                for hs in hs_result.get("hotspots", [])[:5]:
                    res_id = hs.get("residue_id") or hs.get("residue", "")
                    if res_id:
                        hotspots.append(str(res_id))

        if not hotspots:
            chain = task.target.chain or "A"
            hotspots = [f"{chain}{i}" for i in range(30, 40)]

        # Stage 1: design_binder oversampling (adaptive count)
        num_designs = _adaptive_num_designs(
            binder_length, cfg["num_backbones"], target_length=len(target_seq)
        )
        logger.info(
            "  CPX: design_binder(num_designs=%d, binder_length=%d, target_length=%d)",
            num_designs, binder_length, len(target_seq),
        )

        binder_result = await self._call_tool(
            "design_binder",
            {
                "target_pdb": pdb_path,
                "hotspot_residues": hotspots,
                "num_designs": num_designs,
                "binder_length": binder_length,
            },
            output_dir,
        )

        raw_designs: list[dict[str, Any]] = []
        if "error" not in binder_result:
            raw_designs = binder_result.get("designs", [])
            if not raw_designs and "sequence" in binder_result:
                raw_designs = [binder_result]

        # Extract sequences from raw designs
        candidates: list[dict[str, Any]] = []
        for d in raw_designs:
            seq = d.get("sequence", "")
            if not seq:
                continue
            # design_binder may return target+binder concat; extract binder portion
            if len(seq) > binder_length * 1.5:
                seq = seq[-binder_length:]
                logger.warning(
                    "  CPX: extracted binder portion (%d residues) from concat seq (%d)",
                    len(seq), len(d.get("sequence", "")),
                )
            candidates.append({
                "sequence": seq,
                "pipeline_metrics": {k: v for k, v in d.items() if k != "sequence"},
            })

        # Retry with fewer designs if first attempt returned no sequences
        if not candidates and num_designs > 3:
            logger.info("  CPX: retry design_binder with num_designs=3")
            retry_result = await self._call_tool(
                "design_binder",
                {
                    "target_pdb": pdb_path,
                    "hotspot_residues": hotspots,
                    "num_designs": 3,
                    "binder_length": binder_length,
                },
                output_dir,
            )
            if "error" not in retry_result:
                for d in retry_result.get("designs", []):
                    seq = d.get("sequence", "")
                    if seq:
                        candidates.append({
                            "sequence": seq,
                            "pipeline_metrics": {k: v for k, v in d.items() if k != "sequence"},
                        })
                if not candidates and "sequence" in retry_result:
                    candidates.append({"sequence": retry_result["sequence"]})

        # Rescue: backbone PDBs without sequences → call design_sequence
        if not candidates:
            bb_pdbs = []
            for d in raw_designs:
                bb_pdb = self._resolve_backbone_pdb(d)
                if bb_pdb and Path(bb_pdb).exists():
                    bb_pdbs.append(bb_pdb)
            if not bb_pdbs:
                bb_pdbs = self._find_backbone_pdbs_in_dir(output_dir)
                if bb_pdbs:
                    logger.info("  Rescue: found %d backbone PDBs in output dir", len(bb_pdbs))
            for bb_pdb in bb_pdbs[:5]:
                seq_result = await self._call_tool(
                    "design_sequence",
                    {"backbone_pdb": bb_pdb, "num_sequences": 1},
                    output_dir,
                )
                if "error" not in seq_result:
                    for s in seq_result.get("sequences", []):
                        seq = s.get("sequence", "") if isinstance(s, dict) else s
                        if seq and set(seq) <= _VALID_AA:
                            candidates.append({"sequence": seq, "source": "rescue_design_seq"})

        # Last resort: generate_backbone → design_sequence
        if not candidates:
            logger.info("  CPX: all binder attempts failed; fallback to backbone pipeline")
            fallback_designs = await self._fallback_backbone_design(
                binder_length, 3, output_dir
            )
            for d in fallback_designs:
                seq = d.get("sequence", "")
                if seq:
                    candidates.append({"sequence": seq, "source": "fallback_backbone"})

        # Stage 2: Composition filter
        candidates = self._filter_by_composition(candidates)

        if not candidates:
            return []

        # Stage 3: Monomer validation — PARALLEL
        val_calls = [
            ("validate_design", {"sequence": c["sequence"], "predictor": "esmfold"})
            for c in candidates
        ]
        val_results = await self._call_tools_parallel(val_calls, output_dir)
        for c, val_result in zip(candidates, val_results):
            if "error" not in val_result:
                c["plddt"] = val_result.get("plddt", 0)
                c["ptm"] = val_result.get("ptm", 0)
            else:
                c["plddt"] = 0

        # Stage 4: pLDDT filter
        candidates = self._filter_by_plddt(candidates, cfg["filter_plddt"])

        if not candidates:
            return []

        # Stage 5: Complex prediction — PARALLEL
        candidates.sort(key=lambda c: c.get("plddt", 0), reverse=True)
        top_for_complex = candidates[: cfg["top_n_complex"]]

        cx_calls = []
        for c in top_for_complex:
            seqs = [c["sequence"]]
            chain_names = ["design"]
            if target_seq:
                seqs.append(target_seq)
                chain_names.append("target")
            cx_calls.append(("predict_complex", {"sequences": seqs, "chain_names": chain_names}))

        cx_results = await self._call_tools_parallel(cx_calls, output_dir, max_concurrency=2)
        for c, cx_result in zip(top_for_complex, cx_results):
            if "error" not in cx_result:
                c["iptm"] = cx_result.get("iptm", 0)
            else:
                c["iptm"] = 0

        # Stage 6: Interface analysis — PARALLEL
        top_for_complex.sort(key=lambda c: c.get("iptm", 0), reverse=True)
        top_for_interface = top_for_complex[: cfg["top_n_interface"]]

        iface_calls = [
            ("analyze_interface", {"sequence": c["sequence"], "target_sequence": target_seq or ""})
            for c in top_for_interface
        ]
        iface_results = await self._call_tools_parallel(iface_calls, output_dir)
        for c, iface_result in zip(top_for_interface, iface_results):
            if "error" not in iface_result:
                c["bsa"] = iface_result.get("bsa", 0)
                c["interface_score"] = iface_result.get("interface_score", 0)

        # Stage 7: Rosetta refinement — PARALLEL
        top_for_rosetta = top_for_interface[: cfg["top_n_rosetta"]]

        rosetta_calls = []
        for c in top_for_rosetta:
            rosetta_calls.append(("rosetta_relax", {"sequence": c["sequence"]}))
            rosetta_calls.append(("rosetta_score", {"sequence": c["sequence"]}))

        rosetta_results = await self._call_tools_parallel(rosetta_calls, output_dir)
        for i, c in enumerate(top_for_rosetta):
            relax_result = rosetta_results[i * 2]
            score_result = rosetta_results[i * 2 + 1]
            if "error" not in relax_result:
                c["relaxed_pdb"] = relax_result.get("relaxed_pdb", "")
            if "error" not in score_result:
                c["energy"] = score_result.get("total_score", 0)

        # Stage 8: Composite ranking
        ranked = self._rank_composite(
            top_for_rosetta,
            weights={"iptm": 0.4, "plddt": 0.3, "interface_score": 0.2, "energy": 0.1},
        )

        # Stage 9: Diversity selection
        selected = self._select_diverse_top_n(ranked, cfg["final_n"])

        designs = []
        for i, c in enumerate(selected):
            designs.append({
                "name": f"design_{i + 1:03d}",
                "sequence": c["sequence"],
                "length": len(c["sequence"]),
                "quality": {
                    "pLDDT": c.get("plddt", 0),
                    "pTM": c.get("ptm", 0),
                    "ipTM": c.get("iptm", 0),
                },
                "pipeline_metrics": c.get("pipeline_metrics", {}),
            })

        return designs

    # ------------------------------------------------------------------
    # Pipeline 5: Conformational Design (CFD)
    # ------------------------------------------------------------------

    async def _pipeline_cfd(
        self, task: Task, output_dir: Path
    ) -> list[dict[str, Any]]:
        cfg = self._config["conformational_design"]
        pdb_path = self._get_pdb_path(task, output_dir)
        target_seq = self._get_target_sequence(task)

        if not target_seq:
            return []

        # Stage 1: Predict reference structure
        pred_result = await self._call_tool(
            "predict_structure",
            {"sequence": target_seq, "predictor": "esmfold"},
            output_dir,
        )
        predicted_pdb = ""
        ref_plddt: float | None = None
        if "error" not in pred_result:
            predicted_pdb = (
                pred_result.get("predicted_structure_pdb")
                or pred_result.get("pdb_path")
                or ""
            )
            ref_plddt = pred_result.get("plddt", 0)
        structure_pdb = (
            predicted_pdb
            if predicted_pdb and Path(predicted_pdb).exists()
            else pdb_path
        )

        # Stage 2: Primary — ProteinMPNN redesign on backbone structure
        candidates: list[dict[str, Any]] = []

        mpnn_pdb = pdb_path if pdb_path and Path(pdb_path).exists() else structure_pdb
        if mpnn_pdb and Path(mpnn_pdb).exists():
            logger.info("  CFD: ProteinMPNN redesign on %s",
                        "input PDB" if mpnn_pdb == pdb_path else "ESMFold PDB")
            mpnn_result = await self._call_tool(
                "design_sequence",
                {
                    "backbone_pdb": mpnn_pdb,
                    "num_sequences": 5,
                    "sampling_temp": 0.2,
                    "validate": False,
                },
                output_dir,
            )
            if "error" not in mpnn_result:
                for d in mpnn_result.get("designs", []):
                    seq = d.get("sequence", "")
                    if seq and set(seq) <= _VALID_AA:
                        candidates.append({
                            "sequence": seq,
                            "plddt": d.get("plddt", 0),
                            "ptm": d.get("ptm", 0),
                            "opt_target": "proteinmpnn",
                        })

        # Supplementary: optimize_sequence for additional diversity
        targets = cfg["targets"]
        seqs_per_target = _adaptive_seqs_per_target(len(target_seq), cfg["seqs_per_target"])
        logger.info(
            "  CFD: + %d optimize_sequence calls for diversity (seq_len=%d)",
            len(targets) * seqs_per_target, len(target_seq),
        )

        opt_calls = [
            ("optimize_sequence", {
                "current_sequence": target_seq,
                "target_pdb": structure_pdb,
                "optimization_target": opt_target,
            })
            for opt_target in targets
            for _ in range(seqs_per_target)
        ]
        opt_targets_flat = [t for t in targets for _ in range(seqs_per_target)]
        opt_results = await self._call_tools_parallel(opt_calls, output_dir)
        for opt_result, opt_target in zip(opt_results, opt_targets_flat):
            if "error" not in opt_result:
                seq = self._extract_optimized_sequence(opt_result)
                if seq:
                    candidates.append({"sequence": seq, "opt_target": opt_target})

        # Stage 3: Composition filter
        candidates = self._filter_by_composition(candidates)

        if not candidates:
            return []

        # Stage 4: Monomer validation — PARALLEL (skip already-validated ProteinMPNN designs)
        needs_validation = [c for c in candidates if not c.get("plddt")]
        if needs_validation:
            val_calls = [
                ("validate_design", {"sequence": c["sequence"], "predictor": "esmfold"})
                for c in needs_validation
            ]
            val_results = await self._call_tools_parallel(val_calls, output_dir)
            for c, val_result in zip(needs_validation, val_results):
                if "error" not in val_result:
                    c["plddt"] = val_result.get("plddt", 0)
                    c["ptm"] = val_result.get("ptm", 0)
                else:
                    c["plddt"] = 0

        # Stage 5: pLDDT filter (relative to WT for redesign)
        candidates = self._filter_by_plddt(
            candidates, cfg["filter_plddt"], reference_plddt=ref_plddt,
        )

        if not candidates:
            return []

        # Stage 6: Energy minimization (once on reference) + stability scoring — PARALLEL
        em_energy_delta = 0.0
        em_rmsd = 0.0
        if structure_pdb and Path(structure_pdb).exists():
            em_result = await self._call_tool(
                "energy_minimize",
                {"pdb_path": structure_pdb, "num_steps": 500, "solvent": "implicit"},
                output_dir,
            )
            if "error" not in em_result:
                em_energy_delta = em_result.get("energy_delta_kj", 0)
                em_rmsd = em_result.get("rmsd", 0)
        for c in candidates:
            c["energy_delta"] = em_energy_delta
            c["rmsd"] = em_rmsd

        # Stage 7: Stability scoring — PARALLEL
        stab_calls = [
            ("score_stability", {"sequence": c["sequence"], "reference_sequence": target_seq})
            for c in candidates
        ]
        stab_results = await self._call_tools_parallel(stab_calls, output_dir)
        for c, stab_result in zip(candidates, stab_results):
            if "error" not in stab_result:
                c["delta_pll"] = stab_result.get("delta_pll", 0)
                c["pll_score"] = stab_result.get("pll_score", 0)

        # Stage 8: Rosetta refinement — PARALLEL
        candidates.sort(key=lambda c: c.get("plddt", 0), reverse=True)
        top_for_rosetta = candidates[: _adaptive_top_n_rosetta(len(target_seq), cfg["top_n_rosetta"])]

        rosetta_calls = []
        for c in top_for_rosetta:
            rosetta_calls.append(("rosetta_relax", {"sequence": c["sequence"]}))
            rosetta_calls.append(("rosetta_score", {"sequence": c["sequence"]}))

        rosetta_results = await self._call_tools_parallel(rosetta_calls, output_dir)
        for i, c in enumerate(top_for_rosetta):
            relax_result = rosetta_results[i * 2]
            score_result = rosetta_results[i * 2 + 1]
            if "error" not in relax_result:
                c["relaxed_pdb"] = relax_result.get("relaxed_pdb", "")
            if "error" not in score_result:
                c["energy"] = score_result.get("total_score", 0)

        # Stage 9: Composite ranking
        ranked = self._rank_composite(
            top_for_rosetta,
            weights={"plddt": 0.45, "ptm": 0.25, "delta_pll": 0.3},
        )

        # Stage 10: Diversity selection
        selected = self._select_diverse_top_n(ranked, cfg["final_n"])

        designs = []
        for i, c in enumerate(selected):
            designs.append({
                "name": f"design_{i + 1:03d}",
                "sequence": c["sequence"],
                "length": len(c["sequence"]),
                "quality": {
                    "pLDDT": c.get("plddt", 0),
                    "pTM": c.get("ptm", 0),
                },
                "stability": {
                    "pll_score": c.get("pll_score", 0),
                    "delta_pll": c.get("delta_pll", 0),
                },
                "energy": {
                    "energy_delta": c.get("energy_delta", 0),
                    "rmsd": c.get("rmsd", 0),
                },
            })

        return designs

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------

    def _write_outputs(
        self,
        designs: list[dict[str, Any]],
        task: Task,
        output_dir: Path,
        elapsed: float,
    ) -> None:
        """Write designed_sequences.fasta and metrics.json."""
        # FASTA
        fasta_path = output_dir / "designed_sequences.fasta"
        fasta_lines: list[str] = []
        for d in designs:
            name = d.get("name", "design")
            seq = d.get("sequence", "")
            if seq:
                fasta_lines.append(f">{name}")
                for j in range(0, len(seq), 80):
                    fasta_lines.append(seq[j : j + 80])
        fasta_path.write_text("\n".join(fasta_lines) + "\n" if fasta_lines else "")

        # Metrics JSON
        metrics_path = output_dir / "metrics.json"
        metrics: dict[str, Any] = {
            "task_id": task.task_id,
            "agent": "human-expert-agent",
            "designs": [],
            "summary": {
                "total_designs": len(designs),
                "total_time_seconds": round(elapsed, 1),
                "tools_used": list(set(self._tools_used)),
                "total_tool_calls": len(self._tool_call_log),
            },
        }

        for d in designs:
            entry: dict[str, Any] = {
                "id": d.get("name", ""),
                "sequence": d.get("sequence", ""),
                "length": d.get("length", 0),
            }
            if "quality" in d:
                entry["quality"] = d["quality"]
            if "stability" in d:
                entry["stability"] = d["stability"]
            if "energy" in d:
                entry["energy"] = d["energy"]
            if "pipeline_metrics" in d:
                entry["pipeline_metrics"] = d["pipeline_metrics"]
            metrics["designs"].append(entry)

        # Summary stats
        plddts = [
            d.get("quality", {}).get("pLDDT", 0)
            for d in designs
            if d.get("quality")
        ]
        if plddts:
            metrics["summary"]["avg_pLDDT"] = round(sum(plddts) / len(plddts), 1)

        iptms = [
            d.get("quality", {}).get("ipTM", 0)
            for d in designs
            if d.get("quality", {}).get("ipTM")
        ]
        if iptms:
            metrics["summary"]["avg_ipTM"] = round(sum(iptms) / len(iptms), 3)

        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

        logger.info(
            "Wrote %d designs to %s (avg pLDDT: %s)",
            len(designs),
            output_dir,
            metrics["summary"].get("avg_pLDDT", "N/A"),
        )
