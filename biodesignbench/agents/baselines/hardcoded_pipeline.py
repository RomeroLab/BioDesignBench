"""Hardcoded pipeline baseline for BioDesignBench Tier 2 design tasks.

Each DesignTaskType maps to a fixed tool chain:

    de_novo_binder:       design_binder → validate_design
    sequence_optimization: optimize_sequence → validate_design → score_stability
    de_novo_backbone:     generate_backbone → optimize_sequence → validate_design
    complex_engineering:  generate_backbone → optimize_sequence → predict_complex
    conformational_design: predict_structure → optimize_sequence → energy_minimize → validate_design

This deterministic baseline uses NO LLM reasoning — arguments are extracted
directly from task JSON fields and fed through a fixed sequence of MCP tool calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput, ToolCallEntry, summarize_tool_args
from biodesignbench.tasks.schema import Task, TaskTier
from biodesignbench.taxonomy import DesignApproach, MolecularSubject, get_category

logger = logging.getLogger(__name__)

NUM_DESIGNS = 3


def _adaptive_num_designs_for_timeout(seq_len: int) -> int:
    """Reduce NUM_DESIGNS for long sequences to fit within 120min timeout.

    ESM-2 inference in optimize_sequence scales ~O(L) per mutation scan,
    with 3 rounds × 5 mutations = 15 forward passes per call.
    Empirical timing: ~0.3s × seq_len per optimize_sequence call.
    """
    if seq_len <= 200:
        return 3
    elif seq_len <= 400:
        return 2
    else:
        return 1


class HardcodedPipelineAgent(AgentInterface):
    """Deterministic pipeline baseline that maps task type → fixed tool chain.

    Requires a ``ProteinDesignToolProvider`` to call MCP tools.
    Uses USER mode (needs composite tools: design_binder, optimize_sequence).
    """

    def __init__(
        self,
        tool_provider: Any | None = None,
        docker_image: str = "protein-design-mcp:full",
    ):
        self.tool_provider = tool_provider
        self.docker_image = docker_image
        self._tool_call_log: list[ToolCallEntry] = []
        self._tools_used: list[str] = []
        self._iteration: int = 0

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="hardcoded-pipeline",
            name="Hardcoded Pipeline",
            version="1.0.0",
            description="Deterministic tool-chain baseline (no LLM reasoning)",
            provider="baseline",
            model="none",
            is_bio_specific=True,
            capabilities=["deterministic", "protein_design_tools"],
        )

    def setup(self) -> None:
        if self.tool_provider is None:
            raise RuntimeError(
                "HardcodedPipelineAgent requires a ProteinDesignToolProvider. "
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
                reasoning_trace="Hardcoded pipeline only supports Tier 2 design tasks.",
            )

        if output_dir is None:
            import tempfile
            output_dir = Path(tempfile.mkdtemp(prefix="hardcoded_"))
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

        # Dispatch: approach + subject → specific pipeline
        # CFD tasks need the conformational design pipeline regardless of
        # their 2×5 subject classification.
        is_cfd_task = task.task_id.startswith("cfd_")

        if approach == DesignApproach.DE_NOVO:
            if is_cfd_task:
                pipeline_fn = self._pipeline_cfd  # conformational/functional design
            elif subject == MolecularSubject.SCAFFOLD and not task.target.binding_site_residues:
                pipeline_fn = self._pipeline_dnk  # unconditional backbone
            elif subject in (MolecularSubject.BINDER, MolecularSubject.ANTIBODY):
                pipeline_fn = self._pipeline_dnb  # binder generation
            elif subject in (MolecularSubject.ENZYME, MolecularSubject.FLUORESCENT_PROTEIN):
                pipeline_fn = self._pipeline_cfd  # conformational/functional design
            else:
                pipeline_fn = self._pipeline_cpx  # complex/interface generation
        else:
            # REDESIGN
            pipeline_fn = self._pipeline_sqo  # sequence optimization

        logger.info(
            "Running %s/%s pipeline for %s",
            approach.short, subject.short, task.task_id,
        )

        designs = await pipeline_fn(task, output_dir)
        elapsed = time.time() - start_time

        # Write output files
        self._write_outputs(designs, task, output_dir, elapsed)

        return AgentOutput(
            designs=designs,
            output_dir=str(output_dir),
            tools_used=list(set(self._tools_used)),
            tool_call_log=self._tool_call_log,
            api_calls=len(self._tool_call_log),
            iterations=self._iteration,
            reasoning_trace=(
                f"Hardcoded {approach.short}/{subject.short} pipeline: "
                f"{' → '.join(t.tool for t in self._tool_call_log)}"
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
        """Call an MCP tool and record the call."""
        self._iteration += 1
        self._tools_used.append(name)

        summary, values = summarize_tool_args(args)
        try:
            result = await self.tool_provider.call_tool(
                name, args, output_dir, mode="user",
            )
            self._tool_call_log.append(ToolCallEntry(
                tool=name,
                iteration=self._iteration,
                success=True,
                args_summary=summary,
                args_values=values,
            ))
            logger.info("  [%d] %s → OK", self._iteration, name)
            return result
        except Exception as exc:
            error_msg = str(exc)[:200]
            self._tool_call_log.append(ToolCallEntry(
                tool=name,
                iteration=self._iteration,
                success=False,
                error=error_msg,
                args_summary=summary,
                args_values=values,
            ))
            logger.warning("  [%d] %s → FAILED: %s", self._iteration, name, error_msg)
            return {"error": error_msg}

    # ------------------------------------------------------------------
    # Argument extraction helpers
    # ------------------------------------------------------------------

    def _get_pdb_path(self, task: Task, output_dir: Path) -> str:
        """Resolve the target PDB path."""
        pdb_id = task.target.pdb_id
        if pdb_id:
            candidate = output_dir / f"{pdb_id.lower()}.pdb"
            if candidate.exists():
                return str(candidate.resolve())
        # Fallback: find any PDB in output_dir
        for f in output_dir.iterdir():
            if f.suffix == ".pdb" and f.is_file():
                return str(f.resolve())
        return ""

    def _get_hotspot_residues(self, task: Task) -> list[str]:
        """Format binding_site_residues as chain+resid strings."""
        chain = task.target.chain or "A"
        residues = task.target.binding_site_residues or []
        return [f"{chain}{r}" for r in residues]

    def _get_binder_length(self, task: Task) -> int:
        """Get target binder length from design constraints."""
        if task.design_constraints.length_range:
            lo, hi = task.design_constraints.length_range
            return (lo + hi) // 2
        return 70  # reasonable default

    def _get_target_sequence(self, task: Task) -> str:
        """Get target protein sequence."""
        return task.target.sequence or ""

    @staticmethod
    def _extract_backbones(bb_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract backbone entries from generate_backbone response.

        The MCP server returns ``{"designs": [{"pdb_path": ..., "backbone_pdb": ...}]}``.
        After PDB externalization, ``backbone_pdb`` becomes a file path.
        This helper normalises all known formats into ``[{"pdb_path": ...}]``.
        """
        # Primary: MCP returns "designs" list
        backbones = bb_result.get("designs", [])
        # Fallback: older format used "backbones"
        if not backbones:
            backbones = bb_result.get("backbones", [])
        # Single design at top level
        if not backbones and "pdb_path" in bb_result:
            backbones = [{"pdb_path": bb_result["pdb_path"]}]
        # Externalized PDB at top level
        if not backbones:
            for key, val in bb_result.items():
                if isinstance(val, str) and val.endswith(".pdb"):
                    backbones.append({"pdb_path": val})
        return backbones

    @staticmethod
    def _resolve_backbone_pdb(bb: dict[str, Any] | str) -> str:
        """Get the PDB file path from a backbone entry.

        After externalization the ``backbone_pdb`` key holds the written file
        path.  The original ``pdb_path`` key points to the Docker-internal
        path which may not exist on the host.  Try both.
        """
        if isinstance(bb, str):
            return bb
        # Prefer externalized path (always exists on host)
        for key in ("backbone_pdb", "backbone_pdb_file", "pdb_path"):
            candidate = bb.get(key, "")
            if candidate and Path(candidate).exists():
                return candidate
        # Last resort: return whatever pdb_path says
        return bb.get("pdb_path", "") or bb.get("backbone_pdb", "")

    @staticmethod
    def _extract_optimized_sequence(opt_result: dict[str, Any]) -> str:
        """Extract the best sequence from an optimize_sequence response.

        The MCP server returns ``{"optimized_sequence": "..."}``.
        This helper covers that plus legacy key variants.
        """
        # Primary: MCP returns "optimized_sequence" (singular)
        seq = opt_result.get("optimized_sequence", "")
        if seq:
            return seq
        # Legacy variants
        seq = opt_result.get("best_sequence", "")
        if seq:
            return seq
        seq = opt_result.get("sequence", "")
        if seq:
            return seq
        # List format
        candidates = opt_result.get("optimized_sequences", [])
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                return first.get("sequence", "")
            return str(first)
        return ""

    # ------------------------------------------------------------------
    # Pipeline 1: De Novo Binder
    # design_binder → validate_design
    # ------------------------------------------------------------------

    async def _pipeline_dnb(
        self, task: Task, output_dir: Path,
    ) -> list[dict[str, Any]]:
        pdb_path = self._get_pdb_path(task, output_dir)
        hotspots = self._get_hotspot_residues(task)
        binder_length = self._get_binder_length(task)

        # If no hotspots defined, use suggest_hotspots first
        if not hotspots and pdb_path:
            hs_result = await self._call_tool("suggest_hotspots", {
                "target": pdb_path,
                "chain_id": task.target.chain or "A",
                "criteria": "exposed",
            }, output_dir)
            if "error" not in hs_result:
                # Extract hotspot residues from result
                for hs in hs_result.get("hotspots", [])[:5]:
                    res_id = hs.get("residue_id") or hs.get("residue", "")
                    if res_id:
                        hotspots.append(str(res_id))

        if not hotspots:
            # Last resort: use first few residues of the chain
            chain = task.target.chain or "A"
            hotspots = [f"{chain}{i}" for i in range(30, 40)]

        # Step 1: design_binder
        binder_result = await self._call_tool("design_binder", {
            "target_pdb": pdb_path,
            "hotspot_residues": hotspots,
            "num_designs": NUM_DESIGNS,
            "binder_length": binder_length,
        }, output_dir)

        if "error" in binder_result:
            return []

        # Extract designed sequences
        designs = []
        raw_designs = binder_result.get("designs", [])
        if not raw_designs and "sequence" in binder_result:
            raw_designs = [binder_result]

        for i, d in enumerate(raw_designs[:NUM_DESIGNS]):
            seq = d.get("sequence", "")
            if not seq:
                continue

            design_entry: dict[str, Any] = {
                "name": f"design_{i+1:03d}",
                "sequence": seq,
                "length": len(seq),
                "pipeline_metrics": {k: v for k, v in d.items() if k != "sequence"},
            }

            # Step 2: validate_design
            val_result = await self._call_tool("validate_design", {
                "sequence": seq,
                "predictor": "esmfold",
            }, output_dir)

            if "error" not in val_result:
                design_entry["quality"] = {
                    "pLDDT": val_result.get("plddt", 0),
                    "pTM": val_result.get("ptm", 0),
                }

            designs.append(design_entry)

        return designs

    # ------------------------------------------------------------------
    # Pipeline 2: Sequence Optimization
    # optimize_sequence → validate_design → score_stability
    # ------------------------------------------------------------------

    async def _pipeline_sqo(
        self, task: Task, output_dir: Path,
    ) -> list[dict[str, Any]]:
        pdb_path = self._get_pdb_path(task, output_dir)
        target_seq = self._get_target_sequence(task)

        if not target_seq:
            return []

        # Step 1: ProteinMPNN redesign on input PDB (structurally stable sequences)
        mpnn_pdb = pdb_path
        if not mpnn_pdb or not Path(mpnn_pdb).exists():
            # Predict structure if no input PDB
            pred_result = await self._call_tool("predict_structure", {
                "sequence": target_seq, "predictor": "esmfold",
            }, output_dir)
            if "error" not in pred_result:
                mpnn_pdb = pred_result.get("predicted_structure_pdb") or pred_result.get("pdb_path", "")

        if not mpnn_pdb or not Path(mpnn_pdb).exists():
            return []

        ds_result = await self._call_tool("design_sequence", {
            "backbone_pdb": mpnn_pdb,
            "num_sequences": NUM_DESIGNS,
            "sampling_temp": 0.2,
            "validate": True,
        }, output_dir)

        if "error" in ds_result:
            return []

        ds_designs = ds_result.get("designs", [])
        designs = []
        for i, d in enumerate(ds_designs[:NUM_DESIGNS]):
            seq = d.get("sequence", "")
            if not seq:
                continue

            design_entry: dict[str, Any] = {
                "name": f"design_{i+1:03d}",
                "sequence": seq,
                "length": len(seq),
                "quality": {
                    "pLDDT": d.get("plddt", 0),
                    "pTM": d.get("ptm", 0),
                },
            }

            # Step 2: score_stability
            stab_result = await self._call_tool("score_stability", {
                "sequence": seq,
                "reference_sequence": target_seq,
            }, output_dir)

            if "error" not in stab_result:
                design_entry["stability"] = {
                    "pll_score": stab_result.get("pll_score", 0),
                    "delta_pll": stab_result.get("delta_pll", 0),
                    "mean_pll": stab_result.get("mean_pll", 0),
                }

            designs.append(design_entry)

        return designs

    # ------------------------------------------------------------------
    # Pipeline 3: De Novo Backbone
    # generate_backbone → optimize_sequence → validate_design
    # ------------------------------------------------------------------

    async def _pipeline_dnk(
        self, task: Task, output_dir: Path,
    ) -> list[dict[str, Any]]:
        target_length = self._get_binder_length(task)

        # Step 1: generate_backbone (unconditional RFdiffusion)
        bb_result = await self._call_tool("generate_backbone", {
            "length": target_length,
            "num_designs": NUM_DESIGNS,
        }, output_dir)

        if "error" in bb_result:
            return []

        backbones = self._extract_backbones(bb_result)

        designs = []
        for i, bb in enumerate(backbones[:NUM_DESIGNS]):
            bb_pdb = self._resolve_backbone_pdb(bb)
            if not bb_pdb or not Path(bb_pdb).exists():
                continue

            # Step 2: design_sequence (ProteinMPNN on backbone — NOT optimize_sequence)
            ds_result = await self._call_tool("design_sequence", {
                "backbone_pdb": bb_pdb,
                "num_sequences": 4,
                "sampling_temp": 0.1,
                "validate": True,
            }, output_dir)

            if "error" in ds_result:
                continue

            # Pick the best design by pLDDT
            ds_designs = ds_result.get("designs", [])
            if not ds_designs:
                continue

            best = max(ds_designs, key=lambda d: d.get("plddt", 0))
            seq = best.get("sequence", "")
            if not seq:
                continue

            design_entry: dict[str, Any] = {
                "name": f"design_{i+1:03d}",
                "sequence": seq,
                "length": len(seq),
                "backbone_pdb": bb_pdb,
                "pipeline_metrics": {
                    "mpnn_score": best.get("mpnn_score"),
                },
                "quality": {
                    "pLDDT": best.get("plddt", 0),
                    "pTM": best.get("ptm", 0),
                },
            }

            designs.append(design_entry)

        return designs

    # ------------------------------------------------------------------
    # Pipeline 4: Complex Engineering
    # design_binder → predict_complex  (fallback: generate_backbone → optimize_sequence → predict_complex)
    # ------------------------------------------------------------------

    async def _pipeline_cpx(
        self, task: Task, output_dir: Path,
    ) -> list[dict[str, Any]]:
        pdb_path = self._get_pdb_path(task, output_dir)
        hotspots = self._get_hotspot_residues(task)
        binder_length = self._get_binder_length(task)
        target_seq = self._get_target_sequence(task)

        # If no hotspots, use suggest_hotspots
        if not hotspots and pdb_path:
            hs_result = await self._call_tool("suggest_hotspots", {
                "target": pdb_path,
                "chain_id": task.target.chain or "A",
                "criteria": "exposed",
            }, output_dir)
            if "error" not in hs_result:
                for hs in hs_result.get("hotspots", [])[:5]:
                    res_id = hs.get("residue_id") or hs.get("residue", "")
                    if res_id:
                        hotspots.append(str(res_id))

        if not hotspots:
            chain = task.target.chain or "A"
            hotspots = [f"{chain}{i}" for i in range(30, 40)]

        # Primary: design_binder (target-aware binder generation)
        binder_result = await self._call_tool("design_binder", {
            "target_pdb": pdb_path,
            "hotspot_residues": hotspots,
            "num_designs": NUM_DESIGNS,
            "binder_length": binder_length,
        }, output_dir)

        raw_designs = []
        if "error" not in binder_result:
            raw_designs = binder_result.get("designs", [])
            if not raw_designs and "sequence" in binder_result:
                raw_designs = [binder_result]

        # Fallback: generate_backbone → design_sequence (ProteinMPNN)
        if not raw_designs:
            bb_result = await self._call_tool("generate_backbone", {
                "length": binder_length,
                "num_designs": NUM_DESIGNS,
            }, output_dir)

            if "error" not in bb_result:
                backbones = self._extract_backbones(bb_result)
                for bb in backbones[:NUM_DESIGNS]:
                    bb_pdb = self._resolve_backbone_pdb(bb)
                    if not bb_pdb or not Path(bb_pdb).exists():
                        continue
                    ds_result = await self._call_tool("design_sequence", {
                        "backbone_pdb": bb_pdb,
                        "num_sequences": 2,
                        "sampling_temp": 0.1,
                        "validate": False,
                    }, output_dir)
                    if "error" not in ds_result:
                        for d in ds_result.get("designs", [])[:1]:
                            seq = d.get("sequence", "")
                            if seq:
                                raw_designs.append({"sequence": seq})

        # Validate each design with predict_complex
        designs = []
        for i, d in enumerate(raw_designs[:NUM_DESIGNS]):
            seq = d.get("sequence", "")
            if not seq:
                continue

            design_entry: dict[str, Any] = {
                "name": f"design_{i+1:03d}",
                "sequence": seq,
                "length": len(seq),
                "pipeline_metrics": {k: v for k, v in d.items() if k != "sequence"},
            }

            # Validate with predict_complex (designed chain + target)
            if target_seq:
                cx_result = await self._call_tool("predict_complex", {
                    "sequences": [seq, target_seq],
                    "chain_names": ["design", "target"],
                }, output_dir)

                if "error" not in cx_result:
                    design_entry["quality"] = {
                        "ipTM": cx_result.get("iptm", 0),
                        "pTM": cx_result.get("ptm", 0),
                        "pLDDT": cx_result.get("plddt", 0),
                    }
            else:
                val_result = await self._call_tool("validate_design", {
                    "sequence": seq,
                    "predictor": "esmfold",
                }, output_dir)
                if "error" not in val_result:
                    design_entry["quality"] = {
                        "pLDDT": val_result.get("plddt", 0),
                        "pTM": val_result.get("ptm", 0),
                    }

            designs.append(design_entry)

        return designs

    # ------------------------------------------------------------------
    # Pipeline 5: Conformational Design
    # predict_structure → optimize_sequence → energy_minimize → validate_design
    # ------------------------------------------------------------------

    async def _pipeline_cfd(
        self, task: Task, output_dir: Path,
    ) -> list[dict[str, Any]]:
        pdb_path = self._get_pdb_path(task, output_dir)
        target_seq = self._get_target_sequence(task)

        if not target_seq:
            return []

        # Step 1: predict_structure (get reference structure)
        pred_result = await self._call_tool("predict_structure", {
            "sequence": target_seq,
            "predictor": "esmfold",
        }, output_dir)

        predicted_pdb = ""
        if "error" not in pred_result:
            predicted_pdb = (
                pred_result.get("predicted_structure_pdb")
                or pred_result.get("pdb_path")
                or ""
            )

        # Prefer input PDB over ESMFold prediction
        mpnn_pdb = pdb_path if pdb_path and Path(pdb_path).exists() else (
            predicted_pdb if predicted_pdb and Path(predicted_pdb).exists() else ""
        )

        if not mpnn_pdb or not Path(mpnn_pdb).exists():
            return []

        # Step 2: ProteinMPNN redesign on backbone structure
        ds_result = await self._call_tool("design_sequence", {
            "backbone_pdb": mpnn_pdb,
            "num_sequences": NUM_DESIGNS,
            "sampling_temp": 0.2,
            "validate": True,
        }, output_dir)

        if "error" in ds_result:
            return []

        ds_designs = ds_result.get("designs", [])
        designs = []
        for i, d in enumerate(ds_designs[:NUM_DESIGNS]):
            seq = d.get("sequence", "")
            if not seq:
                continue

            design_entry: dict[str, Any] = {
                "name": f"design_{i+1:03d}",
                "sequence": seq,
                "length": len(seq),
                "quality": {
                    "pLDDT": d.get("plddt", 0),
                    "pTM": d.get("ptm", 0),
                },
            }

            # Step 3: validate_design
            val_result = await self._call_tool("validate_design", {
                "sequence": seq,
                "predictor": "esmfold",
            }, output_dir)

            if "error" not in val_result:
                design_entry["quality"] = {
                    "pLDDT": val_result.get("plddt", 0),
                    "pTM": val_result.get("ptm", 0),
                }

            designs.append(design_entry)
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
        fasta_lines = []
        for d in designs:
            name = d.get("name", "design")
            seq = d.get("sequence", "")
            if seq:
                fasta_lines.append(f">{name}")
                # Wrap at 80 chars
                for j in range(0, len(seq), 80):
                    fasta_lines.append(seq[j:j+80])
        fasta_path.write_text("\n".join(fasta_lines) + "\n" if fasta_lines else "")

        # Metrics JSON
        metrics_path = output_dir / "metrics.json"
        metrics: dict[str, Any] = {
            "task_id": task.task_id,
            "agent": "hardcoded-pipeline",
            "designs": [],
            "summary": {
                "total_designs": len(designs),
                "total_time_seconds": round(elapsed, 1),
                "tools_used": list(set(self._tools_used)),
                "pipeline": " → ".join(
                    t.tool for t in self._tool_call_log
                    if t.tool not in ("suggest_hotspots",)
                ),
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
            for d in designs if d.get("quality")
        ]
        if plddts:
            metrics["summary"]["avg_pLDDT"] = round(sum(plddts) / len(plddts), 1)

        iptms = [
            d.get("quality", {}).get("ipTM", 0)
            for d in designs if d.get("quality", {}).get("ipTM")
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
