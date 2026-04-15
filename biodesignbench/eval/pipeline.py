"""Main evaluation pipeline."""

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biodesignbench.tasks.schema import Task, TaskTier, CodingTask, DesignTask
from biodesignbench.tasks.loader import load_task, load_all_tasks, get_task_by_id
from biodesignbench.tasks.prompt_generator import load_prompt
from biodesignbench.tasks.sanitizer import sanitize_task
from biodesignbench.eval.contamination import detect_contamination
from biodesignbench.eval.results import EvaluationResult, BenchmarkResults
from biodesignbench.eval.input_manager import InputManager


class EvaluationPipeline:
    """
    Main evaluation pipeline for running benchmark.

    Output layout:
        results/
          runs/
            run_20250208_143022/
              manifest.json
              summary.json
              agents/
                claude-code/
                  format_001/
                    output/          # Agent-generated files only
                    result.json      # Per-task EvaluationResult
          latest -> runs/run_20250208_143022

    Usage:
        pipeline = EvaluationPipeline(config)
        results = await pipeline.run(agents=["claude", "gpt4"], tier=TaskTier.TIER1)
    """

    def __init__(
        self,
        output_dir: str | Path = "results",
        timeout_minutes: int = 60,
        sandbox: bool = True,
        tool_provider: Any = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_minutes = timeout_minutes
        self.sandbox = sandbox
        self.tool_provider = tool_provider
        self._agents: dict[str, Any] = {}
        self._input_manager = InputManager()
        self._run_id: str | None = None
        self._run_dir: Path | None = None
        self._perturbation_level: str | None = None

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent for evaluation."""
        self._agents[agent_id] = agent

    def _create_run_dir(self, resume_run_id: str | None = None) -> None:
        """Create a timestamped run directory and update the latest symlink.

        If *resume_run_id* is given (e.g. ``"run_20250208_143022_000000"``
        or ``"latest"``), reuse that directory instead of creating a new one.
        """
        if resume_run_id:
            if resume_run_id == "latest":
                latest = self.output_dir / "latest"
                if latest.is_symlink() or latest.exists():
                    resolved = latest.resolve()
                    self._run_dir = resolved
                    self._run_id = resolved.name
                else:
                    raise FileNotFoundError(
                        "No 'latest' run found to resume. Run a benchmark first."
                    )
            else:
                self._run_dir = self.output_dir / "runs" / resume_run_id
                self._run_id = resume_run_id
                if not self._run_dir.exists():
                    raise FileNotFoundError(f"Run directory not found: {self._run_dir}")
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        if self._perturbation_level and self._perturbation_level != "none":
            self._run_id = f"run_{timestamp}_perturbed_{self._perturbation_level}"
        else:
            self._run_id = f"run_{timestamp}"
        self._run_dir = self.output_dir / "runs" / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Update 'latest' symlink
        latest = self.output_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(self._run_dir.resolve(), latest)

    def _write_manifest(
        self, agent_ids: list[str], task_ids: list[str]
    ) -> None:
        """Write manifest.json with run metadata."""
        manifest = {
            "run_id": self._run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "config": {
                "sandbox": self.sandbox,
                "timeout_minutes": self.timeout_minutes,
            },
            "agents": agent_ids,
            "tasks": task_ids,
        }

        manifest_path = self._run_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_task_result(self, result: EvaluationResult) -> None:
        """Save per-task result.json next to the output/ directory."""
        task_dir = self._run_dir / "agents" / result.agent_id / result.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        result_path = task_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    def _load_cached_result(
        self, task_id: str, agent_id: str
    ) -> EvaluationResult | None:
        """Load an existing *successful* result.json if present (for --resume).

        Only successful results are reused. Failed / errored tasks are
        retried so that transient failures (timeouts, API errors) get
        another chance.
        """
        if self._run_dir is None:
            return None
        result_path = (
            self._run_dir / "agents" / agent_id / task_id / "result.json"
        )
        if not result_path.exists():
            return None
        try:
            with open(result_path) as f:
                data = json.load(f)
            result = EvaluationResult.model_validate(data)
            # Only skip if the previous run succeeded
            if result.success:
                return result
            return None
        except Exception:
            return None

    async def evaluate_task(
        self,
        task: Task,
        agent_id: str,
    ) -> EvaluationResult:
        """Evaluate a single task with a single agent."""
        result = EvaluationResult(
            task_id=task.task_id,
            agent_id=agent_id,
            timestamp=datetime.now(UTC),
        )

        if agent_id not in self._agents:
            result.error_message = f"Agent not registered: {agent_id}"
            return result

        agent = self._agents[agent_id]

        # Precomputed baselines bypass solve() + evaluation
        if hasattr(agent, "get_precomputed_result"):
            precomputed = agent.get_precomputed_result(task.task_id)
            if precomputed is not None:
                result = EvaluationResult.model_validate(precomputed)
                result.timestamp = datetime.now(UTC)
                if self._run_dir is not None:
                    self._save_task_result(result)
                return result

        start_time = time.time()

        # Verify standardized prompt file exists for reproducibility
        try:
            tier_name = task.tier.value  # "tier1" or "tier2"
            prompt_content = load_prompt(task.task_id, tier=tier_name)
            result.prompt_file = f"data/{tier_name}/prompts/{task.task_id}.md"
        except FileNotFoundError:
            pass

        # Create per-agent, per-task output directory under run dir
        if self._run_dir is not None:
            task_output_dir = (
                self._run_dir / "agents" / agent_id / task.task_id / "output"
            )
        else:
            # Fallback for evaluate_task() called without run()
            task_output_dir = self.output_dir / agent_id / task.task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Provision inputs centrally (skip in Docker mode - container mounts data/)
        if not self.sandbox:
            self._input_manager.provision_inputs(task, task_output_dir)

        try:
            timeout = max(
                task.constraints.time_limit_minutes, self.timeout_minutes
            ) * 60

            agent_output = await asyncio.wait_for(
                self._run_agent(agent, task, task_output_dir),
                timeout=timeout,
            )

            result.execution_time_seconds = time.time() - start_time
            result.raw_output = agent_output

            # Post-hoc contamination detection (uses original unsanitized task)
            contamination = detect_contamination(
                task_id=task.task_id,
                agent_id=agent_id,
                reasoning_trace=agent_output.get("reasoning_trace", ""),
                code=agent_output.get("code", ""),
                task_source=task.metadata.source,
                task_doi=task.metadata.doi,
            )
            result.contamination_flags = contamination.flags
            result.contamination_score = contamination.contamination_score
            result.contamination_evidence = contamination.evidence

            if isinstance(task, CodingTask):
                await self._evaluate_coding_task(task, agent_output, result)
            elif isinstance(task, DesignTask):
                await self._evaluate_design_task(task, agent_output, result)

            # Zero the score if contamination is detected (threshold: 0.5)
            if result.contamination_score >= 0.5:
                result.partial_score = 0.0
                result.error_message = (
                    f"Score zeroed due to contamination "
                    f"(score={contamination.contamination_score:.2f}, "
                    f"flags={contamination.flags})"
                )

            result.success = True

        except asyncio.TimeoutError:
            result.error_message = (
                f"Task timed out after {task.constraints.time_limit_minutes} minutes"
            )
            result.execution_time_seconds = time.time() - start_time

        except Exception as e:
            error_str = str(e)
            # Classify provider-level content filter rejections so they
            # are distinguishable from agent failures in scoring/reporting.
            if self._is_content_filter_error(error_str):
                result.error_message = f"[content_filter] {error_str}"
                result.failure_modes.append("content_filter")
            else:
                result.error_message = error_str
            result.execution_time_seconds = time.time() - start_time

        # Tag with perturbation level if applicable
        if self._perturbation_level and self._perturbation_level != "none":
            result.perturbation_level = self._perturbation_level

        # Save per-task result if in a run context
        if self._run_dir is not None:
            self._save_task_result(result)

        return result

    @staticmethod
    def _is_content_filter_error(error_str: str) -> bool:
        """Return True if *error_str* looks like a provider content filter rejection.

        OpenAI returns HTTP 400 with ``invalid_prompt`` or messages about
        "limited access" when their biosafety classifier rejects a prompt.
        """
        lower = error_str.lower()
        return (
            "invalid_prompt" in lower
            or "we've limited access" in lower
            or ("content_policy" in lower and "400" in error_str)
        )

    async def _run_agent(
        self, agent: Any, task: Task, output_dir: Path
    ) -> dict[str, Any]:
        """Run agent on task and collect output.

        The task is sanitized before being passed to the agent to prevent
        data contamination (strips source DOIs, expected tools, ground
        truth, test file paths, and tags).
        """
        safe_task = sanitize_task(task)
        if hasattr(agent, "solve"):
            if asyncio.iscoroutinefunction(agent.solve):
                output = await agent.solve(safe_task, output_dir=output_dir)
            else:
                output = agent.solve(safe_task, output_dir=output_dir)

            # Convert AgentOutput to dict if needed
            if hasattr(output, "model_dump"):
                return output.model_dump()
            elif isinstance(output, dict):
                return output
            else:
                return {"code": str(output)}
        else:
            raise ValueError("Agent does not implement solve() method")

    async def _evaluate_coding_task(
        self,
        task: CodingTask,
        agent_output: dict[str, Any],
        result: EvaluationResult,
    ) -> None:
        """Evaluate a coding task output."""
        code = agent_output.get("code", "")
        artifacts_dir = agent_output.get("output_dir")
        artifacts = agent_output.get("artifacts", [])

        # Agent must have produced code OR output artifacts
        if not code and not artifacts and not artifacts_dir:
            result.valid_execution = False
            return

        result.valid_execution = True

        if task.evaluation.test_file and artifacts_dir:
            from biodesignbench.eval.tier1.runner import Tier1TestRunner

            runner = Tier1TestRunner(project_root=self.output_dir.parent)
            test_timeout = task.constraints.time_limit_minutes * 60
            test_results = runner.run_tests(
                task_id=task.task_id,
                output_dir=Path(artifacts_dir),
                timeout_seconds=min(test_timeout, 300),
            )
            result.test_results = test_results["test_results"]
            result.partial_score = test_results["partial_score"]

            if test_results["error_message"]:
                result.error_message = test_results["error_message"]
        else:
            expected_artifacts = set(task.expected_output.artifacts)
            generated_artifacts = set(agent_output.get("artifacts", []))
            if expected_artifacts:
                # Compare by filename, not full path
                generated_names = {Path(a).name for a in generated_artifacts}
                expected_names = {Path(a).name for a in expected_artifacts}
                result.partial_score = (
                    len(expected_names & generated_names) / len(expected_names) * 100
                )

        result.tools_used = agent_output.get("tools_used", [])
        result.api_calls = agent_output.get("api_calls", 0)
        result.iterations = agent_output.get("iterations", 1)

    async def _evaluate_design_task(
        self,
        task: DesignTask,
        agent_output: dict[str, Any],
        result: EvaluationResult,
    ) -> None:
        """Evaluate a design task output using Tier2Evaluator."""
        from biodesignbench.eval.tier2.runner import Tier2Evaluator

        output_dir = agent_output.get("output_dir")
        if not output_dir:
            # No output directory provided
            result.valid_execution = False
            return

        # Convert ToolCallEntry objects to dicts for the evaluator
        raw_log = agent_output.get("tool_call_log", [])
        tool_call_log = [
            e if isinstance(e, dict) else e.model_dump() if hasattr(e, "model_dump") else e
            for e in raw_log
        ]

        evaluator = Tier2Evaluator()
        eval_result = evaluator.evaluate(
            task_id=task.task_id,
            output_dir=Path(output_dir),
            tools_used=agent_output.get("tools_used", []),
            tools_expected=task.metadata.tools_expected,
            tool_call_log=tool_call_log,
            iterations=agent_output.get("iterations", 1),
        )

        result.valid_execution = True
        result.partial_score = eval_result["total_score"]
        result.approach_metrics = eval_result["metrics"].get("approach", {})
        result.orchestration_metrics = eval_result["metrics"].get("orchestration", {})
        result.quality_metrics = eval_result["metrics"].get("quality", {})
        result.novelty_metrics = eval_result["metrics"].get("novelty", {})
        result.diversity_metrics = eval_result["metrics"].get("diversity", {})
        result.feasibility_metrics = eval_result["metrics"].get("feasibility", {})
        result.tools_used = agent_output.get("tools_used", [])
        result.api_calls = agent_output.get("api_calls", 0)
        result.iterations = agent_output.get("iterations", 1)

    async def run(
        self,
        agent_ids: list[str] | None = None,
        tier: TaskTier | None = None,
        task_ids: list[str] | None = None,
        resume: str | None = None,
        perturbation: str | None = None,
        sequential: bool = False,
    ) -> BenchmarkResults:
        """Run full benchmark evaluation.

        Args:
            resume: Run ID to resume (e.g. ``"run_20250208_143022_000000"``
                or ``"latest"``). Tasks with existing result.json are skipped.
            perturbation: Perturbation level ("none", "mild", "moderate",
                "severe"). When set, results are tagged with the level and
                saved to a separate run directory.
            sequential: If True, run agents one at a time instead of
                concurrently. Useful when sharing a single GPU.
        """
        config = {
            "tier": tier.value if tier else "all",
            "sandbox": self.sandbox,
            "timeout_minutes": self.timeout_minutes,
        }
        if perturbation:
            config["perturbation"] = perturbation

        self._perturbation_level = perturbation

        results = BenchmarkResults(config=config)

        if agent_ids is None:
            agent_ids = list(self._agents.keys())

        if task_ids:
            tasks = []
            for tid in task_ids:
                task = get_task_by_id(tid)
                if task is None:
                    print(f"Warning: Task not found: {tid}")
                else:
                    tasks.append(task)
        else:
            tasks = load_all_tasks(tier=tier)

        # Create or reuse run directory
        self._create_run_dir(resume_run_id=resume)
        if not resume:
            self._write_manifest(
                agent_ids=agent_ids,
                task_ids=[t.task_id for t in tasks],
            )

        num_agents = len(agent_ids)
        num_tasks = len(tasks)
        print(f"\n{'='*60}")
        print(f"BioDesignBench Run: {self._run_id}" + (" (resumed)" if resume else ""))
        print(f"Agents: {num_agents} | Tasks: {num_tasks} | Total evaluations: {num_agents * num_tasks}")
        print(f"{'='*60}\n")

        # Run agents (parallel by default, sequential if requested)
        all_results = await self._run_agents_parallel(agent_ids, tasks, sequential=sequential)
        for eval_result in all_results:
            results.add_result(eval_result)

        results.compute_summaries()
        self._save_results(results)

        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for agent_id, summary in results.agent_summaries.items():
            parts = [
                f"  {agent_id}: {summary.successful_tasks}/{summary.total_tasks} passed "
                f"({summary.success_rate:.0%})"
            ]
            if summary.tier1_tasks:
                parts.append(f"tier1 avg: {summary.tier1_avg_partial_score:.1f}")
            if summary.tier2_tasks:
                parts.append(f"tier2 avg: {summary.tier2_avg_partial_score:.1f}")
            print(", ".join(parts))
        print()

        return results

    async def _run_agents_parallel(
        self,
        agent_ids: list[str],
        tasks: list[Task],
        sequential: bool = False,
    ) -> list[EvaluationResult]:
        """Run all agents, each processing tasks sequentially.

        By default agents run concurrently (parallel coroutines). When
        *sequential* is True, agents run one at a time — useful when
        sharing a single GPU to avoid memory conflicts.
        """
        num_agents = len(agent_ids)
        num_tasks = len(tasks)
        # Pad agent labels to align output
        max_label = max(len(aid) for aid in agent_ids) if agent_ids else 0

        async def run_single_agent(agent_id: str) -> list[EvaluationResult]:
            label = agent_id.ljust(max_label)
            agent_results = []
            skipped = 0
            for task_idx, task in enumerate(tasks, 1):
                tier_label = task.tier.value

                # Skip tasks that already have a result.json (resume mode)
                cached = self._load_cached_result(task.task_id, agent_id)
                if cached is not None:
                    agent_results.append(cached)
                    skipped += 1
                    print(
                        f"[{label} {task_idx:>{len(str(num_tasks))}}/{num_tasks}] "
                        f"{task.task_id} ({tier_label}) ... SKIP (cached)"
                    )
                    continue

                print(
                    f"[{label} {task_idx:>{len(str(num_tasks))}}/{num_tasks}] "
                    f"{task.task_id} ({tier_label}) ...",
                    end=" ",
                    flush=True,
                )
                eval_result = await self.evaluate_task(task, agent_id)
                agent_results.append(eval_result)

                if eval_result.success:
                    score = eval_result.get_overall_score()
                    print(f"OK (score: {score:.1f}, {eval_result.execution_time_seconds:.1f}s)")
                elif eval_result.error_message:
                    print(f"FAIL ({eval_result.error_message[:60]})")
                else:
                    print("FAIL")

            done_msg = f"[{label}] done ({len(agent_results)} tasks"
            if skipped:
                done_msg += f", {skipped} cached"
            done_msg += ")"
            print(done_msg)
            return agent_results

        # Launch agents concurrently or sequentially
        if sequential:
            agent_result_lists = []
            for aid in agent_ids:
                agent_result_lists.append(await run_single_agent(aid))
        else:
            agent_result_lists = await asyncio.gather(
                *(run_single_agent(aid) for aid in agent_ids)
            )

        # Flatten results
        all_results = []
        for result_list in agent_result_lists:
            all_results.extend(result_list)
        return all_results

    def _save_results(self, results: BenchmarkResults) -> None:
        """Save results to run directory as summary.json."""
        if self._run_dir is not None:
            output_file = self._run_dir / "summary.json"
        else:
            # Fallback for direct _save_results() calls without run()
            timestamp = results.run_timestamp.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"benchmark_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results.model_dump(mode="json"), f, indent=2, default=str)

        print(f"Results saved to {output_file}")
