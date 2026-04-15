"""BioDesignBench CLI entry point."""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="biodesignbench")
def main():
    """BioDesignBench: Benchmark for Biomolecule Design AI Agents."""
    pass


@main.command()
@click.option("--agent", required=True, help="Agent ID(s), comma-separated. Use 'all' or 'list'.")
@click.option("--task", default=None, help="Task ID(s), comma-separated.")
@click.option("--tier", default=None, type=click.Choice(["tier1", "tier2"]), help="Filter by tier.")
@click.option("--output-dir", default="results", help="Output directory.")
@click.option("--timeout", default=60, type=int, help="Timeout per task in minutes.")
@click.option("--resume", default=None, help="Resume a previous run. Use run ID or 'latest'. Tasks with existing results are skipped.")
def run(agent, task, tier, output_dir, timeout, resume):
    """Run benchmark evaluation."""
    import asyncio
    import sys

    from biodesignbench.utils.config import load_env

    load_env()

    from biodesignbench.agents import get_agent, list_agents, BASELINE_REGISTRY
    from biodesignbench.eval.pipeline import EvaluationPipeline
    from biodesignbench.tasks.schema import TaskTier

    if agent == "list":
        click.echo("Runnable agents:")
        for agent_id in list_agents():
            click.echo(f"  - {agent_id}")
        click.echo("Baselines (pre-computed, not run):")
        for agent_id in sorted(BASELINE_REGISTRY.keys()):
            click.echo(f"  - {agent_id}")
        return

    agent_ids = [a.strip() for a in agent.split(",")] if agent != "all" else [
        aid for aid in list_agents() if aid != "dummy"
    ]

    task_ids = [t.strip() for t in task.split(",")] if task else None
    tier_enum = TaskTier(tier) if tier else None

    click.echo(f"BioDesignBench Runner")
    click.echo(f"  Agents: {', '.join(agent_ids)}")
    click.echo(f"  Tasks: {', '.join(task_ids) if task_ids else tier or 'all'}")
    click.echo(f"  Output: {output_dir}")
    click.echo()

    pipeline = EvaluationPipeline(output_dir=output_dir, timeout_minutes=timeout)

    for agent_id in agent_ids:
        try:
            ag = get_agent(agent_id)
            ag.setup()
            pipeline.register_agent(agent_id, ag)
            click.echo(f"  Registered: {agent_id} ({ag.get_info().model})")
        except Exception as e:
            click.echo(f"  SKIP: {agent_id} ({e})")

    click.echo()
    results = asyncio.run(
        pipeline.run(
            agent_ids=list(pipeline._agents.keys()),
            tier=tier_enum,
            task_ids=task_ids,
            resume=resume,
        )
    )

    leaderboard = results.to_leaderboard()
    if leaderboard:
        click.echo(f"\n{'Rank':<6}{'Agent':<20}{'Success Rate':<15}{'Tasks':<8}{'Avg Time':<12}")
        click.echo("-" * 60)
        for entry in leaderboard:
            click.echo(
                f"{entry['rank']:<6}{entry['agent_id']:<20}"
                f"{entry['success_rate']:.1%}{'':>6}{entry['total_tasks']:<8}"
                f"{entry['avg_time']:.1f}s"
            )


@main.command()
def list_tasks():
    """List all available tasks."""
    from biodesignbench.tasks.loader import load_all_tasks

    tasks = load_all_tasks()
    click.echo(f"Total tasks: {len(tasks)}")
    for task in tasks:
        click.echo(f"  [{task.tier.value}] {task.task_id}: {task.name}")


@main.command()
def list_agents():
    """List all available agents."""
    from biodesignbench.agents import list_agents as _list_agents, BASELINE_REGISTRY

    click.echo("Runnable agents:")
    for agent_id in _list_agents():
        click.echo(f"  - {agent_id}")
    click.echo("Baselines (pre-computed, not run):")
    for agent_id in sorted(BASELINE_REGISTRY.keys()):
        click.echo(f"  - {agent_id}")


if __name__ == "__main__":
    main()
