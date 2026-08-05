# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment tracking CLI commands for Victor.

Provides CLI commands for managing experiments, runs, and artifacts:
- victor experiment list: List all experiments
- victor experiment show <id>: Show experiment details
- victor experiment create: Create new experiment
- victor experiment delete <id>: Delete experiment
- victor run list <experiment_id>: List runs in experiment
- victor run show <id>: Show run details

Usage:
    victor experiment list --status completed --tags optimization
    victor experiment show exp-123
    victor run list exp-123
    victor run show run-456
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

from victor.experiments import (
    ExperimentTracker,
    ExperimentQuery,
    ExperimentStatus,
    RunStatus,
    ArtifactType,
    MetricsAggregator,
    get_default_storage,
)
from victor.ui.commands.utils import setup_logging

logger = logging.getLogger(__name__)

experiment_app = typer.Typer(
    name="experiment",
    help="Manage experiments for tracking workflow runs.",
)
console = Console()


def _format_timestamp(ts: Optional[str]) -> str:
    """Format ISO timestamp for display."""
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds."""
    if not seconds:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


@experiment_app.command("list")
def list_experiments(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Filter by tags (comma-separated)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number to show"),
    log_level: Optional[str] = typer.Option(None, "--log-level", hidden=True),
):
    """List all experiments with optional filtering."""
    setup_logging(command="experiments", cli_log_level=log_level)

    storage = get_default_storage()
    tracker = ExperimentTracker(storage=storage)

    # Build query
    query = ExperimentQuery(limit=limit)

    if status:
        try:
            query.status = ExperimentStatus(status)
        except ValueError:
            console.print(f"[bold red]Error:[/] Invalid status: {status}")
            raise typer.Exit(1)

    if tags:
        query.tags_any = [t.strip() for t in tags.split(",")]

    # List experiments
    experiments = tracker.list_experiments(query)

    if not experiments:
        console.print("[yellow]No experiments found.[/]")
        return

    # Create table
    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Tags")
    table.add_column("Created")
    table.add_column("Runs", style="dim")

    for exp in experiments:
        # Count runs
        runs = storage.list_runs(exp.experiment_id)
        run_count = len(runs)

        # Format tags
        tags_str = ", ".join(exp.tags[:3])
        if len(exp.tags) > 3:
            tags_str += f" +{len(exp.tags) - 3}"

        table.add_row(
            exp.experiment_id[:8],
            exp.name,
            f"[{exp.status}]{exp.status}[/]",
            tags_str,
            _format_timestamp(exp.created_at.isoformat()),
            str(run_count),
        )

    console.print(table)


@experiment_app.command("show")
def show_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    log_level: Optional[str] = typer.Option(None, "--log-level", hidden=True),
):
    """Show detailed information about an experiment."""
    setup_logging(command="experiments", cli_log_level=log_level)

    storage = get_default_storage()
    tracker = ExperimentTracker(storage=storage)

    experiment = tracker.get_experiment(experiment_id)
    if not experiment:
        console.print(f"[bold red]Error:[/] Experiment not found: {experiment_id}")
        raise typer.Exit(1)

    # Get runs
    runs = storage.list_runs(experiment_id)

    # Display experiment info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value")

    info_table.add_row("ID", experiment.experiment_id)
    info_table.add_row("Name", experiment.name)
    info_table.add_row("Description", experiment.description or "-")
    info_table.add_row("Hypothesis", experiment.hypothesis or "-")
    info_table.add_row("Status", f"[{experiment.status}]{experiment.status}[/]")
    info_table.add_row("Tags", ", ".join(experiment.tags) or "-")
    info_table.add_row("Created", _format_timestamp(experiment.created_at.isoformat()))
    info_table.add_row(
        "Started",
        (_format_timestamp(experiment.started_at.isoformat()) if experiment.started_at else "-"),
    )
    info_table.add_row(
        "Completed",
        (
            _format_timestamp(experiment.completed_at.isoformat())
            if experiment.completed_at
            else "-"
        ),
    )
    info_table.add_row("Git Commit", experiment.git_commit_sha or "-")
    info_table.add_row("Git Branch", experiment.git_branch or "-")
    info_table.add_row("Git Dirty", "Yes" if experiment.git_dirty else "No")

    console.print(Panel(info_table, title=f"Experiment: {experiment.name}"))

    # Display parameters if any
    if experiment.parameters:
        console.print("\n[bold]Parameters:[/]")
        console.print(JSON(json.dumps(experiment.parameters, indent=2, default=str)))

    # Display runs
    if runs:
        console.print(f"\n[bold]Runs ({len(runs)}):[/]")

        runs_table = Table()
        runs_table.add_column("ID", style="cyan")
        runs_table.add_column("Name", style="green")
        runs_table.add_column("Status", style="bold")
        runs_table.add_column("Duration")
        runs_table.add_column("Metrics Summary")

        for run in runs[:10]:  # Show first 10 runs
            # Format metrics summary
            metrics_str = ", ".join(
                f"{k}={v:.3f}" for k, v in list(run.metrics_summary.items())[:3]
            )
            if len(run.metrics_summary) > 3:
                metrics_str += f" +{len(run.metrics_summary) - 3}"

            runs_table.add_row(
                run.run_id[:8],
                run.name,
                f"[{run.status}]{run.status}[/]",
                _format_duration(run.duration_seconds),
                metrics_str,
            )

        console.print(runs_table)

        if len(runs) > 10:
            console.print(f"\n[dim]... and {len(runs) - 10} more runs[/]")


@experiment_app.command("create")
def create_experiment(
    name: str = typer.Option(..., "--name", "-n", help="Experiment name"),
    description: str = typer.Option("", "--description", "-d", help="Experiment description"),
    hypothesis: str = typer.Option("", "--hypothesis", help="Hypothesis being tested"),
    tags: str = typer.Option("", "--tags", "-t", help="Comma-separated tags"),
    params: str = typer.Option("", "--params", "-p", help="JSON parameters"),
    log_level: Optional[str] = typer.Option(None, "--log-level", hidden=True),
):
    """Create a new experiment."""
    setup_logging(command="experiments", cli_log_level=log_level)

    tracker = ExperimentTracker()

    # Parse parameters
    parameters = {}
    if params:
        try:
            parameters = json.loads(params)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in params: {e}")
            raise typer.Exit(1)

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    # Create experiment
    experiment = tracker.create_experiment(
        name=name,
        description=description,
        hypothesis=hypothesis,
        tags=tag_list,
        parameters=parameters,
    )

    console.print(f"[green]Created experiment:[/] {experiment.experiment_id}")
    console.print(f"  Name: {experiment.name}")
    console.print(f"  Tags: {', '.join(experiment.tags) or '-'}")


@experiment_app.command("delete")
def delete_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    log_level: Optional[str] = typer.Option(None, "--log-level", hidden=True),
):
    """Delete an experiment and all its runs."""
    setup_logging(command="experiments", cli_log_level=log_level)

    storage = get_default_storage()
    tracker = ExperimentTracker(storage=storage)

    # Check if experiment exists
    experiment = tracker.get_experiment(experiment_id)
    if not experiment:
        console.print(f"[bold red]Error:[/] Experiment not found: {experiment_id}")
        raise typer.Exit(1)

    # Confirm deletion
    if not force:
        console.print(
            f"[bold yellow]Warning:[/] This will delete experiment '{experiment.name}' and all its runs."
        )
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit(0)

    # Delete
    if tracker.delete_experiment(experiment_id):
        console.print(f"[green]Deleted experiment:[/] {experiment_id}")
    else:
        console.print("[bold red]Error:[/] Failed to delete experiment")
        raise typer.Exit(1)


# Run commands

run_app = typer.Typer(
    name="run",
    help="Manage experiment runs.",
)


@run_app.command("list")
def list_runs(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number to show"),
    log_level: Optional[str] = typer.Option(None, "--log-level", hidden=True),
):
    """List runs for an experiment."""
    setup_logging(command="experiments", cli_log_level=log_level)

    storage = get_default_storage()
    tracker = ExperimentTracker(storage=storage)

    # Check if experiment exists
    experiment = tracker.get_experiment(experiment_id)
    if not experiment:
        console.print(f"[bold red]Error:[/] Experiment not found: {experiment_id}")
        raise typer.Exit(1)

    # Get runs
    runs = tracker.list_runs(experiment_id)

    # Filter by status
    if status:
        try:
            filter_status = RunStatus(status)
            runs = [r for r in runs if r.status == filter_status]
        except ValueError:
            console.print(f"[bold red]Error:[/] Invalid status: {status}")
            raise typer.Exit(1)

    # Limit results
    runs = runs[:limit]

    if not runs:
        console.print("[yellow]No runs found.[/]")
        return

    # Create table
    table = Table(title=f"Runs for '{experiment.name}'")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Started")
    table.add_column("Duration")
    table.add_column("Metrics Summary")

    for run in runs:
        # Format metrics summary
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in list(run.metrics_summary.items())[:3])
        if len(run.metrics_summary) > 3:
            metrics_str += f" +{len(run.metrics_summary) - 3}"

        table.add_row(
            run.run_id[:8],
            run.name,
            f"[{run.status}]{run.status}[/]",
            _format_timestamp(run.started_at.isoformat()),
            _format_duration(run.duration_seconds),
            metrics_str,
        )

    console.print(table)


@run_app.command("show")
def show_run(
    run_id: str = typer.Argument(..., help="Run ID"),
    log_level: Optional[str] = typer.Option(None, "--log-level", hidden=True),
):
    """Show detailed information about a run."""
    setup_logging(command="experiments", cli_log_level=log_level)

    storage = get_default_storage()
    tracker = ExperimentTracker(storage=storage)

    run = tracker.get_run(run_id)
    if not run:
        console.print(f"[bold red]Error:[/] Run not found: {run_id}")
        raise typer.Exit(1)

    # Get experiment info
    experiment = tracker.get_experiment(run.experiment_id)

    # Display run info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value")

    info_table.add_row("Run ID", run.run_id)
    info_table.add_row("Name", run.name)
    info_table.add_row("Experiment", experiment.name if experiment else run.experiment_id)
    info_table.add_row("Status", f"[{run.status}]{run.status}[/]")
    info_table.add_row("Started", _format_timestamp(run.started_at.isoformat()))
    info_table.add_row(
        "Completed",
        _format_timestamp(run.completed_at.isoformat()) if run.completed_at else "-",
    )
    info_table.add_row("Duration", _format_duration(run.duration_seconds))

    if run.error_message:
        info_table.add_row("Error", f"[red]{run.error_message}[/]")

    console.print(Panel(info_table, title=f"Run: {run.name}"))

    # Display parameters
    if run.parameters:
        console.print("\n[bold]Parameters:[/]")
        console.print(JSON(json.dumps(run.parameters, indent=2, default=str)))

    # Display metrics summary
    if run.metrics_summary:
        console.print("\n[bold]Metrics Summary:[/]")

        metrics_table = Table(show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        for key, value in sorted(run.metrics_summary.items()):
            metrics_table.add_row(key, f"{value:.6f}")

        console.print(metrics_table)

    # Display metric history
    metrics = storage.get_metrics(run_id)
    if metrics:
        # Group by metric name
        from collections import defaultdict

        metrics_by_key = defaultdict(list)
        for m in metrics:
            metrics_by_key[m.key].append(m)

        console.print("\n[bold]Metric History:[/]")
        for key, history in sorted(metrics_by_key.items()):
            values = [m.value for m in history]
            console.print(
                f"  {key}: {len(values)} points, range=[{min(values):.4f}, {max(values):.4f}]"
            )

    # Display artifacts
    artifacts = storage.get_artifacts(run_id)
    if artifacts:
        console.print(f"\n[bold]Artifacts ({len(artifacts)}):[/]")

        artifacts_table = Table()
        artifacts_table.add_column("Type")
        artifacts_table.add_column("Filename")
        artifacts_table.add_column("Size")
        artifacts_table.add_column("Created")

        for artifact in artifacts:
            size_mb = artifact.file_size_bytes / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{artifact.file_size_bytes} bytes"

            artifacts_table.add_row(
                artifact.artifact_type.value,
                artifact.filename,
                size_str,
                _format_timestamp(artifact.created_at.isoformat()),
            )

        console.print(artifacts_table)

    # Display environment info
    if run.python_version or run.os_info:
        console.print("\n[bold]Environment:[/]")
        env_table = Table(show_header=False)
        env_table.add_column("Field", style="cyan")
        env_table.add_column("Value")

        if run.python_version:
            env_table.add_row("Python", run.python_version)
        if run.os_info:
            env_table.add_row("OS", run.os_info)
        if run.victor_version:
            env_table.add_row("Victor", run.victor_version)
        if run.provider:
            env_table.add_row("Provider", run.provider)
        if run.model:
            env_table.add_row("Model", run.model)

        console.print(env_table)


# Register apps
experiment_app.add_typer(run_app, name="run")
