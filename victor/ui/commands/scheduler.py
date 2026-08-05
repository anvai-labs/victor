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

"""Scheduler CLI commands for Victor.

Provides commands for managing the workflow scheduler service:
- start/stop the scheduler daemon
- list scheduled workflows
- add/remove schedules
- view execution history

This scheduler is a pure Python implementation that:
- Does NOT rely on system cron (crond)
- Parses cron expressions internally
- Runs as an asyncio background service
- Can be deployed as a systemd service

Commands:
    start      - Start the scheduler service (foreground or daemon)
    stop       - Stop a running scheduler daemon
    status     - Show scheduler status and upcoming executions
    list       - List all scheduled workflows
    add        - Schedule a workflow
    remove     - Remove a scheduled workflow
    history    - Show execution history
    install    - Generate systemd service file

Example:
    # Run scheduler in foreground
    victor scheduler start

    # Run as daemon (background)
    victor scheduler start --daemon

    # List schedules
    victor scheduler list

    # Add a schedule
    victor scheduler add my_workflow --cron "0 9 * * *" --yaml path/to/workflow.yaml

    # View history
    victor scheduler history --limit 20

    # Generate systemd service file
    victor scheduler install --output /etc/systemd/system/victor-scheduler.service
"""

import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from victor.config.settings import get_project_paths

scheduler_app = typer.Typer(
    name="scheduler",
    help="Manage the workflow scheduler service.",
)

console = Console()


def _get_default_pid_file() -> Path:
    """Resolve the default scheduler PID file through centralized Victor paths."""
    return get_project_paths().global_victor_dir / "scheduler.pid"


def _resolve_pid_file(pid_file: Optional[Path]) -> Path:
    """Resolve an explicit or default scheduler PID file path."""
    return pid_file or _get_default_pid_file()


def _get_scheduler():
    """Get the global scheduler instance."""
    from victor.workflows.scheduler import get_scheduler

    return get_scheduler()


@scheduler_app.command("start")
def start(
    daemon: bool = typer.Option(
        False,
        "--daemon",
        "-d",
        help="Run as background daemon",
    ),
    pid_file: Optional[Path] = typer.Option(
        None,
        "--pid-file",
        help="PID file for daemon mode",
    ),
    check_interval: float = typer.Option(
        60.0,
        "--interval",
        help="Check interval in seconds",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML config file with schedules",
    ),
) -> None:
    """Start the workflow scheduler service.

    The scheduler runs in the foreground by default. Use --daemon to run
    in the background.

    Example:
        victor scheduler start
        victor scheduler start --daemon
        victor scheduler start --config schedules.yaml
    """
    if daemon:
        _start_daemon(_resolve_pid_file(pid_file), check_interval, config_file)
    else:
        _start_foreground(check_interval, config_file)


def _start_foreground(
    check_interval: float,
    config_file: Optional[Path],
) -> None:
    """Start scheduler in foreground."""
    console.print("[bold green]Starting Victor Workflow Scheduler...[/]")

    scheduler = _get_scheduler()
    scheduler._check_interval = check_interval

    # Load schedules from config if provided
    if config_file:
        _load_schedules_from_config(config_file)

    # Show registered schedules
    schedules = scheduler.list_schedules()
    if schedules:
        console.print(f"[dim]Loaded {len(schedules)} scheduled workflow(s)[/]")
    else:
        console.print(
            "[yellow]No scheduled workflows registered. Use 'victor scheduler add' to add schedules.[/]"
        )

    console.print(f"[dim]Check interval: {check_interval}s[/]")
    console.print("[dim]Press Ctrl+C to stop[/]")
    console.print()

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        await scheduler.start()
        try:
            while scheduler.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await scheduler.stop()

    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down scheduler...[/]")
        loop.run_until_complete(scheduler.stop())
        console.print("[green]Scheduler stopped.[/]")


def _start_daemon(
    pid_file: Path,
    check_interval: float,
    config_file: Optional[Path],
) -> None:
    """Start scheduler as daemon."""
    import os

    # Ensure pid directory exists
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if already running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            console.print(f"[yellow]Scheduler already running (PID {pid})[/]")
            raise typer.Exit(1)
        except (ProcessLookupError, ValueError):
            # Process not running, remove stale pid file
            pid_file.unlink()

    # Fork to background
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process
            console.print(f"[green]Scheduler started (PID {pid})[/]")
            console.print(f"[dim]PID file: {pid_file}[/]")
            raise typer.Exit(0)
    except OSError as e:
        console.print(f"[red]Failed to fork: {e}[/]")
        raise typer.Exit(1)

    # Child process continues
    os.setsid()  # Create new session

    # Write PID file
    pid_file.write_text(str(os.getpid()))

    # Redirect stdout/stderr to log file
    # Note: File handle intentionally kept open for daemon process lifetime
    log_file = pid_file.parent / "scheduler.log"
    sys.stdout = open(log_file, "a", buffering=1)  # Line buffering
    sys.stderr = sys.stdout

    # Run scheduler
    _start_foreground(check_interval, config_file)


@scheduler_app.command("stop")
def stop(
    pid_file: Optional[Path] = typer.Option(
        None,
        "--pid-file",
        help="PID file for daemon",
    ),
) -> None:
    """Stop the scheduler daemon.

    Example:
        victor scheduler stop
    """
    pid_file = _resolve_pid_file(pid_file)

    if not pid_file.exists():
        console.print("[yellow]Scheduler is not running (no PID file)[/]")
        raise typer.Exit(0)

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink()
        console.print(f"[green]Scheduler stopped (PID {pid})[/]")
    except ProcessLookupError:
        pid_file.unlink()
        console.print("[yellow]Scheduler was not running (stale PID file removed)[/]")
    except ValueError:
        console.print("[red]Invalid PID file[/]")
        raise typer.Exit(1)


@scheduler_app.command("status")
def status(
    pid_file: Optional[Path] = typer.Option(
        None,
        "--pid-file",
        help="PID file for daemon",
    ),
) -> None:
    """Show scheduler status and upcoming executions.

    Example:
        victor scheduler status
    """
    pid_file = _resolve_pid_file(pid_file)

    # Check daemon status
    daemon_running = False
    daemon_pid = None

    if pid_file.exists():
        try:
            daemon_pid = int(pid_file.read_text().strip())
            os.kill(daemon_pid, 0)  # Check if process exists
            daemon_running = True
        except (ProcessLookupError, ValueError):
            pass

    # Status panel
    if daemon_running:
        console.print(
            Panel(
                f"[green]Running[/] (PID {daemon_pid})",
                title="Scheduler Status",
            )
        )
    else:
        console.print(
            Panel(
                "[yellow]Not running[/]",
                title="Scheduler Status",
            )
        )

    # Show upcoming executions
    scheduler = _get_scheduler()
    schedules = scheduler.list_schedules()

    if schedules:
        table = Table(title="Upcoming Executions")
        table.add_column("Workflow", style="cyan")
        table.add_column("Next Run", style="green")
        table.add_column("Cron", style="dim")
        table.add_column("Enabled")

        for s in sorted(
            schedules,
            key=lambda x: x.next_run or datetime.max.replace(tzinfo=timezone.utc),
        ):
            next_run = s.next_run.strftime("%Y-%m-%d %H:%M:%S") if s.next_run else "N/A"
            enabled = "[green]Yes[/]" if s.enabled else "[red]No[/]"
            table.add_row(
                s.workflow_name,
                next_run,
                s.schedule.expression,
                enabled,
            )

        console.print(table)
    else:
        console.print("[dim]No scheduled workflows[/]")


@scheduler_app.command("list")
def list_schedules() -> None:
    """List all scheduled workflows.

    Example:
        victor scheduler list
    """
    scheduler = _get_scheduler()
    schedules = scheduler.list_schedules()

    if not schedules:
        console.print("[dim]No scheduled workflows[/]")
        return

    table = Table(title="Scheduled Workflows")
    table.add_column("ID", style="dim")
    table.add_column("Workflow", style="cyan")
    table.add_column("Cron", style="green")
    table.add_column("Next Run")
    table.add_column("Last Run")
    table.add_column("Runs")
    table.add_column("Enabled")

    for s in schedules:
        next_run = s.next_run.strftime("%Y-%m-%d %H:%M") if s.next_run else "N/A"
        last_run = s.last_run.strftime("%Y-%m-%d %H:%M") if s.last_run else "Never"
        enabled = "[green]Yes[/]" if s.enabled else "[red]No[/]"

        table.add_row(
            s.schedule_id,
            s.workflow_name,
            s.schedule.expression,
            next_run,
            last_run,
            str(s.run_count),
            enabled,
        )

    console.print(table)


@scheduler_app.command("add")
def add(
    workflow_name: str = typer.Argument(
        ...,
        help="Name of the workflow to schedule",
    ),
    cron: str = typer.Option(
        ...,
        "--cron",
        "-c",
        help="Cron expression (e.g., '0 9 * * *' for daily at 9 AM)",
    ),
    yaml_path: Optional[Path] = typer.Option(
        None,
        "--yaml",
        "-y",
        help="Path to workflow YAML file",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        help="Initial state as JSON",
    ),
    disabled: bool = typer.Option(
        False,
        "--disabled",
        help="Register as disabled",
    ),
) -> None:
    """Schedule a workflow for recurring execution.

    Example:
        victor scheduler add daily_report --cron "0 9 * * *" --yaml workflows/report.yaml
        victor scheduler add cleanup --cron "@hourly" --context '{"mode": "fast"}'
    """
    from victor.workflows.scheduler import (
        CronSchedule,
        ScheduledWorkflow,
    )

    try:
        schedule = CronSchedule.from_cron(cron)
    except ValueError as e:
        console.print(f"[red]Invalid cron expression: {e}[/]")
        raise typer.Exit(1)

    # Parse context
    initial_state = {}
    if context:
        try:
            initial_state = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON context: {e}[/]")
            raise typer.Exit(1)

    # Validate YAML path if provided
    if yaml_path and not yaml_path.exists():
        console.print(f"[red]Workflow file not found: {yaml_path}[/]")
        raise typer.Exit(1)

    scheduled = ScheduledWorkflow(
        workflow_name=workflow_name,
        workflow_path=str(yaml_path) if yaml_path else None,
        schedule=schedule,
        initial_state=initial_state,
        enabled=not disabled,
    )

    scheduler = _get_scheduler()
    schedule_id = scheduler.register(scheduled)

    console.print(f"[green]Scheduled workflow:[/] {workflow_name}")
    console.print(f"[dim]Schedule ID:[/] {schedule_id}")
    console.print(f"[dim]Cron:[/] {cron}")
    console.print(f"[dim]Next run:[/] {scheduled.next_run}")


@scheduler_app.command("remove")
def remove(
    schedule_id: str = typer.Argument(
        ...,
        help="Schedule ID to remove",
    ),
) -> None:
    """Remove a scheduled workflow.

    Example:
        victor scheduler remove abc123def456
    """
    scheduler = _get_scheduler()

    if scheduler.unregister(schedule_id):
        console.print(f"[green]Removed schedule:[/] {schedule_id}")
    else:
        console.print(f"[red]Schedule not found:[/] {schedule_id}")
        raise typer.Exit(1)


@scheduler_app.command("history")
def history(
    schedule_id: Optional[str] = typer.Option(
        None,
        "--schedule",
        "-s",
        help="Filter by schedule ID",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum entries to show",
    ),
) -> None:
    """Show execution history.

    Example:
        victor scheduler history
        victor scheduler history --schedule abc123 --limit 10
    """
    scheduler = _get_scheduler()
    entries = scheduler.get_execution_history(schedule_id, limit)

    if not entries:
        console.print("[dim]No execution history[/]")
        return

    table = Table(title="Execution History")
    table.add_column("Execution ID", style="dim")
    table.add_column("Workflow", style="cyan")
    table.add_column("Start Time")
    table.add_column("Duration")
    table.add_column("Status")
    table.add_column("Error")

    for entry in reversed(entries):
        start = entry["start_time"].strftime("%Y-%m-%d %H:%M:%S")
        duration = (entry["end_time"] - entry["start_time"]).total_seconds()
        status = "[green]Success[/]" if entry["success"] else "[red]Failed[/]"
        error = entry.get("error", "")[:40] if entry.get("error") else ""

        table.add_row(
            entry["execution_id"],
            entry["workflow_name"],
            start,
            f"{duration:.1f}s",
            status,
            error,
        )

    console.print(table)


@scheduler_app.command("install")
def install(
    output: Path = typer.Option(
        Path("/etc/systemd/system/victor-scheduler.service"),
        "--output",
        "-o",
        help="Output path for service file",
    ),
    user: str = typer.Option(
        os.getenv("USER", "victor"),
        "--user",
        help="User to run service as",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path for schedules",
    ),
) -> None:
    """Generate systemd service file for the scheduler.

    After generating, install with:
        sudo systemctl daemon-reload
        sudo systemctl enable victor-scheduler
        sudo systemctl start victor-scheduler

    Example:
        victor scheduler install --output victor-scheduler.service --user deploy
    """
    # Find victor executable
    victor_path = sys.executable.replace("python", "victor")
    if not Path(victor_path).exists():
        victor_path = "victor"

    config_arg = f" --config {config}" if config else ""

    service_content = f"""[Unit]
Description=Victor Workflow Scheduler
After=network.target

[Service]
Type=simple
User={user}
ExecStart={victor_path} scheduler start{config_arg}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

    try:
        output.write_text(service_content)
        console.print(f"[green]Service file created:[/] {output}")
        console.print()
        console.print("[dim]To install and start the service:[/]")
        console.print("  sudo systemctl daemon-reload")
        console.print(f"  sudo systemctl enable {output.stem}")
        console.print(f"  sudo systemctl start {output.stem}")
    except PermissionError:
        console.print(f"[red]Permission denied:[/] {output}")
        console.print("[dim]Try running with sudo or use a different output path[/]")
        raise typer.Exit(1)


def _load_schedules_from_config(config_file: Path) -> int:
    """Load schedules from a YAML config file.

    Args:
        config_file: Path to config file

    Returns:
        Number of schedules loaded
    """
    import yaml
    from victor.workflows.scheduler import (
        CronSchedule,
        ScheduledWorkflow,
    )

    if not config_file.exists():
        console.print(f"[yellow]Config file not found: {config_file}[/]")
        return 0

    try:
        with open(config_file) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML config: {e}[/]")
        return 0

    schedules = data.get("schedules", [])
    scheduler = _get_scheduler()
    count = 0

    for sched in schedules:
        try:
            cron = CronSchedule.from_cron(sched["cron"])
            workflow = ScheduledWorkflow(
                workflow_name=sched["workflow"],
                workflow_path=sched.get("yaml_path"),
                schedule=cron,
                initial_state=sched.get("context", {}),
                enabled=sched.get("enabled", True),
            )
            scheduler.register(workflow)
            count += 1
        except Exception as e:
            console.print(f"[yellow]Failed to load schedule: {e}[/]")

    return count
