# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Victor Dashboard CLI command.

Provides the `victor dashboard` command for launching the
Textual-based observability dashboard.

Usage:
    victor dashboard                    # Launch dashboard
    victor dashboard --log-file events.jsonl  # Load historical events
    victor dashboard --no-live          # Disable live event streaming
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from victor.core.async_utils import run_sync

dashboard_app = typer.Typer(
    name="dashboard",
    help="Launch the observability dashboard for visualizing events and traces.",
)

console = Console()


@dashboard_app.callback(invoke_without_command=True)
def dashboard(
    ctx: typer.Context,
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        "-f",
        help="Path to a JSONL log file to load historical events.",
    ),
    live: bool = typer.Option(
        True,
        "--live/--no-live",
        help="Enable/disable live event streaming from ObservabilityBus.",
    ),
    wire_log: Optional[str] = typer.Option(
        None,
        "--wire-log",
        help=(
            "Path to a v1 wire-event log (JSONL, or a raw SSE capture of "
            "POST /chat/stream) to replay and tail in the Agent tab."
        ),
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Run in demo mode with simulated events.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set log level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    ),
) -> None:
    """Launch the Victor observability dashboard.

    The dashboard provides a TUI for visualizing:
    - Real-time events from the ObservabilityBus
    - Historical events from JSONL log files
    - Tool execution statistics and history
    - Vertical integration traces

    Examples:
        # Launch with live event streaming
        victor dashboard

        # Load historical events from a file
        victor dashboard --log-file ~/.victor/events.jsonl

        # Disable live streaming, only show historical data
        victor dashboard --log-file events.jsonl --no-live

        # Run demo mode with simulated events
        victor dashboard --demo

        # Replay/tail an agent wire-event stream in the Agent tab
        victor dashboard --wire-log chat.sse

        # Enable debug logging to troubleshoot event loading
        victor dashboard --log-level DEBUG
    """
    if ctx.invoked_subcommand is None:
        # Setup logging with dashboard-specific log level
        from victor.ui.commands.utils import setup_logging

        setup_logging(command="dashboard", cli_log_level=log_level)

        run_sync(
            run_dashboard(
                log_file=log_file,
                live=live,
                demo=demo,
                wire_log=wire_log,
            )
        )


async def run_dashboard(
    log_file: Optional[str] = None,
    live: bool = True,
    demo: bool = False,
    wire_log: Optional[str] = None,
) -> None:
    """Run the observability dashboard.

    Args:
        log_file: Optional path to JSONL log file
        live: Whether to enable live event streaming
        demo: Whether to run in demo mode
        wire_log: Optional v1 wire-event log to replay/tail in the Agent tab
    """
    try:
        from victor.observability.dashboard import ObservabilityDashboard
    except ImportError as e:
        console.print(
            "[bold red]Error:[/] Failed to import dashboard components. "
            "Ensure Textual is installed."
        )
        console.print(f"[dim]{e}[/]")
        raise typer.Exit(1)

    # Validate log file if provided
    if log_file:
        log_path = Path(log_file)
        if not log_path.exists():
            console.print(f"[bold red]Error:[/] Log file not found: {log_file}")
            raise typer.Exit(1)
        if log_path.suffix.lower() not in (".jsonl", ".json", ".log"):
            console.print(
                "[bold yellow]Warning:[/] Log file does not have expected extension "
                "(.jsonl, .json, .log). Proceeding anyway."
            )

    # Run dashboard
    if demo:
        await run_demo_dashboard(log_file, live)
    else:
        app = ObservabilityDashboard(
            log_file=log_file,
            subscribe_to_bus=live,
            wire_log=wire_log,
        )
        await app.run_async()


async def run_demo_dashboard(
    log_file: Optional[str] = None,
    live: bool = True,
) -> None:
    """Run the dashboard in demo mode with simulated events.

    Args:
        log_file: Optional path to JSONL log file
        live: Whether to enable live event streaming
    """
    import random
    from datetime import datetime, timezone

    from victor.observability.dashboard import ObservabilityDashboard
    from victor.core.events import get_observability_bus

    # Create dashboard
    app = ObservabilityDashboard(
        log_file=log_file,
        subscribe_to_bus=live,
    )

    # Get the ObservabilityBus singleton instance
    bus = get_observability_bus()
    await bus.connect()

    # Demo event generator
    async def generate_demo_events() -> None:
        """Generate simulated events for demo mode."""
        tool_names = [
            "read_file",
            "write_file",
            "run_bash",
            "search_codebase",
            "analyze_code",
            "git_status",
            "git_diff",
        ]

        stages = [
            "INITIAL",
            "PLANNING",
            "READING",
            "ANALYZING",
            "EXECUTING",
            "VERIFICATION",
            "COMPLETION",
        ]

        verticals = ["coding", "research", "devops", "dataanalysis"]

        current_stage = "INITIAL"

        # Wait for app to start
        await asyncio.sleep(1)

        while True:
            try:
                # Generate random event
                event_type = random.choice(["tool", "state", "model", "vertical", "error"])

                if event_type == "tool":
                    tool = random.choice(tool_names)
                    # Start event
                    await bus.emit(
                        f"tool.{tool}.start",
                        {
                            "tool_name": tool,
                            "arguments": {"path": "/demo/file.py"},
                        },
                    )
                    await asyncio.sleep(random.uniform(0.1, 0.5))

                    # End event
                    success = random.random() > 0.1
                    await bus.emit(
                        f"tool.{tool}.end",
                        {
                            "tool_name": tool,
                            "success": success,
                            "duration_ms": random.uniform(10, 500),
                            "result": "Demo result" if success else None,
                            "error": "Demo error" if not success else None,
                        },
                    )

                elif event_type == "state":
                    current_idx = stages.index(current_stage)
                    if current_idx < len(stages) - 1:
                        new_stage = stages[current_idx + 1]
                    else:
                        new_stage = stages[0]

                    # Emit state transition event
                    await bus.emit(
                        "state.stage_changed",
                        {
                            "old_stage": current_stage,
                            "new_stage": new_stage,
                            "confidence": random.uniform(0.7, 1.0),
                        },
                    )
                    current_stage = new_stage

                elif event_type == "model":
                    await bus.emit(
                        "model.response",
                        {
                            "provider": "anthropic",
                            "model": "claude-3-5-sonnet",
                            "tokens_used": random.randint(100, 2000),
                            "latency_ms": random.uniform(200, 2000),
                        },
                    )

                elif event_type == "vertical":
                    vertical = random.choice(verticals)
                    await bus.emit(
                        "vertical.applied",
                        {
                            "vertical": vertical,
                            "action": "applied",
                            "config": {
                                "mode": "build",
                                "tool_budget": 25,
                            },
                        },
                    )

                elif event_type == "error":
                    await bus.emit(
                        "error.demo",
                        {
                            "error_message": "Demo error for testing",
                            "component": "demo",
                            "recoverable": True,
                        },
                    )

                await asyncio.sleep(random.uniform(0.5, 2.0))

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    # Start event generator as background task
    if live:
        generator_task = asyncio.create_task(generate_demo_events())
    else:
        generator_task = None

    try:
        await app.run_async()
    finally:
        if generator_task:
            generator_task.cancel()
            try:
                await generator_task
            except asyncio.CancelledError:
                pass


@dashboard_app.command("status")
def dashboard_status() -> None:
    """Show current event bus status and statistics."""
    try:
        from victor.core.events import get_observability_bus
    except ImportError:
        console.print("[red]Error: Could not import ObservabilityBus[/]")
        raise typer.Exit(1)

    # Verify the bus is accessible
    _ = get_observability_bus()

    console.print("\n[bold]ObservabilityBus Status[/]\n")
    console.print("  Status: Running")
    console.print("  Type: In-memory backend")
    console.print("\n[dim]Note: The new ObservabilityBus uses topic-based routing[/]")
    console.print("[dim]instead of category-based routing. See victor.core.events[/]")
    console.print("[dim]for more details on the unified event system.[/]\n")
