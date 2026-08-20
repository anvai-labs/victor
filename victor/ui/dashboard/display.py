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

"""Metrics dashboard display functionality.

Provides terminal-based visualization of metrics using rich library.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from victor.ui.dashboard.data import (
    DashboardData,
    DashboardDataProvider,
    DashboardMetric,
    get_dashboard_provider,
)


class MetricsDashboard:
    """Terminal-based metrics dashboard.

    Provides real-time visualization of agent metrics including:
    - Token usage (input, output, cached, reasoning)
    - Tool calls (count, success rate, duration)
    - LLM calls (count, success rate, duration by model)
    - Performance metrics (duration, state transitions)
    - Recent errors

    Example:
        dashboard = MetricsDashboard()
        dashboard.run(refresh_interval=2.0)

    Or programmatically:
        dashboard = MetricsDashboard()
        data = dashboard.get_provider().get_dashboard_data()
        dashboard.display(data)
    """

    def __init__(
        self,
        provider: Optional[DashboardDataProvider] = None,
        console: Optional[Any] = None,
    ) -> None:
        """Initialize the metrics dashboard.

        Args:
            provider: Optional dashboard data provider
            console: Optional Rich console instance
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is required for the dashboard. " "Install it with: pip install rich"
            )

        self._provider = provider or get_dashboard_provider()
        self._console = console or Console()

    def run(
        self,
        refresh_interval: float = 1.0,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Run the dashboard with auto-refresh.

        Args:
            refresh_interval: Seconds between refreshes
            max_iterations: Maximum number of refresh iterations (None = infinite)
        """
        try:
            with Live(
                self._generate_display(),
                console=self._console,
                refresh_per_second=1.0 / refresh_interval,
            ) as live:
                iterations = 0
                while max_iterations is None or iterations < max_iterations:
                    live.update(self._generate_display())
                    time.sleep(refresh_interval)
                    iterations += 1
        except KeyboardInterrupt:
            self._console.print("\n[yellow]Dashboard stopped.[/yellow]")

    def display(self, data: Optional[DashboardData] = None) -> None:
        """Display dashboard data once.

        Args:
            data: Optional dashboard data (fetches current if None)
        """
        if data is None:
            data = self._provider.get_dashboard_data()

        display = self._generate_display(data)
        self._console.print(display)

    def _generate_display(self, data: Optional[DashboardData] = None) -> Panel:
        """Generate the dashboard display.

        Args:
            data: Optional dashboard data

        Returns:
            Panel with dashboard content
        """
        if data is None:
            data = self._provider.get_dashboard_data()

        content = self._generate_content(data)
        return Panel(
            content,
            title="[bold cyan]Victor Metrics Dashboard[/bold cyan]",
            subtitle=f"[dim]{data.datetime.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def _generate_content(self, data: DashboardData) -> Any:
        """Generate dashboard content.

        Args:
            data: Dashboard data

        Returns:
            Rich renderable content
        """
        # Create a grid layout
        from rich.columns import Columns

        sections = []

        # Token usage section
        sections.append(self._create_token_section(data))

        # Tool usage section
        sections.append(self._create_tool_section(data))

        # LLM usage section
        sections.append(self._create_llm_section(data))

        # Performance section
        sections.append(self._create_performance_section(data))

        # Errors section (if any)
        if data.errors:
            sections.append(self._create_errors_section(data))

        return Columns(sections, expand=True)

    def _create_token_section(self, data: DashboardData) -> Panel:
        """Create token usage section.

        Args:
            data: Dashboard data

        Returns:
            Panel with token metrics
        """
        table = Table(title="[bold]Token Usage[/bold]", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Trend", justify="center")

        # Total tokens
        total = data.token_metrics.get("total")
        if total:
            trend = self._format_trend(total.trend)
            table.add_row("Total", f"{int(total.value):,}", trend)

        # Input tokens
        input_m = data.token_metrics.get("input")
        if input_m:
            trend = self._format_trend(input_m.trend)
            table.add_row("Input", f"{int(input_m.value):,}", trend)

        # Output tokens
        output = data.token_metrics.get("output")
        if output:
            trend = self._format_trend(output.trend)
            table.add_row("Output", f"{int(output.value):,}", trend)

        # Cached tokens
        cached = data.token_usage.get("cached", 0)
        table.add_row("Cached", f"{cached:,}", "")

        return Panel(table, border_style="blue")

    def _create_tool_section(self, data: DashboardData) -> Panel:
        """Create tool usage section.

        Args:
            data: Dashboard data

        Returns:
            Panel with tool metrics
        """
        table = Table(title="[bold]Tool Usage[/bold]", box=box.SIMPLE)
        table.add_column("Metric", style="green")
        table.add_column("Value", justify="right")
        table.add_column("Trend", justify="center")

        # Total calls
        calls = data.tool_metrics.get("total_calls")
        if calls:
            trend = self._format_trend(calls.trend)
            table.add_row("Calls", f"{int(calls.value)}", trend)

        # Success rate
        rate = data.tool_metrics.get("success_rate")
        if rate:
            trend = self._format_trend(rate.trend)
            table.add_row("Success", f"{rate.value:.1%}", trend)

        # Average duration
        duration = data.tool_metrics.get("avg_duration")
        if duration:
            trend = self._format_trend(duration.trend)
            table.add_row("Avg Time", f"{duration.value:.2f}s", trend)

        return Panel(table, border_style="green")

    def _create_llm_section(self, data: DashboardData) -> Panel:
        """Create LLM usage section.

        Args:
            data: Dashboard data

        Returns:
            Panel with LLM metrics
        """
        table = Table(title="[bold]LLM Calls[/bold]", box=box.SIMPLE)
        table.add_column("Metric", style="yellow")
        table.add_column("Value", justify="right")
        table.add_column("Trend", justify="center")

        # Total calls
        calls = data.llm_metrics.get("total_calls")
        if calls:
            trend = self._format_trend(calls.trend)
            table.add_row("Calls", f"{int(calls.value)}", trend)

        # Success rate
        rate = data.llm_metrics.get("success_rate")
        if rate:
            trend = self._format_trend(rate.trend)
            table.add_row("Success", f"{rate.value:.1%}", trend)

        # Average duration
        duration = data.llm_metrics.get("avg_duration")
        if duration:
            trend = self._format_trend(duration.trend)
            table.add_row("Avg Time", f"{duration.value:.2f}s", trend)

        return Panel(table, border_style="yellow")

    def _create_performance_section(self, data: DashboardData) -> Panel:
        """Create performance section.

        Args:
            data: Dashboard data

        Returns:
            Panel with performance metrics
        """
        table = Table(title="[bold]Performance[/bold]", box=box.SIMPLE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", justify="right")

        # Duration
        if data.performance.get("duration"):
            table.add_row("Duration", f"{data.performance['duration']:.2f}s")

        # State transitions
        transitions = data.performance.get("state_transitions", 0)
        table.add_row("State Changes", f"{transitions}")

        # Current state
        state = data.performance.get("current_state")
        if state:
            table.add_row("State", state)

        return Panel(table, border_style="magenta")

    def _create_errors_section(self, data: DashboardData) -> Panel:
        """Create errors section.

        Args:
            data: Dashboard data

        Returns:
            Panel with recent errors
        """
        table = Table(title="[bold red]Recent Errors[/bold red]", box=box.SIMPLE)
        table.add_column("Type", style="red")
        table.add_column("Message", style="dim")
        table.add_column("Time", style="dim")

        # Show last 10 errors
        for error in data.errors[-10:]:
            timestamp = time.strftime(
                "%H:%M:%S", time.localtime(error.get("timestamp", time.time()))
            )
            table.add_row(
                error.get("type", "Unknown"),
                error.get("message", "")[:50],
                timestamp,
            )

        return Panel(table, border_style="red")

    def _format_trend(self, trend: str) -> str:
        """Format trend indicator.

        Args:
            trend: Trend direction

        Returns:
            Formatted trend string
        """
        if trend == "up":
            return "[green]↑[/green]"
        elif trend == "down":
            return "[red]↓[/red]"
        else:
            return "[dim]-[/dim]"

    @property
    def provider(self) -> DashboardDataProvider:
        """Get the dashboard data provider.

        Returns:
            DashboardDataProvider instance
        """
        return self._provider

    def export_json(self, path: str) -> None:
        """Export current dashboard data to JSON.

        Args:
            path: Output file path
        """
        import json

        data = self._provider.get_dashboard_data()

        # Convert to dict for JSON serialization
        output = {
            "timestamp": data.timestamp,
            "datetime": data.datetime.isoformat(),
            "session_id": data.session_id,
            "agent_id": data.agent_id,
            "token_usage": data.token_usage,
            "tool_usage": data.tool_usage,
            "llm_usage": data.llm_usage,
            "performance": data.performance,
            "errors": data.errors,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        self._console.print(f"[green]Exported dashboard data to {path}[/green]")


def display_dashboard(
    refresh_interval: float = 1.0,
    max_iterations: Optional[int] = None,
) -> None:
    """Convenience function to display the metrics dashboard.

    Args:
        refresh_interval: Seconds between refreshes
        max_iterations: Maximum number of refresh iterations
    """
    dashboard = MetricsDashboard()
    dashboard.run(refresh_interval=refresh_interval, max_iterations=max_iterations)


def display_metrics_once() -> None:
    """Display current metrics once (non-refreshing)."""
    dashboard = MetricsDashboard()
    dashboard.display()
