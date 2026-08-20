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

"""Observability CLI commands for Victor.

This module provides CLI commands for interacting with the ObservabilityManager,
including displaying metrics and dashboard data.

Phase 4: Enhance Observability with Unified Dashboard

Commands:
    victor observability dashboard - Show dashboard data
    victor observability metrics [source] - Show metrics for specific source
    victor observability history [hours] - Show historical metrics
    victor observability stats - Show observability manager statistics
"""

import json
import sys
from datetime import datetime
from typing import Optional

import typer

from victor.framework.observability import (
    DashboardData,
    MetricNames,
    MetricsCollection,
    MetricsSnapshot,
    ObservabilityManager,
)
from victor.ui.json_utils import create_json_option, print_json_data

app = typer.Typer(
    name="observability",
    help="Observability commands for monitoring Victor framework metrics.",
)


def format_timestamp(timestamp: float) -> str:
    """Format a timestamp for display.

    Args:
        timestamp: Unix timestamp

    Returns:
        Formatted datetime string
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_percentage(value: float) -> str:
    """Format a value as a percentage.

    Args:
        value: Decimal value (0.0 to 1.0)

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.1f}%"


def format_bytes(bytes_value: int) -> str:
    """Format bytes for human-readable display.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string with appropriate unit
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


@app.command("dashboard")
def dashboard_command(
    json_output: bool = create_json_option(),
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuously update dashboard"),
    interval: float = typer.Option(
        5.0, "--interval", "-i", help="Update interval in seconds (for --watch)"
    ),
    rich: bool = typer.Option(
        False,
        "--rich",
        "-r",
        help="Use rich terminal dashboard (requires rich library)",
    ),
) -> None:
    """Display observability dashboard data.

    Shows aggregated metrics from all registered sources including:
    - Cache performance (hit rate, size, evictions)
    - Tool metrics (calls, errors, latency)
    - Coordinator metrics (operations, errors, latency)
    - Capability metrics (accesses, errors)
    - System metrics (memory, CPU)
    - Alerts and warnings

    Example:
        victor observability dashboard
        victor observability dashboard --json
        victor observability dashboard --watch --interval 10
        victor observability dashboard --rich
    """
    # Use rich dashboard if requested
    if rich:
        try:
            from victor.ui.dashboard.display import MetricsDashboard

            dashboard = MetricsDashboard()

            if watch:
                dashboard.run(refresh_interval=interval)
            else:
                dashboard.display()

            return
        except ImportError:
            typer.echo(
                "Error: Rich dashboard requires rich library. Install it with: pip install rich",
                err=True,
            )
            raise typer.Exit(1)

    def print_dashboard() -> None:
        """Print the dashboard."""
        manager = ObservabilityManager.get_instance()
        data = manager.get_dashboard_data()

        if json_output:
            print_json_data(data.to_dict())
            return

        # Print formatted dashboard
        typer.echo("\n" + "=" * 60)
        typer.echo(f"VICTOR OBSERVABILITY DASHBOARD - {format_timestamp(data.timestamp)}")
        typer.echo("=" * 60 + "\n")

        # Sources summary
        typer.echo("📊 REGISTERED SOURCES")
        typer.echo("-" * 40)
        typer.echo(f"Total Sources: {data.total_sources}")
        if data.sources_by_type:
            for source_type, count in sorted(data.sources_by_type.items()):
                typer.echo(f"  {source_type}: {count}")
        typer.echo("")

        # Cache metrics
        if data.cache_metrics:
            typer.echo("💾 CACHE METRICS")
            typer.echo("-" * 40)
            total = data.cache_metrics["total_hits"] + data.cache_metrics["total_misses"]
            if total > 0:
                typer.echo(f"Hit Rate: {format_percentage(data.cache_metrics['hit_rate'])}")
                typer.echo(f"Total Hits: {data.cache_metrics['total_hits']:,} / {total:,}")
            if data.cache_metrics["total_size"] > 0:
                typer.echo(f"Cache Size: {format_bytes(data.cache_metrics['total_size'])}")
            if data.cache_metrics["evictions"] > 0:
                typer.echo(f"Evictions: {data.cache_metrics['evictions']:,}")
            typer.echo("")

        # Tool metrics
        if data.tool_metrics and data.tool_metrics.get("total_calls", 0) > 0:
            typer.echo("🔧 TOOL METRICS")
            typer.echo("-" * 40)
            typer.echo(f"Total Calls: {data.tool_metrics['total_calls']:,}")
            typer.echo(f"Success Rate: {format_percentage(data.tool_metrics['success_rate'])}")
            if data.tool_metrics["average_latency"] > 0:
                typer.echo(f"Avg Latency: {data.tool_metrics['average_latency'] * 1000:.1f}ms")
            typer.echo("")

        # Coordinator metrics
        if data.coordinator_metrics and data.coordinator_metrics.get("total_operations", 0) > 0:
            typer.echo("🎯 COORDINATOR METRICS")
            typer.echo("-" * 40)
            typer.echo(f"Total Operations: {data.coordinator_metrics['total_operations']:,}")
            if data.coordinator_metrics["average_latency"] > 0:
                typer.echo(
                    f"Avg Latency: {data.coordinator_metrics['average_latency'] * 1000:.1f}ms"
                )
            typer.echo("")

        # Capability metrics
        if data.capability_metrics and data.capability_metrics.get("capability_count", 0) > 0:
            typer.echo("⚡ CAPABILITY METRICS")
            typer.echo("-" * 40)
            typer.echo(f"Capabilities: {data.capability_metrics['capability_count']}")
            typer.echo(f"Total Accesses: {data.capability_metrics['total_accesses']:,}")
            typer.echo("")

        # Vertical metrics
        if data.vertical_metrics and data.vertical_metrics.get("vertical_count", 0) > 0:
            typer.echo("🚀 VERTICAL METRICS")
            typer.echo("-" * 40)
            typer.echo(f"Verticals: {data.vertical_metrics['vertical_count']}")
            typer.echo(f"Total Requests: {data.vertical_metrics['total_requests']:,}")
            if data.vertical_metrics["average_latency"] > 0:
                typer.echo(f"Avg Latency: {data.vertical_metrics['average_latency'] * 1000:.1f}ms")
            typer.echo("")

        # System metrics
        if data.system_metrics:
            typer.echo("🖥️  SYSTEM METRICS")
            typer.echo("-" * 40)
            if data.system_metrics.get("memory_usage_bytes", 0) > 0:
                typer.echo(f"Memory: {format_bytes(data.system_metrics['memory_usage_bytes'])}")
            if data.system_metrics.get("cpu_usage_percent", 0) > 0:
                typer.echo(f"CPU: {data.system_metrics['cpu_usage_percent']:.1f}%")
            typer.echo("")

        # Alerts
        if data.alerts:
            typer.echo("⚠️  ALERTS")
            typer.echo("-" * 40)
            for alert in data.alerts:
                severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    alert["severity"], "⚪"
                )
                typer.echo(f"{severity_icon} [{alert['severity'].upper()}] {alert['message']}")
            typer.echo("")

        typer.echo("=" * 60)

    if watch:
        import time

        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")
                print_dashboard()
                typer.echo(f"\nRefreshing every {interval}s... (Ctrl+C to exit)")
                time.sleep(interval)
        except KeyboardInterrupt:
            typer.echo("\n\nDashboard stopped.")
    else:
        print_dashboard()


@app.command("metrics")
def metrics_command(
    source: Optional[str] = typer.Argument(
        None, help="Source ID to filter by (shows all if not specified)"
    ),
    source_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by source type"),
    json_output: bool = create_json_option(),
) -> None:
    """Show metrics for a specific source or all sources.

    Args:
        source: Source ID to filter by
        source_type: Filter by source type (e.g., "cache", "tool", "coordinator")
        json_output: Output as JSON

    Example:
        victor observability metrics
        victor observability metrics my-cache
        victor observability metrics --type cache
        victor observability metrics --json
    """
    manager = ObservabilityManager.get_instance()

    if source:
        # Get metrics for specific source
        collection = manager.collect_metrics()
        snapshot = collection.get_by_source_id(source)

        if snapshot is None:
            typer.echo(f"Error: Source '{source}' not found", err=True)
            raise typer.Exit(1)

        snapshots = [snapshot]
    else:
        # Get all metrics, optionally filtered by type
        collection = manager.collect_metrics()

        if source_type:
            snapshots = collection.get_by_source_type(source_type)
            if not snapshots:
                typer.echo(f"Error: No sources found with type '{source_type}'", err=True)
                raise typer.Exit(1)
        else:
            snapshots = collection.snapshots

    if json_output:
        output = [s.to_dict() for s in snapshots]
        print_json_data(output)
        return

    # Print formatted metrics
    for snapshot in snapshots:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"{snapshot.source_id} ({snapshot.source_type})")
        typer.echo(f"Updated: {format_timestamp(snapshot.timestamp)}")
        typer.echo("=" * 60)

        for metric in snapshot.metrics:
            typer.echo(f"\n{metric.name}")
            typer.echo(f"  Description: {metric.description}")

            labels_str = ", ".join(f"{label.key}={label.value}" for label in metric.labels)
            if labels_str:
                typer.echo(f"  Labels: {labels_str}")

            if metric.metric_type.value == "counter":
                from victor.framework.observability import CounterMetric

                if isinstance(metric, CounterMetric):
                    typer.echo(f"  Value: {metric.value:,}")
            elif metric.metric_type.value == "gauge":
                from victor.framework.observability import GaugeMetric

                if isinstance(metric, GaugeMetric):
                    typer.echo(f"  Value: {metric.value:.2f}")
                    if metric.min_value is not None:
                        typer.echo(f"  Min: {metric.min_value:.2f}")
                    if metric.max_value is not None:
                        typer.echo(f"  Max: {metric.max_value:.2f}")
            elif metric.metric_type.value == "histogram":
                from victor.framework.observability import HistogramMetric

                if isinstance(metric, HistogramMetric):
                    typer.echo(f"  Count: {metric.count:,}")
                    typer.echo(f"  Sum: {metric.sum:.2f}")
                    typer.echo(f"  Average: {metric.average:.2f}")
            elif metric.metric_type.value == "summary":
                from victor.framework.observability import SummaryMetric

                if isinstance(metric, SummaryMetric):
                    typer.echo(f"  Count: {metric.count:,}")
                    typer.echo(f"  Average: {metric.average:.2f}")
                    if metric.p50 is not None:
                        typer.echo(f"  P50: {metric.p50:.2f}")
                    if metric.p95 is not None:
                        typer.echo(f"  P95: {metric.p95:.2f}")


@app.command("history")
def history_command(
    hours: float = typer.Option(1.0, "--hours", "-h", help="Hours of history to show"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source ID"),
    source_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by source type"),
    json_output: bool = create_json_option(),
) -> None:
    """Show historical metrics data.

    Args:
        hours: Hours of history to retrieve (default: 1.0)
        source: Filter by source ID
        source_type: Filter by source type
        json_output: Output as JSON

    Example:
        victor observability history
        victor observability history --hours 6
        victor observability history --type cache --hours 24
    """
    manager = ObservabilityManager.get_instance()

    history = manager.get_historical_data(source_id=source, source_type=source_type, hours=hours)

    if not history:
        typer.echo("No historical data available")
        return

    if json_output:
        output = [c.to_dict() for c in history]
        print_json_data(output)
        return

    typer.echo(f"\nHistorical metrics for the last {hours} hours\n")

    for collection in history:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Collection: {format_timestamp(collection.timestamp)}")
        typer.echo(f"Sources: {len(collection.snapshots)}")
        typer.echo("=" * 60)

        for snapshot in collection.snapshots:
            typer.echo(
                f"  {snapshot.source_id} ({snapshot.source_type}): {len(snapshot.metrics)} metrics"
            )


@app.command("stats")
def stats_command(
    json_output: bool = create_json_option(),
) -> None:
    """Show observability manager statistics.

    Displays statistics about the observability manager itself,
    including collection counts, errors, and history size.

    Example:
        victor observability stats
        victor observability stats --json
    """
    manager = ObservabilityManager.get_instance()
    stats = manager.get_stats()

    if json_output:
        print_json_data(stats)
        return

    typer.echo("\nObservability Manager Statistics")
    typer.echo("=" * 40)
    typer.echo(f"Collections: {stats['collection_count']:,}")
    typer.echo(f"Errors: {stats['collection_errors']:,}")
    if stats["collection_count"] > 0:
        error_rate = stats["collection_errors"] / stats["collection_count"]
        typer.echo(f"Error Rate: {format_percentage(error_rate)}")
    typer.echo(f"Last Collection Duration: {stats['last_collection_duration']:.3f}s")
    typer.echo(f"Registered Sources: {stats['registered_sources']}")
    typer.echo(f"History Size: {stats['history_size']}")
    typer.echo(f"History Retention: {stats['history_retention_hours']:.1f} hours")


@app.command("sources")
def sources_command(
    json_output: bool = create_json_option(),
) -> None:
    """List all registered metrics sources.

    Example:
        victor observability sources
        victor observability sources --json
    """
    manager = ObservabilityManager.get_instance()
    sources = manager.list_sources()

    if json_output:
        print_json_data({"sources": sources})
        return

    if not sources:
        typer.echo("No metrics sources registered")
        return

    typer.echo(f"\nRegistered Metrics Sources ({len(sources)})\n")
    for i, source in enumerate(sorted(sources), 1):
        typer.echo(f"  {i}. {source}")


__all__ = ["app"]
