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

"""Bayesian orchestration monitoring CLI command."""

from typing import Optional

import typer

from victor.framework.rl.monitoring.reporting import (
    DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    get_bayesian_monitoring_service,
    parse_agent_ids,
)

app = typer.Typer(help="Bayesian orchestration monitoring and metrics")


@app.command()
def summary(
    days: int = typer.Option(
        DEFAULT_BAYESIAN_LOOKBACK_DAYS,
        "--days",
        help="Number of days to look back",
    ),
    output: Optional[str] = typer.Option(None, help="Export summary to JSON file"),
):
    """Show system summary of Bayesian orchestration metrics."""
    service = get_bayesian_monitoring_service()
    typer.echo(service.render_summary(days))
    if output:
        service.export_summary_json(output, days)
        typer.echo(f"\nSummary exported to {output}")


@app.command()
def reliability(
    agent_ids: Optional[str] = typer.Option(
        None,
        "--agents",
        help="Comma-separated list of agent IDs",
    ),
    days: int = typer.Option(
        DEFAULT_BAYESIAN_LOOKBACK_DAYS,
        "--days",
        help="Number of days to look back",
    ),
    export: Optional[str] = typer.Option(None, help="Export to CSV file"),
):
    """Show agent reliability trends."""
    service = get_bayesian_monitoring_service()
    parsed_agent_ids = parse_agent_ids(agent_ids)
    typer.echo(service.render_reliability(parsed_agent_ids, days))
    if export:
        service.export_reliability_csv(export, parsed_agent_ids, days)
        typer.echo(f"\nReliability trends exported to {export}")


@app.command()
def consensus(
    days: int = typer.Option(
        DEFAULT_BAYESIAN_LOOKBACK_DAYS,
        "--days",
        help="Number of days to look back",
    ),
):
    """Show consensus performance statistics."""
    service = get_bayesian_monitoring_service()
    typer.echo(service.render_consensus(days))


@app.command()
def voi(
    agent_id: Optional[str] = typer.Option(None, "--agent", help="Specific agent ID"),
    days: int = typer.Option(
        DEFAULT_BAYESIAN_LOOKBACK_DAYS,
        "--days",
        help="Number of days to look back",
    ),
):
    """Show Value of Information statistics."""
    service = get_bayesian_monitoring_service()
    typer.echo(service.render_voi(agent_id, days))


@app.command()
def correlations(
    agents: str = typer.Argument(..., help="Comma-separated list of agent IDs"),
    days: int = typer.Option(
        DEFAULT_BAYESIAN_LOOKBACK_DAYS,
        "--days",
        help="Number of days to look back",
    ),
):
    """Show correlation matrix heatmap."""
    parsed_agent_ids = parse_agent_ids(agents)
    if not parsed_agent_ids:
        raise typer.BadParameter("At least one agent ID is required.")
    service = get_bayesian_monitoring_service()
    typer.echo(service.render_correlations(parsed_agent_ids, days))


@app.command()
def belief(
    belief_id: str = typer.Argument(..., help="Belief state ID"),
    export: Optional[str] = typer.Option(None, help="Export evolution to CSV file"),
):
    """Show belief state evolution."""
    service = get_bayesian_monitoring_service()
    typer.echo(service.render_belief(belief_id))
    if export:
        service.export_belief_csv(belief_id, export)
        typer.echo(f"\nBelief evolution exported to {export}")


if __name__ == "__main__":
    app()
