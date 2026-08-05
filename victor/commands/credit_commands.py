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

"""
Credit Assignment CLI Commands.

Provides CLI commands for analyzing credit assignment in agent workflows:
- Analyze trajectories and assign credit
- Visualize credit distribution
- Export attribution data
- Compare different methodologies
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from victor.framework.rl import (
    CreditAssignmentIntegration,
    CreditGranularity,
    CreditMethodology,
    ActionMetadata,
    CreditSignal,
    compute_credit_metrics,
    visualize_credit_assignment,
)

app = typer.Typer(
    name="credit",
    help="Credit assignment analysis for agent workflows",
    add_completion=False,
)


@app.command()
def analyze(
    trajectory_file: Path = typer.Argument(
        ...,
        help="Path to trajectory JSON file",
        exists=True,
    ),
    methodology: str = typer.Option(
        "gae",
        "--methodology",
        "-m",
        help="Credit assignment methodology",
        show_choices=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """Analyze a trajectory and assign credit to actions.

    Example:
        victor credit analyze trajectory.json --methodology gae -o results.json
    """
    # Load trajectory
    try:
        with open(trajectory_file) as f:
            data = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading trajectory: {e}", err=True)
        raise typer.Exit(1)

    # Parse methodology
    try:
        method = CreditMethodology(methodology.lower())
    except ValueError:
        typer.echo(f"Unknown methodology: {methodology}", err=True)
        typer.echo(f"Valid options: {[m.value for m in CreditMethodology]}")
        raise typer.Exit(1)

    # Extract trajectory and rewards
    trajectory_data = data.get("trajectory", [])
    rewards = data.get("rewards", [])

    if len(trajectory_data) != len(rewards):
        typer.echo("Error: trajectory and rewards must have same length", err=True)
        raise typer.Exit(1)

    # Create ActionMetadata objects
    trajectory = []
    for i, item in enumerate(trajectory_data):
        metadata = ActionMetadata(
            agent_id=item.get("agent_id", "unknown"),
            action_id=item.get("action_id", f"action_{i}"),
            turn_index=item.get("turn_index", 0),
            step_index=item.get("step_index", i),
            tool_name=item.get("tool_name"),
            method_name=item.get("method_name"),
        )
        trajectory.append(metadata)

    # Assign credit
    integration = CreditAssignmentIntegration()
    signals = integration.assign_credit(trajectory, rewards, methodology=method)

    # Compute metrics
    metrics = compute_credit_metrics(signals)

    # Display results
    if verbose:
        typer.echo("\n" + "=" * 60)
        typer.echo("CREDIT ASSIGNMENT RESULTS")
        typer.echo("=" * 60)
        typer.echo(f"\nMethodology: {method.value}")
        typer.echo(f"Trajectory length: {len(trajectory)}")
        typer.echo(f"Total reward: {sum(rewards):.3f}")
        typer.echo("\nMetrics:")
        for key, value in metrics.items():
            typer.echo(f"  {key}: {value}")

        typer.echo("\nVisualization:")
        typer.echo(visualize_credit_assignment(signals))

    # Prepare output
    results = {
        "methodology": method.value,
        "trajectory_length": len(trajectory),
        "total_reward": sum(rewards),
        "metrics": metrics,
        "signals": [s.to_dict() for s in signals],
    }

    # Save or print results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"\nResults saved to: {output}")
    else:
        typer.echo("\n" + json.dumps(results, indent=2))


@app.command()
def compare(
    trajectory_file: Path = typer.Argument(
        ...,
        help="Path to trajectory JSON file",
        exists=True,
    ),
    methodologies: List[str] = typer.Option(
        ["gae", "shapley", "hindsight"],
        "--methodology",
        "-m",
        help="Methodologies to compare",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for comparison (JSON format)",
    ),
) -> None:
    """Compare multiple credit assignment methodologies.

    Example:
        victor credit compare trajectory.json -m gae shapley -o comparison.json
    """
    # Load trajectory
    with open(trajectory_file) as f:
        data = json.load(f)

    trajectory_data = data.get("trajectory", [])
    rewards = data.get("rewards", [])

    # Create ActionMetadata objects
    trajectory = []
    for i, item in enumerate(trajectory_data):
        metadata = ActionMetadata(
            agent_id=item.get("agent_id", "unknown"),
            action_id=item.get("action_id", f"action_{i}"),
            turn_index=item.get("turn_index", 0),
            step_index=item.get("step_index", i),
        )
        trajectory.append(metadata)

    # Compare methodologies
    comparison = {}
    for method_str in methodologies:
        try:
            method = CreditMethodology(method_str.lower())
            integration = CreditAssignmentIntegration()
            signals = integration.assign_credit(trajectory, rewards, methodology=method)
            metrics = compute_credit_metrics(signals)

            comparison[method_str] = {
                "metrics": metrics,
                "signals": [s.to_dict() for s in signals],
            }
        except ValueError as e:
            typer.echo(f"Warning: Invalid methodology '{method_str}': {e}")

    # Display comparison
    typer.echo("\n" + "=" * 60)
    typer.echo("METHODOLOGY COMPARISON")
    typer.echo("=" * 60)

    for method_name, results in comparison.items():
        typer.echo(f"\n{method_name.upper()}:")
        metrics = results["metrics"]
        typer.echo(f"  Total credit: {metrics['total_credit']:.3f}")
        typer.echo(f"  Avg confidence: {metrics['avg_confidence']:.3f}")
        typer.echo(f"  Positive ratio: {metrics['positive_ratio']:.3f}")

    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(comparison, f, indent=2)
        typer.echo(f"\nComparison saved to: {output}")


@app.command()
def visualize(
    results_file: Path = typer.Argument(
        ...,
        help="Path to credit assignment results JSON file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save visualization to file",
    ),
    format: str = typer.Option(
        "ascii",
        "--format",
        "-f",
        help="Visualization format",
    ),
) -> None:
    """Visualize credit assignment results.

    Example:
        victor credit visualize results.json -o visualization.txt
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    # Reconstruct signals
    signals = []
    for signal_dict in data.get("signals", []):
        # Simple reconstruction for visualization
        signals.append(
            CreditSignal(
                action_id=signal_dict["action_id"],
                raw_reward=signal_dict["raw_reward"],
                credit=signal_dict["credit"],
                confidence=signal_dict.get("confidence", 0.0),
                methodology=(
                    CreditMethodology(signal_dict["methodology"])
                    if signal_dict.get("methodology")
                    else None
                ),
                granularity=CreditGranularity(signal_dict["granularity"]),
            )
        )

    # Generate visualization
    viz = visualize_credit_assignment(signals, output=str(output) if output else None)

    typer.echo(viz)

    if output:
        typer.echo(f"\nVisualization saved to: {output}")


@app.command()
def export(
    results_file: Path = typer.Argument(
        ...,
        help="Path to credit assignment results JSON file",
        exists=True,
    ),
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Export format",
    ),
    output: Path = typer.Option(
        Path("credit_export.csv"),
        "--output",
        "-o",
        help="Output file",
    ),
) -> None:
    """Export credit assignment data to various formats.

    Example:
        victor credit export results.json -f csv -o attribution.csv
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    signals = data.get("signals", [])

    output.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "csv":
        import csv

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "action_id",
                    "raw_reward",
                    "credit",
                    "confidence",
                    "methodology",
                    "granularity",
                    "agent_id",
                ]
            )

            for signal_dict in signals:
                writer.writerow(
                    [
                        signal_dict["action_id"],
                        signal_dict["raw_reward"],
                        signal_dict["credit"],
                        signal_dict.get("confidence", 0.0),
                        signal_dict.get("methodology", ""),
                        signal_dict["granularity"],
                        signal_dict.get("metadata", {}).get("agent_id", ""),
                    ]
                )

    elif format.lower() == "json":
        with open(output, "w") as f:
            json.dump(signals, f, indent=2)

    elif format.lower() == "md":
        # Markdown table
        with open(output, "w") as f:
            f.write("# Credit Assignment Results\n\n")
            f.write("| Action ID | Reward | Credit | Confidence | Methodology |\n")
            f.write("|-----------|--------|--------|------------|-------------|\n")

            for signal_dict in signals:
                f.write(
                    f"| {signal_dict['action_id']} "
                    f"| {signal_dict['raw_reward']:.3f} "
                    f"| {signal_dict['credit']:.3f} "
                    f"| {signal_dict.get('confidence', 0.0):.3f} "
                    f"| {signal_dict.get('methodology', 'N/A')} |\n"
                )

    else:
        typer.echo(f"Unknown format: {format}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Exported to: {output}")


@app.command()
def template(
    output: Path = typer.Option(
        Path("trajectory_template.json"),
        "--output",
        "-o",
        help="Output file for template",
    ),
) -> None:
    """Generate a trajectory template file.

    Example:
        victor credit template -o my_trajectory.json
    """
    template_data = {
        "description": "Trajectory template for credit assignment analysis",
        "trajectory": [
            {
                "action_id": "action_0",
                "agent_id": "agent_1",
                "turn_index": 0,
                "step_index": 0,
                "tool_name": "search",
                "method_name": "code_search",
            },
            {
                "action_id": "action_1",
                "agent_id": "agent_1",
                "turn_index": 0,
                "step_index": 1,
                "tool_name": "read",
                "method_name": "read_file",
            },
            {
                "action_id": "action_2",
                "agent_id": "agent_2",
                "turn_index": 1,
                "step_index": 2,
                "tool_name": "edit",
                "method_name": "edit_file",
            },
        ],
        "rewards": [0.3, 0.5, 0.2],
        "metadata": {
            "task": "Example task",
            "total_duration_ms": 1500,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(template_data, f, indent=2)

    typer.echo(f"Template saved to: {output}")
    typer.echo("\nEdit the file with your trajectory data, then run:")
    typer.echo(f"  victor credit analyze {output}")


if __name__ == "__main__":
    app()
