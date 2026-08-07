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

"""Metrics and analytics slash commands: cost, metrics, serialization, learning, mlstats.

This module combines the previously duplicate _cmd_learning implementations
into a single unified command that shows stats from both:
1. The runtime-intelligence integration (AdaptiveModeController)
2. The RL model selector (RLCoordinator)
"""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


def _get_capability(agent, capability_name: str, default=None):
    """Get a capability value from the agent via CapabilityRegistry or fallback.

    Replaces scattered hasattr/getattr chains with a single lookup path.
    """
    # Primary: CapabilityRegistryMixin.get_capability_value()
    try:
        return agent.get_capability_value(capability_name)
    except (KeyError, TypeError, AttributeError):
        pass
    # Fallback: direct attribute access
    return getattr(agent, capability_name, default)


def _get_runtime_intelligence_integration(agent):
    """Resolve the canonical runtime-intelligence integration surface."""
    integration = _get_capability(agent, "runtime_intelligence_integration")
    if integration is not None:
        return integration
    return _get_capability(agent, "intelligent_pipeline")


@register_command
class CostCommand(BaseSlashCommand):
    """Show estimated token usage and cost for this session."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="cost",
            description="Show estimated token usage and cost for this session",
            usage="/cost",
            aliases=["usage"],
            category="metrics",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        analytics = _get_capability(ctx.agent, "usage_analytics")

        if analytics:
            summary = analytics.get_session_summary()
            content = (
                f"[bold]Session Statistics[/]\n\n"
                f"[bold]Messages:[/] {ctx.agent.conversation.message_count()}\n"
                f"[bold]Tool Calls:[/] {summary.get('tool_calls', 0)}\n"
                f"[bold]Total Tokens:[/] {summary.get('total_tokens', 0):,}\n"
                f"[bold]Estimated Cost:[/] ${summary.get('estimated_cost', 0):.4f}\n"
            )

            if summary.get("provider_breakdown"):
                content += "\n[bold]By Provider:[/]\n"
                for provider, stats in summary["provider_breakdown"].items():
                    content += f"  {provider}: {stats.get('tokens', 0):,} tokens\n"
        else:
            # Get tool call count via capability or public attribute
            tool_metrics = _get_capability(ctx.agent, "tool_metrics", {})
            tool_calls = tool_metrics.get("call_count", 0) if isinstance(tool_metrics, dict) else 0
            if not tool_calls:
                tool_calls = getattr(ctx.agent, "tool_calls_used", 0)

            content = (
                f"[bold]Session Statistics[/]\n\n"
                f"[bold]Messages:[/] {ctx.agent.conversation.message_count()}\n"
                f"[bold]Tool Calls:[/] {tool_calls}\n"
                f"[bold]Provider:[/] {ctx.agent.provider_name}\n"
                f"[bold]Model:[/] {ctx.agent.model}\n"
            )

        ctx.console.print(Panel(content, title="Usage Statistics", border_style="cyan"))


@register_command
class GainCommand(BaseSlashCommand):
    """Show cumulative shell-output condensation savings (rtk `gain` analogue)."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="gain",
            description="Show token savings from shell output condensation",
            usage="/gain",
            aliases=["savings", "condensation"],
            category="metrics",
        )

    def execute(self, ctx: CommandContext) -> None:
        try:
            from victor.agent.usage_analytics import UsageAnalytics

            stats = UsageAnalytics.get_instance().get_condensation_stats()
        except Exception as e:
            ctx.console.print(f"[red]Error fetching condensation stats:[/] {e}")
            return

        if not stats["total_events"]:
            ctx.console.print(
                "[dim]No condensation events yet — savings accumulate as shell "
                "commands (pytest, git, npm, ...) run.[/]"
            )
            return

        table = Table(title="Output Condensation Savings")
        table.add_column("Condenser", style="cyan")
        table.add_column("Events", justify="right")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Saved", justify="right", style="green")

        for name, s in sorted(
            stats["per_condenser"].items(),
            key=lambda kv: kv[1]["original_chars"] - kv[1]["condensed_chars"],
            reverse=True,
        ):
            saved = s["original_chars"] - s["condensed_chars"]
            pct = 100.0 * saved / s["original_chars"] if s["original_chars"] else 0.0
            table.add_row(
                name,
                str(s["count"]),
                f"{s['original_chars']:,}",
                f"{s['condensed_chars']:,}",
                f"{saved:,} ({pct:.0f}%)",
            )

        ctx.console.print(table)
        ctx.console.print(
            Panel(
                f"[bold]Total saved:[/] {stats['chars_saved']:,} chars "
                f"(~{stats['est_tokens_saved']:,} tokens, {stats['savings_pct']:.0f}%)\n"
                f"[dim]Token estimate uses chars/4 — percentages are reliable, "
                f"absolute numbers approximate.[/]",
                title="Condensation Gain",
                border_style="green",
            )
        )


@register_command
class MetricsCommand(BaseSlashCommand):
    """Show streaming performance metrics and provider stats."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="metrics",
            description="Show streaming performance metrics and provider stats",
            usage="/metrics [summary|history|export] [--json|--csv]",
            aliases=["perf", "performance"],
            category="metrics",
        )

    def execute(self, ctx: CommandContext) -> None:
        subcommand = self._get_arg(ctx, 0, "summary").lower()
        export_format = None

        if self._has_flag(ctx, "--json"):
            export_format = "json"
        elif self._has_flag(ctx, "--csv"):
            export_format = "csv"

        try:
            from victor.observability.streaming_metrics import get_metrics_collector

            collector = get_metrics_collector()

            if subcommand == "export":
                if export_format == "json":
                    import json

                    data = collector.export_json()
                    ctx.console.print(json.dumps(data, indent=2))
                elif export_format == "csv":
                    csv_data = collector.export_csv()
                    ctx.console.print(csv_data)
                else:
                    ctx.console.print("[dim]Specify format: --json or --csv[/]")
                return

            if subcommand == "history":
                history = collector.get_history(limit=20)
                if not history:
                    ctx.console.print("[dim]No metrics history yet[/]")
                    return

                table = Table(title="Recent Requests")
                table.add_column("Time", style="dim")
                table.add_column("Provider", style="cyan")
                table.add_column("Tokens", justify="right")
                table.add_column("Latency", justify="right")
                table.add_column("Status", style="green")

                for entry in history:
                    table.add_row(
                        entry.get("timestamp", "")[:19],
                        entry.get("provider", "?"),
                        str(entry.get("tokens", 0)),
                        f"{entry.get('latency_ms', 0):.0f}ms",
                        entry.get("status", "ok"),
                    )

                ctx.console.print(table)
                return

            # Default: summary
            summary = collector.get_summary()
            content = (
                f"[bold]Streaming Metrics[/]\n\n"
                f"[bold]Total Requests:[/] {summary.get('total_requests', 0)}\n"
                f"[bold]Success Rate:[/] {summary.get('success_rate', 0):.1%}\n"
                f"[bold]Avg Latency:[/] {summary.get('avg_latency_ms', 0):.0f}ms\n"
                f"[bold]Total Tokens:[/] {summary.get('total_tokens', 0):,}\n"
            )

            if summary.get("by_provider"):
                content += "\n[bold]By Provider:[/]\n"
                for provider, stats in summary["by_provider"].items():
                    content += f"  {provider}: {stats.get('requests', 0)} requests, {stats.get('avg_latency', 0):.0f}ms avg\n"

            ctx.console.print(Panel(content, title="Performance Metrics", border_style="magenta"))

        except ImportError:
            ctx.console.print("[yellow]Metrics collector not available[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error fetching metrics:[/] {e}")


@register_command
class SerializationCommand(BaseSlashCommand):
    """Show token-optimized serialization statistics and savings."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="serialization",
            description="Show token-optimized serialization statistics and savings",
            usage="/serialization [summary|tools|formats|clear]",
            aliases=["serialize", "ser"],
            category="metrics",
        )

    def execute(self, ctx: CommandContext) -> None:
        subcommand = self._get_arg(ctx, 0, "summary").lower()

        try:
            from victor.processing.serialization.adaptive import get_adaptive_serializer

            serializer = get_adaptive_serializer()
            stats = serializer.get_stats()

            if subcommand == "clear":
                serializer.reset_stats()
                ctx.console.print("[green]Serialization stats cleared[/]")
                return

            if subcommand == "tools":
                tool_stats = stats.get("by_tool", {})
                if not tool_stats:
                    ctx.console.print("[dim]No tool-level stats yet[/]")
                    return

                table = Table(title="Serialization by Tool")
                table.add_column("Tool", style="cyan")
                table.add_column("Format", style="green")
                table.add_column("Uses", justify="right")
                table.add_column("Savings", justify="right", style="yellow")

                for tool, ts in sorted(tool_stats.items()):
                    table.add_row(
                        tool,
                        ts.get("preferred_format", "?"),
                        str(ts.get("usage_count", 0)),
                        f"{ts.get('avg_savings_percent', 0):.1f}%",
                    )

                ctx.console.print(table)
                return

            if subcommand == "formats":
                format_stats = stats.get("by_format", [])
                if not format_stats:
                    ctx.console.print("[dim]No format stats yet[/]")
                    return

                table = Table(title="Serialization by Format")
                table.add_column("Format", style="cyan")
                table.add_column("Uses", justify="right")
                table.add_column("Avg Savings", justify="right", style="yellow")

                for fs in format_stats:
                    table.add_row(
                        fs.get("format", "?"),
                        str(fs.get("usage_count", 0)),
                        f"{fs.get('avg_savings_percent', 0):.1f}%",
                    )

                ctx.console.print(table)
                return

            # Default: summary
            content = (
                f"[bold]Serialization Statistics[/]\n\n"
                f"[bold]Total Serializations:[/] {stats.get('total_serializations', 0)}\n"
                f"[bold]Total Tokens Saved:[/] {stats.get('total_tokens_saved', 0):,}\n"
                f"[bold]Avg Savings:[/] {stats.get('avg_savings_percent', 0):.1f}%\n"
            )

            format_stats = stats.get("by_format", [])
            if format_stats:
                content += "\n[bold]Top Formats:[/]\n"
                for stat in format_stats[:4]:
                    fmt = stat.get("format", "?")
                    usage = stat.get("usage_count", 0)
                    pct = stat.get("avg_savings_percent", 0)
                    content += f"  [cyan]{fmt}[/]: {usage} uses, {pct:.1f}% avg savings\n"

            content += "\n[dim]Use /serialization tools for per-tool breakdown[/]"
            content += "\n[dim]Use /serialization formats for format comparison[/]"

            ctx.console.print(
                Panel(content, title="Serialization Statistics", border_style="green")
            )

        except ImportError:
            ctx.console.print("[yellow]Adaptive serializer not available[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error fetching stats:[/] {e}")


@register_command
class LearningCommand(BaseSlashCommand):
    """Show RL/Q-learning stats from runtime intelligence and the model selector.

    This is the unified command that combines stats from:
    1. AdaptiveModeController (mode transitions, exploration rate)
    2. ModelSelectorLearner (provider rankings, Q-values)
    """

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="learning",
            description="Show Q-learning stats, adjust exploration, or reset session",
            usage="/learning [stats|explore <rate>|reset|recommend|strategy <name>]",
            aliases=["qlearn", "rl"],
            category="metrics",
        )

    def execute(self, ctx: CommandContext) -> None:
        subcommand = self._get_arg(ctx, 0, "stats").lower()

        # Handle explore subcommand
        if subcommand == "explore":
            rate_str = self._get_arg(ctx, 1)
            if not rate_str:
                ctx.console.print("[yellow]Usage: /learning explore <rate>[/]")
                ctx.console.print("[dim]Example: /learning explore 0.2[/]")
                return

            try:
                rate = float(rate_str)
                if not 0.0 <= rate <= 1.0:
                    ctx.console.print("[red]Rate must be between 0.0 and 1.0[/]")
                    return

                # Set on mode controller if available
                if ctx.agent:
                    mode_controller = getattr(ctx.agent, "mode_controller", None)
                    if mode_controller:
                        mode_controller.adjust_exploration_rate(rate)
                        ctx.console.print(
                            f"[green]Mode controller exploration rate set to:[/] {rate:.2f}"
                        )

                # Set on model selector learner
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")
                if learner and hasattr(learner, "set_exploration_rate"):
                    learner.set_exploration_rate(rate)
                    ctx.console.print(
                        f"[green]Model selector exploration rate set to:[/] {rate:.2f}"
                    )

            except ValueError:
                ctx.console.print(f"[red]Invalid rate:[/] {rate_str}")
            return

        # Handle reset subcommand
        if subcommand == "reset":
            # Reset runtime-intelligence integration
            if ctx.agent:
                integration = _get_runtime_intelligence_integration(ctx.agent)
                if integration:
                    if hasattr(integration, "reset_session"):
                        integration.reset_session()
                        ctx.console.print("[green]Runtime-intelligence session reset[/]")
                    else:
                        # Fallback to _pipeline attribute
                        pipeline = getattr(integration, "_pipeline", None)
                        if pipeline:
                            pipeline.reset_session()
                            ctx.console.print("[green]Runtime-intelligence session reset[/]")

            # Reset model selector
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner and hasattr(learner, "reset"):
                learner.reset()
                ctx.console.print("[green]Model selector Q-values reset[/]")
            return

        # Handle recommend subcommand
        if subcommand == "recommend":
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if not learner:
                ctx.console.print("[yellow]Model selector learner not available[/]")
                return

            task_type = self._get_arg(ctx, 1, "coding")
            rec = learner.recommend(ctx.agent.provider_name if ctx.agent else "unknown", task_type)

            if rec:
                ctx.console.print(
                    Panel(
                        f"[bold]Recommended Provider:[/] [cyan]{rec.value}[/]\n"
                        f"[bold]Confidence:[/] {rec.confidence:.2f}\n"
                        f"[bold]Reason:[/] {rec.reason}",
                        title="RL Recommendation",
                        border_style="magenta",
                    )
                )
            else:
                ctx.console.print("[dim]No recommendation available (need more data)[/]")
            return

        # Handle strategy subcommand
        if subcommand == "strategy":
            from victor.framework.rl.coordinator import get_rl_coordinator
            from victor.framework.rl.learners.model_selector import SelectionStrategy

            strategy_name = self._get_arg(ctx, 1)
            if not strategy_name:
                ctx.console.print("[yellow]Usage: /learning strategy <name>[/]")
                ctx.console.print("[dim]Available: epsilon_greedy, ucb, exploit[/]")
                return

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if not learner:
                ctx.console.print("[yellow]Model selector learner not available[/]")
                return

            try:
                strategy = SelectionStrategy(strategy_name.lower())
                learner.set_strategy(strategy)
                ctx.console.print(f"[green]Selection strategy set to:[/] {strategy.value}")
            except ValueError:
                ctx.console.print(f"[red]Invalid strategy:[/] {strategy_name}")
                ctx.console.print("[dim]Available: epsilon_greedy, ucb, exploit[/]")
            return

        # Default: show stats
        content = "[bold]Reinforcement Learning Statistics[/]\n\n"

        # 1. Runtime-intelligence stats (mode controller)
        if ctx.agent:
            integration = _get_runtime_intelligence_integration(ctx.agent)
            if integration:
                if hasattr(integration, "get_stats"):
                    stats = integration.get_stats()
                else:
                    # Fallback to _pipeline attribute
                    pipeline = getattr(integration, "_pipeline", None)
                    if pipeline and hasattr(pipeline, "get_stats"):
                        stats = pipeline.get_stats()
                    else:
                        stats = None

                if stats:
                    content += "[bold cyan]Runtime-Intelligence:[/]\n"
                    content += f"  Session Duration: {stats.session_duration:.1f}s\n"
                    content += f"  Total Requests: {stats.total_requests}\n"
                    content += f"  Enhanced Requests: {stats.enhanced_requests}\n"
                    content += f"  Quality Validations: {stats.quality_validations}\n"

                    if stats.avg_quality_score > 0:
                        content += f"  Avg Quality Score: {stats.avg_quality_score:.2f}\n"
                        content += f"  Avg Grounding Score: {stats.avg_grounding_score:.2f}\n"

                    # Get mode controller via capability registry or attribute
                    if hasattr(integration, "mode_controller"):
                        mode_controller = integration.mode_controller
                    elif hasattr(integration, "_pipeline"):
                        pipeline = getattr(integration, "_pipeline", None)
                        mode_controller = (
                            getattr(pipeline, "_mode_controller", None) if pipeline else None
                        )
                    else:
                        mode_controller = None

                    if mode_controller:
                        session_stats = mode_controller.get_session_stats()
                        content += "\n[bold cyan]Mode Learning:[/]\n"
                        content += f"  Profile: {session_stats.get('profile_name', 'unknown')}\n"
                        content += f"  Total Reward: {session_stats.get('total_reward', 0):.2f}\n"
                        content += (
                            f"  Mode Transitions: {session_stats.get('mode_transitions', 0)}\n"
                        )
                        content += (
                            f"  Exploration Rate: {session_stats.get('exploration_rate', 0):.2f}\n"
                        )

                        modes_visited = session_stats.get("modes_visited", [])
                        if modes_visited:
                            content += f"  Modes Visited: {' -> '.join(modes_visited)}\n"

                    content += "\n"

        # 2. Model Selector learner stats
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")

            if learner:
                rankings = learner.get_provider_rankings()
                if rankings:
                    content += "[bold cyan]Model Selector (Provider Rankings):[/]\n"
                    for rank in rankings[:5]:
                        current = (
                            " (current)"
                            if ctx.agent
                            and rank["provider"].lower() == ctx.agent.provider_name.lower()
                            else ""
                        )
                        content += f"  {rank['provider']}: Q={rank['q_value']:.2f} (n={rank['sample_count']}){current}\n"

                    exploration = (
                        learner.get_exploration_rate()
                        if hasattr(learner, "get_exploration_rate")
                        else getattr(learner, "_exploration_rate", 0.1)
                    )
                    strategy = (
                        learner.get_strategy()
                        if hasattr(learner, "get_strategy")
                        else getattr(learner, "_strategy", None)
                    )
                    content += f"\n  Strategy: {strategy.value if strategy else 'epsilon_greedy'}\n"
                    content += f"  Exploration Rate: {exploration:.2f}\n"
        except Exception as e:
            logger.debug(f"Could not get model selector stats: {e}")

        content += "\n[dim]Use /learning explore <rate> to adjust exploration[/]"
        content += "\n[dim]Use /learning recommend to get provider recommendation[/]"
        content += "\n[dim]Use /learning reset to reset Q-values[/]"

        ctx.console.print(Panel(content, title="Q-Learning Stats", border_style="magenta"))


@register_command
class MLStatsCommand(BaseSlashCommand):
    """Show ML-friendly aggregated session statistics for RL training."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="mlstats",
            description="Show ML-friendly aggregated session statistics for RL training",
            usage="/mlstats [providers|families|sizes|export]",
            aliases=["ml", "analytics"],
            category="metrics",
        )

    def execute(self, ctx: CommandContext) -> None:
        subcommand = self._get_arg(ctx, 0, "summary").lower()

        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()

            if subcommand == "export":
                # Export all RL data as JSON
                import json

                export_data = coordinator.export_all_learner_data()
                ctx.console.print(json.dumps(export_data, indent=2))
                return

            if subcommand == "providers":
                learner = coordinator.get_learner("model_selector")
                if not learner:
                    ctx.console.print("[yellow]Model selector not available[/]")
                    return

                rankings = learner.get_provider_rankings()
                if not rankings:
                    ctx.console.print("[dim]No provider data yet[/]")
                    return

                table = Table(title="Provider Performance")
                table.add_column("Provider", style="cyan")
                table.add_column("Q-Value", justify="right", style="green")
                table.add_column("Samples", justify="right")
                table.add_column("Avg Reward", justify="right", style="yellow")

                for r in rankings:
                    table.add_row(
                        r["provider"],
                        f"{r['q_value']:.3f}",
                        str(r["sample_count"]),
                        f"{r.get('avg_reward', 0):.3f}",
                    )

                ctx.console.print(table)
                return

            if subcommand in ("families", "sizes"):
                ctx.console.print("[dim]Family/size aggregation coming soon[/]")
                return

            # Default: summary
            content = "[bold]ML/RL Training Statistics[/]\n\n"

            # Get all learner stats
            learners = coordinator.list_learners()
            content += f"[bold]Active Learners:[/] {len(learners)}\n"

            for name in learners[:5]:
                learner = coordinator.get_learner(name)
                if learner and hasattr(learner, "get_sample_count"):
                    count = learner.get_sample_count()
                    content += f"  {name}: {count} samples\n"

            # Get coordinator-level stats
            coord_stats = coordinator.get_stats()
            content += "\n[bold]Coordinator Stats:[/]\n"
            content += f"  Total Outcomes: {coord_stats.get('total_outcomes', 0)}\n"
            content += f"  Sessions: {coord_stats.get('sessions', 0)}\n"

            ctx.console.print(Panel(content, title="ML Statistics", border_style="blue"))

        except ImportError:
            ctx.console.print("[yellow]RL coordinator not available[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error fetching ML stats:[/] {e}")
