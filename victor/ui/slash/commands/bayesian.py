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

"""Bayesian orchestration monitoring slash command."""

from __future__ import annotations

from rich.panel import Panel

from victor.framework.rl.monitoring.reporting import (
    DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    get_bayesian_monitoring_service,
    parse_agent_ids,
)
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command


@register_command
class BayesianCommand(BaseSlashCommand):
    """Inspect Bayesian orchestration metrics during a chat session."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="bayesian",
            description="Inspect Bayesian orchestration metrics and historical learning",
            usage=(
                "/bayesian [summary|reliability|consensus|voi|correlations|belief] "
                "[args] [--days N]"
            ),
            aliases=["bayes"],
            category="metrics",
        )

    def execute(self, ctx: CommandContext) -> None:
        subcommand = self._get_arg(ctx, 0, "summary").lower()
        service = get_bayesian_monitoring_service()

        if subcommand == "summary":
            days = self._resolve_days(ctx, self._positionals_after_subcommand(ctx))
            report = service.render_summary(days)
            self._print_report(ctx, report, "Bayesian Summary")
            output_path = self._get_flag_value(ctx, "--output")
            if output_path:
                service.export_summary_json(output_path, days)
                ctx.console.print(f"[green]Summary exported to {output_path}[/]")
            return

        if subcommand == "reliability":
            positionals = self._positionals_after_subcommand(ctx)
            raw_agents = self._get_flag_value(ctx, "--agents")
            day_index = 0
            if raw_agents is None and positionals and not self._looks_like_int(positionals[0]):
                raw_agents = positionals[0]
                day_index = 1
            days = self._resolve_days(ctx, positionals, index=day_index)
            report = service.render_reliability(parse_agent_ids(raw_agents), days)
            self._print_report(ctx, report, "Bayesian Reliability")
            export_path = self._get_flag_value(ctx, "--export")
            if export_path:
                service.export_reliability_csv(export_path, parse_agent_ids(raw_agents), days)
                ctx.console.print(f"[green]Reliability trends exported to {export_path}[/]")
            return

        if subcommand == "consensus":
            days = self._resolve_days(ctx, self._positionals_after_subcommand(ctx))
            self._print_report(ctx, service.render_consensus(days), "Bayesian Consensus")
            return

        if subcommand == "voi":
            positionals = self._positionals_after_subcommand(ctx)
            agent_id = self._get_flag_value(ctx, "--agent")
            day_index = 0
            if agent_id is None and positionals and not self._looks_like_int(positionals[0]):
                agent_id = positionals[0]
                day_index = 1
            days = self._resolve_days(ctx, positionals, index=day_index)
            self._print_report(ctx, service.render_voi(agent_id, days), "Bayesian VoI")
            return

        if subcommand == "correlations":
            positionals = self._positionals_after_subcommand(ctx)
            raw_agents = self._get_flag_value(ctx, "--agents")
            day_index = 0
            if raw_agents is None and positionals:
                raw_agents = positionals[0]
                day_index = 1
            if not raw_agents:
                ctx.console.print(
                    "[yellow]Usage: /bayesian correlations <agent_a,agent_b,...> [days][/]"
                )
                return
            days = self._resolve_days(ctx, positionals, index=day_index)
            self._print_report(
                ctx,
                service.render_correlations(parse_agent_ids(raw_agents) or [], days),
                "Bayesian Correlations",
            )
            return

        if subcommand == "belief":
            positionals = self._positionals_after_subcommand(ctx)
            belief_id = positionals[0] if positionals else None
            if not belief_id:
                ctx.console.print("[yellow]Usage: /bayesian belief <belief_id>[/]")
                return
            self._print_report(ctx, service.render_belief(belief_id), "Bayesian Belief")
            export_path = self._get_flag_value(ctx, "--export")
            if export_path:
                service.export_belief_csv(belief_id, export_path)
                ctx.console.print(f"[green]Belief evolution exported to {export_path}[/]")
            return

        ctx.console.print(
            "[yellow]Usage: /bayesian [summary|reliability|consensus|voi|correlations|belief][/]"
        )

    @staticmethod
    def _print_report(ctx: CommandContext, report: str, title: str) -> None:
        ctx.console.print(Panel(report, title=title, border_style="cyan"))

    def _resolve_days(
        self,
        ctx: CommandContext,
        positionals: list[str],
        index: int = 0,
        default: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    ) -> int:
        flag_value = self._get_flag_value(ctx, "--days")
        if flag_value is not None:
            return self._safe_int(flag_value, default)
        if index < len(positionals):
            return self._safe_int(positionals[index], default)
        return default

    @staticmethod
    def _safe_int(raw_value: str, default: int) -> int:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _looks_like_int(raw_value: str) -> bool:
        try:
            int(raw_value)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _positionals_after_subcommand(ctx: CommandContext) -> list[str]:
        positionals: list[str] = []
        skip_next = False
        flags_with_values = {"--days", "--agents", "--agent", "--output", "--export"}
        for arg in ctx.args[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg in flags_with_values:
                skip_next = True
                continue
            if arg.startswith("--"):
                continue
            positionals.append(arg)
        return positionals
