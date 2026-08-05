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

"""Delegate follow-up slash command.

This command is intentionally a thin UI surface over workflow execution. The
canonical resume/review/merge behavior remains owned by workflow TeamStep and
the delegate follow-up coordinator.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from victor.ui.commands.chat import run_workflow_mode
from victor.ui.delegate_follow_up import (
    DelegateFollowUpContractError,
    build_delegate_follow_up_suggestions,
    load_delegate_follow_up_contract_file,
)
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command


@register_command
class DelegateFollowUpCommand(BaseSlashCommand):
    """Run a delegate follow-up contract through a workflow TeamStep."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="delegate-follow-up",
            description="Resume delegate review/retry/merge work from a follow-up contract",
            usage=(
                "/delegate-follow-up <workflow.yaml> <contract.json> [step_id] | "
                "/delegate-follow-up list <workflow.yaml> <contract.json>"
            ),
            aliases=["dfu"],
            category="workflow",
            requires_agent=False,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if ctx.args and ctx.args[0].lower() in {"list", "steps"}:
            self._handle_list(ctx, ctx.args[1:])
            return

        if len(ctx.args) < 2:
            self._show_help(ctx)
            return

        workflow_path = ctx.args[0]
        contract_path = ctx.args[1]
        step_id = ctx.args[2] if len(ctx.args) > 2 else None

        ctx.console.print("[dim]Routing delegate follow-up through workflow TeamStep runtime...[/]")
        try:
            await run_workflow_mode(
                workflow_path=workflow_path,
                delegate_follow_up_contract=contract_path,
                delegate_next_step_id=step_id,
            )
        except typer.Exit:
            return

    def _show_help(self, ctx: CommandContext) -> None:
        ctx.console.print(
            "[bold]Delegate Follow-Up[/]\n\n"
            "Resume delegate review/retry/merge work through a workflow TeamStep.\n\n"
            "[dim]Usage:[/] /delegate-follow-up <workflow.yaml> <contract.json> [step_id]\n"
            "[dim]List:[/] /delegate-follow-up list <workflow.yaml> <contract.json>\n"
            "[dim]Alias:[/] /dfu\n"
            "[dim]Example:[/] /dfu workflows/delegate-resume.yaml "
            "delegate-follow-up.json resume_delegate_retry"
        )

    def _handle_list(self, ctx: CommandContext, args: list[str]) -> None:
        if len(args) < 2:
            self._show_help(ctx)
            return

        workflow_path = args[0]
        contract_path = args[1]
        try:
            contract = load_delegate_follow_up_contract_file(Path(contract_path))
            suggestions = build_delegate_follow_up_suggestions(
                workflow_path=workflow_path,
                contract_path=contract_path,
                contract=contract,
            )
        except FileNotFoundError:
            ctx.console.print(f"[red]Delegate follow-up contract not found:[/] {contract_path}")
            return
        except DelegateFollowUpContractError as e:
            ctx.console.print(f"[red]Invalid delegate follow-up contract:[/] {e}")
            return

        table = Table(title="Delegate Follow-Up Steps")
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Summary")
        table.add_column("Command", style="dim")
        for suggestion in suggestions:
            table.add_row(
                str(suggestion["step_id"]),
                str(suggestion["description"]),
                str(suggestion["command"]),
            )
        ctx.console.print(table)
        ctx.console.print("[dim]Selectable commands:[/]")
        for suggestion in suggestions:
            ctx.console.print(str(suggestion["command"]))
