# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt optimization slash command (GEPA v2).

Commands:
- /prompt-optimize: Run evolution cycle on all evolvable sections
- /prompt-optimize ASI: Evolve a specific section
- /prompt-optimize --status: Show candidates and scores
- /prompt-optimize --pareto: Show Pareto frontier (v2)
- /prompt-optimize --tier [economic|balanced|performance]: Show/set tier

Example:
    /prompt-optimize                    # Evolve all sections
    /prompt-optimize ASI                # Evolve ASI_TOOL_EFFECTIVENESS_GUIDANCE
    /prompt-optimize --status           # Show current candidates
    /prompt-optimize --pareto           # Show Pareto frontier
    /prompt-optimize --tier performance # Force performance tier
"""

from __future__ import annotations

import difflib
import logging

from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class PromptOptimizeCommand(BaseSlashCommand):
    """Run GEPA-inspired prompt evolution cycle."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="prompt-optimize",
            description="Evolve system prompt sections using execution trace analysis (GEPA)",
            usage="/prompt-optimize [section|--status|--show SECTION|--diff SECTION]",
            aliases=["optimize-prompt", "evolve-prompt"],
            category="advanced",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        args = ctx.args if ctx.args else []
        show_status = "--status" in args or "status" in args
        show_section = self._option_value(args, "--show")
        diff_section = self._option_value(args, "--diff")
        resolved_provider, resolved_model = self._resolve_prompt_identity(ctx)

        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")

            if learner is None:
                ctx.console.print(
                    "[yellow]Prompt optimizer learner not available. "
                    "Ensure RL coordinator is initialized.[/]"
                )
                return

            if show_status:
                self._show_status(ctx, learner, coordinator.db_path)
                return

            if show_section:
                self._show_candidate(ctx, learner, show_section, coordinator.db_path, args)
                return

            if diff_section:
                self._diff_candidates(ctx, learner, diff_section, coordinator.db_path, args)
                return

            # Determine which sections to evolve
            from victor.framework.rl.learners.prompt_optimizer import (
                PromptOptimizerLearner,
            )

            sections = (
                learner.get_evolvable_sections()
                if hasattr(learner, "get_evolvable_sections")
                else PromptOptimizerLearner.EVOLVABLE_SECTIONS
            )
            if args and args[0] not in ("--status", "status"):
                # User specified a section name or abbreviation
                target = args[0].upper()
                matched = [s for s in sections if target in s]
                if matched:
                    sections = matched
                else:
                    ctx.console.print(
                        f"[red]Unknown section '{args[0]}'.[/] " f"Available: {', '.join(sections)}"
                    )
                    return

            section_text = self._get_section_text()

            # Push active session model into tier manager before evolving.
            # resolved_provider/model come from ctx.agent (the live session LLM),
            # not from settings defaults — this is what we want GEPA to call.
            # Also capture base_url so provider-specific endpoints (e.g. ZAI coding plan)
            # are preserved when GEPA creates its own provider instance.
            if resolved_provider and resolved_provider != "default":
                if hasattr(learner, "set_main_model_spec"):
                    active_base_url = getattr(ctx.agent, "provider_base_url", None) or ""
                    learner.set_main_model_spec(
                        resolved_provider, resolved_model, base_url=active_base_url
                    )

            # Run evolution with a per-section spinner showing reflect/mutate phases
            results = Table(title="GEPA Prompt Evolution Results")
            results.add_column("Section", style="cyan")
            results.add_column("Provider", style="magenta")
            results.add_column("Ordinal", style="green")
            results.add_column("Status", style="bold")
            results.add_column("Change", style="dim")
            results.add_column("Lineage", style="dim")

            with Status("", console=ctx.console, spinner="dots") as evolve_status:
                for section in sections:
                    current = section_text.get(section)
                    if current is None:
                        results.add_row(section, "-", "-", "[yellow]Not available[/]", "-", "-")
                        continue

                    short = (
                        section.replace("ASI_", "")
                        .replace("_GUIDANCE", "")
                        .replace("_RULES", "")
                        .replace("_EXAMPLES", "")
                    )

                    def _on_phase(sec: str, phase: str, strat: str = "", _s: str = short) -> None:
                        phase_label = "reflecting…" if phase == "reflect" else "mutating…"
                        evolve_status.update(f"[cyan]{_s}[/] [dim]{phase_label}[/]")

                    evolve_status.update(f"[cyan]{short}[/] [dim]collecting traces…[/]")
                    candidate = learner.evolve(
                        section, current, provider=resolved_provider, on_phase=_on_phase
                    )
                    if candidate:
                        added = len(candidate.text) - len(current)
                        change = f"+{added} chars" if added > 0 else f"{added} chars"
                        results.add_row(
                            section,
                            candidate.provider,
                            str(candidate.generation),
                            "[green]Evolved[/]",
                            change,
                            f"{candidate.parent_hash[:8]} -> {candidate.text_hash[:8]}",
                        )
                    else:
                        results.add_row(section, "-", "-", "[dim]No change[/]", "-", "-")

            ctx.console.print(results)
            ctx.console.print(
                Panel(
                    f"Persisted in: [bold]{coordinator.db_path}[/]\n"
                    f"Prompt identity: [bold]{resolved_provider or 'default'}[/]"
                    f" / [bold]{resolved_model or 'default'}[/]\n"
                    "Ordinal is candidate creation order per (section, provider). Shared parent "
                    "hashes indicate sibling candidates from the same baseline, not a strict "
                    "linear prompt chain.",
                    title="Prompt Evolution Notes",
                    border_style="blue",
                )
            )

        except Exception as e:
            ctx.console.print(f"[red]Prompt optimization failed:[/] {e}")
            logger.exception("Prompt optimization error")

    @staticmethod
    def _option_value(args: list[str], flag: str) -> str | None:
        """Return one flag value from slash-command args."""
        if flag not in args:
            return None
        index = args.index(flag)
        if index + 1 >= len(args):
            return None
        return args[index + 1]

    @staticmethod
    def _resolve_prompt_identity(ctx: CommandContext) -> tuple[str, str]:
        """Resolve provider/model from the active session LLM when available."""
        agent = getattr(ctx, "agent", None)
        if agent is not None:
            provider_name = (
                getattr(agent, "provider_name", None)
                or getattr(getattr(agent, "provider", None), "name", None)
                or getattr(getattr(agent, "_provider", None), "name", None)
            )
            model_name = getattr(agent, "model", None)
            if provider_name or model_name:
                return str(provider_name or "default"), str(model_name or "")

        settings = getattr(ctx, "settings", None)
        provider_settings = getattr(settings, "provider", None)
        default_provider = getattr(provider_settings, "default_provider", None)
        default_model = getattr(provider_settings, "default_model", None)
        return str(default_provider or "default"), str(default_model or "")

    @staticmethod
    def _get_section_text() -> dict[str, str]:
        """Return the static baseline text for evolvable prompt sections."""
        from victor.agent.prompt_section_registry import get_section_registry

        registry = get_section_registry()
        return {
            section.name: section.default_text
            for section in registry.get_all()
            if section.evolvable
        }

    def _resolve_candidate_or_baseline(
        self,
        learner,
        section: str,
        provider: str,
        selector: str,
    ) -> tuple[str, str] | None:
        """Resolve a text selector into (label, text) for show/diff."""
        normalized = selector.strip()
        if normalized.lower() == "baseline":
            baseline_text = self._get_section_text().get(section)
            if baseline_text is None:
                return None
            return (f"{section}:baseline", baseline_text)

        candidate = learner.resolve_candidate(
            section_name=section,
            provider=provider,
            selector=normalized,
        )
        if candidate is None:
            return None
        return (f"{section}:{provider}:{candidate.generation}", candidate.text)

    def _show_candidate(self, ctx: CommandContext, learner, section: str, db_path, args) -> None:
        """Display the full text for one stored prompt candidate."""
        resolved_provider, _ = self._resolve_prompt_identity(ctx)
        provider = self._option_value(args, "--provider") or resolved_provider
        selector = self._option_value(args, "--hash") or self._option_value(args, "--ordinal")
        selector = selector or "baseline"
        if selector.strip().lower() == "baseline":
            resolved = self._resolve_candidate_or_baseline(learner, section, provider, selector)
            if resolved is None:
                ctx.console.print(
                    f"[red]No prompt candidate found for {section} selector '{selector}' (provider={provider}).[/]"
                )
                return
            label, text = resolved
            subtitle = f"Database: {db_path}"
        else:
            candidate = learner.resolve_candidate(
                section_name=section,
                provider=provider,
                selector=selector,
            )
            if candidate is None:
                ctx.console.print(
                    f"[red]No prompt candidate found for {section} selector '{selector}' (provider={provider}).[/]"
                )
                return
            label = f"{section}:{provider}:{candidate.generation}" f" [{candidate.text_hash[:8]}]"
            text = candidate.text
            subtitle = f"Hash: {candidate.text_hash} | Parent: {candidate.parent_hash} | Database: {db_path}"
        ctx.console.print(
            Panel(
                text,
                title=f"Prompt Candidate: {label}",
                subtitle=subtitle,
                border_style="blue",
            )
        )

    def _diff_candidates(self, ctx: CommandContext, learner, section: str, db_path, args) -> None:
        """Display a unified diff between two prompt candidates or baseline."""
        resolved_provider, _ = self._resolve_prompt_identity(ctx)
        provider = self._option_value(args, "--provider") or resolved_provider
        from_selector = self._option_value(args, "--from") or "baseline"
        to_selector = self._option_value(args, "--to")
        if not to_selector:
            ctx.console.print("[red]Missing --to selector for /prompt-optimize --diff.[/]")
            return

        left = self._resolve_candidate_or_baseline(learner, section, provider, from_selector)
        right = self._resolve_candidate_or_baseline(learner, section, provider, to_selector)
        if left is None or right is None:
            ctx.console.print(
                f"[red]Unable to resolve diff selectors for {section} "
                f"(from={from_selector}, to={to_selector}, provider={provider}).[/]"
            )
            return

        left_label, left_text = left
        right_label, right_text = right
        diff = "\n".join(
            difflib.unified_diff(
                left_text.splitlines(),
                right_text.splitlines(),
                fromfile=left_label,
                tofile=right_label,
                lineterm="",
            )
        )
        ctx.console.print(
            Panel(
                diff or "[no textual diff]",
                title=f"Prompt Diff: {section}",
                subtitle=f"Database: {db_path}",
                border_style="blue",
            )
        )

    def _show_status(self, ctx: CommandContext, learner, db_path) -> None:
        """Display current prompt candidates and their scores."""
        metrics = learner.export_metrics()

        if metrics["total_candidates"] == 0:
            ctx.console.print(
                Panel(
                    "No evolved prompt candidates yet.\n\n"
                    "Run [bold]/prompt-optimize[/] to start evolution cycle.\n"
                    "Requires execution trace data in ~/.victor/logs/usage.jsonl\n"
                    f"Stored in: {db_path}",
                    title="Prompt Optimizer Status",
                    border_style="blue",
                )
            )
            return

        table = Table(title="Prompt Candidates")
        table.add_column("Section", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Ordinal", style="green")
        table.add_column("Parent", style="dim")
        table.add_column("Hash", style="dim")
        table.add_column("Live", style="bold")
        table.add_column("Bench", style="bold")
        table.add_column("α/β", style="dim")
        table.add_column("Mean", style="bold")
        table.add_column("Samples", style="dim")
        table.add_column("Strategy", style="dim")
        table.add_column("Preview", style="dim", max_width=50)

        for row in learner.export_candidate_rows():
            table.add_row(
                row["section"],
                row["provider"],
                str(row["ordinal"]),
                row["parent_hash"][:8],
                row["text_hash"][:8],
                "yes" if row["active"] else "no",
                "yes" if row["benchmark_passed"] else "no",
                f"{row['alpha']:.1f}/{row['beta']:.1f}",
                f"{row['mean']:.2f}",
                str(row["sample_count"]),
                row["strategy"],
                row["preview"],
            )

        ctx.console.print(table)

        # GEPA v2: Pareto info
        pareto_info = metrics.get("pareto", {})
        if pareto_info:
            pareto_table = Table(title="Pareto Frontier (GEPA v2)")
            pareto_table.add_column("Section", style="cyan")
            pareto_table.add_column("Hash", style="dim")
            pareto_table.add_column("Gen", style="green")
            pareto_table.add_column("Coverage", style="bold")
            pareto_table.add_column("Chars", style="dim")

            for section, info in pareto_info.items():
                for c in info.get("candidates", []):
                    pareto_table.add_row(
                        section,
                        c["hash"][:8],
                        str(c["gen"]),
                        str(c["coverage"]),
                        str(c["chars"]),
                    )
            ctx.console.print(pareto_table)

        ctx.console.print(
            f"\n[dim]Total: {metrics['total_candidates']} candidates "
            f"across {len(metrics['sections'])} section/provider keys"
            f"{' (Pareto v2)' if metrics.get('use_pareto') else ''}[/]"
        )
        ctx.console.print(f"[dim]Database: {db_path}[/]")
        ctx.console.print(
            Panel(
                "Ordinals are creation order, not guaranteed linear ancestry. "
                "Compare parent/hash pairs to see sibling candidates evolved from the same baseline.",
                title="Lineage Semantics",
                border_style="blue",
            )
        )
