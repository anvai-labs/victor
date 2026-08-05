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

"""Model and provider slash commands: model, profile, provider."""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


def _parse_provider_model(target: str) -> tuple[str | None, str]:
    """Parse provider:model syntax.

    Returns:
        Tuple of (provider, model). Provider is None if not specified.
    """
    if ":" in target:
        parts = target.split(":", 1)
        return parts[0], parts[1]
    return None, target


def _provider_info_value(info: dict, preferred: str, legacy: str, default: str = "N/A") -> str:
    """Read provider info across old/new key names used by command surfaces."""
    value = info.get(preferred)
    if value is None:
        value = info.get(legacy, default)
    return str(value)


def _show_resume_selection(ctx: CommandContext) -> None:
    """Show session selection for resume."""
    from victor.agent.conversation.store import ConversationStore

    try:
        store = ConversationStore()
        sessions = store.list_sessions(limit=20)

        if not sessions:
            ctx.console.print("[yellow]No sessions to resume[/]")
            return

        table = Table(title="Select Session to Resume")
        table.add_column("#", style="cyan", no_wrap=True, width=4)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")
        table.add_column("Model", style="yellow")
        table.add_column("Messages", justify="right")

        for idx, session in enumerate(sessions, 1):
            title = session.title or "Untitled"
            if len(title) > 30:
                title = title[:30] + "..."
            table.add_row(
                str(idx),
                session.session_id,
                title,
                session.model or "unknown",
                str(len(session.messages)),
            )

        ctx.console.print(table)
        ctx.console.print("\n[dim]Usage: /model <model> --resume <number>[/]")

    except Exception as e:
        ctx.console.print(f"[red]Error listing sessions:[/] {e}")
        logger.exception("Error listing sessions")


def _resume_session(ctx: CommandContext, session_id: str = None, index: int = None) -> bool:
    """Resume a session.

    Returns:
        True if resume succeeded
    """
    from victor.agent.conversation.store import ConversationStore

    try:
        store = ConversationStore()

        if index is not None:
            sessions = store.list_sessions(limit=20)
            if not sessions or index > len(sessions):
                ctx.console.print(f"[red]Invalid session index:[/] {index}")
                return False
            session_id = sessions[index - 1].session_id

        session_data = store.load_session(session_id)
        if not session_data:
            ctx.console.print(f"[red]Session not found:[/] {session_id}")
            return False

        from victor.agent.message_history import MessageHistory
        from victor.agent.conversation.state_machine import ConversationStateMachine

        metadata = session_data.get("metadata", {})
        conversation_dict = session_data.get("conversation", {})

        ctx.agent.conversation = MessageHistory.from_dict(conversation_dict)
        ctx.agent.active_session_id = session_id

        conversation_state_dict = session_data.get("conversation_state")
        if conversation_state_dict:
            try:
                ctx.agent.conversation_state = ConversationStateMachine.from_dict(
                    conversation_state_dict
                )
            except Exception as e:
                logger.warning(f"Failed to restore conversation state: {e}")

        ctx.console.print(
            f"[green]\u2713[/] Resumed: {metadata.get('title', 'Untitled')} "
            f"({metadata.get('message_count', 0)} messages)\n"
        )
        return True

    except Exception as e:
        ctx.console.print(f"[red]Failed to resume session:[/] {e}")
        logger.exception("Error resuming session")
        return False


@register_command
class ModelCommand(BaseSlashCommand):
    """List available models, switch model/provider, or resume sessions.

    Usage:
        /model                                  # Show current model + provider info
        /model list                             # List available models
        /model <model_name>                     # Switch model (same provider)
        /model <provider>:<model>               # Switch provider and model
        /model <model> --resume [session_id]    # Switch and resume session
        /model --resume                         # Show sessions to resume

    Examples:
        /model gemma4:31b
        /model deepseek:deepseek-chat
        /model --resume
        /model gemma4:31b --resume 3
    """

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="model",
            description="Show/switch model, list models, or resume sessions",
            usage="/model [model|provider:model] [--resume [session_id]]",
            aliases=["models", "switch"],
            category="model",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if not ctx.args:
            self._show_current(ctx)
            return

        args = list(ctx.args)
        resume_session_id = None
        resume_index = None

        # Check for --resume flag
        if "--resume" in args:
            resume_idx = args.index("--resume")
            args.pop(resume_idx)

            # Check if session_id provided after --resume
            if resume_idx < len(args):
                session_id = args[resume_idx]
                if session_id.isdigit():
                    resume_index = int(session_id)
                else:
                    resume_session_id = session_id
                args.pop(resume_idx)

        # Handle "list" subcommand
        if args and args[0] == "list":
            await self._list_models(ctx)
            return

        # If no more args and --resume was used, show session selection
        if not args and resume_session_id is None and resume_index is None:
            _show_resume_selection(ctx)
            return

        if not args:
            # --resume with session specified but no model — just resume
            if resume_session_id or resume_index is not None:
                _resume_session(ctx, resume_session_id, resume_index)
            return

        # Parse model/provider argument
        target = args[0]
        provider, model = _parse_provider_model(target)

        # Resume session if requested
        if resume_session_id or resume_index is not None:
            if not _resume_session(ctx, resume_session_id, resume_index):
                return

        # Perform the switch
        if provider and model:
            await self._switch_provider(ctx, provider, model)
        elif model:
            await self._switch_model(ctx, model)

    def _show_current(self, ctx: CommandContext) -> None:
        """Show current model/provider information."""
        try:
            info = ctx.agent.get_current_provider_info()

            ctx.console.print(
                Panel(
                    f"[bold]Provider:[/] [cyan]{info.get('provider', 'N/A')}[/]\n"
                    f"[bold]Model:[/] [yellow]{info.get('model', 'N/A')}[/]\n"
                    f"[bold]Native Tools:[/] {info.get('native_tool_calls', 'N/A')}\n"
                    f"[bold]Thinking Mode:[/] {info.get('thinking_mode', 'N/A')}\n"
                    f"[bold]Tool Budget:[/] {info.get('tool_budget', 'N/A')}",
                    title="Current Model/Provider",
                    border_style="cyan",
                )
            )
            ctx.console.print("[dim]Switch: /model <name> | /model <provider>:<model>[/]")
            ctx.console.print("[dim]List:   /model list[/]")
            ctx.console.print("[dim]Resume: /model --resume[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error getting provider info:[/] {e}")

    async def _list_models(self, ctx: CommandContext) -> None:
        """List available models from the current provider."""
        ctx.console.print("[dim]Fetching available models...[/]")

        try:
            # Try to get models from the current provider
            info = ctx.agent.get_current_provider_info()
            provider_name = info.get("provider", "").lower()

            # Try provider-agnostic listing first
            provider = getattr(ctx.agent, "_provider", None)
            if provider and hasattr(provider, "list_models"):
                models_list = await provider.list_models()
            else:
                # Fallback to Ollama
                from victor.providers.ollama_provider import OllamaProvider

                provider_settings = ctx.settings.get_provider_settings("ollama")
                ollama = OllamaProvider(**provider_settings)
                models_list = await ollama.list_models()
                await ollama.close()
                provider_name = "ollama"

            if not models_list:
                ctx.console.print("[yellow]No models found[/]")
                return

            table = Table(title=f"Available Models ({provider_name})", show_header=True)
            table.add_column("Model", style="cyan")
            table.add_column("Size", style="yellow")
            table.add_column("Status", style="green")

            current_model = ctx.agent.model if ctx.agent else None

            for model in models_list:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size else 0

                status = "current" if name == current_model else ""
                table.add_row(
                    name,
                    f"{size_gb:.1f} GB" if size_gb > 0 else "?",
                    status,
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Switch: /model <model_name>[/]")

        except Exception as e:
            ctx.console.print(f"[red]Error listing models:[/] {e}")
            ctx.console.print("Make sure your provider is running")

    async def _switch_model(self, ctx: CommandContext, model: str) -> None:
        """Switch to a different model."""
        try:
            if await ctx.agent.switch_model(model):
                info = ctx.agent.get_current_provider_info()
                ctx.console.print(
                    f"[green]\u2713[/] Switched to [cyan]{model}[/]\n"
                    f"  [dim]Native tools: {info['supports_tool_calling']}, "
                    f"Streaming: {info['supports_streaming']}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch model to {model}[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error switching model:[/] {e}")
            logger.exception("Error switching model")

    async def _switch_provider(self, ctx: CommandContext, provider: str, model: str) -> None:
        """Switch to a different provider and model."""
        try:
            if await ctx.agent.switch_provider(provider_name=provider, model=model):
                info = ctx.agent.get_current_provider_info()
                ctx.console.print(
                    f"[green]\u2713[/] Switched to [cyan]{provider}:{model}[/]\n"
                    f"  [dim]Native tools: {info.get('supports_tool_calling', 'N/A')}, "
                    f"Streaming: {info.get('supports_streaming', 'N/A')}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch to {provider}:{model}[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error switching provider:[/] {e}")
            logger.exception("Error switching provider")


@register_command
class ProfileCommand(BaseSlashCommand):
    """Show or switch profile."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="profile",
            description="Show or switch profile",
            usage="/profile [profile_name]",
            aliases=["profiles"],
            category="model",
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        profiles = ctx.settings.load_profiles()

        if ctx.args:
            # Switch to specified profile
            profile_name = ctx.args[0]
            if profile_name not in profiles:
                ctx.console.print(f"[red]Profile not found:[/] {profile_name}")
                ctx.console.print(f"Available: {', '.join(profiles.keys())}")
                return

            if ctx.agent is None:
                ctx.console.print("[yellow]No active session to switch profile[/]")
                return

            profile_config = profiles[profile_name]

            ctx.console.print(
                f"[dim]Switching to profile '{profile_name}' "
                f"({profile_config.provider}:{profile_config.model})...[/]"
            )

            if await ctx.agent.switch_provider(
                provider_name=profile_config.provider,
                model=profile_config.model,
            ):
                info = ctx.agent.get_current_provider_info()
                provider_label = _provider_info_value(info, "provider_name", "provider")
                model_label = _provider_info_value(info, "model_name", "model")
                ctx.console.print(f"[green]Switched to profile:[/] [cyan]{profile_name}[/]")
                ctx.console.print(f"  [dim]Provider: {provider_label}, Model: {model_label}[/]")
                ctx.console.print(
                    f"  [dim]Native tools: {info.get('supports_tool_calling', 'N/A')}, "
                    f"Streaming: {info.get('supports_streaming', 'N/A')}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch to profile {profile_name}[/]")
                ctx.console.print(
                    "[yellow]You may need to restart: "
                    f"[bold]victor --profile {profile_name}[/][/]"
                )
            return

        # Show profiles
        current_provider = ctx.agent.provider_name if ctx.agent else None
        current_model = ctx.agent.model if ctx.agent else None

        # Get RL Q-values for provider ranking
        rl_rankings = {}
        rl_best_provider = None
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner:
                rankings = learner.get_provider_rankings()
                rl_rankings = {r["provider"].lower(): r["q_value"] for r in rankings}
                if rankings:
                    rl_best_provider = rankings[0]["provider"].lower()
        except Exception:
            pass

        table = Table(title="Configured Profiles", show_header=True)
        table.add_column("Profile", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Q-Value", style="magenta", justify="right")
        table.add_column("Status", style="dim")

        for name, config in profiles.items():
            is_current = config.provider == current_provider and config.model == current_model
            status = "current" if is_current else ""

            provider_lower = config.provider.lower()
            if provider_lower == rl_best_provider and not is_current:
                status = "RL best" if not status else f"{status}, RL best"

            q_val = rl_rankings.get(provider_lower)
            q_str = f"{q_val:.2f}" if q_val is not None else "-"

            table.add_row(name, config.provider, config.model, q_str, status)

        ctx.console.print(table)
        ctx.console.print("\n[dim]Switch profile: /profile <name>[/]")
        if rl_rankings:
            ctx.console.print("[dim]RL recommends based on historical performance[/]")


@register_command
class ProviderCommand(BaseSlashCommand):
    """Show current provider info or switch provider."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="provider",
            description="Show current provider info or switch provider",
            usage="/provider [provider_name]",
            aliases=["providers"],
            category="model",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.providers.registry import ProviderRegistry

        info = ctx.agent.get_current_provider_info()
        available_providers = ProviderRegistry.list_providers()

        if not ctx.args:
            # Show current provider info
            ctx.console.print(
                Panel(
                    f"[bold]Provider:[/] [cyan]{info.get('provider', 'unknown')}[/]\n"
                    f"[bold]Model:[/] [yellow]{info.get('model', 'unknown')}[/]\n"
                    f"[bold]Supports Tools:[/] {info.get('supports_tools', False)}\n"
                    f"[bold]Native Tool Calls:[/] {info.get('native_tool_calls', False)}\n"
                    f"[bold]Thinking Mode:[/] {info.get('thinking_mode', False)}\n"
                    f"[bold]Tool Budget:[/] {info.get('tool_budget', 'N/A')}",
                    title="Current Provider",
                    border_style="cyan",
                )
            )

            # Show RL rankings
            try:
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")
                if learner:
                    rankings = learner.get_provider_rankings()
                    if rankings:
                        ctx.console.print("\n[bold]Provider Rankings (RL):[/]")
                        for r in rankings[:5]:
                            indicator = (
                                "current"
                                if r["provider"].lower() == info.get("provider", "").lower()
                                else ""
                            )
                            ctx.console.print(
                                f"  {r['provider']}: Q={r['q_value']:.2f} "
                                f"(samples: {r['sample_count']}) {indicator}"
                            )
            except Exception:
                pass

            ctx.console.print(f"\n[dim]Available providers: {', '.join(available_providers)}[/]")
            ctx.console.print("[dim]Tip: use /model provider:model to switch[/]")
            return

        # Switch provider
        provider_name = ctx.args[0]
        model = ctx.args[1] if len(ctx.args) > 1 else None

        # Handle provider:model syntax
        if ":" in provider_name and model is None:
            provider_name, model = provider_name.split(":", 1)

        if provider_name not in available_providers:
            ctx.console.print(f"[red]Unknown provider:[/] {provider_name}")
            ctx.console.print(f"Available: {', '.join(available_providers)}")
            return

        ctx.console.print(f"[dim]Switching to {provider_name}...[/]")

        if await ctx.agent.switch_provider(provider_name=provider_name, model=model):
            info = ctx.agent.get_current_provider_info()
            provider_label = _provider_info_value(info, "provider_name", "provider")
            model_label = _provider_info_value(info, "model_name", "model")
            ctx.console.print(f"[green]Switched to:[/] {provider_label}:{model_label}")
            ctx.console.print(
                f"  [dim]Native tools: {info.get('supports_tool_calling', 'N/A')}, "
                f"Streaming: {info.get('supports_streaming', 'N/A')}[/]"
            )
        else:
            ctx.console.print(f"[red]Failed to switch to {provider_name}[/]")
