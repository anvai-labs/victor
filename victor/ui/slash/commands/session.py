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

"""Session management slash commands: save, load, sessions, resume, compact."""

from __future__ import annotations

import logging
from typing import Any

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


def _resolve_conversation_owner(agent: Any) -> Any:
    """Resolve the conversation-bearing runtime object from a slash-command agent.

    The slash handler receives the framework ``Agent`` facade, which does not
    expose ``conversation`` / ``conversation_controller`` directly — those live
    on the underlying ``AgentOrchestrator``. Resolve through
    ``Agent.get_orchestrator()`` so commands work against both the facade and a
    raw orchestrator. Mirrors the resolution pattern in ``chat.py``.

    Args:
        agent: The object in ``CommandContext.agent`` (Agent facade or orchestrator).

    Returns:
        The object exposing ``conversation`` / ``conversation_controller``.
    """
    if agent is None:
        return None
    if hasattr(agent, "conversation"):
        return agent
    orchestrator_getter = getattr(agent, "get_orchestrator", None)
    if callable(orchestrator_getter):
        try:
            return orchestrator_getter()
        except Exception:
            logger.debug(
                "get_orchestrator() failed during slash-command resolution",
                exc_info=True,
            )
    return agent


def _preview_count(preview_messages: object) -> int:
    """Return a stable preview-message count for persisted/session history data."""
    return len(preview_messages) if isinstance(preview_messages, list) else 0


def _preview_path_summary(preview_messages: object, *, max_paths: int = 3) -> str | None:
    """Return a compact summary of preview file paths for session UI."""
    if not isinstance(preview_messages, list):
        return None

    paths: list[str] = []
    seen: set[str] = set()
    for preview in preview_messages:
        if not isinstance(preview, dict):
            continue
        metadata = preview.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        preview_path = str(metadata.get("preview_path", "")).strip()
        if not preview_path or preview_path in seen:
            continue
        seen.add(preview_path)
        paths.append(preview_path)

    if not paths:
        return None

    visible_paths = paths[:max_paths]
    summary = ", ".join(visible_paths)
    remaining = len(paths) - len(visible_paths)
    if remaining > 0:
        summary = f"{summary}, +{remaining} more"

    return summary


def _session_preview_count(session: object) -> int:
    """Return a stable preview-message count for persisted session objects."""
    preview_messages = getattr(session, "preview_messages", None)
    return len(preview_messages) if isinstance(preview_messages, list) else 0


@register_command
class SaveCommand(BaseSlashCommand):
    """Save current conversation to a session file."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="save",
            description="Save current conversation to SQLite session (updates active session or creates new)",
            usage="/save [--new] [name]",
            category="session",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.session import get_session_manager
        from victor.agent.conversation.store import ConversationStore

        # Check for --new flag
        force_new = self._has_flag(ctx, "--new", "-n")

        # Extract title (remove --new flag if present)
        args = [arg for arg in ctx.args if arg not in ("--new", "-n")]
        title = " ".join(args) if args else None

        try:
            store = ConversationStore()
            preview_messages = getattr(ctx.agent.conversation, "preview_messages", [])
            message_count = ctx.agent.conversation.message_count()
            preview_count = _preview_count(preview_messages)
            preview_paths = _preview_path_summary(preview_messages)

            # Determine session_id to use
            if force_new:
                # Create new session (ignore active_session_id)
                session_id = None
                action = "Created new session"
            elif getattr(ctx.agent, "active_session_id", None):
                # Update existing active session
                session_id = ctx.agent.active_session_id
                action = f"Updated session {session_id}"
            else:
                # No active session, create new
                session_id = None
                action = "Created new session"

            # Save to SQLite
            # Get execution state for restoration on resume
            execution_state = None
            if hasattr(ctx.agent, "session_state_accessor"):
                execution_state = ctx.agent.session_state_accessor.session_state.execution_state

            # Get session ledger for persistence
            session_ledger = getattr(ctx.agent, "session_ledger", None)

            # Get compaction hierarchy state for persistence
            compaction_hierarchy = None
            cc = getattr(ctx.agent, "conversation_controller", None)
            if cc:
                hm = getattr(cc, "_hierarchical_manager", None)
                if hm and hasattr(hm, "to_dict"):
                    compaction_hierarchy = hm.to_dict()

            session_id = store.save_session(
                conversation=ctx.agent.conversation,
                model=ctx.agent.model,
                provider=ctx.agent.provider_name,
                profile=getattr(ctx.settings, "current_profile", "default"),
                session_id=session_id,  # Use existing or None for new
                title=title,
                conversation_state=getattr(ctx.agent, "conversation_state", None),
                execution_state=execution_state,
                session_ledger=session_ledger,
                compaction_hierarchy=compaction_hierarchy,
            )

            if session_id:
                # Set active_session_id on agent
                ctx.agent.active_session_id = session_id
                preview_paths_line = (
                    f"[bold]Preview Files:[/] {preview_paths}\n" if preview_paths else ""
                )

                ctx.console.print(
                    Panel(
                        f"{action}!\n\n"
                        f"[bold]Session ID:[/] {session_id}\n"
                        f"[bold]Title:[/] {title or 'Auto-generated'}\n\n"
                        f"[bold]Messages:[/] {message_count}\n"
                        f"[bold]Previews:[/] {preview_count}\n"
                        f"{preview_paths_line}\n"
                        f"[dim]Use '/resume {session_id}' to restore[/]\n"
                        f"[dim]Use '/save --new' to create a new session[/]",
                        title="Session Saved",
                        border_style="green",
                    )
                )
            else:
                ctx.console.print("[red]Failed to save session[/]")
                logger.error("Session ID was empty after save")

        except Exception as e:
            ctx.console.print(f"[red]Failed to save session:[/] {e}")
            logger.exception("Error saving session")


@register_command
class LoadCommand(BaseSlashCommand):
    """Load a saved session."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="load",
            description="Load a saved session",
            usage="/load <session_id>",
            category="session",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not ctx.args:
            ctx.console.print("[yellow]Usage:[/] /load <session_id>")
            ctx.console.print("[dim]Use '/sessions' to list available sessions[/]")
            return

        if not self._require_agent(ctx):
            return

        from victor.agent.conversation.state_machine import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.session import get_session_manager

        session_id = ctx.args[0]

        try:
            session_manager = get_session_manager()
            session = session_manager.load_session(session_id)

            if session is None:
                ctx.console.print(f"[red]Session not found:[/] {session_id}")
                return

            # Restore conversation
            conversation_dict = session.conversation
            preview_messages = conversation_dict.get("preview_messages", [])
            preview_count = _preview_count(preview_messages)
            preview_paths = _preview_path_summary(preview_messages)
            preview_paths_line = (
                f"[bold]Preview Files:[/] {preview_paths}\n" if preview_paths else ""
            )

            ctx.agent.conversation = MessageHistory.from_dict(conversation_dict)

            # Set active session ID for parallel session support
            ctx.agent.active_session_id = session_id

            # Restore conversation state machine if available
            if session.conversation_state:
                ctx.agent.conversation_state = ConversationStateMachine.from_dict(
                    session.conversation_state
                )
                logger.info(
                    f"Restored conversation state: stage={ctx.agent.conversation_state.get_stage().name}"
                )

            ctx.console.print(
                Panel(
                    f"Session loaded successfully!\n\n"
                    f"[bold]Title:[/] {session.metadata.title}\n"
                    f"[bold]Model:[/] {session.metadata.model}\n"
                    f"[bold]Provider:[/] {session.metadata.provider}\n"
                    f"[bold]Messages:[/] {session.metadata.message_count}\n"
                    f"[bold]Previews:[/] {preview_count}\n"
                    f"{preview_paths_line}"
                    f"[bold]Created:[/] {session.metadata.created_at}",
                    title="Session Loaded",
                    border_style="green",
                )
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to load session:[/] {e}")
            logger.exception("Error loading session")


@register_command
class SessionsCommand(BaseSlashCommand):
    """List saved sessions."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="sessions",
            description="List saved sessions",
            usage="/sessions [limit]",
            aliases=["history"],
            category="session",
        )

    def execute(self, ctx: CommandContext) -> None:
        from victor.agent.conversation.store import ConversationStore

        limit = self._parse_int_arg(ctx, 0, default=10)

        try:
            store = ConversationStore()
            sessions = store.list_sessions(limit=limit)

            if not sessions:
                ctx.console.print("[dim]No saved sessions found[/]")
                return

            table = Table(title=f"Saved Sessions (last {len(sessions)})")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Provider", style="blue")
            table.add_column("Messages", justify="right")
            table.add_column("Previews", justify="right")
            table.add_column("Created", style="dim")

            for session in sessions:
                try:
                    date_str = session.created_at.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = str(session.created_at)[:16]

                title = session.title or "Untitled"
                if len(title) > 40:
                    title = title[:40] + "..."

                table.add_row(
                    session.session_id,
                    title,
                    session.model or "unknown",
                    session.provider or "unknown",
                    str(len(session.messages)),
                    str(_session_preview_count(session)),
                    date_str,
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Use '/resume <session_id>' to restore a session[/]")
            ctx.console.print(
                "[dim]Or '/switch <model> --resume <session_id>' to resume and switch[/]"
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to list sessions:[/] {e}")
            logger.exception("Error listing sessions")


@register_command
class ResumeCommand(BaseSlashCommand):
    """Resume a session with interactive selection from SQLite."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="resume",
            description="Resume a session from SQLite history",
            usage="/resume [session_id]",
            category="session",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.conversation.state_machine import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.conversation.store import ConversationStore

        # If session_id provided as argument, load it directly
        if ctx.args:
            session_id = ctx.args[0]
            self._load_session(ctx, session_id)
            return

        # Otherwise, show interactive selection
        try:
            store = ConversationStore()
            sessions = store.list_sessions(limit=20)

            if not sessions:
                ctx.console.print("[yellow]No sessions found[/]")
                ctx.console.print("[dim]Start a conversation to create sessions[/]")
                return

            # Display sessions with numbers
            table = Table(title="Recent Sessions")
            table.add_column("#", style="cyan", no_wrap=True, width=4)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Messages", justify="right")
            table.add_column("Previews", justify="right")
            table.add_column("Date", style="dim")

            for idx, session in enumerate(sessions, 1):
                try:
                    date_str = session.created_at.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = str(session.created_at)[:16]

                title = session.title or "Untitled"
                if len(title) > 40:
                    title = title[:40] + "..."

                table.add_row(
                    str(idx),
                    session.session_id,
                    title,
                    session.model or "unknown",
                    str(len(session.messages)),
                    str(_session_preview_count(session)),
                    date_str,
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Enter session number to resume (1-{})[/]", len(sessions))
            ctx.console.print("[dim]Or use: /resume <session_id>[/]")

        except Exception as e:
            ctx.console.print(f"[red]Failed to list sessions:[/] {e}")
            logger.exception("Error listing sessions")

    def _load_session(self, ctx: CommandContext, session_id: str) -> None:
        """Load and restore a session.

        Args:
            ctx: Command context
            session_id: Session ID to load
        """
        from victor.agent.conversation.state_machine import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.conversation.store import ConversationStore

        try:
            store = ConversationStore()
            session_data = store.load_session(session_id)

            if session_data is None:
                ctx.console.print(f"[red]Session not found:[/] {session_id}")
                return

            # Restore conversation
            metadata = session_data.get("metadata", {})
            conversation_dict = session_data.get("conversation", {})
            preview_messages = conversation_dict.get("preview_messages", [])
            preview_count = _preview_count(preview_messages)
            preview_paths = _preview_path_summary(preview_messages)
            preview_paths_line = (
                f"[bold]Preview Files:[/] {preview_paths}\n" if preview_paths else ""
            )

            ctx.agent.conversation = MessageHistory.from_dict(conversation_dict)

            # Set active session ID for parallel session support
            ctx.agent.active_session_id = session_id

            # Restore conversation state if available
            conversation_state_dict = session_data.get("conversation_state")
            if conversation_state_dict:
                try:
                    ctx.agent.conversation_state = ConversationStateMachine.from_dict(
                        conversation_state_dict
                    )
                    logger.info(
                        f"Restored conversation state: stage={ctx.agent.conversation_state.get_stage().name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore conversation state: {e}")

            # Restore ledger + execution state via SessionContextLinker
            from victor.agent.session_context_linker import SessionContextLinker

            linker = SessionContextLinker(conversation_store=store)
            resume_ctx = linker.build_resume_context(session_id)

            if resume_ctx.ledger and hasattr(ctx.agent, "session_ledger"):
                ctx.agent.session_ledger = resume_ctx.ledger
                logger.info(f"Restored session ledger: {len(resume_ctx.ledger.entries)} entries")

            if resume_ctx.execution_state and hasattr(ctx.agent, "session_state_accessor"):
                accessor = ctx.agent.session_state_accessor
                accessor.observed_files = resume_ctx.execution_state.observed_files
                accessor.executed_tools = resume_ctx.execution_state.executed_tools
                accessor.tool_calls_used = resume_ctx.execution_state.tool_calls_used
                logger.info("Restored execution state via SessionContextLinker")

            # Restore compaction hierarchy
            hierarchy_data = session_data.get("compaction_hierarchy")
            if hierarchy_data:
                cc = getattr(ctx.agent, "conversation_controller", None)
                if cc and hasattr(cc, "_hierarchical_manager"):
                    try:
                        from victor.agent.compaction_hierarchy import (
                            HierarchicalCompactionManager,
                        )

                        cc._hierarchical_manager = HierarchicalCompactionManager.from_dict(
                            hierarchy_data
                        )
                        logger.info("Restored compaction hierarchy")
                    except Exception as e:
                        logger.warning(f"Failed to restore compaction hierarchy: {e}")

            # Inject restored compaction context so the LLM is oriented on next turn
            _cc = getattr(ctx.agent, "conversation_controller", None)
            if _cc:
                _hm = getattr(_cc, "_hierarchical_manager", None)
                if _hm:
                    active_ctx = _hm.get_active_context()
                    if (
                        active_ctx
                        and hasattr(_cc, "_compaction_summaries")
                        and not _cc._compaction_summaries
                    ):
                        _cc._compaction_summaries.append(active_ctx)
                    try:
                        _cc.inject_compaction_context()
                        logger.info("Injected compaction context from restored hierarchy")
                    except Exception as e:
                        logger.debug(f"Compaction context injection skipped on resume: {e}")

            ctx.console.print(
                Panel(
                    f"Session restored from SQLite!\n\n"
                    f"[bold]ID:[/] {metadata.get('session_id', session_id)}\n"
                    f"[bold]Title:[/] {metadata.get('title', 'Untitled')}\n"
                    f"[bold]Model:[/] {metadata.get('model', 'N/A')}\n"
                    f"[bold]Provider:[/] {metadata.get('provider', 'N/A')}\n"
                    f"[bold]Messages:[/] {metadata.get('message_count', 0)}\n"
                    f"[bold]Previews:[/] {preview_count}\n"
                    f"{preview_paths_line}"
                    f"[bold]Created:[/] {metadata.get('created_at', 'N/A')}\n"
                    f"[bold]Resume:[/] {resume_ctx.resume_summary}",
                    title="Session Resumed",
                    border_style="green",
                )
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to load session:[/] {e}")
            logger.exception("Error loading session")


@register_command
class CompactCommand(BaseSlashCommand):
    """Compress conversation history to reduce context size."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="compact",
            description="Compress conversation history (use --smart for AI summarization)",
            usage="/compact [--smart]",
            aliases=["summarize"],
            category="session",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        # ctx.agent is the framework Agent facade; ``conversation`` lives on the
        # underlying orchestrator (see orchestrator.py). Resolve it explicitly.
        owner = _resolve_conversation_owner(ctx.agent)
        conversation = getattr(owner, "conversation", None)
        if conversation is None:
            ctx.console.print(
                "[red]Cannot compact:[/] active session exposes no conversation history."
            )
            return

        original_count = conversation.message_count()

        if original_count < 5:
            ctx.console.print("[dim]Conversation is already small enough, nothing to compact[/]")
            return

        use_smart = self._has_flag(ctx, "--smart", "-s")
        keep_recent = 6

        # Parse --keep N
        keep_val = self._get_flag_value(ctx, "--keep") or self._get_flag_value(ctx, "-k")
        if keep_val:
            try:
                keep_recent = int(keep_val)
            except ValueError:
                pass

        keep_recent = min(keep_recent, len(conversation.messages) // 2)
        messages = conversation.messages

        if use_smart:
            ctx.console.print("[dim]Generating AI summary...[/]")

            conversation_text = "\n".join(
                [
                    (
                        f"{msg.role}: {msg.content[:200]}..."
                        if len(msg.content) > 200
                        else f"{msg.role}: {msg.content}"
                    )
                    for msg in messages[:-keep_recent]
                ]
            )

            summary_prompt = (
                "Summarize this conversation concisely, focusing on key decisions, "
                f"code changes, and context needed for continuation:\n\n{conversation_text}"
            )

            try:
                summary_response = await ctx.agent.chat(summary_prompt)
                summary = summary_response.content

                # Create new conversation with summary + recent messages
                from victor.agent.message_types import Message

                new_messages = [
                    Message(
                        role="system",
                        content=f"[Previous conversation summary]\n{summary}",
                    ),
                    *messages[-keep_recent:],
                ]
                # MessageHistory.messages is read-only (returns a copy); mutate via
                # the public clear()/append_message() API — mirrors the canonical
                # ConversationController.compact_history path.
                conversation.clear()
                for _msg in new_messages:
                    conversation.append_message(_msg)

                # Record in controller tracking systems (mirrors auto-compact in conversation_controller)
                _cc = getattr(owner, "conversation_controller", None)
                if _cc:
                    _compact_record = f"[Manual compact: {summary[:300]}]"
                    if hasattr(_cc, "_compaction_summaries"):
                        _cc._compaction_summaries.append(_compact_record)
                    _hm = getattr(_cc, "_hierarchical_manager", None)
                    if _hm:
                        _hm.add_summary(_compact_record, len(messages))

                ctx.console.print(
                    Panel(
                        f"Compacted {original_count} messages to {len(new_messages)}.\n\n"
                        f"[bold]Summary:[/]\n{summary[:500]}...",
                        title="Smart Compaction Complete",
                        border_style="green",
                    )
                )
            except Exception as e:
                ctx.console.print(f"[red]Smart compaction failed:[/] {e}")
        else:
            # Simple truncation — preserve a leading system message plus the
            # recent window via the public mutation API (messages is read-only).
            _system_msg = None
            if messages and messages[0].role == "system":
                _system_msg = messages[0]
            recent = messages[-keep_recent:]
            conversation.clear()
            if _system_msg is not None:
                conversation.append_message(_system_msg)
            for _msg in recent:
                conversation.append_message(_msg)
            new_count = conversation.message_count()

            ctx.console.print(
                f"[green]Compacted:[/] {original_count} -> {new_count} messages\n"
                "[dim]Use /compact --smart for AI-powered summarization[/]"
            )
