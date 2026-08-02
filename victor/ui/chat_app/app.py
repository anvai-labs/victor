# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Chainlit chat surface for Victor.

Run via ``victor ui`` (which shells out to ``chainlit run`` on this file). This module is
imported by Chainlit's runner, not by the framework, so ``chainlit`` is a hard import here
and the dependency is gated behind the optional ``chat-ui`` extra.

Architecture: the UI layer talks to the framework **only** through ``VictorClient`` (per the
UI-layer mandate in CLAUDE.md) — no orchestrator, no settings mutation. It streams
``client.stream()`` in-process and renders Victor's event model with the Chainlit-free
mapping in :mod:`victor.ui.chat_app.event_mapping`.
"""

from __future__ import annotations

import asyncio
import logging
import os

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput

from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from victor.ui.chat_app.approval import chainlit_approval_handler
from victor.ui.chat_app.event_mapping import (
    RenderKind,
    history_messages,
    map_event,
    provider_switch_hint,
)
from victor.ui.rendering.markdown_presenters import (
    tool_call_summary,
    tool_result_markdown,
    turn_cost_footer,
)
from victor.ui.rendering.utils import format_duration

logger = logging.getLogger(__name__)


def _format_follow_ups(suggestions: list[dict]) -> str:
    """Render follow-up suggestions as a compact markdown hint block.

    Suggestion dicts carry a human ``suggestion`` and/or a ``command``/``tool``
    (see ``victor/agent/tool_pipeline.py``); show the most human-readable field.
    """
    lines = []
    for suggestion in suggestions[:3]:
        if not isinstance(suggestion, dict):
            continue
        text = suggestion.get("suggestion") or suggestion.get("command") or suggestion.get("tool")
        if text:
            lines.append(f"- {text}")
    if not lines:
        return ""
    return "\n\n💡 **Next steps:**\n" + "\n".join(lines)


_CLIENT_KEY = "victor_client"
# The in-flight turn's asyncio.Task, tracked per session so a Stop action can cancel it and
# on_chat_end can drain it before closing the client.
_TASK_KEY = "active_turn_task"

# Agent profile (from ~/.victor/profiles.yaml) selected via `victor ui --profile`,
# passed through the chainlit subprocess as an env var.
_PROFILE_ENV = "VICTOR_UI_PROFILE"

# Tools that mutate state or run code require explicit approval in the web UI, where a
# user is one click away. Read-only tools stay friction-free.
_ASK_ON_TOOLS = [
    "bash",
    "shell",
    "run_shell",
    "execute_command",
    "exec_command",
    "code_execution",
    "write_file",
    "edit_file",
    "delete_file",
    "git_commit",
    "git_push",
]


def _session_config(
    profile: str | None,
    provider: str | None = None,
    model: str | None = None,
    approval: bool = True,
) -> SessionConfig:
    """Build a SessionConfig from chat controls — shared by startup and ChatSettings updates.

    Tool approval gating is coupled to the toggle: when off, ``ask_on_tools`` is emptied so no
    tool prompts fire. Empty provider/model/profile fall back to the active defaults.
    """
    return SessionConfig.from_cli_flags(
        agent_profile=profile or None,
        provider=provider or None,
        model=model or None,
        tool_preview=True,
        tool_approval_enabled=approval,
        ask_on_tools=_ASK_ON_TOOLS if approval else [],
        ask_fallback="deny",
    )


def _build_client() -> VictorClient:
    """Construct the default per-session VictorClient (profile from ``victor ui --profile``)."""
    return VictorClient(_session_config(os.environ.get(_PROFILE_ENV) or None))


def _profile_names() -> list[str]:
    """List configured agent-profile names for the settings dropdown (best-effort)."""
    try:
        from victor.framework.runtime_discovery import list_runtime_profiles

        names = [p.name for p in list_runtime_profiles() if getattr(p, "name", None)]
        return names or ["default"]
    except Exception:  # profile listing is best-effort
        logger.debug("profile listing failed", exc_info=True)
        return ["default"]


async def _send_settings(client: VictorClient) -> None:
    """Render the ChatSettings panel (provider / profile / model / approval)."""
    try:
        providers = client.get_available_providers()
    except Exception:
        providers = []
    current_provider = client.provider_name
    if current_provider and current_provider not in providers:
        providers = [current_provider, *providers]
    profiles = _profile_names()
    await cl.ChatSettings(
        [
            Select(
                id="provider",
                label="Provider",
                values=providers or [current_provider or "default"],
                initial=current_provider or (providers[0] if providers else ""),
            ),
            Select(id="profile", label="Profile", values=profiles, initial=profiles[0]),
            TextInput(id="model", label="Model (override)", initial=client.model or ""),
            Switch(
                id="tool_approval",
                label="Require approval for risky tools",
                initial=True,
            ),
        ]
    ).send()


def _get_client() -> VictorClient:
    """Return the session's VictorClient, rebuilding one if the session was reset."""
    client = cl.user_session.get(_CLIENT_KEY)
    if client is None:
        client = _build_client()
        cl.user_session.set(_CLIENT_KEY, client)
    return client


@cl.on_chat_start
async def on_chat_start() -> None:
    """Create a per-session VictorClient and greet the user."""
    client = _get_client()

    # Hand the interactive approval handler to the client, which registers it into the DI
    # container it actually builds (after bootstrap, before middleware build). Registering
    # into the global container here instead would be lost: the first turn bootstraps a new
    # container and disposes the one this handler lived in, so every ASK-gated tool would
    # silently hit ask_fallback (deny). Safe to call on every on_chat_start / reconnect.
    try:
        client.set_approval_handler(chainlit_approval_handler)
    except Exception:  # approval is best-effort; chat still works without it
        logger.debug("Approval handler registration skipped", exc_info=True)

    # Best-effort in-session replay: a reconnected session whose client already holds history
    # replays its prior turns; a fresh client raises (uninitialized) and we fall through.
    restored = False
    try:
        for author, content in history_messages(await client.get_messages(limit=50)):
            await cl.Message(content=content, author=author).send()
            restored = True
    except Exception:
        logger.debug("no in-session history to restore", exc_info=True)

    if not restored:
        # Cross-visit resume: offer to continue a previously stored session.
        actions = _resume_actions(client)
        greeting = (
            "**Victor** is ready. Ask me to write code, search the repo, run tools — "
            "I'll stream my reasoning and tool calls as I work."
        )
        if actions:
            greeting += "\n\n_Or continue a previous session:_"
        await cl.Message(content=greeting, actions=actions or None).send()

    await _send_settings(client)


def _resume_actions(client: "VictorClient") -> list:
    """Build 'continue previous session' picker actions from stored sessions."""
    try:
        sessions = client.list_recent_sessions(limit=5)
    except Exception:
        logger.debug("could not list recent sessions", exc_info=True)
        return []
    actions = []
    for session in sessions:
        session_id = session.get("session_id")
        if not session_id:
            continue
        title = session.get("title") or "Untitled session"
        count = session.get("message_count", 0)
        actions.append(
            cl.Action(
                name="resume_session",
                payload={"session_id": session_id},
                label=f"↩ {title} ({count} msgs)",
            )
        )
    return actions


@cl.action_callback("resume_session")
async def _on_resume_session(action: "cl.Action") -> None:
    """Resume a stored session: hydrate the agent, then replay its turns."""
    session_id = (action.payload or {}).get("session_id")
    if not session_id:
        return
    client = _get_client()
    try:
        await client.initialize()
        metadata = await client.resume_session(session_id)
    except Exception as exc:
        logger.warning("resume_session failed: %s", exc, exc_info=True)
        await cl.Message(content=f"⚠️ Could not resume that session: {exc}").send()
        return
    if metadata is None:
        await cl.Message(content="⚠️ That session could not be found.").send()
        return

    title = metadata.get("title", "session")
    await cl.Message(content=f"**Resumed:** {title} — the agent recalls the prior turns.").send()
    # Replay the restored turns into the transcript so the user sees the context.
    try:
        for author, content in history_messages(await client.get_messages(limit=50)):
            await cl.Message(content=content, author=author).send()
    except Exception:
        logger.debug("resume replay skipped", exc_info=True)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """Rebuild the session's VictorClient from the ChatSettings panel selections."""
    # Drain any in-flight turn first (PR3 guard), then release the old client.
    task = cl.user_session.get(_TASK_KEY)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            logger.debug("active turn drained on settings update", exc_info=True)
        finally:
            cl.user_session.set(_TASK_KEY, None)

    old = cl.user_session.get(_CLIENT_KEY)
    if old is not None:
        try:
            await old.close()
        except Exception:
            logger.debug("old client close failed on settings update", exc_info=True)

    approval = bool(settings.get("tool_approval", True))
    client = VictorClient(
        _session_config(
            profile=settings.get("profile") or None,
            provider=settings.get("provider") or None,
            model=settings.get("model") or None,
            approval=approval,
        )
    )
    cl.user_session.set(_CLIENT_KEY, client)
    await cl.Message(
        content=(
            "⚙️ Settings applied — provider: **{provider}**, profile: **{profile}**, "
            "approval: **{approval}**.".format(
                provider=settings.get("provider") or "default",
                profile=settings.get("profile") or "default",
                approval="on" if approval else "off",
            )
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream a Victor turn as a cancellable task with a Stop control."""
    await _start_turn(message.content)


async def _start_turn(content: str) -> None:
    """Run one turn as a tracked, cancellable asyncio task alongside a Stop control.

    The turn body runs in ``_run_turn`` as a task stored in the session so the Stop action can
    cancel it; cancellation propagates ``CancelledError`` into ``client.stream()``, whose in-task
    ``aclose()`` drains the provider iterator cleanly.
    """
    client = _get_client()
    task = asyncio.create_task(_run_turn(client, content))
    cl.user_session.set(_TASK_KEY, task)
    stop = cl.Message(content="", actions=[cl.Action(name="stop_turn", payload={}, label="⏹ Stop")])
    await stop.send()
    try:
        await task
    except asyncio.CancelledError:
        await cl.Message(content="⏹ _Turn stopped._").send()
    except Exception:  # _run_turn renders its own recovery card; this is a backstop
        logger.debug("turn task raised after handling", exc_info=True)
    finally:
        cl.user_session.set(_TASK_KEY, None)
        try:
            await stop.remove()
        except Exception:
            logger.debug("stop control removal failed", exc_info=True)


@cl.action_callback("stop_turn")
async def _on_stop_turn(action: "cl.Action") -> None:
    """Cancel the in-flight turn task for this session."""
    task = cl.user_session.get(_TASK_KEY)
    if task is not None and not task.done():
        task.cancel()


@cl.action_callback("retry_turn")
async def _on_retry_turn(action: "cl.Action") -> None:
    """Re-run the failed turn with its original message (carried on the action payload)."""
    content = (getattr(action, "payload", None) or {}).get("content", "")
    if content:
        await _start_turn(content)


async def _run_turn(client: VictorClient, content: str) -> None:
    """Stream a Victor turn, rendering tokens, reasoning, and tool calls."""
    # P1.4 first-token feedback: a transient "Thinking…" message so the user never stares at
    # a blank bubble; removed the instant any real output (token/reasoning/tool) arrives.
    thinking = cl.Message(content="🔄 _Thinking…_")
    await thinking.send()
    state = {"thinking_cleared": False}

    async def _clear_thinking() -> None:
        if not state["thinking_cleared"]:
            state["thinking_cleared"] = True
            try:
                await thinking.remove()
            except Exception:
                logger.debug("thinking placeholder removal failed", exc_info=True)

    # Natural per-iteration flow (terminal-like): instead of one long message with all tool
    # steps piling at the end, stream each iteration's text into its OWN message segment and
    # group that iteration's tool calls under one collapsible parent step between segments —
    # timeline reads text → tools → text → tools …. Contract: event_mapping.segment_turn.
    phase = "text"  # "text" | "tools"
    current_msg: cl.Message | None = None
    tool_group: cl.Step | None = None
    pending: dict[str, dict] = {}  # call_id -> {tool_name, arguments}
    reasoning_step: cl.Step | None = None
    reasoning_text = ""

    async def _finalize_text() -> None:
        nonlocal current_msg
        if current_msg is not None:
            try:
                await current_msg.update()
            except Exception:
                logger.debug("text segment finalize failed", exc_info=True)
            current_msg = None

    async def _close_tool_group() -> None:
        nonlocal tool_group
        if tool_group is not None:
            try:
                await tool_group.__aexit__(None, None, None)
            except Exception:
                logger.debug("tool group finalize failed", exc_info=True)
            tool_group = None

    async def _open_tool_group() -> None:
        nonlocal tool_group
        if tool_group is None:
            tool_group = cl.Step(name="🔧 tools", type="tool")
            await tool_group.__aenter__()

    async def _emit_text(text: str) -> None:
        nonlocal current_msg, phase
        # Text resuming after a tool run begins a new iteration segment.
        if phase == "tools":
            await _close_tool_group()
            await _finalize_text()
            phase = "text"
        if current_msg is None:
            current_msg = cl.Message(content="")
            await current_msg.send()
        await current_msg.stream_token(text)

    try:
        async for event in client.stream(content):
            action = map_event(event)

            if action.kind is RenderKind.TOKEN:
                await _clear_thinking()
                await _emit_text(action.text)

            elif action.kind is RenderKind.THINKING:
                await _clear_thinking()
                if action.text:
                    reasoning_text += action.text
                    if reasoning_step is None:
                        reasoning_step = cl.Step(name="reasoning", type="llm")
                        await reasoning_step.__aenter__()
                    reasoning_step.output = reasoning_text
                    await reasoning_step.update()

            elif action.kind is RenderKind.TOOL_START:
                await _clear_thinking()
                phase = "tools"
                await _open_tool_group()
                key = action.call_id or action.tool_name or "tool"
                pending[key] = {
                    "tool_name": action.tool_name or "tool",
                    "arguments": action.metadata.get("arguments", {}),
                }

            elif action.kind is RenderKind.TOOL_END:
                await _clear_thinking()
                phase = "tools"
                await _open_tool_group()
                key = action.call_id or action.tool_name or "tool"
                info = pending.pop(key, {})
                tool_name = info.get("tool_name") or action.tool_name or "tool"
                args = info.get("arguments", {})
                # Append duration to the step label so the user sees how long it took
                # (e.g. "read(path=…) · 12ms").
                label = tool_call_summary(tool_name, args)
                if action.elapsed:
                    label = f"{label} · {format_duration(action.elapsed)}"
                # Child step nests under the iteration's "🔧 tools" group (parallel calls grouped).
                async with cl.Step(name=label, type="tool") as step:
                    step.input = args
                    output = tool_result_markdown(
                        tool_name, args, action.text, success=action.success
                    )
                    if action.was_pruned:
                        output = f"{output}\n\n_(output truncated for length)_"
                    # Keep follow-up hints grouped with the tool that produced them.
                    follow_ups = _format_follow_ups(action.follow_up_suggestions or [])
                    if follow_ups:
                        output = f"{output}{follow_ups}"
                    step.output = output
                    if not action.success:
                        step.is_error = True

            elif action.kind is RenderKind.ERROR:
                await _clear_thinking()
                await _emit_text(f"\n\n⚠️ {action.text}")

    except Exception as exc:  # surface failures in-chat with a recovery path
        logger.exception("Victor chat turn failed")
        await _clear_thinking()
        from victor.framework.contextual_errors import format_exception_for_user

        friendly = format_exception_for_user(
            exc, {"operation": "chat", "provider": client.provider_name}
        )
        body = f"⚠️ {friendly}"
        try:
            hint = provider_switch_hint(client.provider_name, client.get_available_providers())
        except Exception:  # provider listing is best-effort
            hint = ""
        if hint:
            body = f"{body}\n\n{hint}"
        await cl.Message(
            content=body,
            actions=[cl.Action(name="retry_turn", payload={"content": content}, label="🔄 Retry")],
        ).send()
    finally:
        await _clear_thinking()
        await _close_tool_group()
        if reasoning_step is not None:
            try:
                await reasoning_step.__aexit__(None, None, None)
            except Exception:
                logger.debug("reasoning step finalize failed", exc_info=True)
        await _finalize_text()
        # L3: surface the C0 per-turn cost/latency record as a compact footer (tokens ×
        # round-trips, $, latency) so the savings from L1/L2 are visible. Best-effort.
        try:
            footer = turn_cost_footer(client.get_last_turn_cost())
            if footer:
                await cl.Message(content=footer).send()
        except Exception:
            logger.debug("cost footer render failed", exc_info=True)


@cl.on_chat_end
async def on_chat_end() -> None:
    """Drain any in-flight turn, then release the session's VictorClient resources."""
    # Cancel + await the active turn first so close() never races a live stream (it pairs with
    # VictorClient.close()'s own _active_streams drain guard).
    task = cl.user_session.get(_TASK_KEY)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            logger.debug("active turn drained on chat end", exc_info=True)
        finally:
            cl.user_session.set(_TASK_KEY, None)

    client: VictorClient | None = cl.user_session.get(_CLIENT_KEY)
    if client is not None:
        try:
            await client.close()
        except Exception:  # best-effort cleanup
            logger.debug("VictorClient close failed", exc_info=True)
